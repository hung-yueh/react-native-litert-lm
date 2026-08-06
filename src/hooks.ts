import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { LLMConfig, MemoryWarningLevel } from "./index";
import { createLLM } from "./modelFactory";
import type { LiteRTLMInstance } from "./modelFactory";
import type { MemoryTracker, MemoryTrackerSummary } from "./memoryTracker";
import type { MemoryEstimate, MemoryForecast } from "./memoryEstimator";
import { ModelRegistry } from "./modelRegistry";
import { extractFileName } from "./modelPath";

export interface UseModelConfig extends LLMConfig {
  autoLoad?: boolean;
  /**
   * Enable memory tracking using native ArrayBuffers (v0.35+).
   * When enabled, memory usage is tracked after each inference call
   * using `NitroModules.createNativeArrayBuffer()` for zero-copy storage.
   * @default false
   */
  enableMemoryTracking?: boolean;
  /**
   * Maximum number of memory snapshots to store.
   * Each snapshot uses 32 bytes of native memory.
   * @default 256
   */
  maxMemorySnapshots?: number;
}

export interface UseModelResult {
  model: LiteRTLMInstance | null;
  isReady: boolean;
  isGenerating: boolean;
  downloadProgress: number;
  error: string | null;
  generate: (prompt: string) => Promise<string>;
  reset: () => void;
  /**
   * Delete the model file. If no fileName is provided, derives it from
   * the URL/path passed to useModel.
   */
  deleteModel: (fileName?: string) => Promise<void>;
  load: () => Promise<void>;
  /**
   * Memory tracker instance (available when enableMemoryTracking is true).
   * Uses native ArrayBuffers allocated via `NitroModules.createNativeArrayBuffer()`
   * for efficient, zero-copy memory usage tracking.
   */
  memoryTracker: MemoryTracker | null;
  /**
   * Current memory tracking summary (null if tracking is disabled).
   * Updates automatically after each inference call.
   */
  memorySummary: MemoryTrackerSummary | null;
  /**
   * Pre-flight memory estimate for the loaded model (null until loaded).
   * Refreshed after each load.
   */
  memoryEstimate: MemoryEstimate | null;
  /**
   * Context-window / KV-cache growth forecast for the active conversation.
   * Refreshed after each generate() call — watch `nearingLimit` to warn users
   * before the context (and its memory) runs out.
   */
  memoryForecast: MemoryForecast | null;
  /**
   * Last OS memory-pressure warning received while this hook was mounted
   * (null if none). Reset on each successful load.
   */
  memoryWarning: MemoryWarningLevel | null;
}

export function useModel(
  pathOrUrl: string,
  config?: UseModelConfig,
): UseModelResult {
  const modelRef = useRef<LiteRTLMInstance | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [memorySummary, setMemorySummary] =
    useState<MemoryTrackerSummary | null>(null);
  const [memoryEstimate, setMemoryEstimate] = useState<MemoryEstimate | null>(
    null,
  );
  const [memoryForecast, setMemoryForecast] = useState<MemoryForecast | null>(
    null,
  );
  const [memoryWarning, setMemoryWarning] =
    useState<MemoryWarningLevel | null>(null);

  // Destructure config into primitive values for stable dependency arrays.
  // This prevents infinite re-render loops when consumers pass inline config
  // objects (e.g. useModel(url, { backend: 'cpu' })) without useMemo.
  const autoLoad = config?.autoLoad ?? true;
  const enableMemoryTracking = config?.enableMemoryTracking ?? false;
  const maxMemorySnapshots = config?.maxMemorySnapshots ?? 256;
  const backend = config?.backend;
  const systemPrompt = config?.systemPrompt;
  const maxTokens = config?.maxTokens;
  const maxContextTokens = config?.maxContextTokens;
  const maxOutputTokens = config?.maxOutputTokens;
  const temperature = config?.temperature;
  const topK = config?.topK;
  const topP = config?.topP;
  const multimodal = config?.multimodal;
  const tools = config?.tools;
  const enableSpeculativeDecoding = config?.enableSpeculativeDecoding;
  const enableStructuredOutput = config?.enableStructuredOutput;
  const streamToolCalls = config?.streamToolCalls;
  const toolCallChannelName = config?.toolCallChannelName;
  const numThreads = config?.numThreads;
  const prefillChunkSize = config?.prefillChunkSize;
  const activationDataType = config?.activationDataType;
  const loraPath = config?.loraPath;
  const audioLoraPath = config?.audioLoraPath;
  const loraRank = config?.loraRank;
  const forceLoad = config?.forceLoad;
  const toolsKey = tools ? JSON.stringify(tools) : undefined;

  // Build a stable config object from the destructured primitives
  const nativeConfig = useMemo<LLMConfig>(
    () => ({
      ...(backend !== undefined && { backend }),
      ...(systemPrompt !== undefined && { systemPrompt }),
      ...(maxTokens !== undefined && { maxTokens }),
      ...(maxContextTokens !== undefined && { maxContextTokens }),
      ...(maxOutputTokens !== undefined && { maxOutputTokens }),
      ...(temperature !== undefined && { temperature }),
      ...(topK !== undefined && { topK }),
      ...(topP !== undefined && { topP }),
      ...(multimodal !== undefined && { multimodal }),
      ...(tools !== undefined && { tools }),
      ...(enableSpeculativeDecoding !== undefined && {
        enableSpeculativeDecoding,
      }),
      ...(enableStructuredOutput !== undefined && { enableStructuredOutput }),
      ...(streamToolCalls !== undefined && { streamToolCalls }),
      ...(toolCallChannelName !== undefined && { toolCallChannelName }),
      ...(numThreads !== undefined && { numThreads }),
      ...(prefillChunkSize !== undefined && { prefillChunkSize }),
      ...(activationDataType !== undefined && { activationDataType }),
      ...(loraPath !== undefined && { loraPath }),
      ...(audioLoraPath !== undefined && { audioLoraPath }),
      ...(loraRank !== undefined && { loraRank }),
      ...(forceLoad !== undefined && { forceLoad }),
    }),
    // `tools` is tracked via its serialized `toolsKey` so inline arrays stay stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      backend,
      systemPrompt,
      maxTokens,
      maxContextTokens,
      maxOutputTokens,
      temperature,
      topK,
      topP,
      multimodal,
      toolsKey,
      enableSpeculativeDecoding,
      enableStructuredOutput,
      streamToolCalls,
      toolCallChannelName,
      numThreads,
      prefillChunkSize,
      activationDataType,
      loraPath,
      audioLoraPath,
      loraRank,
      forceLoad,
    ],
  );

  /**
   * Refresh memory summary from the tracker's native buffer.
   */
  const refreshMemorySummary = useCallback(() => {
    if (modelRef.current?.memoryTracker) {
      setMemorySummary(modelRef.current.memoryTracker.getSummary());
    }
  }, []);

  // Initialize the model instance
  useEffect(() => {
    modelRef.current = createLLM({
      enableMemoryTracking,
      maxMemorySnapshots,
    });

    // Reset ready state — the new instance has no model loaded yet.
    // This prevents stale isReady=true after Fast Refresh (which
    // preserves useState but re-runs useEffect).
    setIsReady(false);

    // Cleanup on unmount
    return () => {
      try {
        modelRef.current?.close();
      } catch (e) {
        console.warn("Failed to close model", e);
      }
    };
  }, [enableMemoryTracking, maxMemorySnapshots]);

  const load = useCallback(async () => {
    setIsReady(false);
    setError(null);
    setDownloadProgress(0);

    try {
      if (modelRef.current) {
        // Delegate URL handling + download to the factory's loadModel,
        // passing our progress setter as the callback (eliminates
        // duplicate download logic that was previously in this hook).
        await modelRef.current.loadModel(
          pathOrUrl,
          nativeConfig,
          (progress) => {
            setDownloadProgress(progress);
          },
        );
        // Surface memory intel and subscribe to OS pressure warnings
        setMemoryWarning(null);
        setMemoryEstimate(modelRef.current.estimateMemory());
        setMemoryForecast(modelRef.current.getMemoryForecast());
        try {
          modelRef.current.setMemoryWarningCallback((level) => {
            setMemoryWarning(level);
          });
        } catch {
          // Non-critical — older native builds may not support warnings
        }
        setIsReady(true);
      }
    } catch (e: any) {
      setError(e.message || "Failed to load model");
      console.error(e);
    }
  }, [pathOrUrl, nativeConfig]);

  useEffect(() => {
    if (autoLoad) {
      load();
    }
  }, [autoLoad, load]);

  const generate = useCallback(
    async (prompt: string): Promise<string> => {
      if (!modelRef.current || !isReady) {
        throw new Error("Model not ready");
      }

      setIsGenerating(true);
      try {
        const response = await modelRef.current.execute(
          [{ type: "text", text: prompt }],
          undefined
        );
        refreshMemorySummary();
        setMemoryForecast(modelRef.current.getMemoryForecast());
        return response;
      } catch (e: any) {
        setError(e.message || "Generation failed");
        throw e;
      } finally {
        setIsGenerating(false);
      }
    },
    [isReady, refreshMemorySummary],
  );

  const reset = useCallback(() => {
    if (modelRef.current) {
      modelRef.current.resetConversation();
    }
  }, []);

  const deleteModel = useCallback(
    async (fileName?: string): Promise<void> => {
      const resolvedName = fileName ?? extractFileName(pathOrUrl);
      ModelRegistry.deleteFile(resolvedName);
      setIsReady(false);
      setDownloadProgress(0);
    },
    [pathOrUrl],
  );

  return {
    model: modelRef.current,
    isReady,
    isGenerating,
    downloadProgress,
    error,
    generate,
    reset,
    deleteModel,
    load,
    memoryTracker: modelRef.current?.memoryTracker ?? null,
    memorySummary,
    memoryEstimate,
    memoryForecast,
    memoryWarning,
  };
}
