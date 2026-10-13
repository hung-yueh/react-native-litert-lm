import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { LLMConfig, MemoryWarningLevel } from "./index";
import { createLLM } from "./modelFactory";
import { createEmbedder, type EmbedderInstance, type EmbeddingInput } from "./embedder";
import type { EmbedderConfig, EmbeddingOptions } from "./specs/Embedder.nitro";
import type { LiteRTLMInstance } from "./modelFactory";
import type { MemoryTracker, MemoryTrackerSummary } from "./memoryTracker";
import type { MemoryEstimate, MemoryForecast } from "./core/memoryEstimator";
import { ModelRegistry } from "./modelRegistry";
import { extractFileName } from "./core/modelPath";
import type { ModelInfo } from "./core/modelInfo";
import { LiteRTLMError, type LiteRTLMErrorCode } from "./core/errors";

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
  /**
   * Typed code of the last load/generate failure (null if none or untyped),
   * e.g. `'MEMORY_ESTIMATE_EXCEEDED'`, `'MODEL_TYPE_MISMATCH'`, `'INVALID_INPUT'`.
   */
  errorCode: LiteRTLMErrorCode | null;
  /**
   * What the loaded bundle declares about itself — modalities, context limit,
   * backends, thinking/tool support, sampler defaults (null until loaded).
   */
  modelInfo: ModelInfo | null;
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
  const [errorCode, setErrorCode] = useState<LiteRTLMErrorCode | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
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
  const thinking = config?.thinking;
  const toolsKey = tools ? JSON.stringify(tools) : undefined;
  const thinkingKey = thinking ? JSON.stringify(thinking) : undefined;

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
      ...(thinking !== undefined && { thinking }),
    }),
    // `tools`/`thinking` are tracked via their serialized keys so inline objects stay stable.
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
      thinkingKey,
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
    setErrorCode(null);
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
        // Surface bundle + memory intel and subscribe to OS pressure warnings
        setModelInfo(modelRef.current.getModelInfo());
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
      setErrorCode(e instanceof LiteRTLMError ? e.code : null);
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
        setErrorCode(e instanceof LiteRTLMError ? e.code : null);
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
    errorCode,
    modelInfo,
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

export interface UseEmbedderConfig extends EmbedderConfig {
  /** Load as soon as the hook mounts (default true) */
  autoLoad?: boolean;
}

export interface UseEmbedderResult {
  embedder: EmbedderInstance | null;
  isReady: boolean;
  downloadProgress: number;
  error: string | null;
  /** Typed code of the last failure, e.g. `'MODEL_TYPE_MISMATCH'` */
  errorCode: LiteRTLMErrorCode | null;
  /** What the loaded bundle declares: dimension, input lengths, modalities (null until loaded) */
  modelInfo: ModelInfo | null;
  /** One vector for one input (rejects until ready) */
  embed: (input: EmbeddingInput, options?: EmbeddingOptions) => Promise<Float32Array>;
  /** One vector per input, in one engine call */
  embedBatch: (inputs: EmbeddingInput[], options?: EmbeddingOptions) => Promise<Float32Array[]>;
  load: () => Promise<void>;
}

/**
 * React hook managing an embedding model's lifecycle: download, pre-flight,
 * load, and cleanup on unmount — the embedding counterpart of `useModel`.
 *
 * @example
 * ```tsx
 * const { isReady, embed } = useEmbedder(EMBEDDING_GEMMA_2_TEXT_270M);
 * const vector = isReady ? await embed(EmbeddingTask.searchQuery(text), { normalize: true }) : null;
 * ```
 */
export function useEmbedder(pathOrUrl: string, config?: UseEmbedderConfig): UseEmbedderResult {
  const embedderRef = useRef<EmbedderInstance | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<LiteRTLMErrorCode | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Primitives keep the config stable across renders with inline objects.
  const autoLoad = config?.autoLoad ?? true;
  const backend = config?.backend;
  const visionBackend = config?.visionBackend;
  const audioBackend = config?.audioBackend;
  const maxInputLength = config?.maxInputLength;
  const visionTokensPerImage = config?.visionTokensPerImage;
  const activationDataType = config?.activationDataType;
  const forceLoad = config?.forceLoad;
  const embedderConfig = useMemo<EmbedderConfig>(
    () => ({
      ...(backend !== undefined && { backend }),
      ...(visionBackend !== undefined && { visionBackend }),
      ...(audioBackend !== undefined && { audioBackend }),
      ...(maxInputLength !== undefined && { maxInputLength }),
      ...(visionTokensPerImage !== undefined && { visionTokensPerImage }),
      ...(activationDataType !== undefined && { activationDataType }),
      ...(forceLoad !== undefined && { forceLoad }),
    }),
    [backend, visionBackend, audioBackend, maxInputLength, visionTokensPerImage, activationDataType, forceLoad],
  );

  useEffect(() => {
    embedderRef.current = createEmbedder();
    setIsReady(false);
    return () => {
      try {
        embedderRef.current?.close();
      } catch (e) {
        console.warn("Failed to close embedder", e);
      }
    };
  }, []);

  const load = useCallback(async () => {
    setIsReady(false);
    setError(null);
    setErrorCode(null);
    setDownloadProgress(0);
    try {
      if (embedderRef.current) {
        await embedderRef.current.loadModel(pathOrUrl, embedderConfig, setDownloadProgress);
        setModelInfo(embedderRef.current.getModelInfo());
        setIsReady(true);
      }
    } catch (e: any) {
      setError(e.message || "Failed to load embedding model");
      setErrorCode(e instanceof LiteRTLMError ? e.code : null);
    }
  }, [pathOrUrl, embedderConfig]);

  useEffect(() => {
    if (autoLoad) load();
  }, [autoLoad, load]);

  const embed = useCallback(
    (input: EmbeddingInput, options?: EmbeddingOptions) => {
      if (!embedderRef.current || !isReady) {
        return Promise.reject(new LiteRTLMError("MODEL_NOT_LOADED", "The embedding model is not loaded yet."));
      }
      return embedderRef.current.embed(input, options);
    },
    [isReady],
  );

  const embedBatch = useCallback(
    (inputs: EmbeddingInput[], options?: EmbeddingOptions) => {
      if (!embedderRef.current || !isReady) {
        return Promise.reject(new LiteRTLMError("MODEL_NOT_LOADED", "The embedding model is not loaded yet."));
      }
      return embedderRef.current.embedBatch(inputs, options);
    },
    [isReady],
  );

  return {
    embedder: embedderRef.current,
    isReady,
    downloadProgress,
    error,
    errorCode,
    modelInfo,
    embed,
    embedBatch,
    load,
  };
}
