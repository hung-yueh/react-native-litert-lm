import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { LLMConfig } from "./index";
import { createLLM } from "./modelFactory";
import type { LiteRTLMInstance } from "./modelFactory";
import type { MemoryTracker, MemoryTrackerSummary } from "./memoryTracker";

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
}

/**
 * Extract a filename from a URL or file path.
 */
function extractFileName(pathOrUrl: string): string {
  return pathOrUrl.split("/").pop() || "model.bin";
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
  const [memorySummary, setMemorySummary] = useState<MemoryTrackerSummary | null>(null);

  // Destructure config into primitive values for stable dependency arrays.
  // This prevents infinite re-render loops when consumers pass inline config
  // objects (e.g. useModel(url, { backend: 'cpu' })) without useMemo.
  const autoLoad = config?.autoLoad ?? true;
  const enableMemoryTracking = config?.enableMemoryTracking ?? false;
  const maxMemorySnapshots = config?.maxMemorySnapshots ?? 256;
  const backend = config?.backend;
  const systemPrompt = config?.systemPrompt;
  const maxTokens = config?.maxTokens;
  const temperature = config?.temperature;
  const topK = config?.topK;
  const topP = config?.topP;
  const validate = config?.validate;
  const multimodal = config?.multimodal;
  const tools = config?.tools;
  const toolsKey = JSON.stringify(tools); // stable primitive for deps — avoids infinite re-renders from non-primitive array references
  const enableSpeculativeDecoding = config?.enableSpeculativeDecoding;

  // Build a stable config object from the destructured primitives
  const nativeConfig = useMemo<LLMConfig>(
    () => ({
      ...(backend !== undefined && { backend }),
      ...(systemPrompt !== undefined && { systemPrompt }),
      ...(maxTokens !== undefined && { maxTokens }),
      ...(temperature !== undefined && { temperature }),
      ...(topK !== undefined && { topK }),
      ...(topP !== undefined && { topP }),
       ...(validate !== undefined && { validate }),
      ...(multimodal !== undefined && { multimodal }),
      ...(tools !== undefined && { tools }),
      ...(enableSpeculativeDecoding !== undefined && {
        enableSpeculativeDecoding,
      }),
    }),
    [backend, systemPrompt, maxTokens, temperature, topK, topP, validate, multimodal, enableSpeculativeDecoding, toolsKey],
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
        return new Promise<string>((resolve, reject) => {
          let fullResponse = "";
          try {
            modelRef.current?.sendMessageAsync(
              prompt, (token: string, done: boolean) => {
                fullResponse += token;
                if (done) {
                  refreshMemorySummary();
                  resolve(fullResponse);
                }
              }).catch(reject);
          } catch (e: any) {
            reject(e);
          }
        });
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
      if (modelRef.current) {
        const resolvedName = fileName ?? extractFileName(pathOrUrl);
        await modelRef.current.deleteModel(resolvedName);
        setIsReady(false);
        setDownloadProgress(0);
      }
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
  };
}
