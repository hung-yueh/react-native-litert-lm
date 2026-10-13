import { NitroModules } from "react-native-nitro-modules";
import type { Embedder, EmbedderConfig, EmbeddingOptions } from "./specs/Embedder.nitro";
import type { MemoryUsage, MultimodalPart } from "./specs/LiteRTLM.nitro";
import { ModelRegistry } from "./modelRegistry";
import { inspectModel, tryGetDeviceInfo } from "./modelInspector";
import { LiteRTLMError, MemoryError } from "./core/errors";
import { toLiteRTLMError, type NativePhase } from "./core/nativeError";
import { estimateMemory } from "./core/memoryEstimator";
import { preflightEmbedder, type ModelInfo } from "./core/modelInfo";

/** Text, or the same multimodal parts `execute()` takes. */
export type EmbeddingInput = string | MultimodalPart[];

/** A loaded embedding model. Create one with {@link createEmbedder}. */
export interface EmbedderInstance {
  /**
   * Download (if a URL) and load an embedding bundle, after the same pre-flight
   * checks as `createLLM().loadModel()`: model type, runtime version, backend,
   * memory. Rejects an LLM bundle with `MODEL_TYPE_MISMATCH`.
   */
  loadModel(
    pathOrUrl: string,
    config?: EmbedderConfig,
    onDownloadProgress?: (progress: number) => void,
  ): Promise<void>;
  /**
   * One vector for one input. The `Float32Array` views native memory — no
   * copy, no JSON. Apply an {@link EmbeddingTask} prefix to text first.
   */
  embed(input: EmbeddingInput, options?: EmbeddingOptions): Promise<Float32Array>;
  /** One vector per input, computed in a single engine call. */
  embedBatch(inputs: EmbeddingInput[], options?: EmbeddingOptions): Promise<Float32Array[]>;
  /** What the loaded bundle declares (dimension, input lengths, modalities…). */
  getModelInfo(): ModelInfo | null;
  isReady(): boolean;
  getMemoryUsage(): MemoryUsage;
  /** Free the engine; the instance can load again. */
  unload(): Promise<void>;
  /** Free the engine for good. */
  close(): void;
}

/** A native call whose failure, thrown or rejected, becomes a typed LiteRTLMError. */
async function typed<T>(phase: NativePhase, call: () => Promise<T>): Promise<T> {
  try {
    return await call();
  } catch (e) {
    throw toLiteRTLMError(e, phase);
  }
}

const toParts = (input: EmbeddingInput): MultimodalPart[] =>
  typeof input === "string"
    ? [{ type: "text", text: input }]
    : input.map((part) =>
        part.path?.startsWith("file://") ? { ...part, path: part.path.slice(7) } : part,
      );

/**
 * Create an on-device embedding model (LiteRT-LM EmbeddingEngine) — for
 * semantic search, RAG, clustering, classification and deduplication without
 * a server. Separate from `createLLM()`: an embedder and an LLM are separate
 * engines and can be loaded side by side.
 *
 * @example
 * ```ts
 * const embedder = createEmbedder();
 * await embedder.loadModel(EMBEDDING_GEMMA_2_TEXT_270M);
 * const q = await embedder.embed(EmbeddingTask.searchQuery("cheap flights to Tokyo"), {
 *   normalize: true,
 *   outputSize: 256,
 * });
 * const docs = await embedder.embedBatch(notes.map(EmbeddingTask.document), {
 *   normalize: true,
 *   outputSize: 256,
 * });
 * const best = topK(q, docs, 3);
 * ```
 */
export function createEmbedder(): EmbedderInstance {
  const native = NitroModules.createHybridObject<Embedder>("Embedder");
  let modelInfo: ModelInfo | null = null;

  return {
    async loadModel(pathOrUrl, config, onDownloadProgress) {
      const modelPath = await ModelRegistry.resolveModel(pathOrUrl, { onProgress: onDownloadProgress });

      let info: ModelInfo | null = null;
      try {
        info = await inspectModel(modelPath);
      } catch (e) {
        if (e instanceof LiteRTLMError && e.code === "MODEL_NOT_FOUND") throw e;
        console.warn(
          `[react-native-litert-lm] Could not read the metadata of ${modelPath} ` +
            `(${e instanceof Error ? e.message : String(e)}); loading without the bundle pre-flight.`,
        );
      }
      let effective = config;
      if (info) {
        const preflight = preflightEmbedder(info, config, { device: tryGetDeviceInfo() });
        if (preflight.changed) effective = preflight.config;
        for (const note of preflight.notes) console.warn(`[react-native-litert-lm] ${note}`);
      }

      const fileSize = ModelRegistry.getFileSizeBytes(modelPath);
      if (fileSize >= 0 && !effective?.forceLoad) {
        const estimate = estimateMemory({
          modelFileSizeBytes: fileSize,
          availableMemoryBytes: native.getMemoryUsage().availableMemoryBytes,
          // No KV cache: activations scale with the input length instead.
          config: { backend: effective?.backend, maxContextTokens: effective?.maxInputLength ?? 512 },
        });
        if (estimate.verdict === "critical") {
          throw new MemoryError(
            `Refusing to load embedding model (${Math.round(fileSize / (1024 * 1024))} MB): ` +
              `${estimate.recommendation} Pass { forceLoad: true } to override.`,
            estimate,
          );
        }
      }

      await typed("load", () => native.loadModel(modelPath, effective));
      modelInfo = info;
    },

    async embed(input, options) {
      const buffer = await typed("embed", () => native.embed(toParts(input), options));
      return new Float32Array(buffer);
    },

    async embedBatch(inputs, options) {
      if (inputs.length === 0) return [];
      const buffers = await typed("embed", () => native.embedBatch(inputs.map(toParts), options));
      return buffers.map((buffer) => new Float32Array(buffer));
    },

    getModelInfo: () => modelInfo,
    isReady: () => native.isReady(),
    getMemoryUsage: () => native.getMemoryUsage(),

    async unload() {
      await native.unload();
      modelInfo = null;
    },

    close() {
      native.close();
      modelInfo = null;
    },
  };
}
