import type { HybridObject } from "react-native-nitro-modules";
import type {
  ActivationDataType,
  Backend,
  MemoryUsage,
  MultimodalPart,
} from "./LiteRTLM.nitro";

/** Load-time settings for an embedding model. */
export interface EmbedderConfig {
  /** Backend for the text encoder (default: the bundle's preference, else CPU) */
  backend?: Backend;
  /** Backend for the image encoder (default: the bundle's preference; unset = none) */
  visionBackend?: Backend;
  /** Backend for the audio encoder (default: the bundle's preference; unset = none) */
  audioBackend?: Backend;
  /**
   * Longest input, in tokens, the encoder is prepared for. Picks the
   * smallest compiled signature that fits (EmbeddingGemma 2: 128–8192), so a
   * smaller value means less memory and faster calls.
   */
  maxInputLength?: number;
  /** Visual tokens per image (EmbeddingGemma 2: 70 or 140) */
  visionTokensPerImage?: number;
  /** Activation precision; support depends on model and backend */
  activationDataType?: ActivationDataType;
  /** Skip the pre-flight memory check (like `LLMConfig.forceLoad`) */
  forceLoad?: boolean;
}

/** Per-call embedding settings. */
export interface EmbeddingOptions {
  /** Return a unit-length vector, so dot product equals cosine similarity */
  normalize?: boolean;
  /**
   * Matryoshka truncation: keep only the first N dimensions (EmbeddingGemma 2:
   * 128, 256, 512 or 768). Smaller vectors cost less to store and compare.
   */
  outputSize?: number;
  /** Add the model's special tokens around text input (engine default when unset) */
  insertSpecialTokens?: boolean;
  /** Visual tokens for images in this call */
  visionTokensPerImage?: number;
}

/**
 * A loaded embedding model (LiteRT-LM EmbeddingEngine). Use `createEmbedder()`
 * rather than this raw object: it adds downloads, the pre-flight checks and
 * `Float32Array` results.
 */
export interface Embedder
  extends HybridObject<{ ios: "swift"; android: "kotlin" }> {
  loadModel(modelPath: string, config?: EmbedderConfig): Promise<void>;
  /** One vector for one input; the buffer holds little-endian float32 values. */
  embed(parts: MultimodalPart[], options?: EmbeddingOptions): Promise<ArrayBuffer>;
  /** One vector per input, computed in a single engine call. */
  embedBatch(inputs: MultimodalPart[][], options?: EmbeddingOptions): Promise<ArrayBuffer[]>;
  isReady(): boolean;
  getMemoryUsage(): MemoryUsage;
  /** Free the engine; the instance can load again. */
  unload(): Promise<void>;
  close(): void;
}
