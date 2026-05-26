import type { HybridObject } from "react-native-nitro-modules";

/**
 * Backend types for LLM inference.
 * - 'cpu': CPU inference (slowest, always available)
 * - 'gpu': GPU acceleration (fast, recommended for most devices)
 * - 'npu': NPU/Neural Engine (fastest on supported hardware)
 *
 * @remarks
 * NPU acceleration requires compatible hardware (e.g., Qualcomm Hexagon, MediaTek APU).
 * If NPU is unavailable, LiteRT-LM automatically falls back to GPU.
 */
export type Backend = "cpu" | "gpu" | "npu";

/**
 * Message roles for conversation.
 */
export type Role = "user" | "model" | "system";

/**
 * Definition for a function/tool that the model can request to execute.
 */
export interface ToolDefinition {
  /** Name of the function/tool */
  name: string;
  /** Human-readable description of what the function/tool does */
  description: string;
  /** JSON schema defining parameter names and types (stringified) */
  parametersJson: string;
}

/**
 * The part type for a multimodal message content part.
 */
export type PartType = "text" | "image" | "audio";

/**
 * A part of a unified multimodal message payload.
 */
export interface MultimodalPart {
  /** The part type: 'text', 'image', or 'audio' */
  type: PartType;
  /** The plain text content, if type is 'text' */
  text?: string;
  /** Raw image binary data, if type is 'image' (zero-copy ArrayBuffer mapping) */
  imageBuffer?: ArrayBuffer;
  /** Raw audio binary data, if type is 'audio' (zero-copy ArrayBuffer mapping) */
  audioBuffer?: ArrayBuffer;
}

/**
 * Configuration options for loading an LLM.
 */
export interface LLMConfig {
  /**
   * System prompt to set the model's behavior.
   * This is prepended to the conversation to guide model responses.
   * @example "You are a helpful coding assistant."
   */
  systemPrompt?: string;

  /**
   * Primary compute backend for text generation.
   * - 'cpu': CPU inference (safe default, always available)
   * - 'gpu': GPU acceleration (fast, Metal on iOS, GPU delegate on Android)
   * - 'npu': NPU/Neural Engine (fastest on supported devices)
   *
   * If not specified, defaults to 'cpu'.
   * If specified backend is unavailable, falls back automatically.
   *
   * @remarks
   * Vision encoder is always set to GPU (required by Gemma models).
   * Audio encoder is always set to CPU (optimal for audio processing).
   *
   * @default 'cpu'
   */
  backend?: Backend;

  /**
   * Maximum number of tokens to generate.
   * @default 1024
   */
  maxTokens?: number;

  /**
   * Sampling temperature (0.0 = deterministic, 1.0 = creative).
   * @default 0.7
   */
  temperature?: number;

  /**
   * Top-K sampling (number of top tokens to consider).
   * @default 40
   */
  topK?: number;

  /**
   * Top-P (nucleus) sampling threshold.
   * @default 0.95
   */
  topP?: number;

  /**
   * Whether to run engine validation after loading the model.
   * When enabled, sends a quick test inference ("Hi") and waits up to 30s
   * for a response to confirm the backend works. This is useful for GPU/NPU
   * backends that may silently fail during inference (they can initialize
   * without error but produce no tokens).
   *
   * Validation is **always a no-op on CPU** — the CPU backend is inherently
   * reliable and never needs validation.
   *
   * Disabled by default because it adds significant latency (5-30s) to model loading.
   * Enable only to catch GPU/NPU silent failure issues during development.
   *
   * @default false
   */
  validate?: boolean;

  /**
   * Whether this is a multimodal model.
   * When enabled, the engine handles image/audio tokens properly.
   * If not specified, the system will fall back to filename sniffing.
   */
  multimodal?: boolean;

  /**
   * List of tools/functions that the model can call.
   */
  tools?: ToolDefinition[];

  /**
   * Whether to enable speculative decoding (multi-token prediction) if supported by the model.
   * @default false
   */
  enableSpeculativeDecoding?: boolean;
}

/**
 * A simple message in the conversation.
 * For multimodal, use sendMessageWithImage/sendMessageWithAudio instead.
 */
export interface Message {
  /** Role of the message sender */
  role: Role;
  /** Text content of the message */
  content: string;
}

/**
 * Generation statistics returned after completion.
 */
export interface GenerationStats {
  /** Number of tokens in the prompt */
  promptTokens: number;
  /** Number of tokens generated */
  completionTokens: number;
  /** Total tokens (prompt + completion) */
  totalTokens: number;
  /** Time to first token in milliseconds */
  timeToFirstToken: number;
  /** Total generation time in milliseconds */
  totalTime: number;
  /** Tokens per second */
  tokensPerSecond: number;
}

/**
 * Real memory usage statistics from the native runtime.
 * Measured from OS-level APIs, not estimated.
 */
export interface MemoryUsage {
  /** Native heap allocated bytes (Debug.getNativeHeapAllocatedSize on Android, malloc_size on iOS) */
  nativeHeapBytes: number;
  /** Total process resident set size (RSS) in bytes */
  residentBytes: number;
  /** Available system memory in bytes */
  availableMemoryBytes: number;
  /** Whether the system considers memory low */
  isLowMemory: boolean;
}

/**
 * LiteRT-LM: High-performance LLM inference engine.
 * Supports Gemma 4, Gemma 3n, Phi-4, Qwen, and other .litertlm models.
 *
 * @example
 * ```typescript
 * const llm = createLLM();
 * llm.loadModel('/path/to/gemma-4-E2B-it.litertlm', { backend: 'cpu' });
 *
 * // Blocking generation
 * const response = llm.sendMessage('What is the capital of France?');
 *
 * // Streaming generation
 * llm.sendMessageAsync('Tell me a story', (token, done) => {
 *   process.stdout.write(token);
 * });
 *
 * llm.close();
 * ```
 */
export interface LiteRTLM extends HybridObject<{
  ios: "swift";
  android: "kotlin";
}> {
  /**
   * Load a .litertlm model file.
   * @param config Optional configuration for backend and sampling.
   * @throws Error if the model cannot be loaded.
   */
  loadModel(modelPath: string, config?: LLMConfig): Promise<void>;

  /**
   * Send a text message and get the complete response (blocking).
   * @param message User message text.
   * @returns The model's response text.
   */
  sendMessage(message: string): Promise<string>;

  /**
   * Send a text message with an image (multimodal).
   * @param message User message text.
   * @param imagePath Absolute path to an image file.
   * @returns The model's response text.
   */
  sendMessageWithImage(message: string, imagePath: string): Promise<string>;

  /**
   * Send a text message with an image and get a streaming response.
   * Tokens are delivered via callback as they are generated.
   * @param message User message text.
   * @param imagePath Absolute path to an image file.
   * @param onToken Callback invoked for each token (token, isDone).
   */
  sendMessageWithImageAsync(
    message: string,
    imagePath: string,
    onToken: (token: string, done: boolean) => void,
  ): Promise<void>;

  /**
   * Download a model file from a URL.
   * @param url URL to download from.
   * @param fileName Filename to save as (in app's files directory).
   * @param onProgress Callback for download progress (0.0 - 1.0).
   * @returns Absolute path to the downloaded file.
   */
  downloadModel(
    url: string,
    fileName: string,
    onProgress?: (progress: number) => void,
  ): Promise<string>;

  /**
   * Delete a downloaded model file.
   * @param fileName Filename to delete (in app's files directory).
   */
  deleteModel(fileName: string): Promise<void>;

  /**
   * Send a text message with audio (multimodal).
   * @param message User message text.
   * @param audioPath Absolute path to an audio file (WAV).
   * @returns The model's response text.
   */
  sendMessageWithAudio(message: string, audioPath: string): Promise<string>;

  /**
   * Send a text message with audio and get a streaming response.
   * Tokens are delivered via callback as they are generated.
   * @param message User message text.
   * @param audioPath Absolute path to an audio file (WAV).
   * @param onToken Callback invoked for each token (token, isDone).
   */
  sendMessageWithAudioAsync(
    message: string,
    audioPath: string,
    onToken: (token: string, done: boolean) => void,
  ): Promise<void>;

  /**
   * Send a unified multimodal message containing text and/or zero-copy binary buffers.
   * @param parts The message content parts (text, image, and/or audio).
   * @returns The model's response text.
   */
  sendMultimodalMessage(parts: MultimodalPart[]): Promise<string>;

  /**
   * Send a message with streaming response.
   * Tokens are delivered via callback as they are generated.
   * @param message User message text.
   * @param onToken Callback invoked for each token (token, isDone).
   */
  sendMessageAsync(
    message: string,
    onToken: (token: string, done: boolean) => void,
  ): Promise<void>;

  /**
   * Get the current conversation history.
   * @returns Array of messages in the conversation.
   */
  getHistory(): Message[];

  /**
   * Clear the conversation context and start fresh.
   */
  resetConversation(): void;

  /**
   * Check if a model is loaded and ready for inference.
   */
  isReady(): boolean;

  /**
   * Get the last generation statistics.
   */
  getStats(): GenerationStats;

  /**
   * Count tokens in a text string. Returns -1 if unavailable.
   */
  countTokens(text: string): number;

  /**
   * Get real memory usage from the native runtime.
   * Uses OS-level APIs to report actual memory consumption.
   */
  getMemoryUsage(): MemoryUsage;

  /**
   * Release all native resources.
   * Call this when done with the LLM instance.
   */
  close(): void;
}
