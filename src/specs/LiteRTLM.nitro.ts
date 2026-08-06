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
  /**
   * Local filesystem path to an image or audio file, if type is 'image' or 'audio'.
   * Takes precedence over `imageBuffer`/`audioBuffer` if both are set.
   */
  path?: string;
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
   * Total engine KV-cache budget in tokens (prompt + history + output combined).
   * Must be ≥ the total number of input tokens + desired output tokens.
   * For compiled bundles, should not exceed the bundle's compiled cache_length.
   * @default 4096
   */
  maxContextTokens?: number;

  /**
   * Maximum number of tokens the model generates per response.
   * For compiled bundles, must not exceed the bundle's compiled decode-chunk size.
   * Applies on both platforms (LiteRT-LM 0.15+); override per message via
   * `ExecuteOptions.maxOutputTokens`.
   *
   * @default 1024
   */
  maxOutputTokens?: number;

  /**
   * Maximum number of tokens to generate.
   * @deprecated Use `maxContextTokens` and `maxOutputTokens` instead.
   * When set alone (without the new fields), maps to both for backward compatibility.
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

  /**
   * Initialize the constraint provider (LLGuidance) so structured output can
   * be requested per message via `ExecuteOptions.responseSchema` /
   * `responseRegex` (LiteRT-LM 0.15+). Off by default — enabling adds a small
   * one-time initialization cost at conversation creation.
   * @default false
   */
  enableStructuredOutput?: boolean;

  /**
   * Session-default thinking/reasoning controls (Gemma 4, LiteRT-LM 0.15+).
   * Override per message via `ExecuteOptions.thinking`.
   */
  thinking?: ThinkingOptions;

  /**
   * Number of CPU threads for text generation (CPU backend).
   * Lower values reduce peak memory bandwidth pressure at some speed cost.
   *
   * @remarks **iOS only** — the Android Kotlin SDK does not expose thread control.
   */
  numThreads?: number;

  /**
   * Prefill chunk size (CPU backend only). Smaller chunks reduce peak
   * activation memory during long-prompt prefill at some latency cost.
   *
   * @remarks **iOS only** — the Android Kotlin SDK does not expose this.
   */
  prefillChunkSize?: number;

  /**
   * Activation tensor data type. Lower precision reduces activation memory:
   * 'f16' roughly halves activation memory vs 'f32'.
   *
   * @remarks **iOS only** — the Android Kotlin SDK does not expose this.
   */
  activationDataType?: ActivationDataType;

  /**
   * Path to a LoRA adapter weights file to apply on top of the base model.
   * Lets one base model serve multiple tasks without shipping full model copies.
   */
  loraPath?: string;

  /**
   * Path to an audio LoRA adapter weights file.
   */
  audioLoraPath?: string;

  /**
   * LoRA rank of the adapter (must match how the adapter was trained).
   *
   * @remarks **iOS only** — the Android Kotlin SDK derives the rank internally.
   */
  loraRank?: number;

  /**
   * Stream tool-call tokens through the token callback as they are generated
   * (LiteRT-LM v0.14+). Tool-call content arrives wrapped in channel markers;
   * use `executeWithEvents()` to receive them as typed events instead of raw text.
   * @default false
   */
  streamToolCalls?: boolean;

  /**
   * Channel name used to delimit streamed tool-call tokens.
   * @default "tool_call"
   */
  toolCallChannelName?: string;

  /**
   * Skip the automatic pre-flight memory check in `loadModel()` and load even
   * when the estimate says the model likely does not fit. JS-side only —
   * the native engine ignores this field.
   * @default false
   */
  forceLoad?: boolean;
}

/**
 * Activation tensor data types (memory/precision trade-off).
 */
export type ActivationDataType = "f32" | "f16" | "i16" | "i8";

/**
 * Thinking/reasoning generation controls (Gemma 4 models, LiteRT-LM 0.15+).
 * Set in `LLMConfig.thinking` as the session default, or per message via
 * `ExecuteOptions.thinking`.
 */
export interface ThinkingOptions {
  /**
   * Whether thinking/reasoning generation is enabled.
   * @default true (engine default)
   */
  enabled?: boolean;

  /**
   * Token budget for reasoning generation. `-1` means unlimited.
   * Lower budgets reduce latency and memory at some quality cost on
   * reasoning-heavy prompts.
   * @default -1
   */
  tokenBudget?: number;
}

/**
 * Per-message options for `execute()` (LiteRT-LM v0.14+).
 */
export interface ExecuteOptions {
  /**
   * Cap the number of output tokens for this message only, overriding the
   * session-level `maxOutputTokens`. Both platforms (LiteRT-LM 0.15+).
   */
  maxOutputTokens?: number;

  /**
   * Visual token budget for this message (multimodal models).
   * Lower values reduce memory spent encoding images.
   *
   * @remarks **iOS only.**
   */
  visualTokenBudget?: number;

  /**
   * Constrain this response to match a JSON Schema (constrained decoding via
   * LLGuidance, LiteRT-LM 0.15+). Pass the schema as a JSON string, e.g.
   * `JSON.stringify({ type: 'object', properties: { name: { type: 'string' } } })`.
   * The engine guarantees the output parses against the schema.
   *
   * Requires `enableStructuredOutput: true` in `LLMConfig` at load time.
   * Takes precedence over `responseRegex` when both are set.
   *
   * @remarks For best schema adherence, use low-temperature sampling
   * (`temperature: 0`) in the load config.
   */
  responseSchema?: string;

  /**
   * Constrain this response to match a regular expression (constrained
   * decoding via LLGuidance, LiteRT-LM 0.15+).
   *
   * Requires `enableStructuredOutput: true` in `LLMConfig` at load time.
   * Ignored when `responseSchema` is also set.
   */
  responseRegex?: string;

  /**
   * Multiplicative repetition penalty for this message (HuggingFace style).
   * `1.0` = no penalty; values are clamped to ≥ 1.0 by the engine.
   * Applies only to tokens generated in this response.
   */
  repetitionPenalty?: number;

  /**
   * Subtractive presence penalty (OpenAI style): subtracted from a token's
   * logit once it has appeared in the generated window. Positive values
   * discourage repetition. @default 0
   */
  presencePenalty?: number;

  /**
   * Subtractive frequency penalty (OpenAI style): subtracted per prior
   * occurrence of the token in the generated window. @default 0
   */
  frequencyPenalty?: number;

  /**
   * How many recent generated tokens the penalties consider.
   * `0` = the entire generation history. @default 0
   */
  penaltyWindowSize?: number;

  /**
   * Ban exact n-gram repeats of this size during generation
   * (`0` disables). Useful against looping output.
   */
  noRepeatNgramSize?: number;

  /**
   * Window of recent tokens checked for repeating n-grams.
   * `0` = entire generation history. @default 0
   */
  noRepeatNgramWindowSize?: number;

  /**
   * Token IDs that must never be generated in this response —
   * their logits are forced to -inf.
   */
  suppressTokens?: number[];

  /**
   * Thinking/reasoning override for this message
   * (defaults to `LLMConfig.thinking`).
   */
  thinking?: ThinkingOptions;
}

/**
 * Severity of an OS memory-pressure notification.
 * - 'moderate': the system is under pressure; consider freeing caches.
 * - 'critical': the app is close to being killed; free memory now.
 */
export type MemoryWarningLevel = "moderate" | "critical";

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
   * @deprecated Use ModelRegistry instead.
   */
  downloadModel(
    url: string,
    fileName: string,
    onProgress?: (progress: number) => void,
  ): Promise<string>;

  /**
   * Delete a downloaded model file.
   * @deprecated Use ModelRegistry instead.
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
   * Number of tokens currently held in the conversation KV cache
   * (prompt + history + generated tokens). Returns -1 if unavailable.
   *
   * Combine with `maxContextTokens` to warn before the context window
   * (and its memory) runs out — see `getMemoryForecast()` in the JS API.
   */
  getContextTokenCount(): number;

  /**
   * Release the loaded model and conversation, freeing all engine memory,
   * while keeping this instance reusable — call `loadModel()` again to reload.
   * Unlike `close()`, which permanently invalidates the instance.
   */
  unload(): Promise<void>;

  /**
   * Register a callback fired when the OS signals memory pressure
   * (`onTrimMemory` on Android, dispatch memory-pressure source on iOS).
   * Only one callback is active at a time; registering replaces the previous one.
   */
  setMemoryWarningCallback(
    onWarning: (level: MemoryWarningLevel, usage: MemoryUsage) => void,
  ): void;

  /**
   * Remove the memory warning callback registered via `setMemoryWarningCallback`.
   */
  clearMemoryWarningCallback(): void;

  /**
   * Execute a unified multimodal inference request.
   *
   * A single deep entry point that replaces all modalality-specific send methods.
   * Accepts text, image (path or buffer), and audio (path or buffer) parts.
   * When onToken is provided, streams tokens incrementally; otherwise resolves
   * with the complete response.
   *
   * @param parts Array of content parts (text, image, audio)
   * @param onToken Optional streaming callback - invoked per token with (token, isDone)
   * @param options Optional per-message options (maxOutputTokens, visualTokenBudget)
   * @returns Promise resolving to the full response string
   */
  execute(
    parts: MultimodalPart[],
    onToken?: (token: string, done: boolean) => void,
    options?: ExecuteOptions,
  ): Promise<string>;

  /**
   * Release all native resources.
   * Call this when done with the LLM instance.
   */
  close(): void;
}

export interface ModelFile {
  fileName: string;
  absolutePath: string;
  sizeBytes: number;
  lastModifiedMs: number;
}

export interface ModelStore extends HybridObject<{
  ios: "swift";
  android: "kotlin";
}> {
  isCached(fileName: string): boolean;
  getFilePath(fileName: string): string;
  /**
   * Size in bytes of an arbitrary file path (not restricted to the model cache).
   * Returns -1 if the file does not exist. Used by the pre-flight memory estimator.
   */
  getFileSizeBytes(absolutePath: string): number;
  listCachedFiles(): ModelFile[];
  deleteFile(fileName: string): void;
  downloadFile(
    url: string,
    fileName: string,
    headersJson: string,
    onProgress: (progress: number) => void
  ): Promise<string>;
}
