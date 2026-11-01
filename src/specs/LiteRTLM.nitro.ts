import type { HybridObject } from "react-native-nitro-modules";

/**
 * Backend types for LLM inference.
 * - 'cpu': CPU inference (always available)
 * - 'gpu': GPU acceleration (Metal on iOS, OpenCL on Android)
 * - 'npu': NPU (bundles compiled for a specific SoC, e.g. Qualcomm or MediaTek)
 *
 * @remarks
 * `recommendBackend(await inspectModel(path))` says which of these a bundle
 * supports on this device. If the engine can't be created on the requested
 * backend, `loadModel()` falls back to CPU.
 */
export type Backend = "cpu" | "gpu" | "npu";

/**
 * Message roles for conversation. `'tool'` turns carry a tool's result back
 * to the model (see `toolResultPart()`).
 */
export type Role = "user" | "model" | "system" | "tool";

/**
 * A tool call the model made. `parseToolCalls()` turns these (or the
 * `<tool_call>` markers in a response) into objects with parsed arguments.
 */
export interface ToolCall {
  /** Name of the tool the model wants to run */
  name: string;
  /** The call's arguments as a JSON object; integers stay integers */
  argumentsJson: string;
}

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
 * The part type for a multimodal message content part. A `'toolResult'`
 * part answers a tool call; a message of tool results takes no other parts.
 */
export type PartType = "text" | "image" | "audio" | "toolResult";

/**
 * A part of a unified multimodal message payload.
 */
export interface MultimodalPart {
  /** The part type: 'text', 'image', 'audio' or 'toolResult' */
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
  /** The tool's name, if type is 'toolResult' (as the model called it) */
  name?: string;
  /** The tool's result as JSON, if type is 'toolResult' — build it with `toolResultPart()` */
  responseJson?: string;
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
   * - 'cpu': CPU inference (always available)
   * - 'gpu': GPU acceleration (Metal on iOS, OpenCL on Android)
   * - 'npu': NPU (per-SoC bundles on supported devices)
   *
   * When omitted, `loadModel()` uses the bundle's preferred backend that this
   * device can run (see `recommendBackend()`), else CPU. If the engine can't
   * be created on the requested backend, it falls back to CPU.
   *
   * @remarks
   * Image and audio encoders run on the backends the bundle declares for
   * them; `multimodal: false` leaves them out.
   *
   * @default the bundle's preference, else 'cpu'
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
   * List of tools/functions that the model can call. Calls come back as
   * `<tool_call>` markers in the response (`parseToolCalls()` reads them);
   * your app runs the tool and answers with `toolResultPart()`.
   */
  tools?: ToolDefinition[];

  /**
   * Whether to enable speculative decoding (multi-token prediction) if supported by the model.
   * The session default; a conversation can override it
   * (`createConversation({ speculativeDecoding })`).
   * @default false
   */
  enableSpeculativeDecoding?: boolean;

  /**
   * A Jinja chat template that replaces the bundle's own — the fix for a
   * community bundle whose template is missing or wrong. Start from upstream's
   * `models/<family>/chat_template.jinja`; the engine adds the BOS token
   * itself, so the template must not. An invalid template fails the load.
   */
  chatTemplate?: string;

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
   * Both platforms (Android since LiteRT-LM 0.18's `Backend.CPU(threadCount)`).
   */
  numThreads?: number;

  /**
   * Cap on the visual tokens one image may use, set when the engine loads
   * (Gemma 4: 70, 140, 280, 560 or 1120; `inspectModel()` lists a bundle's
   * options). Lower caps cut image memory and latency;
   * `ExecuteOptions.visualTokenBudget` picks a budget per message, up to this cap.
   */
  maxVisionTokensPerImage?: number;

  /** GPU-backend options. */
  gpu?: GpuOptions;

  /**
   * Run the engine on a single thread — for debugging and reproducibility,
   * not speed.
   *
   * @remarks **iOS only** — the Android Kotlin SDK does not expose it.
   */
  singleThreadedExecution?: boolean;

  /**
   * Prefill chunk size (CPU backend only). Smaller chunks reduce peak
   * activation memory during long-prompt prefill at some latency cost.
   *
   * @remarks **iOS only** — the Android Kotlin SDK does not expose this.
   */
  prefillChunkSize?: number;

  /**
   * Activation tensor data type. Lower precision reduces activation memory:
   * 'f16' roughly halves activation memory vs 'f32'. Mainly a GPU setting —
   * GPU backends default to 'f16'; on CPU the engine picks its own kernels
   * and, for quantized bundles such as Gemma 4 E2B, generates identically
   * whatever the value. A type the model and backend cannot run fails the load.
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

/** GPU-backend options (`LLMConfig.gpu`). */
export interface GpuOptions {
  /**
   * Keep model weights resident in GPU memory with a Metal residency set
   * (iOS 18+), so they aren't paged between calls. Ignored on older iOS and
   * by non-GPU backends.
   *
   * @remarks **iOS only.**
   * @default false
   */
  metalResidencySet?: boolean;
}

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
   * Visual token budget for images in this message (Gemma 4: 70, 140, 280,
   * 560 or 1120), up to `LLMConfig.maxVisionTokensPerImage`. Lower values
   * reduce memory and time spent encoding images. Both platforms; on Android
   * the SDK's budget is process-wide, so messages that set it run one at a time.
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
   *
   * Supported on both platforms (Android since LiteRT-LM 0.17.0, which fixed
   * the JNI binding that aborted the process through 0.16.1).
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
 * One turn of the conversation transcript (`getHistory()`). Images and audio
 * appear as `[Image]` / `[Audio]` placeholders in `content`.
 */
export interface Message {
  /** Role of the message sender */
  role: Role;
  /**
   * Text content of the message. For a `'tool'` turn, the tool's result as
   * JSON; for a `'model'` turn that called tools, any text around the calls.
   */
  content: string;
  /** Tool calls the model made in this turn (`'model'` turns only) */
  toolCalls?: ToolCall[];
  /** The tool whose result this turn carries (`'tool'` turns only) */
  toolName?: string;
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
 * Runs any .litertlm bundle whose model family the engine supports (Gemma 4,
 * Gemma 3n, Qwen, LFM2, Phi-4, …).
 *
 * @example
 * ```typescript
 * const llm = createLLM();
 * await llm.loadModel('/path/to/gemma-4-E2B-it.litertlm');
 *
 * // Blocking generation
 * const response = await llm.execute([{ type: 'text', text: 'What is the capital of France?' }]);
 *
 * // Streaming generation
 * await llm.execute([{ type: 'text', text: 'Tell me a story' }], (token, done) => {
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
   *
   * Optionally seed the fresh context from a prior transcript and/or override
   * the session `systemPrompt` — the native primitive behind
   * `createConversation()` multi-conversation switching.
   *
   * @param historyJson JSON array of `Message` turns (the shape `getHistory()`
   *   returns) to seed the new conversation with. Tool exchanges replay as
   *   the model's tool calls and `'tool'` results. The seeded transcript is
   *   re-prefilled by the engine on the next message.
   * @param systemPrompt Override the session-level system prompt for the new
   *   conversation (undefined = keep the session default).
   * @param speculativeDecoding Turn speculative decoding on or off for the new
   *   conversation (undefined = the session default). `true` loads the
   *   bundle's drafter on first use if the engine started without it.
   *
   * `getHistory()` reflects the new transcript at once, and messages sent
   * after this run in the new conversation. A message sent before it that is
   * still generating or waiting to run is cancelled (its promise rejects with
   * `INFERENCE_CANCELLED`). iOS returns at once and reports a failure to
   * create the new conversation from the next `execute()`; Android waits up
   * to 2 s for a cancelled message to return.
   */
  resetConversation(historyJson?: string, systemPrompt?: string, speculativeDecoding?: boolean): void;

  /**
   * Check if a model is loaded and ready for inference. False the moment
   * `unload()` or `close()` is called.
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
   * Number of tokens held in the conversation KV cache (prompt + history +
   * generated tokens) as of the last completed message — it doesn't move
   * while one generates. The engine prefills the system prompt and tool
   * schemas with the first message, so the count includes them from then on:
   * right after `loadModel()` or a reset, it leaves them out. Returns -1 while
   * unknown: no model loaded, or just after a reset (on Android, until the
   * next message completes when the reset replays a transcript or gives up
   * waiting for a message it cancelled).
   *
   * Combine with `maxContextTokens` to warn before the context window
   * (and its memory) runs out — see `getMemoryForecast()` in the JS API.
   */
  getContextTokenCount(): number;

  /**
   * Release the loaded model and conversation, freeing all engine memory,
   * while keeping this instance reusable — call `loadModel()` again to reload.
   * Unlike `close()`, which permanently invalidates the instance.
   *
   * Reports not-ready and cancels a message still generating at once
   * (`INFERENCE_CANCELLED`); resolves once the engine is freed.
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
   * @param onToken Optional streaming callback - invoked per token with (token, isDone).
   *   The last call has isDone true. A stream that fails ends with an `Error: …`
   *   token; one that's cancelled (reset, unload, close) ends with an empty one.
   * @param options Optional per-message options (maxOutputTokens, visualTokenBudget)
   * @returns Promise resolving to the full response string. It rejects with
   *   `INFERENCE_CANCELLED` if `resetConversation()`, `loadModel()`,
   *   `unload()` or `close()` is called before the message finishes.
   */
  execute(
    parts: MultimodalPart[],
    onToken?: (token: string, done: boolean) => void,
    options?: ExecuteOptions,
  ): Promise<string>;

  /**
   * Release all native resources.
   * Call this when done with the LLM instance.
   *
   * Returns at once: the instance reports not-ready, stops memory warnings
   * and cancels a message still generating immediately; the engine is freed
   * in the background once that message has stopped. A model loaded
   * afterwards, on any instance, waits for that, so the two never sit in
   * memory together.
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
  /** Deletes the cached file and any partial download of it. */
  deleteFile(fileName: string): void;
  /**
   * Download `url` into the cache as `fileName`. A failed or interrupted
   * download keeps its partial file when the server sent a validator and the
   * size, and the next call for the same URL continues it with a Range
   * request. `sha256` (lowercase hex, "" = no check) is checked before the
   * file is moved into place.
   */
  downloadFile(
    url: string,
    fileName: string,
    headersJson: string,
    sha256: string,
    onProgress: (progress: number) => void
  ): Promise<string>;
}
