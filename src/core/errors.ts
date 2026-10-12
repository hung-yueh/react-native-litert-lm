import type { MemoryEstimate } from "./memoryEstimator";
import type { NativeStatus } from "./nativeError";

/**
 * Machine-readable error codes for react-native-litert-lm failures.
 * Branch on `error.code` instead of parsing messages.
 *
 * | Code | Meaning |
 * | --- | --- |
 * | `MEMORY_ESTIMATE_EXCEEDED` | `loadModel()` refused: the pre-flight estimate says it won't fit |
 * | `OUT_OF_MEMORY` | The engine ran out of memory (`RESOURCE_EXHAUSTED`) |
 * | `MODEL_NOT_FOUND` | No file at the model path |
 * | `INVALID_MODEL` | The file is not a readable `.litertlm` bundle |
 * | `MODEL_NOT_LOADED` | Inference before `loadModel()` (or after `unload()`) |
 * | `MODEL_TYPE_MISMATCH` | An embedding bundle loaded as an LLM, or vice versa |
 * | `RUNTIME_TOO_OLD` | The bundle needs a newer LiteRT-LM than this package ships |
 * | `ENGINE_INIT_FAILED` | Engine creation failed for another reason |
 * | `INVALID_CONFIG` | The load config was rejected (unsupported backend/dtype, …) |
 * | `INVALID_INPUT` | A message or its options were rejected (bad schema, missing image, …) |
 * | `INFERENCE_FAILED` | Generation failed for another reason |
 * | `INFERENCE_CANCELLED` | Generation was cancelled |
 * | `EMBEDDING_FAILED` | Embedding computation failed for another reason |
 * | `UNSUPPORTED` | The engine or platform does not implement the request |
 * | `DOWNLOAD_FAILED` | Fetching the model failed |
 */
export type LiteRTLMErrorCode =
  | "MEMORY_ESTIMATE_EXCEEDED"
  | "OUT_OF_MEMORY"
  | "MODEL_NOT_FOUND"
  | "INVALID_MODEL"
  | "MODEL_NOT_LOADED"
  | "MODEL_TYPE_MISMATCH"
  | "RUNTIME_TOO_OLD"
  | "ENGINE_INIT_FAILED"
  | "INVALID_CONFIG"
  | "INVALID_INPUT"
  | "INFERENCE_FAILED"
  | "INFERENCE_CANCELLED"
  | "EMBEDDING_FAILED"
  | "UNSUPPORTED"
  | "DOWNLOAD_FAILED";

/** Optional details attached to a {@link LiteRTLMError}. */
export interface LiteRTLMErrorOptions {
  /** The engine's own status, when the failure came from native code */
  nativeStatus?: NativeStatus;
  /** The underlying error */
  cause?: unknown;
}

/**
 * Base error class for all typed react-native-litert-lm errors.
 */
export class LiteRTLMError extends Error {
  readonly code: LiteRTLMErrorCode;
  /**
   * The engine's canonical status (code, name, failing phase, raw message)
   * when the failure originated natively — e.g. `RESOURCE_EXHAUSTED` during
   * `execute`. Undefined for failures detected in JS.
   */
  readonly nativeStatus?: NativeStatus;

  constructor(code: LiteRTLMErrorCode, message: string, options?: LiteRTLMErrorOptions) {
    super(message);
    this.name = "LiteRTLMError";
    this.code = code;
    if (options?.nativeStatus) this.nativeStatus = options.nativeStatus;
    if (options?.cause !== undefined) this.cause = options.cause;
  }
}

/** Codes a {@link MemoryError} can carry. */
export type MemoryErrorCode = "MEMORY_ESTIMATE_EXCEEDED" | "OUT_OF_MEMORY";

/**
 * A memory failure, with the estimate that explains it so apps can show a
 * useful message (or lower `maxContextTokens` and retry).
 *
 * - `MEMORY_ESTIMATE_EXCEEDED` — `loadModel()` refused before loading because
 *   the pre-flight estimate says the model likely does not fit.
 * - `OUT_OF_MEMORY` — the engine itself ran out of memory while loading or
 *   generating; `estimate` is the pre-flight math for the same model/config.
 */
export class MemoryError extends LiteRTLMError {
  readonly estimate: MemoryEstimate;

  constructor(
    message: string,
    estimate: MemoryEstimate,
    options?: LiteRTLMErrorOptions & { code?: MemoryErrorCode },
  ) {
    super(options?.code ?? "MEMORY_ESTIMATE_EXCEEDED", message, options);
    this.name = "MemoryError";
    this.estimate = estimate;
  }
}

/**
 * Type guard for {@link MemoryError}: true for a pre-flight refusal and for
 * an engine out-of-memory failure, so one check covers "this did not fit".
 */
export function isMemoryError(e: unknown): e is MemoryError {
  if (!(e instanceof Error)) return false;
  const code = (e as Partial<MemoryError>).code;
  if (code === "MEMORY_ESTIMATE_EXCEEDED") return true;
  return code === "OUT_OF_MEMORY" && (e as Partial<MemoryError>).estimate !== undefined;
}
