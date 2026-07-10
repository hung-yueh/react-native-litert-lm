import type { MemoryEstimate } from "./memoryEstimator";

/**
 * Machine-readable error codes for react-native-litert-lm failures.
 * Branch on `error.code` instead of parsing messages.
 */
export type LiteRTLMErrorCode =
  | "MEMORY_ESTIMATE_EXCEEDED"
  | "MODEL_NOT_FOUND"
  | "MODEL_NOT_LOADED"
  | "ENGINE_INIT_FAILED"
  | "DOWNLOAD_FAILED";

/**
 * Base error class for all typed react-native-litert-lm errors.
 */
export class LiteRTLMError extends Error {
  readonly code: LiteRTLMErrorCode;

  constructor(code: LiteRTLMErrorCode, message: string) {
    super(message);
    this.name = "LiteRTLMError";
    this.code = code;
  }
}

/**
 * Thrown by `loadModel()` when the pre-flight memory estimate says the model
 * likely does not fit in available memory and `forceLoad` was not set.
 *
 * The attached {@link MemoryEstimate} explains the math so apps can show a
 * useful message (or lower `maxContextTokens` and retry).
 */
export class MemoryError extends LiteRTLMError {
  readonly estimate: MemoryEstimate;

  constructor(message: string, estimate: MemoryEstimate) {
    super("MEMORY_ESTIMATE_EXCEEDED", message);
    this.name = "MemoryError";
    this.estimate = estimate;
  }
}

/**
 * Type guard for {@link MemoryError}.
 */
export function isMemoryError(e: unknown): e is MemoryError {
  return e instanceof Error && (e as MemoryError).code === "MEMORY_ESTIMATE_EXCEEDED";
}
