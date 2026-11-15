/**
 * Decoding of native failures into typed {@link LiteRTLMError}s.
 *
 * Native code (Swift and Kotlin) tags every failure it raises with
 *
 *   `[LiteRTLM:<phase>:<STATUS>]` or `[LiteRTLM:<phase>:<STATUS>:<CODE>]`
 *
 * where STATUS is the engine's canonical status name (google.rpc.Code, as
 * reported by `litert_lm_get_last_error_code()` on iOS and `absl::Status`
 * strings on Android) and the optional CODE is a {@link LiteRTLMErrorCode} for
 * failures the wrapper itself detects. The tag survives every bridge hop —
 * Swift's `String(describing:)` and fbjni's `Throwable.toString()` both keep
 * the message intact — so this one parser serves both platforms.
 */
import { LiteRTLMError, type LiteRTLMErrorCode } from "./errors";

/** Canonical status names (google.rpc.Code / absl::StatusCode), indexed by code. */
export const CANONICAL_STATUS_NAMES = [
  "OK",
  "CANCELLED",
  "UNKNOWN",
  "INVALID_ARGUMENT",
  "DEADLINE_EXCEEDED",
  "NOT_FOUND",
  "ALREADY_EXISTS",
  "PERMISSION_DENIED",
  "RESOURCE_EXHAUSTED",
  "FAILED_PRECONDITION",
  "ABORTED",
  "OUT_OF_RANGE",
  "UNIMPLEMENTED",
  "INTERNAL",
  "UNAVAILABLE",
  "DATA_LOSS",
  "UNAUTHENTICATED",
] as const;

/** A canonical engine status name, e.g. `"RESOURCE_EXHAUSTED"`. */
export type NativeStatusName = (typeof CANONICAL_STATUS_NAMES)[number];

/** The operation that failed natively. */
export type NativePhase = "load" | "execute" | "embed" | "inspect";

/** The engine's own account of a failure, attached to {@link LiteRTLMError.nativeStatus}. */
export interface NativeStatus {
  /** Canonical status code, 0–16 (`google.rpc.Code`) */
  code: number;
  /** Canonical status name, e.g. `"INVALID_ARGUMENT"` */
  name: NativeStatusName;
  /** Which operation failed */
  phase: NativePhase;
  /** The engine's message, without bridge prefixes */
  message: string;
}

const TAG = /\[LiteRTLM:(load|execute|embed|inspect):([A-Z_]+)(?::([A-Z_]+))?\]\s*/;

/** Bridge noise that precedes the message: Java exception class names. */
const JAVA_PREFIX = /^(?:[a-z][\w$]*\.)+[A-Z][\w$]*(?:Exception|Error|Throwable):\s*/;

/** An NSError rendered by `String(describing:)`: `Error Domain=D Code=N "msg" UserInfo={…}`. */
const NSERROR_DESCRIPTION = /^Error Domain=\S+ Code=-?\d+ "([\s\S]*?)"(?: UserInfo=[\s\S]*)?$/;

/**
 * Strip bridge noise from an untagged native message: the Java exception
 * class prefix fbjni adds, or the NSError dump Swift's `String(describing:)`
 * produces.
 */
export function cleanNativeMessage(message: string): string {
  const ns = NSERROR_DESCRIPTION.exec(message);
  if (ns) return ns[1];
  return message.replace(JAVA_PREFIX, "");
}

const ERROR_CODES: ReadonlySet<string> = new Set<LiteRTLMErrorCode>([
  "MEMORY_ESTIMATE_EXCEEDED",
  "OUT_OF_MEMORY",
  "CONTEXT_FULL",
  "MODEL_NOT_FOUND",
  "INVALID_MODEL",
  "MODEL_NOT_LOADED",
  "MODEL_TYPE_MISMATCH",
  "RUNTIME_TOO_OLD",
  "ENGINE_INIT_FAILED",
  "INVALID_CONFIG",
  "INVALID_INPUT",
  "INFERENCE_FAILED",
  "INFERENCE_CANCELLED",
  "EMBEDDING_FAILED",
  "UNSUPPORTED",
  "DOWNLOAD_FAILED",
]);

/**
 * Parse the native tag out of an error message. Returns null for messages
 * without one (JS-side errors, or native paths that predate the protocol).
 */
export function parseNativeStatus(
  message: string,
): { status: NativeStatus; code?: LiteRTLMErrorCode } | null {
  const m = TAG.exec(message);
  if (!m) return null;
  const name = m[2] as NativeStatusName;
  const code = CANONICAL_STATUS_NAMES.indexOf(name);
  const status: NativeStatus = {
    code: code >= 0 ? code : CANONICAL_STATUS_NAMES.indexOf("UNKNOWN"),
    name: code >= 0 ? name : "UNKNOWN",
    phase: m[1] as NativePhase,
    message: message.slice(m.index + m[0].length).trim(),
  };
  const explicit = m[3] && ERROR_CODES.has(m[3]) ? (m[3] as LiteRTLMErrorCode) : undefined;
  return { status, code: explicit };
}

/**
 * The engine's account of a message that doesn't fit the context window
 * (LiteRT-LM 0.19+): "Max number of tokens reached, the context window is
 * full." or "Not enough space left in the context window for the input".
 */
const CONTEXT_FULL_MESSAGE = /context window/i;

/**
 * Map a canonical status to the error code a caller can act on, per phase.
 * `message` is the engine's text, which tells a full context window apart
 * from running out of memory (both are `RESOURCE_EXHAUSTED`).
 */
export function errorCodeForStatus(
  name: NativeStatusName,
  phase: NativePhase,
  message?: string,
): LiteRTLMErrorCode {
  switch (name) {
    case "RESOURCE_EXHAUSTED":
      return phase === "execute" && message !== undefined && CONTEXT_FULL_MESSAGE.test(message)
        ? "CONTEXT_FULL"
        : "OUT_OF_MEMORY";
    case "CANCELLED":
      return "INFERENCE_CANCELLED";
    case "UNIMPLEMENTED":
      return "UNSUPPORTED";
    case "NOT_FOUND":
      return phase === "load" || phase === "inspect" ? "MODEL_NOT_FOUND" : "INVALID_INPUT";
    case "INVALID_ARGUMENT":
    case "OUT_OF_RANGE":
    case "FAILED_PRECONDITION":
      if (phase === "load") return "INVALID_CONFIG";
      if (phase === "inspect") return "INVALID_MODEL";
      return "INVALID_INPUT";
    default:
      return defaultCodeFor(phase);
  }
}

function defaultCodeFor(phase: NativePhase): LiteRTLMErrorCode {
  switch (phase) {
    case "load":
      return "ENGINE_INIT_FAILED";
    case "inspect":
      return "INVALID_MODEL";
    case "embed":
      return "EMBEDDING_FAILED";
    case "execute":
      return "INFERENCE_FAILED";
  }
}

/** Errors made from a tag with an explicit code: the wrapper raised them itself. */
const wrapperRejections = new WeakSet<LiteRTLMError>();

/**
 * True for a failure the native wrapper detected itself (its tag carries an
 * explicit code): a missing file, a part the model has no encoder for, an
 * option the load didn't enable. The engine never saw the call.
 */
export function isWrapperRejection(e: unknown): boolean {
  return e instanceof LiteRTLMError && wrapperRejections.has(e);
}

/**
 * True for a message that failed because the context window was already
 * full ("Max number of tokens reached, the context window is full."). Unlike
 * a message too large for the space left, it leaves the engine's conversation
 * usable (LiteRT-LM 0.19.1): every later message fails the same way, each on
 * its own.
 */
export function isContextWindowFull(e: unknown): boolean {
  return e instanceof LiteRTLMError && e.code === "CONTEXT_FULL" && /context window is full/i.test(e.message);
}

function messageOf(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  return String(e);
}

/**
 * Convert anything a native call rejected with into a {@link LiteRTLMError}.
 * Already-typed errors pass through unchanged; untagged errors get the
 * phase's default code so every failure out of the public API is typed. The
 * original error is kept as `cause`.
 */
export function toLiteRTLMError(e: unknown, phase: NativePhase): LiteRTLMError {
  if (e instanceof LiteRTLMError) return e;
  const raw = messageOf(e);
  const parsed = parseNativeStatus(raw);
  if (parsed) {
    const code = parsed.code ?? errorCodeForStatus(parsed.status.name, phase, parsed.status.message);
    const error = new LiteRTLMError(code, parsed.status.message || raw, {
      nativeStatus: parsed.status,
      cause: e,
    });
    if (parsed.code) wrapperRejections.add(error);
    return error;
  }
  return new LiteRTLMError(defaultCodeFor(phase), cleanNativeMessage(raw), { cause: e });
}
