package com.margelo.nitro.dev.litert.litertlm

import java.io.FileNotFoundException

/**
 * A failure carrying the engine's canonical status, rendered as
 * `[LiteRTLM:<phase>:<STATUS>(:<CODE>)] <detail>` — the protocol
 * src/core/nativeError.ts decodes into a typed LiteRTLMError (mirrors
 * ios/LiteRTLMNativeError.swift). fbjni hands a rejection to JS as
 * `Throwable.toString()` ("<class>: <message>"); the JS parser finds the tag
 * behind the class name.
 *
 * @param phase "load", "execute", "embed" or "inspect"
 * @param status canonical status name (google.rpc.Code), e.g. "RESOURCE_EXHAUSTED"
 * @param code explicit LiteRTLMErrorCode, for failures the wrapper itself detects
 */
class LiteRTLMNativeException(
    val phase: String,
    val status: String,
    val code: String? = null,
    val detail: String,
    cause: Throwable? = null,
) : RuntimeException("[LiteRTLM:$phase:$status${code?.let { ":$it" } ?: ""}] $detail", cause) {

    companion object {
        /** google.rpc.Code names as `absl::Status::ToString()` prints them. */
        private val STATUS_NAMES = listOf(
            "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
            "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION",
            "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE", "DATA_LOSS",
            "UNAUTHENTICATED",
        )
        private val STATUS_PATTERN = Regex("\\b(${STATUS_NAMES.joinToString("|")}):")

        /**
         * Classify a LiteRT-LM SDK failure. The SDK's JNI layer throws with
         * `absl::Status::ToString()` text ("INVALID_ARGUMENT: …"), often behind
         * its own prefix ("Failed to create engine settings: INVALID_ARGUMENT: …");
         * the first status name in the message wins. Without one, fall back to
         * the exception type.
         */
        fun from(phase: String, context: String, t: Throwable): LiteRTLMNativeException {
            if (t is LiteRTLMNativeException) return t
            val message = t.message ?: t.javaClass.simpleName
            val status = statusIn(message) ?: when (t) {
                is OutOfMemoryError -> "RESOURCE_EXHAUSTED"
                is FileNotFoundException -> "NOT_FOUND"
                is IllegalArgumentException -> "INVALID_ARGUMENT"
                is IllegalStateException -> "FAILED_PRECONDITION"
                is UnsupportedOperationException -> "UNIMPLEMENTED"
                is InterruptedException -> "CANCELLED"
                else -> if (message.contains("not found", ignoreCase = true)) "NOT_FOUND" else "UNKNOWN"
            }
            return LiteRTLMNativeException(phase, status, null, "$context: $message", t)
        }

        /** The first canonical status name written as "NAME:" in [message], if any. */
        internal fun statusIn(message: String): String? =
            STATUS_PATTERN.find(message)?.groupValues?.get(1)
    }
}
