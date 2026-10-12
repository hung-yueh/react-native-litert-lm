//
//  LiteRTLMNativeError.swift
//  react-native-litert-lm
//
//  Typed native failures. Each one renders as
//  `[LiteRTLM:<phase>:<STATUS>(:<CODE>)] <message>`, which
//  src/core/nativeError.ts decodes into a LiteRTLMError on the JS side.
//

import Foundation
import CLiteRTLM

/// The operation a failure belongs to (mirrors `NativePhase` in TypeScript).
enum NativePhase: String {
    case load, execute, embed, inspect
}

/// A failure carrying the engine's canonical status. Nitro hands a rejected
/// Swift error to JS via `String(describing:)` — i.e. `description` below;
/// `LocalizedError` makes `localizedDescription` (logs, NSError bridging)
/// show the same text.
struct LiteRTLMNativeError: Error, CustomStringConvertible, LocalizedError {
    let phase: NativePhase
    /// Canonical status name (google.rpc.Code), e.g. "RESOURCE_EXHAUSTED".
    let status: String
    /// Explicit `LiteRTLMErrorCode`, for failures the wrapper itself detects.
    let code: String?
    let message: String

    init(_ phase: NativePhase, status: String, code: String? = nil, _ message: String) {
        self.phase = phase
        self.status = status
        self.code = code
        self.message = message
    }

    var description: String {
        var tag = "\(phase.rawValue):\(status)"
        if let code = code { tag += ":\(code)" }
        return "[LiteRTLM:\(tag)] \(message)"
    }

    var errorDescription: String? { description }

    /// `LiteRtLmStatusCode` values are frozen and 1:1 with google.rpc.Code.
    static let statusNames = [
        "OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
        "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED",
        "INTERNAL", "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED",
    ]

    /// Reset the calling thread's engine error state. Call right before a C
    /// call whose failure you will diagnose: successful calls never clear it,
    /// so without this a later read can surface an older failure.
    static func clearEngineError() {
        litert_lm_clear_last_error()
    }

    /// Build an error from the engine's thread-local last error.
    ///
    /// Must run on the thread that made the failing C call, immediately after
    /// it and before any other LiteRT-LM call (c/error_reporter.h) — i.e.
    /// inside the same `queue` block. The message is copied here, since the
    /// engine's pointer dies at the next failure on this thread.
    static func fromEngine(_ phase: NativePhase, context: String) -> LiteRTLMNativeError {
        let raw = Int(litert_lm_get_last_error_code())
        let detail = litert_lm_get_last_error_message().map { String(cString: $0) }
        guard raw > 0, let detail = detail, !detail.isEmpty else {
            return LiteRTLMNativeError(phase, status: "UNKNOWN", context)
        }
        let status = raw < statusNames.count ? statusNames[raw] : "UNKNOWN"
        // The engine's message is absl::Status::ToString() — "NAME: text";
        // the name already lives in the tag.
        let text = detail.hasPrefix(status + ": ") ? String(detail.dropFirst(status.count + 2)) : detail
        return LiteRTLMNativeError(phase, status: status, "\(context): \(text)")
    }

    /// Classify a status-prefixed message ("RESOURCE_EXHAUSTED: …", the
    /// absl::Status string form). Streaming errors arrive as text only.
    static func fromStatusMessage(_ phase: NativePhase, _ message: String) -> LiteRTLMNativeError {
        for name in statusNames.dropFirst() where message.hasPrefix(name + ":") {
            let rest = message.dropFirst(name.count + 1).trimmingCharacters(in: .whitespaces)
            return LiteRTLMNativeError(phase, status: name, rest)
        }
        return LiteRTLMNativeError(phase, status: "UNKNOWN", message)
    }
}
