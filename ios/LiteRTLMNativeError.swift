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

    /// The canonical name of a status the engine returned ("UNKNOWN" for a
    /// value this build doesn't know, as c/error_reporter.h asks).
    static func statusName(_ status: LiteRtLmStatusCode) -> String {
        let raw = Int(status.rawValue)
        return raw > 0 && raw < statusNames.count ? statusNames[raw] : "UNKNOWN"
    }

    /// Reset the calling thread's engine error message. Optional since C API
    /// 1.0 (a failing call always records its own); `check` does it anyway,
    /// so a message can never come from an older failure.
    static func clearEngineError() {
        litert_lm_clear_last_error()
    }

    /// The error for a C call that returned `status` (not OK).
    ///
    /// Must run on the thread that made the failing call, immediately after
    /// it and before any other LiteRT-LM call (c/error_reporter.h): the
    /// message is thread-local. It is copied here, since the engine's pointer
    /// dies at the next failure on this thread.
    static func fromEngine(_ phase: NativePhase, status: LiteRtLmStatusCode, context: String) -> LiteRTLMNativeError {
        let name = statusName(status)
        guard let detail = litert_lm_get_last_error_message().map({ String(cString: $0) }), !detail.isEmpty else {
            return LiteRTLMNativeError(phase, status: name, context)
        }
        // The engine's message is absl::Status::ToString() — "NAME: text";
        // the name already lives in the tag.
        let text = detail.hasPrefix(name + ": ") ? String(detail.dropFirst(name.count + 2)) : detail
        return LiteRTLMNativeError(phase, status: name, "\(context): \(text)")
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

    // MARK: - C API 1.0 calls

    /// Make one C API 1.0 call and throw its failure: every function that can
    /// fail returns a `LiteRtLmStatusCode` (c/error_reporter.h §6). The error
    /// carries the returned status, `phase`, `context` and the engine's own
    /// message. `call` must be the C call itself, so the thread-local message
    /// is read on its thread right after it.
    static func check(
        _ phase: NativePhase, _ context: @autoclosure () -> String,
        _ call: () -> LiteRtLmStatusCode
    ) throws {
        clearEngineError()
        let status = call()
        if status != kLiteRtLmStatusOk {
            throw fromEngine(phase, status: status, context: context())
        }
    }

    /// `check` for a call that hands its result back through an `out_*`
    /// parameter, which starts as `initial`. A pointer result can be NULL on
    /// success: a value that is legitimately absent.
    static func read<T>(
        _ phase: NativePhase, _ context: @autoclosure () -> String, _ initial: T,
        _ call: (UnsafeMutablePointer<T>) -> LiteRtLmStatusCode
    ) throws -> T {
        var value = initial
        try check(phase, context()) { withUnsafeMutablePointer(to: &value) { call($0) } }
        return value
    }

    /// `read` for a call that creates an object, which it never leaves NULL
    /// on success.
    static func created(
        _ phase: NativePhase, _ context: @autoclosure () -> String,
        _ call: (UnsafeMutablePointer<OpaquePointer?>) -> LiteRtLmStatusCode
    ) throws -> OpaquePointer {
        guard let object = try read(phase, context(), nil, call) else {
            throw LiteRTLMNativeError(phase, status: "INTERNAL", "\(context()): the engine returned nothing")
        }
        return object
    }
}

/// The C API version of the linked framework against the one this module was
/// compiled for (c/engine.h, `LITERT_LM_C_API_VERSION_*`).
enum EngineAPI {
    /// The linked library's version, or nil if it can't say.
    static let linked: (major: Int32, minor: Int32, patch: Int32)? = {
        var major: Int32 = 0, minor: Int32 = 0, patch: Int32 = 0
        guard litert_lm_get_api_version(&major, &minor, &patch) == kLiteRtLmStatusOk else { return nil }
        return (major, minor, patch)
    }()

    static var compiled: String {
        "\(LITERT_LM_C_API_VERSION_MAJOR).\(LITERT_LM_C_API_VERSION_MINOR).\(LITERT_LM_C_API_VERSION_PATCH)"
    }

    /// Throws unless the linked framework speaks the C API this module was
    /// compiled against: the same major (another major changes the ABI), and
    /// no older minor (it may lack functions the headers declare). Call it
    /// before anything else touches the engine.
    static func requireCompatible(_ phase: NativePhase) throws {
        if let v = linked, v.major == LITERT_LM_C_API_VERSION_MAJOR, v.minor >= LITERT_LM_C_API_VERSION_MINOR {
            return
        }
        let found = linked.map { "\($0.major).\($0.minor).\($0.patch)" } ?? "an unknown version"
        throw LiteRTLMNativeError(
            phase, status: "FAILED_PRECONDITION", code: "ENGINE_INIT_FAILED",
            "The linked CLiteRTLM.xcframework implements LiteRT-LM C API \(found), but react-native-litert-lm " +
            "was compiled against C API \(compiled). ios/Frameworks is probably left over from another " +
            "version of the package: reinstall react-native-litert-lm (its postinstall fetches the matching " +
            "framework, see ios/Frameworks/.engine-tag), run pod install, and rebuild.")
    }
}
