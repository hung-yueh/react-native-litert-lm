//
//  CMappings.swift
//  react-native-litert-lm
//
//  Small bridges between Nitro types and the LiteRT-LM C API, shared by the
//  LLM, the embedder and the inspector.
//

import Foundation
import CLiteRTLM

extension ActivationDataType {
    /// The C enum (raw values match: 0 F32, 1 F16, 2 I16, 3 I8).
    var cValue: LiteRtLmActivationDataType {
        switch self {
        case .f16: return kLiteRtLmActivationDataTypeFloat16
        case .i16: return kLiteRtLmActivationDataTypeInt16
        case .i8: return kLiteRtLmActivationDataTypeInt8
        default: return kLiteRtLmActivationDataTypeFloat32
        }
    }
}

extension SpeculativeDecodingMode {
    /// The C enum (LiteRT-LM 0.19, C API 1.0).
    var cValue: LiteRtLmSpeculativeDecodingMode {
        switch self {
        case .off: return kLiteRtLmSpeculativeDecodingModeOff
        case .on: return kLiteRtLmSpeculativeDecodingModeOn
        case .adaptive: return kLiteRtLmSpeculativeDecodingModeAdaptive
        }
    }

    /// What a caller asked for: the mode wins over the older boolean, where
    /// `true` is `.on` and `false` is `.off`. Nil leaves the default (the
    /// engine's for a conversation; off for the engine).
    static func resolve(_ mode: SpeculativeDecodingMode?, legacy enabled: Bool?) -> SpeculativeDecodingMode? {
        if let mode = mode { return mode }
        return enabled.map { $0 ? .on : .off }
    }

    /// Build a speculative-decoding config with this mode and hand it to
    /// `apply`, which copies it (engine settings or a session config).
    func apply(_ phase: NativePhase, _ apply: (OpaquePointer) -> LiteRtLmStatusCode) throws {
        let config = try LiteRTLMNativeError.created(phase, "Failed to create a speculative decoding config") {
            litert_lm_speculative_decoding_config_create($0)
        }
        defer { litert_lm_speculative_decoding_config_delete(config) }
        try LiteRTLMNativeError.check(phase, "Failed to set speculative decoding to \(stringValue)") {
            litert_lm_speculative_decoding_config_set_mode(config, cValue)
        }
        try LiteRTLMNativeError.check(phase, "Failed to apply speculative decoding \(stringValue)") { apply(config) }
    }
}

/// `withCString` for an optional string: `nil` passes a NULL pointer.
func withOptionalCString<R>(_ string: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    if let string = string { return string.withCString { body($0) } }
    return body(nil)
}
