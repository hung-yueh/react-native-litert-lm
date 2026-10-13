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

/// `withCString` for an optional string: `nil` passes a NULL pointer.
func withOptionalCString<R>(_ string: String?, _ body: (UnsafePointer<CChar>?) -> R) -> R {
    if let string = string { return string.withCString { body($0) } }
    return body(nil)
}
