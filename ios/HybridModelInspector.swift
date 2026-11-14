//
//  HybridModelInspector.swift
//  react-native-litert-lm
//
//  Reads .litertlm bundle metadata through LiteRT-LM's model-info C API
//  (c/model_info.h) without creating an engine, and describes the device.
//  Backs inspectModel() / recommendBackend() and loadModel()'s modality setup.
//

import Foundation
import Metal
import NitroModules
import CLiteRTLM

public class HybridModelInspector: HybridModelInspectorSpec_base, HybridModelInspectorSpec_protocol {
    private let queue = DispatchQueue(label: "dev.litert.inspect", qos: .userInitiated)

    public func inspect(modelPath: String) throws -> Promise<NativeModelInfo> {
        let promise = Promise<NativeModelInfo>()
        queue.async {
            do {
                promise.resolve(withResult: try ModelBundle.read(modelPath))
            } catch {
                promise.reject(withError: error)
            }
        }
        return promise
    }

    public func getDeviceInfo() throws -> NativeDeviceInfo {
        DeviceFacts.current()
    }
}

/// Bundle metadata reads. Not free: besides the header, the engine reads
/// each vision encoder section into memory to learn its signatures (hundreds
/// of MB on a multimodal bundle). Readings are cached per file version, so
/// inspectModel() and the loadModel() that follows read a bundle once.
enum ModelBundle {
    private struct FileVersion: Equatable {
        let size: UInt64
        let modified: Date?
    }
    private static let cacheLock = NSLock()
    private static var cache: [String: (version: FileVersion, info: NativeModelInfo)] = [:]

    private static func fileVersion(_ path: String) -> FileVersion? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path) else { return nil }
        return FileVersion(size: (attrs[.size] as? NSNumber)?.uint64Value ?? 0,
                           modified: attrs[.modificationDate] as? Date)
    }

    private static let modalities: [(Modality, LiteRtLmModality)] = [
        (.text, kLiteRtLmModalityText),
        (.vision, kLiteRtLmModalityVision),
        (.audio, kLiteRtLmModalityAudio),
        (.video, kLiteRtLmModalityVideo),
    ]

    /// Open a bundle for metadata queries. Must run on one thread end to end
    /// (the engine's error state is thread-local).
    private static func open(_ path: String) throws -> OpaquePointer {
        try EngineAPI.requireCompatible(.inspect)
        guard FileManager.default.fileExists(atPath: path) else {
            throw LiteRTLMNativeError(.inspect, status: "NOT_FOUND", code: "MODEL_NOT_FOUND",
                                      "Model file not found: \(path)")
        }
        return try LiteRTLMNativeError.created(.inspect, "Not a readable LiteRT-LM bundle") {
            litert_lm_loaded_file_create(path, $0)
        }
    }

    static func read(_ path: String) throws -> NativeModelInfo {
        let version = fileVersion(path)
        if let version = version {
            cacheLock.lock()
            let hit = cache[path]
            cacheLock.unlock()
            if let hit = hit, hit.version == version { return hit.info }
        }
        let info = try readUncached(path)
        if let version = version {
            cacheLock.lock()
            cache[path] = (version, info)
            cacheLock.unlock()
        }
        return info
    }

    /// A model-info read. Since C API 1.0 a value the bundle doesn't declare
    /// (no vision budget, no sampler defaults on an embedding bundle, …) is
    /// `kLiteRtLmStatusNotFound`: nil here, not an error. Any other failure
    /// throws.
    private static func declared<T>(
        _ what: String, _ initial: T, _ call: (UnsafeMutablePointer<T>) -> LiteRtLmStatusCode
    ) throws -> T? {
        var value = initial
        LiteRTLMNativeError.clearEngineError()
        let status = withUnsafeMutablePointer(to: &value) { call($0) }
        if status == kLiteRtLmStatusNotFound { return nil }
        guard status == kLiteRtLmStatusOk else {
            throw LiteRTLMNativeError.fromEngine(.inspect, status: status, context: "Failed to read the bundle's \(what)")
        }
        return value
    }

    /// A model-info read the engine always answers.
    private static func value<T>(
        _ what: String, _ initial: T, _ call: (UnsafeMutablePointer<T>) -> LiteRtLmStatusCode
    ) throws -> T {
        try LiteRTLMNativeError.read(.inspect, "Failed to read the bundle's \(what)", initial, call)
    }

    private static func readUncached(_ path: String) throws -> NativeModelInfo {
        let file = try open(path)
        defer { litert_lm_loaded_file_delete(file) }

        let modalityInfos: [NativeModalityInfo] = try modalities.map { modality, cModality in
            let name = modality.stringValue
            let soc = try value("\(name) SoC name", nil) { litert_lm_loaded_file_modality_soc_name(file, cModality, $0) }
                .map { String(cString: $0) }
                .flatMap { $0.isEmpty ? nil : $0 }
            return NativeModalityInfo(
                modality: modality,
                supported: try value("\(name) support", false) {
                    litert_lm_loaded_file_supports_input_modality(file, cModality, $0)
                },
                backends: try declaredBackends(file, cModality),
                npuBrand: npuBrand(from: try value("\(name) NPU brand", kLiteRtLmNpuBrandUnknown) {
                    litert_lm_loaded_file_modality_npu_brand(file, cModality, $0)
                }),
                socName: soc
            )
        }

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?
            .doubleValue ?? 0
        let minRuntime = try value("minimum runtime version", nil) { litert_lm_loaded_file_min_runtime_version(file, $0) }
            .map { String(cString: $0) }
            .flatMap { $0.isEmpty ? nil : $0 }

        // Absent values keep the sentinels NativeModelInfo documents: 0 for the
        // context and the sampler defaults, -1 for the vision budget and the
        // embedding width, [] for the signature lists.
        return NativeModelInfo(
            modelType: modelType(from: try value("model type", kLiteRtLmModelTypeUnknown) {
                litert_lm_loaded_file_model_type(file, $0)
            }),
            fileSizeBytes: fileSize,
            maxContextTokens: Double(try declared("context size", UInt32(0)) {
                litert_lm_loaded_file_max_context_tokens(file, $0)
            } ?? 0),
            isDynamicContext: try value("dynamic context", false) { litert_lm_loaded_file_is_dynamic_context(file, $0) },
            minRuntimeVersion: minRuntime,
            modalities: modalityInfos,
            supportsThinking: try value("thinking support", false) { litert_lm_loaded_file_supports_thinking(file, $0) },
            supportsFunctionCalling: try value("function-calling support", false) {
                litert_lm_loaded_file_supports_function_calling(file, $0)
            },
            hasSpeculativeDecoding: try value("drafter", false) {
                litert_lm_loaded_file_has_speculative_decoding_support(file, $0)
            },
            samplerType: Double((try declared("sampler type", kLiteRtLmSamplerTypeUnspecified) {
                litert_lm_loaded_file_sampler_type(file, $0)
            } ?? kLiteRtLmSamplerTypeUnspecified).rawValue),
            samplerTemperature: Double(try declared("sampler temperature", Float(0)) {
                litert_lm_loaded_file_sampler_temperature(file, $0)
            } ?? 0),
            samplerTopK: Double(try declared("sampler top-k", Int32(0)) {
                litert_lm_loaded_file_sampler_top_k(file, $0)
            } ?? 0),
            samplerTopP: Double(try declared("sampler top-p", Float(0)) {
                litert_lm_loaded_file_sampler_top_p(file, $0)
            } ?? 0),
            maxVisionTokenBudget: Double(try declared("vision token budget", Int32(0)) {
                litert_lm_loaded_file_max_vision_token_budget(file, $0)
            } ?? -1),
            visionSignatureSelection: try intList("vision signatures") {
                litert_lm_loaded_file_vision_signature_selection(file, $0, $1, $2)
            },
            embeddingDimension: Double(try declared("embedding dimension", Int32(0)) {
                litert_lm_loaded_file_embedding_dimension(file, $0)
            } ?? -1),
            embeddingSignatureSelection: try intList("embedding signatures") {
                litert_lm_loaded_file_embedding_signature_selection(file, $0, $1, $2)
            }
        )
    }

    /// The backend each encoder should run on — the first the bundle declares
    /// that this device can run ("cpu"/"gpu") — or nil for an encoder the
    /// bundle lacks. Undeclared encoders keep 0.7's choice: vision on GPU when
    /// the device has one, audio on CPU. Nil when the metadata is unreadable.
    static func encoderPlan(_ path: String) -> (vision: String?, audio: String?)? {
        guard let info = try? read(path) else { return nil }
        func plan(_ modality: Modality, undeclared: String) -> String? {
            guard let m = info.modalities.first(where: { $0.modality == modality }), m.supported else { return nil }
            // A real vision encoder always reports its compiled signatures; the
            // text-only EmbeddingGemma 2 bundle claims vision support without one.
            if modality == .vision && info.visionSignatureSelection.isEmpty { return nil }
            if m.backends.isEmpty { return undeclared }
            for backend in m.backends {
                if backend == .cpu { return "cpu" }
                if backend == .gpu && DeviceFacts.gpuAvailable { return "gpu" }
            }
            return "cpu"
        }
        return (
            plan(.vision, undeclared: DeviceFacts.gpuAvailable ? "gpu" : "cpu"),
            plan(.audio, undeclared: "cpu")
        )
    }

    /// The backends the bundle declares for `modality`, most preferred first
    /// (count with a NULL buffer, then fill; 0 for an unsupported modality).
    private static func declaredBackends(_ file: OpaquePointer, _ modality: LiteRtLmModality) throws -> [Backend] {
        let what = "backends"
        let count = try value(what, 0) { litert_lm_loaded_file_modality_supported_backends(file, modality, nil, 0, $0) }
        guard count > 0 else { return [] }
        var raw = [LiteRtLmBackendType](repeating: kLiteRtLmBackendTypeCpu, count: count)
        let n = try raw.withUnsafeMutableBufferPointer { buf in
            try value(what, 0) { litert_lm_loaded_file_modality_supported_backends(file, modality, buf.baseAddress, buf.count, $0) }
        }
        return raw.prefix(min(n, count)).compactMap(backend(from:))
    }

    /// Read a C API 1.0 "count, then fill" int list: `fill(buffer, maxSize,
    /// outCount)`, first with a NULL buffer for the count. Undeclared
    /// (`kLiteRtLmStatusNotFound`) is an empty list.
    private static func intList(
        _ what: String,
        _ fill: (UnsafeMutablePointer<Int32>?, Int, UnsafeMutablePointer<Int>) -> LiteRtLmStatusCode
    ) throws -> [Double] {
        guard let count = try declared(what, 0, { fill(nil, 0, $0) }), count > 0 else { return [] }
        var values = [Int32](repeating: 0, count: count)
        let n = try values.withUnsafeMutableBufferPointer { buf in
            try declared(what, 0) { fill(buf.baseAddress, buf.count, $0) }
        } ?? 0
        return values.prefix(min(n, count)).map(Double.init)
    }

    private static func backend(from raw: LiteRtLmBackendType) -> Backend? {
        switch raw {
        case kLiteRtLmBackendTypeCpu: return .cpu
        case kLiteRtLmBackendTypeGpu: return .gpu
        case kLiteRtLmBackendTypeNpu: return .npu
        default: return nil
        }
    }

    private static func npuBrand(from raw: LiteRtLmNpuBrand) -> NpuBrand {
        switch raw {
        case kLiteRtLmNpuBrandQualcomm: return .qualcomm
        case kLiteRtLmNpuBrandGoogleTensor: return .googletensor
        case kLiteRtLmNpuBrandMediaTek: return .mediatek
        case kLiteRtLmNpuBrandIntel: return .intel
        case kLiteRtLmNpuBrandSamsung: return .samsung
        default: return .unknown
        }
    }

    private static func modelType(from raw: LiteRtLmModelType) -> ModelType {
        switch raw {
        case kLiteRtLmModelTypeLlm: return .llm
        case kLiteRtLmModelTypeEmbedding: return .embedding
        default: return .unknown
        }
    }
}

/// Device facts that decide which bundle variant and backend fit.
enum DeviceFacts {
    /// Whether Metal (the GPU backend's API) is available.
    static let gpuAvailable: Bool = MTLCreateSystemDefaultDevice() != nil

    static func current() -> NativeDeviceInfo {
        var sys = utsname()
        uname(&sys)
        let machine = withUnsafeBytes(of: &sys.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        #if targetEnvironment(simulator)
        let model = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] ?? machine
        let isEmulator = true
        #else
        let model = machine
        let isEmulator = false
        #endif
        let v = ProcessInfo.processInfo.operatingSystemVersion
        return NativeDeviceInfo(
            platform: "ios",
            osVersion: "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
            model: model,
            socModel: nil,
            socManufacturer: nil,
            totalMemoryBytes: Double(ProcessInfo.processInfo.physicalMemory),
            gpuAvailable: gpuAvailable,
            isEmulator: isEmulator
        )
    }
}
