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
        guard FileManager.default.fileExists(atPath: path) else {
            throw LiteRTLMNativeError(.inspect, status: "NOT_FOUND", code: "MODEL_NOT_FOUND",
                                      "Model file not found: \(path)")
        }
        LiteRTLMNativeError.clearEngineError()
        guard let file = litert_lm_loaded_file_create(path) else {
            throw LiteRTLMNativeError.fromEngine(.inspect, context: "Not a readable LiteRT-LM bundle")
        }
        return file
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

    private static func readUncached(_ path: String) throws -> NativeModelInfo {
        let file = try open(path)
        defer { litert_lm_loaded_file_delete(file) }

        let modalityInfos: [NativeModalityInfo] = modalities.map { modality, cModality in
            let backends = declaredBackends(file, cModality)
            let soc = litert_lm_loaded_file_modality_soc_name(file, cModality)
                .map { String(cString: $0) }
                .flatMap { $0.isEmpty ? nil : $0 }
            return NativeModalityInfo(
                modality: modality,
                supported: litert_lm_loaded_file_supports_input_modality(file, cModality),
                backends: backends,
                npuBrand: npuBrand(from: litert_lm_loaded_file_modality_npu_brand(file, cModality)),
                socName: soc
            )
        }

        let fileSize = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? NSNumber)?
            .doubleValue ?? 0
        let minRuntime = litert_lm_loaded_file_min_runtime_version(file)
            .map { String(cString: $0) }
            .flatMap { $0.isEmpty ? nil : $0 }

        return NativeModelInfo(
            modelType: modelType(from: litert_lm_loaded_file_model_type(file)),
            fileSizeBytes: fileSize,
            maxContextTokens: Double(litert_lm_loaded_file_max_context_tokens(file)),
            isDynamicContext: litert_lm_loaded_file_is_dynamic_context(file),
            minRuntimeVersion: minRuntime,
            modalities: modalityInfos,
            supportsThinking: litert_lm_loaded_file_supports_thinking(file),
            supportsFunctionCalling: litert_lm_loaded_file_supports_function_calling(file),
            hasSpeculativeDecoding: litert_lm_loaded_file_has_speculative_decoding_support(file),
            samplerType: Double(litert_lm_loaded_file_sampler_type(file).rawValue),
            samplerTemperature: Double(litert_lm_loaded_file_sampler_temperature(file)),
            samplerTopK: Double(litert_lm_loaded_file_sampler_top_k(file)),
            samplerTopP: Double(litert_lm_loaded_file_sampler_top_p(file)),
            maxVisionTokenBudget: Double(litert_lm_loaded_file_max_vision_token_budget(file)),
            visionSignatureSelection: intList { litert_lm_loaded_file_vision_signature_selection(file, $0, $1) },
            embeddingDimension: Double(litert_lm_loaded_file_embedding_dimension(file)),
            embeddingSignatureSelection: intList { litert_lm_loaded_file_embedding_signature_selection(file, $0, $1) }
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

    private static func declaredBackends(_ file: OpaquePointer, _ modality: LiteRtLmModality) -> [Backend] {
        let count = litert_lm_loaded_file_modality_supported_backends(file, modality, nil, 0)
        guard count > 0 else { return [] }
        var raw = [LiteRtLmBackendType](repeating: kLiteRtLmBackendTypeCpu, count: Int(count))
        let n = litert_lm_loaded_file_modality_supported_backends(file, modality, &raw, count)
        return raw.prefix(Int(max(0, n))).compactMap(backend(from:))
    }

    /// Read a C "count, then fill" int list (count with a NULL buffer first).
    private static func intList(_ fill: (UnsafeMutablePointer<Int32>?, Int32) -> Int32) -> [Double] {
        let count = fill(nil, 0)
        guard count > 0 else { return [] }
        var values = [Int32](repeating: 0, count: Int(count))
        let n = values.withUnsafeMutableBufferPointer { fill($0.baseAddress, count) }
        return values.prefix(Int(max(0, n))).map(Double.init)
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
