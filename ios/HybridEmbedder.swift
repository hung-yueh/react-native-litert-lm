//
//  HybridEmbedder.swift
//  react-native-litert-lm
//
//  On-device embeddings through LiteRT-LM's EmbeddingEngine C API
//  (c/embedding_engine.h, 0.17+): text, image and audio inputs to one vector,
//  returned as a native-owned ArrayBuffer of little-endian float32 values.
//

import Foundation
import NitroModules
import CLiteRTLM

public class HybridEmbedder: HybridEmbedderSpec_base, HybridEmbedderSpec_protocol {
    /// Engine work never touches the JS thread. Separate from the LLM's queue:
    /// an embedder and an LLM are separate engines and may run concurrently.
    private let queue = DispatchQueue(label: "dev.litert.embedding", qos: .userInitiated)
    /// Touched only on `queue` (and in deinit).
    private var engine: OpaquePointer?

    /// Readiness and size, which the JS thread reads without waiting for
    /// engine work on `queue`.
    private let stateLock = NSLock()
    private var loaded = false
    private var loadedModelSizeBytes = 0

    private func setState(loaded: Bool, sizeBytes: Int) {
        stateLock.lock()
        defer { stateLock.unlock() }
        self.loaded = loaded
        self.loadedModelSizeBytes = sizeBytes
    }

    /// Model weight size, so the JS garbage collector sees the native cost.
    public var memorySize: Int {
        stateLock.lock()
        defer { stateLock.unlock() }
        return loadedModelSizeBytes
    }

    deinit {
        if let engine = engine { litert_lm_embedding_engine_delete(engine) }
    }

    // MARK: - Lifecycle

    public func loadModel(modelPath: String, config: EmbedderConfig?) throws -> Promise<Void> {
        let promise = Promise<Void>()
        queue.async {
            self.unloadInternal()
            guard FileManager.default.fileExists(atPath: modelPath) else {
                promise.reject(withError: LiteRTLMNativeError(
                    .load, status: "NOT_FOUND", code: "MODEL_NOT_FOUND", "Model file not found: \(modelPath)"))
                return
            }
            // Encoders default to the bundle's own declaration (e.g. none for
            // a text-only embedder); an explicit backend wins.
            let plan = ModelBundle.encoderPlan(modelPath)
            let main = config?.backend?.stringValue ?? "cpu"
            let explicitVision = config?.visionBackend?.stringValue
            let explicitAudio = config?.audioBackend?.stringValue
            let vision = explicitVision ?? plan?.vision
            let audio = explicitAudio ?? plan?.audio
            // Bundle metadata can claim an encoder the file lacks (audio has no
            // signal to check), so retry without encoders the caller didn't ask
            // for. A failed creation costs a fraction of a second.
            var attempts: [(vision: String?, audio: String?)] = [(vision, audio)]
            for fallback in [(vision, explicitAudio), (explicitVision, audio), (explicitVision, explicitAudio)]
            where !attempts.contains(where: { $0.vision == fallback.0 && $0.audio == fallback.1 }) {
                attempts.append(fallback)
            }
            var lastError: LiteRTLMNativeError?
            for attempt in attempts {
                switch Self.createEngine(modelPath, config, main: main, vision: attempt.vision, audio: attempt.audio) {
                case .success(let engine):
                    self.engine = engine
                    let attributes = try? FileManager.default.attributesOfItem(atPath: modelPath)
                    self.setState(loaded: true, sizeBytes: attributes?[.size] as? Int ?? 0)
                    promise.resolve()
                    return
                case .failure(let error):
                    lastError = error
                }
            }
            promise.reject(withError: lastError
                ?? LiteRTLMNativeError(.load, status: "UNKNOWN", "Failed to create the embedding engine"))
        }
        return promise
    }

    /// One engine-creation attempt. Run on `queue` (thread-local engine errors).
    private static func createEngine(
        _ modelPath: String, _ config: EmbedderConfig?, main: String, vision: String?, audio: String?
    ) -> Result<OpaquePointer, LiteRTLMNativeError> {
        LiteRTLMNativeError.clearEngineError()
        let settings = modelPath.withCString { modelC in
            main.withCString { mainC in
                withOptionalCString(vision) { visionC in
                    withOptionalCString(audio) { audioC in
                        litert_lm_embedding_engine_settings_create(modelC, mainC, visionC, audioC)
                    }
                }
            }
        }
        guard let settings = settings else {
            return .failure(LiteRTLMNativeError.fromEngine(.load, context: "Invalid embedding engine settings"))
        }
        defer { litert_lm_embedding_engine_settings_delete(settings) }

        if let length = config?.maxInputLength {
            litert_lm_embedding_engine_settings_set_max_input_length(settings, Int32(length))
        }
        if let tokens = config?.visionTokensPerImage {
            litert_lm_embedding_engine_settings_set_vision_tokens_per_image(settings, Int32(tokens))
        }
        if let adt = config?.activationDataType {
            litert_lm_embedding_engine_settings_set_activation_data_type(settings, adt.cValue)
        }
        (modelPath as NSString).deletingLastPathComponent.withCString {
            litert_lm_embedding_engine_settings_set_cache_dir(settings, $0)
        }

        LiteRTLMNativeError.clearEngineError()
        guard let engine = litert_lm_embedding_engine_create(settings) else {
            return .failure(LiteRTLMNativeError.fromEngine(
                .load, context: "Failed to create the embedding engine (\(main))"))
        }
        return .success(engine)
    }

    public func isReady() throws -> Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return loaded
    }

    public func getMemoryUsage() throws -> MemoryUsage {
        HybridLiteRTLM.processMemoryUsage()
    }

    public func unload() throws -> Promise<Void> {
        let promise = Promise<Void>()
        queue.async {
            self.unloadInternal()
            promise.resolve()
        }
        return promise
    }

    /// Returns at once: the instance reports not-ready immediately, and the
    /// engine is freed on `queue` as soon as any in-flight work finishes.
    public func close() throws {
        setState(loaded: false, sizeBytes: 0)
        queue.async { self.unloadInternal() }
    }

    /// Run on `queue`.
    private func unloadInternal() {
        if let engine = engine {
            litert_lm_embedding_engine_delete(engine)
            self.engine = nil
        }
        setState(loaded: false, sizeBytes: 0)
    }

    // MARK: - Embedding

    public func embed(parts: [MultimodalPart], options: EmbeddingOptions?) throws -> Promise<ArrayBuffer> {
        let promise = Promise<ArrayBuffer>()
        // JS buffers are copied now (they aren't ours after this call); files
        // are read later, on the queue.
        let inputs: [EmbeddingInput]
        do { inputs = try parts.map(EmbeddingInput.init) } catch {
            promise.reject(withError: error)
            return promise
        }
        queue.async {
            do {
                let engine = try self.requireEngine()
                promise.resolve(withResult: try Self.computeOne(engine, inputs, options))
            } catch {
                promise.reject(withError: error)
            }
        }
        return promise
    }

    public func embedBatch(inputs batch: [[MultimodalPart]], options: EmbeddingOptions?) throws -> Promise<[ArrayBuffer]> {
        let promise = Promise<[ArrayBuffer]>()
        let groups: [[EmbeddingInput]]
        do { groups = try batch.map { try $0.map(EmbeddingInput.init) } } catch {
            promise.reject(withError: error)
            return promise
        }
        queue.async {
            do {
                let engine = try self.requireEngine()
                promise.resolve(withResult: try Self.computeBatch(engine, groups, options))
            } catch {
                promise.reject(withError: error)
            }
        }
        return promise
    }

    private func requireEngine() throws -> OpaquePointer {
        guard let engine = engine else {
            throw LiteRTLMNativeError(.embed, status: "FAILED_PRECONDITION", code: "MODEL_NOT_LOADED",
                                      "No embedding model loaded. Call loadModel() first.")
        }
        return engine
    }

    /// Run on `queue`. Inputs and the response are freed before returning.
    private static func computeOne(
        _ engine: OpaquePointer, _ inputs: [EmbeddingInput], _ options: EmbeddingOptions?
    ) throws -> ArrayBuffer {
        let handles = try createAll(inputs)
        defer { handles.forEach { litert_lm_input_data_delete($0) } }
        let opts = makeOptions(options)
        defer { if let opts = opts { litert_lm_embedding_options_delete(opts) } }

        LiteRTLMNativeError.clearEngineError()
        let response: OpaquePointer? = handles.map { Optional($0) }.withUnsafeBufferPointer { buf in
            litert_lm_embedding_engine_compute_embedding(engine, buf.baseAddress, buf.count, opts)
        }
        guard let response = response else {
            throw LiteRTLMNativeError.fromEngine(.embed, context: "Embedding failed")
        }
        defer { litert_lm_embedding_response_delete(response) }
        return try copyVector(response)
    }

    /// Run on `queue`. One engine call for the whole batch.
    private static func computeBatch(
        _ engine: OpaquePointer, _ groups: [[EmbeddingInput]], _ options: EmbeddingOptions?
    ) throws -> [ArrayBuffer] {
        guard !groups.isEmpty else { return [] }
        var handles: [[OpaquePointer]] = []
        defer { handles.forEach { $0.forEach { litert_lm_input_data_delete($0) } } }
        for group in groups { handles.append(try createAll(group)) }
        let opts = makeOptions(options)
        defer { if let opts = opts { litert_lm_embedding_options_delete(opts) } }

        // C wants `const LiteRtLmInputData* const* const*`: an array of
        // per-input arrays, plus their lengths. Keep every row alive for the call.
        let rows: [UnsafeMutablePointer<OpaquePointer?>] = handles.map { group in
            let row = UnsafeMutablePointer<OpaquePointer?>.allocate(capacity: max(1, group.count))
            for (i, h) in group.enumerated() { row[i] = h }
            return row
        }
        defer { rows.forEach { $0.deallocate() } }
        var rowPointers: [UnsafePointer<OpaquePointer?>?] = rows.map { UnsafePointer($0) }
        var counts: [Int] = handles.map { $0.count }

        LiteRTLMNativeError.clearEngineError()
        let responses = rowPointers.withUnsafeMutableBufferPointer { rowBuf in
            counts.withUnsafeMutableBufferPointer { countBuf in
                litert_lm_embedding_engine_compute_embedding_batch(
                    engine, rowBuf.baseAddress, countBuf.baseAddress, groups.count, opts)
            }
        }
        guard let responses = responses else {
            throw LiteRTLMNativeError.fromEngine(.embed, context: "Batch embedding failed")
        }
        defer { litert_lm_embedding_responses_delete(responses) }
        let n = litert_lm_embedding_responses_get_size(responses)
        return try (0..<n).map { index in
            guard let response = litert_lm_embedding_responses_get_at(responses, index) else {
                throw LiteRTLMNativeError(.embed, status: "INTERNAL", "Missing batch result \(index)")
            }
            return try copyVector(response)
        }
    }

    /// Engine inputs for one embedding; none survive a failure partway through.
    private static func createAll(_ inputs: [EmbeddingInput]) throws -> [OpaquePointer] {
        var handles: [OpaquePointer] = []
        do {
            for input in inputs { handles.append(try input.create()) }
        } catch {
            handles.forEach { litert_lm_input_data_delete($0) }
            throw error
        }
        return handles
    }

    private static func makeOptions(_ options: EmbeddingOptions?) -> OpaquePointer? {
        guard let options = options, let opts = litert_lm_embedding_options_create() else { return nil }
        if let normalize = options.normalize {
            litert_lm_embedding_options_set_normalize(opts, normalize)
        }
        if let size = options.outputSize {
            litert_lm_embedding_options_set_output_size(opts, Int32(size))
        }
        if let special = options.insertSpecialTokens {
            litert_lm_embedding_options_set_insert_special_tokens(opts, special)
        }
        if let tokens = options.visionTokensPerImage {
            litert_lm_embedding_options_set_vision_tokens_per_image(opts, Int32(tokens))
        }
        return opts
    }

    /// Copy a response's floats into a native-owned buffer JS reads as a Float32Array.
    private static func copyVector(_ response: OpaquePointer) throws -> ArrayBuffer {
        let count = litert_lm_embedding_response_get_size(response)
        guard count > 0, let values = litert_lm_embedding_response_get_values(response) else {
            throw LiteRTLMNativeError(.embed, status: "INTERNAL", "The engine returned an empty embedding")
        }
        let bytes = count * MemoryLayout<Float>.size
        return ArrayBuffer.copy(of: UnsafeRawPointer(values).assumingMemoryBound(to: UInt8.self), size: bytes)
    }
}

/// An input part captured on the JS thread, so the engine can run off it.
private enum EmbeddingInput {
    case text(String)
    case image(Media)
    case audio(Media)

    /// A JS buffer is copied at capture (it isn't ours after the call); a file
    /// is read when the engine needs it, on the embedding queue.
    enum Media {
        case bytes(Data)
        case file(String)

        func read(_ label: String) throws -> Data {
            switch self {
            case .bytes(let data):
                return data
            case .file(let path):
                guard let data = FileManager.default.contents(atPath: path) else {
                    throw LiteRTLMNativeError(.embed, status: "NOT_FOUND", code: "INVALID_INPUT",
                                              "\(label) file not found: \(path)")
                }
                return data
            }
        }
    }

    init(_ part: MultimodalPart) throws {
        switch part.type {
        case .text:
            self = .text(part.text ?? "")
        case .image:
            self = .image(try Self.media(part.path, part.imageBuffer, label: "Image"))
        case .audio:
            self = .audio(try Self.media(part.path, part.audioBuffer, label: "Audio"))
        default:
            throw LiteRTLMNativeError(.embed, status: "INVALID_ARGUMENT", code: "INVALID_INPUT",
                                      "Embeddings take text, image and audio parts only")
        }
    }

    private static func media(_ path: String?, _ buffer: ArrayBuffer?, label: String) throws -> Media {
        if let path = path { return .file(path) }
        if let buffer = buffer { return .bytes(buffer.toData(copyIfNeeded: true)) }
        throw LiteRTLMNativeError(.embed, status: "INVALID_ARGUMENT", code: "INVALID_INPUT",
                                  "\(label) part needs a path or a buffer")
    }

    /// Create the engine-side input (the engine copies the bytes). Run on the queue.
    func create() throws -> OpaquePointer {
        let handle: OpaquePointer?
        switch self {
        case .text(let text):
            handle = text.withCString { litert_lm_input_data_create(kLiteRtLmInputDataTypeText, $0, strlen($0)) }
        case .image(let media):
            let data = try media.read("Image")
            handle = data.withUnsafeBytes { litert_lm_input_data_create(kLiteRtLmInputDataTypeImage, $0.baseAddress, data.count) }
        case .audio(let media):
            let data = try media.read("Audio")
            handle = data.withUnsafeBytes { litert_lm_input_data_create(kLiteRtLmInputDataTypeAudio, $0.baseAddress, data.count) }
        }
        guard let handle = handle else {
            throw LiteRTLMNativeError.fromEngine(.embed, context: "Failed to create an embedding input")
        }
        return handle
    }
}
