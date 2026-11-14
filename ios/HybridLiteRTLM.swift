//
//  HybridLiteRTLM.swift
//  react-native-litert-lm
//
//  Created by Antigravity on 2026-05-19.
//  Copyright © 2026 Margelo. All rights reserved.
//

import Foundation
import NitroModules
import CLiteRTLM
import os

public class HybridLiteRTLM: HybridLiteRTLMSpec_base, HybridLiteRTLMSpec_protocol {
    
    // MARK: - Internal (for extension files)
    
    /// Serial queue for all engine work: load, conversation changes, and
    /// non-streaming generation, which can hold it for seconds. Nothing the
    /// JS thread calls waits on it — see `Snapshot`.
    let queue = DispatchQueue(label: "dev.litert.engine", qos: .userInteractive)

    /// Opaque pointer to the active conversation state. Used on `queue`, and
    /// written there under `conversationLock` so cancelGeneration() can reach
    /// it from any thread.
    private(set) var conversation: OpaquePointer?
    private let conversationLock = NSLock()

    /// Why the last conversation change failed, reported by the next execute()
    /// (resetConversation() returns before the engine does the work). `queue` only.
    var conversationFailure: LiteRTLMNativeError?

    var loadedModelPath: String?
    let modelStore = HybridModelStore()

    // MARK: - State the JS thread reads

    /// What the JS thread sees. Reading it never waits for engine work on
    /// `queue`; work on `queue` keeps it current, and close(), unload(),
    /// loadModel() and resetConversation() update it the moment they're called.
    struct Snapshot {
        var isLoaded = false
        var history: [Message] = []
        var stats = GenerationStats.zero
        /// Tokens in the context after the last completed turn; -1 when unknown.
        var contextTokens: Double = -1
        var modelSizeBytes = 0
        /// Moves whenever the conversation is replaced (reset, close, unload,
        /// load). A turn that started under an older epoch keeps its result
        /// out of the transcript, and one that hadn't started yet is cancelled.
        var epoch = 0
        /// Moves on load, close and unload only, so a load finishing after a
        /// close() doesn't report itself ready.
        var lifecycle = 0
        /// Set by close(): the instance loads nothing after it.
        var closed = false
        /// Streams generating on the engine's own thread.
        var streamsInFlight = 0
        var memoryWarningCallback: ((MemoryWarningLevel, MemoryUsage) -> Void)?
        var memoryPressureSource: DispatchSourceMemoryPressure?
    }
    private let snapshotLock = NSLock()
    private var snapshot = Snapshot()

    /// Read or update the snapshot. Keep the body short: it runs under a lock
    /// the JS thread takes.
    @discardableResult
    func withSnapshot<T>(_ body: (inout Snapshot) -> T) -> T {
        snapshotLock.lock()
        defer { snapshotLock.unlock() }
        return body(&snapshot)
    }

    /// Streams in flight, so work on `queue` can wait for them to end before
    /// it deletes their conversation.
    let streams = DispatchGroup()

    // MARK: - Private state

    /// Opaque pointer to the LiteRT LM C Engine. Written on `queue` under
    /// `engineLock`, which countTokens() holds while it uses the engine.
    private var engine: OpaquePointer?
    private let engineLock = NSLock()

    /// Memory-pressure events run here, so they arrive even while `queue` is
    /// busy with a load or a generation.
    private let memoryQueue = DispatchQueue(label: "dev.litert.engine.memory", qos: .utility)

    // Default configuration variables
    /// Readable by tests: a failed GPU/NPU init falls back to CPU and updates this.
    private(set) var backend: Backend = .cpu
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxContextTokens: Int = 4096
    private var maxOutputTokens: Int = 1024
    private var systemPrompt: String?
    private var tools: [ToolDefinition]?
    /// The engine's speculative-decoding mode (LLMConfig), nil for the default (off).
    private var speculativeDecodingMode: SpeculativeDecodingMode?
    /// Whether the loaded bundle ships a drafter (e.g. Gemma 4's MTP). Without
    /// one, a request to speculate is skipped.
    private var bundleHasDrafter = false
    /// LLMConfig.exactStats: the live engine is in benchmark mode, so getStats() can read its numbers.
    private(set) var exactStats: Bool = false
    private var chatTemplate: String?
    private var numThreads: Int?
    private var maxVisionTokensPerImage: Int?
    private var metalResidencySet: Bool?
    private(set) var singleThreadedExecution: Bool?
    private var prefillChunkSize: Int?
    private var activationDataType: ActivationDataType?
    private var loraPath: String?
    private var audioLoraPath: String?
    private var loraRank: Int?
    private var streamToolCalls: Bool = false
    var enableStructuredOutput: Bool = false
    private var thinkingOptions: ThinkingOptions?
    private var toolCallChannelName: String = "tool_call"

    /// Encoders the live engine was created with; execute() rejects media
    /// parts for a missing one with a typed error instead of an engine failure.
    private(set) var hasVisionEncoder = false
    private(set) var hasAudioEncoder = false

    /// Actual model weight size to inform the JS engine's garbage collection.
    public var memorySize: Int {
        withSnapshot { $0.modelSizeBytes }
    }

    deinit {
        stopMemoryPressureMonitoring()
        closeInternal()
    }

    // MARK: - Core Hybrid Object API

    public func isReady() throws -> Bool {
        withSnapshot { $0.isLoaded }
    }

    public func getHistory() throws -> [Message] {
        withSnapshot { $0.history }
    }

    /// Returns at once. The transcript and stats switch immediately, a message
    /// still generating in the old conversation is cancelled, and `queue`
    /// replaces the conversation before any message sent after this call. A
    /// failure to create the new conversation is reported by the next execute().
    public func resetConversation(
        historyJson: String?,
        systemPrompt: String?,
        speculativeDecoding: Bool? = nil,
        speculativeDecodingMode: SpeculativeDecodingMode? = nil
    ) throws {
        // The mode wins over the boolean; unset inherits the engine's.
        let speculative = SpeculativeDecodingMode.resolve(speculativeDecodingMode, legacy: speculativeDecoding)
        // Mirror the replayed transcript into the wrapper's own history so
        // getHistory() stays truthful for the restored conversation.
        let transcript = Self.parseTranscript(historyJson)
        let epoch: Int = withSnapshot { s in
            s.epoch += 1
            s.history = transcript
            s.stats = .zero
            s.contextTokens = -1
            return s.epoch
        }
        cancelGeneration()
        queue.async {
            // Skip it if a newer reset, close or load has superseded it.
            guard self.engine != nil, self.withSnapshot({ $0.epoch }) == epoch else { return }
            // Replay in the engine's own message format, tool exchanges included.
            let messagesJson = Self.engineMessagesJson(transcript)
            self.conversationFailure = self.createNewConversation(
                initialMessagesJson: messagesJson,
                systemPromptOverride: systemPrompt,
                speculativeDecoding: speculative
            )
            // The engine prefills a replayed transcript with the next message,
            // so until that one completes the count stays -1, as on Android:
            // the 0 it reads now would understate what the message builds on.
            guard messagesJson == nil else { return }
            let tokens = self.conversation.map(Self.tokenCount) ?? -1
            self.withSnapshot { s in
                if s.epoch == epoch { s.contextTokens = tokens }
            }
        }
    }

    public func getStats() throws -> GenerationStats {
        withSnapshot { $0.stats }
    }

    /// The one engine call made off `queue`: tokenizing reads only the
    /// tokenizer, so it needn't wait behind a generation. `engineLock` keeps
    /// the engine from being freed meanwhile.
    public func countTokens(text: String) throws -> Double {
        engineLock.lock()
        defer { engineLock.unlock() }
        guard let engine = engine,
              let result = try? LiteRTLMNativeError.created(.execute, "Failed to tokenize", {
                  litert_lm_engine_tokenize(engine, text, $0)
              })
        else { return -1.0 }
        defer { litert_lm_tokenize_result_delete(result) }
        let count = try? LiteRTLMNativeError.read(.execute, "Failed to count the tokens", 0) {
            litert_lm_tokenize_result_get_num_tokens(result, $0)
        }
        return count.map(Double.init) ?? -1.0
    }

    /// Tokens in `conversation`'s context (KV cache), or -1 when the engine
    /// can't say. On `queue`.
    static func tokenCount(_ conversation: OpaquePointer) -> Double {
        let count = try? LiteRTLMNativeError.read(.execute, "Failed to count the context's tokens", 0) {
            litert_lm_conversation_get_token_count(conversation, $0)
        }
        return count.map(Double.init) ?? -1
    }
    
    public func getMemoryUsage() throws -> MemoryUsage {
        Self.processMemoryUsage()
    }

    /// Process memory, shared with the embedder (one app, one memory budget).
    static func processMemoryUsage() -> MemoryUsage {
        var residentBytes: Double = 0.0
        var nativeHeapBytes: Double = 0.0
        
        // Retrieve process resident set size (RSS) via Mach basic task info
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            residentBytes = Double(info.resident_size)
            nativeHeapBytes = Double(info.resident_size)
        }
        
        // os_proc_available_memory reports actual headroom available before Jetsam termination (iOS 13+)
        var availableBytes = Double(os_proc_available_memory())
        #if targetEnvironment(simulator)
        if availableBytes == 0 {
            // iOS Simulator fallback: default to 4GB of simulated headroom, minus resident size
            availableBytes = max(0.0, 4.0 * 1024.0 * 1024.0 * 1024.0 - residentBytes)
        }
        #endif
        
        // Flag memory warning at ~200MB remaining headroom
        let isLowMemory = availableBytes < 200.0 * 1024.0 * 1024.0
        
        return MemoryUsage(
            nativeHeapBytes: nativeHeapBytes,
            residentBytes: residentBytes,
            availableMemoryBytes: availableBytes,
            isLowMemory: isLowMemory
        )
    }
    
    /// Tokens in the context as of the last completed turn (-1 when unknown).
    public func getContextTokenCount() throws -> Double {
        withSnapshot { $0.contextTokens }
    }

    /// Reports not-ready at once and cancels a message still generating; the
    /// promise resolves once `queue` has freed the engine.
    public func unload() throws -> Promise<Void> {
        markUnloaded()
        cancelGeneration()
        let promise = Promise<Void>()
        let teardown = Teardowns.begin()
        queue.async {
            self.closeInternal()
            Teardowns.end(teardown)
            promise.resolve()
        }
        return promise
    }

    public func setMemoryWarningCallback(
        onWarning: @escaping (_ level: MemoryWarningLevel, _ usage: MemoryUsage) -> Void
    ) throws {
        withSnapshot { $0.memoryWarningCallback = onWarning }
        startMemoryPressureMonitoring()
    }

    public func clearMemoryWarningCallback() throws {
        withSnapshot { $0.memoryWarningCallback = nil }
        stopMemoryPressureMonitoring()
    }

    private func startMemoryPressureMonitoring() {
        let source: DispatchSourceMemoryPressure? = withSnapshot { s in
            guard s.memoryPressureSource == nil else { return nil }
            let source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.warning, .critical], queue: memoryQueue)
            s.memoryPressureSource = source
            return source
        }
        guard let source = source else { return }
        source.setEventHandler { [weak self, weak source] in
            guard let self = self, let source = source,
                  let callback = self.withSnapshot({ $0.memoryWarningCallback }) else { return }
            let level: MemoryWarningLevel = source.data.contains(.critical) ? .critical : .moderate
            callback(level, Self.processMemoryUsage())
        }
        source.resume()
    }

    private func stopMemoryPressureMonitoring() {
        let source: DispatchSourceMemoryPressure? = withSnapshot { s in
            defer { s.memoryPressureSource = nil }
            return s.memoryPressureSource
        }
        source?.cancel()
    }

    /// Returns at once: the instance reports not-ready and stops memory
    /// warnings immediately, a message still generating is cancelled, and
    /// `queue` frees the engine once that message has stopped.
    public func close() throws {
        withSnapshot { s in
            s.memoryWarningCallback = nil
            s.closed = true
        }
        stopMemoryPressureMonitoring()
        markUnloaded()
        cancelGeneration()
        let teardown = Teardowns.begin()
        queue.async {
            self.closeInternal()
            Teardowns.end(teardown)
        }
    }

    /// The not-loaded state, published to the JS thread ahead of the teardown
    /// on `queue`; also moves the epoch so work still in flight lands nowhere.
    private func markUnloaded() {
        withSnapshot { s in
            s.epoch += 1
            s.lifecycle += 1
            s.isLoaded = false
            s.history = []
            s.stats = .zero
            s.contextTokens = -1
            s.modelSizeBytes = 0
        }
    }
    
    // MARK: - Async Operations
    
    public func loadModel(modelPath: String, config: LLMConfig?) throws -> Promise<Void> {
        let promise = Promise<Void>()
        // A new load replaces the conversation: work still in flight in the
        // old one lands nowhere, and a close() or unload() that follows wins.
        let started: Int? = withSnapshot { s in
            guard !s.closed else { return nil }
            s.epoch += 1
            s.lifecycle += 1
            return s.lifecycle
        }
        guard let lifecycle = started else {
            promise.reject(withError: LiteRTLMNativeError(
                .load, status: "FAILED_PRECONDITION",
                "This LiteRTLM instance is closed; create a new one with createLLM()."))
            return promise
        }
        cancelGeneration()
        let teardownsBefore = Teardowns.mark()

        queue.async {
            // Teardown any previous contexts
            self.closeInternal()
            // An engine another instance closed before this call is still
            // being freed on its own queue; don't hold both in memory.
            Teardowns.wait(through: teardownsBefore)

            // A framework from another C API version must not be called at all.
            do { try EngineAPI.requireCompatible(.load) } catch {
                promise.reject(withError: error)
                return
            }

            // A missing file would otherwise burn through all four engine
            // attempts below and surface as a generic creation failure.
            guard FileManager.default.fileExists(atPath: modelPath) else {
                promise.reject(withError: LiteRTLMNativeError(
                    .load, status: "NOT_FOUND", code: "MODEL_NOT_FOUND",
                    "Model file not found: \(modelPath)"))
                return
            }
            
            // Extract configurations
            if let config = config {
                if let b = config.backend { self.backend = b }
                if let t = config.temperature { self.temperature = t }
                if let k = config.topK { self.topK = Int(k) }
                if let p = config.topP { self.topP = p }
                // New split fields take priority over legacy maxTokens
                if let ctx = config.maxContextTokens { self.maxContextTokens = Int(ctx) }
                if let out = config.maxOutputTokens { self.maxOutputTokens = Int(out) }
                // Legacy: if only maxTokens is set, map to both for backward compat
                if config.maxContextTokens == nil && config.maxOutputTokens == nil,
                   let m = config.maxTokens {
                    self.maxContextTokens = Int(m)
                    self.maxOutputTokens = Int(m)
                }
                if let s = config.systemPrompt { self.systemPrompt = s }
                self.tools = config.tools
                self.speculativeDecodingMode = SpeculativeDecodingMode.resolve(
                    config.speculativeDecodingMode, legacy: config.enableSpeculativeDecoding)
                self.exactStats = config.exactStats ?? false
                self.chatTemplate = config.chatTemplate
                self.numThreads = config.numThreads.map { Int($0) }
                self.maxVisionTokensPerImage = config.maxVisionTokensPerImage.map { Int($0) }
                self.metalResidencySet = config.gpu?.metalResidencySet
                self.singleThreadedExecution = config.singleThreadedExecution
                self.prefillChunkSize = config.prefillChunkSize.map { Int($0) }
                self.activationDataType = config.activationDataType
                self.loraPath = config.loraPath
                self.audioLoraPath = config.audioLoraPath
                self.loraRank = config.loraRank.map { Int($0) }
                self.streamToolCalls = config.streamToolCalls ?? false
                if let ch = config.toolCallChannelName { self.toolCallChannelName = ch }
                self.enableStructuredOutput = config.enableStructuredOutput ?? false
                self.thinkingOptions = config.thinking
            } else {
                self.tools = nil
                self.speculativeDecodingMode = nil
                self.exactStats = false
                self.chatTemplate = nil
                self.numThreads = nil
                self.maxVisionTokensPerImage = nil
                self.metalResidencySet = nil
                self.singleThreadedExecution = nil
                self.prefillChunkSize = nil
                self.activationDataType = nil
                self.loraPath = nil
                self.audioLoraPath = nil
                self.loraRank = nil
                self.streamToolCalls = false
                self.toolCallChannelName = "tool_call"
                self.enableStructuredOutput = false
                self.thinkingOptions = nil
            }
            
            // Map main backend string
            let mainBackendStr = self.backend == .gpu ? "gpu" : (self.backend == .npu ? "gpu" : "cpu")
            
            // Encoders come from the bundle's own metadata (LiteRT-LM 0.18 model
            // info): with `multimodal` unset, every encoder the bundle has loads,
            // each on the backend the bundle declares (Gemma 4 E2B: CPU for
            // both — ~45 MB and ~0.3 s at load; images add their memory only
            // when sent). `multimodal: false` keeps the engine text-only. Only
            // when the metadata is unreadable does 0.7's file-name default
            // apply; that fallback is deprecated and goes in 1.0.
            let plan = ModelBundle.encoderPlan(modelPath)
            let wantEncoders = config?.multimodal ?? (plan != nil || Self.legacyMultimodalHint(modelPath))
            // Unreadable metadata (older bundles): 0.7's choice — GPU vision, CPU audio.
            let visionBackend = wantEncoders ? (plan == nil ? "gpu" : plan?.vision) : nil
            let audioBackend = wantEncoders ? (plan == nil ? "cpu" : plan?.audio) : nil
            if config?.multimodal == true, let p = plan, p.vision == nil && p.audio == nil {
                NSLog("[LiteRTLM] multimodal: true, but this bundle has no vision or audio encoder — loading text-only")
            }
            
            // Whether the bundle ships a drafter; without one, speculation is skipped.
            self.bundleHasDrafter = (try? ModelBundle.read(modelPath))?.hasSpeculativeDecoding ?? false

            var rawEngine: OpaquePointer? = nil
            /// The engine's own reason for the most recent failed attempt.
            var lastEngineError: LiteRTLMNativeError?

            // Set LiteRT C Log Level to WARNING for clean production output.
            // v0.15: takes a typed LiteRtLmLogSeverity (previously a raw int,
            // where 2 was INFO — WARNING is the intended level).
            do {
                try LiteRTLMNativeError.check(.load, "Failed to set the engine's log level") {
                    litert_lm_set_min_log_level(kLiteRtLmLogSeverityWarning)
                }
            } catch {
                NSLog("[LiteRTLM] %@", "\(error)")
            }

            // One attempt: the engine, or nil with the engine's reason in
            // `lastEngineError`. A setting the engine rejects throws instead:
            // every attempt would fail on it the same way.
            let createEngine = { (main: String, vision: String?, audio: String?) throws -> OpaquePointer? in
                let settings: OpaquePointer
                do {
                    settings = try LiteRTLMNativeError.created(.load, "Invalid engine settings (\(main))") { out in
                        modelPath.withCString { modelC in
                            withOptionalCString(main) { mainC in
                                withOptionalCString(vision) { visionC in
                                    withOptionalCString(audio) { audioC in
                                        litert_lm_engine_settings_create(modelC, mainC, visionC, audioC, out)
                                    }
                                }
                            }
                        }
                    }
                } catch let error as LiteRTLMNativeError {
                    lastEngineError = error
                    return nil
                }
                defer { litert_lm_engine_settings_delete(settings) }

                try self.configureEngineSettings(settings, modelPath: modelPath)

                do {
                    return try LiteRTLMNativeError.created(.load, "Failed to create the engine (\(main))") {
                        litert_lm_engine_create(settings, $0)
                    }
                } catch let error as LiteRTLMNativeError {
                    lastEngineError = error
                    return nil
                }
            }
            
            // The requested configuration first, then progressively more
            // conservative CPU ones — each distinct attempt tried once, and never
            // with an encoder the caller didn't ask for or the bundle lacks.
            let vision = visionBackend, audio = audioBackend
            var attempts: [(main: String, vision: String?, audio: String?)] = [(mainBackendStr, vision, audio)]
            for fallback in [
                ("cpu", vision, audio),                                  // keep the encoders' backends
                ("cpu", vision == nil ? nil : "cpu", audio == nil ? nil : "cpu"), // everything on CPU
                ("cpu", nil as String?, nil as String?),                 // text only
            ] where !attempts.contains(where: { $0.main == fallback.0 && $0.vision == fallback.1 && $0.audio == fallback.2 }) {
                attempts.append(fallback)
            }
            for (index, attempt) in attempts.enumerated() {
                if index == 1 {
                    NSLog("[LiteRTLM] %@ engine failed — trying the CPU fallbacks", mainBackendStr.uppercased())
                }
                do {
                    rawEngine = try createEngine(attempt.main, attempt.vision, attempt.audio)
                } catch {
                    promise.reject(withError: error)
                    return
                }
                if rawEngine != nil {
                    self.hasVisionEncoder = attempt.vision != nil
                    self.hasAudioEncoder = attempt.audio != nil
                    if index > 0 {
                        NSLog("[LiteRTLM] Loaded with the fallback (main %@, vision %@, audio %@)",
                              attempt.main, attempt.vision ?? "none", attempt.audio ?? "none")
                        self.backend = .cpu
                    }
                    break
                }
            }
            
            guard let engine = rawEngine else {
                // Report the last attempt (the most permissive config): if even
                // CPU text-only failed, its reason is the real one.
                let reason = lastEngineError
                    ?? LiteRTLMNativeError(.load, status: "UNKNOWN", "Failed to create the LiteRT-LM engine")
                promise.reject(withError: LiteRTLMNativeError(
                    .load, status: reason.status, code: reason.code,
                    "\(reason.message) — tried \(mainBackendStr), then the CPU fallbacks"))
                return
            }
            
            self.engineLock.lock()
            self.engine = engine
            self.engineLock.unlock()
            let conversationFailure = self.createNewConversation()
            self.loadedModelPath = modelPath

            if let failure = conversationFailure {
                self.closeInternal()
                promise.reject(withError: failure)
                return
            }

            let attributes = try? FileManager.default.attributesOfItem(atPath: modelPath)
            let tokens = self.conversation.map(Self.tokenCount) ?? -1
            self.withSnapshot { s in
                // A close() or unload() called during the load wins; its
                // teardown is already queued behind this block.
                guard s.lifecycle == lifecycle else { return }
                s.isLoaded = true
                s.modelSizeBytes = attributes?[.size] as? Int ?? 0
                s.contextTokens = tokens
            }
            promise.resolve()
        }
        
        return promise
    }
    
    // Legacy inference — shapes mirror src/inferenceRouting.ts; JS createLLM routes via execute.
    public func sendMessage(message: String) throws -> Promise<String> {
        try execute(parts: [.textPart(message)], onToken: nil, options: nil)
    }

    public func sendMessageAsync(
        message: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message)], onToken: onToken)
    }

    public func sendMessageWithImage(message: String, imagePath: String) throws -> Promise<String> {
        try execute(parts: [.textPart(message), .imagePart(imagePath)], onToken: nil, options: nil)
    }

    public func sendMessageWithImageAsync(
        message: String, imagePath: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message), .imagePart(imagePath)], onToken: onToken)
    }

    public func sendMessageWithAudioAsync(
        message: String, audioPath: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        try executeVoid(parts: [.textPart(message), .audioPart(audioPath)], onToken: onToken)
    }

    public func sendMessageWithAudio(message: String, audioPath: String) throws -> Promise<String> {
        try execute(parts: [.textPart(message), .audioPart(audioPath)], onToken: nil, options: nil)
    }

    public func sendMultimodalMessage(parts: [MultimodalPart]) throws -> Promise<String> {
        try execute(parts: parts, onToken: nil, options: nil)
    }

    public func downloadModel(
        url: String,
        fileName: String,
        onProgress: ((Double) -> Void)?
    ) throws -> Promise<String> {
        return try modelStore.downloadFile(
            url: url,
            fileName: fileName,
            headersJson: "{}",
            sha256: "",
            onProgress: { progress in
                onProgress?(progress)
            }
        )
    }
    
    public func deleteModel(fileName: String) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            do {
                try self.modelStore.deleteFile(fileName: fileName)
                let currentlyLoadedName = self.loadedModelPath.map { ($0 as NSString).lastPathComponent.lowercased() }
                if let loadedName = currentlyLoadedName, loadedName == fileName.lowercased() {
                    if self.engine != nil {
                        // A stream's turn still waiting to be recorded lands nowhere.
                        self.withSnapshot { $0.epoch += 1 }
                        self.closeInternal()
                    }
                    // The backing file is gone — clear the stale path even if
                    // no engine was live (closeInternal handles the loaded case).
                    self.loadedModelPath = nil
                }
                promise.resolve()
            } catch {
                promise.reject(withError: error)
            }
        }
        
        return promise
    }
    
    // MARK: - Internal Engine Helpers

    /// Apply the load's settings to one engine attempt. Throws when the engine
    /// rejects one (C API 1.0 setters return a status). Runs on `queue`.
    private func configureEngineSettings(_ s: OpaquePointer, modelPath: String) throws {
        func set(_ what: String, _ call: () -> LiteRtLmStatusCode) throws {
            try LiteRTLMNativeError.check(.load, "Failed to set \(what)", call)
        }
        try set("maxContextTokens \(maxContextTokens)") {
            litert_lm_engine_settings_set_max_num_tokens(s, Int32(maxContextTokens))
        }
        if exactStats {
            try set("benchmark mode (exactStats)") { litert_lm_engine_settings_enable_benchmark(s) }
        }

        // Memory tuning knobs (v0.14 C API)
        if let threads = numThreads {
            try set("numThreads \(threads)") { litert_lm_engine_settings_set_num_threads(s, Int32(threads)) }
        }
        if let chunk = prefillChunkSize {
            try set("prefillChunkSize \(chunk)") { litert_lm_engine_settings_set_prefill_chunk_size(s, Int32(chunk)) }
        }
        // LiteRT-LM 0.18 engine knobs; unset leaves the engine default.
        if let tokens = maxVisionTokensPerImage {
            try set("maxVisionTokensPerImage \(tokens)") {
                litert_lm_engine_settings_set_max_vision_tokens_per_image(s, Int32(tokens))
            }
        }
        if let residency = metalResidencySet {
            try set("gpu.metalResidencySet") { litert_lm_engine_settings_set_gpu_enable_metal_residency_set(s, residency) }
        }
        if let single = singleThreadedExecution {
            try set("singleThreadedExecution") { litert_lm_engine_settings_set_single_threaded_execution(s, single) }
        }
        if let adt = activationDataType {
            try set("activationDataType") { litert_lm_engine_settings_set_activation_data_type(s, adt.cValue) }
        }
        if let rank = loraRank {
            try set("loraRank \(rank)") { litert_lm_engine_settings_set_lora_rank(s, Int32(rank)) }
            let ranks: [Int32] = [Int32(rank)]
            try set("the supported LoRA ranks") {
                ranks.withUnsafeBufferPointer { litert_lm_engine_settings_set_supported_lora_ranks(s, $0.baseAddress, $0.count) }
            }
        }

        // The engine's default for every conversation. On or adaptive loads
        // the bundle's drafter now; a bundle without one loads without it.
        if let mode = speculativeDecodingMode {
            if mode == .off || bundleHasDrafter {
                try mode.apply(.load) { litert_lm_engine_settings_set_speculative_decoding_config(s, $0) }
            } else {
                NSLog("[LiteRTLM] speculativeDecoding %@ skipped: this bundle has no drafter", mode.stringValue)
            }
        }

        // Cache dir set to parent directory of model path
        let cacheDir = (modelPath as NSString).deletingLastPathComponent
        try set("the cache directory") {
            cacheDir.withCString { litert_lm_engine_settings_set_cache_dir(s, $0) }
        }
    }

    /// Replace the active conversation. Returns the failure (nil on success);
    /// must run on `queue` so the engine's thread-local error is readable.
    @discardableResult
    private func createNewConversation(
        initialMessagesJson: String? = nil,
        systemPromptOverride: String? = nil,
        speculativeDecoding: SpeculativeDecodingMode? = nil
    ) -> LiteRTLMNativeError? {
        guard let engine = self.engine else {
            return LiteRTLMNativeError(.load, status: "FAILED_PRECONDITION", code: "MODEL_NOT_LOADED",
                                       "No model loaded. Call loadModel() first.")
        }

        if self.conversation != nil {
            settleStreams()
            if let oldConv = swapConversation(nil) {
                litert_lm_conversation_delete(oldConv)
            }
        }

        do {
            let created = try makeConversation(
                engine,
                initialMessagesJson: initialMessagesJson,
                systemPromptOverride: systemPromptOverride,
                speculativeDecoding: speculativeDecoding)
            swapConversation(created)
            return nil
        } catch let error as LiteRTLMNativeError {
            return error
        } catch {
            return LiteRTLMNativeError(.load, status: "UNKNOWN", "Failed to create the conversation: \(error)")
        }
    }

    /// A new conversation on `engine` with this load's settings. Throws the
    /// engine's reason when a setting or the creation fails. On `queue`.
    private func makeConversation(
        _ engine: OpaquePointer,
        initialMessagesJson: String?,
        systemPromptOverride: String?,
        speculativeDecoding: SpeculativeDecodingMode?
    ) throws -> OpaquePointer {
        func set(_ what: String, _ call: () -> LiteRtLmStatusCode) throws {
            try LiteRTLMNativeError.check(.load, "Failed to set \(what)", call)
        }

        let convConfig = try LiteRTLMNativeError.created(.load, "Failed to allocate a conversation config") {
            litert_lm_conversation_config_create($0)
        }
        defer { litert_lm_conversation_config_delete(convConfig) }

        let sessionConfig = try LiteRTLMNativeError.created(.load, "Failed to allocate a session config") {
            litert_lm_session_config_create($0)
        }
        defer { litert_lm_session_config_delete(sessionConfig) }

        try set("maxOutputTokens \(maxOutputTokens)") {
            litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(maxOutputTokens))
        }

        // v0.14: LiteRtLmSamplerParams is an opaque handle (create/set/delete)
        let sampler = try LiteRTLMNativeError.created(.load, "Failed to allocate the sampler parameters") {
            litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP, $0)
        }
        defer { litert_lm_sampler_params_delete(sampler) }
        try set("topK \(topK)") { litert_lm_sampler_params_set_top_k(sampler, Int32(topK)) }
        try set("topP \(topP)") { litert_lm_sampler_params_set_top_p(sampler, Float(topP)) }
        try set("temperature \(temperature)") { litert_lm_sampler_params_set_temperature(sampler, Float(temperature)) }
        try set("the sampler seed") { litert_lm_sampler_params_set_seed(sampler, 0) }
        try set("the sampler parameters") { litert_lm_session_config_set_sampler_params(sessionConfig, sampler) }

        // A LoRA file the engine can't open is reported and skipped, as before 0.19.
        if let loraPath = loraPath {
            do {
                try set("the LoRA path \(loraPath)") {
                    loraPath.withCString { litert_lm_session_config_set_lora_path(sessionConfig, $0) }
                }
            } catch {
                NSLog("[LiteRTLM] %@ — continuing without it", "\(error)")
            }
        }
        if let audioLoraPath = audioLoraPath {
            do {
                try set("the audio LoRA path \(audioLoraPath)") {
                    audioLoraPath.withCString { litert_lm_session_config_set_audio_lora_path(sessionConfig, $0) }
                }
            } catch {
                NSLog("[LiteRTLM] %@ — continuing without it", "\(error)")
            }
        }

        // Per conversation: unset inherits the engine's mode; on or adaptive
        // loads the bundle's drafter lazily on first use, so a bundle without
        // one skips it.
        if let mode = speculativeDecoding {
            if mode == .off || bundleHasDrafter {
                try mode.apply(.load) { litert_lm_session_config_set_speculative_decoding_config(sessionConfig, $0) }
            } else {
                NSLog("[LiteRTLM] speculativeDecoding %@ skipped: this bundle has no drafter", mode.stringValue)
            }
        }

        try set("the session config") { litert_lm_conversation_config_set_session_config(convConfig, sessionConfig) }

        if let template = chatTemplate {
            try set("chatTemplate") {
                template.withCString { litert_lm_conversation_config_set_prompt_template(convConfig, $0) }
            }
        }

        if streamToolCalls {
            try set("streamToolCalls") {
                toolCallChannelName.withCString { litert_lm_conversation_config_set_stream_tool_calls(convConfig, true, $0) }
            }
        }

        if enableStructuredOutput {
            // v0.15: initialize the LLGuidance constraint provider so per-message
            // responseSchema/responseRegex constraints can be applied.
            try set("enableStructuredOutput") {
                litert_lm_conversation_config_set_enable_constrained_decoding(convConfig, true)
            }
            try set("the LLGuidance constraint provider") {
                litert_lm_conversation_config_set_constraint_provider(convConfig, kLiteRtLmConstraintProviderTypeLlGuidance)
            }
        }

        if let thinking = thinkingOptions {
            let thinkingConfig = try LiteRTLMNativeError.created(.load, "Failed to allocate a thinking config") {
                litert_lm_thinking_config_create($0)
            }
            defer { litert_lm_thinking_config_delete(thinkingConfig) }
            try set("thinking.enabled") {
                litert_lm_thinking_config_set_enable_thinking(thinkingConfig, thinking.enabled ?? true)
            }
            if let budget = thinking.tokenBudget {
                try set("thinking.tokenBudget \(budget)") {
                    litert_lm_thinking_config_set_thinking_token_budget(thinkingConfig, Int32(budget))
                }
            }
            try set("the thinking config") { litert_lm_conversation_config_set_thinking_config(convConfig, thinkingConfig) }
        }

        if let systemPrompt = systemPromptOverride ?? self.systemPrompt {
            // The engine wraps this payload itself ({"role":"system","content":<payload>}),
            // so pass content only, as a content-part ARRAY: the Gemma chat template
            // iterates non-string content as a list of {type, text} parts. (Verified
            // by integration test: a bare {"type":"text","text":…} object — the form
            // upstream's engine_test stores — does NOT render; the array form does.
            // The old {"role":…,"content":…} payload double-wrapped and rendered empty.)
            let contentParts: [[String: Any]] = [["type": "text", "text": systemPrompt]]
            if let data = try? JSONSerialization.data(withJSONObject: contentParts),
               let systemMsgJson = String(data: data, encoding: .utf8) {
                try set("the system prompt") {
                    systemMsgJson.withCString { litert_lm_conversation_config_set_system_message(convConfig, $0) }
                }
            }
        }

        if let messagesJson = initialMessagesJson, !messagesJson.isEmpty {
            // Seed the conversation with a prior transcript (multi-conversation
            // switching). The engine re-prefills these on the next message.
            try set("the replayed transcript") {
                messagesJson.withCString { litert_lm_conversation_config_set_messages(convConfig, $0) }
            }
        }

        if let tools = tools, !tools.isEmpty {
            try set("the tools") {
                Self.toolsJson(tools).withCString { litert_lm_conversation_config_set_tools(convConfig, $0) }
            }
        }

        return try LiteRTLMNativeError.created(.load, "Failed to create the conversation") {
            litert_lm_conversation_create(engine, convConfig, $0)
        }
    }
    

    /// Free the conversation and engine. Runs on `queue` (or in deinit).
    private func closeInternal() {
        hasVisionEncoder = false
        hasAudioEncoder = false
        loadedModelPath = nil
        conversationFailure = nil

        if self.conversation != nil {
            settleStreams()
            if let conversation = swapConversation(nil) {
                litert_lm_conversation_delete(conversation)
            }
        }
        // countTokens() holds engineLock while it uses the engine; once the
        // pointer is cleared under it, no caller can still be using it.
        engineLock.lock()
        let engine = self.engine
        self.engine = nil
        engineLock.unlock()
        if let engine = engine {
            litert_lm_engine_delete(engine)
        }

        withSnapshot { s in
            s.isLoaded = false
            s.history = []
            s.stats = .zero
            s.contextTokens = -1
            s.modelSizeBytes = 0
        }
    }

    // MARK: - Generations in flight

    /// Install a conversation (or none) and return the one it replaces. The
    /// caller, on `queue`, deletes the old one: once it's out of the property,
    /// cancelGeneration() can't reach it.
    @discardableResult
    private func swapConversation(_ new: OpaquePointer?) -> OpaquePointer? {
        conversationLock.lock()
        defer { conversationLock.unlock() }
        let old = conversation
        conversation = new
        return old
    }

    /// Stop whatever the current conversation is generating — a blocking
    /// send on `queue` or a stream on the engine's thread — without waiting:
    /// it ends at its next decode step with a CANCELLED error. Callable from
    /// any thread. Only for a conversation about to be replaced or freed,
    /// since LiteRT-LM leaves a cancelled session unusable.
    private func cancelGeneration() {
        conversationLock.lock()
        defer { conversationLock.unlock() }
        if let conversation = conversation {
            Self.cancel(conversation)
        }
    }

    /// Request a cancel (c/conversation.h: it doesn't wait). Fails only for a
    /// NULL conversation, which never reaches here; logged if it ever does.
    private static func cancel(_ conversation: OpaquePointer) {
        do {
            try LiteRTLMNativeError.check(.execute, "Failed to cancel the message generating") {
                litert_lm_conversation_cancel_process(conversation)
            }
        } catch {
            NSLog("[LiteRTLM] %@", "\(error)")
        }
    }

    /// Before a conversation is deleted: a stream still generating on the
    /// engine's thread is cancelled, and this waits for its last callback so
    /// the conversation isn't freed under it. The engine completes its own
    /// work on the conversation before that callback, and the session's
    /// destructor waits for the task to return. Runs on `queue`, never on the
    /// JS thread.
    private func settleStreams() {
        guard withSnapshot({ $0.streamsInFlight }) > 0, let conversation = conversation else { return }
        Self.cancel(conversation)
        // A cancel lands at the next decode step, but a prefill runs to the
        // end first, which can take a while for a long prompt. Freeing the
        // conversation before the stream's last callback would pull it out
        // from under the engine, so wait it out.
        while streams.wait(timeout: .now() + 10) == .timedOut {
            NSLog("[LiteRTLM] Still waiting for a cancelled stream to end before freeing its conversation")
        }
    }

    /// Call on `queue` right before starting a stream.
    func streamStarted() {
        withSnapshot { $0.streamsInFlight += 1 }
        streams.enter()
    }

    /// Call exactly once per started stream, from its final or error callback
    /// (the engine's thread) or when it failed to start.
    func streamEnded() {
        withSnapshot { $0.streamsInFlight -= 1 }
        streams.leave()
    }

    /// On `queue`, before a message with `exactStats` counts the engine's
    /// turns: wait for streams still generating, which would otherwise add
    /// their own turns (a reply's thinking refill among them) after the count.
    /// The engine runs the two messages one after the other anyway.
    func awaitStreams() {
        while streams.wait(timeout: .now() + 10) == .timedOut {
            NSLog("[LiteRTLM] Still waiting for a stream to end before counting the next message's turns")
        }
    }
    
    /// 0.7's default for `multimodal` when unset (on for Gemma 3n / Gemma 3
    /// file names), now only for bundles whose metadata can't be read.
    /// Deprecated — the JS layer warns; removed in 1.0.
    static func legacyMultimodalHint(_ modelPath: String) -> Bool {
        let name = (modelPath as NSString).lastPathComponent.lowercased()
        return name.contains("3n") || name.contains("gemma3")
    }

    // MARK: - Internal Preprocessing Helpers (for extension files)
    
    private let kControlTokens = [
        "<end_of_turn>",
        "<start_of_turn>model",
        "<start_of_turn>user",
        "<start_of_turn>",
        "<eos>"
    ]
    
    func stripControlTokens(_ text: String) -> String {
        var result = text
        for tok in kControlTokens {
            result = result.replacingOccurrences(of: tok, with: "")
        }
        return result
    }
    
    func safeEmitLength(_ text: String) -> Int {
        let chars = Array(text)
        guard let lastAngleIdx = chars.lastIndex(of: "<") else {
            return chars.count
        }
        let suffix = String(chars[lastAngleIdx...])
        for tok in kControlTokens {
            if tok.hasPrefix(suffix) && suffix.count < tok.count {
                return lastAngleIdx
            }
        }
        return chars.count
    }
    
    func extractTextFromResponse(_ jsonResponse: String) -> String {
        guard let data = jsonResponse.data(using: .utf8) else {
            return stripControlTokens(jsonResponse)
        }
        do {
            if let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] {
                var textResult = ""
                if let content = json["content"] {
                    if let contentString = content as? String {
                        textResult += contentString
                    } else if let contentArray = content as? [[String: Any]] {
                        for part in contentArray {
                            if let type = part["type"] as? String, type == "text", let text = part["text"] as? String {
                                textResult += text
                            }
                        }
                    }
                }
                // Tool calls arrive as {"role":"assistant","tool_calls":[{"type":
                // "function","function":{"name":…,"arguments":{…}}}]} (verified on
                // Gemma 4 E2B). Surface them in the <tool_call>{json}</tool_call>
                // marker protocol the JS event parser understands instead of
                // leaking the raw message JSON into the token stream. Made from
                // the response's text, so the arguments keep the model's order.
                if json["tool_calls"] is [[String: Any]],
                   let calls = JSONText.member("tool_calls", of: jsonResponse).flatMap(JSONText.elements(of:)) {
                    textResult += calls.map(Self.toolCallMarker).joined()
                }
                // A message-shaped payload never leaks raw JSON — if it carried
                // neither text nor tool calls, it contributes nothing.
                if json["role"] != nil || json["content"] != nil || json["tool_calls"] != nil {
                    return stripControlTokens(textResult)
                }
            }
        } catch {}
        return stripControlTokens(jsonResponse)
    }
    
}

extension GenerationStats {
    /// Stats before any turn has run.
    static var zero: GenerationStats {
        GenerationStats(promptTokens: 0, completionTokens: 0, totalTokens: 0,
                        timeToFirstToken: 0, totalTime: 0, tokensPerSecond: 0)
    }
}

/// Engine teardowns that close() or unload() queued, on any instance, and
/// that haven't finished. A load waits for the ones queued before it was
/// called: models are gigabytes, and one loaded right after another was
/// closed shouldn't sit beside it in memory. A load never waits for a
/// teardown queued after it was called, so the waits can't form a cycle.
enum Teardowns {
    private static let condition = NSCondition()
    private static var issued = 0
    private static var pending = Set<Int>()

    static func begin() -> Int {
        condition.lock()
        defer { condition.unlock() }
        issued += 1
        pending.insert(issued)
        return issued
    }

    static func end(_ ticket: Int) {
        condition.lock()
        defer { condition.unlock() }
        pending.remove(ticket)
        condition.broadcast()
    }

    /// The latest teardown so far — what a load called now waits for.
    static func mark() -> Int {
        condition.lock()
        defer { condition.unlock() }
        return issued
    }

    /// Block until every teardown up to `ticket` has finished. Never call it
    /// on the JS thread.
    static func wait(through ticket: Int) {
        condition.lock()
        defer { condition.unlock() }
        while pending.contains(where: { $0 <= ticket }) {
            condition.wait()
        }
    }
}
