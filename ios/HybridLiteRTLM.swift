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
    
    /// Dedicated background serial queue to protect the JSI/JS thread from blocking and deadlocks (User Rule #1).
    let queue = DispatchQueue(label: "dev.litert.engine", qos: .userInteractive)
    
    /// Opaque pointer to the active conversation state.
    var conversation: OpaquePointer?
    
    /// Conversation history.
    var history: [Message] = []
    
    /// Latest inference generation statistics.
    var lastStats = GenerationStats(
        promptTokens: 0.0,
        completionTokens: 0.0,
        totalTokens: 0.0,
        timeToFirstToken: 0.0,
        totalTime: 0.0,
        tokensPerSecond: 0.0
    )
    
    var loadedModelPath: String?
    let modelStore = HybridModelStore()
    
    // MARK: - Private state
    
    /// Opaque pointer to the LiteRT LM C Engine.
    private var engine: OpaquePointer?
    
    /// Thread-safe status flag.
    private var isLoaded = false
    
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
    private var enableSpeculativeDecoding: Bool = false
    private var chatTemplate: String?
    private var numThreads: Int?
    private var maxVisionTokensPerImage: Int?
    private var metalResidencySet: Bool?
    private var singleThreadedExecution: Bool?
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

    /// Size of the loaded model file in bytes (0 when unloaded).
    /// Read lock-free from `memorySize` (GC may query mid-load; a stale read
    /// is harmless, blocking the JS thread behind a model load is not).
    private var loadedModelSizeBytes: Int = 0

    /// OS memory-pressure listener (DISPATCH_SOURCE_TYPE_MEMORYPRESSURE).
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private var memoryWarningCallback: ((MemoryWarningLevel, MemoryUsage) -> Void)?

    /// Actual model weight size to inform the JS engine's garbage collection.
    public var memorySize: Int {
        return loadedModelSizeBytes
    }

    deinit {
        stopMemoryPressureMonitoring()
        closeInternal()
    }
    
    // MARK: - Core Hybrid Object API
    
    public func isReady() throws -> Bool {
        return queue.sync { isLoaded }
    }
    
    public func getHistory() throws -> [Message] {
        return queue.sync { history }
    }
    
    public func resetConversation(historyJson: String?, systemPrompt: String?, speculativeDecoding: Bool? = nil) throws {
        let failure: LiteRTLMNativeError? = queue.sync {
            // Mirror the replayed transcript into the wrapper's own history so
            // getHistory() stays truthful for the restored conversation.
            history = Self.parseTranscript(historyJson)
            lastStats = GenerationStats(
                promptTokens: 0.0,
                completionTokens: 0.0,
                totalTokens: 0.0,
                timeToFirstToken: 0.0,
                totalTime: 0.0,
                tokensPerSecond: 0.0
            )
            guard isLoaded && engine != nil else { return nil }
            // Replay in the engine's own message format, tool exchanges included.
            return createNewConversation(
                initialMessagesJson: Self.engineMessagesJson(history),
                systemPromptOverride: systemPrompt,
                speculativeDecoding: speculativeDecoding
            )
        }
        if let failure = failure { throw failure }
    }
    
    public func getStats() throws -> GenerationStats {
        return queue.sync { lastStats }
    }
    
    public func countTokens(text: String) throws -> Double {
        return queue.sync {
            guard let engine = self.engine else {
                return -1.0
            }
            guard let result = litert_lm_engine_tokenize(engine, text) else {
                return -1.0
            }
            let numTokens = litert_lm_tokenize_result_get_num_tokens(result)
            litert_lm_tokenize_result_delete(result)
            return Double(numTokens)
        }
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
    
    public func getContextTokenCount() throws -> Double {
        return queue.sync {
            guard let conversation = self.conversation else { return -1.0 }
            return Double(litert_lm_conversation_get_token_count(conversation))
        }
    }

    public func unload() throws -> Promise<Void> {
        let promise = Promise<Void>()
        queue.async {
            self.closeInternal()
            promise.resolve()
        }
        return promise
    }

    public func setMemoryWarningCallback(
        onWarning: @escaping (_ level: MemoryWarningLevel, _ usage: MemoryUsage) -> Void
    ) throws {
        queue.sync {
            self.memoryWarningCallback = onWarning
            self.startMemoryPressureMonitoring()
        }
    }

    public func clearMemoryWarningCallback() throws {
        queue.sync {
            self.memoryWarningCallback = nil
            self.stopMemoryPressureMonitoring()
        }
    }

    /// Must be called on `queue`.
    private func startMemoryPressureMonitoring() {
        guard memoryPressureSource == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical], queue: queue)
        source.setEventHandler { [weak self] in
            guard let self = self, let callback = self.memoryWarningCallback else { return }
            let event = source.data
            let level: MemoryWarningLevel = event.contains(.critical) ? .critical : .moderate
            if let usage = try? self.getMemoryUsage() {
                callback(level, usage)
            }
        }
        source.resume()
        memoryPressureSource = source
    }

    private func stopMemoryPressureMonitoring() {
        memoryPressureSource?.cancel()
        memoryPressureSource = nil
    }

    public func close() throws {
        queue.sync {
            self.memoryWarningCallback = nil
            self.stopMemoryPressureMonitoring()
            closeInternal()
        }
    }
    
    // MARK: - Async Operations
    
    public func loadModel(modelPath: String, config: LLMConfig?) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            // Teardown any previous contexts
            self.closeInternal()

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
                self.enableSpeculativeDecoding = config.enableSpeculativeDecoding ?? false
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
                self.enableSpeculativeDecoding = false
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
            
            var rawEngine: OpaquePointer? = nil
            /// The engine's own reason for the most recent failed attempt.
            var lastEngineError: LiteRTLMNativeError?
            
            // Set LiteRT C Log Level to WARNING for clean production output.
            // v0.15: takes a typed LiteRtLmLogSeverity (previously a raw int,
            // where 2 was INFO — WARNING is the intended level).
            litert_lm_set_min_log_level(kLiteRtLmLogSeverityWarning)
            
            // Creation helper with scoped FFI pointer lifetime
            let createEngine = { (main: String, vision: String?, audio: String?) -> OpaquePointer? in
                LiteRTLMNativeError.clearEngineError()
                let settings = modelPath.withCString { modelC in
                    withOptionalCString(main) { mainC in
                        withOptionalCString(vision) { visionC in
                            withOptionalCString(audio) { audioC in
                                return litert_lm_engine_settings_create(modelC, mainC, visionC, audioC)
                            }
                        }
                    }
                }
                
                guard let s = settings else {
                    lastEngineError = LiteRTLMNativeError.fromEngine(
                        .load, context: "Invalid engine settings (\(main))")
                    return nil
                }
                defer { litert_lm_engine_settings_delete(s) }
                
                litert_lm_engine_settings_set_max_num_tokens(s, Int32(self.maxContextTokens))
                litert_lm_engine_settings_enable_benchmark(s)

                // Memory tuning knobs (v0.14 C API)
                if let threads = self.numThreads {
                    litert_lm_engine_settings_set_num_threads(s, Int32(threads))
                }
                if let chunk = self.prefillChunkSize {
                    litert_lm_engine_settings_set_prefill_chunk_size(s, Int32(chunk))
                }
                // LiteRT-LM 0.18 engine knobs; unset leaves the engine default.
                if let tokens = self.maxVisionTokensPerImage {
                    litert_lm_engine_settings_set_max_vision_tokens_per_image(s, Int32(tokens))
                }
                if let residency = self.metalResidencySet {
                    litert_lm_engine_settings_set_gpu_enable_metal_residency_set(s, residency)
                }
                if let single = self.singleThreadedExecution {
                    litert_lm_engine_settings_set_single_threaded_execution(s, single)
                }
                if let adt = self.activationDataType {
                    litert_lm_engine_settings_set_activation_data_type(s, adt.cValue)
                }
                if let rank = self.loraRank {
                    litert_lm_engine_settings_set_lora_rank(s, Int32(rank))
                    var ranks: [Int32] = [Int32(rank)]
                    _ = ranks.withUnsafeBufferPointer { buf in
                        litert_lm_engine_settings_set_supported_lora_ranks(s, buf.baseAddress, 1)
                    }
                }

                if self.enableSpeculativeDecoding {
                    if let loadedFile = litert_lm_loaded_file_create((modelPath as NSString).utf8String) {
                        let hasMtp = litert_lm_loaded_file_has_speculative_decoding_support(loadedFile)
                        litert_lm_loaded_file_delete(loadedFile)
                        if hasMtp {
                            litert_lm_engine_settings_set_enable_speculative_decoding(s, true)
                        }
                    }
                }
                
                // Cache dir set to parent directory of model path
                let cacheDir = (modelPath as NSString).deletingLastPathComponent
                cacheDir.withCString { cacheC in
                    litert_lm_engine_settings_set_cache_dir(s, cacheC)
                }
                
                // Read the engine's thread-local error right after the call,
                // on this same queue thread (c/error_reporter.h).
                LiteRTLMNativeError.clearEngineError()
                let created = litert_lm_engine_create(s)
                if created == nil {
                    lastEngineError = LiteRTLMNativeError.fromEngine(
                        .load, context: "Failed to create the engine (\(main))")
                }
                return created
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
                rawEngine = createEngine(attempt.main, attempt.vision, attempt.audio)
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
            
            self.engine = engine
            let conversationFailure = self.createNewConversation()
            self.loadedModelPath = modelPath
            self.loadedModelSizeBytes =
                (try? FileManager.default.attributesOfItem(atPath: modelPath)[.size] as? Int).flatMap { $0 } ?? 0
            
            if let failure = conversationFailure {
                self.closeInternal()
                promise.reject(withError: failure)
                return
            }
            
            self.isLoaded = true
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
                    if self.isLoaded {
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
    
    /// Replace the active conversation. Returns the failure (nil on success);
    /// must run on `queue` so the engine's thread-local error is readable.
    @discardableResult
    private func createNewConversation(
        initialMessagesJson: String? = nil,
        systemPromptOverride: String? = nil,
        speculativeDecoding: Bool? = nil
    ) -> LiteRTLMNativeError? {
        guard let engine = self.engine else {
            return LiteRTLMNativeError(.load, status: "FAILED_PRECONDITION", code: "MODEL_NOT_LOADED",
                                       "No model loaded. Call loadModel() first.")
        }
        
        if let oldConv = self.conversation {
            litert_lm_conversation_delete(oldConv)
            self.conversation = nil
        }
        
        guard let convConfig = litert_lm_conversation_config_create() else {
            return LiteRTLMNativeError(.load, status: "INTERNAL", "Failed to allocate a conversation config")
        }
        defer { litert_lm_conversation_config_delete(convConfig) }
        
        guard let sessionConfig = litert_lm_session_config_create() else {
            return LiteRTLMNativeError(.load, status: "INTERNAL", "Failed to allocate a session config")
        }
        defer { litert_lm_session_config_delete(sessionConfig) }
        
        litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(self.maxOutputTokens))
        
        // v0.14: LiteRtLmSamplerParams is an opaque handle (create/set/delete)
        if let sampler = litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP) {
            litert_lm_sampler_params_set_top_k(sampler, Int32(self.topK))
            litert_lm_sampler_params_set_top_p(sampler, Float(self.topP))
            litert_lm_sampler_params_set_temperature(sampler, Float(self.temperature))
            litert_lm_sampler_params_set_seed(sampler, 0)
            litert_lm_session_config_set_sampler_params(sessionConfig, sampler)
            litert_lm_sampler_params_delete(sampler)
        }

        if let loraPath = self.loraPath {
            let status = loraPath.withCString { litert_lm_session_config_set_lora_path(sessionConfig, $0) }
            if status != 0 {
                NSLog("[LiteRTLM] Failed to set LoRA path (status %d): %@", status, loraPath)
            }
        }
        if let audioLoraPath = self.audioLoraPath {
            let status = audioLoraPath.withCString { litert_lm_session_config_set_audio_lora_path(sessionConfig, $0) }
            if status != 0 {
                NSLog("[LiteRTLM] Failed to set audio LoRA path (status %d): %@", status, audioLoraPath)
            }
        }

        // Per conversation: unset inherits the engine's setting; true loads the
        // bundle's drafter lazily on first use.
        if let speculative = speculativeDecoding {
            litert_lm_session_config_set_enable_speculative_decoding(sessionConfig, speculative)
        }

        litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)

        if let template = self.chatTemplate {
            template.withCString { litert_lm_conversation_config_set_prompt_template(convConfig, $0) }
        }

        if self.streamToolCalls {
            self.toolCallChannelName.withCString { channelC in
                litert_lm_conversation_config_set_stream_tool_calls(convConfig, true, channelC)
            }
        }

        if self.enableStructuredOutput {
            // v0.15: initialize the LLGuidance constraint provider so per-message
            // responseSchema/responseRegex constraints can be applied.
            litert_lm_conversation_config_set_enable_constrained_decoding(convConfig, true)
            var provider = kLiteRtLmConstraintProviderTypeLlGuidance
            litert_lm_conversation_config_set_constraint_provider(convConfig, &provider)
        }

        if let thinking = self.thinkingOptions,
           let thinkingConfig = litert_lm_thinking_config_create() {
            litert_lm_thinking_config_set_enable_thinking(thinkingConfig, thinking.enabled ?? true)
            if let budget = thinking.tokenBudget {
                litert_lm_thinking_config_set_thinking_token_budget(thinkingConfig, Int32(budget))
            }
            litert_lm_conversation_config_set_thinking_config(convConfig, thinkingConfig)
            litert_lm_thinking_config_delete(thinkingConfig)
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
                systemMsgJson.withCString { systemMsgC in
                    litert_lm_conversation_config_set_system_message(convConfig, systemMsgC)
                }
            }
        }

        if let messagesJson = initialMessagesJson, !messagesJson.isEmpty {
            // Seed the conversation with a prior transcript (multi-conversation
            // switching). The engine re-prefills these on the next message.
            messagesJson.withCString { messagesC in
                litert_lm_conversation_config_set_messages(convConfig, messagesC)
            }
        }
        
        if let tools = self.tools, !tools.isEmpty {
            var toolsArray: [[String: Any]] = []
            for tool in tools {
                var functionMap: [String: Any] = ["name": tool.name, "description": tool.description]
                if let data = tool.parametersJson.data(using: .utf8),
                   let parsedParams = try? JSONSerialization.jsonObject(with: data, options: []) {
                    functionMap["parameters"] = parsedParams
                } else {
                    // Don't silently drop the schema — the model would see a
                    // tool with no parameters and hallucinate arguments.
                    NSLog("[LiteRTLM] Tool '%@' has unparseable parametersJson — sending empty schema", tool.name)
                    functionMap["parameters"] = [String: Any]()
                }
                toolsArray.append(["type": "function", "function": functionMap])
            }
            if let data = try? JSONSerialization.data(withJSONObject: toolsArray, options: []),
               let jsonString = String(data: data, encoding: .utf8) {
                jsonString.withCString { toolsC in
                    litert_lm_conversation_config_set_tools(convConfig, toolsC)
                }
            }
        }
        
        LiteRTLMNativeError.clearEngineError()
        self.conversation = litert_lm_conversation_create(engine, convConfig)
        if self.conversation == nil {
            return LiteRTLMNativeError.fromEngine(.load, context: "Failed to create the conversation")
        }
        return nil
    }
    
    private func closeInternal() {
        isLoaded = false
        hasVisionEncoder = false
        hasAudioEncoder = false
        history.removeAll()
        loadedModelPath = nil
        loadedModelSizeBytes = 0
        
        if let conversation = self.conversation {
            litert_lm_conversation_delete(conversation)
            self.conversation = nil
        }
        if let engine = self.engine {
            litert_lm_engine_delete(engine)
            self.engine = nil
        }
        
        lastStats = GenerationStats(
            promptTokens: 0.0,
            completionTokens: 0.0,
            totalTokens: 0.0,
            timeToFirstToken: 0.0,
            totalTime: 0.0,
            tokensPerSecond: 0.0
        )
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
                // leaking the raw message JSON into the token stream.
                if let toolCalls = json["tool_calls"] as? [[String: Any]] {
                    for call in toolCalls {
                        let function = (call["function"] as? [String: Any]) ?? call
                        var payload: [String: Any] = [:]
                        payload["name"] = function["name"] ?? ""
                        payload["arguments"] = function["arguments"] ?? [String: Any]()
                        if let payloadData = try? JSONSerialization.data(withJSONObject: payload),
                           let payloadJson = String(data: payloadData, encoding: .utf8) {
                            textResult += "<tool_call>\(payloadJson)</tool_call>"
                        }
                    }
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
