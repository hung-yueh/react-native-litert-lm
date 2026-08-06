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
    private var backend: Backend = .cpu
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxContextTokens: Int = 4096
    private var maxOutputTokens: Int = 1024
    private var systemPrompt: String?
    private var tools: [ToolDefinition]?
    private var enableSpeculativeDecoding: Bool = false
    private var numThreads: Int?
    private var prefillChunkSize: Int?
    private var activationDataType: ActivationDataType?
    private var loraPath: String?
    private var audioLoraPath: String?
    private var loraRank: Int?
    private var streamToolCalls: Bool = false
    private var toolCallChannelName: String = "tool_call"

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
    
    public func resetConversation() throws {
        queue.sync {
            history.removeAll()
            lastStats = GenerationStats(
                promptTokens: 0.0,
                completionTokens: 0.0,
                totalTokens: 0.0,
                timeToFirstToken: 0.0,
                totalTime: 0.0,
                tokensPerSecond: 0.0
            )
            if isLoaded && engine != nil {
                createNewConversation()
            }
        }
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
                self.numThreads = config.numThreads.map { Int($0) }
                self.prefillChunkSize = config.prefillChunkSize.map { Int($0) }
                self.activationDataType = config.activationDataType
                self.loraPath = config.loraPath
                self.audioLoraPath = config.audioLoraPath
                self.loraRank = config.loraRank.map { Int($0) }
                self.streamToolCalls = config.streamToolCalls ?? false
                if let ch = config.toolCallChannelName { self.toolCallChannelName = ch }
            } else {
                self.tools = nil
                self.enableSpeculativeDecoding = false
                self.numThreads = nil
                self.prefillChunkSize = nil
                self.activationDataType = nil
                self.loraPath = nil
                self.audioLoraPath = nil
                self.loraRank = nil
                self.streamToolCalls = false
                self.toolCallChannelName = "tool_call"
            }
            
            // Map main backend string
            let mainBackendStr = self.backend == .gpu ? "gpu" : (self.backend == .npu ? "gpu" : "cpu")
            
            //Sniff multimodal support
            let isMultimodal = config?.multimodal ?? (modelPath.lowercased().contains("3n") || modelPath.lowercased().contains("gemma3"))
            let visionBackend = isMultimodal ? "gpu" : nil
            let audioBackend = isMultimodal ? "cpu" : nil
            
            var rawEngine: OpaquePointer? = nil
            
            // Set LiteRT C Log Level to WARNING for clean production output.
            // v0.15: takes a typed LiteRtLmLogSeverity (previously a raw int,
            // where 2 was INFO — WARNING is the intended level).
            litert_lm_set_min_log_level(kLiteRtLmLogSeverityWarning)
            
            // Creation helper with scoped FFI pointer lifetime
            let createEngine = { (main: String, vision: String?, audio: String?) -> OpaquePointer? in
                let settings = modelPath.withCString { modelC in
                    self.withOptionalCString(main) { mainC in
                        self.withOptionalCString(vision) { visionC in
                            self.withOptionalCString(audio) { audioC in
                                return litert_lm_engine_settings_create(modelC, mainC, visionC, audioC)
                            }
                        }
                    }
                }
                
                guard let s = settings else { return nil }
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
                if let adt = self.activationDataType {
                    // v0.15: takes a typed LiteRtLmActivationDataType (raw values
                    // unchanged: 0=F32, 1=F16, 2=I16, 3=I8)
                    let cType: LiteRtLmActivationDataType
                    switch adt.rawValue {
                    case 1: cType = kLiteRtLmActivationDataTypeFloat16
                    case 2: cType = kLiteRtLmActivationDataTypeInt16
                    case 3: cType = kLiteRtLmActivationDataTypeInt8
                    default: cType = kLiteRtLmActivationDataTypeFloat32
                    }
                    litert_lm_engine_settings_set_activation_data_type(s, cType)
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
                
                return litert_lm_engine_create(s)
            }
            
            // Attempt primary backend configuration
            rawEngine = createEngine(mainBackendStr, visionBackend, audioBackend)
            
            // Fallback sequence if GPU/NPU fails to initialize
            if rawEngine == nil {
                if mainBackendStr != "cpu" {
                    NSLog("[LiteRTLM] %@ backend failed — trying fallback chain...", mainBackendStr.uppercased())
                }
                // Fallback 1: CPU execution with GPU acceleration for heavy Vision parameters
                rawEngine = createEngine("cpu", "gpu", "cpu")
                
                if rawEngine == nil {
                    // Fallback 2: Full CPU execution for all modalities
                    rawEngine = createEngine("cpu", "cpu", "cpu")
                }
                
                if rawEngine == nil {
                    // Fallback 3: Text-only CPU execution (skip vision executor mapping)
                    rawEngine = createEngine("cpu", nil, nil)
                }
                
                if rawEngine != nil {
                    NSLog("[LiteRTLM] %@ backend unavailable — fell back to CPU successfully", mainBackendStr.uppercased())
                    self.backend = .cpu
                }
            }
            
            guard let engine = rawEngine else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "Failed to construct LiteRT-LM engine. Checked backends and fallback chains."]))
                return
            }
            
            self.engine = engine
            self.createNewConversation()
            self.loadedModelPath = modelPath
            self.loadedModelSizeBytes =
                (try? FileManager.default.attributesOfItem(atPath: modelPath)[.size] as? Int).flatMap { $0 } ?? 0
            
            guard self.conversation != nil else {
                self.closeInternal()
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "Failed to create conversation context."]))
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
    
    private func createNewConversation() {
        guard let engine = self.engine else { return }
        
        if let oldConv = self.conversation {
            litert_lm_conversation_delete(oldConv)
            self.conversation = nil
        }
        
        guard let convConfig = litert_lm_conversation_config_create() else { return }
        defer { litert_lm_conversation_config_delete(convConfig) }
        
        guard let sessionConfig = litert_lm_session_config_create() else { return }
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

        litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)

        if self.streamToolCalls {
            self.toolCallChannelName.withCString { channelC in
                litert_lm_conversation_config_set_stream_tool_calls(convConfig, true, channelC)
            }
        }
        
        if let systemPrompt = self.systemPrompt {
            let systemMsgJson = "{\"role\":\"system\",\"content\":\"" + escapeJson(systemPrompt) + "\"}"
            systemMsgJson.withCString { systemMsgC in
                litert_lm_conversation_config_set_system_message(convConfig, systemMsgC)
            }
        }
        
        if let tools = self.tools, !tools.isEmpty {
            var toolsArray: [[String: Any]] = []
            for tool in tools {
                var functionMap: [String: Any] = ["name": tool.name, "description": tool.description]
                if let data = tool.parametersJson.data(using: .utf8),
                   let parsedParams = try? JSONSerialization.jsonObject(with: data, options: []) {
                    functionMap["parameters"] = parsedParams
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
        
        self.conversation = litert_lm_conversation_create(engine, convConfig)
    }
    
    private func closeInternal() {
        isLoaded = false
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
    
    // MARK: - Internal Preprocessing Helpers (for extension files)
    
    private let kControlTokens = [
        "<end_of_turn>",
        "<start_of_turn>model",
        "<start_of_turn>user",
        "<start_of_turn>",
        "<eos>"
    ]
    
    func escapeJson(_ input: String) -> String {
        var output = ""
        for char in input {
            switch char {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case "\u{0008}": output += "\\b"
            case "\u{000c}": output += "\\f"
            default: output.append(char)
            }
        }
        return output
    }
    
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
                if let content = json["content"] {
                    if let contentString = content as? String {
                        return stripControlTokens(contentString)
                    } else if let contentArray = content as? [[String: Any]] {
                        var textResult = ""
                        for part in contentArray {
                            if let type = part["type"] as? String, type == "text", let text = part["text"] as? String {
                                textResult += text
                            }
                        }
                        return stripControlTokens(textResult)
                    }
                }
            }
        } catch {}
        return stripControlTokens(jsonResponse)
    }
    
    private func withOptionalCString<R>(_ string: String?, _ block: (UnsafePointer<CChar>?) -> R) -> R {
        if let string = string {
            return string.withCString { block($0) }
        } else {
            return block(nil)
        }
    }

}
