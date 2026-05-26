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

/// A stream context passed to the low-level C FFI callback to forward chunks safely to the JS thread.
private class StreamContext {
    let userMessage: String
    let startTime: Date
    let onToken: (_ token: String, _ done: Bool) -> Void
    let promise: Promise<Void>
    let parent: HybridLiteRTLM
    
    var rawResponse: String = ""
    var fullResponse: String = ""
    var lastEmittedLength: Int = 0
    var tokenCount: Int = 0
    
    init(
        userMessage: String,
        startTime: Date,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void,
        promise: Promise<Void>,
        parent: HybridLiteRTLM
    ) {
        self.userMessage = userMessage
        self.startTime = startTime
        self.onToken = onToken
        self.promise = promise
        self.parent = parent
    }
}

public class HybridLiteRTLM: HybridLiteRTLMSpec_base, HybridLiteRTLMSpec_protocol {
    
    /// Dedicated background serial queue to protect the JSI/JS thread from blocking and deadlocks (User Rule #1).
    private let queue = DispatchQueue(label: "dev.litert.engine", qos: .userInteractive)
    
    /// Opaque pointer to the LiteRT LM C Engine.
    private var engine: OpaquePointer?
    
    /// Opaque pointer to the active conversation state.
    private var conversation: OpaquePointer?
    
    /// Thread-safe status flag.
    private var isLoaded = false
    
    /// Conversation history.
    private var history: [Message] = []
    
    /// Latest inference generation statistics.
    private var lastStats = GenerationStats(
        promptTokens: 0.0,
        completionTokens: 0.0,
        totalTokens: 0.0,
        timeToFirstToken: 0.0,
        totalTime: 0.0,
        tokensPerSecond: 0.0
    )
    
    // Default configuration variables
    private var backend: Backend = .cpu
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxTokens: Int = 1024
    private var systemPrompt: String?
    private var tools: [ToolDefinition]?
    private var enableSpeculativeDecoding: Bool = false
    
    /// Approximate model weight size to inform the JS engine's garbage collection.
    public var memorySize: Int {
        return 1024 * 1024 * 1024 // ~1GB proxy
    }
    
    deinit {
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
        return try queue.sync {
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
        let availableBytes = Double(os_proc_available_memory())
        
        // Flag memory warning at ~200MB remaining headroom
        let isLowMemory = availableBytes < 200.0 * 1024.0 * 1024.0
        
        return MemoryUsage(
            nativeHeapBytes: nativeHeapBytes,
            residentBytes: residentBytes,
            availableMemoryBytes: availableBytes,
            isLowMemory: isLowMemory
        )
    }
    
    public func close() throws {
        queue.sync {
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
                if let m = config.maxTokens { self.maxTokens = Int(m) }
                if let s = config.systemPrompt { self.systemPrompt = s }
                self.tools = config.tools
                self.enableSpeculativeDecoding = config.enableSpeculativeDecoding ?? false
            } else {
                self.tools = nil
                self.enableSpeculativeDecoding = false
            }
            
            // Map main backend string
            let mainBackendStr = self.backend == .gpu ? "gpu" : (self.backend == .npu ? "gpu" : "cpu")
            
            //Sniff multimodal support
            let isMultimodal = config?.multimodal ?? (modelPath.lowercased().contains("3n") || modelPath.lowercased().contains("gemma3"))
            let visionBackend = isMultimodal ? "gpu" : nil
            let audioBackend = isMultimodal ? "cpu" : nil
            
            var rawEngine: OpaquePointer? = nil
            
            // Set LiteRT C Log Level to WARNING (2) for clean production output
            litert_lm_set_min_log_level(2)
            
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
                
                litert_lm_engine_settings_set_max_num_tokens(s, Int32(self.maxTokens))
                litert_lm_engine_settings_enable_benchmark(s)
                
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
            if rawEngine == nil && mainBackendStr != "cpu" {
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
                    self.backend = .cpu
                }
            }
            
            guard let engine = rawEngine else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "Failed to construct LiteRT-LM engine. Checked backends and fallback chains."]))
                return
            }
            
            self.engine = engine
            self.createNewConversation()
            
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
    
    public func sendMessage(message: String) throws -> Promise<String> {
        let promise = Promise<String>()
        
        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }
            
            let msgJson = self.buildTextMessageJson(text: message)
            let startTime = Date()
            
            // Synchronous FFI call blocks only this interactive queue
            guard let response = litert_lm_conversation_send_message(conversation, msgJson, nil, nil) else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "LiteRT-LM: sendMessage failed"]))
                return
            }
            defer { litert_lm_json_response_delete(response) }
            
            var result = ""
            if let responseStr = litert_lm_json_response_get_string(response) {
                result = self.extractTextFromResponse(String(cString: responseStr))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            let endTime = Date()
            let totalTime = endTime.timeIntervalSince(startTime)
            
            var completionTokens = 0.0
            var tokensPerSecond = 0.0
            var ttft = 0.0
            
            if let benchInfo = litert_lm_conversation_get_benchmark_info(conversation) {
                let numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchInfo)
                if numDecodeTurns > 0 {
                    let lastIdx = numDecodeTurns - 1
                    tokensPerSecond = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(benchInfo, lastIdx)
                    completionTokens = Double(litert_lm_benchmark_info_get_decode_token_count_at(benchInfo, lastIdx))
                }
                ttft = litert_lm_benchmark_info_get_time_to_first_token(benchInfo)
                litert_lm_benchmark_info_delete(benchInfo)
            }
            
            let promptTokens = Double(message.count) / 4.0
            if completionTokens == 0.0 {
                completionTokens = Double(result.count) / 4.0
            }
            
            self.lastStats = GenerationStats(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens,
                timeToFirstToken: ttft,
                totalTime: totalTime,
                tokensPerSecond: tokensPerSecond > 0.0 ? tokensPerSecond : (completionTokens / totalTime)
            )
            
            self.history.append(Message(role: .user, content: message))
            self.history.append(Message(role: .model, content: result))
            
            promise.resolve(withResult: result)
        }
        
        return promise
    }
    
    public func sendMessageAsync(
        message: String,
        onToken: @escaping (_ token: String, _ done: Bool) -> Void
    ) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }
            
            let msgJson = self.buildTextMessageJson(text: message)
            let startTime = Date()
            
            let context = StreamContext(
                userMessage: message,
                startTime: startTime,
                onToken: onToken,
                promise: promise,
                parent: self
            )
            
            let callbackData = Unmanaged.passRetained(context).toOpaque()
            
            let callback: LiteRtLmStreamCallback = { callbackData, chunk, isFinal, errorMsg in
                guard let callbackData = callbackData else { return }
                let ctx = Unmanaged<StreamContext>.fromOpaque(callbackData).takeUnretainedValue()
                
                if let errorMsg = errorMsg {
                    let errorStr = String(cString: errorMsg)
                    ctx.onToken("Error: \(errorStr)", true)
                    ctx.promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: errorStr]))
                    Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                    return
                }
                
                if isFinal {
                    let endTime = Date()
                    let totalTime = endTime.timeIntervalSince(ctx.startTime)
                    
                    let cleaned = ctx.parent.stripControlTokens(ctx.rawResponse)
                    var finalCleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !ctx.userMessage.isEmpty && finalCleaned.hasPrefix(ctx.userMessage) {
                        finalCleaned = String(finalCleaned.dropFirst(ctx.userMessage.count))
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    
                    if finalCleaned.count > ctx.lastEmittedLength {
                        let startIdx = finalCleaned.index(finalCleaned.startIndex, offsetBy: ctx.lastEmittedLength)
                        let remaining = String(finalCleaned[startIdx...])
                        ctx.onToken(remaining, false)
                    }
                    ctx.fullResponse = finalCleaned
                    
                    var completionTokens = Double(ctx.tokenCount)
                    var tokensPerSecond = 0.0
                    var ttft = 0.0
                    
                    if let benchInfo = litert_lm_conversation_get_benchmark_info(ctx.parent.conversation) {
                        let numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchInfo)
                        if numDecodeTurns > 0 {
                            let lastIdx = numDecodeTurns - 1
                            tokensPerSecond = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(benchInfo, lastIdx)
                            completionTokens = Double(litert_lm_benchmark_info_get_decode_token_count_at(benchInfo, lastIdx))
                        }
                        ttft = litert_lm_benchmark_info_get_time_to_first_token(benchInfo)
                        litert_lm_benchmark_info_delete(benchInfo)
                    }
                    
                    let promptTokens = Double(ctx.userMessage.count) / 4.0
                    if completionTokens == 0.0 {
                        completionTokens = Double(ctx.fullResponse.count) / 4.0
                    }
                    
                    ctx.parent.lastStats = GenerationStats(
                        promptTokens: promptTokens,
                        completionTokens: completionTokens,
                        totalTokens: promptTokens + completionTokens,
                        timeToFirstToken: ttft,
                        totalTime: totalTime,
                        tokensPerSecond: tokensPerSecond > 0.0 ? tokensPerSecond : (completionTokens / totalTime)
                    )
                    
                    ctx.parent.history.append(Message(role: .user, content: ctx.userMessage))
                    ctx.parent.history.append(Message(role: .model, content: ctx.fullResponse))
                    
                    ctx.onToken("", true)
                    ctx.promise.resolve()
                    Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                    return
                }
                
                if let chunk = chunk {
                    let token = String(cString: chunk)
                    let raw: String
                    if token.hasPrefix("{") && token.contains("\"role\"") {
                        raw = ctx.parent.extractTextFromResponse(token)
                    } else {
                        raw = token
                    }
                    
                    ctx.rawResponse += raw
                    let cleaned = ctx.parent.stripControlTokens(ctx.rawResponse)
                        .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
                    
                    var processed = cleaned
                    if !ctx.userMessage.isEmpty && processed.hasPrefix(ctx.userMessage) {
                        processed = String(processed.dropFirst(ctx.userMessage.count))
                            .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
                    }
                    
                    let safeLen = ctx.parent.safeEmitLength(processed)
                    if safeLen > ctx.lastEmittedLength {
                        let chars = Array(processed)
                        let newText = String(chars[ctx.lastEmittedLength..<safeLen])
                        ctx.lastEmittedLength = safeLen
                        ctx.tokenCount += 1
                        ctx.onToken(newText, false)
                    }
                }
            }
            
            let status = litert_lm_conversation_send_message_stream(
                conversation,
                msgJson,
                nil,
                nil,
                callback,
                callbackData
            )
            
            if status != 0 {
                Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                promise.reject(withError: NSError(domain: "LiteRTLM", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "Failed to start streaming conversation."]))
            }
        }
        
        return promise
    }
    
    public func sendMessageWithImage(message: String, imagePath: String) throws -> Promise<String> {
        let promise = Promise<String>()
        
        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }
            
            if !FileManager.default.fileExists(atPath: imagePath) {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 404, userInfo: [NSLocalizedDescriptionKey: "Image file not found: \(imagePath)"]))
                return
            }
            
            let msgJson = self.buildImageMessageJson(text: message, imagePath: imagePath)
            let startTime = Date()
            
            guard let response = litert_lm_conversation_send_message(conversation, msgJson, nil, nil) else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "LiteRT-LM: sendMessageWithImage failed"]))
                return
            }
            defer { litert_lm_json_response_delete(response) }
            
            var result = ""
            if let responseStr = litert_lm_json_response_get_string(response) {
                result = self.extractTextFromResponse(String(cString: responseStr))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            let endTime = Date()
            let totalTime = endTime.timeIntervalSince(startTime)
            
            self.lastStats = GenerationStats(
                promptTokens: Double(message.count) / 4.0,
                completionTokens: Double(result.count) / 4.0,
                totalTokens: Double(message.count + result.count) / 4.0,
                timeToFirstToken: 0.0,
                totalTime: totalTime,
                tokensPerSecond: Double(result.count) / 4.0 / totalTime
            )
            
            self.history.append(Message(role: .user, content: message + " [image: \(imagePath)]"))
            self.history.append(Message(role: .model, content: result))
            
            promise.resolve(withResult: result)
        }
        
        return promise
    }

    public func sendMessageWithImageAsync(message: String, imagePath: String, onToken: @escaping (_ token: String, _ done: Bool) -> Void) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }
            
            if !FileManager.default.fileExists(atPath: imagePath) {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 404, userInfo: [NSLocalizedDescriptionKey: "Image file not found: \(imagePath)"]))
                return
            }
            
            let msgJson = self.buildImageMessageJson(text: message, imagePath: imagePath)
            let startTime = Date()
            
            let historyUserContent = message + " [image: \(imagePath)]"
            let context = StreamContext(
                userMessage: message,
                startTime: startTime,
                onToken: onToken,
                promise: promise,
                parent: self
            )
            
            let callbackData = Unmanaged.passRetained(context).toOpaque()
            
            let callback: LiteRtLmStreamCallback = { callbackData, chunk, isFinal, errorMsg in
                guard let callbackData = callbackData else { return }
                let ctx = Unmanaged<StreamContext>.fromOpaque(callbackData).takeUnretainedValue()
                
                if let errorMsg = errorMsg {
                    let errorStr = String(cString: errorMsg)
                    ctx.onToken("Error: \(errorStr)", true)
                    ctx.promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: errorStr]))
                    Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                    return
                }
                
                if isFinal {
                    let endTime = Date()
                    let totalTime = endTime.timeIntervalSince(ctx.startTime)
                    
                    let cleaned = ctx.parent.stripControlTokens(ctx.rawResponse)
                    var finalCleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !ctx.userMessage.isEmpty && finalCleaned.hasPrefix(ctx.userMessage) {
                        finalCleaned = String(finalCleaned.dropFirst(ctx.userMessage.count))
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    
                    if finalCleaned.count > ctx.lastEmittedLength {
                        let startIdx = finalCleaned.index(finalCleaned.startIndex, offsetBy: ctx.lastEmittedLength)
                        let remaining = String(finalCleaned[startIdx...])
                        ctx.onToken(remaining, false)
                    }
                    ctx.fullResponse = finalCleaned
                    
                    var completionTokens = Double(ctx.tokenCount)
                    var tokensPerSecond = 0.0
                    var ttft = 0.0
                    if let benchInfo = litert_lm_conversation_get_benchmark_info(ctx.parent.conversation) {
                        let numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchInfo)
                        if numDecodeTurns > 0 {
                            let lastIdx = numDecodeTurns - 1
                            tokensPerSecond = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(benchInfo, lastIdx)
                            completionTokens = Double(litert_lm_benchmark_info_get_decode_token_count_at(benchInfo, lastIdx))
                        }
                        ttft = litert_lm_benchmark_info_get_time_to_first_token(benchInfo)
                        litert_lm_benchmark_info_delete(benchInfo)
                    }

                    let promptTokens = Double(ctx.userMessage.count) / 4.0
                    if completionTokens == 0.0 {
                        completionTokens = Double(ctx.fullResponse.count) / 4.0
                    }
                    ctx.parent.lastStats = GenerationStats(
                        promptTokens: promptTokens,
                        completionTokens: completionTokens,
                        totalTokens: promptTokens + completionTokens,
                        timeToFirstToken: ttft,
                        totalTime: totalTime,
                        tokensPerSecond: tokensPerSecond > 0.0 ? tokensPerSecond : (completionTokens / totalTime)
                    )
                    ctx.parent.history.append(Message(role: .user, content: historyUserContent))
                    ctx.parent.history.append(Message(role: .model, content: ctx.fullResponse))
                    ctx.onToken("", true)
                    ctx.promise.resolve()
                    Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                    return
                }

                if let chunk = chunk {
                    let token = String(cString: chunk)
                    let raw: String
                    if token.hasPrefix("{") && token.contains("\"role\"") {
                        raw = ctx.parent.extractTextFromResponse(token)
                    } else {
                        raw = token
                    }

                    ctx.rawResponse += raw
                    let cleaned = ctx.parent.stripControlTokens(ctx.rawResponse)
                        .trimmingLeadingCharacters(in: .whitespacesAndNewlines)

                    var processed = cleaned
                    if !ctx.userMessage.isEmpty && processed.hasPrefix(ctx.userMessage) {
                        processed = String(processed.dropFirst(ctx.userMessage.count))
                            .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
                    }

                    let safeLen = ctx.parent.safeEmitLength(processed)
                    if safeLen > ctx.lastEmittedLength {
                        let chars = Array(processed)
                        let newText = String(chars[ctx.lastEmittedLength..<safeLen])
                        ctx.lastEmittedLength = safeLen
                        ctx.tokenCount += 1
                        ctx.onToken(newText, false)
                    }
                }
            }

            let status = litert_lm_conversation_send_message_stream(
                conversation,
                msgJson,
                nil,
                nil,
                callback,
                callbackData
            )
            if status != 0 {
                Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                promise.reject(withError: NSError(domain: "LiteRTLM", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "Failed to start streaming conversation."]))
            }
        }

        return promise
    }
    
    public func sendMessageWithAudioAsync(message: String, audioPath: String, onToken: @escaping (_ token: String, _ done: Bool) -> Void) throws -> Promise<Void> {
        let promise = Promise<Void>()

        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }

            if !FileManager.default.fileExists(atPath: audioPath) {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 404, userInfo: [NSLocalizedDescriptionKey: "Audio file not found: \(audioPath)"]))
                return
            }

            let msgJson = self.buildAudioMessageJson(text: message, audioPath: audioPath)
            let startTime = Date()

            let historyUserContent = message + " [audio: \(audioPath)]"
            let context = StreamContext(
                userMessage: message,
                startTime: startTime,
                onToken: onToken,
                promise: promise,
                parent: self
            )

            let callbackData = Unmanaged.passRetained(context).toOpaque()

            let callback: LiteRtLmStreamCallback = { callbackData, chunk, isFinal, errorMsg in
                guard let callbackData = callbackData else { return }
                let ctx = Unmanaged<StreamContext>.fromOpaque(callbackData).takeUnretainedValue()

                if let errorMsg = errorMsg {
                    let errorStr = String(cString: errorMsg)
                    ctx.onToken("Error: \(errorStr)", true)
                    ctx.promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: errorStr]))
                    Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                    return
                }

                if isFinal {
                    let endTime = Date()
                    let totalTime = endTime.timeIntervalSince(ctx.startTime)

                    let cleaned = ctx.parent.stripControlTokens(ctx.rawResponse)
                    var finalCleaned = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !ctx.userMessage.isEmpty && finalCleaned.hasPrefix(ctx.userMessage) {
                        finalCleaned = String(finalCleaned.dropFirst(ctx.userMessage.count))
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                    }

                    if finalCleaned.count > ctx.lastEmittedLength {
                        let startIdx = finalCleaned.index(finalCleaned.startIndex, offsetBy: ctx.lastEmittedLength)
                        let remaining = String(finalCleaned[startIdx...])
                        ctx.onToken(remaining, false)
                    }
                    ctx.fullResponse = finalCleaned

                    var completionTokens = Double(ctx.tokenCount)
                    var tokensPerSecond = 0.0
                    var ttft = 0.0
                    if let benchInfo = litert_lm_conversation_get_benchmark_info(ctx.parent.conversation) {
                        let numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchInfo)
                        if numDecodeTurns > 0 {
                            let lastIdx = numDecodeTurns - 1
                            tokensPerSecond = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(benchInfo, lastIdx)
                            completionTokens = Double(litert_lm_benchmark_info_get_decode_token_count_at(benchInfo, lastIdx))
                        }
                        ttft = litert_lm_benchmark_info_get_time_to_first_token(benchInfo)
                        litert_lm_benchmark_info_delete(benchInfo)
                    }

                    let promptTokens = Double(ctx.userMessage.count) / 4.0
                    if completionTokens == 0.0 {
                        completionTokens = Double(ctx.fullResponse.count) / 4.0
                    }
                    ctx.parent.lastStats = GenerationStats(
                        promptTokens: promptTokens,
                        completionTokens: completionTokens,
                        totalTokens: promptTokens + completionTokens,
                        timeToFirstToken: ttft,
                        totalTime: totalTime,
                        tokensPerSecond: tokensPerSecond > 0.0 ? tokensPerSecond : (completionTokens / totalTime)
                    )
                    ctx.parent.history.append(Message(role: .user, content: historyUserContent))
                    ctx.parent.history.append(Message(role: .model, content: ctx.fullResponse))
                    ctx.onToken("", true)
                    ctx.promise.resolve()
                    Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                    return
                }

                if let chunk = chunk {
                    let token = String(cString: chunk)
                    let raw: String
                    if token.hasPrefix("{") && token.contains("\"role\"") {
                        raw = ctx.parent.extractTextFromResponse(token)
                    } else {
                        raw = token
                    }

                    ctx.rawResponse += raw
                    let cleaned = ctx.parent.stripControlTokens(ctx.rawResponse)
                        .trimmingLeadingCharacters(in: .whitespacesAndNewlines)

                    var processed = cleaned
                    if !ctx.userMessage.isEmpty && processed.hasPrefix(ctx.userMessage) {
                        processed = String(processed.dropFirst(ctx.userMessage.count))
                            .trimmingLeadingCharacters(in: .whitespacesAndNewlines)
                    }

                    let safeLen = ctx.parent.safeEmitLength(processed)
                    if safeLen > ctx.lastEmittedLength {
                        let chars = Array(processed)
                        let newText = String(chars[ctx.lastEmittedLength..<safeLen])
                        ctx.lastEmittedLength = safeLen
                        ctx.tokenCount += 1
                        ctx.onToken(newText, false)
                    }
                }
            }

            let status = litert_lm_conversation_send_message_stream(
                conversation,
                msgJson,
                nil,
                nil,
                callback,
                callbackData
            )
            if status != 0 {
                Unmanaged<StreamContext>.fromOpaque(callbackData).release()
                promise.reject(withError: NSError(domain: "LiteRTLM", code: Int(status), userInfo: [NSLocalizedDescriptionKey: "Failed to start streaming conversation."]))
            }
        }

        return promise
    }

    public func sendMessageWithAudio(message: String, audioPath: String) throws -> Promise<String> {
        let promise = Promise<String>()
        
        queue.async {
            guard let conversation = self.conversation else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }
            
            if !FileManager.default.fileExists(atPath: audioPath) {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 404, userInfo: [NSLocalizedDescriptionKey: "Audio file not found: \(audioPath)"]))
                return
            }
            
            let msgJson = self.buildAudioMessageJson(text: message, audioPath: audioPath)
            let startTime = Date()
            
            guard let response = litert_lm_conversation_send_message(conversation, msgJson, nil, nil) else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "LiteRT-LM: sendMessageWithAudio failed"]))
                return
            }
            defer { litert_lm_json_response_delete(response) }
            
            var result = ""
            if let responseStr = litert_lm_json_response_get_string(response) {
                result = self.extractTextFromResponse(String(cString: responseStr))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            
            let endTime = Date()
            let totalTime = endTime.timeIntervalSince(startTime)
            
            self.lastStats = GenerationStats(
                promptTokens: Double(message.count) / 4.0,
                completionTokens: Double(result.count) / 4.0,
                totalTokens: Double(message.count + result.count) / 4.0,
                timeToFirstToken: 0.0,
                totalTime: totalTime,
                tokensPerSecond: Double(result.count) / 4.0 / totalTime
            )
            
            self.history.append(Message(role: .user, content: message + " [audio: \(audioPath)]"))
            self.history.append(Message(role: .model, content: result))
            
            promise.resolve(withResult: result)
        }
        
        return promise
    }
    
    public func sendMultimodalMessage(parts: [MultimodalPart]) throws -> Promise<String> {
        let promise = Promise<String>()
        
        queue.async {
            guard let engine = self.engine else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: No model loaded. Call loadModel() first."]))
                return
            }
            
            // Create session config
            guard let sessionConfig = litert_lm_session_config_create() else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: Failed to create session config."]))
                return
            }
            defer { litert_lm_session_config_delete(sessionConfig) }
            
            litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(self.maxTokens))
            
            var sampler = LiteRtLmSamplerParams()
            sampler.type = kLiteRtLmSamplerTypeTopP
            sampler.top_k = Int32(self.topK)
            sampler.top_p = Float(self.topP)
            sampler.temperature = Float(self.temperature)
            sampler.seed = 0
            withUnsafePointer(to: &sampler) { samplerPtr in
                litert_lm_session_config_set_sampler_params(sessionConfig, samplerPtr)
            }
            
            guard let session = litert_lm_engine_create_session(engine, sessionConfig) else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: Failed to create session."]))
                return
            }
            defer { litert_lm_session_delete(session) }
            
            // Construct inputs array
            var inputs: [LiteRtLmInputData] = []
            var allocatedStrings: [UnsafeMutablePointer<CChar>] = []
            
            defer {
                for ptr in allocatedStrings {
                    free(ptr)
                }
            }
            
            for part in parts {
                switch part.type {
                case .text:
                    if let text = part.text {
                        let cStr = strdup(text)!
                        allocatedStrings.append(cStr)
                        inputs.append(LiteRtLmInputData(type: kLiteRtLmInputDataTypeText, data: cStr, size: text.utf8.count))
                    }
                case .image:
                    if let imageBuffer = part.imageBuffer {
                        inputs.append(LiteRtLmInputData(type: kLiteRtLmInputDataTypeImage, data: imageBuffer.data, size: imageBuffer.size))
                    }
                case .audio:
                    if let audioBuffer = part.audioBuffer {
                        inputs.append(LiteRtLmInputData(type: kLiteRtLmInputDataTypeAudio, data: audioBuffer.data, size: audioBuffer.size))
                    }
                }
            }
            
            let startTime = Date()
            
            // Run session inference
            guard let responses = litert_lm_session_generate_content(session, inputs, inputs.count) else {
                promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "LiteRTLM: Session generate content failed."]))
                return
            }
            defer { litert_lm_responses_delete(responses) }
            
            var result = ""
            let numCandidates = litert_lm_responses_get_num_candidates(responses)
            if numCandidates > 0 {
                if let responseStr = litert_lm_responses_get_response_text_at(responses, 0) {
                    result = String(cString: responseStr).trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
            
            let endTime = Date()
            let totalTime = endTime.timeIntervalSince(startTime)
            
            // Update last stats using benchmark info from session
            var completionTokens = 0.0
            var tokensPerSecond = 0.0
            var ttft = 0.0
            
            if let benchInfo = litert_lm_session_get_benchmark_info(session) {
                let numDecodeTurns = litert_lm_benchmark_info_get_num_decode_turns(benchInfo)
                if numDecodeTurns > 0 {
                    let lastIdx = numDecodeTurns - 1
                    tokensPerSecond = litert_lm_benchmark_info_get_decode_tokens_per_sec_at(benchInfo, lastIdx)
                    completionTokens = Double(litert_lm_benchmark_info_get_decode_token_count_at(benchInfo, lastIdx))
                }
                ttft = litert_lm_benchmark_info_get_time_to_first_token(benchInfo)
                litert_lm_benchmark_info_delete(benchInfo)
            }
            
            let totalInputLen = parts.reduce(0) { $0 + ($1.text?.count ?? 0) }
            let promptTokens = Double(totalInputLen) / 4.0
            if completionTokens == 0.0 {
                completionTokens = Double(result.count) / 4.0
            }
            
            self.lastStats = GenerationStats(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens,
                timeToFirstToken: ttft,
                totalTime: totalTime,
                tokensPerSecond: tokensPerSecond > 0.0 ? tokensPerSecond : (completionTokens / totalTime)
            )
            
            // Append to history
            var userTextRepresentation = ""
            for part in parts {
                if part.type == .text, let text = part.text {
                    userTextRepresentation += text + " "
                } else if part.type == .image {
                    userTextRepresentation += "[Image Buffer] "
                } else if part.type == .audio {
                    userTextRepresentation += "[Audio Buffer] "
                }
            }
            userTextRepresentation = userTextRepresentation.trimmingCharacters(in: .whitespacesAndNewlines)
            
            self.history.append(Message(role: .user, content: userTextRepresentation))
            self.history.append(Message(role: .model, content: result))
            
            promise.resolve(withResult: result)
        }
        
        return promise
    }
    
    public func downloadModel(
        url: String,
        fileName: String,
        onProgress: ((Double) -> Void)?
    ) throws -> Promise<String> {
        let promise = Promise<String>()
        
        queue.async {
            do {
                if fileName.contains("..") || fileName.contains("/") || fileName.contains("\\") {
                    promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid filename: path traversal or directory separators are not allowed."]))
                    return
                }
                
                let cachesDir = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true).first ?? NSTemporaryDirectory()
                let modelsDir = (cachesDir as NSString).appendingPathComponent("litert_models")
                
                let fileManager = FileManager.default
                if !fileManager.fileExists(atPath: modelsDir) {
                    try fileManager.createDirectory(atPath: modelsDir, withIntermediateDirectories: true, attributes: nil)
                }
                
                let destPath = (modelsDir as NSString).appendingPathComponent(fileName)
                
                // Fast cache check
                if fileManager.fileExists(atPath: destPath) {
                    let attrs = try fileManager.attributesOfItem(atPath: destPath)
                    if let fileSize = attrs[.size] as? UInt64, fileSize > 0 {
                        onProgress?(1.0)
                        promise.resolve(withResult: destPath)
                        return
                    }
                }
                
                guard let downloadUrl = URL(string: url), downloadUrl.scheme?.lowercased() == "https" else {
                    promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid download URL: HTTPS is required for security."]))
                    return
                }
                
                onProgress?(0.0)
                
                let sessionConfig = URLSessionConfiguration.default
                sessionConfig.timeoutIntervalForRequest = 30
                sessionConfig.timeoutIntervalForResource = 3600
                
                let session = URLSession(configuration: sessionConfig)
                var progressHandler: NSKeyValueObservation?
                
                let task = session.downloadTask(with: downloadUrl) { location, response, error in
                    progressHandler?.invalidate()
                    
                    if let error = error {
                        promise.reject(withError: error)
                        return
                    }
                    
                    if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode >= 400 {
                        promise.reject(withError: NSError(domain: "LiteRTLM", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: "HTTP \(httpResponse.statusCode)"]))
                        return
                    }
                    
                    guard let location = location else {
                        promise.reject(withError: NSError(domain: "LiteRTLM", code: 500, userInfo: [NSLocalizedDescriptionKey: "No download location found."]))
                        return
                    }
                    
                    do {
                        if fileManager.fileExists(atPath: destPath) {
                            try fileManager.removeItem(atPath: destPath)
                        }
                        try fileManager.moveItem(at: location, to: URL(fileURLWithPath: destPath))
                        onProgress?(1.0)
                        promise.resolve(withResult: destPath)
                    } catch {
                        promise.reject(withError: error)
                    }
                }
                
                if let onProgress = onProgress {
                    var lastUpdate = Date()
                    progressHandler = task.observe(\.countOfBytesReceived, options: [.new]) { task, _ in
                        let expected = task.countOfBytesExpectedToReceive
                        if expected > 0 {
                            let now = Date()
                            // Throttled progress notifications to 10Hz
                            if now.timeIntervalSince(lastUpdate) > 0.1 {
                                let progress = Double(task.countOfBytesReceived) / Double(expected)
                                onProgress(progress)
                                lastUpdate = now
                            }
                        }
                    }
                }
                
                task.resume()
                session.finishTasksAndInvalidate()
            } catch {
                promise.reject(withError: error)
            }
        }
        
        return promise
    }
    
    public func deleteModel(fileName: String) throws -> Promise<Void> {
        let promise = Promise<Void>()
        
        queue.async {
            do {
                if fileName.contains("..") || fileName.contains("/") || fileName.contains("\\") {
                    promise.reject(withError: NSError(domain: "LiteRTLM", code: 400, userInfo: [NSLocalizedDescriptionKey: "Invalid filename: path traversal or directory separators are not allowed."]))
                    return
                }
                
                let cachesDir = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true).first ?? NSTemporaryDirectory()
                let modelsDir = (cachesDir as NSString).appendingPathComponent("litert_models")
                let destPath = (modelsDir as NSString).appendingPathComponent(fileName)
                
                let fileManager = FileManager.default
                if fileManager.fileExists(atPath: destPath) {
                    try fileManager.removeItem(atPath: destPath)
                    if self.isLoaded {
                        self.closeInternal()
                    }
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
        
        litert_lm_session_config_set_max_output_tokens(sessionConfig, Int32(self.maxTokens))
        
        var sampler = LiteRtLmSamplerParams()
        sampler.type = kLiteRtLmSamplerTypeTopP
        sampler.top_k = Int32(self.topK)
        sampler.top_p = Float(self.topP)
        sampler.temperature = Float(self.temperature)
        sampler.seed = 0
        withUnsafePointer(to: &sampler) { samplerPtr in
            litert_lm_session_config_set_sampler_params(sessionConfig, samplerPtr)
        }
        
        litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)
        
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
    
    // MARK: - String and JSON Preprocessing Helpers
    
    private let kControlTokens = [
        "<end_of_turn>",
        "<start_of_turn>model",
        "<start_of_turn>user",
        "<start_of_turn>",
        "<eos>"
    ]
    
    private func escapeJson(_ input: String) -> String {
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
    
    private func buildTextMessageJson(text: String) -> String {
        return "{\"role\":\"user\",\"content\":\"" + escapeJson(text) + "\"}"
    }
    
    private func buildImageMessageJson(text: String, imagePath: String) -> String {
        return "{\"role\":\"user\",\"content\":[" +
               "{\"type\":\"text\",\"text\":\"" + escapeJson(text) + "\"}," +
               "{\"type\":\"image\",\"path\":\"" + escapeJson(imagePath) + "\"}" +
               "]}"
    }
    
    private func buildAudioMessageJson(text: String, audioPath: String) -> String {
        return "{\"role\":\"user\",\"content\":[" +
               "{\"type\":\"text\",\"text\":\"" + escapeJson(text) + "\"}," +
               "{\"type\":\"audio\",\"path\":\"" + escapeJson(audioPath) + "\"}" +
               "]}"
    }
    
    private func stripControlTokens(_ text: String) -> String {
        var result = text
        for tok in kControlTokens {
            result = result.replacingOccurrences(of: tok, with: "")
        }
        return result
    }
    
    private func safeEmitLength(_ text: String) -> Int {
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
    
    private func extractTextFromResponse(_ jsonResponse: String) -> String {
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

// MARK: - String Trimming Extension

private extension String {
    func trimmingLeadingCharacters(in characterSet: CharacterSet) -> String {
        guard let index = firstIndex(where: { char in
            !char.unicodeScalars.allSatisfy { characterSet.contains($0) }
        }) else {
            return ""
        }
        return String(self[index...])
    }
}
