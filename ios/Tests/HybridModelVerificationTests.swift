import XCTest
import NitroModules
import CLiteRTLM
@testable import LiteRTLM

/// The on-device battery a non-Gemma model passes before it gets a `Models`
/// constant: facts, multi-turn memory, streaming, a system prompt and
/// structured output must work; tool calls and thinking are reported.
///
/// Skipped unless a bundle is given:
///
///   xcodebuild test … -only-testing:react-native-litert-lm-Unit-Tests/HybridModelVerificationTests \
///     TEST_RUNNER_LITERTLM_VERIFY_MODEL=$HOME/.litert-models/community/Qwen3.5-2B_int8.litertlm
final class HybridModelVerificationTests: XCTestCase {

    static var path: String? {
        guard let path = ProcessInfo.processInfo.environment["LITERTLM_VERIFY_MODEL"],
              FileManager.default.fileExists(atPath: path) else { return nil }
        return path
    }

    /// Optional Jinja template file to load with (`TEST_RUNNER_LITERTLM_VERIFY_TEMPLATE`).
    static var template: String? {
        ProcessInfo.processInfo.environment["LITERTLM_VERIFY_TEMPLATE"]
            .flatMap { try? String(contentsOfFile: $0, encoding: .utf8) }
    }

    /// Optional sampler `temperature,topK,topP` (`TEST_RUNNER_LITERTLM_VERIFY_SAMPLER`); default greedy.
    static var sampler: (Double, Double, Double) {
        let parts = (ProcessInfo.processInfo.environment["LITERTLM_VERIFY_SAMPLER"] ?? "")
            .split(separator: ",").compactMap { Double($0) }
        return parts.count == 3 ? (parts[0], parts[1], parts[2]) : (0.0, 1.0, 1.0)
    }

    private func text(_ s: String) -> MultimodalPart { .textPart(s) }

    private func config(tools: [ToolDefinition]? = nil) -> LLMConfig {
        LLMConfig(
            systemPrompt: nil, backend: .cpu, maxContextTokens: 2048, maxOutputTokens: 256, maxTokens: nil,
            temperature: Self.sampler.0, topK: Self.sampler.1, topP: Self.sampler.2, multimodal: false, tools: tools,
            enableSpeculativeDecoding: nil, chatTemplate: Self.template, enableStructuredOutput: true, thinking: nil,
            numThreads: nil, maxVisionTokensPerImage: nil, gpu: nil, singleThreadedExecution: nil,
            prefillChunkSize: nil, activationDataType: nil, loraPath: nil, audioLoraPath: nil, loraRank: nil,
            streamToolCalls: nil, toolCallChannelName: nil, forceLoad: true, kvCacheBytesPerToken: nil
        )
    }

    private func options(maxOutputTokens: Double? = nil, responseSchema: String? = nil) -> ExecuteOptions {
        ExecuteOptions(maxOutputTokens: maxOutputTokens, visualTokenBudget: nil, responseSchema: responseSchema,
                       responseRegex: nil, repetitionPenalty: nil, presencePenalty: nil, frequencyPenalty: nil,
                       penaltyWindowSize: nil, noRepeatNgramSize: nil, noRepeatNgramWindowSize: nil,
                       suppressTokens: nil, thinking: nil)
    }

    private func ask(_ bridge: HybridLiteRTLM, _ prompt: String, _ opts: ExecuteOptions? = nil) async throws -> String {
        try await bridge.execute(parts: [text(prompt)], onToken: nil, options: opts ?? options()).await()
    }

    /// The answer with any reasoning block removed, for content checks.
    private func answer(_ reply: String) -> String {
        guard let end = reply.range(of: "</think>") else { return reply }
        return String(reply[end.upperBound...])
    }

    /// Diagnostic: the prompts that failed after resetConversation(), each as
    /// the first turn of a freshly loaded engine, then again after a reset.
    func testFreshEngineProbe() async throws {
        guard let path = Self.path, ProcessInfo.processInfo.environment["LITERTLM_VERIFY_PROBE"] == "fresh" else {
            throw XCTSkip("Set TEST_RUNNER_LITERTLM_VERIFY_PROBE=fresh with TEST_RUNNER_LITERTLM_VERIFY_MODEL")
        }
        let math = "I have 3 apples, buy 5 more, then eat 2. How many apples do I have? End with the number."
        for (label, system, prompt) in [("math", nil, math),
                                        ("pirate", "You are a pirate. Begin every reply with the word Arr.", "Say hello to me.")] {
            let cfg = LLMConfig(
                systemPrompt: system, backend: .cpu, maxContextTokens: 2048, maxOutputTokens: 128, maxTokens: nil,
                temperature: 0.0, topK: 1, topP: 1.0, multimodal: false, tools: nil,
                enableSpeculativeDecoding: nil, chatTemplate: nil, enableStructuredOutput: nil, thinking: nil,
                numThreads: nil, maxVisionTokensPerImage: nil, gpu: nil, singleThreadedExecution: nil,
                prefillChunkSize: nil, activationDataType: nil, loraPath: nil, audioLoraPath: nil, loraRank: nil,
                streamToolCalls: nil, toolCallChannelName: nil, forceLoad: true, kvCacheBytesPerToken: nil)
            let bridge = HybridLiteRTLM()
            _ = try await bridge.loadModel(modelPath: path, config: cfg).await()
            let first = try await ask(bridge, prompt, options(maxOutputTokens: 96))
            _ = try await ask(bridge, "Thanks.", options(maxOutputTokens: 16))
            try bridge.resetConversation(historyJson: nil, systemPrompt: system)
            let afterReset = try await ask(bridge, prompt, options(maxOutputTokens: 96))
            print("[probe] \(label) fresh engine: \(first.prefix(160))")
            print("[probe] \(label) after reset:  \(afterReset.prefix(160))")
            try bridge.close()
        }
    }

    func testVerifyModel() async throws {
        guard let path = Self.path else {
            throw XCTSkip("Set TEST_RUNNER_LITERTLM_VERIFY_MODEL to a .litertlm bundle")
        }
        let name = (path as NSString).lastPathComponent + (Self.template != nil ? " (custom chat template)" : "") +
            " sampler=\(Self.sampler)"
        let info = try await HybridModelInspector().inspect(modelPath: path).await()
        print("[verify] \(name): type=\(info.modelType.stringValue) ctx=\(Int(info.maxContextTokens)) " +
              "dynamic=\(info.isDynamicContext) thinking=\(info.supportsThinking) tools=\(info.supportsFunctionCalling) " +
              "mtp=\(info.hasSpeculativeDecoding) text-backends=\(info.modalities.first { $0.modality == .text }?.backends.map(\.stringValue) ?? []) " +
              "sampler=\(Int(info.samplerType)) t=\(info.samplerTemperature) k=\(Int(info.samplerTopK)) p=\(info.samplerTopP)")

        let bridge = HybridLiteRTLM()
        defer { try? bridge.close() }
        var started = Date()
        _ = try await bridge.loadModel(modelPath: path, config: config()).await()
        print(String(format: "[verify] loaded in %.1fs", Date().timeIntervalSince(started)))

        // Facts
        let capital = try await ask(bridge, "What is the capital of France? Answer with just the city name.")
        print("[verify] capital: \(capital)")
        XCTAssertTrue(answer(capital).lowercased().contains("paris"), "facts: \(capital)")

        // Multi-turn memory, from a fresh conversation (a one-word answer just
        // before primes small models to keep answering in one word)
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)
        let tokensBefore = try bridge.getContextTokenCount()
        let ack = try await ask(bridge, "My favorite color is teal. Please remember it.", options(maxOutputTokens: 48))
        let tokensAfter = try bridge.getContextTokenCount()
        let color = try await ask(bridge, "What is my favorite color? Answer with one word.")
        print("[verify] context tokens \(Int(tokensBefore)) → \(Int(tokensAfter)) → \(Int(try bridge.getContextTokenCount())); " +
              "ack: \(ack.prefix(120))")
        print("[verify] transcript: \(try bridge.getHistory().map { "\($0.role.stringValue): \($0.content.prefix(60))" })")
        print("[verify] recall: \(color)")
        XCTAssertTrue(answer(color).lowercased().contains("teal"), "multi-turn: \(color)")

        // Streaming
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)
        var chunks = 0
        var done = false
        started = Date()
        let story = try await bridge.execute(
            parts: [text("Write two sentences about the ocean.")],
            onToken: { _, isDone in if isDone { done = true } else { chunks += 1 } },
            options: options(maxOutputTokens: 96)).await()
        let stats = try bridge.getStats()
        print(String(format: "[verify] streamed %d chunks in %.1fs, ttft %.0f ms, decode %.2f tok/s: %@",
                     chunks, Date().timeIntervalSince(started), stats.timeToFirstToken,
                     stats.tokensPerSecond, String(story.prefix(160))))
        XCTAssertTrue(done && chunks > 1, "streaming")

        // System prompt
        try bridge.resetConversation(historyJson: nil, systemPrompt: "You are a pirate. Begin every reply with the word Arr.")
        let pirate = try await ask(bridge, "Say hello to me.", options(maxOutputTokens: 48))
        print("[verify] system prompt: \(pirate)")
        XCTAssertTrue(answer(pirate).lowercased().contains("arr"), "system prompt: \(pirate)")

        // Structured output
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)
        let schema = #"{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"]}"#
        let json = try await ask(bridge, "Name one city in Japan and its country.",
                                 options(maxOutputTokens: 64, responseSchema: schema))
        print("[verify] structured: \(json)")
        let parsed = try? JSONSerialization.jsonObject(with: Data(answer(json).utf8)) as? [String: Any]
        XCTAssertNotNil(parsed?["city"] as? String, "structured output: \(json)")

        // Reasoning (reported: whether a <think> block appears)
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)
        let math = try await ask(bridge, "I have 3 apples, buy 5 more, then eat 2. How many apples do I have? End with the number.",
                                 options(maxOutputTokens: 256))
        print("[verify] reasoning (think block: \(math.contains("<think>"))): \(math.prefix(300))")
        XCTAssertTrue(answer(math).contains("6"), "arithmetic: \(math)")
        try bridge.close()

        // Tools (reported, not gated)
        let weather = ToolDefinition(
            name: "get_current_weather", description: "Get the current weather for a location.",
            parametersJson: #"{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}"#)
        let agent = HybridLiteRTLM()
        defer { try? agent.close() }
        _ = try await agent.loadModel(modelPath: path, config: config(tools: [weather])).await()
        let call = try await ask(agent, "What is the weather in Paris right now? Use the tool.")
        print("[verify] tool call: \(call)")
        if call.contains("<tool_call>") {
            let result = MultimodalPart(
                type: .toolresult, text: nil, path: nil, imageBuffer: nil, audioBuffer: nil,
                name: "get_current_weather",
                responseJson: #"{"location":"Paris","temperature_c":27,"condition":"hailstorm"}"#)
            let reply = try await agent.execute(parts: [result], onToken: nil, options: options()).await()
            print("[verify] tool answer: \(reply)")
            print("[verify] TOOLS: \(reply.contains("27") || reply.lowercased().contains("hail") ? "round trip OK" : "result ignored")")
        } else {
            print("[verify] TOOLS: no tool call emitted")
        }
    }
}
