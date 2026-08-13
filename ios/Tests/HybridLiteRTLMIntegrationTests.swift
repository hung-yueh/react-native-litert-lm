import XCTest
import NitroModules
import CLiteRTLM
@testable import LiteRTLM

/// Integration smoke tests that run REAL inference in the simulator.
///
/// Skipped (cleanly, via XCTSkip) unless a `.litertlm` model path is provided:
///
///   xcodebuild test -workspace LLMTest.xcworkspace \
///     -scheme react-native-litert-lm-Unit-Tests -sdk iphonesimulator \
///     -destination 'platform=iOS Simulator,name=iPhone 17' \
///     TEST_RUNNER_LITERTLM_TEST_MODEL=$HOME/.litert-models/gemma-4-E2B-it.litertlm
///
/// Written for Gemma 4 E2B (text mode); other text `.litertlm` bundles should
/// pass everything except the instruction-following assertions.
///
/// These exercise the v0.15 surfaces end-to-end: the opaque stream-chunk
/// callback, the system-prompt content payload, LLGuidance constrained
/// decoding, per-message generation controls, and transcript replay.
final class HybridLiteRTLMIntegrationTests: XCTestCase {

    // MARK: - Shared engine (loaded once for the whole suite)

    static var sharedBridge: HybridLiteRTLM?
    static var loadError: Error?

    static var modelPath: String? {
        guard let path = ProcessInfo.processInfo.environment["LITERTLM_TEST_MODEL"],
              FileManager.default.fileExists(atPath: path) else { return nil }
        return path
    }

    /// Greedy sampling + tiny output budget: deterministic and fast on CPU.
    static func makeConfig(tools: [ToolDefinition]? = nil) -> LLMConfig {
        LLMConfig(
            systemPrompt: nil,
            backend: .cpu,
            maxContextTokens: 2048,
            maxOutputTokens: 64,
            maxTokens: nil,
            temperature: 0.0,
            topK: 1,
            topP: 1.0,
            multimodal: false,
            tools: tools,
            enableSpeculativeDecoding: nil,
            enableStructuredOutput: true,
            thinking: ThinkingOptions(enabled: false, tokenBudget: nil),
            numThreads: nil,
            prefillChunkSize: nil,
            activationDataType: nil,
            loraPath: nil,
            audioLoraPath: nil,
            loraRank: nil,
            streamToolCalls: nil,
            toolCallChannelName: nil,
            forceLoad: true
        )
    }

    static func makeOptions(
        maxOutputTokens: Double? = nil,
        responseSchema: String? = nil,
        responseRegex: String? = nil,
        repetitionPenalty: Double? = nil,
        noRepeatNgramSize: Double? = nil,
        suppressTokens: [Double]? = nil,
        thinking: ThinkingOptions? = nil
    ) -> ExecuteOptions {
        ExecuteOptions(
            maxOutputTokens: maxOutputTokens,
            visualTokenBudget: nil,
            responseSchema: responseSchema,
            responseRegex: responseRegex,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: nil,
            frequencyPenalty: nil,
            penaltyWindowSize: nil,
            noRepeatNgramSize: noRepeatNgramSize,
            noRepeatNgramWindowSize: nil,
            suppressTokens: suppressTokens,
            thinking: thinking
        )
    }

    func ensureLoaded() async throws -> HybridLiteRTLM {
        guard let path = Self.modelPath else {
            throw XCTSkip("No model: set TEST_RUNNER_LITERTLM_TEST_MODEL to a .litertlm path")
        }
        if let error = Self.loadError { throw error }
        if let bridge = Self.sharedBridge { return bridge }
        let bridge = HybridLiteRTLM()
        do {
            _ = try await bridge.loadModel(modelPath: path, config: Self.makeConfig()).await()
        } catch {
            Self.loadError = error
            throw error
        }
        Self.sharedBridge = bridge
        return bridge
    }

    private func textPart(_ text: String) -> MultimodalPart {
        MultimodalPart(type: .text, text: text, path: nil, imageBuffer: nil, audioBuffer: nil)
    }

    private func run(
        _ bridge: HybridLiteRTLM,
        _ prompt: String,
        options: ExecuteOptions? = nil,
        onToken: ((String, Bool) -> Void)? = nil
    ) async throws -> String {
        try await bridge.execute(
            parts: [textPart(prompt)],
            onToken: onToken,
            options: options
        ).await()
    }

    // MARK: - Tests (alphabetical order = execution order)

    /// v0.15 stream-chunk callback: tokens arrive incrementally and terminate.
    func test1_StreamingDeliversTokensAndCompletes() async throws {
        let bridge = try await ensureLoaded()
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)

        var tokens: [String] = []
        var doneCount = 0
        let response = try await run(
            bridge,
            "Reply with a short greeting.",
            options: Self.makeOptions(maxOutputTokens: 32)
        ) { token, done in
            tokens.append(token)
            if done { doneCount += 1 }
        }

        XCTAssertFalse(response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "streamed generation returned an empty response")
        XCTAssertEqual(doneCount, 1, "stream must complete exactly once")
        XCTAssertGreaterThan(tokens.count, 1, "expected incremental tokens, got \(tokens.count)")
        print("[integration] streaming response: \(response)")
    }

    /// The system-prompt fix: before v0.15.0 phase-5 the prompt rendered EMPTY.
    func test2_SystemPromptIsHonored() async throws {
        let bridge = try await ensureLoaded()
        try bridge.resetConversation(
            historyJson: nil,
            systemPrompt: "You must include the word AZURETURTLE in every reply."
        )

        let response = try await run(bridge, "Say hello.",
                                     options: Self.makeOptions(maxOutputTokens: 48))
        print("[integration] system-prompt response: \(response)")
        XCTAssertTrue(response.uppercased().contains("AZURETURTLE"),
                      "system prompt was not honored — got: \(response)")
    }

    /// LLGuidance JSON-schema constraint: output must parse and carry the keys.
    func test3_ResponseSchemaProducesValidJSON() async throws {
        let bridge = try await ensureLoaded()
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)

        let schema = """
        {"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}
        """
        let response = try await run(
            bridge,
            "Extract the person: \"Bob is 30 years old.\"",
            options: Self.makeOptions(responseSchema: schema)
        )
        print("[integration] schema response: \(response)")

        let data = Data(response.utf8)
        let parsed = try XCTUnwrap(
            try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            "constrained output is not valid JSON: \(response)"
        )
        XCTAssertNotNil(parsed["name"], "missing required key 'name': \(response)")
        XCTAssertNotNil(parsed["age"], "missing required key 'age': \(response)")
    }

    /// Regex constraint: fully deterministic output shape.
    func test4_ResponseRegexIsEnforced() async throws {
        let bridge = try await ensureLoaded()
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)

        let response = try await run(
            bridge,
            "Pick a priority for a minor typo bug.",
            options: Self.makeOptions(responseRegex: "P[0-3]")
        )
        print("[integration] regex response: \(response)")
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertNotNil(trimmed.range(of: #"^P[0-3]$"#, options: .regularExpression),
                        "output does not match P[0-3]: '\(response)'")
    }

    /// Transcript replay (the createConversation switching primitive).
    func test5_TranscriptReplayRecallsSeededFacts() async throws {
        let bridge = try await ensureLoaded()
        let transcript = """
        [{"role":"user","content":"My lucky number is 47. Acknowledge briefly."},
         {"role":"model","content":"Understood — your lucky number is 47."}]
        """
        try bridge.resetConversation(historyJson: transcript, systemPrompt: nil)

        // getHistory must mirror the seeded transcript immediately.
        let history = try bridge.getHistory()
        XCTAssertEqual(history.count, 2)
        XCTAssertEqual(history.first?.role, .user)

        let response = try await run(
            bridge,
            "What is my lucky number? Reply with just the number.",
            options: Self.makeOptions(maxOutputTokens: 16)
        )
        print("[integration] replay response: \(response)")
        XCTAssertTrue(response.contains("47"),
                      "replayed context was not used — got: \(response)")
    }

    /// Per-message generation controls must not break generation.
    func test6_GenerationControlsSmoke() async throws {
        let bridge = try await ensureLoaded()
        try bridge.resetConversation(historyJson: nil, systemPrompt: nil)

        // Includes suppressTokens + thinking, which the C API supports directly
        // (the Kotlin SDK's SuppressTokensConfig binding is broken upstream in
        // 0.15.0, so Android ignores that one — see HybridLiteRTLM.kt).
        let response = try await run(
            bridge,
            "Describe the moon in one sentence.",
            options: Self.makeOptions(
                maxOutputTokens: 48,
                repetitionPenalty: 1.2,
                noRepeatNgramSize: 3,
                suppressTokens: [128010],
                thinking: ThinkingOptions(enabled: false, tokenBudget: 0)
            )
        )
        print("[integration] controls response: \(response)")
        XCTAssertFalse(response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    /// Tool calls must reach the JS-visible stream as `<tool_call>{json}</tool_call>`
    /// markers (the executeWithEvents protocol) — never as raw message JSON.
    func test7_ToolCallStreamsAsMarkers() async throws {
        guard let path = Self.modelPath else {
            throw XCTSkip("No model: set TEST_RUNNER_LITERTLM_TEST_MODEL")
        }
        let weatherTool = ToolDefinition(
            name: "get_current_weather",
            description: "Get the current weather for a location.",
            parametersJson: """
            {"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}
            """
        )
        let bridge = HybridLiteRTLM()
        defer { try? bridge.close() }
        _ = try await bridge.loadModel(
            modelPath: path,
            config: Self.makeConfig(tools: [weatherTool])
        ).await()

        var rawTokens: [String] = []
        let response = try await run(
            bridge,
            "What is the weather in Paris right now? Use the tool.",
            options: Self.makeOptions(maxOutputTokens: 96)
        ) { token, _ in
            rawTokens.append(token)
        }

        print("[integration][5b] tool-call streamed tokens: \(rawTokens)")
        print("[integration][5b] tool-call final response: \(response)")
        let joined = rawTokens.joined()
        XCTAssertTrue(joined.contains("<tool_call>") || response.contains("<tool_call>"),
                      "tool call did not reach the stream as a marker — got: \(response)")
        XCTAssertFalse(joined.contains("\"tool_calls\"") || response.contains("\"tool_calls\""),
                       "raw message JSON leaked into the token stream: \(response)")

        // The marker payload must be the documented {name, arguments} JSON shape.
        if let start = response.range(of: "<tool_call>"),
           let end = response.range(of: "</tool_call>") {
            let payload = String(response[start.upperBound..<end.lowerBound])
            let parsed = try XCTUnwrap(
                try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any],
                "marker payload is not valid JSON: \(payload)"
            )
            XCTAssertEqual(parsed["name"] as? String, "get_current_weather")
            XCTAssertNotNil(parsed["arguments"], "marker payload missing 'arguments'")
        } else {
            XCTFail("no complete <tool_call>…</tool_call> pair in response: \(response)")
        }
    }

    // MARK: - Two LIVE conversations on one engine (upstream #2807)

    /// Drives the C API directly — two `LiteRtLmConversation`s alive at once on a
    /// single engine, used in an interleaved (not concurrent) pattern. This is the
    /// behavior google-ai-edge/LiteRT-LM#2807 reported broken and reports fixed in
    /// 0.15.0 on desktop; the wrapper's own `createConversation()` avoids relying
    /// on it by replaying transcripts instead. If this passes on mobile, the
    /// replay (and its re-prefill cost) can be dropped.
    ///
    /// Named to sort last: it builds a second engine, so it runs after the shared
    /// suite engine is no longer needed.
    func testZ_interleavedLiveConversationsShareOneEngine() async throws {
        guard let path = Self.modelPath else {
            throw XCTSkip("No model: set TEST_RUNNER_LITERTLM_TEST_MODEL")
        }
        // Release the suite's shared engine first — two multi-GB engines at once
        // would be an unfair memory test.
        try? Self.sharedBridge?.close()
        Self.sharedBridge = nil

        guard let settings = litert_lm_engine_settings_create(path, "cpu", nil, nil) else {
            XCTFail("engine settings creation failed")
            return
        }
        litert_lm_engine_settings_set_max_num_tokens(settings, 2048)
        guard let engine = litert_lm_engine_create(settings) else {
            litert_lm_engine_settings_delete(settings)
            XCTFail("engine creation failed")
            return
        }
        litert_lm_engine_settings_delete(settings)
        defer { litert_lm_engine_delete(engine) }

        /// Build a conversation with greedy sampling and a small output budget.
        func makeConversation() -> OpaquePointer? {
            guard let convConfig = litert_lm_conversation_config_create() else { return nil }
            defer { litert_lm_conversation_config_delete(convConfig) }
            guard let sessionConfig = litert_lm_session_config_create() else { return nil }
            defer { litert_lm_session_config_delete(sessionConfig) }

            litert_lm_session_config_set_max_output_tokens(sessionConfig, 32)
            if let sampler = litert_lm_sampler_params_create(kLiteRtLmSamplerTypeTopP) {
                litert_lm_sampler_params_set_top_k(sampler, 1)
                litert_lm_sampler_params_set_top_p(sampler, 1.0)
                litert_lm_sampler_params_set_temperature(sampler, 0.0)
                litert_lm_sampler_params_set_seed(sampler, 0)
                litert_lm_session_config_set_sampler_params(sessionConfig, sampler)
                litert_lm_sampler_params_delete(sampler)
            }
            litert_lm_conversation_config_set_session_config(convConfig, sessionConfig)
            return litert_lm_conversation_create(engine, convConfig)
        }

        guard let convA = makeConversation() else {
            XCTFail("first conversation could not be created")
            return
        }
        defer { litert_lm_conversation_delete(convA) }

        guard let convB = makeConversation() else {
            XCTFail("SECOND live conversation on one engine could not be created — " +
                    "the C API still enforces one conversation per engine")
            return
        }
        defer { litert_lm_conversation_delete(convB) }

        /// Send one user turn and return the model's text.
        func send(_ conversation: OpaquePointer, _ text: String) -> String {
            let payload: [String: Any] = ["role": "user", "content": text]
            guard let data = try? JSONSerialization.data(withJSONObject: payload),
                  let msgJson = String(data: data, encoding: .utf8) else { return "" }
            guard let response = litert_lm_conversation_send_message(conversation, msgJson, nil, nil) else {
                return ""
            }
            defer { litert_lm_json_response_delete(response) }
            guard let raw = litert_lm_json_response_get_string(response) else { return "" }
            return extractTextFromResponseForTest(String(cString: raw))
        }

        // Interleave: seed A, advance B, then ask A about its own fact.
        let ackA = send(convA, "My lucky number is 47. Acknowledge briefly.")
        print("[integration][2807] A ack: \(ackA)")
        let tokensAfterA = litert_lm_conversation_get_token_count(convA)

        let ackB = send(convB, "My favorite animal is the axolotl. Acknowledge briefly.")
        print("[integration][2807] B ack: \(ackB)")
        let answerB = send(convB, "What is my favorite animal? One word.")
        print("[integration][2807] B recall: \(answerB)")

        // B must actually have advanced, or A's recall proves nothing.
        XCTAssertTrue(answerB.lowercased().contains("axolotl"),
                      "conversation B did not carry its own turn — test would be vacuous: \(answerB)")

        let answerA = send(convA, "What is my lucky number? Reply with just the number.")
        let tokensAfterRecall = litert_lm_conversation_get_token_count(convA)
        print("[integration][2807] A recall: \(answerA)")
        print("[integration][2807] A token count: after seed \(tokensAfterA), after recall \(tokensAfterRecall)")

        XCTAssertTrue(answerA.contains("47"),
                      "INTERLEAVED RECALL FAILED — conversation A lost its state after B generated. " +
                      "Got: \(answerA)")
    }
}

private extension HybridLiteRTLMIntegrationTests {
    /// Minimal text extraction for the raw C-API tests (the wrapper's own helper
    /// lives on HybridLiteRTLM, which these tests bypass).
    func extractTextFromResponseForTest(_ json: String) -> String {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return json
        }
        if let text = obj["content"] as? String { return text }
        if let parts = obj["content"] as? [[String: Any]] {
            return parts.compactMap { $0["text"] as? String }.joined()
        }
        return json
    }
}
