package com.margelo.nitro.dev.litert.litertlm

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.Message as SdkMessage
import com.margelo.nitro.JNIOnLoad
import dev.litert.litertlm.LiteRTLMInitProvider
import org.json.JSONArray
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.concurrent.thread

/**
 * On-device integration tests that run REAL inference through the Kotlin
 * bridge — the Android counterpart of `ios/Tests/HybridLiteRTLMIntegrationTests.swift`.
 *
 * Robolectric cannot execute the LiteRT-LM engine, so these are the only tests
 * that prove the Android wiring (ConversationConfig fields, the `tool()`
 * adapter, per-call generation params, tool-call marker surfacing) against the
 * real SDK rather than just compiling against it.
 *
 * Requires a `.litertlm` bundle in the test app's external files dir; skipped
 * (via `assumeTrue`) when absent, so `connectedAndroidTest` stays green on a
 * bare device.
 *
 * Install the APK BEFORE pushing the model, and run via `am instrument`:
 * `connectedAndroidTest` uninstalls the test APK when it finishes, which
 * deletes `/sdcard/Android/data/<pkg>/` — and with it the pushed model — so a
 * Gradle run can only ever skip. `am instrument` also prints the test's own
 * output (generated text) straight to the console.
 *
 *   cd example/android
 *   ANDROID_HOME=~/Library/Android/sdk ./gradlew :react-native-litert-lm:assembleDebugAndroidTest
 *   adb install -r -g ../../android/build/outputs/apk/androidTest/debug/react-native-litert-lm-debug-androidTest.apk
 *   adb shell mkdir -p /sdcard/Android/data/dev.litert.litertlm.test/files
 *   adb push gemma-4-E2B-it.litertlm /sdcard/Android/data/dev.litert.litertlm.test/files/
 *   adb shell am instrument -w -r \
 *     -e class com.margelo.nitro.dev.litert.litertlm.HybridLiteRTLMInstrumentedTest \
 *     dev.litert.litertlm.test/androidx.test.runner.AndroidJUnitRunner
 *
 * Method order is fixed so the engine loads at most twice (with and without
 * tools) instead of once per test — each load memory-maps a multi-GB bundle.
 */
@RunWith(AndroidJUnit4::class)
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
class HybridLiteRTLMInstrumentedTest {

    companion object {
        private const val LOAD_TIMEOUT_SEC = 180L
        private const val GENERATE_TIMEOUT_SEC = 180L

        private var bridge: HybridLiteRTLM? = null
        private var bridgeHasTools = false

        /**
         * A minimal Gemma-format template whose system turn forces one fixed
         * word — if the reply obeys, the override replaced the bundle's
         * template. Same as the iOS suite's.
         */
        val BANANA_TEMPLATE = """
            <|turn>system
            Whatever the user says, reply with exactly one word: BANANA<turn|>
            {% for message in messages -%}
            <|turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}
            {% if message['content'] is string %}{{ message['content'] }}{% else %}{% for part in message['content'] %}{% if part['type'] == 'text' %}{{ part['text'] }}{% endif %}{% endfor %}{% endif %}<turn|>
            {% endfor -%}
            {% if add_generation_prompt %}<|turn>model
            {% endif %}
        """.trimIndent()

        /** `am instrument -e backend gpu` selects the GPU backend; default CPU. */
        fun testBackend(): Backend =
            if (InstrumentationRegistry.getArguments().getString("backend") == "gpu") Backend.GPU
            else Backend.CPU

        fun modelFile(): File? {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val dir = context.getExternalFilesDir(null) ?: return null
            // Embedding bundles sit alongside; they belong to HybridEmbedderInstrumentedTest.
            return dir.listFiles()?.firstOrNull {
                it.name.endsWith(".litertlm") && !it.name.startsWith("embeddinggemma")
            }
        }

        /** example/test.jpeg / test.wav, pushed next to the model; null if absent. */
        fun fixture(name: String): File? {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            return context.getExternalFilesDir(null)?.let { File(it, name) }?.takeIf { it.exists() }
        }

        @AfterClass
        @JvmStatic
        fun tearDownClass() {
            bridge?.close()
            bridge = null
        }
    }

    /**
     * A React Native app bootstraps these at startup; a bare test APK must do
     * it itself. fbjni's HybridData (used by Nitro's Promise) loads its own
     * .so through NativeLoader, which throws until it is initialized.
     */
    private fun ensureNativeInit() {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(SystemDelegate())
        }
        JNIOnLoad.initializeNativeNitro()

        // The init ContentProvider supplies the application context in a real
        // app; in a self-instrumenting test APK, set it explicitly.
        if (LiteRTLMInitProvider.applicationContext == null) {
            val field = LiteRTLMInitProvider::class.java.getDeclaredField("applicationContext")
            field.isAccessible = true
            field.set(null, InstrumentationRegistry.getInstrumentation().targetContext)
        }
    }

    /** Load (or reuse) an engine; `withTools` registers a tool definition. */
    private fun engine(withTools: Boolean): HybridLiteRTLM {
        val model = modelFile()
        assumeTrue(
            "No .litertlm in the test app's external files dir — skipping on-device inference",
            model != null,
        )
        ensureNativeInit()

        bridge?.let { existing ->
            if (bridgeHasTools == withTools) return existing
            existing.close()
            bridge = null
        }

        val tools = if (withTools) {
            arrayOf(
                ToolDefinition(
                    "get_current_weather",
                    "Get the current weather for a location.",
                    """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
                )
            )
        } else null

        val fresh = HybridLiteRTLM()
        // Greedy sampling keeps generations deterministic and short on CPU.
        fresh.loadModel(
            model!!.absolutePath,
            llmConfig(tools = tools, structuredOutput = true),
        )

        // loadModel resolves through a Promise; poll the plain isReady() flag so
        // the assertions never depend on Promise/JSI plumbing.
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(LOAD_TIMEOUT_SEC)
        while (!fresh.isReady() && System.nanoTime() < deadline) {
            Thread.sleep(250)
        }
        assertTrue(
            "Engine failed to load within ${LOAD_TIMEOUT_SEC}s (withTools=$withTools) — check logcat",
            fresh.isReady(),
        )
        bridge = fresh
        bridgeHasTools = withTools
        return fresh
    }

    private fun textPart(text: String) =
        MultimodalPart(PartType.TEXT, text, null, null, null, null, null)

    /**
     * Greedy, short-output config for these tests (named, so new LLMConfig
     * fields don't shift it). Override what a test needs.
     */
    private fun llmConfig(
        backend: Backend = testBackend(),
        multimodal: Boolean? = false,
        tools: Array<ToolDefinition>? = null,
        structuredOutput: Boolean? = null,
        activationDataType: ActivationDataType? = null,
        chatTemplate: String? = null,
        maxVisionTokensPerImage: Double? = null,
        numThreads: Double? = null,
    ) = LLMConfig(
        systemPrompt = null, backend = backend, maxContextTokens = 2048.0, maxOutputTokens = 64.0,
        maxTokens = null, temperature = 0.0, topK = 1.0, topP = 1.0, multimodal = multimodal,
        tools = tools, enableSpeculativeDecoding = null, chatTemplate = chatTemplate,
        enableStructuredOutput = structuredOutput, thinking = ThinkingOptions(false, null),
        numThreads = numThreads, maxVisionTokensPerImage = maxVisionTokensPerImage, gpu = null,
        singleThreadedExecution = null, prefillChunkSize = null,
        activationDataType = activationDataType, loraPath = null, audioLoraPath = null, loraRank = null,
        streamToolCalls = null, toolCallChannelName = null, forceLoad = true,
    )

    private fun options(
        maxOutputTokens: Double? = null,
        visualTokenBudget: Double? = null,
        responseSchema: String? = null,
        responseRegex: String? = null,
        repetitionPenalty: Double? = null,
        noRepeatNgramSize: Double? = null,
        suppressTokens: DoubleArray? = null,
        thinking: ThinkingOptions? = null,
    ) = ExecuteOptions(
        maxOutputTokens, visualTokenBudget, responseSchema, responseRegex,
        repetitionPenalty, null, null, null,
        noRepeatNgramSize, null, suppressTokens, thinking,
    )

    /**
     * Run one generation and return everything the JS layer would see. Uses the
     * streaming callback (a plain Kotlin lambda) for both the text and the
     * completion signal, so no Promise bridging is involved.
     */
    private fun generate(
        llm: HybridLiteRTLM,
        prompt: String,
        options: ExecuteOptions? = null,
    ): Pair<String, List<String>> = generateParts(llm, arrayOf(textPart(prompt)), options)

    /** The context token count once it is known (a reset reads it in the background), or -1 after 10 s. */
    private fun knownContextTokens(llm: HybridLiteRTLM): Double {
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10)
        var tokens = llm.getContextTokenCount()
        while (tokens < 0 && System.nanoTime() < deadline) {
            Thread.sleep(50)
            tokens = llm.getContextTokenCount()
        }
        return tokens
    }

    /** [generate] for any parts (e.g. tool results). */
    private fun generateParts(
        llm: HybridLiteRTLM,
        parts: Array<MultimodalPart>,
        options: ExecuteOptions? = null,
    ): Pair<String, List<String>> {
        val tokens = mutableListOf<String>()
        val done = CountDownLatch(1)
        val sb = StringBuilder()
        llm.execute(parts, { token, isDone ->
            tokens.add(token)
            sb.append(token)
            if (isDone) done.countDown()
        }, options)
        assertTrue(
            "Generation did not complete within ${GENERATE_TIMEOUT_SEC}s",
            done.await(GENERATE_TIMEOUT_SEC, TimeUnit.SECONDS),
        )
        return sb.toString() to tokens
    }

    // ── Engine without tools ────────────────────────────────────────────────

    /**
     * Throughput baseline for engine bumps — reported, not gated. One warm-up
     * plus three measured runs of a fixed prompt under greedy sampling. The
     * Kotlin SDK exposes no benchmark info, so rates are measured from the
     * stream itself: TTFT = first chunk, decode = chunks/s after the first
     * (one chunk per decoded token on this path). Sorts first, so the shared
     * engine's load is timed here.
     */
    @Test
    fun test0_benchmarkDecodeThroughput() {
        val loadStart = System.nanoTime()
        val llm = engine(withTools = false)
        val loadSec = (System.nanoTime() - loadStart) / 1e9

        val prompt = "Write a detailed paragraph about the history of the bicycle."
        val ttftsMs = mutableListOf<Double>()
        val rates = mutableListOf<Double>()
        for (run in 0 until 4) {
            llm.resetConversation(null, null, null)
            var first = 0L
            var last = 0L
            var chunks = 0
            val done = CountDownLatch(1)
            val start = System.nanoTime()
            llm.execute(arrayOf(textPart(prompt)), { token, isDone ->
                if (token.isNotEmpty() && !isDone) {
                    val now = System.nanoTime()
                    if (first == 0L) first = now
                    last = now
                    chunks++
                }
                if (isDone) done.countDown()
            }, options(maxOutputTokens = 128.0))
            assertTrue("benchmark run $run timed out", done.await(GENERATE_TIMEOUT_SEC, TimeUnit.SECONDS))
            val ttftMs = (first - start) / 1e6
            val rate = if (chunks > 1 && last > first) (chunks - 1) / ((last - first) / 1e9) else 0.0
            println("[android-bench] run $run: ttft=${"%.0f".format(ttftMs)}ms decode=${"%.2f".format(rate)} tok/s chunks=$chunks")
            if (run == 0) continue // warm-up
            ttftsMs.add(ttftMs)
            rates.add(rate)
        }
        fun median(xs: List<Double>) = xs.sorted()[xs.size / 2]
        println(
            "[android-bench] SUMMARY backend=${testBackend()} load=${"%.1f".format(loadSec)}s " +
                "median_ttft=${"%.0f".format(median(ttftsMs))}ms median_decode=${"%.2f".format(median(rates))} tok/s"
        )
        assertTrue("no decode throughput measured", median(rates) > 0)
    }

    @Test
    fun test1_streamingDeliversTokensAndCompletes() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, null, null)

        val (response, tokens) = generate(llm, "Reply with a short greeting.", options(maxOutputTokens = 32.0))
        println("[android-int] streaming response: $response")
        assertFalse("streamed generation was empty", response.isBlank())
        assertTrue("expected incremental tokens, got ${tokens.size}", tokens.size > 1)
        // Milliseconds on both platforms, as documented.
        val stats = llm.getStats()
        assertTrue("totalTime should be in milliseconds: ${stats.totalTime}", stats.totalTime > 50)
        assertTrue(stats.timeToFirstToken > 0 && stats.timeToFirstToken <= stats.totalTime)
    }

    @Test
    fun test2_systemPromptIsHonored() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, "You must include the word AZURETURTLE in every reply.", null)

        val (response, _) = generate(llm, "Say hello.", options(maxOutputTokens = 48.0))
        println("[android-int] system-prompt response: $response")
        assertTrue(
            "system prompt was not honored — got: $response",
            response.uppercase().contains("AZURETURTLE"),
        )
    }

    /** enableResponseFormat + per-call ResponseFormat.json (constrained decoding). */
    @Test
    fun test3_responseSchemaProducesValidJson() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, null, null)

        val schema =
            """{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"""
        val (response, _) = generate(
            llm,
            "Extract the person: \"Bob is 30 years old.\"",
            options(responseSchema = schema),
        )
        println("[android-int] schema response: $response")
        val parsed = JSONObject(response.trim()) // throws if not valid JSON
        assertTrue("missing required key 'name': $response", parsed.has("name"))
        assertTrue("missing required key 'age': $response", parsed.has("age"))
    }

    @Test
    fun test4_responseRegexIsEnforced() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, null, null)

        val (response, _) = generate(
            llm,
            "Pick a priority for a minor typo bug.",
            options(responseRegex = "P[0-3]"),
        )
        println("[android-int] regex response: $response")
        assertTrue(
            "output does not match P[0-3]: '$response'",
            Regex("^P[0-3]$").matches(response.trim()),
        )
    }

    /** ConversationConfig.initialMessages — the createConversation() switching primitive. */
    @Test
    fun test5_transcriptReplayRecallsSeededFacts() {
        val llm = engine(withTools = false)
        val transcript = JSONArray(
            listOf(
                JSONObject(mapOf("role" to "user", "content" to "My lucky number is 47. Acknowledge briefly.")),
                JSONObject(mapOf("role" to "model", "content" to "Understood — your lucky number is 47.")),
            )
        ).toString()
        llm.resetConversation(transcript, null, null)

        // The wrapper's own history must mirror the replayed transcript.
        val history = llm.getHistory()
        assertEquals(2, history.size)
        assertEquals(Role.USER, history[0].role)

        val (response, _) = generate(
            llm,
            "What is my lucky number? Reply with just the number.",
            options(maxOutputTokens = 16.0),
        )
        println("[android-int] replay response: $response")
        assertTrue("replayed context was not used — got: $response", response.contains("47"))
    }

    /** Per-message maxOutputTokens — newly effective on Android in 0.15. */
    @Test
    fun test6_perMessageMaxOutputTokensLimitsLength() {
        val llm = engine(withTools = false)

        llm.resetConversation(null, null, null)
        val (long, _) = generate(llm, "Explain what the ocean is.", options(maxOutputTokens = 64.0))
        llm.resetConversation(null, null, null)
        val (short, _) = generate(llm, "Explain what the ocean is.", options(maxOutputTokens = 8.0))

        println("[android-int] long (64 tok): $long")
        println("[android-int] short (8 tok): $short")
        assertFalse("short generation was empty", short.isBlank())
        assertTrue(
            "an 8-token cap did not shorten the response (short=${short.length} chars, long=${long.length} chars)",
            short.length < long.length,
        )
    }

    /**
     * Penalties / n-gram ban / token suppression must not break generation.
     * suppressTokens is the regression guard for google-ai-edge/LiteRT-LM#3229:
     * through 0.16.1 constructing SuppressTokensConfig aborted the process.
     */
    @Test
    fun test7_generationControlsSmoke() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, null, null)

        val (response, _) = generate(
            llm,
            "Describe the moon in one sentence.",
            options(
                maxOutputTokens = 48.0,
                repetitionPenalty = 1.2,
                noRepeatNgramSize = 3.0,
                suppressTokens = doubleArrayOf(128010.0),
                thinking = ThinkingOptions(false, 0.0),
            ),
        )
        println("[android-int] controls response: $response")
        assertFalse(response.isBlank())
    }

    // ── Engine with tools (loads a second engine; keep last) ─────────────────

    /**
     * Regression test for the crash fixed in 0.15: registering tools used to
     * force-cast OpenApiTool to ToolProvider (always a ClassCastException), so
     * any loadModel with tools rejected.
     */
    @Test
    fun test8_loadModelWithToolsSucceeds() {
        val llm = engine(withTools = true)
        assertTrue("engine with tools should be ready", llm.isReady())
    }

    /** Tool calls must reach the JS-visible stream as <tool_call>{json}</tool_call>. */
    @Test
    fun test9_toolCallSurfacesAsMarker() {
        val llm = engine(withTools = true)
        llm.resetConversation(null, null, null)

        val (response, tokens) = generate(
            llm,
            "What is the weather in Paris right now? Use the tool.",
            options(maxOutputTokens = 96.0),
        )
        println("[android-int] tool-call tokens: $tokens")
        println("[android-int] tool-call response: $response")

        val start = response.indexOf("<tool_call>")
        assumeTrue(
            "model did not emit a tool call for this prompt — nothing to assert",
            start >= 0,
        )
        val end = response.indexOf("</tool_call>", start)
        assertTrue("unterminated tool_call marker: $response", end > start)
        val payload = response.substring(start + "<tool_call>".length, end)
        val parsed = JSONObject(payload)
        assertEquals("get_current_weather", parsed.getString("name"))
        assertNotNull("marker payload missing 'arguments'", parsed.opt("arguments"))
        assertFalse(
            "raw message JSON leaked into the token stream: $response",
            response.contains("\"tool_calls\""),
        )
    }

    /**
     * The agent loop: the model calls a tool, the app answers with a tool
     * result, and the reply uses it. The exchange lands in getHistory() as
     * structured turns and survives a replay into a new conversation.
     */
    @Test
    fun test9b_toolResultRoundTrip() {
        val llm = engine(withTools = true)
        llm.resetConversation(null, null, null)

        val (call, _) = generate(
            llm, "What is the weather in Paris right now? Use the tool.", options(maxOutputTokens = 96.0),
        )
        println("[android-int][tools] call: $call")
        assertTrue("the model did not call the tool — got: $call", call.contains("<tool_call>"))

        // Values the model cannot guess, so the answer proves it read the result.
        val result = MultimodalPart(
            PartType.TOOLRESULT, null, null, null, null, "get_current_weather",
            """{"location":"Paris","temperature_c":27,"condition":"hailstorm"}""",
        )
        val (answer, _) = generateParts(llm, arrayOf(result), options(maxOutputTokens = 96.0))
        println("[android-int][tools] answer: $answer")
        assertTrue(
            "the answer ignored the tool result — got: $answer",
            answer.contains("27") || answer.lowercase().contains("hail"),
        )

        val history = llm.getHistory()
        assertEquals(listOf(Role.USER, Role.MODEL, Role.TOOL, Role.MODEL), history.map { it.role })
        assertEquals("get_current_weather", history[1].toolCalls?.firstOrNull()?.name)
        assertFalse("tool calls must leave the model text", history[1].content.contains("<tool_call>"))
        assertEquals("get_current_weather", history[2].toolName)
        assertTrue(history[2].content.contains("hailstorm"))

        // Mixing a tool result with other parts is refused, typed.
        val refused = awaitResult(llm.execute(arrayOf(result, textPart("and tomorrow?")), null, null))
            .exceptionOrNull()
        assertEquals("INVALID_INPUT", (refused as LiteRTLMNativeException).code)

        // Replay the transcript (the JSON getHistory() gives JS) into a new conversation.
        val transcript = JSONArray()
        history.forEach { message ->
            val item = JSONObject().put("role", message.role.name.lowercase()).put("content", message.content)
            message.toolCalls?.let { calls ->
                item.put("toolCalls", JSONArray(calls.map { JSONObject().put("name", it.name).put("argumentsJson", it.argumentsJson) }))
            }
            message.toolName?.let { item.put("toolName", it) }
            transcript.put(item)
        }
        llm.resetConversation(transcript.toString(), null, null)
        assertEquals(listOf(Role.USER, Role.MODEL, Role.TOOL, Role.MODEL), llm.getHistory().map { it.role })
        val (recall, _) = generate(
            llm, "What condition did the weather tool report? Answer in one word.", options(maxOutputTokens = 16.0),
        )
        println("[android-int][tools] recall after replay: $recall")
        assertTrue("the replayed tool result was lost — got: $recall", recall.lowercase().contains("hail"))
    }

    /** Frees the shared engine so a test's own engine never sits beside it. */
    private fun releaseSharedEngine() {
        bridge?.close()
        bridge = null
    }

    @Test
    fun test9c_chatTemplateReplacesTheBundles() {
        val model = modelFile()
        assumeTrue("No .litertlm pushed — skipping", model != null)
        ensureNativeInit()
        releaseSharedEngine()
        val llm = HybridLiteRTLM()
        try {
            llm.loadModelBlocking(model!!.absolutePath, llmConfig(chatTemplate = BANANA_TEMPLATE))
            val (reply, _) = generate(llm, "What is the capital of France?", options(maxOutputTokens = 8.0))
            println("[android-int][template] reply: $reply")
            assertTrue("the custom chat template was not applied — got: $reply", reply.uppercase().contains("BANANA"))
        } finally {
            llm.close()
        }
    }

    /**
     * Calls the JS thread makes while a message generates return at once, and
     * a reset or close() cancels the message instead of waiting it out — the
     * counterpart of the iOS suite's testJ. Through 0.7, close() could wait
     * behind a whole load, and a reset or close() during a blocking send
     * freed the conversation under it (the SDK's close() frees native state
     * at once).
     */
    @Test
    fun testJ_jsThreadNeverWaitsForTheEngine() {
        val model = modelFile()
        assumeTrue("No .litertlm pushed — skipping", model != null)
        ensureNativeInit()
        releaseSharedEngine()
        val essay = arrayOf(textPart("Write a long, detailed essay about the history of the bicycle."))
        val long = options(maxOutputTokens = 512.0)
        val limitMs = 50.0

        fun ms(label: String, call: () -> Unit): Double {
            val start = System.nanoTime()
            call()
            val elapsed = (System.nanoTime() - start) / 1e6
            println("[android-int][nonblocking] $label took ${"%.2f".format(elapsed)} ms")
            return elapsed
        }
        /** A message's outcome: the latch opens when it settles, with its error if it failed. */
        class Sent(val settled: CountDownLatch = CountDownLatch(1), val error: AtomicReference<Throwable?> = AtomicReference(null))
        fun send(llm: HybridLiteRTLM, onToken: ((String, Boolean) -> Unit)? = null): Sent {
            val sent = Sent()
            llm.execute(essay, onToken, long)
                .then { sent.settled.countDown() }
                .catch { sent.error.set(it); sent.settled.countDown() }
            return sent
        }
        fun assertCancelled(label: String, sent: Sent) {
            assertTrue("$label: the message never settled", sent.settled.await(30, TimeUnit.SECONDS))
            val error = sent.error.get()
            assertNotNull("$label: the message finished instead of being cancelled", error)
            val native = error as? LiteRTLMNativeException ?: error?.cause as? LiteRTLMNativeException
            assertEquals("$label: $error", "CANCELLED", native?.status)
        }

        val llm = HybridLiteRTLM()
        llm.loadModelBlocking(model!!.absolutePath, llmConfig())

        // A blocking send holds a worker — and the conversation — for many seconds.
        var sent = send(llm)
        Thread.sleep(2_000)
        assertTrue(ms("isReady") { assertTrue(llm.isReady()) } < limitMs)
        assertTrue(ms("getHistory") { llm.getHistory() } < limitMs)
        assertTrue(ms("getStats") { llm.getStats() } < limitMs)
        assertTrue(ms("setMemoryWarningCallback") { llm.setMemoryWarningCallback { _, _ -> } } < limitMs)
        assertTrue(ms("clearMemoryWarningCallback") { llm.clearMemoryWarningCallback() } < limitMs)

        // Reset cancels the message and waits only for it to return; the next
        // message runs in the new conversation, without the cancelled one.
        var started = System.nanoTime()
        ms("resetConversation") { llm.resetConversation(null, null, null) }
        assertCancelled("reset", sent)
        println("[android-int][nonblocking] reset cancelled the message in ${(System.nanoTime() - started) / 1_000_000} ms")
        val (hello, _) = generate(llm, "Say hello.", options(maxOutputTokens = 16.0))
        assertTrue(hello.isNotBlank())
        assertEquals("the cancelled message must stay out of the transcript",
            listOf(Role.USER, Role.MODEL), llm.getHistory().map { it.role })

        // Reset mid-stream: the old conversation's count is gone at once, and
        // the new conversation's is read in the background, without waiting
        // for its first message.
        val beforeReset = llm.getContextTokenCount()
        val streaming = CountDownLatch(1)
        sent = send(llm) { token, done -> if (!done && token.isNotEmpty()) streaming.countDown() }
        assertTrue("no streamed token", streaming.await(GENERATE_TIMEOUT_SEC, TimeUnit.SECONDS))
        ms("resetConversation mid-stream") { llm.resetConversation(null, null, null) }
        val atOnce = llm.getContextTokenCount()
        assertCancelled("reset mid-stream", sent)
        val read = knownContextTokens(llm)
        println("[android-int][nonblocking] context tokens before the reset $beforeReset, at once $atOnce, read $read")
        assertTrue("no context token count after the reset: $read", read >= 0)
        assertTrue("the old conversation's count after a reset: $atOnce", atOnce == -1.0 || atOnce == read)
        generate(llm, "Say hello.", options(maxOutputTokens = 16.0))
        assertTrue("no context token count after the next message", llm.getContextTokenCount() > read)

        // close() during a blocking send: not ready at once, the message
        // cancelled, and the engine freed once it has returned.
        sent = send(llm)
        Thread.sleep(2_000)
        started = System.nanoTime()
        assertTrue(ms("close") { llm.close() } < limitMs)
        assertFalse(llm.isReady())
        assertCancelled("close", sent)
        Teardowns.await(Teardowns.mark())
        val closeSec = (System.nanoTime() - started) / 1e9
        assertTrue("close() must cancel the message, not wait it out ($closeSec s)", closeSec < 10)

        // close() mid-stream: the stream is cancelled, and the release — which
        // waits for the stream to return — completes.
        val streamer = HybridLiteRTLM()
        streamer.loadModelBlocking(model.absolutePath, llmConfig())
        val firstToken = CountDownLatch(1)
        val lastToken = AtomicReference<String?>(null)
        sent = send(streamer) { token, done ->
            if (!done && token.isNotEmpty()) firstToken.countDown()
            if (done) lastToken.set(token)
        }
        assertTrue("no streamed token", firstToken.await(GENERATE_TIMEOUT_SEC, TimeUnit.SECONDS))
        assertTrue(ms("close mid-stream") { streamer.close() } < limitMs)
        assertCancelled("close mid-stream", sent)
        // A cancel is a deliberate stop: the stream ends without error text.
        assertEquals("a cancelled stream must end with an empty done token", "", lastToken.get())
        Teardowns.await(Teardowns.mark())
    }

    /**
     * A reset reads the new conversation's count on a background thread,
     * claimed like a message. Holding that read (through readContextTokens)
     * puts a message, a second reset, close() or unload() inside it.
     */
    @Test
    fun testJb_resetReadsTheCountInTheBackground() {
        val model = modelFile()
        assumeTrue("No .litertlm pushed — skipping", model != null)
        ensureNativeInit()
        releaseSharedEngine()
        Teardowns.await(Teardowns.mark())
        val short = options(maxOutputTokens = 16.0)

        fun ms(label: String, call: () -> Unit): Double {
            val start = System.nanoTime()
            call()
            val elapsed = (System.nanoTime() - start) / 1e6
            println("[android-int][reset-count] $label took ${"%.2f".format(elapsed)} ms")
            return elapsed
        }

        /** Holds the [n]-th background read (counting from 1) before or after it reads the engine. */
        class HeldRead(llm: HybridLiteRTLM, private val n: Int, private val afterEngine: Boolean) {
            val reached = CountDownLatch(1)
            val release = CountDownLatch(1)
            @Volatile var engineMs = -1.0
            val calls = AtomicInteger()
            init {
                val read = llm.readContextTokens
                llm.readContextTokens = { conversation ->
                    val held = calls.incrementAndGet() == n
                    if (held && !afterEngine) { reached.countDown(); release.await(60, TimeUnit.SECONDS) }
                    val start = System.nanoTime()
                    read(conversation).also {
                        if (held) engineMs = (System.nanoTime() - start) / 1e6
                        if (held && afterEngine) { reached.countDown(); release.await(60, TimeUnit.SECONDS) }
                    }
                }
            }
        }

        val llm = HybridLiteRTLM()
        llm.loadModelBlocking(model!!.absolutePath, llmConfig())
        val read = llm.readContextTokens
        try {
            // Without a transcript the count is read at once, system prompt or not.
            llm.resetConversation(null, null, null)
            val empty = knownContextTokens(llm)
            llm.resetConversation(null, "You are a pirate. Begin every reply with the word Arr.", null)
            val prompted = knownContextTokens(llm)
            println("[android-int][reset-count] after a reset $empty, with a system prompt $prompted")
            assertTrue("no count after a reset: $empty", empty >= 0)
            assertTrue("no count after a reset with a system prompt: $prompted", prompted >= 0)

            // A message sent right after a reset ends with its own count, 20 times.
            fun hello(): Pair<String, Double> {
                val (reply, _) = generate(llm, "Say hello.", short)
                Thread.sleep(1_000)
                return reply to llm.getContextTokenCount()
            }
            llm.resetConversation(null, null, null)
            knownContextTokens(llm)
            val (firstReply, firstCount) = hello()
            println("[android-int][reset-count] reference: $firstCount, '$firstReply'")
            var compared = 0
            repeat(20) { i ->
                val started = System.nanoTime()
                llm.resetConversation(null, null, null)
                val (reply, count) = hello()
                Thread.sleep(1_000)
                val later = llm.getContextTokenCount()
                val sec = (System.nanoTime() - started) / 1e9
                println("[android-int][reset-count] message right after a reset #${i + 1}: $count, 1 s later $later, ${"%.1f".format(sec)} s" +
                    if (reply == firstReply) "" else ", reply differs: '$reply'")
                assertTrue("#${i + 1} took $sec s", sec < 60)
                assertTrue("#${i + 1}: no count after the message: $count", count > 0)
                assertEquals("#${i + 1}: the count moved after the message", count, later, 0.0)
                if (reply == firstReply) {
                    assertEquals("#${i + 1}: not the message's own count", firstCount, count, 0.0)
                    compared++
                }
            }
            println("[android-int][reset-count] $compared of 20 had the reference reply")

            // The read has its number, then a message finishes: the message's count stays.
            run {
                val hold = HeldRead(llm, 1, afterEngine = true)
                llm.resetConversation(null, null, null)
                assertTrue("the read never ran", hold.reached.await(10, TimeUnit.SECONDS))
                generate(llm, "Say hello.", short)
                val message = llm.getContextTokenCount()
                hold.release.countDown()
                Thread.sleep(1_000)
                val after = llm.getContextTokenCount()
                println("[android-int][reset-count] read held after the engine: message $message, after the read $after")
                assertTrue("no count after the message: $message", message > 0)
                assertEquals("a stale read replaced the message's count", message, after, 0.0)
                llm.readContextTokens = read
            }

            // A message starts while the read is held, before or after it reads the
            // engine, and the read is let go mid-decode: it records nothing, so the
            // count stays -1 while the message generates and is then the message's own.
            for (afterEngine in listOf(false, true)) {
                val hold = HeldRead(llm, 1, afterEngine)
                llm.resetConversation(null, null, null)
                assertTrue("the read never ran", hold.reached.await(10, TimeUnit.SECONDS))
                val firstToken = CountDownLatch(1)
                val done = CountDownLatch(1)
                llm.execute(arrayOf(textPart("Write two sentences about the ocean.")), { token, isDone ->
                    if (!isDone && token.isNotEmpty()) firstToken.countDown()
                    if (isDone) done.countDown()
                }, options(maxOutputTokens = 96.0))
                assertTrue("no streamed token", firstToken.await(GENERATE_TIMEOUT_SEC, TimeUnit.SECONDS))
                hold.release.countDown()
                Thread.sleep(300)
                val generating = llm.getContextTokenCount()
                val stillGenerating = done.count == 1L
                assertTrue("the message never finished", done.await(60, TimeUnit.SECONDS))
                val message = llm.getContextTokenCount()
                Thread.sleep(1_000)
                val after = llm.getContextTokenCount()
                println("[android-int][reset-count] read held ${if (afterEngine) "after" else "before"} the engine, let go mid-decode: " +
                    "the engine answered it in ${"%.1f".format(hold.engineMs)} ms; count while generating $generating, " +
                    "after the message $message, 1 s later $after")
                assertTrue("the message finished before the check", stillGenerating)
                assertEquals("the count moved while the message generated", -1.0, generating, 0.0)
                assertTrue("no count after the message: $message", message > 0)
                assertEquals("the read changed the message's count", message, after, 0.0)
                llm.readContextTokens = read
            }

            // A second reset while the first read holds its claim, let go during the
            // reset's 2 s wait: the reset waits for it, and the second count lands.
            run {
                val hold = HeldRead(llm, 1, afterEngine = true)
                llm.resetConversation(null, null, null)
                assertTrue("the read never ran", hold.reached.await(10, TimeUnit.SECONDS))
                val letGo = thread { Thread.sleep(500); hold.release.countDown() }
                ms("resetConversation while the first read is held for 0.5 s") { llm.resetConversation(null, null, null) }
                letGo.join()
                val second = knownContextTokens(llm)
                Thread.sleep(1_000)
                val after = llm.getContextTokenCount()
                println("[android-int][reset-count] second reset: $second, 1 s later $after")
                assertTrue("the second reset's count never landed: $second", second >= 0)
                assertEquals("the first read changed the second reset's count", second, after, 0.0)
                llm.readContextTokens = read
            }

            // A second reset while the first read holds its claim past the reset's
            // 2 s wait, as a cancelled message that hasn't returned would: no read
            // starts while that claim is out, so the count stays -1, also once the
            // first read has returned, and the next message records its own count.
            run {
                val hold = HeldRead(llm, 1, afterEngine = true)
                llm.resetConversation(null, null, null)
                assertTrue("the read never ran", hold.reached.await(10, TimeUnit.SECONDS))
                ms("resetConversation while the first read is held past its wait") { llm.resetConversation(null, null, null) }
                Thread.sleep(1_000)
                val whileHeld = llm.getContextTokenCount()
                hold.release.countDown()
                Thread.sleep(1_000)
                val returned = llm.getContextTokenCount()
                generate(llm, "Say hello.", short)
                val message = llm.getContextTokenCount()
                println("[android-int][reset-count] second reset, first read held past its wait: count while held $whileHeld, " +
                    "after the first read returned $returned, after the next message $message, reads ${hold.calls.get()}")
                assertEquals("a read started while the first read held its claim", 1, hold.calls.get())
                assertEquals("a count while the first read held its claim", -1.0, whileHeld, 0.0)
                assertEquals("a count after the first read returned", -1.0, returned, 0.0)
                assertTrue("no count after the next message: $message", message > 0)
                llm.readContextTokens = read
            }
        } finally {
            llm.close()
        }
        Teardowns.await(Teardowns.mark())

        // close() during the read: not ready at once, and the engine is freed
        // only after the read has returned.
        val closing = HybridLiteRTLM()
        closing.loadModelBlocking(model.absolutePath, llmConfig())
        val heldForClose = HeldRead(closing, 1, afterEngine = false)
        closing.resetConversation(null, null, null)
        assertTrue("the read never ran", heldForClose.reached.await(10, TimeUnit.SECONDS))
        assertTrue(ms("close during the read") { closing.close() } < 50.0)
        assertFalse(closing.isReady())
        val ticket = Teardowns.mark()
        val freed = CountDownLatch(1)
        thread { Teardowns.await(ticket); freed.countDown() }
        assertFalse("close() freed the engine under the read", freed.await(2, TimeUnit.SECONDS))
        heldForClose.release.countDown()
        assertTrue("the engine was never freed", freed.await(30, TimeUnit.SECONDS))
        assertEquals(-1.0, closing.getContextTokenCount(), 0.0)

        // unload() during the read: it settles only after the read has returned.
        val unloading = HybridLiteRTLM()
        unloading.loadModelBlocking(model.absolutePath, llmConfig())
        val heldForUnload = HeldRead(unloading, 1, afterEngine = false)
        unloading.resetConversation(null, null, null)
        assertTrue("the read never ran", heldForUnload.reached.await(10, TimeUnit.SECONDS))
        val settled = CountDownLatch(1)
        unloading.unload().then { settled.countDown() }.catch { settled.countDown() }
        assertFalse(unloading.isReady())
        assertFalse("unload() freed the engine under the read", settled.await(2, TimeUnit.SECONDS))
        heldForUnload.release.countDown()
        assertTrue("unload() never settled", settled.await(30, TimeUnit.SECONDS))
        assertEquals(-1.0, unloading.getContextTokenCount(), 0.0)
        unloading.close()
        Teardowns.await(Teardowns.mark())
    }

    /** numThreads reaches the CPU backend (LiteRT-LM 0.18's Backend.CPU(threadCount)); speed reported. */
    @Test
    fun testK_cpuThreadCount() {
        val model = modelFile()
        assumeTrue("No .litertlm pushed — skipping", model != null)
        ensureNativeInit()
        releaseSharedEngine()
        fun decodeTps(threads: Double?): Double {
            val llm = HybridLiteRTLM()
            try {
                llm.loadModelBlocking(model!!.absolutePath, llmConfig(backend = Backend.CPU, numThreads = threads))
                generate(llm, "Hi", options(maxOutputTokens = 4.0)) // warm-up
                llm.resetConversation(null, null, null)
                val (text, _) = generate(
                    llm, "Write a detailed paragraph about the history of the bicycle.", options(maxOutputTokens = 64.0),
                )
                assertTrue("generated nothing with numThreads=$threads", text.isNotBlank())
                return llm.getStats().tokensPerSecond
            } finally {
                llm.close()
            }
        }
        val one = decodeTps(1.0)
        val default = decodeTps(null)
        println("[android-int][threads] decode ${"%.2f".format(one)} tok/s on 1 thread, ${"%.2f".format(default)} tok/s default")
    }

    /** Speculative decoding on for one conversation (Gemma 4 E2B ships an MTP drafter); speed reported. */
    @Test
    fun testS_speculativeDecodingPerConversation() {
        val llm = engine(withTools = false)
        val prompt = "Write a detailed paragraph about the history of the bicycle."
        llm.resetConversation(null, null, null)
        val (plain, _) = generate(llm, prompt, options(maxOutputTokens = 64.0))
        val plainTps = llm.getStats().tokensPerSecond
        llm.resetConversation(null, null, true)
        val (fast, _) = generate(llm, prompt, options(maxOutputTokens = 64.0))
        val fastTps = llm.getStats().tokensPerSecond
        println("[android-int][mtp] decode ${"%.2f".format(plainTps)} tok/s default, ${"%.2f".format(fastTps)} tok/s speculative")
        println("[android-int][mtp] speculative text: ${fast.take(120)}")
        assertTrue(plain.isNotBlank())
        assertTrue(fast.isNotBlank())
        llm.resetConversation(null, null, null)
    }

    /**
     * Visual-token budgets: a per-image cap at load and a per-message budget,
     * both through the SDK's process-wide flag, which must be restored after
     * every use so no other engine or message inherits it.
     */
    @OptIn(com.google.ai.edge.litertlm.ExperimentalApi::class)
    @Test
    fun testV_visualTokenBudgets() {
        val model = modelFile()
        val image = fixture("test.jpeg")
        assumeTrue("model or test.jpeg not pushed — skipping", model != null && image != null)
        ensureNativeInit()
        releaseSharedEngine()
        val llm = HybridLiteRTLM()
        try {
            llm.loadModelBlocking(
                model!!.absolutePath,
                llmConfig(backend = Backend.CPU, multimodal = true, maxVisionTokensPerImage = 280.0),
            )
            assertNull("the load must restore the global budget", com.google.ai.edge.litertlm.ExperimentalFlags.visualTokenBudget)
            assumeTrue("this bundle has no vision encoder", llm.hasVisionEncoder)
            fun describe(budget: Double?): Pair<String, Long> {
                llm.resetConversation(null, null, null)
                val start = System.nanoTime()
                val (text, _) = generateParts(
                    llm,
                    arrayOf(textPart("Describe this picture in one sentence."), mediaPart(PartType.IMAGE, image!!)),
                    options(maxOutputTokens = 48.0, visualTokenBudget = budget),
                )
                return text to (System.nanoTime() - start) / 1_000_000
            }
            val (small, smallMs) = describe(70.0)
            val (capped, cappedMs) = describe(null)
            println("[android-int][vision] budget 70: $smallMs ms — $small")
            println("[android-int][vision] cap 280: $cappedMs ms — $capped")
            assertTrue(small.isNotBlank())
            assertTrue(capped.isNotBlank())
            assertNull("a send must restore the global budget", com.google.ai.edge.litertlm.ExperimentalFlags.visualTokenBudget)
        } finally {
            llm.close()
        }
    }

    // ── Typed failures from the real SDK (src/core/nativeError.ts decodes these) ──

    /** Run a load that must fail; returns the typed exception it threw. */
    private fun failedLoad(path: String, config: LLMConfig? = null): LiteRTLMNativeException {
        ensureNativeInit()
        val llm = HybridLiteRTLM()
        try {
            llm.loadModelBlocking(path, config)
        } catch (e: LiteRTLMNativeException) {
            println("[android-int][errors] ${e.message}")
            assertFalse("a failed load must leave the instance unloaded", llm.isReady())
            return e
        } finally {
            llm.close()
        }
        throw AssertionError("loading $path should have failed")
    }

    @Test
    fun testE1_missingFileIsModelNotFound() {
        val e = failedLoad("/data/local/tmp/does-not-exist.litertlm")
        assertEquals("load", e.phase)
        assertEquals("NOT_FOUND", e.status)
        assertEquals("MODEL_NOT_FOUND", e.code)
    }

    /** The SDK's own status must reach JS, not a generic "failed to load". */
    @Test
    fun testE2_invalidBundleCarriesTheSdkStatus() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val junk = File(context.cacheDir, "not_a_model.litertlm").apply { writeBytes(ByteArray(4096) { 0x42 }) }
        try {
            val e = failedLoad(junk.absolutePath)
            assertEquals("load", e.phase)
            assertTrue("SDK status was not captured: ${e.message}", e.status != "UNKNOWN")
            assertTrue("missing fallback context: ${e.message}", e.detail.contains("then the CPU fallbacks"))
        } finally {
            junk.delete()
        }
    }

    /**
     * activationDataType reaches the engine (EngineConfig, new on Android in
     * LiteRT-LM 0.18.0): greedy output under INT8 activations must differ from
     * FLOAT32. The CPU backend accepts INT8 for this quantized bundle rather
     * than rejecting it, so a numerics difference is the observable signal.
     * Skips — rather than fails — if the outputs coincide, since then the run
     * proves nothing either way. Frees the suite's shared engine first.
     */
    @Test
    fun testE3_activationDataTypeReachesTheEngine() {
        val model = modelFile()
        assumeTrue("No .litertlm — skipping", model != null)
        bridge?.close()
        bridge = null
        fun greedy(adt: ActivationDataType): String {
            ensureNativeInit()
            val llm = HybridLiteRTLM()
            try {
                llm.loadModelBlocking(
                    model!!.absolutePath,
                    llmConfig(backend = Backend.CPU, activationDataType = adt),
                )
                return generate(llm, "Write one sentence about rivers.", options(maxOutputTokens = 32.0)).first
            } finally {
                llm.close()
            }
        }
        val f32 = greedy(ActivationDataType.F32)
        val i8 = greedy(ActivationDataType.I8)
        println("[android-int][adt] f32: $f32")
        println("[android-int][adt] i8:  $i8")
        assertFalse("FLOAT32 generation was empty", f32.isBlank())
        assumeTrue("identical output under F32 and I8 — cannot tell whether the type was applied", f32 != i8)
    }

    /** Await a Nitro Promise from a test thread; returns its value or failure. */
    private fun <T> awaitResult(promise: com.margelo.nitro.core.Promise<T>): Result<T> {
        val latch = CountDownLatch(1)
        var result: Result<T>? = null
        promise
            .then { result = Result.success(it); latch.countDown() }
            .catch { result = Result.failure(it); latch.countDown() }
        assertTrue("promise did not settle", latch.await(GENERATE_TIMEOUT_SEC, TimeUnit.SECONDS))
        return result!!
    }

    private fun mediaPart(type: PartType, file: File) =
        MultimodalPart(type, null, file.absolutePath, null, null, null, null)

    // ── Bundle model info + encoders (LiteRT-LM 0.18) ───────────────────────

    @Test
    fun testI_inspectRealBundle() {
        val model = modelFile()
        assumeTrue("No .litertlm — skipping", model != null)
        ensureNativeInit()
        val start = System.nanoTime()
        val info = ModelBundle.read(model!!.absolutePath)
        val ms = (System.nanoTime() - start) / 1e6
        println(
            "[android-int][info] inspected in ${ms.toInt()} ms: type=${info.modelType} ctx=${info.maxContextTokens.toInt()} " +
                "dynamic=${info.isDynamicContext} mtp=${info.hasSpeculativeDecoding} " +
                "vision=${info.maxVisionTokenBudget.toInt()} ${info.visionSignatureSelection.map { it.toInt() }}",
        )
        info.modalities.forEach {
            println("[android-int][info]   ${it.modality}=${it.supported} backends=${it.backends.toList()} npu=${it.npuBrand}")
        }
        assertEquals(ModelType.LLM, info.modelType)
        assertTrue(info.modalities.first { it.modality == Modality.TEXT }.supported)
        val device = DeviceFacts.current()
        println("[android-int][info] device: ${device.model} Android ${device.osVersion} soc=${device.socModel} gpu=${device.gpuAvailable} emulator=${device.isEmulator}")
    }

    /**
     * Encoders are opt-in and follow the bundle: a text-only engine refuses
     * media with a typed error; a multimodal one wires exactly the bundle's
     * encoders and answers about a real image and audio clip. Frees the
     * suite's shared engine first.
     */
    @Test
    fun testM_multimodalEncodersFollowTheBundle() {
        val model = modelFile()
        assumeTrue("No .litertlm — skipping", model != null)
        bridge?.close()
        bridge = null
        ensureNativeInit()
        val path = model!!.absolutePath
        val plan = ModelBundle.encoderPlan(path)!!
        val bundleVision = plan.vision != null
        val bundleAudio = plan.audio != null
        println("[android-int][encoders] bundle plan: vision=${plan.vision} audio=${plan.audio}")
        fun config(multimodal: Boolean) = llmConfig(backend = Backend.CPU, multimodal = multimodal)

        val textOnly = HybridLiteRTLM()
        try {
            var t0 = System.nanoTime()
            textOnly.loadModelBlocking(path, config(multimodal = false))
            println("[android-int][encoders] text-only load ${"%.1f".format((System.nanoTime() - t0) / 1e9)}s")
            assertFalse(textOnly.hasVisionEncoder)
            assertFalse(textOnly.hasAudioEncoder)
            val refused = awaitResult(
                textOnly.execute(
                    arrayOf(textPart("What is in this picture?"), MultimodalPart(PartType.IMAGE, null, "/x.jpg", null, null, null, null)),
                    null, null,
                ),
            ).exceptionOrNull()
            assertTrue("expected a typed refusal, got $refused", refused is LiteRTLMNativeException)
            assertEquals("UNSUPPORTED", (refused as LiteRTLMNativeException).code)
        } finally {
            textOnly.close()
        }

        val multi = HybridLiteRTLM()
        try {
            val t0 = System.nanoTime()
            multi.loadModelBlocking(path, config(multimodal = true))
            println("[android-int][encoders] multimodal load ${"%.1f".format((System.nanoTime() - t0) / 1e9)}s")
            assertEquals(bundleVision, multi.hasVisionEncoder)
            assertEquals(bundleAudio, multi.hasAudioEncoder)
            val image = fixture("test.jpeg")
            if (bundleVision && image != null) {
                val turn = System.nanoTime()
                val reply = awaitResult(
                    multi.execute(
                        arrayOf(textPart("Describe this picture in one short sentence."), mediaPart(PartType.IMAGE, image)),
                        null, options(maxOutputTokens = 48.0),
                    ),
                ).getOrThrow()
                println("[android-int][encoders] image (${"%.1f".format((System.nanoTime() - turn) / 1e9)}s): $reply")
                assertFalse(reply.isBlank())
            }
            val audio = fixture("test.wav")
            if (bundleAudio && audio != null) {
                multi.resetConversation(null, null, null)
                val turn = System.nanoTime()
                val reply = awaitResult(
                    multi.execute(
                        arrayOf(textPart("What do you hear? One short sentence."), mediaPart(PartType.AUDIO, audio)),
                        null, options(maxOutputTokens = 48.0),
                    ),
                ).getOrThrow()
                println("[android-int][encoders] audio (${"%.1f".format((System.nanoTime() - turn) / 1e9)}s): $reply")
                assertFalse(reply.isBlank())
            }
        } finally {
            multi.close()
        }
    }

    // ── Two LIVE conversations on one engine (upstream #2807) ────────────────

    /**
     * Drives the LiteRT-LM Kotlin SDK directly — two `Conversation`s alive at once
     * on a single `Engine`, used in an interleaved (not concurrent) pattern. This
     * is the behavior google-ai-edge/LiteRT-LM#2807 reported broken and reports
     * fixed in 0.15.0 on desktop; the wrapper's `createConversation()` avoids
     * relying on it by replaying transcripts. If this passes on a phone, the
     * replay (and its re-prefill cost per switch) can be dropped.
     *
     * Named to sort last: it builds a second engine, so the shared suite engine is
     * closed first rather than holding two multi-GB engines at once.
     */
    @Test
    fun testZ_interleavedLiveConversationsShareOneEngine() {
        val model = modelFile()
        assumeTrue("No .litertlm — skipping", model != null)
        JNIOnLoad.initializeNativeNitro()

        // Free the suite's shared engine first.
        bridge?.close()
        bridge = null

        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val engine = Engine(
            EngineConfig(
                modelPath = model!!.absolutePath,
                backend = com.google.ai.edge.litertlm.Backend.CPU(),
                maxNumTokens = 2048,
                cacheDir = context.cacheDir.absolutePath,
            )
        )
        try {
            engine.initialize()
            val greedy = SamplerConfig(topK = 1, topP = 1.0, temperature = 0.0)
            fun conversation() = engine.createConversation(
                ConversationConfig(
                    samplerConfig = greedy,
                    automaticToolCalling = false,
                    maxOutputToken = 32,
                )
            )

            val convA = conversation()
            // If the SDK enforced one conversation per engine, this is where it fails.
            val convB = conversation()

            fun send(conversation: Conversation, text: String): String =
                conversation.sendMessage(SdkMessage.user(text))
                    .contents.contents
                    .filterIsInstance<Content.Text>()
                    .joinToString("") { it.text }

            try {
                val ackA = send(convA, "My lucky number is 47. Acknowledge briefly.")
                println("[android-int][2807] A ack: $ackA")

                val ackB = send(convB, "My favorite animal is the axolotl. Acknowledge briefly.")
                println("[android-int][2807] B ack: $ackB")
                val answerB = send(convB, "What is my favorite animal? One word.")
                println("[android-int][2807] B recall: $answerB")

                // B must actually have advanced, or A's recall proves nothing.
                assertTrue(
                    "conversation B did not carry its own turn — test would be vacuous: $answerB",
                    answerB.lowercase().contains("axolotl"),
                )

                val answerA = send(convA, "What is my lucky number? Reply with just the number.")
                println("[android-int][2807] A recall: $answerA")
                assertTrue(
                    "INTERLEAVED RECALL FAILED — conversation A lost its state after B " +
                        "generated. Got: $answerA",
                    answerA.contains("47"),
                )
            } finally {
                runCatching { convA.close() }
                runCatching { convB.close() }
            }
        } finally {
            runCatching { engine.close() }
        }
    }
}
