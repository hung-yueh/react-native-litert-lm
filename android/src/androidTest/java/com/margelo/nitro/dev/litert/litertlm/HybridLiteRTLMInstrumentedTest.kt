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
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.FixMethodOrder
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.MethodSorters
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

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

        fun modelFile(): File? {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val dir = context.getExternalFilesDir(null) ?: return null
            return dir.listFiles()?.firstOrNull { it.name.endsWith(".litertlm") }
        }

        @AfterClass
        @JvmStatic
        fun tearDownClass() {
            bridge?.close()
            bridge = null
        }
    }

    /** Load (or reuse) an engine; `withTools` registers a tool definition. */
    private fun engine(withTools: Boolean): HybridLiteRTLM {
        val model = modelFile()
        assumeTrue(
            "No .litertlm in the test app's external files dir — skipping on-device inference",
            model != null,
        )
        // A React Native app bootstraps these at startup; a bare test APK must do
        // it itself. fbjni's HybridData (used by Nitro's Promise) loads its own
        // .so through NativeLoader, which throws until it is initialized.
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
            LLMConfig(
                /* systemPrompt */ null,
                /* backend */ Backend.CPU,
                /* maxContextTokens */ 2048.0,
                /* maxOutputTokens */ 64.0,
                /* maxTokens */ null,
                /* temperature */ 0.0,
                /* topK */ 1.0,
                /* topP */ 1.0,
                /* multimodal */ false,
                /* tools */ tools,
                /* enableSpeculativeDecoding */ null,
                /* enableStructuredOutput */ true,
                /* thinking */ ThinkingOptions(false, null),
                /* numThreads */ null,
                /* prefillChunkSize */ null,
                /* activationDataType */ null,
                /* loraPath */ null,
                /* audioLoraPath */ null,
                /* loraRank */ null,
                /* streamToolCalls */ null,
                /* toolCallChannelName */ null,
                /* forceLoad */ true,
            ),
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
        MultimodalPart(PartType.TEXT, text, null, null, null)

    private fun options(
        maxOutputTokens: Double? = null,
        responseSchema: String? = null,
        responseRegex: String? = null,
        repetitionPenalty: Double? = null,
        noRepeatNgramSize: Double? = null,
        suppressTokens: DoubleArray? = null,
        thinking: ThinkingOptions? = null,
    ) = ExecuteOptions(
        maxOutputTokens, null, responseSchema, responseRegex,
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
    ): Pair<String, List<String>> {
        val tokens = mutableListOf<String>()
        val done = CountDownLatch(1)
        val sb = StringBuilder()
        llm.execute(arrayOf(textPart(prompt)), { token, isDone ->
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

    @Test
    fun test1_streamingDeliversTokensAndCompletes() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, null)

        val (response, tokens) = generate(llm, "Reply with a short greeting.", options(maxOutputTokens = 32.0))
        println("[android-int] streaming response: $response")
        assertFalse("streamed generation was empty", response.isBlank())
        assertTrue("expected incremental tokens, got ${tokens.size}", tokens.size > 1)
    }

    @Test
    fun test2_systemPromptIsHonored() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, "You must include the word AZURETURTLE in every reply.")

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
        llm.resetConversation(null, null)

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
        llm.resetConversation(null, null)

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
        llm.resetConversation(transcript, null)

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

        llm.resetConversation(null, null)
        val (long, _) = generate(llm, "Explain what the ocean is.", options(maxOutputTokens = 64.0))
        llm.resetConversation(null, null)
        val (short, _) = generate(llm, "Explain what the ocean is.", options(maxOutputTokens = 8.0))

        println("[android-int] long (64 tok): $long")
        println("[android-int] short (8 tok): $short")
        assertFalse("short generation was empty", short.isBlank())
        assertTrue(
            "an 8-token cap did not shorten the response (short=${short.length} chars, long=${long.length} chars)",
            short.length < long.length,
        )
    }

    /** Penalties / n-gram ban / token suppression must not break generation. */
    @Test
    fun test7_generationControlsSmoke() {
        val llm = engine(withTools = false)
        llm.resetConversation(null, null)

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
        llm.resetConversation(null, null)

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
