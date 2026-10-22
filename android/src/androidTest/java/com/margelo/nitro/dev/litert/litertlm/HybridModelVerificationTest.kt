package com.margelo.nitro.dev.litert.litertlm

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import com.margelo.nitro.JNIOnLoad
import com.margelo.nitro.core.Promise
import dev.litert.litertlm.LiteRTLMInitProvider
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The on-device battery a non-Gemma model passes before it gets a `Models`
 * constant — the Android counterpart of ios/Tests/HybridModelVerificationTests.swift.
 * Facts, multi-turn memory, streaming, a system prompt and structured output
 * must work; tool calls and thinking are reported.
 *
 * Skipped unless a bundle sits in the test app's `files/verify/` folder (kept
 * apart from the Gemma suite's model) and is named on the command line:
 *
 *   adb shell mkdir -p /sdcard/Android/data/dev.litert.litertlm.test/files/verify
 *   adb push Qwen3.5-2B_int8.litertlm /sdcard/Android/data/dev.litert.litertlm.test/files/verify/
 *   adb shell am instrument -w -r -e verifyModel Qwen3.5-2B_int8.litertlm \
 *     -e class com.margelo.nitro.dev.litert.litertlm.HybridModelVerificationTest \
 *     dev.litert.litertlm.test/androidx.test.runner.AndroidJUnitRunner
 */
@RunWith(AndroidJUnit4::class)
class HybridModelVerificationTest {

    private fun modelPath(): String? {
        val name = InstrumentationRegistry.getArguments().getString("verifyModel") ?: return null
        val dir = InstrumentationRegistry.getInstrumentation().targetContext.getExternalFilesDir(null) ?: return null
        return File(File(dir, "verify"), name).takeIf { it.exists() }?.absolutePath
    }

    private fun ensureNativeInit() {
        if (!NativeLoader.isInitialized()) NativeLoader.init(SystemDelegate())
        JNIOnLoad.initializeNativeNitro()
        if (LiteRTLMInitProvider.applicationContext == null) {
            val field = LiteRTLMInitProvider::class.java.getDeclaredField("applicationContext")
            field.isAccessible = true
            field.set(null, InstrumentationRegistry.getInstrumentation().targetContext)
        }
    }

    private fun config(tools: Array<ToolDefinition>? = null) = LLMConfig(
        systemPrompt = null, backend = Backend.CPU, maxContextTokens = 2048.0, maxOutputTokens = 256.0,
        maxTokens = null, temperature = 0.0, topK = 1.0, topP = 1.0, multimodal = false, tools = tools,
        enableSpeculativeDecoding = null, chatTemplate = null, enableStructuredOutput = true, thinking = null,
        numThreads = null, maxVisionTokensPerImage = null, gpu = null, singleThreadedExecution = null,
        prefillChunkSize = null, activationDataType = null, loraPath = null, audioLoraPath = null, loraRank = null,
        streamToolCalls = null, toolCallChannelName = null, forceLoad = true,
    )

    private fun options(maxOutputTokens: Double? = null, responseSchema: String? = null) = ExecuteOptions(
        maxOutputTokens, null, responseSchema, null, null, null, null, null, null, null, null, null,
    )

    private fun text(s: String) = MultimodalPart(PartType.TEXT, s, null, null, null, null, null)

    /** One generation through the streaming callback; returns the text and the chunk count. */
    private fun run(llm: HybridLiteRTLM, parts: Array<MultimodalPart>, opts: ExecuteOptions? = options()): Pair<String, Int> {
        val done = CountDownLatch(1)
        val sb = StringBuilder()
        var chunks = 0
        llm.execute(parts, { token, isDone ->
            sb.append(token)
            if (isDone) done.countDown() else chunks++
        }, opts)
        assertTrue("generation did not finish", done.await(300, TimeUnit.SECONDS))
        return sb.toString() to chunks
    }

    private fun ask(llm: HybridLiteRTLM, prompt: String, opts: ExecuteOptions? = options()) =
        run(llm, arrayOf(text(prompt)), opts).first

    /** Wait for a Nitro Promise on the test thread; returns its value or throws its error. */
    private fun <T> await(promise: Promise<T>): T {
        val settled = CountDownLatch(1)
        var result: Result<T>? = null
        promise
            .then { result = Result.success(it); settled.countDown() }
            .catch { result = Result.failure(it); settled.countDown() }
        assertTrue("promise did not settle", settled.await(300, TimeUnit.SECONDS))
        return result!!.getOrThrow()
    }

    /** The answer with any reasoning block removed, for content checks. */
    private fun answer(reply: String) = reply.substringAfter("</think>")

    @Test
    fun verifyModel() {
        val path = modelPath()
        assumeTrue("pass -e verifyModel <file> with the bundle in files/verify/ — skipping", path != null)
        ensureNativeInit()
        val name = File(path!!).name

        val llm = HybridLiteRTLM()
        try {
            var started = System.nanoTime()
            llm.loadModelBlocking(path, config())
            println("[verify] $name loaded in ${"%.1f".format((System.nanoTime() - started) / 1e9)}s")
            val loaded = llm.getContextTokenCount()

            val capital = ask(llm, "What is the capital of France? Answer with just the city name.")
            val streamed = llm.getContextTokenCount()
            println("[verify] capital: $capital")
            assertTrue("facts: $capital", answer(capital).lowercase().contains("paris"))

            // Context token count: grows with a streamed and a blocking message
            await(llm.execute(arrayOf(text("And of Italy?")), null, options(maxOutputTokens = 16.0)))
            val blocking = llm.getContextTokenCount()
            println("[verify] context tokens: loaded $loaded, streamed $streamed, blocking $blocking")
            assertTrue("context tokens after load: $loaded", loaded >= 0)
            assertTrue("context tokens after a streamed message: $loaded -> $streamed", streamed > loaded)
            assertTrue("context tokens after a blocking message: $streamed -> $blocking", blocking > streamed)

            // Multi-turn memory, from a fresh conversation (a one-word answer just
            // before primes small models to keep answering in one word)
            llm.resetConversation(null, null, null)
            assertEquals("context tokens after a reset", -1.0, llm.getContextTokenCount(), 0.0)
            ask(llm, "My favorite color is teal. Please remember it.", options(maxOutputTokens = 48.0))
            val afterReset = llm.getContextTokenCount()
            println("[verify] context tokens after the reset's first message: $afterReset")
            assertTrue("context tokens after the reset's first message: $afterReset", afterReset > 0)
            val color = ask(llm, "What is my favorite color? Answer with one word.")
            println("[verify] recall: $color")
            assertTrue("multi-turn: $color", answer(color).lowercase().contains("teal"))

            llm.resetConversation(null, null, null)
            started = System.nanoTime()
            val (story, chunks) = run(llm, arrayOf(text("Write two sentences about the ocean.")), options(maxOutputTokens = 96.0))
            println("[verify] streamed $chunks chunks in ${"%.1f".format((System.nanoTime() - started) / 1e9)}s: ${story.take(160)}")
            assertTrue("streaming", chunks > 1)

            llm.resetConversation(null, "You are a pirate. Begin every reply with the word Arr.", null)
            val pirate = ask(llm, "Say hello to me.", options(maxOutputTokens = 48.0))
            println("[verify] system prompt: $pirate")
            assertTrue("system prompt: $pirate", answer(pirate).lowercase().contains("arr"))

            llm.resetConversation(null, null, null)
            val schema = """{"type":"object","properties":{"city":{"type":"string"},"country":{"type":"string"}},"required":["city","country"]}"""
            val json = ask(llm, "Name one city in Japan and its country.", options(maxOutputTokens = 64.0, responseSchema = schema))
            println("[verify] structured: $json")
            assertNotNull("structured output: $json", runCatching { JSONObject(answer(json)).getString("city") }.getOrNull())

            llm.resetConversation(null, null, null)
            val math = ask(
                llm, "I have 3 apples, buy 5 more, then eat 2. How many apples do I have? End with the number.",
                options(maxOutputTokens = 256.0),
            )
            println("[verify] reasoning (think block: ${math.contains("<think>")}): ${math.take(300)}")
            assertTrue("arithmetic: $math", answer(math).contains("6"))

            val unloading = llm.unload()
            assertEquals("context tokens after unload()", -1.0, llm.getContextTokenCount(), 0.0)
            await(unloading)
        } finally {
            llm.close()
        }

        // Tools: reported, not gated
        val weather = ToolDefinition(
            "get_current_weather", "Get the current weather for a location.",
            """{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}""",
        )
        val agent = HybridLiteRTLM()
        try {
            agent.loadModelBlocking(path, config(tools = arrayOf(weather)))
            val call = ask(agent, "What is the weather in Paris right now? Use the tool.")
            println("[verify] tool call: $call")
            if (call.contains("<tool_call>")) {
                val result = MultimodalPart(
                    PartType.TOOLRESULT, null, null, null, null, "get_current_weather",
                    """{"location":"Paris","temperature_c":27,"condition":"hailstorm"}""",
                )
                val (reply, _) = run(agent, arrayOf(result))
                println("[verify] tool answer: $reply")
                println("[verify] TOOLS: ${if (reply.contains("27") || reply.lowercase().contains("hail")) "round trip OK" else "result ignored"}")
            } else {
                println("[verify] TOOLS: no tool call emitted")
            }
        } finally {
            agent.close()
        }
    }
}
