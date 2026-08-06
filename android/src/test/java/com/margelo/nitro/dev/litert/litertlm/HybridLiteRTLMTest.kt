package com.margelo.nitro.dev.litert.litertlm

import org.junit.Assert.*
import org.junit.Before
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import dev.litert.litertlm.LiteRTLMInitProvider
import java.lang.IllegalArgumentException

@RunWith(RobolectricTestRunner::class)
class HybridLiteRTLMTest {
    private lateinit var bridge: HybridLiteRTLM

    @Before
    fun setUp() {
        // Initialize the static applicationContext inside LiteRTLMInitProvider via reflection
        try {
            val field = LiteRTLMInitProvider::class.java.getDeclaredField("applicationContext")
            field.isAccessible = true
            field.set(null, RuntimeEnvironment.getApplication())
        } catch (e: Exception) {
            e.printStackTrace()
        }
        
        bridge = HybridLiteRTLM()
    }

    @After
    fun tearDown() {
        bridge.close()
    }

    @Test
    fun testAndroidPathTraversalPrevention() {
        val traversals = arrayOf("../secret", "/etc/hosts", "nested\\..\\file", "..", "../", "..\\")
        for (traversal in traversals) {
            val promise = bridge.deleteModel(traversal)
            assertNotNull("Promise should not be null", promise)
            assertTrue("Promise should be completed", promise.isCompleted)
            assertNotNull("Promise should have rejected with an error for filename: $traversal", promise.error)
            val error = promise.error!!
            val errMsg = error.message ?: error.cause?.message ?: ""
            assertTrue("Expected message to contain traversal warning, got: $errMsg",
                errMsg.contains("path traversal or directory separators are not allowed"))
        }
    }

    @Test
    fun testAndroidHTTPSDownloadEnforcement() {
        val promise = bridge.downloadModel("http://insecure.site/model.bin", "model.bin", null)
        assertNotNull("Promise should not be null", promise)
        assertTrue("Promise should be completed", promise.isCompleted)
        assertNotNull("Promise should have rejected with an error", promise.error)
        val error = promise.error!!
        val errMsg = error.message ?: error.cause?.message ?: ""
        assertTrue("Expected message to contain HTTPS warning, got: $errMsg",
            errMsg.contains("HTTPS is required for security"))
    }

    @Test
    fun testAndroidMemoryTelemetry() {
        val mem = bridge.getMemoryUsage()
        assertNotNull(mem)
        assertTrue(mem.nativeHeapBytes >= 0.0)
        assertTrue(mem.residentBytes >= 0.0)
        assertTrue(mem.availableMemoryBytes >= 0.0)
    }

    @Test
    fun testSendMessageWithImageAsyncRejectsWithoutModel() {
        val promise = bridge.sendMessageWithImageAsync("hello", "/tmp/image.jpg") { _, _ -> }
        assertNotNull("Promise should not be null", promise)
        assertTrue("Promise should be completed", promise.isCompleted)
        assertNotNull("Promise should have rejected without model", promise.error)
        val errMsg = promise.error!!.message ?: promise.error!!.cause?.message ?: ""
        assertTrue("Expected no-model error, got: $errMsg",
            errMsg.contains("No model loaded"))
    }

    @Test
    fun testSendMessageWithAudioAsyncRejectsWithoutModel() {
        val promise = bridge.sendMessageWithAudioAsync("hello", "/tmp/audio.wav") { _, _ -> }
        assertNotNull("Promise should not be null", promise)
        assertTrue("Promise should be completed", promise.isCompleted)
        assertNotNull("Promise should have rejected without model", promise.error)
        val errMsg = promise.error!!.message ?: promise.error!!.cause?.message ?: ""
        assertTrue("Expected no-model error, got: $errMsg",
            errMsg.contains("No model loaded"))
    }

    @Test
    fun testAndroidInitialStats() {
        val stats = bridge.getStats()
        assertNotNull(stats)
        assertEquals(0.0, stats.promptTokens, 0.0)
        assertEquals(0.0, stats.completionTokens, 0.0)
        assertEquals(0.0, stats.totalTokens, 0.0)
        assertEquals(0.0, stats.timeToFirstToken, 0.0)
        assertEquals(0.0, stats.totalTime, 0.0)
        assertEquals(0.0, stats.tokensPerSecond, 0.0)
    }

    @Test
    fun testDeleteModelCleanupLogic() {
        val loadedPathField = HybridLiteRTLM::class.java.getDeclaredField("loadedModelPath")
        loadedPathField.isAccessible = true
        loadedPathField.set(bridge, "/path/to/my_loaded_model.litertlm")

        val promise1 = bridge.deleteModel("other_model.litertlm")
        assertNotNull(promise1)
        while (!promise1.isCompleted) { Thread.sleep(10) }
        assertEquals("/path/to/my_loaded_model.litertlm", loadedPathField.get(bridge))

        val promise2 = bridge.deleteModel("my_loaded_model.litertlm")
        assertNotNull(promise2)
        while (!promise2.isCompleted) { Thread.sleep(10) }
        assertNull(loadedPathField.get(bridge))
    }

    // ── Memory pressure: onTrimMemory → MemoryWarningLevel mapping ──────────

    private fun registeredCallbacks(): android.content.ComponentCallbacks2 {
        val field = HybridLiteRTLM::class.java.getDeclaredField("componentCallbacks")
        field.isAccessible = true
        return field.get(bridge) as android.content.ComponentCallbacks2
    }

    @Test
    fun testTrimMemoryLevelMapping() {
        val received = mutableListOf<MemoryWarningLevel>()
        bridge.setMemoryWarningCallback { level, usage ->
            assertTrue(usage.residentBytes >= 0.0)
            received.add(level)
        }
        val callbacks = registeredCallbacks()

        // TRIM_MEMORY_RUNNING_MODERATE (5) / RUNNING_LOW (10) → MODERATE
        callbacks.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE)
        callbacks.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW)
        // RUNNING_CRITICAL (15) and COMPLETE (80) → CRITICAL
        callbacks.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL)
        callbacks.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
        // Below every threshold → no event
        callbacks.onTrimMemory(0)

        assertEquals(
            listOf(
                MemoryWarningLevel.MODERATE,
                MemoryWarningLevel.MODERATE,
                MemoryWarningLevel.CRITICAL,
                MemoryWarningLevel.CRITICAL,
            ),
            received,
        )
    }

    @Test
    fun testLowMemoryMapsToCritical() {
        val received = mutableListOf<MemoryWarningLevel>()
        bridge.setMemoryWarningCallback { level, _ -> received.add(level) }
        @Suppress("DEPRECATION")
        registeredCallbacks().onLowMemory()
        assertEquals(listOf(MemoryWarningLevel.CRITICAL), received)
    }

    @Test
    fun testClearMemoryWarningCallbackStopsEvents() {
        val received = mutableListOf<MemoryWarningLevel>()
        bridge.setMemoryWarningCallback { level, _ -> received.add(level) }
        val callbacks = registeredCallbacks()
        bridge.clearMemoryWarningCallback()
        callbacks.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
        assertTrue(received.isEmpty())
    }

    // ── Config plumbing: loadModel must land config in engine state ─────────

    @Test
    fun testLoadModelAppliesConfigBeforeEngineCreation() {
        val config = LLMConfig(
            systemPrompt = "You are terse.",
            backend = Backend.CPU,
            maxContextTokens = 8192.0,
            maxOutputTokens = 512.0,
            maxTokens = null,
            temperature = 0.3,
            topK = 12.0,
            topP = 0.8,
            multimodal = null,
            tools = null,
            enableSpeculativeDecoding = true,
            enableStructuredOutput = true,
            thinking = null,
            numThreads = null,
            prefillChunkSize = null,
            activationDataType = null,
            loraPath = "/lora/adapter.bin",
            audioLoraPath = null,
            loraRank = null,
            streamToolCalls = null,
            toolCallChannelName = null,
            forceLoad = null,
        )

        // The model file doesn't exist, so the load rejects — but config is
        // applied before engine creation and must stick.
        val promise = bridge.loadModel("/nonexistent/model.litertlm", config)
        while (!promise.isCompleted) { Thread.sleep(10) }
        assertNotNull("Load of a nonexistent model should reject", promise.error)

        fun field(name: String): Any? {
            val f = HybridLiteRTLM::class.java.getDeclaredField(name)
            f.isAccessible = true
            return f.get(bridge)
        }
        assertEquals("You are terse.", field("systemPrompt"))
        assertEquals(0.3, field("temperature") as Double, 1e-9)
        assertEquals(12, field("topK"))
        assertEquals(0.8, field("topP") as Double, 1e-9)
        assertEquals(8192, field("maxContextTokens"))
        assertEquals(512, field("maxOutputTokens"))
        assertEquals(true, field("enableSpeculativeDecoding"))
        assertEquals(true, field("enableStructuredOutput"))
        assertEquals("/lora/adapter.bin", field("loraPath"))
    }

    // ── Concurrency smoke: overlapping calls must not crash ─────────────────

    @Test
    fun testConcurrentExecuteAndUnloadDoesNotCrash() {
        val errors = java.util.concurrent.ConcurrentLinkedQueue<Throwable>()
        val threads = listOf(
            Thread {
                repeat(20) {
                    try {
                        val p = bridge.sendMessageAsync("hi") { _, _ -> }
                        while (!p.isCompleted) { Thread.sleep(1) }
                    } catch (t: Throwable) {
                        // A "no model loaded" rejection is the expected outcome;
                        // anything else (NPE, deadlock timeout, JNI crash) is not.
                        if (t.message?.contains("No model loaded") != true) {
                            errors.add(t)
                        }
                    }
                }
            },
            Thread {
                repeat(20) {
                    try {
                        bridge.unload()
                        bridge.resetConversation()
                    } catch (t: Throwable) {
                        if (t.message?.contains("No model loaded") != true) {
                            errors.add(t)
                        }
                    }
                }
            },
        )
        threads.forEach { it.start() }
        threads.forEach { it.join(15_000) }
        assertTrue("Concurrent execute/unload raised: ${errors.firstOrNull()}", errors.isEmpty())
    }
}
