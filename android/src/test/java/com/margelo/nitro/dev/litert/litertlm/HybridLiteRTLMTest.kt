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
}
