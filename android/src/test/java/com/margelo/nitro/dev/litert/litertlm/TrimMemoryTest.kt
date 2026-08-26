package com.margelo.nitro.dev.litert.litertlm

import android.content.ComponentCallbacks2
import org.junit.Assert.*
import org.junit.Before
import org.junit.After
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import dev.litert.litertlm.LiteRTLMInitProvider

/**
 * Regression tests for issue #24: engines were released on every screen lock
 * because onTrimMemory treated TRIM_MEMORY_UI_HIDDEN (20) as memory pressure.
 */
@RunWith(RobolectricTestRunner::class)
class TrimMemoryTest {
    private lateinit var bridge: HybridLiteRTLM

    @Before
    fun setUp() {
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
    fun testLifecycleTrimLevelsAreNotMemoryEmergencies() {
        // UI_HIDDEN fires on every screen lock / home press; BACKGROUND and
        // MODERATE are routine LRU positions. None justify dropping an engine.
        assertFalse(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE))
        assertFalse(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW))
        assertFalse(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN))
        assertFalse(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_BACKGROUND))
        assertFalse(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_MODERATE))
    }

    @Test
    fun testGenuineEmergenciesAreMemoryEmergencies() {
        assertTrue(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL))
        assertTrue(LiteRTLMRegistry.isMemoryEmergency(ComponentCallbacks2.TRIM_MEMORY_COMPLETE))
    }

    @Test
    fun testClassifyTrimLevelIgnoresUiHidden() {
        assertNull(HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN))
    }

    @Test
    fun testClassifyTrimLevelMapsEventCodes() {
        assertEquals(MemoryWarningLevel.CRITICAL,
            HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL))
        assertEquals(MemoryWarningLevel.CRITICAL,
            HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_COMPLETE))
        assertEquals(MemoryWarningLevel.MODERATE,
            HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE))
        assertEquals(MemoryWarningLevel.MODERATE,
            HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW))
        assertEquals(MemoryWarningLevel.MODERATE,
            HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_BACKGROUND))
        assertEquals(MemoryWarningLevel.MODERATE,
            HybridLiteRTLM.classifyTrimLevel(ComponentCallbacks2.TRIM_MEMORY_MODERATE))
    }

    @Test
    fun testInstanceStaysReloadableAfterRegistryTrim() {
        // A registry trim must not put the instance into the closed state:
        // the app has to be able to recover with loadModel().
        LiteRTLMRegistry.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)

        val promise = bridge.loadModel("/nonexistent/model.litertlm", null)
        assertTrue("Promise should be completed", promise.isCompleted)
        val errMsg = promise.error?.message ?: promise.error?.cause?.message ?: ""
        assertFalse("loadModel must not be refused as closed after a trim, got: $errMsg",
            errMsg.contains("instance is closed"))
    }

    @Test
    fun testCloseStillRefusesLoadModel() {
        bridge.close()
        val promise = bridge.loadModel("/nonexistent/model.litertlm", null)
        assertTrue("Promise should be completed", promise.isCompleted)
        val errMsg = promise.error?.message ?: promise.error?.cause?.message ?: ""
        assertTrue("Expected closed-instance error, got: $errMsg",
            errMsg.contains("instance is closed"))
    }

    @Test
    fun testRegistryTrimKeepsMemoryWarningSubscription() {
        // The old code called close() on trim, which silently unregistered the
        // app's memory-warning callback along with the engine.
        bridge.setMemoryWarningCallback { _, _ -> }

        val field = HybridLiteRTLM::class.java.getDeclaredField("componentCallbacks")
        field.isAccessible = true
        assertNotNull("Callback should be registered", field.get(bridge))

        LiteRTLMRegistry.onTrimMemory(ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
        assertNotNull("Trim must not drop the warning subscription", field.get(bridge))

        bridge.close()
        assertNull("close() should unregister the subscription", field.get(bridge))
    }
}
