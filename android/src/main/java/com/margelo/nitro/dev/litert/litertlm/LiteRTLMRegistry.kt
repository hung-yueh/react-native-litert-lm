package com.margelo.nitro.dev.litert.litertlm

import android.content.ComponentCallbacks2
import java.util.Collections
import java.util.WeakHashMap
import android.util.Log

/**
 * Global registry to track active LiteRTLM instances.
 * Used for memory trimming and cleanup.
 */
object LiteRTLMRegistry {
    private const val TAG = "LiteRTLMRegistry"

    // Use WeakSet-like structure to prevent leaks
    private val instances = Collections.newSetFromMap(WeakHashMap<HybridLiteRTLM, Boolean>())

    fun register(instance: HybridLiteRTLM) {
        synchronized(instances) {
            instances.add(instance)
        }
    }

    /**
     * Whether a trim level justifies dropping loaded engines. Trim levels are
     * event codes, not a severity scale: TRIM_MEMORY_UI_HIDDEN (20) fires on
     * every screen lock / home press and says nothing about memory pressure,
     * so a `>=` threshold must not be used here. Only two events mean the
     * process is about to die without intervention: RUNNING_CRITICAL
     * (foreground, memory critical) and COMPLETE (cached, next to be killed).
     */
    fun isMemoryEmergency(level: Int): Boolean =
        level == ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL ||
            level >= ComponentCallbacks2.TRIM_MEMORY_COMPLETE

    fun onTrimMemory(level: Int) {
        Log.w(TAG, "Memory emergency (level=$level). Releasing engines...")
        synchronized(instances) {
            // Release the heavy native resources but keep each instance
            // reloadable — close() would set isClosed and strand the JS side.
            instances.forEach { it.releaseUnderMemoryPressure() }
        }
    }
}
