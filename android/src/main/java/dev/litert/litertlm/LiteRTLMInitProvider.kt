package dev.litert.litertlm

import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.util.Log

class LiteRTLMInitProvider : ContentProvider() {
    companion object {
        private const val TAG = "LiteRTLMInitProvider"
        var applicationContext: Context? = null
            private set
    }

    override fun onCreate(): Boolean {
        applicationContext = context?.applicationContext
        Log.i(TAG, "LiteRTLMInitProvider initialized with context: $applicationContext")
        
        applicationContext?.registerComponentCallbacks(object : android.content.ComponentCallbacks2 {
            override fun onTrimMemory(level: Int) {
                // Trim levels are event codes, not severities — a `>=` check
                // matches TRIM_MEMORY_UI_HIDDEN (20), which fires on every
                // screen lock and would drop loaded engines (issue #24).
                if (com.margelo.nitro.dev.litert.litertlm.LiteRTLMRegistry.isMemoryEmergency(level)) {
                    com.margelo.nitro.dev.litert.litertlm.LiteRTLMRegistry.onTrimMemory(level)
                }
            }

            override fun onConfigurationChanged(newConfig: android.content.res.Configuration) {}
            override fun onLowMemory() {
                com.margelo.nitro.dev.litert.litertlm.LiteRTLMRegistry.onTrimMemory(android.content.ComponentCallbacks2.TRIM_MEMORY_COMPLETE)
            }
        })
        
        return true
    }

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?
    ): Cursor? = null

    override fun getType(uri: Uri): String? = null

    override fun insert(uri: Uri, values: ContentValues?): Uri? = null

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0

    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?
    ): Int = 0
}
