///
/// HybridLiteRTLM.kt
/// Kotlin implementation of LiteRTLM HybridObject using LiteRT-LM Android SDK.
///

package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import android.os.Debug
import android.app.ActivityManager
import android.content.Context
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import dev.litert.litertlm.LiteRTLMInitProvider
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.SamplerConfig
import com.margelo.nitro.dev.litert.litertlm.Backend
import com.margelo.nitro.dev.litert.litertlm.GenerationStats
import com.margelo.nitro.dev.litert.litertlm.HybridLiteRTLMSpec
import com.margelo.nitro.dev.litert.litertlm.LLMConfig
import com.margelo.nitro.dev.litert.litertlm.Message
import com.margelo.nitro.dev.litert.litertlm.Role
import com.margelo.nitro.core.Promise
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.LoraConfig
import com.google.ai.edge.litertlm.NoRepeatNgramConfig
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.RepetitionPenaltyConfig
import com.google.ai.edge.litertlm.ResponseFormat
import com.google.ai.edge.litertlm.SuppressTokensConfig
import com.google.ai.edge.litertlm.ThinkingConfig
import com.google.ai.edge.litertlm.tool
import org.json.JSONException
import org.json.JSONObject
import com.google.ai.edge.litertlm.ToolProvider
import android.content.ComponentCallbacks2
import android.content.res.Configuration
import java.io.File



// Alias to avoid confusion with our generated Message type
// Alias to avoid confusion
typealias LiteRTMessage = com.google.ai.edge.litertlm.Message



/**
 * Kotlin implementation of LiteRTLM using the LiteRT-LM Android SDK.
 * This class bridges between React Native (via Nitro) and the Google LiteRT-LM Engine.
 */
@DoNotStrip
@Keep
class HybridLiteRTLM : HybridLiteRTLMSpec() {

    companion object {
        private const val TAG = "HybridLiteRTLM"
        private val initLock = Any()

        /**
         * Map a [ComponentCallbacks2] trim level to a JS-facing warning, or
         * null for events that carry no memory-pressure signal. Trim levels
         * are event codes, not a severity scale — TRIM_MEMORY_UI_HIDDEN (20)
         * fires on every screen lock / home press (issue #24).
         */
        internal fun classifyTrimLevel(level: Int): MemoryWarningLevel? = when (level) {
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL,
            ComponentCallbacks2.TRIM_MEMORY_COMPLETE -> MemoryWarningLevel.CRITICAL
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_MODERATE,
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW,
            ComponentCallbacks2.TRIM_MEMORY_BACKGROUND,
            ComponentCallbacks2.TRIM_MEMORY_MODERATE -> MemoryWarningLevel.MODERATE
            else -> null
        }

        /** Process memory, shared with the embedder (one app, one memory budget). */
        internal fun processMemoryUsage(): MemoryUsage {
            // Native heap: allocated bytes from Debug APIs (most accurate for native allocations)
            val nativeHeapBytes = Debug.getNativeHeapAllocatedSize().toDouble()

            // Process RSS: read from /proc/self/status (VmRSS) in kB
            var residentBytes = 0.0
            try {
                java.io.File("/proc/self/status").forEachLine { line ->
                    if (line.startsWith("VmRSS:")) {
                        val kb = line.substringAfter("VmRSS:").trim().split("\\s+".toRegex())[0].toDoubleOrNull()
                        if (kb != null) {
                            residentBytes = kb * 1024.0
                        }
                        return@forEachLine
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to read /proc/self/status: ${e.message}")
            }

            // Available memory and low-memory flag from ActivityManager
            var availableMemoryBytes = 0.0
            var isLowMemory = false
            try {
                val context = LiteRTLMInitProvider.applicationContext
                if (context != null) {
                    val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
                    val memInfo = ActivityManager.MemoryInfo()
                    activityManager.getMemoryInfo(memInfo)
                    availableMemoryBytes = memInfo.availMem.toDouble()
                    isLowMemory = memInfo.lowMemory
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to get ActivityManager memory info: ${e.message}")
            }

            return MemoryUsage(
                nativeHeapBytes = nativeHeapBytes,
                residentBytes = residentBytes,
                availableMemoryBytes = availableMemoryBytes,
                isLowMemory = isLowMemory
            )
        }

        /**
         * 0.7's default for `multimodal` when unset (on for Gemma 3n / Gemma 3
         * file names), now only for bundles whose metadata can't be read.
         * Deprecated — the JS layer warns; removed in 1.0.
         */
        internal fun legacyMultimodalHint(modelPath: String): Boolean {
            val name = modelPath.substringAfterLast("/").lowercase()
            return name.contains("3n") || name.contains("gemma3")
        }

        /**
         * Initialize the native library.
         * Must be called from Application.onCreate() to register the HybridObject.
         */
        fun initialize() {
            try {
                // Call generated internal OnLoad to load the library
                LiteRTLMOnLoad.initializeNative()
            } catch (e: Throwable) {
                Log.e(TAG, "Failed to initialize LiteRTLM native library", e)
            }
        }
    }

    init {
        LiteRTLMRegistry.register(this)
    }

    // LiteRT-LM Engine and Conversation
    private var engine: Engine? = null
    private var conversation: Conversation? = null
    
    @Volatile
    private var isClosed = false

    /** Latch of the in-flight streaming worker, if any. Used to wait for cancellation to settle. */
    @Volatile
    private var activeStreamingLatch: CountDownLatch? = null

    private val modelStore = HybridModelStore()
    private var loadedModelPath: String? = null

    // Conversation history for getHistory()
    // Synchronized to prevent ConcurrentModificationException: history is
    // written from Promise.parallel workers and sendMessageAsync SDK callbacks,
    // and read from getHistory() which may be called from the JS thread.
    private val history: MutableList<Message> = Collections.synchronizedList(mutableListOf())

    // Last generation stats
    private var lastStats = GenerationStats(
        promptTokens = 0.0,
        completionTokens = 0.0,
        totalTokens = 0.0,
        timeToFirstToken = 0.0,
        totalTime = 0.0,
        tokensPerSecond = 0.0
    )

    // Configuration
    private var backend: Backend = Backend.CPU
    private var temperature: Double = 0.7
    private var topK: Int = 40
    private var topP: Double = 0.95
    private var maxContextTokens: Int = 4096
    private var maxOutputTokens: Int = 1024
    private var enableStructuredOutput: Boolean = false
    private var sessionThinking: ThinkingOptions? = null
    private var systemPrompt: String? = null
    private var tools: Array<ToolDefinition>? = null
    private var enableSpeculativeDecoding: Boolean = false
    private var loraPath: String? = null
    private var audioLoraPath: String? = null
    private var activationDataType: ActivationDataType? = null

    /**
     * Encoders the live engine was created with; execute() rejects media parts
     * for a missing one with a typed error instead of an engine failure.
     */
    @Volatile
    internal var hasVisionEncoder = false
        private set
    @Volatile
    internal var hasAudioEncoder = false
        private set

    /** Size of the loaded model file in bytes (0 when unloaded). */
    @Volatile
    private var loadedModelSizeBytes: Long = 0L

    /** OS memory-pressure listener registered on the application context. */
    private var memoryWarningCallback: ((MemoryWarningLevel, MemoryUsage) -> Unit)? = null
    private var componentCallbacks: ComponentCallbacks2? = null

    override val memorySize: Long
        get() = loadedModelSizeBytes

    // -------------------------------------------------------------------------
    // loadModel - Initialize LiteRT-LM Engine and Conversation
    // -------------------------------------------------------------------------
    override fun loadModel(modelPath: String, config: LLMConfig?): Promise<Unit> {
        return Promise.parallel { loadModelBlocking(modelPath, config) }
    }

    /**
     * The body of [loadModel], run on a Promise worker. Internal so on-device
     * tests can observe the typed failure without Promise/JSI plumbing.
     */
    internal fun loadModelBlocking(modelPath: String, config: LLMConfig?) {
        // Serialize initialization to prevent OOM from concurrent loads
        synchronized(initLock) {
            if (isClosed) {
                throw LiteRTLMNativeException(
                    "load", "FAILED_PRECONDITION", null,
                    "This LiteRTLM instance is closed; create a new one with createLLM().",
                )
            }
            
            Log.i(TAG, "loadModel: $modelPath")

            // Clean up existing resources
            // We call internal cleanup that doesn't set isClosed
            cleanupInternal()

            // Apply configuration
            config?.let { cfg ->
                cfg.backend?.let { backend = it }
                cfg.temperature?.let { temperature = it }
                cfg.topK?.let { topK = it.toInt() }
                cfg.topP?.let { topP = it }
                // New split fields take priority over legacy maxTokens
                cfg.maxContextTokens?.let { maxContextTokens = it.toInt() }
                cfg.maxOutputTokens?.let { maxOutputTokens = it.toInt() }
                // Legacy: if only maxTokens is set, map to both for backward compat
                if (cfg.maxContextTokens == null && cfg.maxOutputTokens == null) {
                    cfg.maxTokens?.let {
                        maxContextTokens = it.toInt()
                        maxOutputTokens = it.toInt()
                    }
                }
                cfg.systemPrompt?.let { systemPrompt = it }
                tools = cfg.tools
                enableSpeculativeDecoding = cfg.enableSpeculativeDecoding ?: false
                enableStructuredOutput = cfg.enableStructuredOutput ?: false
                sessionThinking = cfg.thinking
                loraPath = cfg.loraPath
                audioLoraPath = cfg.audioLoraPath
                // Exposed by the Kotlin SDK since LiteRT-LM 0.18.0.
                activationDataType = cfg.activationDataType
                // numThreads / prefillChunkSize / loraRank are iOS-only: the
                // Kotlin SDK does not expose them (see LLMConfig docs).
                if (cfg.streamToolCalls == true) {
                    Log.w(TAG, "streamToolCalls (mid-generation channel streaming) is not " +
                        "wired on Android yet. Completed tool calls ARE surfaced as typed " +
                        "toolCall events when generation finishes; only token-by-token " +
                        "tool-call streaming is iOS-only for now.")
                }
            }

            // A missing file would otherwise burn through every engine
            // attempt below and surface as a generic creation failure.
            if (!File(modelPath).exists()) {
                throw LiteRTLMNativeException(
                    "load", "NOT_FOUND", "MODEL_NOT_FOUND", "Model file not found: $modelPath",
                )
            }

            try {
                // Early GPU hardware check: LiteRT-LM's GPU delegate needs an
                // OpenCL driver, which most Samsung/Qualcomm devices lack. Only a
                // warning — the fallback chain below handles it.
                if (backend == Backend.GPU) {
                    if (!GpuProbe.openCLAvailable()) {
                        Log.w(TAG, "OpenCL library not found — GPU backend will likely fail, fallback chain will attempt CPU")
                    } else {
                        Log.i(TAG, "OpenCL library found — GPU backend is available")
                    }
                }

                // Encoders come from the bundle's own metadata (LiteRT-LM 0.18
                // ModelInfo): with `multimodal` unset, every encoder the bundle has
                // loads, each on the backend the bundle declares (Gemma 4 E2B: CPU
                // for both). `multimodal: false` keeps the engine text-only. Only
                // when the metadata is unreadable does 0.7's file-name default
                // apply; that fallback is deprecated and goes in 1.0.
                val plan = ModelBundle.encoderPlan(modelPath)
                val wantEncoders = config?.multimodal ?: (plan != null || legacyMultimodalHint(modelPath))
                // Unreadable metadata (older bundles): 0.7's choice — GPU vision, CPU audio.
                val visionBackend = if (!wantEncoders) null else if (plan == null) Backend.GPU else plan.vision
                val audioBackend = if (!wantEncoders) null else if (plan == null) Backend.CPU else plan.audio
                if (config?.multimodal == true && plan != null && plan.vision == null && plan.audio == null) {
                    Log.w(TAG, "multimodal: true, but this bundle has no vision or audio encoder — loading text-only")
                }

                // Get cache directory from application context
                val cacheDirectory = LiteRTLMInitProvider.applicationContext?.cacheDir?.absolutePath
                Log.i(TAG, "Using cache directory: $cacheDirectory")

                if (enableSpeculativeDecoding) {
                    @OptIn(ExperimentalApi::class)
                    ExperimentalFlags.enableSpeculativeDecoding = true
                }

                // The SDK's own reason for the most recent failed attempt.
                var lastEngineError: Throwable? = null

                // Helper: attempt engine creation with given backends, return null on failure
                fun tryCreateEngine(
                    mainBackend: com.google.ai.edge.litertlm.Backend,
                    visionBackend: com.google.ai.edge.litertlm.Backend?,
                    audioBackend: com.google.ai.edge.litertlm.Backend?
                ): Engine? {
                    return try {
                        // Each encoder independently: a vision-only bundle keeps vision.
                        val cfg = EngineConfig(
                            modelPath = modelPath,
                            backend = mainBackend,
                            visionBackend = visionBackend,
                            audioBackend = audioBackend,
                            maxNumTokens = maxContextTokens,
                            cacheDir = cacheDirectory,
                            activationDataType = activationDataType?.toSdk(),
                        )
                        Engine(cfg).also { it.initialize() }
                    } catch (e: Exception) {
                        Log.w(TAG, "Engine creation failed with backend $mainBackend: ${e.message}")
                        lastEngineError = e
                        null
                    }
                }

                // Map our Backend enum to LiteRT-LM Backend sealed class
                val lmBackend = when (backend) {
                    Backend.GPU -> com.google.ai.edge.litertlm.Backend.GPU()
                    Backend.NPU -> {
                        Log.i(TAG, "NPU backend requested - requires hardware support")
                        com.google.ai.edge.litertlm.Backend.NPU()
                    }
                    else -> com.google.ai.edge.litertlm.Backend.CPU()
                }

                Log.i(TAG, "Backend config: main=$lmBackend, vision=${visionBackend ?: "none"}, audio=${audioBackend ?: "none"}")

                if (isClosed) return@synchronized

                // The requested configuration first, then progressively more
                // conservative CPU ones — each distinct attempt tried once, and
                // never with an encoder the caller didn't ask for or the bundle lacks.
                val requested = Attempt(backend, visionBackend, audioBackend)
                val attempts = listOf(
                    requested,
                    Attempt(Backend.CPU, visionBackend, audioBackend), // keep the encoders' backends
                    Attempt(Backend.CPU, visionBackend?.let { Backend.CPU }, audioBackend?.let { Backend.CPU }), // all CPU
                    Attempt(Backend.CPU, null, null), // text only
                ).distinct()

                var eng: Engine? = null
                for ((index, attempt) in attempts.withIndex()) {
                    if (index == 1) Log.w(TAG, "${backend.name} engine failed — trying the CPU fallbacks")
                    eng = tryCreateEngine(attempt.main.toSdk(), attempt.vision?.toSdk(), attempt.audio?.toSdk())
                    if (eng != null) {
                        hasVisionEncoder = attempt.vision != null
                        hasAudioEncoder = attempt.audio != null
                        if (index > 0) {
                            Log.w(TAG, "Loaded with the fallback: $attempt")
                            backend = Backend.CPU
                        }
                        break
                    }
                }

                // Report the last attempt (the most permissive config): if even
                // CPU text-only failed, its reason is the real one.
                engine = eng ?: throw LiteRTLMNativeException.from(
                    "load",
                    "Failed to create the engine — tried ${backend.name.lowercase()}, then the CPU fallbacks",
                    lastEngineError ?: IllegalStateException("no engine was created"),
                )
                Log.i(TAG, "Engine created and initialized successfully")

                // Create Conversation
                createNewConversation()
                Log.i(TAG, "Conversation created successfully")
                loadedModelPath = modelPath
                loadedModelSizeBytes = try { File(modelPath).length() } catch (_: Exception) { 0L }

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load model: ${e.message}", e)
                // Clean up partial state so isReady() returns false
                cleanupInternal()
                throw LiteRTLMNativeException.from("load", "Failed to load the model", e)
            }
        }
    }

    // Legacy inference — shapes mirror src/inferenceRouting.ts; JS createLLM routes via execute.
    override fun sendMessage(message: String): Promise<String> =
        execute(parts = arrayOf(MultimodalPartFactories.textPart(message)), onToken = null, options = null)

    override fun sendMessageAsync(message: String, onToken: (String, Boolean) -> Unit): Promise<Unit> =
        executeVoid(parts = arrayOf(MultimodalPartFactories.textPart(message)), onToken = onToken)

    // -------------------------------------------------------------------------
    // Multimodal methods
    // -------------------------------------------------------------------------
    
    /**
     * Resize image if dimensions exceed maxDimension to prevent OOM.
     * Gemma 3n's vision encoder is optimized for 512x512 or 1024x1024.
     * Passing larger images can spike memory 500MB+.
     */
    private fun resizeImageIfNeeded(imagePath: String, maxDimension: Int = 1024): String {
        val originalBitmap = android.graphics.BitmapFactory.decodeFile(imagePath)
            ?: throw LiteRTLMNativeException(
                "execute", "INVALID_ARGUMENT", "INVALID_INPUT", "Failed to decode image: $imagePath",
            )

        val width = originalBitmap.width
        val height = originalBitmap.height

        // If already within bounds, return original path
        if (width <= maxDimension && height <= maxDimension) {
            originalBitmap.recycle()
            return imagePath
        }

        Log.i(TAG, "Resizing image from ${width}x${height} to fit ${maxDimension}px")

        val scale = maxDimension.toFloat() / maxOf(width, height)
        val newWidth = (width * scale).toInt()
        val newHeight = (height * scale).toInt()

        val resizedBitmap = android.graphics.Bitmap.createScaledBitmap(originalBitmap, newWidth, newHeight, true)
        originalBitmap.recycle()

        // Save to temp file
        val cacheDir = LiteRTLMInitProvider.applicationContext?.cacheDir
            ?: throw RuntimeException("Application context not available for image resizing")
        val tempFile = java.io.File(cacheDir, "resized_${java.util.UUID.randomUUID()}.jpg")
        java.io.FileOutputStream(tempFile).use { out ->
            resizedBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
        }
        resizedBitmap.recycle()

        Log.i(TAG, "Resized image saved to: ${tempFile.absolutePath} (${newWidth}x${newHeight})")
        return tempFile.absolutePath
    }

    override fun sendMessageWithImage(message: String, imagePath: String): Promise<String> =
        execute(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.imagePart(imagePath)),
            onToken = null,
            options = null,
        )

    override fun sendMessageWithImageAsync(message: String, imagePath: String, onToken: (String, Boolean) -> Unit): Promise<Unit> =
        executeVoid(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.imagePart(imagePath)),
            onToken = onToken,
        )

    override fun downloadModel(url: String, fileName: String, onProgress: ((Double) -> Unit)?): Promise<String> {
        return modelStore.downloadFile(url, fileName, "{}", onProgress ?: {})
    }

    override fun deleteModel(fileName: String): Promise<Unit> {
        return Promise.parallel {
            modelStore.deleteFile(fileName)
            val currentlyLoadedName = loadedModelPath?.substringAfterLast("/")?.lowercase()
            if (currentlyLoadedName != null && currentlyLoadedName == fileName.lowercase()) {
                if (engine != null) {
                    cleanupInternal()
                }
                // The backing file is gone — clear the stale path even if no
                // engine was live (cleanupInternal handles the loaded case).
                loadedModelPath = null
            }
        }
    }

    override fun sendMessageWithAudioAsync(message: String, audioPath: String, onToken: (String, Boolean) -> Unit): Promise<Unit> =
        executeVoid(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.audioPart(audioPath)),
            onToken = onToken,
        )

    override fun sendMessageWithAudio(message: String, audioPath: String): Promise<String> =
        execute(
            parts = arrayOf(MultimodalPartFactories.textPart(message), MultimodalPartFactories.audioPart(audioPath)),
            onToken = null,
            options = null,
        )

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    override fun getHistory(): Array<Message> {
        // Synchronized list requires manual sync for iteration/copy
        synchronized(history) {
            return history.toTypedArray()
        }
    }

    override fun resetConversation(historyJson: String?, systemPrompt: String?) {
        // Signal native inference to cancel and await the in-flight streaming
        // worker to settle before closing the conversation underneath it.
        cancelInFlightGeneration()
        synchronized(history) {
            history.clear()
            // Mirror the replayed transcript into the wrapper's own history so
            // getHistory() stays truthful for the restored conversation.
            parseTranscript(historyJson).forEach { (role, content) ->
                val nitroRole = when (role) {
                    "model" -> Role.MODEL
                    "system" -> Role.SYSTEM
                    else -> Role.USER
                }
                history.add(Message(nitroRole, content))
            }
        }
        createNewConversation(
            initialMessagesJson = historyJson,
            systemPromptOverride = systemPrompt,
        )
    }

    /** Parse a `[{role, content}]` transcript JSON into (role, content) pairs. */
    private fun parseTranscript(historyJson: String?): List<Pair<String, String>> {
        if (historyJson.isNullOrEmpty()) return emptyList()
        return try {
            val arr = org.json.JSONArray(historyJson)
            (0 until arr.length()).mapNotNull { i ->
                val obj = arr.optJSONObject(i) ?: return@mapNotNull null
                val role = obj.optString("role", "user")
                val content = obj.optString("content", "")
                role to content
            }
        } catch (e: JSONException) {
            Log.e(TAG, "resetConversation: unparseable historyJson — starting empty", e)
            emptyList()
        }
    }

    override fun isReady(): Boolean {
        return isLoaded_
    }
    
    // Property backing field for isReady check
    private val isLoaded_: Boolean
        get() = engine != null

    override fun getStats(): GenerationStats {
        return lastStats
    }

    override fun getMemoryUsage(): MemoryUsage = processMemoryUsage()


    override fun getContextTokenCount(): Double {
        // The LiteRT-LM Kotlin SDK does not expose a conversation-level token
        // count (the C API has litert_lm_conversation_get_token_count but the
        // Kotlin wrapper omits it). Return -1 to signal "unknown" — callers
        // already handle this gracefully.
        return -1.0
    }

    override fun unload(): Promise<Unit> {
        return Promise.parallel {
            Log.d(TAG, "Unloading model (instance stays reusable)")
            cleanupInternal()
        }
    }

    override fun setMemoryWarningCallback(onWarning: (MemoryWarningLevel, MemoryUsage) -> Unit) {
        synchronized(initLock) {
            memoryWarningCallback = onWarning
            if (componentCallbacks == null) {
                val context = LiteRTLMInitProvider.applicationContext
                if (context == null) {
                    Log.w(TAG, "setMemoryWarningCallback: no application context available")
                    return
                }
                val callbacks = object : ComponentCallbacks2 {
                    override fun onTrimMemory(level: Int) {
                        val cb = memoryWarningCallback ?: return
                        val warningLevel = classifyTrimLevel(level) ?: return
                        try {
                            cb(warningLevel, getMemoryUsage())
                        } catch (e: Exception) {
                            Log.w(TAG, "Memory warning callback failed: ${e.message}")
                        }
                    }

                    override fun onConfigurationChanged(newConfig: Configuration) {}

                    @Deprecated("Deprecated in ComponentCallbacks")
                    override fun onLowMemory() {
                        val cb = memoryWarningCallback ?: return
                        try {
                            cb(MemoryWarningLevel.CRITICAL, getMemoryUsage())
                        } catch (e: Exception) {
                            Log.w(TAG, "Memory warning callback failed: ${e.message}")
                        }
                    }
                }
                context.registerComponentCallbacks(callbacks)
                componentCallbacks = callbacks
            }
        }
    }

    override fun clearMemoryWarningCallback() {
        synchronized(initLock) {
            memoryWarningCallback = null
            componentCallbacks?.let { callbacks ->
                try {
                    LiteRTLMInitProvider.applicationContext?.unregisterComponentCallbacks(callbacks)
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to unregister component callbacks: ${e.message}")
                }
            }
            componentCallbacks = null
        }
    }

    override fun close() {
        Log.d(TAG, "Closing resources")
        isClosed = true
        clearMemoryWarningCallback()
        cleanupInternal()
    }

    /**
     * Called by [LiteRTLMRegistry] on a genuine OS memory emergency. Frees the
     * heavy native engine but — unlike [close] — leaves the instance open and
     * the memory-warning subscription registered, so the app can recover with
     * loadModel(). The app still observes the event: the registry's component
     * callback was registered at process start (before any JS could call
     * setMemoryWarningCallback), so the per-instance callback fires after this
     * release and reports the same trim event as CRITICAL.
     */
    internal fun releaseUnderMemoryPressure() {
        if (engine == null) return
        Log.w(TAG, "Releasing engine under memory pressure (instance stays reloadable)")
        cleanupInternal()
    }

    /**
     * Signals the native inference thread to cancel and awaits the in-flight
     * streaming worker to settle before closing or deleting the conversation.
     * Prevents native race conditions where close() deletes native conversation
     * state while the decode loop is still running on a background worker thread.
     */
    private fun cancelInFlightGeneration() {
        try {
            conversation?.cancelProcess()
            val latch = activeStreamingLatch
            if (latch != null) {
                val settled = latch.await(2, TimeUnit.SECONDS)
                if (!settled) {
                    Log.w(TAG, "cancelInFlightGeneration: worker did not settle within timeout")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "cancelInFlightGeneration error: ${e.message}")
        }
    }

    private fun cleanupInternal() {
        synchronized(initLock) {
            try {
                // Abort any in-flight generation and let worker settle before close
                cancelInFlightGeneration()
                conversation?.close()
                conversation = null
                engine?.close()        // Direct call
                engine = null
                hasVisionEncoder = false
                hasAudioEncoder = false
                loadedModelPath = null
                loadedModelSizeBytes = 0L
            } catch (e: Exception) {
                Log.e(TAG, "Error closing resources", e)
            }
        }
    }

    private fun ensureLoaded() {
        if (engine == null) {
            throw LiteRTLMNativeException(
                "execute", "FAILED_PRECONDITION", "MODEL_NOT_LOADED",
                "No model loaded. Call loadModel() first.",
            )
        }
    }

    /** Mirrors iOS validateMediaPath: a typed error instead of a decode failure later. */
    private fun requireMediaFile(path: String, label: String) {
        if (!File(path).exists()) {
            throw LiteRTLMNativeException("execute", "NOT_FOUND", "INVALID_INPUT", "$label file not found: $path")
        }
    }

    private fun createNewConversation(
        initialMessagesJson: String? = null,
        systemPromptOverride: String? = null,
    ) {
        ensureLoaded()
        // v0.10.2 enforces single-session: close existing conversation first
        conversation?.let { oldConv ->
            try {
                oldConv.close()
            } catch (e: Exception) {
                Log.w(TAG, "Failed to close old conversation: ${e.message}")
            }
            conversation = null
        }
        // Map tools. The SDK's tool() adapter requires a {name, description,
        // parameters} description JSON — parameters alone makes it throw, and
        // the old `as ToolProvider` cast of an OpenApiTool always threw
        // ClassCastException (OpenApiTool is an interface, ToolProvider an
        // unrelated abstract class).
        val lmTools: List<ToolProvider>? = tools?.map { toolDef ->
            val apiTool = object : OpenApiTool {
                override fun getToolDescriptionJsonString(): String {
                    val params = try {
                        JSONObject(toolDef.parametersJson)
                    } catch (e: JSONException) {
                        Log.e(TAG, "Tool '${toolDef.name}' has unparseable parametersJson — sending empty schema", e)
                        JSONObject()
                    }
                    return JSONObject()
                        .put("name", toolDef.name)
                        .put("description", toolDef.description)
                        .put("parameters", params)
                        .toString()
                }
                override fun execute(paramsJsonString: String): String {
                    // Never invoked: automaticToolCalling is disabled — tool
                    // execution happens on the JS side.
                    return "{}"
                }
            }
            tool(apiTool)
        }

        // Create conversation with explicit SamplerConfig (required by Gallery pattern).
        // GPU backend may fail silently without proper sampler params.
        val convConfig = ConversationConfig(
            samplerConfig = SamplerConfig(
                topK = topK,
                topP = topP.toDouble(),
                temperature = temperature.toDouble(),
            ),
            systemInstruction = (systemPromptOverride ?: systemPrompt)
                ?.let { Contents.of(Content.Text(it)) },
            // Seed with a prior transcript (multi-conversation switching); the
            // engine re-prefills these on the next message.
            initialMessages = parseTranscript(initialMessagesJson).map { (role, content) ->
                if (role == "model") LiteRTMessage.model(content) else LiteRTMessage.user(content)
            },
            tools = lmTools ?: emptyList(),
            // The SDK defaults to auto-running tool calls against OpenApiTool.execute()
            // (a stub here). Tool execution belongs to the JS side — surface the
            // calls instead (see execute()).
            automaticToolCalling = false,
            loraConfig = if (loraPath != null || audioLoraPath != null) {
                LoraConfig(loraPath, audioLoraPath)
            } else {
                null
            },
            // Available since LiteRT-LM 0.15.0 (previously iOS-only)
            maxOutputToken = maxOutputTokens,
            // Initializes the LLGuidance constraint provider for per-call
            // responseFormat (structured output)
            enableResponseFormat = enableStructuredOutput,
            thinkingConfig = sessionThinking?.toSdkThinkingConfig(),
        )
        conversation = engine!!.createConversation(convConfig)
    }



    override fun sendMultimodalMessage(parts: Array<MultimodalPart>): Promise<String> {
        return execute(parts = parts, onToken = null, options = null)
    }

    /** Streaming adapter for legacy `Promise<Unit>` APIs — all inference runs through [execute]. */
    private fun executeVoid(
        parts: Array<MultimodalPart>,
        onToken: (String, Boolean) -> Unit,
    ): Promise<Unit> {
        val voidPromise = Promise<Unit>()
        try {
            execute(parts, onToken, null)
                .then { voidPromise.resolve(Unit) }
                .catch { voidPromise.reject(it) }
        } catch (e: Throwable) {
            voidPromise.reject(e)
        }
        return voidPromise
    }

    private class PreprocessedPart(
        val type: PartType,
        val text: String?,
        val path: String?,
        val bytes: ByteArray?
    )

    override fun execute(
        parts: Array<MultimodalPart>,
        onToken: ((token: String, done: Boolean) -> Unit)?,
        options: ExecuteOptions?,
    ): Promise<String> {
        // visualTokenBudget has no Kotlin SDK surface — session defaults apply.
        if (options?.visualTokenBudget != null) {
            Log.w(TAG, "per-message visualTokenBudget is not supported on Android — session config applies")
        }
        // Preprocess synchronously on the JS/JSI thread to safely extract JS buffer bytes
        val preprocessed = parts.map { part ->
            val bytes = when (part.type) {
                PartType.IMAGE -> part.imageBuffer?.let { buf ->
                    val javaBuf = buf.getBuffer(false)
                    val arr = ByteArray(javaBuf.remaining())
                    javaBuf.get(arr)
                    arr
                }
                PartType.AUDIO -> part.audioBuffer?.let { buf ->
                    val javaBuf = buf.getBuffer(false)
                    val arr = ByteArray(javaBuf.remaining())
                    javaBuf.get(arr)
                    arr
                }
                else -> null
            }
            PreprocessedPart(
                type = part.type,
                text = part.text,
                path = part.path,
                bytes = bytes
            )
        }

        return Promise.parallel {
            ensureLoaded()
            preprocessed.firstOrNull { (it.type == PartType.IMAGE && !hasVisionEncoder) || (it.type == PartType.AUDIO && !hasAudioEncoder) }
                ?.let { part ->
                    val missing = if (part.type == PartType.IMAGE) "vision" else "audio"
                    throw LiteRTLMNativeException(
                        "execute", "FAILED_PRECONDITION", "UNSUPPORTED",
                        "This model was loaded without its $missing encoder. Pass multimodal: true to " +
                            "loadModel() (inspectModel() shows whether the bundle has one).",
                    )
                }

            val tempFiles = mutableListOf<java.io.File>()

            try {
                val contents = mutableListOf<Content>()
                var userTextRepresentation = ""

                for (part in preprocessed) {
                    when (part.type) {
                        PartType.TEXT -> part.text?.let {
                            contents.add(Content.Text(it))
                            userTextRepresentation += "$it "
                        }
                        PartType.IMAGE -> {
                            val imagePath = when {
                                part.path != null -> part.path
                                part.bytes != null -> {
                                    val tmp = java.io.File(
                                        LiteRTLMInitProvider.applicationContext!!.cacheDir,
                                        "litert_buf_${java.util.UUID.randomUUID()}.jpg"
                                    )
                                    tmp.writeBytes(part.bytes)
                                    tempFiles.add(tmp)
                                    tmp.absolutePath
                                }
                                else -> null
                            }
                            if (imagePath != null) {
                                requireMediaFile(imagePath, "Image")
                                val processedPath = resizeImageIfNeeded(imagePath)
                                if (processedPath != imagePath) tempFiles.add(java.io.File(processedPath))
                                contents.add(Content.ImageFile(processedPath))
                                userTextRepresentation += "[Image] "
                            }
                        }
                        PartType.AUDIO -> {
                            val audioPath = when {
                                part.path != null -> part.path
                                part.bytes != null -> {
                                    val tmp = java.io.File(
                                        LiteRTLMInitProvider.applicationContext!!.cacheDir,
                                        "litert_buf_${java.util.UUID.randomUUID()}.wav"
                                    )
                                    tmp.writeBytes(part.bytes)
                                    tempFiles.add(tmp)
                                    tmp.absolutePath
                                }
                                else -> null
                            }
                            if (audioPath != null) {
                                requireMediaFile(audioPath, "Audio")
                                contents.add(Content.AudioFile(audioPath))
                                userTextRepresentation += "[Audio] "
                            }
                        }
                    }
                }

                userTextRepresentation = userTextRepresentation.trim()
                history.add(Message(Role.USER, userTextRepresentation))

                val userMsg = LiteRTMessage.user(Contents.of(contents))

                // v0.15: per-message constrained decoding. Schema takes precedence
                // over regex, mirroring iOS.
                val responseFormat = options?.responseSchema?.let { ResponseFormat.json(it) }
                    ?: options?.responseRegex?.let { ResponseFormat.regex(it) }
                if (responseFormat != null && !enableStructuredOutput) {
                    throw LiteRTLMNativeException(
                        "execute", "FAILED_PRECONDITION", "INVALID_CONFIG",
                        "responseSchema/responseRegex require enableStructuredOutput: true in the loadModel() config.",
                    )
                }

                // v0.15: per-turn generation controls
                val repetitionPenaltyConfig = if (
                    options?.repetitionPenalty != null || options?.presencePenalty != null ||
                    options?.frequencyPenalty != null || options?.penaltyWindowSize != null
                ) {
                    RepetitionPenaltyConfig(
                        repetitionPenalty = options?.repetitionPenalty?.toFloat(),
                        presencePenalty = options?.presencePenalty?.toFloat(),
                        frequencyPenalty = options?.frequencyPenalty?.toFloat(),
                        windowSize = options?.penaltyWindowSize?.toInt(),
                    )
                } else null
                val noRepeatNgramConfig = if (
                    options?.noRepeatNgramSize != null || options?.noRepeatNgramWindowSize != null
                ) {
                    NoRepeatNgramConfig(
                        noRepeatNgramSize = options?.noRepeatNgramSize?.toInt(),
                        windowSize = options?.noRepeatNgramWindowSize?.toInt(),
                    )
                } else null
                // Safe since LiteRT-LM 0.17.0: the JNI layer takes the ids as int[]
                // (google-ai-edge/LiteRT-LM#3229). Through 0.16.1 it looked up a
                // Kotlin-mangled accessor and aborted the process.
                val suppressTokensConfig = options?.suppressTokens?.let { ids ->
                    SuppressTokensConfig(ids.map { it.toInt() })
                }
                val perCallThinking = options?.thinking?.toSdkThinkingConfig()
                val perCallMaxOutput = options?.maxOutputTokens?.toInt()

                if (onToken != null) {
                    // ── Streaming path ────────────────────────────────────────────────
                    val latch = CountDownLatch(1)
                    activeStreamingLatch = latch
                    val errorRef = AtomicReference<Throwable?>(null)
                    val fullResponseBuilder = StringBuilder()

                    val listener = StreamingCallbackListener(
                        onToken = { token, done ->
                            onToken(token, done)
                            if (done) latch.countDown()
                        },
                        responseBuilder = fullResponseBuilder,
                        history = history,
                        userMessage = userTextRepresentation,
                        onStatsReady = { stats -> lastStats = stats },
                        onFailure = { e -> errorRef.set(e) }
                    )

                    try {
                        try {
                            conversation!!.sendMessageAsync(
                                message = userMsg,
                                callback = listener,
                                repetitionPenaltyConfig = repetitionPenaltyConfig,
                                noRepeatNgramConfig = noRepeatNgramConfig,
                                suppressTokensConfig = suppressTokensConfig,
                                maxOutputToken = perCallMaxOutput,
                                thinkingConfig = perCallThinking,
                                responseFormat = responseFormat,
                            )
                        } catch (e: Exception) {
                            Log.e(TAG, "execute streaming failed", e)
                            errorRef.set(e)
                            onToken("Error: ${e.message}", true)
                            latch.countDown()
                        }

                        latch.await()
                        errorRef.get()?.let { throw LiteRTLMNativeException.from("execute", "Inference failed", it) }
                        fullResponseBuilder.toString()
                    } finally {
                        if (activeStreamingLatch === latch) {
                            activeStreamingLatch = null
                        }
                    }

                } else {
                    // ── Blocking path ─────────────────────────────────────────────────
                    val startTime = System.nanoTime()
                    val responseMsg = try {
                        conversation!!.sendMessage(
                            message = userMsg,
                            repetitionPenaltyConfig = repetitionPenaltyConfig,
                            noRepeatNgramConfig = noRepeatNgramConfig,
                            suppressTokensConfig = suppressTokensConfig,
                            maxOutputToken = perCallMaxOutput,
                            thinkingConfig = perCallThinking,
                            responseFormat = responseFormat,
                        )
                    } catch (e: Exception) {
                        throw LiteRTLMNativeException.from("execute", "Inference failed", e)
                    }
                    val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0

                    val textResponse = responseMsg.contents.contents
                        .filterIsInstance<Content.Text>()
                        .joinToString("") { it.text }

                    // Surface parsed tool calls (previously silently dropped) in the
                    // same <tool_call>{json}</tool_call> marker protocol the JS
                    // layer parses into typed events.
                    val response = textResponse + serializeToolCallMarkers(responseMsg.toolCalls)

                    history.add(Message(Role.MODEL, textResponse))

                    val promptTokens = userTextRepresentation.length / 4.0
                    val completionTokens = response.length / 4.0
                    lastStats = GenerationStats(
                        promptTokens = promptTokens,
                        completionTokens = completionTokens,
                        totalTokens = promptTokens + completionTokens,
                        timeToFirstToken = 0.0,
                        totalTime = elapsedMs,
                        tokensPerSecond = if (elapsedMs > 0) completionTokens / (elapsedMs / 1000.0) else 0.0
                    )
                    response
                }
            } finally {
                // Clean up all temp files created during this execute call
                for (f in tempFiles) {
                    try { f.delete() } catch (e: Exception) {
                        Log.w(TAG, "Failed to delete temp file: ${f.absolutePath}")
                    }
                }
            }
        }
    }

    override fun countTokens(text: String): Double {
        return -1.0
    }
}

/** Map the Nitro ThinkingOptions to the LiteRT-LM SDK config (engine defaults: enabled, unlimited budget). */
private fun ThinkingOptions.toSdkThinkingConfig(): ThinkingConfig =
    ThinkingConfig(
        enableThinking = enabled ?: true,
        thinkingTokenBudget = tokenBudget?.toInt() ?: -1,
    )

/**
 * Serialize SDK-parsed tool calls as `<tool_call>{"name":…,"arguments":{…}}</tool_call>`
 * markers — the shape `executeWithEvents()` documents and parses on the JS side.
 */
internal fun serializeToolCallMarkers(
    toolCalls: List<com.google.ai.edge.litertlm.ToolCall>?,
): String =
    toolCalls.orEmpty().joinToString("") { call ->
        val json = JSONObject()
            .put("name", call.name)
            .put("arguments", JSONObject(call.arguments))
        "<tool_call>$json</tool_call>"
    }

/** One engine-creation attempt in loadModel()'s fallback chain. */
private data class Attempt(val main: Backend, val vision: Backend?, val audio: Backend?)
