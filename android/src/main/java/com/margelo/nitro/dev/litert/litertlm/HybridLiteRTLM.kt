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
import com.google.ai.edge.litertlm.OpenApiTool
import com.google.ai.edge.litertlm.ToolProvider
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference


// Alias to avoid confusion with our generated Message type
// Alias to avoid confusion
typealias LiteRTMessage = com.google.ai.edge.litertlm.Message

/**
 * Named implementation of the LiteRT-LM MessageCallback for streaming inference.
 *
 * Extracted from the anonymous inline class in sendMessageAsync for testability.
 * Accumulates response chunks, forwards tokens to JS, and appends the final
 * response to the conversation history.
 */
internal class StreamingCallbackListener(
    private val onToken: (String, Boolean) -> Unit,
    private val responseBuilder: StringBuilder,
    private val history: MutableList<Message>,
    private val userMessage: String,
    private val onStatsReady: (GenerationStats) -> Unit,
) : com.google.ai.edge.litertlm.MessageCallback {

    private val startTime = System.nanoTime()
    private var firstTokenTime = 0L
    private var tokenCount = 0

    override fun onMessage(message: com.google.ai.edge.litertlm.Message) {
        val chunk = message.contents.contents
            .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
            .joinToString("") { it.text }

        if (firstTokenTime == 0L && chunk.isNotEmpty()) {
            firstTokenTime = System.nanoTime()
        }
        if (chunk.isNotEmpty()) {
            tokenCount++
        }

        onToken(chunk, false)

        if (chunk.isNotEmpty()) {
            responseBuilder.append(chunk)
        }
    }

    override fun onDone() {
        onToken("", true)
        val fullResponse = responseBuilder.toString()
        history.add(Message(Role.MODEL, fullResponse))

        // Compute stats using heuristic token counts (~4 chars/token)
        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
        val ttftMs = if (firstTokenTime > 0) (firstTokenTime - startTime) / 1_000_000.0 else 0.0
        val promptTokens = userMessage.length / 4.0
        val completionTokens = fullResponse.length / 4.0
        onStatsReady(GenerationStats(
            promptTokens = promptTokens,
            completionTokens = completionTokens,
            totalTokens = promptTokens + completionTokens,
            timeToFirstToken = ttftMs,
            totalTime = elapsedMs,
            tokensPerSecond = if (elapsedMs > 0) completionTokens / (elapsedMs / 1000.0) else 0.0
        ))

        Log.d("StreamingCallbackListener", "Streaming done. Length: ${fullResponse.length}, TTFT: ${ttftMs.toLong()}ms, Total: ${elapsedMs.toLong()}ms")
    }

    override fun onError(throwable: Throwable) {
        Log.e("StreamingCallbackListener", "Async generation failed", throwable)
        onToken("Error: ${throwable.message}", true)
    }
}

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

        /** Cached result of OpenCL availability probe (null = not yet checked). */
        @Volatile
        private var openCLAvailable: Boolean? = null
        
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
    private var maxTokens: Int = 1024
    private var systemPrompt: String? = null
    private var tools: Array<ToolDefinition>? = null
    private var enableSpeculativeDecoding: Boolean = false

    override val memorySize: Long
        get() = 1024L * 1024L * 1024L // ~1GB (models are large)

    // -------------------------------------------------------------------------
    // loadModel - Initialize LiteRT-LM Engine and Conversation
    // -------------------------------------------------------------------------
    override fun loadModel(modelPath: String, config: LLMConfig?): Promise<Unit> {
        return Promise.parallel {
            // Serialize initialization to prevent OOM from concurrent loads
            synchronized(initLock) {
                if (isClosed) {
                    throw RuntimeException("Cannot load model: LiteRTLM instance is closed")
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
                    cfg.maxTokens?.let { maxTokens = it.toInt() }
                    cfg.systemPrompt?.let { systemPrompt = it }
                    tools = cfg.tools
                    enableSpeculativeDecoding = cfg.enableSpeculativeDecoding ?: false
                }
    
                // Whether to run engine validation after loading
                val shouldValidate = config?.validate?: false
    
                try {
                    // Early GPU hardware check: probe for OpenCL library before
                    // spending time on engine creation. LiteRT-LM's GPU delegate
                    // requires OpenCL, which is absent on most Samsung/Qualcomm devices.
                    if (backend == Backend.GPU) {
                        val hasOpenCL = openCLAvailable ?: run {
                            val result = try {
                                System.loadLibrary("OpenCL")
                                true
                            } catch (_: UnsatisfiedLinkError) {
                                try {
                                    // Some devices have it at a non-standard path
                                    System.load("/system/vendor/lib64/libOpenCL.so")
                                    true
                                } catch (_: UnsatisfiedLinkError) {
                                    false
                                }
                            }
                            openCLAvailable = result
                            result
                        }
                        if (!hasOpenCL) {
                            throw RuntimeException(
                                "GPU backend is not supported on this device (OpenCL library not found). " +
                                "Please use CPU backend instead."
                            )
                        }
                        Log.i(TAG, "OpenCL library found — GPU backend is available")
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
                    
                    // Detect multimodal support. Check config.multimodal flag first, then fall back to filename sniffing.
                    // Only Gemma 3n bundles vision/audio executors; Gemma 4 E2B is text-only.
                    // Passing vision/audio backends to a text-only model causes
                    // vision_litert_compiled_model_executor init failures.
                    val modelFileName = modelPath.substringAfterLast("/").lowercase()
                    val isMultimodal = config?.multimodal ?: (modelFileName.contains("3n") || modelFileName.contains("gemma3"))
    
                    val lmVisionBackend = if (isMultimodal) com.google.ai.edge.litertlm.Backend.GPU() else null
                    val lmAudioBackend = if (isMultimodal) com.google.ai.edge.litertlm.Backend.CPU() else null
    
                    Log.i(TAG, "Backend config: main=$lmBackend, vision=$lmVisionBackend, audio=$lmAudioBackend, multimodal=$isMultimodal")
    
                    // Get cache directory from application context
                    val cacheDirectory = LiteRTLMInitProvider.applicationContext?.cacheDir?.absolutePath
                    Log.i(TAG, "Using cache directory: $cacheDirectory")
    
                    // Create Engine configuration — visionBackend/audioBackend are optional
                    val engineConfig = if (isMultimodal) {
                        EngineConfig(
                            modelPath = modelPath,
                            backend = lmBackend,
                            visionBackend = lmVisionBackend!!,
                            audioBackend = lmAudioBackend!!,
                            maxNumTokens = maxTokens,
                            cacheDir = cacheDirectory
                        )
                    } else {
                        EngineConfig(
                            modelPath = modelPath,
                            backend = lmBackend,
                            maxNumTokens = maxTokens,
                            cacheDir = cacheDirectory
                        )
                    }
    
                    if (isClosed) return@synchronized
                    
                    if (enableSpeculativeDecoding) {
                        @OptIn(ExperimentalApi::class)
                        ExperimentalFlags.enableSpeculativeDecoding = true
                    }

                    // Initialize Engine
                    engine = Engine(engineConfig).also { it.initialize() }
                    Log.i(TAG, "Engine created and initialized successfully")
    
                    // Create Conversation
                    createNewConversation()
                    Log.i(TAG, "Conversation created successfully")

                    // Validate the engine actually works with a quick test inference.
                    // GPU/NPU backends can initialize without error but silently fail to
                    // produce tokens — enabling this catches those failures at load time.
                    // CPU is always reliable so validation is never run on it, even when
                    // the `validate` flag is set.
                    if (shouldValidate) {
                        if (backend == Backend.GPU || backend == Backend.NPU) {
                            validateEngine()
                        } else {
                            Log.i(TAG, "Validation skipped: CPU backend is always reliable")
                        }
                    }
    
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load model: ${e.message}", e)
                    // Clean up partial state so isReady() returns false
                    cleanupInternal()
                    throw RuntimeException("Failed to load model: ${e.message}", e)
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // sendMessage - Helper for one-shot generation (internally uses Async)
    // -------------------------------------------------------------------------
    override fun sendMessage(message: String): Promise<String> {
        // Implement Promise-based sendMessage using suspend coroutine logic wrapped in Promise
        // Since Promise.parallel expects a blocking block returning T, 
        // and sendMessageAsync is callback-based, we need to bridge them.
        // HOWEVER, we can just use the synchronous `sendMessage` API of the SDK 
        // inside the `Promise.parallel` block, which moves it off the main thread!
        return Promise.parallel {
            ensureLoaded()

            // Add user message to history
            history.add(Message(Role.USER, message))
            Log.i(TAG, "sendMessage (Promise): $message")
            
            // Blocking inference (safe here because we are in Promise.parallel worker thread)
            val userMsg = LiteRTMessage.user(message)
            val startTime = System.nanoTime()
            val responseMsg = conversation!!.sendMessage(message = userMsg)
            val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
            
            // Extract text
            val response = responseMsg.contents.contents
                .filterIsInstance<com.google.ai.edge.litertlm.Content.Text>()
                .joinToString("") { it.text } 
            
            // Add model response to history
            history.add(Message(Role.MODEL, response))
            
            // Update stats with real timing data
            // Token count heuristic: LiteRT-LM Android SDK does not expose
            // actual token counts from inference. We approximate using
            // ~4 chars/token. iOS uses the C API benchmark info for real counts.
            val promptTokens = message.length / 4.0
            val completionTokens = response.length / 4.0
            lastStats = GenerationStats(
                promptTokens = promptTokens,
                completionTokens = completionTokens,
                totalTokens = promptTokens + completionTokens,
                timeToFirstToken = 0.0, // Not available from sync API
                totalTime = elapsedMs,
                tokensPerSecond = if (elapsedMs > 0) completionTokens / (elapsedMs / 1000.0) else 0.0
            )
            
            response // Return the string
        }
    }

    // -------------------------------------------------------------------------
    // sendMessageAsync - Streaming inference
    // -------------------------------------------------------------------------
    override fun sendMessageAsync(message: String, onToken: (String, Boolean) -> Unit): Promise<Unit> {
        return Promise.parallel {
            val latch = CountDownLatch(1)
            val errorRef = AtomicReference<Throwable?>(null)

            ensureLoaded()

            // Add user message to history
            history.add(Message(Role.USER, message))
            Log.d(TAG, "sendMessageAsync: $message")

            val fullResponseBuilder = StringBuilder()
            
            val listener = StreamingCallbackListener(
                onToken = { token, done ->
                    onToken(token, done)
                    if (done) {
                        latch.countDown()
                    }
                },
                responseBuilder = fullResponseBuilder,
                history = history,
                userMessage = message,
                onStatsReady = { stats -> lastStats = stats },
            )

            try {
                val userMsg = LiteRTMessage.user(message)
                conversation!!.sendMessageAsync(message = userMsg, callback = listener)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initiate async generation", e)
                errorRef.set(e)
                onToken("Error: ${e.message}", true)
                latch.countDown()
            }

            // Wait for completion or error
            latch.await()
            val err = errorRef.get()
            if (err != null) {
                throw RuntimeException("Async inference failed: ${err.message}", err)
            }
        }
    }

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
            ?: throw RuntimeException("Failed to decode image: $imagePath")

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
        val tempFile = java.io.File(cacheDir, "resized_${System.currentTimeMillis()}.jpg")
        java.io.FileOutputStream(tempFile).use { out ->
            resizedBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 90, out)
        }
        resizedBitmap.recycle()

        Log.i(TAG, "Resized image saved to: ${tempFile.absolutePath} (${newWidth}x${newHeight})")
        return tempFile.absolutePath
    }

    override fun sendMessageWithImage(message: String, imagePath: String): Promise<String> {
        return Promise.parallel {
            ensureLoaded()
            Log.i(TAG, "sendMessageWithImage: $message, path=$imagePath")

            // Resize image to prevent OOM on high-resolution photos
            val processedImagePath = resizeImageIfNeeded(imagePath)

            // Create multimodal message
            // Use factory method Message.of passing a list of Content
            val textContent = Content.Text(message)
            
            val userMsg = LiteRTMessage.user(Contents.of(textContent, Content.ImageFile(processedImagePath)))

            // Add to history
            history.add(Message(Role.USER, "$message [Image]"))

            val responseMsg = conversation!!.sendMessage(message = userMsg)
            
            val response = responseMsg.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }

            history.add(Message(Role.MODEL, response))

            // Clean up temp resized image to prevent cache dir bloat
            if (processedImagePath != imagePath) {
                try {
                    java.io.File(processedImagePath).delete()
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to clean up temp image: ${e.message}")
                }
            }
            
            response
        }
    }

    override fun sendMessageWithImageAsync(message: String, imagePath: String, onToken: (String, Boolean) -> Unit): Promise<Unit> {
        return Promise.parallel {
            val latch = CountDownLatch(1)
            val errorRef = AtomicReference<Throwable?>(null)

            ensureLoaded()

            Log.i(TAG, "sendMessageWithImageAsync: $message, path=$imagePath")

            // Resize image to prevent OOM on high-resolution photos
            val processedImagePath = resizeImageIfNeeded(imagePath)

            val fullResponseBuilder = StringBuilder()
            
            val listener = StreamingCallbackListener(
                onToken = { token, done ->
                    onToken(token, done)
                    if (done) {
                        latch.countDown()
                    }
                },
                responseBuilder = fullResponseBuilder,
                history = history,
                userMessage = message,
                onStatsReady = { stats -> lastStats = stats },
            )

            try {
                val textContent = Content.Text(message)
                val userMsg = LiteRTMessage.user(Contents.of(textContent, Content.ImageFile(processedImagePath)))

                history.add(Message(Role.USER, "$message [Image]"))

                conversation!!.sendMessageAsync(message = userMsg, callback = listener)
            } catch (e: Exception) {
                // Clean up temp resized image to prevent cache dir bloat
                if (processedImagePath != imagePath) {
                    try {
                        java.io.File(processedImagePath).delete()
                    } catch (e: Exception) {
                        Log.w(TAG, "Failed to clean up temp image: ${e.message}")
                    }
                }

                Log.e(TAG, "Failed to initiate async multimodal generation", e)
                errorRef.set(e)
                onToken("Error: ${e.message}", true)
                latch.countDown()
            }

            // Wait for completion or error
            latch.await()

            // Clean up temp resized image to prevent cache dir bloat
            if (processedImagePath != imagePath) {
                try {
                    java.io.File(processedImagePath).delete()
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to clean up temp image: ${e.message}")
                }
            }

            val err = errorRef.get()
            if (err != null) {
                throw RuntimeException("Async multimodal inference failed: ${err.message}", err)
            }
        }
    }

    override fun downloadModel(url: String, fileName: String, onProgress: ((Double) -> Unit)?): Promise<String> {
        return Promise.parallel {
            Log.i(TAG, "downloadModel: $url -> $fileName")
            
            if (!url.startsWith("https://", ignoreCase = true)) {
                throw IllegalArgumentException("Invalid download URL: HTTPS is required for security.")
            }
            
            if (fileName.contains("..") || fileName.contains("/") || fileName.contains("\\")) {
                throw IllegalArgumentException("Invalid filename: path traversal or directory separators are not allowed.")
            }
            
            val context = LiteRTLMInitProvider.applicationContext ?: throw RuntimeException("Context not available")
            val modelsDir = java.io.File(context.filesDir, "models")
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }
            
            val modelFile = java.io.File(modelsDir, fileName)
            val tempFile = java.io.File(modelsDir, "$fileName.tmp")
            
            // Check if file exists and has content
            if (modelFile.exists() && modelFile.length() > 0) {
                Log.i(TAG, "Model already exists at: ${modelFile.absolutePath}")
                onProgress?.invoke(1.0)
                return@parallel modelFile.absolutePath
            }
            
            Log.i(TAG, "Downloading model to temp file: ${tempFile.absolutePath}")
            onProgress?.invoke(0.0)
            
            try {
                val connection = java.net.URL(url).openConnection() as java.net.HttpURLConnection
                connection.connectTimeout = 15000 // 15s
                connection.readTimeout = 0 // Infinite for large files
                connection.doInput = true
                connection.connect()
                
                if (connection.responseCode != java.net.HttpURLConnection.HTTP_OK) {
                    throw RuntimeException("Failed to download model: HTTP ${connection.responseCode}")
                }
                
                val contentLength = connection.contentLengthLong // Use long for large files
                val input = connection.inputStream
                val output = java.io.FileOutputStream(tempFile)
                
                val buffer = ByteArray(8 * 1024)
                var bytesRead: Int
                var totalBytesRead = 0L
                var lastProgressUpdate = 0L
                
                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead
                    
                    if (contentLength > 0 && onProgress != null) {
                        val currentTime = System.currentTimeMillis()
                        // Update roughly every 100ms to avoid flooding JS bridge
                        if (currentTime - lastProgressUpdate > 100) {
                            val progress = totalBytesRead.toDouble() / contentLength.toDouble()
                            onProgress(progress)
                            lastProgressUpdate = currentTime
                        }
                    }
                }
                
                output.flush()
                output.close()
                input.close()
                connection.disconnect()
                
                // Atomic rename
                if (tempFile.renameTo(modelFile)) {
                    Log.i(TAG, "Download complete and renamed to: ${modelFile.absolutePath}")
                    onProgress?.invoke(1.0)
                    return@parallel modelFile.absolutePath
                } else {
                    throw RuntimeException("Failed to rename temp file to model file")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                if (tempFile.exists()) {
                    tempFile.delete()
                }
                throw RuntimeException("Download failed: ${e.message}", e)
            }
        }
    }

    override fun deleteModel(fileName: String): Promise<Unit> {
        return Promise.parallel {
            Log.i(TAG, "deleteModel: $fileName")
            
            if (fileName.contains("..") || fileName.contains("/") || fileName.contains("\\")) {
                throw IllegalArgumentException("Invalid filename: path traversal or directory separators are not allowed.")
            }
            
            val context = LiteRTLMInitProvider.applicationContext ?: throw RuntimeException("Context not available")
            val modelsDir = java.io.File(context.filesDir, "models")
            val modelFile = java.io.File(modelsDir, fileName)
            
            if (modelFile.exists()) {
                val deleted = modelFile.delete()
                if (deleted) {
                    Log.i(TAG, "Deleted model: ${modelFile.absolutePath}")
                    // Ensure engine references are cleared if they point to this file
                    // We use cleanupInternal() which releases resources WITHOUT marking the instance as closed.
                    if (engine != null) {
                        Log.i(TAG, "Cleaning up engine after deleting model file.")
                        cleanupInternal()
                    }
                } else {
                    Log.e(TAG, "Failed to delete model: ${modelFile.absolutePath}")
                    throw RuntimeException("Failed to delete model: ${modelFile.absolutePath}")
                }
            } else {
                Log.w(TAG, "Model not found for deletion: ${modelFile.absolutePath}")
            }
        }
    }

    override fun sendMessageWithAudioAsync(message: String, audioPath: String, onToken: (String, Boolean) -> Unit): Promise<Unit> {
        return Promise.parallel {
            val latch = CountDownLatch(1)
            val errorRef = AtomicReference<Throwable?>(null)

            ensureLoaded()

            Log.i(TAG, "sendMessageWithAudioAsync: $message, path=$audioPath")

            val fullResponseBuilder = StringBuilder()

            val listener = StreamingCallbackListener(
                onToken = { token, done ->
                    onToken(token, done)
                    if (done) {
                        latch.countDown()
                    }
                },
                responseBuilder = fullResponseBuilder,
                history = history,
                userMessage = message,
                onStatsReady = { stats -> lastStats = stats },
            )

            try {
                val userMsg = LiteRTMessage.user(Contents.of(
                    Content.Text(message),
                    Content.AudioFile(audioPath)
                ))

                history.add(Message(Role.USER, "$message [Audio]"))

                conversation!!.sendMessageAsync(message = userMsg, callback = listener)
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initiate async audio generation", e)
                errorRef.set(e)
                onToken("Error: ${e.message}", true)
                latch.countDown()
            }

            latch.await()
            val err = errorRef.get()
            if (err != null) {
                throw RuntimeException("Async audio inference failed: ${err.message}", err)
            }
        }
    }

    override fun sendMessageWithAudio(message: String, audioPath: String): Promise<String> {
        return Promise.parallel {
            ensureLoaded()
            Log.i(TAG, "sendMessageWithAudio: $message, path=$audioPath")

            // Load audio
            
            val userMsg = LiteRTMessage.user(Contents.of(
                Content.Text(message),
                Content.AudioFile(audioPath)
            ))

            history.add(Message(Role.USER, "$message [Audio]"))
            
            val responseMsg = conversation!!.sendMessage(message = userMsg)
            
            val response = responseMsg.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }
                
            history.add(Message(Role.MODEL, response))
            
            response
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------
    override fun getHistory(): Array<Message> {
        // Synchronized list requires manual sync for iteration/copy
        synchronized(history) {
            return history.toTypedArray()
        }
    }

    override fun resetConversation() {
        synchronized(history) {
            history.clear()
        }
        createNewConversation()
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

    override fun getMemoryUsage(): MemoryUsage {
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

    override fun close() {
        Log.d(TAG, "Closing resources")
        isClosed = true
        cleanupInternal()
    }

    private fun cleanupInternal() {
        try {
            conversation?.close()
            conversation = null
            engine?.close()        // Direct call
            engine = null 
        } catch (e: Exception) {
            Log.e(TAG, "Error closing resources", e)
        }
    }

    private fun ensureLoaded() {
        if (engine == null) {
            throw RuntimeException("LiteRTLM: No model loaded. Call loadModel() first.")
        }
    }

    private fun createNewConversation() {
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
        // Map tools
        val lmTools: List<ToolProvider>? = tools?.map { tool ->
            val apiTool = object : OpenApiTool {
                override fun getToolDescriptionJsonString(): String {
                    return tool.parametersJson
                }
                override fun execute(paramsJsonString: String): String {
                    return "{}"
                }
            }
            (apiTool as Any) as ToolProvider
        }

        // Create conversation with explicit SamplerConfig (required by Gallery pattern).
        // GPU backend may fail silently without proper sampler params.
        val convConfig = ConversationConfig(
            samplerConfig = SamplerConfig(
                topK = topK,
                topP = topP.toDouble(),
                temperature = temperature.toDouble(),
            ),
            systemInstruction = systemPrompt?.let { Contents.of(Content.Text(it)) },
            tools = lmTools ?: emptyList()
        )
        conversation = engine!!.createConversation(convConfig)
    }

    /**
     * Validate that the engine can actually produce inference output.
     *
     * Some GPU backends initialize without error but silently hang during inference.
     * This sends a minimal test prompt ("Hi") and waits up to 30s for any token.
     * If no token arrives, we throw so the model does NOT appear as loaded.
     */
    private fun validateEngine() {
        val backendName = when (backend) {
            Backend.GPU -> "GPU"
            Backend.NPU -> "NPU"
            else -> "CPU"
        }
        Log.i(TAG, "Validating $backendName backend with test inference...")

        val latch = CountDownLatch(1)
        val gotToken = AtomicBoolean(false)
        val errorRef = AtomicReference<String?>(null)

        // Use the existing conversation for validation (single-session constraint).
        val validationConv = conversation
            ?: throw RuntimeException("$backendName backend: no conversation available for validation")

        try {
            val testMsg = LiteRTMessage.user("Hi")
            validationConv.sendMessageAsync(
                message = testMsg,
                callback = object : com.google.ai.edge.litertlm.MessageCallback {
                    override fun onMessage(msg: com.google.ai.edge.litertlm.Message) {
                        gotToken.set(true)
                        latch.countDown()
                    }
                    override fun onDone() {
                        latch.countDown()
                    }
                    override fun onError(t: Throwable) {
                        errorRef.set(t.message)
                        latch.countDown()
                    }
                }
            )
        } catch (e: Exception) {
            throw RuntimeException(
                "$backendName backend failed to run inference: ${e.message}. " +
                "This device may not support the $backendName backend. Please try CPU.",
                e
            )
        }

        // Wait up to 30s for any response
        val completed = latch.await(30, TimeUnit.SECONDS)

        val error = errorRef.get()
        if (error != null) {
            throw RuntimeException(
                "$backendName backend inference error: $error. " +
                "This device may not support the $backendName backend. Please try CPU."
            )
        }
        if (!completed || !gotToken.get()) {
            throw RuntimeException(
                "$backendName backend produced no response within 30 seconds. " +
                "This device may not support the $backendName backend. Please try CPU."
            )
        }

        Log.i(TAG, "$backendName backend validated successfully")

        // Re-create the real conversation (validation consumed one turn)
        createNewConversation()
    }

    override fun sendMultimodalMessage(parts: Array<MultimodalPart>): Promise<String> {
        return Promise.parallel {
            ensureLoaded()
            val contents = mutableListOf<Content>()
            var userTextRepresentation = ""
            for (part in parts) {
                when (part.type) {
                    PartType.TEXT -> part.text?.let { 
                        contents.add(Content.Text(it))
                        userTextRepresentation += "$it "
                    }
                    PartType.IMAGE -> part.imageBuffer?.let { buffer ->
                        val byteBuffer = buffer.getBuffer(false)
                        val bytes = ByteArray(byteBuffer.remaining())
                        byteBuffer.get(bytes)
                        contents.add(Content.ImageBytes(bytes))
                        userTextRepresentation += "[Image Buffer] "
                    }
                    PartType.AUDIO -> part.audioBuffer?.let { buffer ->
                        val byteBuffer = buffer.getBuffer(false)
                        val bytes = ByteArray(byteBuffer.remaining())
                        byteBuffer.get(bytes)
                        contents.add(Content.AudioBytes(bytes))
                        userTextRepresentation += "[Audio Buffer] "
                    }
                }
            }
            userTextRepresentation = userTextRepresentation.trim()
            history.add(Message(Role.USER, userTextRepresentation))

            val userMsg = LiteRTMessage.user(Contents.of(contents))
            val startTime = System.nanoTime()
            val responseMsg = conversation!!.sendMessage(message = userMsg)
            val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0

            val response = responseMsg.contents.contents
                .filterIsInstance<Content.Text>()
                .joinToString("") { it.text }

            history.add(Message(Role.MODEL, response))
            
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
    }

    override fun countTokens(text: String): Double {
        return -1.0
    }
}
