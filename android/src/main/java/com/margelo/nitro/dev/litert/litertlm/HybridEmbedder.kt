package com.margelo.nitro.dev.litert.litertlm

import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.google.ai.edge.litertlm.EmbeddingEngine
import com.google.ai.edge.litertlm.EmbeddingEngineConfig
import com.google.ai.edge.litertlm.InputData
import com.margelo.nitro.core.ArrayBuffer
import com.margelo.nitro.core.Promise
import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import com.google.ai.edge.litertlm.EmbeddingOptions as SdkEmbeddingOptions

/**
 * On-device embeddings through the LiteRT-LM SDK's EmbeddingEngine (0.17+):
 * text, image and audio inputs to one vector, returned as a native-owned
 * ArrayBuffer of native-order float32 values (a Float32Array in JS).
 */
@DoNotStrip
@Keep
class HybridEmbedder : HybridEmbedderSpec() {
    /**
     * Engine work runs here, one task at a time in call order (like the iOS
     * serial queue), so the JS thread never waits on it. The thread exits
     * when idle.
     */
    private val worker = ThreadPoolExecutor(0, 1, 30L, TimeUnit.SECONDS, LinkedBlockingQueue<Runnable>()) { task ->
        Thread(task, "litert-embedding").apply { isDaemon = true }
    }

    /** Serializes the blocking bodies, which on-device tests also call directly. */
    private val lock = Any()

    @Volatile
    private var engine: EmbeddingEngine? = null

    /** What the JS thread reads; cleared at once by [close]. */
    @Volatile
    private var ready = false

    @Volatile
    private var loadedModelSizeBytes = 0L

    override val memorySize: Long
        get() = loadedModelSizeBytes

    private fun <T> onWorker(run: () -> T): Promise<T> {
        val promise = Promise<T>()
        worker.execute {
            try {
                promise.resolve(run())
            } catch (e: Throwable) {
                promise.reject(e)
            }
        }
        return promise
    }

    // ── Lifecycle ───────────────────────────────────────────────────────────

    override fun loadModel(modelPath: String, config: EmbedderConfig?): Promise<Unit> =
        onWorker { loadModelBlocking(modelPath, config) }

    /** The body of [loadModel]; internal so on-device tests can call it directly. */
    internal fun loadModelBlocking(modelPath: String, config: EmbedderConfig?) {
        synchronized(lock) {
            unloadInternal()
            if (!File(modelPath).exists()) {
                throw LiteRTLMNativeException("load", "NOT_FOUND", "MODEL_NOT_FOUND", "Model file not found: $modelPath")
            }
            // Encoders default to the bundle's own declaration; an explicit
            // backend wins. Bundle metadata can claim an encoder the file lacks
            // (audio has no signal to check), so retry without encoders the
            // caller didn't ask for — a failed creation is fast.
            val plan = ModelBundle.encoderPlan(modelPath)
            val vision = config?.visionBackend ?: plan?.vision
            val audio = config?.audioBackend ?: plan?.audio
            val attempts = listOf(
                vision to audio,
                vision to config?.audioBackend,
                config?.visionBackend to audio,
                config?.visionBackend to config?.audioBackend,
            ).distinct()
            var lastError: Exception? = null
            for ((attemptVision, attemptAudio) in attempts) {
                val engineConfig = EmbeddingEngineConfig(
                    modelPath = modelPath,
                    backend = (config?.backend ?: Backend.CPU).toSdk(),
                    visionBackend = attemptVision?.toSdk(),
                    audioBackend = attemptAudio?.toSdk(),
                    cacheDir = LiteRTLMInitProvider.applicationContext?.cacheDir?.absolutePath,
                    maxInputLength = config?.maxInputLength?.toInt(),
                    visionTokensPerImage = config?.visionTokensPerImage?.toInt(),
                    activationDataType = config?.activationDataType?.toSdk(),
                )
                try {
                    engine = EmbeddingEngine(engineConfig).also { it.initialize() }
                    loadedModelSizeBytes = File(modelPath).length()
                    ready = true
                    return
                } catch (e: Exception) {
                    lastError = e
                }
            }
            throw LiteRTLMNativeException.from(
                "load", "Failed to create the embedding engine",
                lastError ?: IllegalStateException("no engine was created"),
            )
        }
    }

    override fun isReady(): Boolean = ready

    override fun getMemoryUsage(): MemoryUsage = HybridLiteRTLM.processMemoryUsage()

    override fun unload(): Promise<Unit> = onWorker { synchronized(lock) { unloadInternal() } }

    /**
     * Returns at once: the instance reports not-ready immediately, and the
     * engine is freed on the worker as soon as any in-flight work finishes.
     */
    override fun close() {
        ready = false
        worker.execute { synchronized(lock) { unloadInternal() } }
    }

    private fun unloadInternal() {
        ready = false
        try {
            engine?.close()
        } catch (_: Exception) {
        }
        engine = null
        loadedModelSizeBytes = 0L
    }

    // ── Embedding ───────────────────────────────────────────────────────────

    override fun embed(parts: Array<MultimodalPart>, options: EmbeddingOptions?): Promise<ArrayBuffer> {
        val inputs = try {
            parts.map { it.capture() }
        } catch (e: Exception) {
            return Promise.rejected(e)
        }
        return onWorker { embedBlocking(inputs.map { it() }, options) }
    }

    override fun embedBatch(inputs: Array<Array<MultimodalPart>>, options: EmbeddingOptions?): Promise<Array<ArrayBuffer>> {
        val batch = try {
            inputs.map { group -> group.map { it.capture() } }
        } catch (e: Exception) {
            return Promise.rejected(e)
        }
        return onWorker { embedBatchBlocking(batch.map { group -> group.map { it() } }, options) }
    }

    internal fun embedBlocking(inputs: List<InputData>, options: EmbeddingOptions?): ArrayBuffer =
        synchronized(lock) {
            val response = try {
                requireEngine().computeEmbedding(inputs, options.toSdk())
            } catch (e: LiteRTLMNativeException) {
                throw e
            } catch (e: Exception) {
                throw LiteRTLMNativeException.from("embed", "Embedding failed", e)
            }
            response.embedding.toArrayBuffer()
        }

    internal fun embedBatchBlocking(batch: List<List<InputData>>, options: EmbeddingOptions?): Array<ArrayBuffer> =
        synchronized(lock) {
            if (batch.isEmpty()) return@synchronized emptyArray()
            val responses = try {
                requireEngine().computeEmbeddingBatch(batch, options.toSdk())
            } catch (e: LiteRTLMNativeException) {
                throw e
            } catch (e: Exception) {
                throw LiteRTLMNativeException.from("embed", "Batch embedding failed", e)
            }
            responses.map { it.embedding.toArrayBuffer() }.toTypedArray()
        }

    private fun requireEngine(): EmbeddingEngine =
        engine ?: throw LiteRTLMNativeException(
            "embed", "FAILED_PRECONDITION", "MODEL_NOT_LOADED",
            "No embedding model loaded. Call loadModel() first.",
        )
}

/**
 * Captures a part on the JS thread. A JS buffer is copied now (it isn't ours
 * after the call); a file is read when the returned function runs, on the
 * worker.
 */
private fun MultimodalPart.capture(): () -> InputData = when (type) {
    PartType.TEXT -> {
        val value = text ?: ""
        { InputData.Text(value) }
    }
    PartType.IMAGE -> media(path, imageBuffer, "Image") { InputData.Image(it) }
    PartType.AUDIO -> media(path, audioBuffer, "Audio") { InputData.Audio(it) }
    PartType.TOOLRESULT -> throw LiteRTLMNativeException(
        "embed", "INVALID_ARGUMENT", "INVALID_INPUT", "Embeddings take text, image and audio parts only",
    )
}

private fun media(path: String?, buffer: ArrayBuffer?, label: String, make: (ByteArray) -> InputData): () -> InputData {
    if (path != null) {
        return {
            val file = File(path)
            if (!file.exists()) {
                throw LiteRTLMNativeException("embed", "NOT_FOUND", "INVALID_INPUT", "$label file not found: $path")
            }
            make(file.readBytes())
        }
    }
    if (buffer != null) {
        val source = buffer.getBuffer(false)
        val bytes = ByteArray(source.remaining()).also { source.get(it) }
        return { make(bytes) }
    }
    throw LiteRTLMNativeException("embed", "INVALID_ARGUMENT", "INVALID_INPUT", "$label part needs a path or a buffer")
}

private fun EmbeddingOptions?.toSdk(): SdkEmbeddingOptions = SdkEmbeddingOptions(
    normalize = this?.normalize,
    insertSpecialTokens = this?.insertSpecialTokens,
    outputSize = this?.outputSize?.toInt(),
    visionTokensPerImage = this?.visionTokensPerImage?.toInt(),
)

/** Native byte order, so JS reads the buffer as a Float32Array directly. */
private fun FloatArray.toArrayBuffer(): ArrayBuffer {
    val buffer = ArrayBuffer.allocate(size * Float.SIZE_BYTES)
    buffer.getBuffer(false).order(ByteOrder.nativeOrder()).asFloatBuffer().put(this)
    return buffer
}
