///
/// HybridLayaModel.kt
/// Kotlin implementation of LayaModel HybridObject using LiteRT.
///

package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.google.ai.edge.litert.Accelerator
import com.margelo.nitro.core.Promise
import com.margelo.nitro.dev.litert.litertlm.laya.DecisionGraph
import com.margelo.nitro.dev.litert.litertlm.laya.LayaCalibration
import com.margelo.nitro.dev.litert.litertlm.laya.LayaDecoder
import com.margelo.nitro.dev.litert.litertlm.laya.LayaEmbeddings
import com.margelo.nitro.dev.litert.litertlm.laya.LayaJson
import com.margelo.nitro.dev.litert.litertlm.laya.LayaPromptBuilder
import com.margelo.nitro.dev.litert.litertlm.laya.LayaTokenizer
import java.io.File

/**
 * Android native implementation of the Laya Multilingual LiteRT System-1 Decision Model.
 * Orchestrates memory-mapped embeddings table gather, pure Kotlin BPE tokenization,
 * prompt sequence assembly, main model inference on GPU/CPU, action head execution,
 * and calibrated softmax decoding.
 */
@DoNotStrip
@Keep
class HybridLayaModel : HybridLayaModelSpec() {

    companion object {
        private const val TAG = "HybridLayaModel"
        private const val DEFAULT_WINDOW = 256
        private const val DEFAULT_MAIN_GRAPH = "laya_ml_s256_embeds_wfp16.tflite"
        private const val DEFAULT_ACT_GRAPH = "laya_ml_act_head_fp32.tflite"
        private const val DEFAULT_CALIBRATION = "laya_ml_calibration.json"
        private const val DEFAULT_EMBEDDINGS_BIN = "token_embeddings_fp16.bin"
        private const val DEFAULT_EMBEDDINGS_JSON = "token_embeddings.json"
        private const val DEFAULT_TOKENIZER_JSON = "tokenizer.json"
    }

    private val lock = Any()

    private var tokenizer: LayaTokenizer? = null
    private var calibration: LayaCalibration = LayaCalibration.identity()
    private var embeddings: LayaEmbeddings? = null
    private var mainGraph: DecisionGraph? = null
    private var actGraph: DecisionGraph? = null

    private var currentWindow: Int = DEFAULT_WINDOW
    private var embedsBuffer: FloatArray = FloatArray(DEFAULT_WINDOW * LayaEmbeddings.WIDTH)
    private var activeBackend: String = "none"

    @Volatile
    private var isModelLoaded: Boolean = false

    override fun loadModel(modelDir: String, config: LayaModelConfig?): Promise<Unit> {
        return Promise.parallel {
            synchronized(lock) {
                try {
                    val cleanPath = if (modelDir.startsWith("file://")) modelDir.substring(7) else modelDir
                    val dir = File(cleanPath)
                    if (!dir.exists() || !dir.isDirectory) {
                        throw IllegalArgumentException("Laya model directory not found at: $cleanPath")
                    }

                    closeInternal()

                    val window = config?.window?.toInt() ?: DEFAULT_WINDOW
                    val mainGraphName = config?.mainGraphName ?: DEFAULT_MAIN_GRAPH
                    val actGraphName = config?.actGraphName ?: DEFAULT_ACT_GRAPH
                    val calibrationName = config?.calibrationFile ?: DEFAULT_CALIBRATION
                    val numThreads = config?.numThreads?.toInt() ?: 4

                    val tokenizerFile = File(dir, DEFAULT_TOKENIZER_JSON)
                    if (!tokenizerFile.isFile) {
                        throw IllegalArgumentException("tokenizer.json not found in: $cleanPath")
                    }
                    val tokenizerInstance = LayaTokenizer(tokenizerFile)

                    val calibrationFile = File(dir, calibrationName)
                    val calibrationInstance = if (calibrationFile.isFile) {
                        LayaCalibration.load(calibrationFile)
                    } else {
                        Log.w(TAG, "Calibration file $calibrationName not found in $cleanPath; using uncalibrated temperatures")
                        LayaCalibration.identity()
                    }

                    val embeddingsBin = File(dir, DEFAULT_EMBEDDINGS_BIN)
                    val embeddingsMeta = File(dir, DEFAULT_EMBEDDINGS_JSON)
                    if (!embeddingsBin.isFile) {
                        throw IllegalArgumentException("Token embeddings table $DEFAULT_EMBEDDINGS_BIN not found in: $cleanPath")
                    }
                    val embeddingsInstance = LayaEmbeddings(embeddingsBin, embeddingsMeta)

                    val mainFile = File(dir, mainGraphName)
                    val actFile = File(dir, actGraphName)

                    val requestedAccelerator = config?.accelerator?.lowercase()
                    val tryGpu = requestedAccelerator != "cpu"

                    var compiledMain: DecisionGraph? = null
                    var compiledAct: DecisionGraph? = null
                    var chosenBackend = "cpu"

                    if (tryGpu) {
                        try {
                            compiledMain = DecisionGraph.create(
                                mainFile,
                                Accelerator.GPU,
                                listOf("attention_mask", "inputs_embeds", "qtype_onehot"),
                                listOf("pooled_cls", "token_logits")
                            )
                            compiledAct = DecisionGraph.create(
                                actFile,
                                Accelerator.GPU,
                                listOf("feats", "pooled_cls"),
                                listOf("act_logits")
                            )
                            chosenBackend = "gpu"
                            Log.i(TAG, "Successfully initialized Laya graphs with GPU accelerator (FP32)")
                        } catch (gpuError: Throwable) {
                            Log.w(TAG, "GPU initialization failed, falling back to CPU: ${gpuError.message}")
                            compiledMain?.close()
                            compiledMain = null
                            compiledAct?.close()
                            compiledAct = null
                        }
                    }

                    if (compiledMain == null || compiledAct == null) {
                        compiledMain = DecisionGraph.create(
                            mainFile,
                            Accelerator.CPU,
                            listOf("attention_mask", "inputs_embeds", "qtype_onehot"),
                            listOf("pooled_cls", "token_logits"),
                            numThreads
                        )
                        compiledAct = DecisionGraph.create(
                            actFile,
                            Accelerator.CPU,
                            listOf("feats", "pooled_cls"),
                            listOf("act_logits"),
                            numThreads
                        )
                        chosenBackend = "cpu"
                        Log.i(TAG, "Initialized Laya graphs with CPU accelerator ($numThreads threads)")
                    }

                    tokenizer = tokenizerInstance
                    calibration = calibrationInstance
                    embeddings = embeddingsInstance
                    mainGraph = compiledMain
                    actGraph = compiledAct
                    currentWindow = window
                    embedsBuffer = FloatArray(window * LayaEmbeddings.WIDTH)
                    activeBackend = chosenBackend
                    isModelLoaded = true

                    Log.i(TAG, "Loaded Laya model from $cleanPath (window=$window, backend=$chosenBackend)")
                } catch (e: Exception) {
                    closeInternal()
                    Log.e(TAG, "Failed to load Laya model: ${e.message}", e)
                    throw e
                }
            }
        }
    }

    override fun decide(request: LayaDecisionRequest): Promise<LayaDecisionResponse> {
        return Promise.parallel {
            val startTime = System.nanoTime()

            val tok: LayaTokenizer
            val cal: LayaCalibration
            val emb: LayaEmbeddings
            val main: DecisionGraph
            val act: DecisionGraph
            val window: Int
            val embeds: FloatArray

            synchronized(lock) {
                if (!isModelLoaded || mainGraph == null || actGraph == null || embeddings == null || tokenizer == null) {
                    throw IllegalStateException("Laya model is not loaded. Call loadModel() first.")
                }
                tok = tokenizer!!
                cal = calibration
                emb = embeddings!!
                main = mainGraph!!
                act = actGraph!!
                window = currentWindow
                embeds = embedsBuffer
            }

            // Parse criteria
            val criteriaParsed: Any? = when {
                !request.criteria.isNullOrBlank() -> {
                    try {
                        LayaJson.parse(request.criteria!!)
                    } catch (e: Exception) {
                        request.criteria
                    }
                }
                request.options != null -> {
                    val map = LinkedHashMap<String, Any?>()
                    request.options!!.forEach { map[it] = null }
                    map
                }
                else -> null
            }

            val questionMap = linkedMapOf<String, Any?>(
                "type" to request.type,
                "instructions" to request.instructions,
                "criteria" to criteriaParsed
            )

            val builder = LayaPromptBuilder(tok, maxLen = window, headMaxLen = window)
            val stateText = request.state ?: ""
            val sequence = builder.build(stateText, questionMap)

            synchronized(lock) {
                if (!isModelLoaded) {
                    throw IllegalStateException("Laya model was closed during inference preparation.")
                }

                emb.gather(sequence.ids, window, embeds)

                main.write("inputs_embeds", embeds)
                val attention = FloatArray(window) { if (it < sequence.ids.size) 1f else 0f }
                main.write("attention_mask", attention)
                val qtypeOneHot = FloatArray(3).also { it[sequence.question.qtype] = 1f }
                main.write("qtype_onehot", qtypeOneHot)

                main.run()

                val tokens = main.read("token_logits")
                val pooled = main.read("pooled_cls")

                check(tokens.all { it.isFinite() } && pooled.all { it.isFinite() }) {
                    "Laya model returned non-finite logits on $activeBackend"
                }

                val markers = LayaDecoder.gather(tokens, sequence.markers)

                act.write("pooled_cls", pooled)
                act.write("feats", LayaDecoder.actFeatures(markers))

                act.run()

                val actionLogits = act.read("act_logits")
                val answer = LayaDecoder.decode(markers, actionLogits, sequence.question, cal)

                val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0

                val type = answer["type"] as String
                val choice = answer["choice"] as? String
                val score = (answer["score"] as? Number)?.toDouble()
                val noul = (answer["noul"] as? Number)?.toDouble()
                val decision = if (type == "noul") (noul != null && noul >= 0.5) else null

                val rawProbabilities = answer["probabilities"]
                val probabilitiesMap = mutableMapOf<String, Double>()
                if (rawProbabilities is Map<*, *>) {
                    for ((k, v) in rawProbabilities) {
                        if (k != null && v is Number) {
                            probabilitiesMap[k.toString()] = v.toDouble()
                        }
                    }
                }

                val confidence = (answer["confidence"] as? Number)?.toDouble() ?: 0.0
                val actionMap = answer["action"] as? Map<*, *>
                val actProbability = (actionMap?.get("act_probability") as? Number)?.toDouble() ?: 1.0

                LayaDecisionResponse(
                    type = type,
                    choice = choice,
                    score = score,
                    noul = noul,
                    decision = decision,
                    probabilities = probabilitiesMap,
                    confidence = confidence,
                    actProbability = actProbability,
                    latencyMs = latencyMs
                )
            }
        }
    }

    override fun isReady(): Boolean {
        synchronized(lock) {
            return isModelLoaded && mainGraph != null && actGraph != null
        }
    }

    override fun unload(): Promise<Unit> {
        return Promise.parallel {
            synchronized(lock) {
                closeInternal()
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            closeInternal()
        }
    }

    private fun closeInternal() {
        isModelLoaded = false
        activeBackend = "none"
        try {
            mainGraph?.close()
        } catch (_: Throwable) {}
        mainGraph = null

        try {
            actGraph?.close()
        } catch (_: Throwable) {}
        actGraph = null

        try {
            embeddings?.close()
        } catch (_: Throwable) {}
        embeddings = null

        tokenizer = null
        calibration = LayaCalibration.identity()
    }
}
