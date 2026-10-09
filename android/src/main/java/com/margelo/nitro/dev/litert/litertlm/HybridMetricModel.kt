///
/// HybridMetricModel.kt
/// Kotlin implementation of MetricModel HybridObject using LiteRT (TensorFlow Lite).
///

package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import org.tensorflow.lite.Interpreter
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * Kotlin implementation of MetricModel HybridObject using LiteRT.
 * Evaluates language along 13 different classification axes, converting
 * quantized [-128, 128] int8 outputs into [0.0, 1.0] float ratings.
 */
@DoNotStrip
@Keep
class HybridMetricModel : HybridMetricModelSpec() {

    companion object {
        private const val TAG = "HybridMetricModel"
        const val METRIC_DIMENSION_COUNT = 13

        val METRIC_INDEX_MAP: Array<String> = arrayOf(
            "Warmth",                  // 0
            "Sentiment",               // 1
            "Directness",              // 2
            "Passive-Aggressive",      // 3
            "Chemistry",               // 4
            "Professional Impression", // 5
            "Urgency",                 // 6
            "Formality",               // 7
            "Tension",                 // 8
            "Empathy",                 // 9
            "Clarity",                 // 10
            "Assertiveness",           // 11
            "Playfulness"              // 12
        )

        const val PAD_TOKEN_ID: Long = 0L
        const val BOS_TOKEN_ID: Long = 1L
        const val EOS_TOKEN_ID: Long = 2L
        const val UNK_TOKEN_ID: Long = 3L
        const val DEFAULT_MAX_SEQ_LENGTH = 128
        const val DEFAULT_VOCAB_SIZE = 30000

        /**
         * Deterministic hash-based subword tokenizer matching DefaultMetricTokenizer.
         */
        fun hashToken(token: String, vocabSize: Int = DEFAULT_VOCAB_SIZE): Long {
            var hash = 5381L
            for (ch in token) {
                hash = (hash * 33) xor ch.code.toLong()
            }
            val positive = kotlin.math.abs(hash)
            return (positive % (vocabSize - 4)) + 4L
        }

        /**
         * Tokenizes a text string into token IDs and attention mask arrays.
         */
        fun tokenizeText(
            text: String,
            seqLength: Int = DEFAULT_MAX_SEQ_LENGTH,
            vocabSize: Int = DEFAULT_VOCAB_SIZE
        ): Pair<LongArray, LongArray> {
            val tokenIds = LongArray(seqLength) { PAD_TOKEN_ID }
            val attentionMask = LongArray(seqLength) { 0L }

            val cleaned = text.trim().lowercase()
            if (cleaned.isEmpty()) {
                if (seqLength >= 2) {
                    tokenIds[0] = BOS_TOKEN_ID
                    tokenIds[1] = EOS_TOKEN_ID
                    attentionMask[0] = 1L
                    attentionMask[1] = 1L
                }
                return Pair(tokenIds, attentionMask)
            }

            val words = Regex("\\w+|[^\\s\\w]+").findAll(cleaned).map { it.value }.toList()
            val maxWords = max(0, seqLength - 2)
            val bounded = words.take(maxWords)

            tokenIds[0] = BOS_TOKEN_ID
            attentionMask[0] = 1L

            var idx = 1
            for (word in bounded) {
                tokenIds[idx] = hashToken(word, vocabSize)
                attentionMask[idx] = 1L
                idx++
            }

            tokenIds[idx] = EOS_TOKEN_ID
            attentionMask[idx] = 1L

            return Pair(tokenIds, attentionMask)
        }
    }

    private val lock = Any()
    private var interpreter: Interpreter? = null
    @Volatile
    private var isLoaded: Boolean = false
    private var expectedSeqLen: Int = DEFAULT_MAX_SEQ_LENGTH

    override fun loadModel(modelPath: String, config: MetricModelConfig?): Promise<Unit> {
        return Promise.parallel {
            synchronized(lock) {
                try {
                    val cleanPath = if (modelPath.startsWith("file://")) modelPath.substring(7) else modelPath
                    val modelFile = File(cleanPath)
                    if (!modelFile.exists()) {
                        throw IllegalArgumentException("Metric model file not found at: $cleanPath")
                    }

                    interpreter?.close()
                    interpreter = null
                    isLoaded = false

                    val numThreads = config?.numThreads?.toInt() ?: 2
                    val options = Interpreter.Options().apply {
                        setNumThreads(numThreads)
                    }

                    val interp = Interpreter(modelFile, options)
                    val inputCount = interp.inputTensorCount
                    if (inputCount < 1) {
                        interp.close()
                        throw IllegalStateException("Model has no input tensors")
                    }

                    val outputCount = interp.outputTensorCount
                    if (outputCount < 1) {
                        interp.close()
                        throw IllegalStateException("Model has no output tensors")
                    }

                    val inShape0 = interp.getInputTensor(0).shape()
                    val expectedSeq = if (inShape0.size >= 2 && inShape0[1] > 0) inShape0[1] else (config?.maxSequenceLength?.toInt() ?: DEFAULT_MAX_SEQ_LENGTH)

                    expectedSeqLen = expectedSeq
                    interpreter = interp
                    isLoaded = true
                    Log.i(TAG, "Loaded metric model from $cleanPath with seqLen=$expectedSeqLen, threads=$numThreads")
                } catch (e: Exception) {
                    isLoaded = false
                    interpreter?.close()
                    interpreter = null
                    Log.e(TAG, "Failed to load metric model: ${e.message}", e)
                    throw e
                }
            }
        }
    }

    override fun evaluate(input: MetricEvaluationInput): Promise<MetricEvaluationResult> {
        return Promise.parallel {
            val startTime = System.nanoTime()
            val interp: Interpreter
            val seqLen: Int

            synchronized(lock) {
                if (!isLoaded || interpreter == null) {
                    throw IllegalStateException("Metric model is not loaded. Call loadModel() first.")
                }
                interp = interpreter!!
                seqLen = input.maxLength?.toInt() ?: expectedSeqLen
            }

            val tokenIdsArr: LongArray
            val attentionMaskArr: LongArray

            if (input.tokenIds != null && input.tokenIds!!.isNotEmpty()) {
                val userTokens = input.tokenIds!!
                val userMask = input.attentionMask
                tokenIdsArr = LongArray(seqLen) { PAD_TOKEN_ID }
                attentionMaskArr = LongArray(seqLen) { 0L }

                val count = min(seqLen, userTokens.size)
                for (i in 0 until count) {
                    tokenIdsArr[i] = userTokens[i].toLong()
                    attentionMaskArr[i] = if (userMask != null && i < userMask.size) userMask[i].toLong() else 1L
                }
            } else if (input.text != null) {
                val pair = tokenizeText(input.text!!, seqLen)
                tokenIdsArr = pair.first
                attentionMaskArr = pair.second
            } else {
                throw IllegalArgumentException("Either 'text' or 'tokenIds' must be provided to evaluate().")
            }

            val inputTensors = arrayOfNulls<Any>(interp.inputTensorCount)
            val inTensor0 = interp.getInputTensor(0)
            if (inTensor0.dataType() == org.tensorflow.lite.DataType.INT32) {
                val intTokens = Array(1) { IntArray(seqLen) { i -> tokenIdsArr[i].toInt() } }
                inputTensors[0] = intTokens
            } else {
                val longTokens = Array(1) { tokenIdsArr }
                inputTensors[0] = longTokens
            }

            if (interp.inputTensorCount >= 2) {
                val inTensor1 = interp.getInputTensor(1)
                if (inTensor1.dataType() == org.tensorflow.lite.DataType.INT32) {
                    val intMask = Array(1) { IntArray(seqLen) { i -> attentionMaskArr[i].toInt() } }
                    inputTensors[1] = intMask
                } else {
                    val longMask = Array(1) { attentionMaskArr }
                    inputTensors[1] = longMask
                }
            }

            val outTensor = interp.getOutputTensor(0)
            val outputDimensionCount = if (outTensor.shape().size >= 2 && outTensor.shape()[1] > 0) outTensor.shape()[1] else METRIC_DIMENSION_COUNT
            val outputBytes = Array(1) { ByteArray(outputDimensionCount) }
            val outputsMap = mapOf<Int, Any>(0 to outputBytes)

            synchronized(lock) {
                if (!isLoaded || interpreter == null) {
                    throw IllegalStateException("Metric model was unloaded during evaluation preparation.")
                }
                interp.runForMultipleInputsOutputs(inputTensors, outputsMap)
            }

            val rawBytes = outputBytes[0]
            val scores = mutableMapOf<String, Double>()
            val values = mutableListOf<Double>()
            val rawValues = mutableListOf<Double>()

            for (i in 0 until outputDimensionCount) {
                val rawInt = rawBytes[i].toInt() // -128..127
                val clamped = max(-128, min(128, rawInt))
                // Convert quantized [-128, 128] to [0.0, 1.0]
                val normalized = max(0.0, min(1.0, (clamped + 128.0) / 256.0))

                val axisName = if (i < METRIC_INDEX_MAP.size) METRIC_INDEX_MAP[i] else "Metric_$i"
                scores[axisName] = normalized
                values.add(normalized)
                rawValues.add(clamped.toDouble())
            }

            val latencyMs = (System.nanoTime() - startTime) / 1_000_000.0

            MetricEvaluationResult(
                scores = scores,
                values = values.toDoubleArray(),
                rawValues = rawValues.toDoubleArray(),
                latencyMs = latencyMs
            )
        }
    }

    override fun isReady(): Boolean {
        synchronized(lock) {
            return isLoaded && interpreter != null
        }
    }

    override fun unload(): Promise<Unit> {
        return Promise.parallel {
            synchronized(lock) {
                interpreter?.close()
                interpreter = null
                isLoaded = false
            }
        }
    }

    override fun close() {
        synchronized(lock) {
            interpreter?.close()
            interpreter = null
            isLoaded = false
        }
    }
}
