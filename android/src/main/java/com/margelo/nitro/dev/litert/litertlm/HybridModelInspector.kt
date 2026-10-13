package com.margelo.nitro.dev.litert.litertlm

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import com.google.ai.edge.litertlm.BackendType as SdkBackendType
import com.google.ai.edge.litertlm.Modality as SdkModality
import com.google.ai.edge.litertlm.ModelInfo as SdkModelInfo
import com.google.ai.edge.litertlm.ModelType as SdkModelType
import com.google.ai.edge.litertlm.NpuBrand as SdkNpuBrand

/**
 * Reads .litertlm bundle metadata through the LiteRT-LM SDK's ModelInfo
 * (0.18.0+) without creating an engine, and describes the device. Backs
 * inspectModel() / recommendBackend() and loadModel()'s modality setup.
 */
@DoNotStrip
@Keep
class HybridModelInspector : HybridModelInspectorSpec() {
    override fun inspect(modelPath: String): Promise<NativeModelInfo> =
        Promise.parallel { ModelBundle.read(modelPath) }

    override fun getDeviceInfo(): NativeDeviceInfo = DeviceFacts.current()
}

/**
 * Bundle metadata reads. Not free: besides the header, the engine reads each
 * vision encoder section into memory to learn its signatures (hundreds of MB
 * on a multimodal bundle). Readings are cached per file version, so
 * inspectModel() and the loadModel() that follows read a bundle once.
 */
internal object ModelBundle {
    private data class FileVersion(val size: Long, val modified: Long)

    private val cache = java.util.concurrent.ConcurrentHashMap<String, Pair<FileVersion, NativeModelInfo>>()

    private val MODALITIES = listOf(
        Modality.TEXT to SdkModality.TEXT,
        Modality.VISION to SdkModality.VISION,
        Modality.AUDIO to SdkModality.AUDIO,
        Modality.VIDEO to SdkModality.VIDEO,
    )

    private fun open(path: String): SdkModelInfo {
        if (!File(path).exists()) {
            throw LiteRTLMNativeException("inspect", "NOT_FOUND", "MODEL_NOT_FOUND", "Model file not found: $path")
        }
        return try {
            SdkModelInfo.from(path)
        } catch (e: Exception) {
            throw LiteRTLMNativeException.from("inspect", "Not a readable LiteRT-LM bundle", e)
        }
    }

    fun read(path: String): NativeModelInfo {
        val file = File(path)
        val version = if (file.exists()) FileVersion(file.length(), file.lastModified()) else null
        if (version != null) {
            cache[path]?.let { (cached, info) -> if (cached == version) return info }
        }
        return readUncached(path).also { info -> if (version != null) cache[path] = version to info }
    }

    private fun readUncached(path: String): NativeModelInfo = open(path).use { info ->
        val supported = info.inputModalities()
        val modalities = MODALITIES.map { (modality, sdk) ->
            NativeModalityInfo(
                modality = modality,
                supported = when (modality) {
                    Modality.TEXT -> supported.text
                    Modality.VISION -> supported.vision
                    Modality.AUDIO -> supported.audio
                    Modality.VIDEO -> supported.video
                },
                backends = info.supportedBackends(sdk).mapNotNull { it.toNitro() }.toTypedArray(),
                npuBrand = info.npuBrand(sdk).toNitro(),
                socName = info.socName(sdk)?.takeIf { it.isNotEmpty() },
            )
        }
        val llm = info as? SdkModelInfo.Llm
        val embedding = info as? SdkModelInfo.Embedding
        val sampler = llm?.defaultSamplerParams()
        NativeModelInfo(
            modelType = when (info.modelType) {
                SdkModelType.LLM -> ModelType.LLM
                SdkModelType.EMBEDDING -> ModelType.EMBEDDING
                else -> ModelType.UNKNOWN
            },
            fileSizeBytes = File(path).length().toDouble(),
            maxContextTokens = info.maxContextTokens().toDouble(),
            isDynamicContext = info.isDynamicContext(),
            minRuntimeVersion = info.minRuntimeVersion()?.takeIf { it.isNotEmpty() },
            modalities = modalities.toTypedArray(),
            supportsThinking = llm?.supportsThinking() ?: false,
            supportsFunctionCalling = llm?.supportsFunctionCalling() ?: false,
            hasSpeculativeDecoding = llm?.hasSpeculativeDecodingSupport() ?: false,
            samplerType = (sampler?.type ?: 0).toDouble(),
            samplerTemperature = (sampler?.temperature ?: 0f).toDouble(),
            samplerTopK = (sampler?.topK ?: 0).toDouble(),
            samplerTopP = (sampler?.topP ?: 0f).toDouble(),
            maxVisionTokenBudget = info.maxVisionTokenBudget().toDouble(),
            visionSignatureSelection = info.visionSignatureSelection().toDoubles(),
            embeddingDimension = (embedding?.dimension() ?: -1).toDouble(),
            embeddingSignatureSelection = embedding?.signatureSelection().toDoubles(),
        )
    }

    /** The backend each encoder should run on (null = the bundle lacks it). */
    data class EncoderPlan(val vision: Backend?, val audio: Backend?)

    /**
     * The first backend the bundle declares for each encoder that this device
     * can run, or null for an encoder the bundle lacks. Undeclared encoders
     * keep 0.7's choice: vision on GPU when an OpenCL driver is present (else
     * CPU — without one, LiteRT falls back to WebGPU, which on an emulator is
     * software Vulkan), audio on CPU. Null when the metadata is unreadable.
     */
    fun encoderPlan(path: String): EncoderPlan? {
        val info = try {
            read(path)
        } catch (e: Exception) {
            return null
        }
        fun plan(modality: Modality, undeclared: Backend): Backend? {
            val m = info.modalities.firstOrNull { it.modality == modality }
            if (m == null || !m.supported) return null
            // A real vision encoder always reports its compiled signatures; the
            // text-only EmbeddingGemma 2 bundle claims vision support without one.
            if (modality == Modality.VISION && info.visionSignatureSelection.isEmpty()) return null
            if (m.backends.isEmpty()) return undeclared
            return m.backends.firstOrNull { it == Backend.CPU || (it == Backend.GPU && GpuProbe.openCLAvailable()) }
                ?: Backend.CPU
        }
        return EncoderPlan(
            vision = plan(Modality.VISION, if (GpuProbe.openCLAvailable()) Backend.GPU else Backend.CPU),
            audio = plan(Modality.AUDIO, Backend.CPU),
        )
    }

    private fun IntArray?.toDoubles(): DoubleArray =
        this?.map { it.toDouble() }?.toDoubleArray() ?: DoubleArray(0)

    private fun SdkBackendType.toNitro(): Backend? = when (this) {
        SdkBackendType.CPU -> Backend.CPU
        SdkBackendType.GPU -> Backend.GPU
        SdkBackendType.NPU -> Backend.NPU
    }

    private fun SdkNpuBrand.toNitro(): NpuBrand = when (this) {
        SdkNpuBrand.QUALCOMM -> NpuBrand.QUALCOMM
        SdkNpuBrand.GOOGLE_TENSOR -> NpuBrand.GOOGLETENSOR
        SdkNpuBrand.MEDIATEK -> NpuBrand.MEDIATEK
        SdkNpuBrand.INTEL -> NpuBrand.INTEL
        SdkNpuBrand.SAMSUNG -> NpuBrand.SAMSUNG
        else -> NpuBrand.UNKNOWN
    }
}

/** Device facts that decide which bundle variant and backend fit. */
internal object DeviceFacts {
    fun current(): NativeDeviceInfo {
        val context = LiteRTLMInitProvider.applicationContext
        val totalMemory = try {
            val am = context?.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            am?.let { ActivityManager.MemoryInfo().also(it::getMemoryInfo).totalMem } ?: 0L
        } catch (_: Exception) {
            0L
        }
        val socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL.takeUnless { it == Build.UNKNOWN } else null
        val socManufacturer =
            if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER.takeUnless { it == Build.UNKNOWN } else null
        return NativeDeviceInfo(
            platform = "android",
            osVersion = Build.VERSION.RELEASE,
            model = "${Build.MANUFACTURER} ${Build.MODEL}",
            socModel = socModel,
            socManufacturer = socManufacturer,
            totalMemoryBytes = totalMemory.toDouble(),
            gpuAvailable = GpuProbe.openCLAvailable(),
            isEmulator = isEmulator(),
        )
    }

    private fun isEmulator(): Boolean =
        Build.FINGERPRINT.startsWith("generic") || Build.FINGERPRINT.contains("emulator") ||
            Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish") ||
            Build.PRODUCT.contains("sdk")
}

/**
 * LiteRT-LM's Android GPU backend needs an OpenCL driver, which most Samsung
 * and Qualcomm devices do not expose. Probed once per process.
 */
internal object GpuProbe {
    @Volatile
    private var openCL: Boolean? = null

    fun openCLAvailable(): Boolean = openCL ?: probe().also { openCL = it }

    private fun probe(): Boolean {
        try {
            System.loadLibrary("OpenCL")
            return true
        } catch (_: UnsatisfiedLinkError) {
        }
        val paths = arrayOf(
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/vendor/lib/libOpenCL.so",
            "/system/lib64/libOpenCL.so",
        )
        for (path in paths) {
            try {
                System.load(path)
                return true
            } catch (_: UnsatisfiedLinkError) {
            }
        }
        return false
    }
}
