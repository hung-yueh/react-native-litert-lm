package com.margelo.nitro.dev.litert.litertlm

import com.google.ai.edge.litertlm.ActivationDataType as SdkActivationDataType
import com.google.ai.edge.litertlm.SpeculativeDecodingConfig
import com.google.ai.edge.litertlm.SpeculativeDecodingMode as SdkSpeculativeDecodingMode

/**
 * The speculative-decoding mode a request asks for: the mode when set, else
 * the boolean (`true` is ON, `false` OFF), else null (the engine's default).
 */
internal fun speculativeDecodingModeOf(mode: SpeculativeDecodingMode?, enabled: Boolean?): SpeculativeDecodingMode? =
    mode ?: when (enabled) {
        true -> SpeculativeDecodingMode.ON
        false -> SpeculativeDecodingMode.OFF
        null -> null
    }

/** The SDK config for [mode]; null leaves the engine's (or engine-level) default. */
internal fun SpeculativeDecodingMode?.toSdkConfig(): SpeculativeDecodingConfig? = when (this) {
    null -> null
    SpeculativeDecodingMode.OFF -> SpeculativeDecodingConfig(SdkSpeculativeDecodingMode.OFF)
    SpeculativeDecodingMode.ON -> SpeculativeDecodingConfig(SdkSpeculativeDecodingMode.ON)
    SpeculativeDecodingMode.ADAPTIVE -> SpeculativeDecodingConfig(SdkSpeculativeDecodingMode.ADAPTIVE)
}

/** Map the Nitro Backend to a fresh LiteRT-LM SDK backend (CPU with [cpuThreads] when given). */
internal fun Backend.toSdk(cpuThreads: Int? = null): com.google.ai.edge.litertlm.Backend = when (this) {
    Backend.GPU -> com.google.ai.edge.litertlm.Backend.GPU()
    Backend.NPU -> com.google.ai.edge.litertlm.Backend.NPU()
    Backend.CPU -> com.google.ai.edge.litertlm.Backend.CPU(threadCount = cpuThreads)
}

/** Map the Nitro ActivationDataType to the LiteRT-LM SDK enum (same ordinal values). */
internal fun ActivationDataType.toSdk(): SdkActivationDataType = when (this) {
    ActivationDataType.F32 -> SdkActivationDataType.FLOAT32
    ActivationDataType.F16 -> SdkActivationDataType.FLOAT16
    ActivationDataType.I16 -> SdkActivationDataType.INT16
    ActivationDataType.I8 -> SdkActivationDataType.INT8
}
