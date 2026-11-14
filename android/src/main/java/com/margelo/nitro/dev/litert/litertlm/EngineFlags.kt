package com.margelo.nitro.dev.litert.litertlm

import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags

/**
 * The SDK's [ExperimentalFlags] are process-wide, and it reads them when an
 * engine is created or a message is sent. Every instance sets them only
 * through here, under one lock, and restores them afterwards — so one
 * instance's value never reaches another instance's call.
 */
@OptIn(ExperimentalApi::class)
internal object EngineFlags {
    private val lock = Any()

    /**
     * Create an engine with a per-image vision-token cap (null: the default),
     * in benchmark mode if asked. Speculative decoding is per engine since
     * LiteRT-LM 0.19 (`EngineConfig.speculativeDecodingConfig`), so it is no
     * longer set here.
     */
    fun <T> forEngine(
        maxVisionTokensPerImage: Int?,
        benchmark: Boolean,
        create: () -> T,
    ): T =
        synchronized(lock) {
            val benchmarkBefore = ExperimentalFlags.enableBenchmark
            val budget = ExperimentalFlags.visualTokenBudget
            ExperimentalFlags.enableBenchmark = benchmark
            ExperimentalFlags.visualTokenBudget = maxVisionTokensPerImage
            try {
                create()
            } finally {
                ExperimentalFlags.enableBenchmark = benchmarkBefore
                ExperimentalFlags.visualTokenBudget = budget
            }
        }

    /**
     * Send with this visual-token budget for the message's images (null: the
     * engine's cap). The SDK reads the flag as the send starts, so a blocking
     * send holds the lock until it returns.
     */
    fun <T> forSend(visualTokenBudget: Int?, send: () -> T): T = synchronized(lock) {
        val previous = ExperimentalFlags.visualTokenBudget
        ExperimentalFlags.visualTokenBudget = visualTokenBudget
        try {
            send()
        } finally {
            ExperimentalFlags.visualTokenBudget = previous
        }
    }
}
