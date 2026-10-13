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

    /** Create an engine with speculative decoding and a per-image vision-token cap (null: defaults). */
    fun <T> forEngine(speculativeDecoding: Boolean?, maxVisionTokensPerImage: Int?, create: () -> T): T =
        synchronized(lock) {
            val speculative = ExperimentalFlags.enableSpeculativeDecoding
            val budget = ExperimentalFlags.visualTokenBudget
            ExperimentalFlags.enableSpeculativeDecoding = speculativeDecoding
            ExperimentalFlags.visualTokenBudget = maxVisionTokensPerImage
            try {
                create()
            } finally {
                ExperimentalFlags.enableSpeculativeDecoding = speculative
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
