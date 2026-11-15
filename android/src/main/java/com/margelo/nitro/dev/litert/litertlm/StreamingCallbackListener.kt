package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import com.google.ai.edge.litertlm.BenchmarkInfo
import com.google.ai.edge.litertlm.Content
import java.util.concurrent.CancellationException

/**
 * Thought text in the marker pair JS turns into `thinking` events
 * (NATIVE_THINKING_CHANNEL in src/core/streamEvents.ts; mirrors iOS).
 */
internal fun thinkingMarker(text: String) = "<litertlm:thinking>$text</litertlm:thinking>"

/**
 * The thought text of a streamed message: LiteRT-LM 0.19 parses a bundle's
 * reasoning channel (Gemma 4's "thought") out of the reply into
 * [com.google.ai.edge.litertlm.Message.channels]. A tool-call channel isn't
 * thought.
 */
internal fun thoughtText(message: com.google.ai.edge.litertlm.Message): String =
    message.channels.filterKeys { it != "tool_call" }.values.joinToString("")

/**
 * Named implementation of the LiteRT-LM MessageCallback for streaming inference.
 *
 * Accumulates response chunks, forwards tokens to JS, and appends the final
 * response to the conversation history.
 */
internal class StreamingCallbackListener(
    private val onToken: (String, Boolean) -> Unit,
    private val responseBuilder: StringBuilder,
    private val history: MutableList<Message>,
    private val userMessage: String,
    private val onStatsReady: (GenerationStats) -> Unit,
    private val onFailure: ((Throwable) -> Unit)? = null,
    /** The engine's numbers for the finished message, if it has them (exactStats). */
    private val engineStats: () -> BenchmarkInfo? = { null },
) : com.google.ai.edge.litertlm.MessageCallback {

    private val startTime = System.nanoTime()
    private var firstTokenTime = 0L
    private var tokenCount = 0
    /** The reply carried channel content (thinking), which the engine re-prefills without it. */
    private var sawChannelContent = false

    override fun onMessage(message: com.google.ai.edge.litertlm.Message) {
        if (!sawChannelContent && message.channels.values.any { it.isNotEmpty() }) {
            sawChannelContent = true
        }
        // Thought text goes to JS only: it isn't part of the reply.
        val thought = thoughtText(message)
        if (thought.isNotEmpty()) onToken(thinkingMarker(thought), false)
        // Text content plus any SDK-parsed tool calls (arriving on the final
        // message), the latter wrapped in the marker protocol the JS event
        // parser understands.
        val chunk = message.contents.contents
            .filterIsInstance<Content.Text>()
            .joinToString("") { it.text } +
            serializeToolCallMarkers(message.toolCalls)

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
        val fullResponse = responseBuilder.toString()
        // Tool calls move out of the text into structured toolCalls.
        val (text, calls) = splitToolCallMarkers(fullResponse)
        history.add(Message(Role.MODEL, text, calls.takeIf { it.isNotEmpty() }?.toTypedArray(), null))

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
        val ttftMs = if (firstTokenTime > 0) (firstTokenTime - startTime) / 1_000_000.0 else 0.0
        onStatsReady(
            generationStats(engineStats(), userMessage, fullResponse, elapsedMs, ttftMs, refilled = sawChannelContent),
        )

        Log.d("StreamingCallbackListener", "Streaming done. Length: ${fullResponse.length}, TTFT: ${ttftMs.toLong()}ms, Total: ${elapsedMs.toLong()}ms")

        // Notify JS that streaming is done AFTER updating history and stats
        onToken("", true)
    }

    override fun onError(throwable: Throwable) {
        // Record the failure first: the done token releases the waiting
        // worker, which then reads it.
        onFailure?.invoke(throwable)
        if (throwable is CancellationException) {
            // A cancel (close/unload/reset mid-stream) is a deliberate stop, not
            // a failure to show: end the stream without error text. The promise
            // still rejects with CANCELLED.
            onToken("", true)
        } else {
            Log.e("StreamingCallbackListener", "Async generation failed", throwable)
            onToken("Error: ${throwable.message}", true)
        }
    }
}
