package com.margelo.nitro.dev.litert.litertlm

import com.google.ai.edge.litertlm.BenchmarkInfo
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import java.util.concurrent.CancellationException

@RunWith(RobolectricTestRunner::class)
class StreamingCallbackListenerTest {

    /** Every callback in order, so a test can check the failure lands before the done token. */
    private val events = mutableListOf<String>()
    private var failure: Throwable? = null

    private fun listener() = StreamingCallbackListener(
        onToken = { token, done -> events += "token($token, $done)" },
        responseBuilder = StringBuilder(),
        history = mutableListOf(),
        userMessage = "hi",
        onStatsReady = {},
        onFailure = { e ->
            failure = e
            events += "failure"
        },
    )

    @Test
    fun aCancelledStreamEndsWithoutErrorText() {
        // How the SDK reports a stream that close(), unload() or a reset cancelled.
        val cancel = CancellationException("Task cancelled")
        listener().onError(cancel)
        assertEquals(listOf("failure", "token(, true)"), events)
        assertSame(cancel, failure)
    }

    @Test
    fun aFailedStreamRecordsTheErrorBeforeTheDoneToken() {
        // The done token wakes the worker, which reads the failure right away.
        listener().onError(RuntimeException("RESOURCE_EXHAUSTED: out of memory"))
        assertEquals(listOf("failure", "token(Error: RESOURCE_EXHAUSTED: out of memory, true)"), events)
    }

    private val engine = BenchmarkInfo(
        initTimeInSecond = 1.0,
        // The conversation's first turn: a later message must not report it.
        timeToFirstTokenInSecond = 9.0,
        lastPrefillTokenCount = 20,
        lastDecodeTokenCount = 128,
        lastPrefillTokensPerSecond = 200.0,
        lastDecodeTokensPerSecond = 64.0,
    )

    @Test
    fun aStreamedMessageTakesItsCountsFromTheEngineAndTimesItsOwnFirstToken() {
        var stats: GenerationStats? = null
        val listener = StreamingCallbackListener(
            onToken = { _, _ -> },
            responseBuilder = StringBuilder("a reply"),
            history = mutableListOf(),
            userMessage = "hi",
            onStatsReady = { stats = it },
            engineStats = { engine },
        )
        listener.onDone()
        assertEquals(20.0, stats!!.promptTokens, 0.0)
        assertEquals(128.0, stats!!.completionTokens, 0.0)
        assertEquals(148.0, stats!!.totalTokens, 0.0)
        assertEquals(64.0, stats!!.tokensPerSecond, 0.0)
        assertEquals("no chunk arrived, so no first token", 0.0, stats!!.timeToFirstToken, 0.0)
    }

    @Test
    fun aMessageWithTheEngineNumbersReportsTheLastTurns() {
        val streamed = generationStats(engine, "hi", "a reply", 4_000.0, 90.0)
        assertEquals(148.0, streamed.totalTokens, 0.0)
        assertEquals(90.0, streamed.timeToFirstToken, 0.0)
        assertEquals(4_000.0, streamed.totalTime, 0.0)
        assertEquals(64.0, streamed.tokensPerSecond, 0.0)

        // 20 tokens at 200 tokens/s
        val blocking = generationStats(engine, "hi", "a reply", 4_000.0)
        assertEquals(100.0, blocking.timeToFirstToken, 1e-9)
    }

    @Test
    fun aMessageWithoutTheEngineNumbersIsAnEstimate() {
        val streamed = generationStats(null, "12345678", "1234", 2_000.0, 90.0)
        assertEquals(2.0, streamed.promptTokens, 0.0)
        assertEquals(1.0, streamed.completionTokens, 0.0)
        assertEquals(3.0, streamed.totalTokens, 0.0)
        assertEquals(90.0, streamed.timeToFirstToken, 0.0)
        assertEquals(0.5, streamed.tokensPerSecond, 0.0)

        val blocking = generationStats(null, "12345678", "1234", 2_000.0)
        assertEquals(0.0, blocking.timeToFirstToken, 0.0)
        assertEquals("no time, no rate", 0.0, generationStats(null, "", "", 0.0).tokensPerSecond, 0.0)
    }

    @OptIn(ExperimentalApi::class)
    @Test
    fun benchmarkModeIsSetOnlyWhileTheEngineIsCreated() {
        assertFalse(ExperimentalFlags.enableBenchmark)
        assertTrue(EngineFlags.forEngine(null, null, benchmark = true) { ExperimentalFlags.enableBenchmark })
        assertFalse(ExperimentalFlags.enableBenchmark)
        assertFalse(EngineFlags.forEngine(null, null, benchmark = false) { ExperimentalFlags.enableBenchmark })
        runCatching { EngineFlags.forEngine<Unit>(null, null, benchmark = true) { throw IllegalStateException() } }
        assertFalse("a failed creation restores it too", ExperimentalFlags.enableBenchmark)
    }
}
