package com.margelo.nitro.dev.litert.litertlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
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
}
