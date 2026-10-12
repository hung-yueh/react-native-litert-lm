package com.margelo.nitro.dev.litert.litertlm

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.FileNotFoundException

/**
 * The tag protocol src/core/nativeError.ts decodes, and the classification of
 * LiteRT-LM SDK failures (absl::Status text) into canonical statuses.
 */
class LiteRTLMNativeExceptionTest {

    @Test
    fun rendersTheTagProtocol() {
        val plain = LiteRTLMNativeException("load", "INVALID_ARGUMENT", null, "bad dtype")
        assertEquals("[LiteRTLM:load:INVALID_ARGUMENT] bad dtype", plain.message)

        val explicit = LiteRTLMNativeException(
            "execute", "FAILED_PRECONDITION", "MODEL_NOT_LOADED", "No model loaded.",
        )
        assertEquals("[LiteRTLM:execute:FAILED_PRECONDITION:MODEL_NOT_LOADED] No model loaded.", explicit.message)
        // fbjni hands JS Throwable.toString(): "<class>: <message>"; the tag must survive it.
        assertTrue(explicit.toString().endsWith(": [LiteRTLM:execute:FAILED_PRECONDITION:MODEL_NOT_LOADED] No model loaded."))
    }

    @Test
    fun readsTheStatusFromAbslStatusText() {
        val sdk = RuntimeException("Failed to create engine settings: INVALID_ARGUMENT: unsupported activation type")
        val e = LiteRTLMNativeException.from("load", "Failed to load the model", sdk)
        assertEquals("INVALID_ARGUMENT", e.status)
        assertEquals("load", e.phase)
        assertNull(e.code)
        assertEquals("Failed to load the model: Failed to create engine settings: INVALID_ARGUMENT: unsupported activation type", e.detail)
        assertSame(sdk, e.cause)
    }

    @Test
    fun theFirstStatusInTheMessageWins() {
        assertEquals(
            "RESOURCE_EXHAUSTED",
            LiteRTLMNativeException.statusIn("prefill: RESOURCE_EXHAUSTED: arena; INTERNAL: cleanup"),
        )
        assertNull(LiteRTLMNativeException.statusIn("NOT_FOUNDATION is not a status"))
    }

    @Test
    fun fallsBackToTheExceptionType() {
        fun statusOf(t: Throwable) = LiteRTLMNativeException.from("execute", "ctx", t).status
        assertEquals("RESOURCE_EXHAUSTED", statusOf(OutOfMemoryError("java heap")))
        assertEquals("NOT_FOUND", statusOf(FileNotFoundException("/x")))
        assertEquals("INVALID_ARGUMENT", statusOf(IllegalArgumentException("bad")))
        assertEquals("FAILED_PRECONDITION", statusOf(IllegalStateException("closed")))
        assertEquals("UNIMPLEMENTED", statusOf(UnsupportedOperationException("nope")))
        assertEquals("NOT_FOUND", statusOf(RuntimeException("Model file not found: /x.litertlm")))
        assertEquals("UNKNOWN", statusOf(RuntimeException("something odd")))
    }

    @Test
    fun passesTypedExceptionsThrough() {
        val typed = LiteRTLMNativeException("load", "NOT_FOUND", "MODEL_NOT_FOUND", "missing")
        assertSame(typed, LiteRTLMNativeException.from("execute", "ctx", typed))
    }
}
