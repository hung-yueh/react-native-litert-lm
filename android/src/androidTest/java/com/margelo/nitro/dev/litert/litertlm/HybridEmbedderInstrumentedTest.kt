package com.margelo.nitro.dev.litert.litertlm

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.facebook.soloader.nativeloader.NativeLoader
import com.facebook.soloader.nativeloader.SystemDelegate
import com.google.ai.edge.litertlm.InputData
import com.margelo.nitro.JNIOnLoad
import com.margelo.nitro.core.ArrayBuffer
import dev.litert.litertlm.LiteRTLMInitProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Real on-device embeddings through the Kotlin bridge — the Android
 * counterpart of `ios/Tests/HybridEmbedderIntegrationTests.swift`.
 *
 * Skipped unless EmbeddingGemma 2 bundles sit in the test app's external
 * files dir (push them after installing the APK; see HybridLiteRTLMInstrumentedTest):
 *
 *   adb push embeddinggemma-2-text-270m.litertlm embeddinggemma-2-740m.litertlm \
 *     /sdcard/Android/data/dev.litert.litertlm.test/files/
 */
@RunWith(AndroidJUnit4::class)
class HybridEmbedderInstrumentedTest {

    private fun file(name: String): File? {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        return context.getExternalFilesDir(null)?.let { File(it, name) }?.takeIf { it.exists() }
    }

    private fun embedder(name: String): HybridEmbedder {
        val model = file(name)
        assumeTrue("$name not pushed — skipping", model != null)
        if (!NativeLoader.isInitialized()) NativeLoader.init(SystemDelegate())
        JNIOnLoad.initializeNativeNitro()
        if (LiteRTLMInitProvider.applicationContext == null) {
            val field = LiteRTLMInitProvider::class.java.getDeclaredField("applicationContext")
            field.isAccessible = true
            field.set(null, InstrumentationRegistry.getInstrumentation().targetContext)
        }
        return HybridEmbedder().also { it.loadModelBlocking(model!!.absolutePath, null) }
    }

    private fun options(normalize: Boolean = true, outputSize: Double? = null) =
        EmbeddingOptions(normalize, outputSize, null, null)

    private fun floats(buffer: ArrayBuffer): FloatArray {
        val f = buffer.getBuffer(false).order(ByteOrder.nativeOrder()).asFloatBuffer()
        return FloatArray(f.remaining()).also { f.get(it) }
    }

    private fun dot(a: FloatArray, b: FloatArray) = a.indices.sumOf { (a[it] * b[it]).toDouble() }

    @Test
    fun textEmbeddingsAreSemanticallySound() {
        val embedder = embedder("embeddinggemma-2-text-270m.litertlm")
        try {
            fun embed(text: String, opts: EmbeddingOptions = options()) =
                floats(embedder.embedBlocking(listOf(InputData.Text(text)), opts))
            val start = System.nanoTime()
            val query = embed("task: search result | query: How do I bake bread at home?")
            val ms = (System.nanoTime() - start) / 1e6
            val match = embed("title: none | text: A simple recipe for a crusty sourdough loaf.")
            val other = embed("title: none | text: The stock market fell sharply on Tuesday.")
            println("[android-embed] 1 vector in ${ms.toInt()} ms; bread↔recipe ${"%.3f".format(dot(query, match))}, bread↔stocks ${"%.3f".format(dot(query, other))}")

            assertEquals(768, query.size)
            assertEquals(1.0, dot(query, query), 1e-3)
            assertTrue("query should land nearer the matching passage", dot(query, match) > dot(query, other))
            assertEquals(256, embed("task: search result | query: bread", options(outputSize = 256.0)).size)

            val batch = embedder.embedBatchBlocking(
                listOf(
                    listOf(InputData.Text("title: none | text: A simple recipe for a crusty sourdough loaf.")),
                    listOf(InputData.Text("title: none | text: The stock market fell sharply on Tuesday.")),
                ),
                options(),
            ).map(::floats)
            assertEquals(2, batch.size)
            assertEquals(1.0, dot(batch[0], match), 1e-3)
            assertEquals(1.0, dot(batch[1], other), 1e-3)
        } finally {
            embedder.close()
        }
    }

    @Test
    fun embedWithoutALoadedModelIsTyped() {
        if (!NativeLoader.isInitialized()) NativeLoader.init(SystemDelegate())
        JNIOnLoad.initializeNativeNitro()
        try {
            HybridEmbedder().embedBlocking(listOf(InputData.Text("x")), null)
            throw AssertionError("embedding without a model must fail")
        } catch (e: LiteRTLMNativeException) {
            assertEquals("MODEL_NOT_LOADED", e.code)
            assertEquals("embed", e.phase)
        }
    }

    /**
     * close() never waits for engine work: readiness drops at once, the
     * in-flight batch still completes, and the instance loads again in call
     * order. Media files are read on the worker, so a bad part rejects (typed)
     * rather than throwing on the caller's thread.
     */
    @Test
    fun closeDoesNotWaitForEngineWork() {
        val name = "embeddinggemma-2-text-270m.litertlm"
        val embedder = embedder(name)
        try {
            val notes = Array(32) { arrayOf(textPart("title: none | text: Note $it: water the tomato plants in July.")) }
            val batch = embedder.embedBatch(notes, options())
            val start = System.nanoTime()
            embedder.close()
            val closeMs = (System.nanoTime() - start) / 1e6
            println("[android-embed] close() returned in ${"%.2f".format(closeMs)} ms during a 32-input batch")
            assertTrue("close() must not wait for the batch (took $closeMs ms)", closeMs < 50)
            assertFalse(embedder.isReady())
            assertEquals(32, awaitResult(batch).getOrThrow().size)

            awaitResult(embedder.loadModel(file(name)!!.absolutePath, null)).getOrThrow()
            assertTrue(embedder.isReady())

            val missing = awaitResult(
                embedder.embed(arrayOf(MultimodalPart(PartType.IMAGE, null, "/nope.jpg", null, null, null, null)), null),
            ).exceptionOrNull()
            assertEquals("INVALID_INPUT", (missing as LiteRTLMNativeException).code)
            val empty = awaitResult(
                embedder.embed(arrayOf(MultimodalPart(PartType.AUDIO, null, null, null, null, null, null)), null),
            ).exceptionOrNull()
            assertEquals("INVALID_INPUT", (empty as LiteRTLMNativeException).code)
        } finally {
            embedder.close()
        }
    }

    private fun textPart(text: String) = MultimodalPart(PartType.TEXT, text, null, null, null, null, null)

    /** Await a Nitro Promise from the test thread; returns its value or failure. */
    private fun <T> awaitResult(promise: com.margelo.nitro.core.Promise<T>): Result<T> {
        val latch = CountDownLatch(1)
        var result: Result<T>? = null
        promise
            .then { result = Result.success(it); latch.countDown() }
            .catch { result = Result.failure(it); latch.countDown() }
        assertTrue("promise did not settle", latch.await(120, TimeUnit.SECONDS))
        return result!!
    }

    /** Text, image and audio share one space: captions land nearer their own media. */
    @Test
    fun crossModalEmbeddings() {
        val image = file("test.jpeg")
        val audio = file("test.wav")
        assumeTrue("test.jpeg / test.wav not pushed — skipping", image != null && audio != null)
        val embedder = embedder("embeddinggemma-2-740m.litertlm")
        try {
            fun embed(input: InputData) = floats(embedder.embedBlocking(listOf(input), options()))
            fun text(s: String) = embed(InputData.Text("task: search result | query: $s"))
            var t = System.nanoTime()
            val picture = embed(InputData.Image(image!!.readBytes()))
            val imageMs = (System.nanoTime() - t) / 1e6
            t = System.nanoTime()
            val clip = embed(InputData.Audio(audio!!.readBytes()))
            val audioMs = (System.nanoTime() - t) / 1e6
            val cake = text("a person holding a birthday cake")
            val mountain = text("a snowy mountain landscape")
            val greeting = text("people greeting each other")
            val engine = text("a car engine revving")
            println(
                "[android-embed][mm] image ${imageMs.toInt()} ms, audio ${audioMs.toInt()} ms; " +
                    "image↔cake ${"%.3f".format(dot(picture, cake))} vs mountain ${"%.3f".format(dot(picture, mountain))}; " +
                    "audio↔greeting ${"%.3f".format(dot(clip, greeting))} vs engine ${"%.3f".format(dot(clip, engine))}",
            )
            assertTrue(dot(picture, cake) > dot(picture, mountain))
            assertTrue(dot(clip, greeting) > dot(clip, engine))
        } finally {
            embedder.close()
        }
    }
}
