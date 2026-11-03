package com.margelo.nitro.dev.litert.litertlm

import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.net.InetAddress
import java.net.SocketTimeoutException
import java.security.MessageDigest
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLSocketFactory
import kotlin.concurrent.thread
import okhttp3.mockwebserver.Dispatcher
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.RecordedRequest
import okhttp3.mockwebserver.SocketPolicy
import okhttp3.tls.HandshakeCertificates
import okhttp3.tls.HeldCertificate
import okio.Buffer
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.shadows.ShadowLog

/**
 * Exercises HybridModelStore's real download machinery — HTTPS enforcement is
 * covered elsewhere; these tests cover what happens when the network answers:
 * success + atomic rename, progress reporting, HTTP errors, mid-stream
 * disconnects (no half-written files), the cached-file short-circuit,
 * continuing a partial download with a Range request (from the bytes synced to
 * disk, under a new URL signature), a stalled read, two downloads of one file
 * at once, the sha256 check and the hash kept for a cached file, old or gone
 * partial downloads, a download that would fill the disk, names the cache
 * reserves, and what a failed download leaves open.
 *
 * Uses a local MockWebServer over TLS since the store only accepts https URLs.
 */
@RunWith(RobolectricTestRunner::class)
class HybridModelStoreDownloadTest {
    private lateinit var store: HybridModelStore
    private lateinit var server: MockWebServer
    private lateinit var modelsDir: File
    private var originalSslFactory: SSLSocketFactory? = null
    private var originalVerifier: HostnameVerifier? = null

    @Before
    fun setUp() {
        val field = LiteRTLMInitProvider::class.java.getDeclaredField("applicationContext")
        field.isAccessible = true
        field.set(null, RuntimeEnvironment.getApplication())

        modelsDir = File(RuntimeEnvironment.getApplication().filesDir, "models")
        modelsDir.deleteRecursively()

        // Self-signed localhost certificate trusted by the default HTTPS stack
        // for the duration of the test.
        val localhost = InetAddress.getByName("localhost").canonicalHostName
        val certificate = HeldCertificate.Builder()
            .addSubjectAlternativeName(localhost)
            .build()
        val serverCertificates = HandshakeCertificates.Builder()
            .heldCertificate(certificate)
            .build()
        val clientCertificates = HandshakeCertificates.Builder()
            .addTrustedCertificate(certificate.certificate)
            .build()

        originalSslFactory = HttpsURLConnection.getDefaultSSLSocketFactory()
        originalVerifier = HttpsURLConnection.getDefaultHostnameVerifier()
        HttpsURLConnection.setDefaultSSLSocketFactory(clientCertificates.sslSocketFactory())
        HttpsURLConnection.setDefaultHostnameVerifier { _, _ -> true }

        server = MockWebServer()
        server.useHttps(serverCertificates.sslSocketFactory(), false)
        server.start()

        store = HybridModelStore()
    }

    @After
    fun tearDown() {
        server.shutdown()
        originalSslFactory?.let { HttpsURLConnection.setDefaultSSLSocketFactory(it) }
        originalVerifier?.let { HttpsURLConnection.setDefaultHostnameVerifier(it) }
        modelsDir.deleteRecursively()
    }

    private fun url(path: String): String = server.url(path).toString()

    private fun <T> await(promise: com.margelo.nitro.core.Promise<T>): Pair<T?, Throwable?> {
        val deadline = System.currentTimeMillis() + 10_000
        while (!promise.isCompleted && System.currentTimeMillis() < deadline) {
            Thread.sleep(10)
        }
        assertTrue("Promise did not complete within 10s", promise.isCompleted)
        return Pair(promise.result, promise.error)
    }

    @Test
    fun downloadSucceedsAndRenamesAtomically() {
        val body = ByteArray(256 * 1024) { it.toByte() }
        server.enqueue(MockResponse().setBody(Buffer().write(body)))

        val progress = mutableListOf<Double>()
        val (path, error) = await(
            store.downloadFile(url("/model.litertlm"), "model.litertlm", "{}", "") { progress.add(it) }
        )

        assertNull("Download should succeed, got: $error", error)
        val file = File(path!!)
        assertTrue(file.exists())
        assertEquals(body.size.toLong(), file.length())
        // No temp file left behind
        assertFalse(File(modelsDir, "model.litertlm.tmp").exists())
        // Progress starts at 0, ends at exactly 1, and never decreases
        assertEquals(0.0, progress.first(), 0.0)
        assertEquals(1.0, progress.last(), 0.0)
        assertEquals(progress, progress.sorted())
    }

    @Test
    fun downloadRejectsOnHttpError() {
        server.enqueue(MockResponse().setResponseCode(404))

        val (_, error) = await(
            store.downloadFile(url("/missing.litertlm"), "missing.litertlm", "{}", "") {}
        )

        assertNotNull("Expected HTTP 404 rejection", error)
        assertTrue(error!!.message!!.contains("404"))
        assertFalse(File(modelsDir, "missing.litertlm").exists())
        assertFalse(File(modelsDir, "missing.litertlm.tmp").exists())
    }

    @Test
    fun downloadCleansUpTempFileOnMidStreamDisconnect() {
        val body = ByteArray(512 * 1024) { it.toByte() }
        server.enqueue(
            MockResponse()
                .setBody(Buffer().write(body))
                .setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY)
        )

        val (_, error) = await(
            store.downloadFile(url("/cut.litertlm"), "cut.litertlm", "{}", "") {}
        )

        // Depending on where the cut lands the read either errors or ends
        // early; either way the download must reject and leave no partial
        // file behind — a truncated model must never be installed as complete.
        assertNotNull("Truncated download must reject", error)
        assertFalse(File(modelsDir, "cut.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "cut.litertlm").exists())
    }

    @Test
    fun downloadShortCircuitsWhenAlreadyCached() {
        modelsDir.mkdirs()
        File(modelsDir, "cached.litertlm").writeBytes(ByteArray(128) { 1 })

        val progress = mutableListOf<Double>()
        val (path, error) = await(
            store.downloadFile(url("/cached.litertlm"), "cached.litertlm", "{}", "") { progress.add(it) }
        )

        assertNull(error)
        assertEquals(File(modelsDir, "cached.litertlm").absolutePath, path)
        assertEquals(listOf(1.0), progress)
        assertEquals("No network request should be made for a cached file", 0, server.requestCount)
    }

    @Test
    fun downloadSendsCustomHeaders() {
        server.enqueue(MockResponse().setBody("ok"))

        val (_, error) = await(
            store.downloadFile(
                url("/auth.litertlm"),
                "auth.litertlm",
                """{"Authorization":"Bearer secret-token"}""",
                "",
            ) {}
        )

        assertNull(error)
        val recorded = server.takeRequest()
        assertEquals("Bearer secret-token", recorded.getHeader("Authorization"))
    }

    // ── Continuing a partial download ───────────────────────────────────────

    private val payload = ByteArray(300 * 1024) { (it * 7 + 3).toByte() }

    private fun sha256Of(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    /**
     * A first call that the connection drops after [cut] bytes, from a server
     * that named the file's version and size: the partial file stays.
     */
    private fun leavePartial(name: String, cut: Int, etag: String = "\"v1\"", sha256: String = "", path: String = "/$name") {
        server.enqueue(
            MockResponse()
                .setBody(Buffer().write(payload, 0, cut))
                .setHeader("Content-Length", payload.size)
                .setHeader("ETag", etag)
                .setSocketPolicy(SocketPolicy.DISCONNECT_AT_END)
        )
        val (_, error) = await(store.downloadFile(url(path), name, "{}", sha256) {})
        assertNotNull("A cut download must reject", error)
        assertEquals(cut.toLong(), File(modelsDir, "$name.tmp").length())
        assertFalse(File(modelsDir, name).exists())
        server.takeRequest()
    }

    private fun rest(from: Int, etag: String = "\"v1\""): MockResponse =
        MockResponse()
            .setResponseCode(206)
            .setBody(Buffer().write(payload, from, payload.size - from))
            .setHeader("Content-Range", "bytes $from-${payload.size - 1}/${payload.size}")
            .setHeader("ETag", etag)

    @Test
    fun aDroppedDownloadKeepsItsPartialFileWhenTheServerNamedTheVersion() {
        leavePartial("kept.litertlm", cut = 100 * 1024)
        assertTrue(File(modelsDir, "kept.litertlm.tmp.json").exists())
        // Neither file is a cached model
        assertEquals(0, store.listCachedFiles().size)
        assertFalse(store.isCached("kept.litertlm"))
    }

    @Test
    fun theNextCallContinuesWithARangeRequest() {
        leavePartial("resume.litertlm", cut = 100 * 1024)
        server.enqueue(rest(100 * 1024))

        val progress = mutableListOf<Double>()
        val (path, error) = await(
            store.downloadFile(url("/resume.litertlm"), "resume.litertlm", "{}", "") { progress.add(it) }
        )

        assertNull("The continued download should succeed, got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        val request = server.takeRequest()
        assertEquals("bytes=${100 * 1024}-", request.getHeader("Range"))
        assertEquals("\"v1\"", request.getHeader("If-Range"))
        assertFalse(File(modelsDir, "resume.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "resume.litertlm.tmp.json").exists())
        // Progress counts the bytes already there, ends at 1, never decreases
        assertEquals(100.0 * 1024 / payload.size, progress[1], 1e-9)
        assertEquals(1.0, progress.last(), 0.0)
        assertEquals(progress, progress.sorted())
    }

    @Test
    fun aServerThatIgnoresTheRangeStartsTheFileOver() {
        leavePartial("norange.litertlm", cut = 100 * 1024)
        server.enqueue(MockResponse().setBody(Buffer().write(payload)).setHeader("ETag", "\"v1\""))

        val (path, error) = await(store.downloadFile(url("/norange.litertlm"), "norange.litertlm", "{}", "") {})

        assertNull(error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals(2, server.requestCount)
    }

    @Test
    fun anUnsatisfiableRangeStartsOverWithoutIt() {
        leavePartial("r416.litertlm", cut = 100 * 1024)
        server.enqueue(MockResponse().setResponseCode(416))
        server.enqueue(MockResponse().setBody(Buffer().write(payload)).setHeader("ETag", "\"v1\""))

        val (path, error) = await(store.downloadFile(url("/r416.litertlm"), "r416.litertlm", "{}", "") {})

        assertNull(error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertNotNull(server.takeRequest().getHeader("Range"))
        assertNull("The second request asks for the whole file", server.takeRequest().getHeader("Range"))
    }

    @Test
    fun aRangeOfAnotherVersionIsNotAppended() {
        // The Hugging Face CDN answers 206 whatever If-Range says: the validator
        // of the 206 itself has to match.
        leavePartial("changed.litertlm", cut = 100 * 1024)
        server.enqueue(rest(100 * 1024, etag = "\"v2\""))
        val v2 = payload.reversedArray()
        server.enqueue(MockResponse().setBody(Buffer().write(v2)).setHeader("ETag", "\"v2\""))

        val (path, error) = await(store.downloadFile(url("/changed.litertlm"), "changed.litertlm", "{}", "") {})

        assertNull(error)
        assertArrayEquals(v2, File(path!!).readBytes())
        assertEquals(3, server.requestCount)
    }

    @Test
    fun aServerErrorOnTheContinuationKeepsThePartialFile() {
        leavePartial("busy.litertlm", cut = 100 * 1024)
        server.enqueue(MockResponse().setResponseCode(503))

        val (_, error) = await(store.downloadFile(url("/busy.litertlm"), "busy.litertlm", "{}", "") {})

        assertNotNull(error)
        assertTrue(error!!.message!!.contains("503"))
        assertEquals(100L * 1024, File(modelsDir, "busy.litertlm.tmp").length())
    }

    @Test
    fun aPartialFileFromAnotherUrlIsNotContinued() {
        leavePartial("moved.litertlm", cut = 100 * 1024)
        server.enqueue(MockResponse().setBody(Buffer().write(payload)).setHeader("ETag", "\"v1\""))

        val (path, error) = await(store.downloadFile(url("/elsewhere/moved.litertlm"), "moved.litertlm", "{}", "") {})

        assertNull(error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertNull(server.takeRequest().getHeader("Range"))
    }

    @Test
    fun theRangeAndIfRangeFollowARedirect() {
        leavePartial("redirect.litertlm", cut = 100 * 1024)
        server.enqueue(
            MockResponse().setResponseCode(302).setHeader("Location", url("/cdn/redirect.litertlm"))
        )
        server.enqueue(rest(100 * 1024))

        val (path, error) = await(store.downloadFile(url("/redirect.litertlm"), "redirect.litertlm", "{}", "") {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        server.takeRequest() // the redirect
        val redirected = server.takeRequest()
        assertEquals("/cdn/redirect.litertlm", redirected.path)
        assertEquals("bytes=${100 * 1024}-", redirected.getHeader("Range"))
        assertEquals("\"v1\"", redirected.getHeader("If-Range"))
    }

    @Test
    fun deleteFileRemovesAPartialDownload() {
        leavePartial("gone.litertlm", cut = 100 * 1024)

        store.deleteFile("gone.litertlm")

        assertFalse(File(modelsDir, "gone.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "gone.litertlm.tmp.json").exists())
    }

    @Test
    fun aContinuationTheServerNoLongerHasDiscardsThePartialFile() {
        for (code in listOf(404, 410)) {
            leavePartial("gone$code.litertlm", cut = 100 * 1024)
            server.enqueue(MockResponse().setResponseCode(code))

            val (_, error) = await(store.downloadFile(url("/gone$code.litertlm"), "gone$code.litertlm", "{}", "") {})

            assertTrue("got: $error", error!!.message!!.contains("HTTP $code"))
            assertFalse(File(modelsDir, "gone$code.litertlm.tmp").exists())
            assertFalse(File(modelsDir, "gone$code.litertlm.tmp.json").exists())
            server.takeRequest()
        }
    }

    @Test
    fun aSignedUrlWithANewSignatureContinuesThePartialFile() {
        leavePartial("signed.litertlm", cut = 100 * 1024, path = "/signed.litertlm?X-Amz-Signature=one")
        // The record keeps no signature
        assertFalse(File(modelsDir, "signed.litertlm.tmp.json").readText().contains("Signature"))
        server.enqueue(rest(100 * 1024))

        val (path, error) = await(
            store.downloadFile(url("/signed.litertlm?X-Amz-Signature=two"), "signed.litertlm", "{}", "") {}
        )

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals("bytes=${100 * 1024}-", server.takeRequest().getHeader("Range"))
    }

    // ── What reaches the disk ───────────────────────────────────────────────

    private fun syncedBytes(name: String): Long =
        runCatching { JSONObject(File(modelsDir, "$name.tmp.json").readText()).getLong("syncedBytes") }.getOrDefault(-1L)

    @Test
    fun aDroppedDownloadRecordsWhatItSyncedToDisk() {
        leavePartial("synced.litertlm", cut = 100 * 1024)
        assertEquals(100L * 1024, syncedBytes("synced.litertlm"))
    }

    @Test
    fun aLongDownloadSyncsToDiskAsItGoes() {
        store.syncBytes = 64L * 1024
        server.enqueue(slowly())

        val running = onThread { store.downloadFile(url("/long.litertlm"), "long.litertlm", "{}", "") {} }
        server.takeRequest()
        waitFor("the record counts synced bytes while the download runs") {
            syncedBytes("long.litertlm") >= 64L * 1024 && !File(modelsDir, "long.litertlm").exists()
        }

        val (path, error) = running()
        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
    }

    @Test
    fun aResumeStartsAtTheLastByteSyncedToDisk() {
        leavePartial("powercut.litertlm", cut = 100 * 1024)
        // As after a power cut: the file holds more than the record says reached the disk
        val record = File(modelsDir, "powercut.litertlm.tmp.json")
        record.writeText(JSONObject(record.readText()).put("syncedBytes", 64 * 1024).toString())
        server.enqueue(rest(64 * 1024))

        val (path, error) = await(
            store.downloadFile(url("/powercut.litertlm"), "powercut.litertlm", "{}", sha256Of(payload)) {}
        )

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals("bytes=${64 * 1024}-", server.takeRequest().getHeader("Range"))
    }

    @Test
    fun aRecordWithoutSyncedBytesStartsOver() {
        leavePartial("unsynced.litertlm", cut = 100 * 1024)
        val record = File(modelsDir, "unsynced.litertlm.tmp.json")
        record.writeText(JSONObject(record.readText()).apply { remove("syncedBytes") }.toString())
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (path, error) = await(store.downloadFile(url("/unsynced.litertlm"), "unsynced.litertlm", "{}", "") {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertNull(server.takeRequest().getHeader("Range"))
    }

    @Test
    fun aDownloadThatWouldFillTheDiskFailsBeforeWritingAByte() {
        // Room for the file, not for the margin on top
        store.usableSpace = { payload.size.toLong() }
        server.enqueue(MockResponse().setBody(Buffer().write(payload)).setHeader("ETag", "\"v1\""))

        val (_, error) = await(store.downloadFile(url("/full.litertlm"), "full.litertlm", "{}", "") {})

        assertTrue("got: $error", error!!.message!!.contains("Not enough free storage"))
        assertEquals(emptyList<String>(), modelsDir.list()!!.toList())
    }

    @Test
    fun aContinuationThatWouldFillTheDiskKeepsThePartialFile() {
        leavePartial("fuller.litertlm", cut = 100 * 1024)
        store.usableSpace = { 0L }
        server.enqueue(rest(100 * 1024))

        val (_, error) = await(store.downloadFile(url("/fuller.litertlm"), "fuller.litertlm", "{}", "") {})

        assertTrue("got: $error", error!!.message!!.contains("Not enough free storage"))
        assertEquals(100L * 1024, File(modelsDir, "fuller.litertlm.tmp").length())
        assertTrue(File(modelsDir, "fuller.litertlm.tmp.json").exists())
    }

    @Test
    fun namesTheCacheReservesAreRejected() {
        for (name in listOf("x.tmp", "x.tmp.json", "x.sha256", ".", "")) {
            val (_, error) = await(store.downloadFile(url("/x"), name, "{}", "") {})
            assertNotNull("\"$name\" must be rejected", error)
            assertThrows(IllegalArgumentException::class.java) { store.isCached(name) }
        }
        assertEquals(0, server.requestCount)
    }

    @Test
    fun aContinuationShortOfTheWholeFileIsNotInstalled() {
        leavePartial("short.litertlm", cut = 100 * 1024)
        // A 206 sent chunked (no Content-Length to check) that ends 50 KiB in
        server.enqueue(
            MockResponse()
                .setResponseCode(206)
                .setChunkedBody(Buffer().write(payload, 100 * 1024, 50 * 1024), 16 * 1024)
                .setHeader("Content-Range", "bytes ${100 * 1024}-${payload.size - 1}/${payload.size}")
                .setHeader("ETag", "\"v1\"")
        )

        val (_, error) = await(store.downloadFile(url("/short.litertlm"), "short.litertlm", "{}", "") {})

        assertNotNull("A continuation short of the whole file must reject", error)
        assertTrue(error!!.message!!, error.message!!.contains("Incomplete download: got ${150 * 1024} of ${payload.size} bytes"))
        assertFalse(File(modelsDir, "short.litertlm").exists())
        // What it got stays for the next call, which continues from there
        assertEquals(150L * 1024, File(modelsDir, "short.litertlm.tmp").length())
        assertTrue(File(modelsDir, "short.litertlm.tmp.json").exists())
        server.enqueue(rest(150 * 1024))

        val (path, error2) = await(store.downloadFile(url("/short.litertlm"), "short.litertlm", "{}", "") {})

        assertNull("got: $error2", error2)
        assertArrayEquals(payload, File(path!!).readBytes())
        server.takeRequest() // the 206 that ended early
        assertEquals("bytes=${150 * 1024}-", server.takeRequest().getHeader("Range"))
    }

    // ── A stalled read ──────────────────────────────────────────────────────

    @Test
    fun aStalledReadFailsAndKeepsThePartialFileForTheNextCall() {
        assertEquals(60_000, store.readTimeoutMs)
        store.readTimeoutMs = 1_000
        // 200 KiB, then the server goes quiet for 3 s before the rest
        server.enqueue(
            MockResponse()
                .setBody(Buffer().write(payload))
                .setHeader("ETag", "\"v1\"")
                .throttleBody(200L * 1024, 3, TimeUnit.SECONDS)
        )

        val (_, error) = await(store.downloadFile(url("/stall.litertlm"), "stall.litertlm", "{}", "") {})

        assertNotNull("A stalled read must reject", error)
        assertTrue("got: $error", error!!.cause is SocketTimeoutException)
        // It failed in the quiet time, and kept what it got
        assertEquals(200L * 1024, File(modelsDir, "stall.litertlm.tmp").length())
        assertTrue(File(modelsDir, "stall.litertlm.tmp.json").exists())
        server.enqueue(rest(200 * 1024))

        val (path, error2) = await(store.downloadFile(url("/stall.litertlm"), "stall.litertlm", "{}", "") {})

        assertNull("got: $error2", error2)
        assertArrayEquals(payload, File(path!!).readBytes())
        server.takeRequest()
        assertEquals("bytes=${200 * 1024}-", server.takeRequest().getHeader("Range"))
    }

    // ── Two downloads of one file ───────────────────────────────────────────

    /**
     * Starts a call on a thread of its own, as Nitro's Promise.parallel does
     * (the test Promise runs it on the caller's thread): its result once it returns.
     */
    private fun <T> onThread(call: () -> com.margelo.nitro.core.Promise<T>): () -> Pair<T?, Throwable?> {
        var promise: com.margelo.nitro.core.Promise<T>? = null
        val worker = thread { promise = call() }
        return {
            worker.join(10_000)
            assertFalse("The call did not return within 10s", worker.isAlive)
            Pair(promise!!.result, promise!!.error)
        }
    }

    private fun waitFor(what: String, condition: () -> Boolean) {
        val deadline = System.currentTimeMillis() + 5_000
        while (!condition() && System.currentTimeMillis() < deadline) {
            Thread.sleep(10)
        }
        assertTrue(what, condition())
    }

    /** The whole payload from a slow server: [bytes] every 300 ms. */
    private fun slowly(bytes: Long = 64L * 1024): MockResponse =
        MockResponse()
            .setBody(Buffer().write(payload))
            .setHeader("ETag", "\"v1\"")
            .throttleBody(bytes, 300, TimeUnit.MILLISECONDS)

    @Test
    fun twoDownloadsOfTheSameFileRunOneAfterTheOther() {
        server.enqueue(slowly())

        val first = onThread { store.downloadFile(url("/twice.litertlm"), "twice.litertlm", "{}", "") {} }
        server.takeRequest()
        waitFor("the first download writes its partial file") { File(modelsDir, "twice.litertlm.tmp").length() > 0 }
        val second = onThread { store.downloadFile(url("/twice.litertlm"), "twice.litertlm", "{}", "") {} }

        val (path1, error1) = first()
        val (path2, error2) = second()
        assertNull("got: $error1", error1)
        assertNull("got: $error2", error2)
        assertEquals(path1, path2)
        assertArrayEquals(payload, File(path1!!).readBytes())
        // The second waited for the first and found the file in the cache
        assertEquals(1, server.requestCount)
    }

    @Test
    fun aCallThatWaitedContinuesWhatAFailedDownloadLeft() {
        // The version and size, 192 KiB slowly, then the connection drops
        server.enqueue(
            MockResponse()
                .setBody(Buffer().write(payload, 0, 192 * 1024))
                .setHeader("Content-Length", payload.size)
                .setHeader("ETag", "\"v1\"")
                .throttleBody(64L * 1024, 300, TimeUnit.MILLISECONDS)
                .setSocketPolicy(SocketPolicy.DISCONNECT_AT_END)
        )
        server.enqueue(rest(192 * 1024))

        val first = onThread { store.downloadFile(url("/again.litertlm"), "again.litertlm", "{}", "") {} }
        server.takeRequest()
        waitFor("the first download writes its partial file") { File(modelsDir, "again.litertlm.tmp").length() > 0 }
        val second = onThread { store.downloadFile(url("/again.litertlm"), "again.litertlm", "{}", "") {} }

        val (_, error1) = first()
        val (path2, error2) = second()
        assertNotNull("The dropped download must reject", error1)
        assertNull("got: $error2", error2)
        assertArrayEquals(payload, File(path2!!).readBytes())
        assertEquals("bytes=${192 * 1024}-", server.takeRequest().getHeader("Range"))
    }

    @Test
    fun deleteFileLeavesThePartialFileOfARunningDownload() {
        server.enqueue(slowly())

        val running = onThread { store.downloadFile(url("/writing.litertlm"), "writing.litertlm", "{}", "") {} }
        server.takeRequest()
        waitFor("the download writes its partial file") { File(modelsDir, "writing.litertlm.tmp").length() > 0 }
        store.deleteFile("writing.litertlm")

        assertTrue(File(modelsDir, "writing.litertlm.tmp").exists())
        assertTrue(File(modelsDir, "writing.litertlm.tmp.json").exists())
        val (path, error) = running()
        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
    }

    @Test
    fun aDownloadOfAnotherFileDoesNotWait() {
        server.enqueue(slowly(bytes = 32L * 1024))

        val first = onThread { store.downloadFile(url("/slow.litertlm"), "slow.litertlm", "{}", "") {} }
        server.takeRequest()
        val waiting = onThread { store.downloadFile(url("/slow.litertlm"), "slow.litertlm", "{}", "") {} }
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (other, error) = await(store.downloadFile(url("/other.litertlm"), "other.litertlm", "{}", "") {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(other!!).readBytes())
        assertFalse("The slow download is still running", File(modelsDir, "slow.litertlm").exists())
        assertNull(first().second)
        assertNull(waiting().second)
        assertEquals(2, server.requestCount)
    }

    // ── sha256 ──────────────────────────────────────────────────────────────

    @Test
    fun aMatchingSha256InstallsTheFile() {
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (path, error) = await(store.downloadFile(url("/ok.litertlm"), "ok.litertlm", "{}", sha256Of(payload)) {})

        assertNull(error)
        assertArrayEquals(payload, File(path!!).readBytes())
    }

    @Test
    fun aSha256MismatchLeavesNothingBehind() {
        server.enqueue(MockResponse().setBody(Buffer().write(payload)).setHeader("ETag", "\"v1\""))
        val wrong = "0".repeat(64)

        val (_, error) = await(store.downloadFile(url("/bad.litertlm"), "bad.litertlm", "{}", wrong) {})

        assertNotNull(error)
        assertTrue(error!!.message!!, error.message!!.contains("sha256 mismatch: expected $wrong, got ${sha256Of(payload)}"))
        assertFalse(File(modelsDir, "bad.litertlm").exists())
        assertFalse(File(modelsDir, "bad.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "bad.litertlm.tmp.json").exists())
        assertFalse(store.isCached("bad.litertlm"))
    }

    @Test
    fun aContinuedDownloadIsHashedFromItsFirstByte() {
        leavePartial("hashed.litertlm", cut = 100 * 1024, sha256 = sha256Of(payload))
        server.enqueue(rest(100 * 1024))

        val (path, error) = await(
            store.downloadFile(url("/hashed.litertlm"), "hashed.litertlm", "{}", sha256Of(payload)) {}
        )

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
    }

    @Test
    fun aContinuedDownloadWithTheWrongSha256LeavesNothingBehind() {
        leavePartial("hashbad.litertlm", cut = 100 * 1024)
        server.enqueue(rest(100 * 1024))
        val wrong = "f".repeat(64)

        val (_, error) = await(store.downloadFile(url("/hashbad.litertlm"), "hashbad.litertlm", "{}", wrong) {})

        assertNotNull(error)
        assertTrue(error!!.message!!.contains("sha256 mismatch: expected $wrong, got ${sha256Of(payload)}"))
        assertFalse(File(modelsDir, "hashbad.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "hashbad.litertlm.tmp.json").exists())
        assertFalse(File(modelsDir, "hashbad.litertlm").exists())
    }

    @Test
    fun aContinuationThatStartsOverIsHashedFromItsFirstByte() {
        leavePartial("rehash.litertlm", cut = 100 * 1024, sha256 = sha256Of(payload))
        // The part on disk was hashed before the request; the server ignores the range
        server.enqueue(MockResponse().setBody(Buffer().write(payload)).setHeader("ETag", "\"v1\""))

        val (path, error) = await(
            store.downloadFile(url("/rehash.litertlm"), "rehash.litertlm", "{}", sha256Of(payload)) {}
        )

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
    }

    // ── A cached file and a sha256 ──────────────────────────────────────────

    private fun hashesOfCachedFiles(): Int =
        ShadowLog.getLogsForTag("HybridModelStore").count { it.msg.startsWith("Hashed the cached") }

    @Test
    fun aCachedFileIsHashedOnceAndTheResultKept() {
        modelsDir.mkdirs()
        val cached = File(modelsDir, "cachedsha.litertlm")
        cached.writeBytes(payload)

        repeat(2) {
            val (path, error) = await(
                store.downloadFile(url("/cachedsha.litertlm"), "cachedsha.litertlm", "{}", sha256Of(payload)) {}
            )
            assertNull("got: $error", error)
            assertEquals(cached.absolutePath, path)
        }

        assertEquals(0, server.requestCount)
        assertEquals(1, hashesOfCachedFiles())
        val kept = JSONObject(File(modelsDir, "cachedsha.litertlm.sha256").readText())
        assertEquals(sha256Of(payload), kept.getString("sha256"))
        assertEquals(cached.length(), kept.getLong("size"))
        assertEquals(cached.lastModified(), kept.getLong("modified"))
        // The kept hash is not a cached model of its own
        assertEquals(listOf("cachedsha.litertlm"), store.listCachedFiles().map { it.fileName })
    }

    @Test
    fun aCachedFileOfAnotherRevisionIsDownloadedAgain() {
        modelsDir.mkdirs()
        File(modelsDir, "revised.litertlm").writeBytes(ByteArray(128) { 1 })
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (path, error) = await(
            store.downloadFile(url("/revised.litertlm"), "revised.litertlm", "{}", sha256Of(payload)) {}
        )

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals(1, server.requestCount)
        assertEquals(sha256Of(payload), JSONObject(File(modelsDir, "revised.litertlm.sha256").readText()).getString("sha256"))
    }

    @Test
    fun aDownloadedFileIsNotHashedAgainOnTheNextCall() {
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))
        await(store.downloadFile(url("/fresh.litertlm"), "fresh.litertlm", "{}", sha256Of(payload)) {})

        val (path, error) = await(store.downloadFile(url("/fresh.litertlm"), "fresh.litertlm", "{}", sha256Of(payload)) {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals(1, server.requestCount)
        assertEquals(0, hashesOfCachedFiles())
    }

    @Test
    fun aKeptHashDoesNotVouchForAFileChangedSince() {
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))
        val (first, _) = await(store.downloadFile(url("/changed.litertlm"), "changed.litertlm", "{}", sha256Of(payload)) {})
        // Something outside the store rewrites the file
        File(first!!).writeBytes(ByteArray(256) { 2 })
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (path, error) = await(store.downloadFile(url("/changed.litertlm"), "changed.litertlm", "{}", sha256Of(payload)) {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals(2, server.requestCount)
    }

    @Test
    fun aFileInstalledWithoutASha256DropsTheKeptHash() {
        modelsDir.mkdirs()
        // The hash kept for a file that was deleted outside the store
        File(modelsDir, "plain.litertlm.sha256").writeText("""{"sha256":"${"a".repeat(64)}","size":1,"modified":1}""")
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (path, error) = await(store.downloadFile(url("/plain.litertlm"), "plain.litertlm", "{}", "") {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertFalse(File(modelsDir, "plain.litertlm.sha256").exists())
    }

    @Test
    fun deleteFileRemovesTheKeptHash() {
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))
        await(store.downloadFile(url("/hashkept.litertlm"), "hashkept.litertlm", "{}", sha256Of(payload)) {})
        assertTrue(File(modelsDir, "hashkept.litertlm.sha256").exists())

        store.deleteFile("hashkept.litertlm")

        assertEquals(emptyList<String>(), modelsDir.list()!!.toList())
    }

    // ── Partial downloads nothing has written to for a week ─────────────────

    private val day = 24L * 60 * 60 * 1000

    private fun age(name: String, daysAgo: Int) {
        assertTrue(File(modelsDir, name).setLastModified(System.currentTimeMillis() - daysAgo * day))
    }

    /** A file in the models dir, last written [daysAgo] days ago. */
    private fun place(name: String, daysAgo: Int) {
        modelsDir.mkdirs()
        File(modelsDir, name).writeBytes(ByteArray(64) { 7 })
        age(name, daysAgo)
    }

    @Test
    fun partialDownloadsNotWrittenForAWeekGoWhenAnotherDownloadStarts() {
        place("old.litertlm.tmp", 8)
        place("old.litertlm.tmp.json", 8)
        place("lone.litertlm.tmp", 8)
        place("record.litertlm.tmp.json", 8)
        // A pair is as old as its newer file: the record is written once, the
        // .tmp with every chunk
        place("recent.litertlm.tmp", 6)
        place("recent.litertlm.tmp.json", 8)
        // A model is never a partial download, however old
        place("model.litertlm", 30)
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (path, error) = await(store.downloadFile(url("/other.litertlm"), "other.litertlm", "{}", "") {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals(
            listOf("model.litertlm", "other.litertlm", "recent.litertlm.tmp", "recent.litertlm.tmp.json"),
            modelsDir.list()!!.sorted()
        )
    }

    @Test
    fun aWeekOldPartialDownloadOfTheFileItselfIsContinued() {
        leavePartial("aged.litertlm", cut = 100 * 1024)
        age("aged.litertlm.tmp", 8)
        age("aged.litertlm.tmp.json", 8)
        server.enqueue(rest(100 * 1024))

        val (path, error) = await(
            store.downloadFile(url("/aged.litertlm"), "aged.litertlm", "{}", sha256Of(payload)) {}
        )

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(path!!).readBytes())
        assertEquals("bytes=${100 * 1024}-", server.takeRequest().getHeader("Range"))
    }

    @Test
    fun aWeekOldPartialDownloadThatIsDownloadingStays() {
        leavePartial("held.litertlm", cut = 100 * 1024)
        age("held.litertlm.tmp", 8)
        age("held.litertlm.tmp.json", 8)
        // Its download continues it, but the server takes 2 s to answer: the
        // download holds the file's lock while the partial file is a week old
        server.enqueue(rest(100 * 1024).setHeadersDelay(2, TimeUnit.SECONDS))
        val running = onThread { store.downloadFile(url("/held.litertlm"), "held.litertlm", "{}", "") {} }
        server.takeRequest()
        server.enqueue(MockResponse().setBody(Buffer().write(payload)))

        val (other, error) = await(store.downloadFile(url("/other.litertlm"), "other.litertlm", "{}", "") {})

        assertNull("got: $error", error)
        assertArrayEquals(payload, File(other!!).readBytes())
        // The partial file was still there to continue: the whole file, not just its rest
        val (path, error2) = running()
        assertNull("got: $error2", error2)
        assertArrayEquals(payload, File(path!!).readBytes())
    }

    @Test
    fun aWeekOldPartialDownloadWhoseContinuationFailsIsDiscarded() {
        leavePartial("abandoned.litertlm", cut = 100 * 1024)
        age("abandoned.litertlm.tmp", 8)
        age("abandoned.litertlm.tmp.json", 8)
        server.enqueue(MockResponse().setResponseCode(503))

        val (_, error) = await(store.downloadFile(url("/abandoned.litertlm"), "abandoned.litertlm", "{}", "") {})

        assertTrue("got: $error", error!!.message!!.contains("HTTP 503"))
        assertFalse(File(modelsDir, "abandoned.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "abandoned.litertlm.tmp.json").exists())
    }

    @Test
    fun aWeekOldPartialDownloadThatACallIsAboutToContinueStays() {
        leavePartial("queued.litertlm", cut = 100 * 1024)
        age("queued.litertlm.tmp", 8)
        age("queued.litertlm.tmp.json", 8)
        // A call for the file that has yet to take its lock, as one waiting for it
        val queued = HybridModelStore.downloadsOf("queued.litertlm")
        queued.calls.incrementAndGet()
        try {
            server.enqueue(MockResponse().setBody(Buffer().write(payload)))

            val (_, error) = await(store.downloadFile(url("/other.litertlm"), "other.litertlm", "{}", "") {})

            assertNull("got: $error", error)
            assertTrue(File(modelsDir, "queued.litertlm.tmp").exists())
            assertTrue(File(modelsDir, "queued.litertlm.tmp.json").exists())
        } finally {
            queued.calls.decrementAndGet()
        }
    }

    @Test
    fun aCallThatWaitsDeletesOldPartialDownloadsOnlyWhenItsTurnComes() {
        // The server answers only once the gate opens: until then the first
        // download holds the file's lock
        val gate = CountDownLatch(1)
        server.dispatcher = object : Dispatcher() {
            override fun dispatch(request: RecordedRequest): MockResponse {
                gate.await(10, TimeUnit.SECONDS)
                return MockResponse().setBody(Buffer().write(payload))
            }
        }
        val first = onThread { store.downloadFile(url("/turn.litertlm"), "turn.litertlm", "{}", "") {} }
        server.takeRequest()
        // A week-old partial download placed after the first call cleaned up
        place("stale.litertlm.tmp", 8)
        place("stale.litertlm.tmp.json", 8)
        val second = onThread { store.downloadFile(url("/turn.litertlm"), "turn.litertlm", "{}", "") {} }
        val keptWhileWaiting = try {
            waitFor("the second call waits for the first") {
                ShadowLog.getLogsForTag("HybridModelStore").any {
                    it.msg == "Waiting for the download of turn.litertlm already running"
                }
            }
            File(modelsDir, "stale.litertlm.tmp").exists() && File(modelsDir, "stale.litertlm.tmp.json").exists()
        } finally {
            gate.countDown()
        }

        val (path1, error1) = first()
        val (path2, error2) = second()
        assertTrue("The waiting call cleaned up before it held the file's lock", keptWhileWaiting)
        assertNull("got: $error1", error1)
        assertNull("got: $error2", error2)
        assertEquals(path1, path2)
        assertArrayEquals(payload, File(path1!!).readBytes())
        // Once its turn came it cleaned up, and found the file in the cache
        assertFalse(File(modelsDir, "stale.litertlm.tmp").exists())
        assertFalse(File(modelsDir, "stale.litertlm.tmp.json").exists())
        assertEquals(1, server.requestCount)
    }

    // ── What a failed download leaves open ──────────────────────────────────

    // The test JVM's own counts, through reflection: java.lang.management is not
    // in android.jar, which the unit tests compile against
    private val system = runCatching {
        Class.forName("java.lang.management.ManagementFactory").getMethod("getOperatingSystemMXBean").invoke(null)
    }.getOrNull()
    private val fdCount = runCatching {
        Class.forName("com.sun.management.UnixOperatingSystemMXBean").getMethod("getOpenFileDescriptorCount")
    }.getOrNull()

    private fun openFileDescriptors(): Long = fdCount!!.invoke(system) as Long

    private fun collections(): Long {
        val management = Class.forName("java.lang.management.ManagementFactory")
        val count = Class.forName("java.lang.management.GarbageCollectorMXBean").getMethod("getCollectionCount")
        return (management.getMethod("getGarbageCollectorMXBeans").invoke(null) as List<*>).sumOf { count.invoke(it) as Long }
    }

    /**
     * Runs [download] (a download that fails) 10 times and checks that no file
     * descriptor stays open. A stream or socket left open is closed only when
     * a garbage collection finds it, so the count is taken again when a
     * collection ran in between.
     */
    private fun assertFailuresLeaveNothingOpen(download: (Int) -> Unit) {
        assumeTrue("needs a JVM that counts open file descriptors", system != null && fdCount?.declaringClass?.isInstance(system) == true)
        download(0) // the classes and the TLS session it needs, before counting
        repeat(3) { attempt ->
            val collected = collections()
            val before = openFileDescriptors()
            repeat(10) { download(1 + attempt * 10 + it) }
            // The server closes its side of a connection once the client has
            val deadline = System.currentTimeMillis() + 5_000
            var left = openFileDescriptors() - before
            while (left > 3 && System.currentTimeMillis() < deadline) {
                Thread.sleep(50)
                left = openFileDescriptors() - before
            }
            assertTrue("10 failed downloads left $left more file descriptors open", left <= 3)
            if (collections() == collected) return
        }
    }

    @Test
    fun aConnectionDroppedMidReadLeavesNoFileOrSocketOpen() {
        assertFailuresLeaveNothingOpen { i ->
            // Chunked, so a body that stops short fails the read itself
            server.enqueue(
                MockResponse()
                    .setChunkedBody(Buffer().write(payload, 0, 64 * 1024), 8 * 1024)
                    .setSocketPolicy(SocketPolicy.DISCONNECT_DURING_RESPONSE_BODY)
            )
            val (_, error) = await(store.downloadFile(url("/drop$i.litertlm"), "drop$i.litertlm", "{}", "") {})
            assertNotNull("A dropped download must reject", error)
        }
    }

    @Test
    fun anHttpErrorLeavesNoSocketOpen() {
        assertFailuresLeaveNothingOpen { i ->
            server.enqueue(MockResponse().setResponseCode(404).setBody("Not here"))
            val (_, error) = await(store.downloadFile(url("/none$i.litertlm"), "none$i.litertlm", "{}", "") {})
            assertTrue("got: $error", error!!.message!!.contains("HTTP 404"))
        }
    }

    @Test
    fun aServerErrorOnTheContinuationLeavesNoSocketOpen() {
        leavePartial("again503.litertlm", cut = 100 * 1024)
        assertFailuresLeaveNothingOpen {
            server.enqueue(MockResponse().setResponseCode(503).setBody("Busy"))
            val (_, error) = await(store.downloadFile(url("/again503.litertlm"), "again503.litertlm", "{}", "") {})
            assertTrue("got: $error", error!!.message!!.contains("HTTP 503"))
            assertEquals(100L * 1024, File(modelsDir, "again503.litertlm.tmp").length())
        }
    }
}
