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
 * continuing a partial download with a Range request, a stalled read, two
 * downloads of one file at once, the sha256 check, old partial downloads
 * deleted when another download starts, and what a failed download leaves open.
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
    private fun leavePartial(name: String, cut: Int, etag: String = "\"v1\"", sha256: String = "") {
        server.enqueue(
            MockResponse()
                .setBody(Buffer().write(payload, 0, cut))
                .setHeader("Content-Length", payload.size)
                .setHeader("ETag", etag)
                .setSocketPolicy(SocketPolicy.DISCONNECT_AT_END)
        )
        val (_, error) = await(store.downloadFile(url("/$name"), name, "{}", sha256) {})
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
    fun aCachedFileIsReturnedWithoutHashingIt() {
        modelsDir.mkdirs()
        File(modelsDir, "cachedsha.litertlm").writeBytes(ByteArray(128) { 1 })

        val (path, error) = await(
            store.downloadFile(url("/cachedsha.litertlm"), "cachedsha.litertlm", "{}", "0".repeat(64)) {}
        )

        assertNull(error)
        assertEquals(File(modelsDir, "cachedsha.litertlm").absolutePath, path)
        assertEquals(0, server.requestCount)
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
