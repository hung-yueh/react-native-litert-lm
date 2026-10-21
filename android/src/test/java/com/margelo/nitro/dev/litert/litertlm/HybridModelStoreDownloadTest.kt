package com.margelo.nitro.dev.litert.litertlm

import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.net.InetAddress
import java.security.MessageDigest
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLSocketFactory
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okhttp3.mockwebserver.SocketPolicy
import okhttp3.tls.HandshakeCertificates
import okhttp3.tls.HeldCertificate
import okio.Buffer
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * Exercises HybridModelStore's real download machinery — HTTPS enforcement is
 * covered elsewhere; these tests cover what happens when the network answers:
 * success + atomic rename, progress reporting, HTTP errors, mid-stream
 * disconnects (no half-written files), the cached-file short-circuit,
 * continuing a partial download with a Range request, and the sha256 check.
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
}
