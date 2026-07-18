package com.margelo.nitro.dev.litert.litertlm

import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.net.InetAddress
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
 * disconnects (no half-written files), and the cached-file short-circuit.
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
            store.downloadFile(url("/model.litertlm"), "model.litertlm", "{}") { progress.add(it) }
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
            store.downloadFile(url("/missing.litertlm"), "missing.litertlm", "{}") {}
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
            store.downloadFile(url("/cut.litertlm"), "cut.litertlm", "{}") {}
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
            store.downloadFile(url("/cached.litertlm"), "cached.litertlm", "{}") { progress.add(it) }
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
            ) {}
        )

        assertNull(error)
        val recorded = server.takeRequest()
        assertEquals("Bearer secret-token", recorded.getHeader("Authorization"))
    }
}
