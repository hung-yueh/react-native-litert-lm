package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import org.json.JSONObject

@DoNotStrip
@Keep
class HybridModelStore : HybridModelStoreSpec() {

    private val tag = "HybridModelStore"

    private val modelsDirectory: File
        get() {
            val context = LiteRTLMInitProvider.applicationContext 
                ?: throw RuntimeException("Android Application Context is not available")
            val modelsDir = File(context.filesDir, "models")
            if (!modelsDir.exists()) {
                modelsDir.mkdirs()
            }
            return modelsDir
        }

    private fun sanitizeFileName(fileName: String) {
        if (fileName.contains("..") || fileName.contains("/") || fileName.contains("\\")) {
            throw IllegalArgumentException("Invalid filename: path traversal or directory separators are not allowed.")
        }
    }

    override fun isCached(fileName: String): Boolean {
        sanitizeFileName(fileName)
        val file = File(modelsDirectory, fileName)
        return file.exists() && file.length() > 0
    }

    override fun getFilePath(fileName: String): String {
        sanitizeFileName(fileName)
        return File(modelsDirectory, fileName).absolutePath
    }

    override fun getFileSizeBytes(absolutePath: String): Double {
        return try {
            val file = File(absolutePath)
            if (file.exists() && file.isFile) file.length().toDouble() else -1.0
        } catch (e: Exception) {
            -1.0
        }
    }

    override fun listCachedFiles(): Array<ModelFile> {
        return try {
            val dir = modelsDirectory
            val files = dir.listFiles() ?: return emptyArray()
            val result = mutableListOf<ModelFile>()
            for (f in files) {
                if (f.isFile && !f.name.endsWith(".tmp") && !f.name.endsWith(RESUME_SUFFIX)) {
                    result.add(
                        ModelFile(
                            fileName = f.name,
                            absolutePath = f.absolutePath,
                            sizeBytes = f.length().toDouble(),
                            lastModifiedMs = f.lastModified().toDouble()
                        )
                    )
                }
            }
            result.toTypedArray()
        } catch (e: Exception) {
            Log.e(tag, "Failed to list cached files", e)
            emptyArray()
        }
    }

    override fun deleteFile(fileName: String) {
        sanitizeFileName(fileName)
        // A partial download of the file goes too
        File(modelsDirectory, "$fileName.tmp").delete()
        File(modelsDirectory, "$fileName$RESUME_SUFFIX").delete()
        val file = File(modelsDirectory, fileName)
        if (file.exists()) {
            val deleted = file.delete()
            if (!deleted) {
                throw RuntimeException("Failed to delete model file: ${file.absolutePath}")
            }
        }
    }

    override fun downloadFile(
        url: String,
        fileName: String,
        headersJson: String,
        sha256: String,
        onProgress: (progress: Double) -> Unit
    ): Promise<String> {
        return Promise.parallel {
            Log.i(tag, "downloadFile: $url -> $fileName")
            sanitizeFileName(fileName)

            if (!url.startsWith("https://", ignoreCase = true)) {
                throw IllegalArgumentException("Invalid download URL: HTTPS is required for security.")
            }

            val dir = modelsDirectory
            val modelFile = File(dir, fileName)
            val tempFile = File(dir, "$fileName.tmp")
            val resumeFile = File(dir, "$fileName$RESUME_SUFFIX")

            // Fast cache check
            if (modelFile.exists() && modelFile.length() > 0) {
                Log.i(tag, "Model already exists: ${modelFile.absolutePath}")
                onProgress(1.0)
                return@parallel modelFile.absolutePath
            }

            Log.i(tag, "Downloading model to temp file: ${tempFile.absolutePath}")
            onProgress(0.0)

            // Parse headers
            val headersMap = mutableMapOf<String, String>()
            if (headersJson.isNotEmpty()) {
                try {
                    val json = JSONObject(headersJson)
                    val keys = json.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        headersMap[key] = json.getString(key)
                    }
                } catch (e: Exception) {
                    Log.e(tag, "Failed to parse custom headers JSON", e)
                }
            }

            // A partial file left by an earlier call is continued only when it
            // came from the same URL and that response named its version (an
            // ETag or Last-Modified) and its size.
            val partial = readPartial(resumeFile)?.takeIf {
                it.url == url && tempFile.length() in 1 until it.totalBytes
            }
            if (partial == null) discardPartial(tempFile, resumeFile)
            var offset = if (partial != null) tempFile.length() else 0L
            var keepPartial = partial != null

            try {
                var connection = open(url, headersMap, offset, partial?.validator)
                if (partial != null) {
                    val code = connection.responseCode
                    if (!continues(connection, offset, partial)) {
                        // 200 = the whole file (the server ignored the range, or the
                        // file changed); 416 or a 206 of another version = the partial
                        // file is no use. Anything else fails and keeps it for next time.
                        if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL && code != 416) {
                            throw RuntimeException("Failed to download model: HTTP $code")
                        }
                        Log.i(tag, "Not continuing the partial download (HTTP $code): starting over")
                        discardPartial(tempFile, resumeFile)
                        keepPartial = false
                        offset = 0L
                        if (code != HttpURLConnection.HTTP_OK) {
                            connection.disconnect()
                            connection = open(url, headersMap, 0L, null)
                        }
                    }
                }

                if (offset == 0L && connection.responseCode != HttpURLConnection.HTTP_OK) {
                    throw RuntimeException("Failed to download model: HTTP ${connection.responseCode}")
                }

                val contentLength = connection.contentLengthLong
                val totalBytes = if (offset > 0) partial!!.totalBytes else contentLength
                if (offset == 0L) {
                    // What the next call needs to continue this download
                    val validator = validatorOf(connection)
                    if (validator != null && contentLength > 0) {
                        resumeFile.writeText(
                            JSONObject()
                                .put("url", url)
                                .put("validator", validator)
                                .put("totalBytes", contentLength)
                                .toString()
                        )
                        keepPartial = true
                    }
                }

                val digest = if (sha256.isNotEmpty()) MessageDigest.getInstance("SHA-256") else null
                if (offset > 0) {
                    Log.i(tag, "Continuing the partial download at byte $offset of $totalBytes")
                    if (digest != null) {
                        val started = System.currentTimeMillis()
                        hashPrefix(digest, tempFile, offset)
                        Log.i(tag, "Hashed the $offset bytes already downloaded in ${System.currentTimeMillis() - started} ms")
                    }
                    onProgress(offset.toDouble() / totalBytes.toDouble())
                }

                val input = connection.inputStream
                val output = FileOutputStream(tempFile, offset > 0)

                val buffer = ByteArray(8 * 1024)
                var bytesRead: Int
                var totalBytesRead = 0L
                var lastProgressUpdate = 0L

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    digest?.update(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead

                    if (totalBytes > 0) {
                        val currentTime = System.currentTimeMillis()
                        // Update progress roughly every 100ms to prevent bridge flooding
                        if (currentTime - lastProgressUpdate > 100) {
                            val progress = (offset + totalBytesRead).toDouble() / totalBytes.toDouble()
                            onProgress(progress)
                            lastProgressUpdate = currentTime
                        }
                    }
                }

                output.flush()
                output.close()
                input.close()
                connection.disconnect()

                // A dropped connection can end the read loop early without an
                // exception — never install a truncated model file.
                if (contentLength > 0 && totalBytesRead != contentLength) {
                    throw RuntimeException(
                        "Incomplete download: got $totalBytesRead of $contentLength bytes"
                    )
                }
                Log.i(tag, "Received $totalBytesRead bytes from byte $offset: ${offset + totalBytesRead} of $totalBytes bytes")

                if (digest != null) {
                    val started = System.currentTimeMillis()
                    val actual = digest.digest().joinToString("") { "%02x".format(it) }
                    if (actual != sha256) {
                        keepPartial = false
                        throw RuntimeException("sha256 mismatch: expected $sha256, got $actual")
                    }
                    Log.i(tag, "sha256 matches ($actual), final digest in ${System.currentTimeMillis() - started} ms")
                }

                // Atomic rename
                if (tempFile.renameTo(modelFile)) {
                    resumeFile.delete()
                    Log.i(tag, "Download complete and verified at: ${modelFile.absolutePath}")
                    onProgress(1.0)
                    modelFile.absolutePath
                } else {
                    throw RuntimeException("Failed to rename temporary file to model file")
                }
            } catch (e: Exception) {
                Log.e(tag, "Download failed", e)
                if (keepPartial && tempFile.length() > 0) {
                    Log.i(tag, "Kept ${tempFile.length()} bytes of the partial download for the next call")
                } else {
                    discardPartial(tempFile, resumeFile)
                }
                throw RuntimeException("Download failed: ${e.message}", e)
            }
        }
    }

    /** Where a partial download came from, as its first response described it. */
    private class Partial(val url: String, val validator: String, val totalBytes: Long)

    private fun readPartial(file: File): Partial? = try {
        if (!file.exists()) null
        else JSONObject(file.readText()).let {
            Partial(it.getString("url"), it.getString("validator"), it.getLong("totalBytes"))
        }
    } catch (e: Exception) {
        null
    }

    private fun discardPartial(tempFile: File, resumeFile: File) {
        tempFile.delete()
        resumeFile.delete()
    }

    private fun open(
        url: String,
        headers: Map<String, String>,
        offset: Long,
        validator: String?
    ): HttpURLConnection {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.connectTimeout = 15000 // 15s
        connection.readTimeout = 0 // Infinite read timeout for large files
        connection.doInput = true

        // Apply headers
        for ((key, value) in headers) {
            connection.setRequestProperty(key, value)
        }
        if (offset > 0 && validator != null) {
            connection.setRequestProperty("Range", "bytes=$offset-")
            connection.setRequestProperty("If-Range", validator)
        }

        connection.connect()
        return connection
    }

    /** The version a response is of: a strong ETag, else Last-Modified (what If-Range accepts). */
    private fun validatorOf(connection: HttpURLConnection): String? {
        val etag = connection.getHeaderField("ETag")
        if (etag != null && !etag.startsWith("W/")) return etag
        return connection.getHeaderField("Last-Modified")
    }

    /**
     * A 206 with the rest of the file the partial download holds. The
     * validator is compared here too: some servers answer a Range request
     * without evaluating If-Range (the Hugging Face CDN does).
     */
    private fun continues(connection: HttpURLConnection, offset: Long, partial: Partial): Boolean {
        if (connection.responseCode != HttpURLConnection.HTTP_PARTIAL) return false
        val range = CONTENT_RANGE.matchEntire(connection.getHeaderField("Content-Range")?.trim() ?: "")
            ?: return false
        val (first, last, size) = range.destructured
        return first.toLong() == offset &&
            last.toLong() == partial.totalBytes - 1 &&
            size.toLong() == partial.totalBytes &&
            validatorOf(connection) == partial.validator
    }

    private fun hashPrefix(digest: MessageDigest, file: File, length: Long) {
        FileInputStream(file).use { input ->
            val buffer = ByteArray(1024 * 1024)
            var left = length
            while (left > 0) {
                val n = input.read(buffer, 0, minOf(buffer.size.toLong(), left).toInt())
                if (n < 0) throw IOException("The partial download is shorter than $length bytes")
                digest.update(buffer, 0, n)
                left -= n
            }
        }
    }

    private companion object {
        /** Next to `<file>.tmp`: the URL, validator and size a partial download continues from. */
        const val RESUME_SUFFIX = ".tmp.json"
        val CONTENT_RANGE = Regex("""bytes (\d+)-(\d+)/(\d+)""")
    }
}
