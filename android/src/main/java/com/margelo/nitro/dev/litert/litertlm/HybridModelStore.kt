package com.margelo.nitro.dev.litert.litertlm

import android.system.ErrnoException
import android.system.OsConstants
import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.locks.ReentrantLock
import org.json.JSONObject

@DoNotStrip
@Keep
class HybridModelStore : HybridModelStoreSpec() {

    private val tag = "HybridModelStore"

    /**
     * How long a read waits for data before the download fails and keeps its
     * partial file for the next call. Without it a dead connection (say after
     * a switch from Wi-Fi to cellular) blocks the read forever. Tests shorten it.
     */
    internal var readTimeoutMs = 60_000

    /**
     * A partial download is synced to disk, and its record updated, every
     * this many bytes: a power cut loses what came after, never corrupts it.
     * Tests shorten it.
     */
    internal var syncBytes = 64L * 1024 * 1024

    /** Free bytes on the volume of a directory. Tests replace it. */
    internal var usableSpace: (File) -> Long = { it.usableSpace }

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
        // The cache keeps partial downloads and verified hashes next to the models under these suffixes
        if (fileName.isEmpty() || fileName == "." || RESERVED_SUFFIXES.any { fileName.endsWith(it) }) {
            throw IllegalArgumentException(
                "Invalid filename \"$fileName\": the cache reserves names ending in ${RESERVED_SUFFIXES.joinToString()}."
            )
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
                if (f.isFile && RESERVED_SUFFIXES.none { f.name.endsWith(it) }) {
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
        // A partial download of the file goes too, unless a download is
        // writing it right now
        val lock = downloadsOf(fileName).lock
        if (lock.tryLock()) {
            try {
                File(modelsDirectory, "$fileName.tmp").delete()
                File(modelsDirectory, "$fileName$RESUME_SUFFIX").delete()
            } finally {
                lock.unlock()
            }
        }
        File(modelsDirectory, "$fileName$VERIFIED_SUFFIX").delete()
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

            // One download of a file at a time in this process: a second call
            // waits here (on its own thread) for the first to end, then finds
            // the file in the cache or continues the partial file it left. The
            // count keeps other calls' clean-up off the file meanwhile.
            val downloads = downloadsOf(fileName)
            downloads.calls.incrementAndGet()
            try {
                if (!downloads.lock.tryLock()) {
                    Log.i(tag, "Waiting for the download of $fileName already running")
                    downloads.lock.lock()
                }
                try {
                    // Week-old partial downloads go while this call holds its lock, so another
                    // call's clean-up never deletes the partial file this call continues.
                    expireStalePartials(fileName)
                    download(url, fileName, headersJson, sha256, onProgress)
                } finally {
                    downloads.lock.unlock()
                }
            } finally {
                downloads.calls.decrementAndGet()
            }
        }
    }

    /**
     * Deletes the partial downloads (`<file>.tmp` and `<file>.tmp.json`) that
     * nothing has written to for [PARTIAL_MAX_AGE_MS], except [fileName]'s own
     * (its download continues it, or starts over when the server's version
     * changed) and those of files a call is downloading or waiting to
     * download. A pair is as old as the newer of its files: `.tmp` changes
     * with every write, `.tmp.json` with every sync. A failure here is logged
     * and does not stop the download.
     */
    private fun expireStalePartials(fileName: String) {
        try {
            val dir = modelsDirectory
            val names = (dir.list() ?: return).mapNotNull {
                when {
                    it.endsWith(RESUME_SUFFIX) -> it.removeSuffix(RESUME_SUFFIX)
                    it.endsWith(".tmp") -> it.removeSuffix(".tmp")
                    else -> null
                }
            }.toSet() - fileName
            val now = System.currentTimeMillis()
            for (name in names) {
                val downloads = downloadsOf(name)
                if (!downloads.lock.tryLock()) continue
                try {
                    // A call waiting for the lock is about to continue this partial file
                    if (downloads.calls.get() > 0) continue
                    // Read under the lock: a download may have written it since the listing
                    val files = listOf(File(dir, "$name.tmp"), File(dir, "$name$RESUME_SUFFIX")).filter { it.isFile }
                    val age = now - (files.maxOfOrNull { it.lastModified() } ?: continue)
                    if (age < PARTIAL_MAX_AGE_MS) continue
                    for (file in files) {
                        if (!file.delete()) Log.w(tag, "Could not delete ${file.absolutePath}")
                    }
                    Log.i(tag, "Deleted the partial download of $name: last written ${age / DAY_MS} days ago")
                } finally {
                    downloads.lock.unlock()
                }
            }
        } catch (e: Exception) {
            Log.w(tag, "Could not delete old partial downloads", e)
        }
    }

    private fun download(
        url: String,
        fileName: String,
        headersJson: String,
        sha256: String,
        onProgress: (progress: Double) -> Unit
    ): String {
        val dir = modelsDirectory
        val modelFile = File(dir, fileName)
        val tempFile = File(dir, "$fileName.tmp")
        val resumeFile = File(dir, "$fileName$RESUME_SUFFIX")
        val verifiedFile = File(dir, "$fileName$VERIFIED_SUFFIX")

        // Fast cache check. With a sha256 the cached file must be the one it
        // names: another revision under the same name is downloaded again.
        if (modelFile.exists() && modelFile.length() > 0) {
            if (sha256.isEmpty() || matchesSha256(modelFile, verifiedFile, sha256)) {
                Log.i(tag, "Model already exists: ${modelFile.absolutePath}")
                onProgress(1.0)
                return modelFile.absolutePath
            }
            Log.i(tag, "The cached $fileName is not the file sha256 $sha256 names: downloading it again")
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
        // came from the same URL (its query aside: a signed URL changes with
        // every call) and that response named its version (an ETag or
        // Last-Modified) and its size. Only the bytes synced to disk count:
        // after a power cut the ones after them may never have been written.
        val source = withoutQuery(url)
        var partial = readPartial(resumeFile)?.takeIf {
            it.url == source && it.syncedBytes in 1 until it.totalBytes && tempFile.length() >= it.syncedBytes
        }
        // Before anything writes to the pair: a partial that nothing could add
        // to for a week goes when its continuation fails
        val stale = partial != null &&
            System.currentTimeMillis() - maxOf(tempFile.lastModified(), resumeFile.lastModified()) >= PARTIAL_MAX_AGE_MS
        if (partial == null) {
            discardPartial(tempFile, resumeFile)
        } else if (tempFile.length() > partial.syncedBytes) {
            RandomAccessFile(tempFile, "rw").use { it.setLength(partial.syncedBytes) }
        }
        var offset = partial?.syncedBytes ?: 0L
        var keepPartial = partial != null

        val digest = if (sha256.isNotEmpty()) MessageDigest.getInstance("SHA-256") else null
        if (partial != null) {
            onProgress(offset.toDouble() / partial.totalBytes.toDouble())
            // Hashed before the request, so the response doesn't wait on it
            if (digest != null) {
                try {
                    val started = System.currentTimeMillis()
                    hashPrefix(digest, tempFile, offset)
                    Log.i(tag, "Hashed the $offset bytes already downloaded in ${System.currentTimeMillis() - started} ms")
                } catch (e: IOException) {
                    Log.w(tag, "Could not read the partial download: starting over", e)
                    discardPartial(tempFile, resumeFile)
                    digest.reset()
                    partial = null
                    offset = 0L
                    keepPartial = false
                }
            }
        }

        // However the download ends, its streams are closed and the connection
        // it has open is disconnected (one is disconnected before a second
        // opens): a failed download must not hold a socket or a file until a GC.
        var connection: HttpURLConnection? = null
        return try {
            connection = open(url, headersMap, offset, partial?.validator)
            if (partial != null) {
                val code = connection.responseCode
                if (!continues(connection, offset, partial)) {
                    // 200 = the whole file (the server ignored the range, or the
                    // file changed); 416 or a 206 of another version = the partial
                    // file is no use. Anything else fails and keeps it for next
                    // time, unless the file is gone (404, 410) or the partial has
                    // gone a week without growing: in an app with one model, no
                    // other download's clean-up would ever delete it.
                    if (code != HttpURLConnection.HTTP_OK && code != HttpURLConnection.HTTP_PARTIAL && code != 416) {
                        if (code == HttpURLConnection.HTTP_NOT_FOUND || code == HttpURLConnection.HTTP_GONE || stale) {
                            keepPartial = false
                        }
                        throw RuntimeException("Failed to download model: HTTP $code")
                    }
                    Log.i(tag, "Not continuing the partial download (HTTP $code): starting over")
                    discardPartial(tempFile, resumeFile)
                    digest?.reset()
                    partial = null
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
            // Not one byte is written without room for the rest of the file:
            // a download that fills the disk would leave it full
            if (totalBytes > 0) requireFreeSpace(dir, totalBytes - offset)

            // What the next call needs to continue this download
            var record = partial
            if (offset == 0L) {
                val validator = validatorOf(connection)
                if (validator != null && contentLength > 0) {
                    record = Partial(source, validator, contentLength, 0L)
                    writePartial(resumeFile, record)
                    keepPartial = true
                }
            }

            if (offset > 0) {
                Log.i(tag, "Continuing the partial download at byte $offset of $totalBytes")
            }

            var totalBytesRead = 0L
            connection.inputStream.use { input ->
                FileOutputStream(tempFile, offset > 0).use { output ->
                    val buffer = ByteArray(8 * 1024)
                    var bytesRead: Int
                    var lastProgressUpdate = 0L
                    var synced = offset

                    try {
                        while (input.read(buffer).also { bytesRead = it } != -1) {
                            output.write(buffer, 0, bytesRead)
                            digest?.update(buffer, 0, bytesRead)
                            totalBytesRead += bytesRead

                            if (record != null && offset + totalBytesRead - synced >= syncBytes) {
                                output.fd.sync()
                                synced = offset + totalBytesRead
                                writePartial(resumeFile, record.copy(syncedBytes = synced))
                            }

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
                    } finally {
                        // However the transfer ends, what it wrote reaches the disk
                        // before the record says so, or before the file is installed
                        try {
                            output.fd.sync()
                            if (record != null) writePartial(resumeFile, record.copy(syncedBytes = offset + totalBytesRead))
                        } catch (e: IOException) {
                            Log.w(tag, "Could not sync the download to disk", e)
                        }
                    }
                }
            }

            // A dropped connection can end the read loop early without an
            // exception — never install a truncated model file. A continued
            // download must reach the size its first response gave: a 206
            // sent chunked has no Content-Length to compare with.
            if (offset > 0 && offset + totalBytesRead != totalBytes) {
                throw RuntimeException(
                    "Incomplete download: got ${offset + totalBytesRead} of $totalBytes bytes"
                )
            }
            if (offset == 0L && contentLength > 0 && totalBytesRead != contentLength) {
                throw RuntimeException(
                    "Incomplete download: got $totalBytesRead of $contentLength bytes"
                )
            }
            Log.i(tag, "Received $totalBytesRead bytes from byte $offset: ${offset + totalBytesRead} of $totalBytes bytes")

            if (digest != null) {
                val started = System.currentTimeMillis()
                val actual = hex(digest.digest())
                if (actual != sha256) {
                    keepPartial = false
                    throw RuntimeException("sha256 mismatch: expected $sha256, got $actual")
                }
                Log.i(tag, "sha256 matches ($actual), final digest in ${System.currentTimeMillis() - started} ms")
            }

            // An old hash must never vouch for the new file
            verifiedFile.delete()
            // Atomic rename
            if (tempFile.renameTo(modelFile)) {
                resumeFile.delete()
                if (digest != null) writeVerified(verifiedFile, modelFile, sha256)
                Log.i(tag, "Download complete and verified at: ${modelFile.absolutePath}")
                onProgress(1.0)
                modelFile.absolutePath
            } else {
                throw RuntimeException("Failed to rename temporary file to model file")
            }
        } catch (e: Exception) {
            Log.e(tag, "Download failed", e)
            // A disk the download filled must not stay full
            if (isOutOfSpace(e)) keepPartial = false
            if (keepPartial && tempFile.length() > 0) {
                Log.i(tag, "Kept ${tempFile.length()} bytes of the partial download for the next call")
            } else {
                discardPartial(tempFile, resumeFile)
            }
            throw RuntimeException("Download failed: ${e.message}", e)
        } finally {
            connection?.disconnect()
        }
    }

    /**
     * Where a partial download came from, as its first response described it,
     * and how many of its bytes are synced to disk.
     */
    private data class Partial(val url: String, val validator: String, val totalBytes: Long, val syncedBytes: Long)

    private fun readPartial(file: File): Partial? = try {
        if (!file.exists()) null
        else JSONObject(file.readText()).let {
            Partial(it.getString("url"), it.getString("validator"), it.getLong("totalBytes"), it.optLong("syncedBytes", -1))
        }
    } catch (e: Exception) {
        null
    }

    private fun writePartial(file: File, partial: Partial) {
        file.writeText(
            JSONObject()
                .put("url", partial.url)
                .put("validator", partial.validator)
                .put("totalBytes", partial.totalBytes)
                .put("syncedBytes", partial.syncedBytes)
                .toString()
        )
    }

    private fun discardPartial(tempFile: File, resumeFile: File) {
        tempFile.delete()
        resumeFile.delete()
    }

    /**
     * Whether [model] is the file [sha256] names. The hash a download checked
     * is kept in [verified] with the file's size and modification time. A file
     * without one (cached before 0.8.2, or installed without a sha256), or one
     * changed since, is hashed once and the result kept.
     */
    private fun matchesSha256(model: File, verified: File, sha256: String): Boolean {
        val size = model.length()
        val modified = model.lastModified()
        try {
            val record = JSONObject(verified.readText())
            if (record.getLong("size") == size && record.getLong("modified") == modified) {
                return record.getString("sha256") == sha256
            }
        } catch (e: Exception) {
            // No record, or an unreadable one: hash the file
        }
        val actual = try {
            val started = System.currentTimeMillis()
            val digest = MessageDigest.getInstance("SHA-256")
            hashPrefix(digest, model, size)
            hex(digest.digest()).also {
                Log.i(tag, "Hashed the cached ${model.name} in ${System.currentTimeMillis() - started} ms")
            }
        } catch (e: IOException) {
            Log.w(tag, "Could not hash the cached ${model.name}", e)
            return false
        }
        writeVerified(verified, model, actual)
        return actual == sha256
    }

    private fun writeVerified(verified: File, model: File, sha256: String) {
        try {
            verified.writeText(
                JSONObject()
                    .put("sha256", sha256)
                    .put("size", model.length())
                    .put("modified", model.lastModified())
                    .toString()
            )
        } catch (e: IOException) {
            Log.w(tag, "Could not record the sha256 of ${model.name}", e)
        }
    }

    private fun requireFreeSpace(dir: File, needed: Long) {
        val free = usableSpace(dir)
        if (free < needed + FREE_SPACE_MARGIN_BYTES) {
            throw IOException(
                "Not enough free storage: ${needed / MIB} MB left to download, ${free / MIB} MB free"
            )
        }
    }

    private fun isOutOfSpace(e: Throwable): Boolean = generateSequence(e) { it.cause }.any {
        (it is ErrnoException && it.errno == OsConstants.ENOSPC) ||
            it.message?.contains("ENOSPC") == true ||
            it.message?.contains("No space left on device") == true
    }

    private fun open(
        url: String,
        headers: Map<String, String>,
        offset: Long,
        validator: String?
    ): HttpURLConnection {
        val connection = URL(url).openConnection() as HttpURLConnection
        connection.connectTimeout = 15000 // 15s
        // A stalled read fails like a dropped connection: the partial file stays
        connection.readTimeout = readTimeoutMs
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
                if (n < 0) throw IOException("${file.name} is shorter than $length bytes")
                digest.update(buffer, 0, n)
                left -= n
            }
        }
    }

    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { "%02x".format(it) }

    /** The URL a partial download is recorded under: a signed URL's query changes with every call. */
    private fun withoutQuery(url: String): String = url.substringBefore('#').substringBefore('?')

    internal companion object {
        /** Next to `<file>.tmp`: the URL, validator, size and synced bytes a partial download continues from. */
        const val RESUME_SUFFIX = ".tmp.json"
        /** Next to a model: the sha256 a download checked it against, with its size and modification time. */
        const val VERIFIED_SUFFIX = ".sha256"
        val RESERVED_SUFFIXES = listOf(".tmp", RESUME_SUFFIX, VERIFIED_SUFFIX)
        val CONTENT_RANGE = Regex("""bytes (\d+)-(\d+)/(\d+)""")

        const val DAY_MS = 24L * 60 * 60 * 1000
        /** A partial download nothing has written to for this long goes when another download starts. */
        const val PARTIAL_MAX_AGE_MS = 7 * DAY_MS

        const val MIB = 1024L * 1024
        /** Room a download leaves free on top of the file. */
        const val FREE_SPACE_MARGIN_BYTES = 64 * MIB

        /** A file's download lock, and how many calls are downloading it or waiting to. */
        class FileDownloads {
            val lock = ReentrantLock()
            val calls = AtomicInteger()
        }

        /**
         * One entry per file name for the whole process (HybridLiteRTLM's
         * downloadModel has a store of its own): a download holds the lock from
         * the cache check to the rename, so no two write `<file>.tmp` at once.
         */
        private val fileDownloads = ConcurrentHashMap<String, FileDownloads>()

        fun downloadsOf(fileName: String): FileDownloads = fileDownloads.getOrPut(fileName) { FileDownloads() }
    }
}
