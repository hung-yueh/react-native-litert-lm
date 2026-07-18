package com.margelo.nitro.dev.litert.litertlm

import android.util.Log
import androidx.annotation.Keep
import com.facebook.proguard.annotations.DoNotStrip
import com.margelo.nitro.core.Promise
import dev.litert.litertlm.LiteRTLMInitProvider
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
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
                if (f.isFile && !f.name.endsWith(".tmp")) {
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

            try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.connectTimeout = 15000 // 15s
                connection.readTimeout = 0 // Infinite read timeout for large files
                connection.doInput = true

                // Apply headers
                for ((key, value) in headersMap) {
                    connection.setRequestProperty(key, value)
                }

                connection.connect()

                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    throw RuntimeException("Failed to download model: HTTP ${connection.responseCode}")
                }

                val contentLength = connection.contentLengthLong
                val input = connection.inputStream
                val output = FileOutputStream(tempFile)

                val buffer = ByteArray(8 * 1024)
                var bytesRead: Int
                var totalBytesRead = 0L
                var lastProgressUpdate = 0L

                while (input.read(buffer).also { bytesRead = it } != -1) {
                    output.write(buffer, 0, bytesRead)
                    totalBytesRead += bytesRead

                    if (contentLength > 0) {
                        val currentTime = System.currentTimeMillis()
                        // Update progress roughly every 100ms to prevent bridge flooding
                        if (currentTime - lastProgressUpdate > 100) {
                            val progress = totalBytesRead.toDouble() / contentLength.toDouble()
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

                // Atomic rename
                if (tempFile.renameTo(modelFile)) {
                    Log.i(tag, "Download complete and verified at: ${modelFile.absolutePath}")
                    onProgress(1.0)
                    modelFile.absolutePath
                } else {
                    throw RuntimeException("Failed to rename temporary file to model file")
                }
            } catch (e: Exception) {
                Log.e(tag, "Download failed", e)
                if (tempFile.exists()) {
                    tempFile.delete()
                }
                throw RuntimeException("Download failed: ${e.message}", e)
            }
        }
    }
}
