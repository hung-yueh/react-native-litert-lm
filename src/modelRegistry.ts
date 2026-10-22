import { NitroModules } from "react-native-nitro-modules";
import type { ModelStore, ModelFile } from "./specs/LiteRTLM.nitro";
import { resolveModelFileName } from "./core/modelPath";
import { LiteRTLMError } from "./core/errors";
import { cleanNativeMessage } from "./core/nativeError";

export type { ModelFile } from "./specs/LiteRTLM.nitro";

export interface ModelDownloadOptions {
  headers?: Record<string, string>;
  onProgress?: (progress: number) => void;
  /**
   * Expected SHA-256 of the file: 64 hex characters, any case (for a file on
   * the Hugging Face Hub, its LFS `oid`). The download is hashed as it is
   * written and moved into the cache only when it matches; a mismatch deletes
   * it and rejects with `DOWNLOAD_FAILED`. A file already in the cache is
   * returned without hashing it again.
   */
  sha256?: string;
}

const SHA256_HEX = /^[0-9a-f]{64}$/i;

const nativeStore = NitroModules.createHybridObject<ModelStore>("ModelStore");

/** A download the native store is running, that a call for the same file can join. */
interface Download {
  /** The URL, sha256 and headers it runs with: a call joins only when all three are the same. */
  key: string;
  result: Promise<string>;
  /** The onProgress of every call that waits on it. */
  listeners: Array<(progress: number) => void>;
  /** The last progress the native store reported, sent once to a call that joins. */
  progress?: number;
}

/** The downloads running, by file name (the cache holds one file per name). */
const downloads = new Map<string, Download>();

/**
 * High-performance Model Registry for react-native-litert-lm.
 *
 * Coordinates URL sniffing, HTTPS enforcement, atomic downloading with progress,
 * and cache queries. Serves as the single source of truth for model storage status.
 */
export const ModelRegistry = {
  /**
   * Check if a model file is cached locally.
   * Accepts a filename, local path, or HTTPS URL.
   *
   * @param pathOrUrl Filename, local path, or download URL
   * @returns true if cached and has size > 0
   */
  isCached(pathOrUrl: string): boolean {
    return nativeStore.isCached(resolveModelFileName(pathOrUrl));
  },

  /**
   * Get the absolute local path of a cached model.
   * Accepts a filename, local path, or HTTPS URL.
   *
   * @param pathOrUrl Filename, local path, or download URL
   * @returns The absolute local path
   */
  getFilePath(pathOrUrl: string): string {
    return nativeStore.getFilePath(resolveModelFileName(pathOrUrl));
  },

  /**
   * List all locally cached model files.
   *
   * @returns Array of ModelFile descriptors containing path, size, and mod time
   */
  listCachedFiles(): ModelFile[] {
    return nativeStore.listCachedFiles();
  },

  /**
   * Delete a cached model file, and any partial download of it (a partial
   * file that a download is writing right now is left to that download).
   * Accepts a filename, local path, or HTTPS URL.
   *
   * @param pathOrUrl Filename, local path, or download URL to delete
   */
  deleteFile(pathOrUrl: string): void {
    nativeStore.deleteFile(resolveModelFileName(pathOrUrl));
  },

  /**
   * Size in bytes of a file at an absolute path (not restricted to the cache).
   * Returns -1 if the file does not exist.
   *
   * @param absolutePath Absolute filesystem path
   */
  getFileSizeBytes(absolutePath: string): number {
    return nativeStore.getFileSizeBytes(absolutePath);
  },

  /**
   * Resolve a model path or URL.
   *
   * Sniffs the protocol. If it is an HTTPS URL, downloads it to the native
   * cache directory (and throttles callbacks) if not already cached. If it is a
   * local path, returns it directly.
   *
   * A download that fails or is interrupted (the app killed, the network
   * lost) keeps what it wrote, and the next call for the same URL continues
   * from there with an HTTP Range request when the server sent a validator
   * (ETag or Last-Modified) and the file size.
   *
   * One download of a file runs at a time. A call while one is running joins
   * it when the URL, sha256 and headers are the same (its `onProgress` gets
   * the current progress at once, then the rest), and otherwise starts once
   * that download has ended.
   *
   * @param pathOrUrl Local filepath or HTTPS download URL
   * @param options Custom HTTP headers, progress callback and expected SHA-256
   * @returns Promise resolving to the absolute local path of the model
   */
  async resolveModel(pathOrUrl: string, options?: ModelDownloadOptions): Promise<string> {
    const cleanPath = pathOrUrl.startsWith("file://") ? pathOrUrl.substring(7) : pathOrUrl;
    if (cleanPath.startsWith("http://") || cleanPath.startsWith("https://")) {
      if (cleanPath.startsWith("http://")) {
        throw new LiteRTLMError(
          "INVALID_INPUT",
          "Insecure HTTP URLs are not allowed for model downloads. " +
            "Use HTTPS instead: " +
            cleanPath.replace("http://", "https://")
        );
      }

      const urlWithoutQuery = cleanPath.split("?")[0];
      const fileName = urlWithoutQuery.split("/").pop();
      if (!fileName) {
        throw new LiteRTLMError("INVALID_INPUT", `Invalid model URL: ${cleanPath}`);
      }

      const sha256 = options?.sha256;
      if (sha256 !== undefined && !SHA256_HEX.test(sha256)) {
        throw new LiteRTLMError(
          "INVALID_INPUT",
          `Invalid sha256 (expected 64 hex characters): ${sha256}`,
        );
      }

      const headersJson = options?.headers ? JSON.stringify(options.headers) : "{}";
      const expectedSha256 = sha256?.toLowerCase() ?? "";
      // The same headers in another order are the same request
      const headers = options?.headers ?? {};
      const key = JSON.stringify([
        cleanPath,
        expectedSha256,
        Object.keys(headers).sort().map((name) => [name, headers[name]]),
      ]);

      // One download of a file at a time (the native store keeps to that too):
      // a call for the same download joins the running one and gets its
      // progress; any other call waits for it to end
      for (let running = downloads.get(fileName); running; running = downloads.get(fileName)) {
        if (running.key === key) {
          const onProgress = options?.onProgress;
          if (onProgress) {
            running.listeners.push(onProgress);
            if (running.progress !== undefined) onProgress(running.progress);
          }
          return running.result;
        }
        await running.result.catch(() => undefined);
      }

      const download = {
        key,
        listeners: options?.onProgress ? [options.onProgress] : [],
      } as Download;
      // Registered before the native call: if that throws at once, the
      // finally below still finds it
      downloads.set(fileName, download);
      download.result = (async () => {
        try {
          return await nativeStore.downloadFile(cleanPath, fileName, headersJson, expectedSha256, (progress) => {
            download.progress = progress;
            for (const listener of download.listeners) listener(progress);
          });
        } catch (e) {
          const message = cleanNativeMessage(e instanceof Error ? e.message : String(e));
          throw new LiteRTLMError("DOWNLOAD_FAILED", `Failed to download ${cleanPath}: ${message}`, {
            cause: e,
          });
        } finally {
          // Before the callers resume, so one that calls again starts a new download
          if (downloads.get(fileName) === download) downloads.delete(fileName);
        }
      })();
      return download.result;
    }

    return cleanPath;
  }
};
