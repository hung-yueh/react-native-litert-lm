import { NitroModules } from "react-native-nitro-modules";
import type { ModelStore, ModelFile } from "./specs/LiteRTLM.nitro";
import { resolveModelFileName } from "./modelPath";

export type { ModelFile } from "./specs/LiteRTLM.nitro";

export interface ModelDownloadOptions {
  headers?: Record<string, string>;
  onProgress?: (progress: number) => void;
}

const nativeStore = NitroModules.createHybridObject<ModelStore>("ModelStore");

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
   * Delete a cached model file.
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
   * @param pathOrUrl Local filepath or HTTPS download URL
   * @param options Custom HTTP headers and progress callback
   * @returns Promise resolving to the absolute local path of the model
   */
  async resolveModel(pathOrUrl: string, options?: ModelDownloadOptions): Promise<string> {
    const cleanPath = pathOrUrl.startsWith("file://") ? pathOrUrl.substring(7) : pathOrUrl;
    if (cleanPath.startsWith("http://") || cleanPath.startsWith("https://")) {
      if (cleanPath.startsWith("http://")) {
        throw new Error(
          "Insecure HTTP URLs are not allowed for model downloads. " +
            "Use HTTPS instead: " +
            cleanPath.replace("http://", "https://")
        );
      }

      const urlWithoutQuery = cleanPath.split("?")[0];
      const fileName = urlWithoutQuery.split("/").pop();
      if (!fileName) {
        throw new Error(`Invalid model URL: ${cleanPath}`);
      }

      const headersJson = options?.headers ? JSON.stringify(options.headers) : "{}";
      
      return nativeStore.downloadFile(
        cleanPath,
        fileName,
        headersJson,
        (progress) => {
          options?.onProgress?.(progress);
        }
      );
    }

    return cleanPath;
  }
};
