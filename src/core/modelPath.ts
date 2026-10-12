/**
 * Canonical path/URL helpers for model files.
 * Used by ModelRegistry, hooks, and any download/cache logic.
 */

/** Extract a filename from a URL or file path (query string stripped). */
export function extractFileName(pathOrUrl: string): string {
  const urlWithoutQuery = pathOrUrl.split("?")[0];
  const name = urlWithoutQuery.split("/").pop();
  return name || "model.bin";
}

/** Resolve a cache key from a filename, local path, or HTTPS URL. */
export function resolveModelFileName(pathOrUrl: string): string {
  return pathOrUrl.includes("/") ? extractFileName(pathOrUrl) : pathOrUrl;
}
