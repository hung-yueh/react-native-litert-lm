/**
 * Deprecation warnings: one console.warn per API per process, so a
 * deprecated call in a render loop doesn't flood the console. Everything
 * deprecated keeps working until 1.0.
 */
const warned = new Set<string>();
let enabled = true;

/** Turn deprecation warnings off (or back on) for the whole process. */
export function setDeprecationWarningsEnabled(on: boolean): void {
  enabled = on;
}

/** Warn once that `api` is deprecated in favour of `instead`. */
export function warnDeprecated(api: string, instead: string): void {
  if (!enabled || warned.has(api)) return;
  warned.add(api);
  console.warn(
    `[react-native-litert-lm] ${api} is deprecated and will be removed in 1.0. Use ${instead} instead.`,
  );
}

/** Test helper: forget which APIs already warned. */
export function resetDeprecationWarnings(): void {
  warned.clear();
}
