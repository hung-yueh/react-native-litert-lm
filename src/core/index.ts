/**
 * `react-native-litert-lm/core` — the react-native-free half of the library.
 *
 * Pure, deterministic logic with no native module and no React Native import:
 * memory estimation and forecasting, typed errors, the streaming-event parser,
 * and manifest resolution. Usable from tooling, Node scripts, tests, or any JS
 * runtime; the main entry point re-exports all of it.
 */
export {
  estimateMemory,
  estimateKvCacheBytesPerToken,
  forecastMemory,
  evaluateBudget,
  resolveMaxContextTokens,
  ESTIMATOR_CONSTANTS,
} from "./memoryEstimator";
export type {
  MemoryEstimate,
  MemoryEstimateInputs,
  MemoryForecast,
  MemoryForecastInputs,
  MemoryBudget,
  MemoryVerdict,
  BudgetLevel,
} from "./memoryEstimator";

export { LiteRTLMError, MemoryError, isMemoryError } from "./errors";
export type { LiteRTLMErrorCode } from "./errors";

export { createStreamEventParser, DEFAULT_CHANNELS } from "./streamEvents";
export type {
  StreamEvent,
  StreamEventType,
  StreamChannel,
  StreamEventParser,
} from "./streamEvents";

export {
  resolutionFor,
  resolveVariant,
  resolveFromManifest,
  mergeStreamChannels,
  parseManifest,
  fetchManifest,
  manifestFetchStatus,
  manifestPlatformFor,
  thinkingMarkers,
  declaredChannels,
} from "./manifestResolver";
export type {
  Manifest,
  ManifestFetchError,
  ManifestVariant,
  ManifestCapabilities,
  ManifestRecommendation,
  ManifestPlatform,
  ManifestResolution,
  ResolveFromManifestOptions,
  ResolveVariantOptions,
  VariantResolution,
  DeclaredChannel,
  ThinkingChannel,
} from "./manifestResolver";

export { extractFileName, resolveModelFileName } from "./modelPath";
