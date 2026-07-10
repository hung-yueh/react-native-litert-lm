import type { Backend, LLMConfig } from "./specs/LiteRTLM.nitro";

/**
 * Pure memory estimation math for crash-free model loading.
 *
 * Everything in this module is deterministic and side-effect free so the
 * formulas are unit-testable byte-for-byte. Native inputs (model file size,
 * available memory) are gathered by the caller (`createLLM().loadModel`).
 *
 * The estimates are deliberately conservative: an on-device LLM that fails to
 * load with a clear error is recoverable; an OS kill (iOS jetsam / Android LMK)
 * is not.
 */

/** How confident the estimator is that the model fits. */
export type MemoryVerdict = "safe" | "tight" | "critical";

/**
 * Result of a pre-flight memory estimate for loading a model.
 * All values in bytes.
 */
export interface MemoryEstimate {
  /** Model weights resident in memory (file size × backend load factor) */
  modelBytes: number;
  /** KV cache at full context (`maxContextTokens` × per-token cost) */
  kvCacheBytes: number;
  /** Runtime + activation overhead */
  overheadBytes: number;
  /** Sum of the above — expected peak while the model is loaded */
  totalEstimatedBytes: number;
  /** Available memory headroom reported by the OS at estimate time */
  availableBytes: number;
  /** availableBytes − totalEstimatedBytes (negative = expected to not fit) */
  headroomBytes: number;
  /** Fraction of available memory the load would consume (0–∞) */
  usageFraction: number;
  /**
   * 'safe'     — total ≤ SAFE_FRACTION × available
   * 'tight'    — fits, but with little slack; expect pressure warnings
   * 'critical' — total exceeds available memory; load will likely crash
   */
  verdict: MemoryVerdict;
  /** Human-readable explanation and suggested remediation */
  recommendation: string;
  /** KV-cache cost per token used for this estimate (bytes) */
  kvCacheBytesPerToken: number;
  /** Context budget (tokens) used for this estimate */
  maxContextTokens: number;
}

/** Inputs for {@link estimateMemory}. */
export interface MemoryEstimateInputs {
  /** Size of the .litertlm file on disk, in bytes */
  modelFileSizeBytes: number;
  /** Available memory reported by the OS (jetsam-aware on iOS), in bytes */
  availableMemoryBytes: number;
  /** The load config (backend, maxContextTokens, …) */
  config?: LLMConfig;
  /**
   * Override the per-token KV-cache cost in bytes (e.g. from calibration).
   * Defaults to a heuristic derived from the model file size.
   */
  kvCacheBytesPerTokenOverride?: number;
}

/**
 * Estimator constants. Exported so tests and calibration scripts can assert
 * against — and document — the exact formula.
 */
export const ESTIMATOR_CONSTANTS = {
  /** Weights load factor: CPU maps the file plus runtime copies */
  CPU_LOAD_FACTOR: 1.1,
  /** Weights load factor: GPU keeps a device-memory copy alongside staging */
  GPU_LOAD_FACTOR: 1.3,
  /** KV-cache heuristic: bytes per token per GiB of model file */
  KV_BYTES_PER_TOKEN_PER_GIB: 12 * 1024,
  /** KV-cache per-token floor (tiny models still pay layers × heads) */
  KV_BYTES_PER_TOKEN_MIN: 8 * 1024,
  /** KV-cache per-token ceiling (GQA keeps caches small on big models) */
  KV_BYTES_PER_TOKEN_MAX: 96 * 1024,
  /** Fixed runtime overhead floor (engine, tokenizer, buffers) */
  OVERHEAD_MIN_BYTES: 200 * 1024 * 1024,
  /** Activation overhead as a fraction of weights */
  OVERHEAD_WEIGHTS_FRACTION: 0.1,
  /** Verdict boundary: ≤ this fraction of available memory is 'safe' */
  SAFE_FRACTION: 0.8,
  /** Default context budget when the config does not specify one */
  DEFAULT_MAX_CONTEXT_TOKENS: 4096,
} as const;

const GIB = 1024 * 1024 * 1024;

/**
 * Heuristic per-token KV-cache cost derived from model file size.
 * Overridable per call for calibrated apps.
 */
export function estimateKvCacheBytesPerToken(
  modelFileSizeBytes: number,
): number {
  const c = ESTIMATOR_CONSTANTS;
  const raw = (modelFileSizeBytes / GIB) * c.KV_BYTES_PER_TOKEN_PER_GIB;
  return Math.min(c.KV_BYTES_PER_TOKEN_MAX, Math.max(c.KV_BYTES_PER_TOKEN_MIN, raw));
}

/** Resolve the context-token budget from a config (mirrors native defaults). */
export function resolveMaxContextTokens(config?: LLMConfig): number {
  if (config?.maxContextTokens != null) return config.maxContextTokens;
  // Legacy field maps to both context and output budgets when set alone
  if (config?.maxOutputTokens == null && config?.maxTokens != null) {
    return config.maxTokens;
  }
  return ESTIMATOR_CONSTANTS.DEFAULT_MAX_CONTEXT_TOKENS;
}

function loadFactorForBackend(backend?: Backend): number {
  return backend === "gpu" || backend === "npu"
    ? ESTIMATOR_CONSTANTS.GPU_LOAD_FACTOR
    : ESTIMATOR_CONSTANTS.CPU_LOAD_FACTOR;
}

/**
 * Pre-flight estimate: will loading this model fit in memory?
 *
 * `loadModel()` runs this automatically and rejects with a `MemoryError` on a
 * 'critical' verdict (unless `config.forceLoad` is set). Call it directly to
 * drive a model-picker UI before downloading anything.
 */
export function estimateMemory(inputs: MemoryEstimateInputs): MemoryEstimate {
  const c = ESTIMATOR_CONSTANTS;
  const { modelFileSizeBytes, availableMemoryBytes, config } = inputs;

  const modelBytes = modelFileSizeBytes * loadFactorForBackend(config?.backend);

  const kvCacheBytesPerToken =
    inputs.kvCacheBytesPerTokenOverride ??
    estimateKvCacheBytesPerToken(modelFileSizeBytes);
  const maxContextTokens = resolveMaxContextTokens(config);
  let kvCacheBytes = maxContextTokens * kvCacheBytesPerToken;
  // f16 activations roughly halve KV/activation memory vs f32 default
  if (config?.activationDataType === "f16") kvCacheBytes /= 2;

  const overheadBytes = Math.max(
    c.OVERHEAD_MIN_BYTES,
    modelBytes * c.OVERHEAD_WEIGHTS_FRACTION,
  );

  const totalEstimatedBytes = modelBytes + kvCacheBytes + overheadBytes;
  const headroomBytes = availableMemoryBytes - totalEstimatedBytes;
  const usageFraction =
    availableMemoryBytes > 0 ? totalEstimatedBytes / availableMemoryBytes : Infinity;

  let verdict: MemoryVerdict;
  let recommendation: string;
  if (usageFraction <= c.SAFE_FRACTION) {
    verdict = "safe";
    recommendation = "The model fits comfortably in available memory.";
  } else if (usageFraction <= 1) {
    verdict = "tight";
    recommendation =
      "The model fits, but with little slack — expect OS memory-pressure warnings. " +
      "Consider lowering maxContextTokens or setting activationDataType: 'f16'.";
  } else {
    verdict = "critical";
    const deficitMb = Math.ceil(-headroomBytes / (1024 * 1024));
    recommendation =
      `Estimated usage exceeds available memory by ~${deficitMb} MB — loading will likely ` +
      "crash the app. Use a smaller model or quantization, lower maxContextTokens, " +
      "or free memory before loading.";
  }

  return {
    modelBytes,
    kvCacheBytes,
    overheadBytes,
    totalEstimatedBytes,
    availableBytes: availableMemoryBytes,
    headroomBytes,
    usageFraction,
    verdict,
    recommendation,
    kvCacheBytesPerToken,
    maxContextTokens,
  };
}

/**
 * Forecast of context-window and KV-cache growth for the current conversation.
 * All byte values estimated; token counts are exact (from the engine).
 */
export interface MemoryForecast {
  /** Tokens currently in the conversation KV cache (exact, from engine) */
  contextTokensUsed: number;
  /** Context budget in tokens */
  maxContextTokens: number;
  /** contextTokensUsed / maxContextTokens (0–1) */
  contextUsedFraction: number;
  /** Tokens left before the context window is exhausted */
  remainingTokens: number;
  /** Estimated KV-cache bytes currently allocated */
  kvCacheBytesUsed: number;
  /** Estimated additional bytes if the context fills completely */
  kvCacheBytesRemaining: number;
  /** True when contextUsedFraction ≥ warnAtFraction */
  nearingLimit: boolean;
}

/** Inputs for {@link forecastMemory}. */
export interface MemoryForecastInputs {
  /** Current KV-cache token count from `getContextTokenCount()` (−1 if unknown) */
  contextTokensUsed: number;
  /** Context budget in tokens */
  maxContextTokens: number;
  /** Per-token KV-cache cost in bytes */
  kvCacheBytesPerToken: number;
  /** Fraction of the context at which `nearingLimit` flips (default 0.8) */
  warnAtFraction?: number;
}

/**
 * Combine the engine's exact token count with the KV-cost model to answer
 * "how much longer can this conversation run?".
 */
export function forecastMemory(inputs: MemoryForecastInputs): MemoryForecast {
  const warnAt = inputs.warnAtFraction ?? 0.8;
  const used = Math.max(0, inputs.contextTokensUsed);
  const max = Math.max(1, inputs.maxContextTokens);
  const remaining = Math.max(0, max - used);
  const fraction = used / max;

  return {
    contextTokensUsed: used,
    maxContextTokens: max,
    contextUsedFraction: fraction,
    remainingTokens: remaining,
    kvCacheBytesUsed: used * inputs.kvCacheBytesPerToken,
    kvCacheBytesRemaining: remaining * inputs.kvCacheBytesPerToken,
    nearingLimit: fraction >= warnAt,
  };
}

/**
 * Memory budget thresholds for live monitoring, as fractions of total
 * (available + used) memory consumed by the process.
 */
export interface MemoryBudget {
  /** Fire a 'moderate' budget warning at this usage fraction @default 0.75 */
  warnAtFraction?: number;
  /** Fire a 'critical' budget warning at this usage fraction @default 0.9 */
  criticalAtFraction?: number;
}

export type BudgetLevel = "ok" | "warn" | "critical";

/**
 * Evaluate a memory snapshot against a budget. Pure — the caller decides how
 * to react (the `createLLM` wrapper fires its `onBudgetExceeded` callback on
 * level transitions).
 */
export function evaluateBudget(
  residentBytes: number,
  availableMemoryBytes: number,
  budget?: MemoryBudget,
): BudgetLevel {
  const warnAt = budget?.warnAtFraction ?? 0.75;
  const criticalAt = budget?.criticalAtFraction ?? 0.9;
  const total = residentBytes + availableMemoryBytes;
  if (total <= 0) return "ok";
  const fraction = residentBytes / total;
  if (fraction >= criticalAt) return "critical";
  if (fraction >= warnAt) return "warn";
  return "ok";
}
