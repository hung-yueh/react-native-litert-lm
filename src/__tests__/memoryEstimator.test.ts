/**
 * Golden tests for the memory estimator — the formulas guard crash-free
 * loading, so regressions must be caught byte-for-byte.
 */
import {
  estimateMemory,
  estimateKvCacheBytesPerToken,
  forecastMemory,
  resolveMaxContextTokens,
  evaluateBudget,
  ESTIMATOR_CONSTANTS,
} from "../memoryEstimator";

const GIB = 1024 * 1024 * 1024;
const MIB = 1024 * 1024;

describe("estimateKvCacheBytesPerToken", () => {
  it("scales linearly with model file size (12 KiB per GiB)", () => {
    expect(estimateKvCacheBytesPerToken(1 * GIB)).toBe(12 * 1024);
    expect(estimateKvCacheBytesPerToken(2 * GIB)).toBe(24 * 1024);
  });

  it("clamps to the floor for tiny models", () => {
    expect(estimateKvCacheBytesPerToken(0.1 * GIB)).toBe(
      ESTIMATOR_CONSTANTS.KV_BYTES_PER_TOKEN_MIN,
    );
  });

  it("clamps to the ceiling for huge models", () => {
    expect(estimateKvCacheBytesPerToken(10 * GIB)).toBe(
      ESTIMATOR_CONSTANTS.KV_BYTES_PER_TOKEN_MAX,
    );
  });
});

describe("resolveMaxContextTokens", () => {
  it("defaults to 4096 with no config", () => {
    expect(resolveMaxContextTokens(undefined)).toBe(4096);
    expect(resolveMaxContextTokens({})).toBe(4096);
  });

  it("prefers explicit maxContextTokens", () => {
    expect(resolveMaxContextTokens({ maxContextTokens: 8192, maxTokens: 100 })).toBe(8192);
  });

  it("maps legacy maxTokens when set alone (mirrors native behavior)", () => {
    expect(resolveMaxContextTokens({ maxTokens: 2048 })).toBe(2048);
  });

  it("ignores legacy maxTokens when maxOutputTokens is also set", () => {
    expect(resolveMaxContextTokens({ maxTokens: 2048, maxOutputTokens: 512 })).toBe(4096);
  });
});

describe("estimateMemory — golden values", () => {
  it("computes exact bytes for a 1 GiB model on CPU with defaults", () => {
    const e = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
    });
    // weights: 1 GiB × 1.1
    expect(e.modelBytes).toBe(1 * GIB * 1.1);
    // KV: 4096 tokens × 12 KiB
    expect(e.kvCacheBytesPerToken).toBe(12 * 1024);
    expect(e.kvCacheBytes).toBe(4096 * 12 * 1024);
    // overhead: max(200 MiB, 10% of weights) = 200 MiB
    expect(e.overheadBytes).toBe(200 * MIB);
    expect(e.totalEstimatedBytes).toBe(
      1 * GIB * 1.1 + 4096 * 12 * 1024 + 200 * MIB,
    );
    expect(e.headroomBytes).toBe(4 * GIB - e.totalEstimatedBytes);
    expect(e.verdict).toBe("safe");
  });

  it("applies the GPU load factor", () => {
    const e = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
      config: { backend: "gpu" },
    });
    expect(e.modelBytes).toBe(1 * GIB * 1.3);
  });

  it("uses the 10%-of-weights overhead once it exceeds the floor", () => {
    const e = estimateMemory({
      modelFileSizeBytes: 4 * GIB,
      availableMemoryBytes: 16 * GIB,
    });
    expect(e.overheadBytes).toBe(4 * GIB * 1.1 * 0.1);
  });

  it("halves KV-cache memory for f16 activations", () => {
    const base = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
    });
    const f16 = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
      config: { activationDataType: "f16" },
    });
    expect(f16.kvCacheBytes).toBe(base.kvCacheBytes / 2);
  });

  it("honors the per-token override", () => {
    const e = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
      kvCacheBytesPerTokenOverride: 1000,
    });
    expect(e.kvCacheBytes).toBe(4096 * 1000);
  });

  it("scales KV cache with maxContextTokens — the biggest tuning lever", () => {
    const small = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
      config: { maxContextTokens: 1024 },
    });
    const large = estimateMemory({
      modelFileSizeBytes: 1 * GIB,
      availableMemoryBytes: 4 * GIB,
      config: { maxContextTokens: 8192 },
    });
    expect(large.kvCacheBytes).toBe(small.kvCacheBytes * 8);
  });
});

describe("estimateMemory — verdict boundaries", () => {
  // Fix total usage, vary available memory around the boundaries
  const inputs = { modelFileSizeBytes: 1 * GIB };
  const total = estimateMemory({
    ...inputs,
    availableMemoryBytes: 100 * GIB,
  }).totalEstimatedBytes;

  it("is 'safe' at exactly 80% usage", () => {
    const e = estimateMemory({
      ...inputs,
      availableMemoryBytes: total / ESTIMATOR_CONSTANTS.SAFE_FRACTION,
    });
    expect(e.verdict).toBe("safe");
  });

  it("is 'tight' just past 80% usage but still fitting", () => {
    const e = estimateMemory({
      ...inputs,
      availableMemoryBytes: total / ESTIMATOR_CONSTANTS.SAFE_FRACTION - 1024,
    });
    expect(e.verdict).toBe("tight");
    expect(e.recommendation).toContain("maxContextTokens");
  });

  it("is 'critical' when the estimate exceeds available memory", () => {
    const e = estimateMemory({
      ...inputs,
      availableMemoryBytes: total - 1024,
    });
    expect(e.verdict).toBe("critical");
    expect(e.headroomBytes).toBeLessThan(0);
    expect(e.recommendation).toContain("crash");
  });

  it("is 'critical' with zero available memory (no division blowup)", () => {
    const e = estimateMemory({ ...inputs, availableMemoryBytes: 0 });
    expect(e.verdict).toBe("critical");
    expect(e.usageFraction).toBe(Infinity);
  });
});

describe("forecastMemory", () => {
  it("computes exact usage fractions and byte projections", () => {
    const f = forecastMemory({
      contextTokensUsed: 1024,
      maxContextTokens: 4096,
      kvCacheBytesPerToken: 12 * 1024,
    });
    expect(f.contextUsedFraction).toBe(0.25);
    expect(f.remainingTokens).toBe(3072);
    expect(f.kvCacheBytesUsed).toBe(1024 * 12 * 1024);
    expect(f.kvCacheBytesRemaining).toBe(3072 * 12 * 1024);
    expect(f.nearingLimit).toBe(false);
  });

  it("flips nearingLimit at the warn fraction (default 0.8)", () => {
    const below = forecastMemory({
      contextTokensUsed: 3276,
      maxContextTokens: 4096,
      kvCacheBytesPerToken: 1,
    });
    const above = forecastMemory({
      contextTokensUsed: 3277,
      maxContextTokens: 4096,
      kvCacheBytesPerToken: 1,
    });
    expect(below.nearingLimit).toBe(false);
    expect(above.nearingLimit).toBe(true);
  });

  it("clamps negative token counts (engine returned -1)", () => {
    const f = forecastMemory({
      contextTokensUsed: -1,
      maxContextTokens: 4096,
      kvCacheBytesPerToken: 1,
    });
    expect(f.contextTokensUsed).toBe(0);
    expect(f.remainingTokens).toBe(4096);
  });

  it("honors a custom warn fraction", () => {
    const f = forecastMemory({
      contextTokensUsed: 2048,
      maxContextTokens: 4096,
      kvCacheBytesPerToken: 1,
      warnAtFraction: 0.5,
    });
    expect(f.nearingLimit).toBe(true);
  });
});

describe("evaluateBudget", () => {
  it("reports 'ok' below the warn threshold", () => {
    // 50% usage
    expect(evaluateBudget(1000, 1000)).toBe("ok");
  });

  it("reports 'warn' at the default 75% threshold", () => {
    expect(evaluateBudget(75, 25)).toBe("warn");
  });

  it("reports 'critical' at the default 90% threshold", () => {
    expect(evaluateBudget(90, 10)).toBe("critical");
  });

  it("honors custom thresholds", () => {
    expect(evaluateBudget(60, 40, { warnAtFraction: 0.5 })).toBe("warn");
    expect(
      evaluateBudget(60, 40, { warnAtFraction: 0.5, criticalAtFraction: 0.6 }),
    ).toBe("critical");
  });

  it("returns 'ok' when totals are degenerate", () => {
    expect(evaluateBudget(0, 0)).toBe("ok");
  });
});
