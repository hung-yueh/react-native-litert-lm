/**
 * Guardrail tying the JS memory cost model to real device measurements.
 *
 * scripts/memory-baseline.json records peak-RSS baselines captured by the
 * on-device scenarios in scripts/device-memory-scenarios.md. Once populated,
 * this test fails when estimateMemory() drifts so far from a recorded device
 * peak that the pre-flight verdict would be misleading (estimate under 60%
 * of, or over 250% of, the observed peak).
 *
 * Baseline entry shape (per device key, e.g. "pixel-8" / "iphone-15-pro"):
 *   {
 *     "peakResidentBytes": 3210000000,
 *     "modelFileSizeBytes": 3655827456,
 *     "availableMemoryBytes": 6000000000,
 *     "config": { "backend": "cpu", "maxContextTokens": 4096 }
 *   }
 */
import * as fs from "fs";
import * as path from "path";
import { estimateMemory } from "../memoryEstimator";

const BASELINE_PATH = path.join(
  __dirname,
  "..",
  "..",
  "scripts",
  "memory-baseline.json",
);

interface BaselineEntry {
  peakResidentBytes: number;
  modelFileSizeBytes: number;
  availableMemoryBytes: number;
  config?: { backend?: "cpu" | "gpu" | "npu"; maxContextTokens?: number };
}

describe("memory estimator vs device baselines", () => {
  const raw = JSON.parse(fs.readFileSync(BASELINE_PATH, "utf8"));

  it("baseline file has the expected shape", () => {
    expect(raw).toHaveProperty("baselines");
    expect(typeof raw.baselines).toBe("object");
  });

  const entries = Object.entries(raw.baselines as Record<string, BaselineEntry>);

  if (entries.length === 0) {
    it.skip("no device baselines recorded yet (populate via scripts/device-memory-scenarios.md)", () => {});
    return;
  }

  it.each(entries)(
    "estimate for %s is within tolerance of the recorded device peak",
    (_device, baseline) => {
      expect(baseline.peakResidentBytes).toBeGreaterThan(0);
      expect(baseline.modelFileSizeBytes).toBeGreaterThan(0);

      const estimate = estimateMemory({
        modelFileSizeBytes: baseline.modelFileSizeBytes,
        availableMemoryBytes: baseline.availableMemoryBytes,
        config: baseline.config,
      });

      const ratio = estimate.totalEstimatedBytes / baseline.peakResidentBytes;
      // Underestimating badly means the pre-flight check green-lights loads
      // that OOM; wild overestimating blocks loads that are actually fine.
      expect(ratio).toBeGreaterThanOrEqual(0.6);
      expect(ratio).toBeLessThanOrEqual(2.5);
    },
  );
});
