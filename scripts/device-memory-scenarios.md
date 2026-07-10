# On-Device Memory Scenarios

The unit suite proves the estimation math; these scenarios prove the
**crash-free behavior on real hardware**. Run them before each release (and
after every LiteRT-LM engine bump) on at least one iOS device/simulator and
one physical Android device, using the example app.

Record results in `scripts/memory-baseline.json`.

## Setup

```bash
npm run build && cd example && npm install && npx expo prebuild --clean
npx expo run:android   # or: npx expo run:ios
```

Use Gemma 3n E2B INT4 (~1.3 GB) unless stated otherwise.

## Scenario 1 — OOM prevention (pre-flight rejection)

1. In the example app, attempt to load the model with an absurd context:
   `loadModel(url, { maxContextTokens: 262144 })`.
2. **Pass:** the promise rejects with `MemoryError` (`code:
   "MEMORY_ESTIMATE_EXCEEDED"`), the app stays alive, and a subsequent
   normal `loadModel(url)` succeeds.
3. **Fail:** the OS kills the app (iOS jetsam / Android LMK), or the load
   hangs.

## Scenario 2 — OS pressure warning fires and generation survives

1. Load the model, start a long streamed generation.
2. Mid-generation, simulate memory pressure:
   - iOS simulator: Device menu → *Simulate Memory Warning*
     (or `xcrun simctl spawn booted notifyutil -p org.apple.memory-warning`)
   - Android: `adb shell am send-trim-memory <package> RUNNING_CRITICAL`
3. **Pass:** the `onMemoryWarning` / `useModel().memoryWarning` surface
   reports `moderate`/`critical`; the stream either completes or ends with a
   clean `done` callback; no crash.

## Scenario 3 — Peak-memory regression budget

1. Run the scripted flow: load model → 20-turn streamed conversation
   (alternating short/long prompts) → 1 multimodal turn (image) → `unload()`.
2. Read `memoryTracker.getSummary().peakResidentBytes`.
3. **Pass:** peak RSS ≤ baseline in `memory-baseline.json` × 1.10.
4. Update the baseline only deliberately (engine upgrades that legitimately
   change the footprint), never to quiet a failure you don't understand.

## Scenario 4 — Estimator calibration

1. After Scenario 3, compare `estimateMemory().totalEstimatedBytes` against
   the observed `peakResidentBytes`.
2. **Pass:** the estimate is within −20%/+50% of observed peak (estimates
   should err high). If the estimator is under-predicting, recalibrate
   `ESTIMATOR_CONSTANTS` (see `src/memoryEstimator.ts`) and fix the golden
   tests to match.

## Scenario 5 — Load/unload soak (leak check)

1. Loop 10×: `loadModel(path)` → one short generation → `unload()`.
2. Sample `getMemoryUsage().residentBytes` after each `unload()`.
3. **Pass:** RSS after the 10th unload is within 15% of RSS after the 1st —
   no monotonic growth.

## Scenario 6 — Context forecast accuracy

1. Load with `maxContextTokens: 1024`; run turns until
   `getMemoryForecast().nearingLimit` becomes true.
2. **Pass:** `nearingLimit` flips before the engine starts truncating or
   erroring, and `contextTokensUsed` matches `countTokens` intuition
   (monotonically increasing per turn).
