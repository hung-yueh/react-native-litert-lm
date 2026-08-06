# react-native-litert-lm

High-performance **on-device LLM inference** for React Native, powered by [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [Nitro Modules](https://github.com/mrousavy/nitro). Optimized for **Gemma 4** and other on-device models — with first-class **memory safety** so a 1–4 GB model can't silently OOM-kill your app.

## Highlights

- 🛡️ **Crash-free memory handling** — pre-flight estimation, live tracking, context forecasting, OS pressure warnings, budgets, and deterministic `unload()`. [See below.](#memory-handling)
- ⚡ **Binary multimodal input** — pass image/audio as native `ArrayBuffer`s, no base64 heap blow-up (buffers are staged to temp files for the engine's file-based API).
- 🧩 **Typed streaming events** — `token` / `toolCall` / `thinking` events (LiteRT-LM v0.14 streaming tool calls; channel streaming is iOS-only today, Android streams plain tokens).
- 🏎️ **GPU acceleration** — Metal (iOS), OpenCL delegate (Android/Pixel), with automatic CPU fallback.
- 🧠 **Speculative decoding & tool calling** — multi-token prediction and JSON-schema function calls.
- 📥 **Automatic model download** — HTTPS download with progress and local caching.

## Demo

> Gemma 4 E2B on a Samsung Galaxy S22 (Snapdragon 8 Gen 1, 4 GB RAM) — CPU backend, streaming inference.

<video src="https://github.com/user-attachments/assets/1da527ce-0432-4f8b-8899-474f81b2feea" width="300" controls></video>

## Installation

```bash
npm install react-native-litert-lm react-native-nitro-modules
```

**Expo** — add the plugin to `app.json`, then prebuild:

```json
{ "expo": { "plugins": ["react-native-litert-lm"], "android": { "minSdkVersion": 26 } } }
```

```bash
npx expo prebuild
npx expo run:ios      # or run:android
```

**Bare React Native** — `cd ios && pod install` (iOS) / `cd android && ./gradlew clean` (Android).

> Only ARM devices/simulators are supported. x86_64 Android emulators are not.

## Quick Start

The `useModel` hook manages the full lifecycle — download, load, inference, cleanup — and exposes memory state reactively.

```typescript
import { useModel, GEMMA_4_E2B_IT } from "react-native-litert-lm";

function Chat() {
  const { model, isReady, downloadProgress, error, memoryEstimate } = useModel(
    GEMMA_4_E2B_IT,
    { backend: "cpu", systemPrompt: "You are a helpful assistant.", enableMemoryTracking: true },
  );

  if (error) return <Text>{error}</Text>;
  if (!isReady) return <Text>Loading… {Math.round(downloadProgress * 100)}%</Text>;

  const ask = async () => console.log(await model.sendMessage("Hello!"));
  return <Button title="Generate" onPress={ask} />;
}
```

Prefer imperative control? Use `createLLM()`:

```typescript
import { createLLM } from "react-native-litert-lm";

const llm = createLLM();
await llm.loadModel("https://example.com/model.litertlm", { backend: "gpu" });
const reply = await llm.sendMessage("What is the capital of France?");
llm.unload(); // free the engine; llm stays reusable
```

## Memory Handling

On-device LLMs are the easiest way to get an app OOM-killed: a model that fits on one phone is killed by iOS Jetsam / Android LMK on another. This library turns **"will it fit?"** into a first-class, testable question across three layers — **predict → watch → react**.

### 1. Predict — pre-flight estimation

`loadModel()` estimates *weights + KV cache + overhead* against real OS headroom (jetsam-aware `os_proc_available_memory` on iOS, `ActivityManager.MemoryInfo` on Android) and **rejects with a typed `MemoryError` instead of letting the OS kill your app**:

```typescript
import { isMemoryError } from "react-native-litert-lm";

try {
  await llm.loadModel(modelUrl, { maxContextTokens: 8192 });
} catch (e) {
  if (isMemoryError(e)) {
    console.log(e.estimate.verdict);        // 'safe' | 'tight' | 'critical'
    console.log(e.estimate.recommendation); // how to make it fit
    await llm.loadModel(modelUrl, { maxContextTokens: 2048 }); // retry smaller
  }
}
```

Estimate *before downloading anything* to drive a model picker, and pass `{ forceLoad: true }` to skip the check:

```typescript
import { estimateMemory } from "react-native-litert-lm";

const estimate = estimateMemory({
  modelFileSizeBytes: 2.58e9,
  availableMemoryBytes: llm.getMemoryUsage().availableMemoryBytes,
  config: { backend: "gpu", maxContextTokens: 4096 },
});
if (estimate.verdict !== "safe") suggestSmallerModel();
```

### 2. Watch — live usage & forecasting

`getMemoryUsage()` reads real OS metrics (RSS, native heap, available memory) — no estimation. With `enableMemoryTracking`, snapshots are recorded into a native-backed ring buffer after every inference:

```typescript
const llm = createLLM({ enableMemoryTracking: true, maxMemorySnapshots: 256 });
// … after inference …
const { peakResidentBytes, currentResidentBytes } = llm.memoryTracker!.getSummary();
```

`getMemoryForecast()` combines the engine's KV-cache token count (exact on iOS; approximated on Android, where the SDK doesn't expose tokenizer counts) with the cost model to warn *before* the context window runs out:

```typescript
const forecast = llm.getMemoryForecast();
// { contextTokensUsed, remainingTokens, contextUsedFraction, kvCacheBytesUsed, nearingLimit }
if (forecast?.nearingLimit) summarizeHistoryOrWarn();
```

### 3. React — pressure warnings, budgets & teardown

Subscribe to real OS memory-pressure signals (`onTrimMemory` on Android, dispatch memory-pressure source on iOS) — the callback fires with level `'moderate'` or `'critical'` — or set app-defined budgets:

```typescript
llm.setMemoryWarningCallback((level, usage) => {
  if (level === "critical") llm.unload(); // free ~GBs deterministically
});

const llm = createLLM({
  enableMemoryTracking: true,
  memoryBudget: {
    warnAtFraction: 0.75,
    criticalAtFraction: 0.9,
    onBudgetExceeded: (level) => console.warn(`memory ${level}`),
  },
});
```

`unload()` releases the engine (freeing gigabytes) while keeping the instance reusable — don't wait for GC to reclaim a multi-GB model.

### Tuning knobs

Every knob's memory impact, documented. `maxContextTokens` is the biggest lever.

| Knob | Effect | Platform |
| --- | --- | --- |
| `maxContextTokens` | KV-cache size — the biggest lever | both |
| `activationDataType: 'f16'` | ~halves activation/KV memory | iOS |
| `prefillChunkSize` | caps peak prefill activation memory | iOS |
| `numThreads` | CPU memory-bandwidth pressure | iOS |
| `execute(…, { maxOutputTokens })` | per-message output cap | iOS |
| `loraPath` | one base model + small adapters | both |

### With the `useModel` hook

All of the above is reactive — `memoryEstimate`, `memoryForecast`, and `memoryWarning` are returned alongside `memorySummary`, updating automatically as you load and generate.

## Inference

### Streaming

```typescript
llm.sendMessageAsync("Tell me a story", (token, done) => {
  process.stdout.write(token);
  if (done) console.log("\n— done —");
});
```

### Typed streaming events (tool calls & thinking)

With `streamToolCalls: true` (LiteRT-LM v0.14+, **iOS only** — the Android SDK doesn't expose the tool-call channel yet), tool-call and reasoning tokens stream inside channel markers. `executeWithEvents()` parses them into typed events:

```typescript
await llm.loadModel(modelUrl, { tools, streamToolCalls: true });

await llm.executeWithEvents([{ type: "text", text: "Weather in Tokyo?" }], (event) => {
  switch (event.type) {
    case "token":    ui.appendText(event.text); break;
    case "toolCall": toolBuffer += event.text; break;
    case "thinking": ui.showReasoning(event.text); break;
  }
  if (event.done) runTool(JSON.parse(toolBuffer));
});
```

Markers default to `<tool_call>…</tool_call>` / `<thinking>…</thinking>` and are configurable via `createLLM({ streamChannels })`.

### Multimodal (binary buffers)

Pass native-backed `ArrayBuffer`s directly — no base64 encoding. (Internally the engine's API is file-based, so buffers are staged to temp files that are cleaned up after inference.)

```typescript
const buf = await (await fetch(Image.resolveAssetSource(require("./photo.jpg")).uri)).arrayBuffer();

const reply = await llm.sendMultimodalMessage([
  { type: "image", imageBuffer: buf },
  { type: "text", text: "Describe this image." },
]);
```

Path-based helpers also exist: `sendMessageWithImage(text, path)` and `sendMessageWithAudio(text, path)`. Multimodal requires a multimodal model (e.g. Gemma 4 E2B, Gemma 3n).

### Speculative decoding & tool calling

```typescript
useModel(GEMMA_4_E2B_IT, {
  enableSpeculativeDecoding: true, // multi-token prediction, if the model supports it
  tools: [{
    name: "get_current_weather",
    description: "Get the current weather for a location",
    parametersJson: JSON.stringify({
      type: "object",
      properties: { location: { type: "string" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } },
      required: ["location"],
    }),
  }],
});
```

### Structured output (JSON Schema / regex)

With `enableStructuredOutput: true`, any message can constrain its response via
constrained decoding (LLGuidance, LiteRT-LM 0.15+) — the engine **guarantees**
the output matches, on both platforms:

```typescript
const llm = createLLM();
await llm.loadModel(GEMMA_4_E2B_IT, {
  enableStructuredOutput: true,
  temperature: 0, // greedy sampling improves schema adherence
});

const json = await llm.execute(
  [{ type: 'text', text: 'Extract: "Ada Lovelace, born 1815, London"' }],
  undefined,
  {
    responseSchema: JSON.stringify({
      type: 'object',
      properties: { name: { type: 'string' }, birthYear: { type: 'number' }, city: { type: 'string' } },
      required: ['name', 'birthYear', 'city'],
    }),
  },
);
const person = JSON.parse(json); // guaranteed to parse

// Or a regex constraint:
await llm.execute([{ type: 'text', text: 'Pick a priority.' }], undefined, {
  responseRegex: 'P[0-3]',
});
```

`responseSchema` takes precedence when both are set. Using either without
`enableStructuredOutput` rejects with a clear error.

### Generation controls (thinking, anti-repetition)

Per message via `ExecuteOptions` (both platforms, LiteRT-LM 0.15+):

```typescript
await llm.execute(parts, onToken, {
  maxOutputTokens: 256,            // per-message output cap
  thinking: { enabled: true, tokenBudget: 512 }, // reasoning budget (Gemma 4)
  repetitionPenalty: 1.2,          // ≥ 1.0, HuggingFace-style multiplicative
  presencePenalty: 0.5,            // OpenAI-style subtractive
  frequencyPenalty: 0.3,
  noRepeatNgramSize: 3,            // ban exact 3-gram repeats
  suppressTokens: [128010],        // token IDs forced to -inf
});
```

Session-wide thinking defaults go in the load config: `loadModel(url, { thinking: { tokenBudget: 1024 } })`.
Thinking content still streams as typed `thinking` events through `executeWithEvents()`.

### Multiple conversations, one engine

`createConversation()` gives you independent chats sharing a single loaded
model — no double model load, ideal for "New Chat" UIs and agent side-chains:

```typescript
const support = llm.createConversation({ systemPrompt: 'You are a support agent.' });
const summarizer = llm.createConversation({ systemPrompt: 'You summarize tersely.' });

await support.execute([{ type: 'text', text: 'My app crashes on launch.' }]);
await summarizer.execute([{ type: 'text', text: 'Summarize: …' }]); // context switch
await support.execute([{ type: 'text', text: 'It happens on iOS 18.' }]); // remembers the crash report

support.getHistory();     // this conversation's transcript
await summarizer.release(); // drop a side-chain when done
```

How it works: the engine holds one native context at a time. Switching
conversations replays the target's transcript into a fresh context, so the
**next message after a switch pays a re-prefill cost** (roughly seconds on
long transcripts — engine-side prefix caching that would make this near-free
is [in progress upstream](https://github.com/google-ai-edge/LiteRT-LM/issues/2807)).
Frequent A/B ping-ponging with long histories will feel it; occasional
switching won't. Multimodal turns replay as `[Image]`/`[Audio]` text
placeholders. Once conversations are in use, inference calls are serialized
so a switch can never interrupt a generation; top-level `llm.execute()` acts
as its own "default" conversation.

## Supported Models

All exported URLs are **public — no auth required**. Pass any to `useModel()` / `loadModel()`.

| Constant | Model | Size | Min RAM | Source |
| --- | --- | --- | --- | --- |
| `GEMMA_4_E2B_IT` | Gemma 4 E2B (multimodal) | 2.58 GB | 4 GB+ | HuggingFace |
| `GEMMA_4_E4B_IT` | Gemma 4 E4B (higher quality) | 3.65 GB | 6 GB+ | HuggingFace |
| `GEMMA_3N_E2B_IT_INT4` | Gemma 3n E2B (int4, multimodal) | ~3.66 GB | 6 GB+ | litert.dev |

Other `.litertlm` models (Gemma 3 1B, Phi-4 Mini, Qwen 2.5 1.5B) download manually from [HuggingFace](https://huggingface.co/litert-community).

> **iOS:** models over ~2 GB need the [Extended Virtual Addressing entitlement](#ios-entitlements) — that includes all three models above. For a sub-2 GB option, download Gemma 3 1B manually from [HuggingFace](https://huggingface.co/litert-community).

## API Reference

**`createLLM(options?)`** → instance. Options: `enableMemoryTracking`, `maxMemorySnapshots` (default 256), `memoryBudget`, `streamChannels`.

**`loadModel(path, config?)`** → `Promise<void>`. `path` is a local path or HTTPS URL.

| Config | Default | Notes |
| --- | --- | --- |
| `backend` | `'cpu'` | `'cpu'` \| `'gpu'` \| `'npu'` (auto-fallback to CPU) |
| `systemPrompt` | — | System prompt |
| `temperature` / `topK` / `topP` | `0.7` / `40` / `0.95` | Sampling |
| `maxContextTokens` | `4096` | Total KV-cache budget (tokens) |
| `maxOutputTokens` | `1024` | Max tokens generated per response |
| `streamToolCalls` | `false` | Emit typed tool-call/thinking events (iOS only) |
| `enableStructuredOutput` | `false` | Initialize constrained decoding for per-message `responseSchema`/`responseRegex` |
| `thinking` | engine default | `{ enabled, tokenBudget }` reasoning controls (Gemma 4) |
| `forceLoad` | `false` | Skip the pre-flight memory check |
| memory tuning | — | `numThreads`, `prefillChunkSize`, `activationDataType`, `loraPath` — see [Tuning knobs](#tuning-knobs) |

**Inference:** `sendMessage(text)`, `sendMessageAsync(text, cb)`, `sendMessageWithImage/Audio(text, path)`, `sendMultimodalMessage(parts)`, `execute(parts, onToken?, options?)`, `executeWithEvents(parts, onEvent, options?)`.

**Conversations:** `createConversation(options?)` → handle with `execute`, `executeWithEvents`, `getHistory()`, `release()` — independent chats sharing one engine (see [Multiple conversations](#multiple-conversations-one-engine)).

**Memory:** `estimateMemory(inputs)`, `getMemoryUsage()`, `getMemoryForecast()`, `getContextTokenCount()`, `setMemoryWarningCallback(cb)` / `clearMemoryWarningCallback()`, `memoryTracker`.

**Lifecycle:** `getStats()`, `getHistory()`, `resetConversation()`, `unload()`, `close()`, `deleteModel(fileName)`.

**Utilities:** `checkBackendSupport(backend)`, `checkMultimodalSupport()`, `getRecommendedBackend()` — each returns a warning string (or `undefined`) so you can gate features before loading.

## Requirements & Platform Support

| | |
| --- | --- |
| React Native | 0.76+ |
| react-native-nitro-modules | 0.36.0+ |
| LiteRT-LM engine | 0.15.0 |
| Android | API 26+, arm64-v8a — CPU (all), GPU (OpenCL/Pixel only), NPU |
| iOS | 15.1+, arm64 — CPU, GPU (Metal; auto-fallback to CPU) |

> **Android GPU** requires OpenCL, unavailable on most Samsung/Qualcomm devices — check with `checkBackendSupport('gpu')`; the engine auto-falls back to CPU.

### iOS Entitlements

Models over ~2 GB need **Extended Virtual Addressing** or iOS caps virtual memory at ~2 GB and Jetsam kills the app. Add to your `.entitlements` (requires a paid Apple Developer account):

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

## Architecture

Nitro Modules (JSI) bridges TypeScript to a per-platform native engine:

```
React Native (TypeScript)
        │  Nitro JSI bindings (HybridLiteRTLMSpec)
   ┌────┴─────────────────────┐
   iOS (Swift Direct FFI)     Android (Kotlin)
   CLiteRTLM.xcframework       litertlm-android AAR
```

- **iOS** — native Swift calling the C FFI directly; inference, load, and unload are dispatched on a serial `dev.litert.engine` queue so generation never blocks the JSI thread (lightweight accessors like `getStats()` use a synchronous hop onto that queue). Raw pointers are freed deterministically in `deinit`/`close()`/`unload()` for zero leaks. RSS read via `mach_task_basic_info`.
- **Android** — stateless Kotlin conforming to `HybridLiteRTLMSpec`, with Proguard keep rules and optional `libOpenCL.so` loading for the GPU delegate.

## Testing

Multi-tier suite that runs on CI without a device:

- **JS/TS (Jest):** `npm test` — memory estimator (golden values), forecast/budget logic, stream-event parsing, ring-buffer tracker, hook & factory behavior, HTTPS guard.
- **Android (Robolectric):** `cd example/android && ./gradlew :react-native-litert-lm:testDebugUnitTest` — covers path-traversal/HTTPS guards, memory telemetry, and error paths. Requires a JDK 21+ test launcher (the litertlm-android AAR is built for Java 21); Gradle picks one up via toolchain auto-detection, or set `org.gradle.java.installations.paths`.
- **iOS (XCTest):** the test spec isn't part of Expo's generated Podfile — add `pod 'react-native-litert-lm', :path => '../..', :testspecs => ['Tests']` to `example/ios/Podfile`, run `pod install`, then `cd example/ios && xcodebuild test -workspace LLMTest.xcworkspace -scheme react-native-litert-lm-Unit-Tests -destination 'platform=iOS Simulator,name=iPhone 16,OS=18.6'`.

The Jest tier also includes a **config-parity contract test** (every `LLMConfig` key must be forwarded by `useModel` or explicitly excluded) and a **device-baseline guardrail** that validates the memory cost model against peak-RSS numbers recorded in [`scripts/memory-baseline.json`](scripts/memory-baseline.json).

**Real inference smoke test (opt-in):** the iOS suite includes an end-to-end generation test, skipped unless you point it at a local model: `LITERTLM_TEST_MODEL=$HOME/models/gemma3-1b.litertlm xcodebuild test …`.

On-device memory scenarios (OOM prevention, pressure simulation, peak-RSS regression budget) are documented in [`scripts/device-memory-scenarios.md`](scripts/device-memory-scenarios.md), and the scripted example-app integration pass lives in [`scripts/e2e-example-flow.md`](scripts/e2e-example-flow.md).

## Example App

`example/` is a full showcase app — Chat + Memory dashboard (pre-flight verdict, live RSS sparkline, context forecast, pressure warnings), typed streaming events, and the tuning knobs. Run it with `npm run build`, then `cd example && npm install && npx expo prebuild --clean && npx expo run:ios`.

## License

Code is [MIT](LICENSE).

### ⚠️ AI Model Disclaimer

This library is an execution engine; the models are **not** distributed with it and carry their own licenses — [Gemma](https://ai.google.dev/gemma/terms), [Llama 3](https://www.llama.com/llama3/license/), [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE), [Phi](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE). By downloading a model you accept its license and acceptable-use policy. The author takes no responsibility for model outputs.
