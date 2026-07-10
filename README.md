# react-native-litert-lm

High-performance on-device LLM inference for React Native, powered by [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [Nitro Modules](https://github.com/mrousavy/nitro). Optimized for **Gemma 4** and other on-device language models.

## Features

- 🚀 **Native Swift Bridge (iOS)** — Bypasses Swift actor deadlocks (User Rule #1) via direct C FFI dispatched on a serial `dev.litert.engine` background queue.
- 🤖 **Stateless Kotlin Bridge (Android)** — Fully conforms to `HybridLiteRTLMSpec` using direct JSI memory access.
- ⚡ **Zero-Copy Multimodal API** — Native-owned `ArrayBuffer` mapping straight to FFI inputs for image/audio data without copy overhead (complying with User Rule #2).
- 🧠 **Speculative Decoding** — Active multi-token prediction support with pre-flight model capability validation.
- 🛠️ **Function / Tool Calling** — Native JSON-encoded schema specification support for structured outputs.
- 🏎️ **GPU Acceleration** — Metal (iOS), OpenCL GPU delegate (Android, Pixel devices).
- 🔄 **Streaming Support** — Non-blocking token-by-token callbacks.
- 📊 **Real Memory Tracking** — OS-level memory metrics (RSS, native heap, available memory) via native APIs (complying with User Rule #3).
- 🛡️ **Crash-Free Memory Controls** — Pre-flight memory estimation (`loadModel` rejects with a typed `MemoryError` instead of letting the OS kill your app), context/KV-cache forecasting, OS memory-pressure warnings, memory budgets, and deterministic `unload()`.
- 🧩 **Typed Streaming Events** — `executeWithEvents()` splits the stream into `token` / `toolCall` / `thinking` events (LiteRT-LM v0.14 streaming tool calls).
- 🎛️ **Memory Tuning Knobs** — `maxContextTokens`, `numThreads`, `prefillChunkSize`, `activationDataType`, per-message `maxOutputTokens`, and LoRA adapters (`loraPath`).
- 📥 **Automatic Model Download** — Downloads models from URL with progress tracking and local caching.

## Demo

> Gemma 4 E2B running on-device on a Samsung Galaxy S22 (Snapdragon 8 Gen 1, 4 GB RAM) — CPU backend, streaming inference.

<video src="https://github.com/user-attachments/assets/1da527ce-0432-4f8b-8899-474f81b2feea" width="300" controls></video>

## Installation

```bash
npm install react-native-litert-lm react-native-nitro-modules
```

### Expo

Add to your `app.json`:

```json
{
  "expo": {
    "plugins": ["react-native-litert-lm"],
    "android": {
      "minSdkVersion": 26
    }
  }
}
```

Then create a development build:

```bash
npx expo prebuild
npx expo run:android  # Android
npx expo run:ios      # iOS
```

> **Note**: Only ARM devices/simulators are supported. x86_64 Android emulators are not supported.

### Bare React Native

```bash
# Android
cd android && ./gradlew clean

# iOS
cd ios && pod install
```

## Example App

The `example/` directory contains a fully functional test app with a dark-themed diagnostic UI that demonstrates:

- Model downloading with progress tracking
- Text inference (blocking and streaming)
- Multi-turn conversation with context retention
- Performance benchmarking (tokens/sec, latency)
- Real-time memory tracking
- Speculative decoding & tool calling settings toggles
- Zero-copy multimodal inference loading images/audio directly into ArrayBuffers

### Running the Example

1. **Build the library** (compiles TypeScript to `lib/`):

   ```bash
   npm run build
   ```

2. **Install example dependencies:**

   ```bash
   cd example
   npm install
   ```

3. **Create a development build and run:**

   ```bash
   npx expo prebuild --clean
   npx expo run:android  # Android
   npx expo run:ios      # iOS (pre-linked with CLiteRTLM.xcframework)
   ```

> **Note:** If you change native code (Swift/Kotlin), you must run `npx expo prebuild --clean` again before rebuilding.

## Model Management

LiteRT-LM models (like Gemma 4) are large files (1–4 GB) and cannot be bundled into your app binary. They are downloaded at runtime.

### Automatic Downloading

Pass an HTTPS URL to `useModel()` or `loadModel()` — the library handles the rest:

- **Progress tracking** — real-time download percentage via callbacks
- **Local caching** — downloaded models are cached and reused across app launches
  - **Android**: `files/models/` (app-private)
  - **iOS**: `Library/Caches/litert_models/` (survives app relaunch; reclaimable by iOS under storage pressure)
- **HTTPS enforcement** — only secure URLs are accepted

### Manual Downloading

If you need custom control over downloads (e.g., authentication headers for private model hosting, resumable downloads, or custom caching), use your preferred HTTP client and pass the local file path:

```typescript
import { fetch } from "expo/fetch";
import { File, Paths } from "expo-file-system";
import { useModel } from "react-native-litert-lm";

const MODEL_URL = "https://example.com/private-model.litertlm";

// Download with custom headers using expo/fetch
const response = await fetch(MODEL_URL, {
  headers: { Authorization: `Bearer ${token}` },
});
const modelFile = new File(Paths.cache, "my-model.litertlm");
modelFile.write(await response.bytes());

// Pass the local path — no download occurs
const { model, isReady } = useModel(modelFile.uri, { backend: "cpu" });
```

## Usage

### React Hook (Recommended)

The `useModel` hook manages the full model lifecycle: downloading, loading, inference, and cleanup.

```typescript
import { useModel, GEMMA_4_E2B_IT } from "react-native-litert-lm";
import { Platform } from "react-native";

function App() {
  const {
    model,
    isReady,
    downloadProgress,
    error,
    load,          // Manually trigger load
    deleteModel,   // Delete cached model file
    memorySummary, // Auto-updated memory stats (if tracking enabled)
  } = useModel(GEMMA_4_E2B_IT, {
    backend: 'cpu',
    autoLoad: true, // Default: true. Set false to load manually via load().
    systemPrompt: "You are a helpful assistant.",
    enableMemoryTracking: true,
  });

  if (!isReady) {
    return <Text>Loading... {Math.round(downloadProgress * 100)}%</Text>;
  }

  const generate = async () => {
    const response = await model.sendMessage("Hello!");
    console.log(response);
  };

  return <Button title="Generate" onPress={generate} />;
}
```

### Manual Usage

```typescript
import { createLLM } from "react-native-litert-lm";

const llm = createLLM();

// Load a model from URL (auto-downloads) or local path
await llm.loadModel("https://example.com/model.litertlm", {
  backend: "gpu",
  systemPrompt: "You are a helpful assistant.",
});

// Generate a response
const response = await llm.sendMessage("What is the capital of France?");
console.log(response);

// Clean up
llm.close();
```

### Streaming Generation

```typescript
llm.sendMessageAsync("Tell me a story", (token, done) => {
  process.stdout.write(token);
  if (done) console.log("\n--- Done ---");
});
```

### Multimodal (Image / Audio) & Zero-Copy Buffers

Multimodal features are fully supported via standard file paths or high-performance zero-copy `ArrayBuffer` objects:

#### 1. Zero-Copy Multimodal Messages (Recommended)
This API uses Nitro Modules' native-backed `ArrayBuffer` directly mapped to native memory buffers, avoiding any base64 heap copying overhead (User Rule #2):

```typescript
import { checkMultimodalSupport } from "react-native-litert-lm";

const warning = checkMultimodalSupport();
if (warning) {
  console.warn(warning); // Experimental or unsupported on current platform (e.g. iOS simulator)
} else {
  // Read local assets or files straight into ArrayBuffers using fetch
  const response = await fetch(Image.resolveAssetSource(require("./test.jpeg")).uri);
  const imageBuffer = await response.arrayBuffer();

  const reply = await llm.sendMultimodalMessage([
    { type: "image", imageBuffer },
    { type: "text", text: "Describe what is in this image." }
  ]);
  console.log(reply);
}
```

#### 2. Path-Based Multimodal Messages
```typescript
// Image input
const response = await llm.sendMessageWithImage(
  "What's in this image?",
  "/path/to/image.jpg",
);

// Audio input
const transcription = await llm.sendMessageWithAudio(
  "Transcribe this audio",
  "/path/to/audio.wav",
);
```

### Speculative Decoding & Tools

#### 1. Speculative Decoding (MTP)
Enable speculative decoding in `LLMConfig` to accelerate inference using multi-token prediction when supported by your model:

```typescript
const { model } = useModel(GEMMA_4_E2B_IT, {
  enableSpeculativeDecoding: true,
});
```

#### 2. Function / Tool Calling
Inject tools as an array of definitions, specifying parameter validation using standard JSON schema format:

```typescript
const { model } = useModel(GEMMA_4_E2B_IT, {
  tools: [
    {
      name: "get_current_weather",
      description: "Get the current weather for a location",
      parametersJson: JSON.stringify({
        type: "object",
        properties: {
          location: { type: "string", description: "The city and state, e.g. San Francisco, CA" },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] }
        },
        required: ["location"]
      })
    }
  ]
});
```

### Performance Stats

```typescript
const stats = llm.getStats();
console.log(`Generated ${stats.completionTokens} tokens`);
console.log(`Speed: ${stats.tokensPerSecond.toFixed(1)} tokens/sec`);
console.log(`Time to first token: ${stats.timeToFirstToken.toFixed(0)} ms`);
```

> **Note**: Stats are available for both sync (`sendMessage`) and streaming (`sendMessageAsync`) on both platforms. iOS uses real benchmark data from the C API; Android uses heuristic token counts with precise timing.

### Memory Tracking

The library provides real OS-level memory data — no estimation. It reads directly from `mach_task_basic_info` (iOS) and `Debug.getNativeHeapAllocatedSize()` + `/proc/self/status` (Android).

#### Direct Memory Query

```typescript
const usage = llm.getMemoryUsage();
console.log(
  `Native heap: ${(usage.nativeHeapBytes / 1024 / 1024).toFixed(1)} MB`,
);
console.log(`RSS: ${(usage.residentBytes / 1024 / 1024).toFixed(1)} MB`);
console.log(
  `Available: ${(usage.availableMemoryBytes / 1024 / 1024).toFixed(1)} MB`,
);
console.log(`Low memory: ${usage.isLowMemory}`);
```

#### Automatic Tracking with Native Buffers

Enable memory tracking to automatically record snapshots in a native-backed `ArrayBuffer` after every inference call:

```typescript
const llm = createLLM({
  enableMemoryTracking: true,
  maxMemorySnapshots: 256,
});

await llm.loadModel("/path/to/model.litertlm", { backend: "cpu" });
await llm.sendMessage("Hello!");

const summary = llm.memoryTracker!.getSummary();
console.log(
  `Peak RSS: ${(summary.peakResidentBytes / 1024 / 1024).toFixed(1)} MB`,
);
console.log(
  `RSS Delta: ${(summary.residentDeltaBytes / 1024 / 1024).toFixed(1)} MB`,
);
```

#### Using `useModel` with Memory Tracking

```typescript
const { model, isReady, memorySummary } = useModel(modelUrl, {
  enableMemoryTracking: true,
  maxMemorySnapshots: 100,
});

// memorySummary auto-updates after each inference call
if (memorySummary) {
  console.log(`Current RSS: ${memorySummary.currentResidentBytes}`);
  console.log(`Peak RSS: ${memorySummary.peakResidentBytes}`);
}
```

#### Standalone Memory Tracker

```typescript
import {
  createMemoryTracker,
  createNativeBuffer,
} from "react-native-litert-lm";

const tracker = createMemoryTracker(100);

tracker.record({
  timestamp: Date.now(),
  nativeHeapBytes: 50_000_000,
  residentBytes: 200_000_000,
  availableMemoryBytes: 4_000_000_000,
});

// Access the underlying native buffer (zero-copy transfer to native code)
const buffer = tracker.getNativeBuffer();
```

### Memory Controls (Crash-Free Loading)

On-device LLMs are the easiest way to get your app OOM-killed. The memory
controls make "will this fit?" a first-class, testable question.

#### Pre-flight estimation

`loadModel()` automatically estimates weights + KV cache + overhead against
the OS-reported headroom (jetsam-aware `os_proc_available_memory` on iOS,
`ActivityManager.MemoryInfo` on Android) and **rejects with a typed
`MemoryError` instead of crashing**:

```typescript
import { isMemoryError } from "react-native-litert-lm";

try {
  await llm.loadModel(modelUrl, { maxContextTokens: 8192 });
} catch (e) {
  if (isMemoryError(e)) {
    console.log(e.estimate.verdict); // 'critical'
    console.log(e.estimate.recommendation); // what to do about it
    // e.g. retry with a smaller context:
    await llm.loadModel(modelUrl, { maxContextTokens: 2048 });
  }
}

// Or check before downloading anything:
import { estimateMemory } from "react-native-litert-lm";
const estimate = estimateMemory({
  modelFileSizeBytes: 2.58e9,
  availableMemoryBytes: llm.getMemoryUsage().availableMemoryBytes,
  config: { backend: "gpu", maxContextTokens: 4096 },
});
if (estimate.verdict !== "safe") showSmallerModelSuggestion();
```

Pass `{ forceLoad: true }` to skip the check.

#### Context / KV-cache forecasting

```typescript
const forecast = llm.getMemoryForecast();
// { contextTokensUsed, remainingTokens, contextUsedFraction,
//   kvCacheBytesUsed, kvCacheBytesRemaining, nearingLimit }
if (forecast?.nearingLimit) warnUserOrSummarizeHistory();
```

#### OS memory-pressure warnings & budgets

```typescript
// Native OS signals (onTrimMemory / dispatch memory-pressure source)
llm.setMemoryWarningCallback((level, usage) => {
  if (level === "critical") llm.unload(); // free ~GBs deterministically
});

// Or app-defined budgets evaluated after each inference
const llm = createLLM({
  enableMemoryTracking: true,
  memoryBudget: {
    warnAtFraction: 0.75,
    criticalAtFraction: 0.9,
    onBudgetExceeded: (level) => console.warn(`memory ${level}`),
  },
});
```

With the `useModel` hook all of this is reactive: `memoryEstimate`,
`memoryForecast`, and `memoryWarning` are returned alongside `memorySummary`.

#### Memory tuning knobs

| Knob | Effect | Platform |
| --- | --- | --- |
| `maxContextTokens` | KV-cache size — the biggest lever | both |
| `activationDataType: 'f16'` | ~halves activation/KV memory | iOS |
| `prefillChunkSize` | caps peak prefill activation memory | iOS |
| `numThreads` | CPU memory-bandwidth pressure | iOS |
| `execute(..., { maxOutputTokens })` | per-message output cap | iOS |
| `loraPath` | one base model + small adapters | both |

### Typed Streaming Events (Tool Calls & Thinking)

With `streamToolCalls: true` (LiteRT-LM v0.14+), tool-call tokens stream in
real time wrapped in channel markers. `executeWithEvents()` parses them into
typed events:

```typescript
await llm.loadModel(modelUrl, { tools, streamToolCalls: true });

await llm.executeWithEvents(
  [{ type: "text", text: "What's the weather in Tokyo?" }],
  (event) => {
    switch (event.type) {
      case "token":    ui.appendText(event.text); break;
      case "toolCall": toolBuffer += event.text; break;
      case "thinking": ui.showReasoning(event.text); break;
    }
    if (event.done) runTool(JSON.parse(toolBuffer));
  },
);
```

Channel markers default to `<tool_call>…</tool_call>` /
`<thinking>…</thinking>` and are configurable via
`createLLM({ streamChannels })` for models with different delimiters.

## Supported Models

All exported model URLs are **public — no authentication required**. Pass them directly to `useModel()` or `loadModel()` for automatic downloading with progress tracking and local caching.

| Constant               | Model                           | Size    | Min RAM | Source      |
| :--------------------- | :------------------------------ | :------ | :------ | :---------- |
| `GEMMA_4_E2B_IT`       | Gemma 4 E2B (Multimodal, IT)    | 2.58 GB | 4 GB+   | HuggingFace |
| `GEMMA_4_E4B_IT`       | Gemma 4 E4B (Higher Quality)    | 3.65 GB | 6 GB+   | HuggingFace |
| `GEMMA_3N_E2B_IT_INT4` | Gemma 3n E2B (Int4, Multimodal) | ~1.3 GB | 4 GB+   | litert.dev  |

> **Recommended:** Use `GEMMA_4_E2B_IT` for most use cases — multimodal (text + vision + audio) and the best quality-to-size ratio.
>
> **iOS Note:** Models larger than ~2 GB require the `com.apple.developer.kernel.extended-virtual-addressing` entitlement. See [iOS Entitlements](#ios-entitlements) below. Gemma 3n E2B (~1.3 GB) works without it.

**Other compatible models** (download `.litertlm` files manually from [HuggingFace](https://huggingface.co/litert-community)):

| Model         | Size    | Min RAM | Notes                 |
| ------------- | ------- | ------- | --------------------- |
| Gemma 3 1B    | ~1 GB   | 4 GB+   | Smallest, fastest     |
| Phi-4 Mini    | ~2 GB   | 4 GB+   | Microsoft's small LLM |
| Qwen 2.5 1.5B | ~1.5 GB | 4 GB+   | Multilingual          |

## API Reference

### `createLLM(options?): LiteRTLM`

Creates a new LLM inference engine instance.

- `options.enableMemoryTracking` — enable automatic memory snapshot recording
- `options.maxMemorySnapshots` — max number of snapshots to retain (default: 256)

### `loadModel(path, config?): Promise<void>`

Loads a model from a local path or HTTPS URL.

| Parameter             | Type     | Default | Description                               |
| --------------------- | -------- | ------- | ----------------------------------------- |
| `path`                | `string` | —       | Absolute path to `.litertlm` or HTTPS URL |
| `config.backend`      | `string` | `'cpu'` | `'cpu'`, `'gpu'`, or `'npu'`              |
| `config.systemPrompt` | `string` | —       | System prompt for the model               |
| `config.temperature`  | `number` | `0.7`   | Sampling temperature                      |
| `config.topK`         | `number` | `40`    | Top-K sampling                            |
| `config.topP`         | `number` | `0.95`  | Top-P (nucleus) sampling                  |
| `config.maxTokens`    | `number` | `1024`  | Maximum generation length                 |

#### Backend Options

| Backend | Engine                         | Speed   | Notes                                                                              |
| ------- | ------------------------------ | ------- | ---------------------------------------------------------------------------------- |
| `'cpu'` | CPU inference                  | Slowest | Always available on all devices                                                    |
| `'gpu'` | Metal (iOS) / OpenCL (Android) | Fast    | iOS: always available. Android: requires OpenCL (Pixel only, not Samsung/Qualcomm) |
| `'npu'` | NPU / Neural Engine            | Fastest | Requires supported hardware; experimental                                          |

> **iOS**: Both `'cpu'` and `'gpu'` (Metal) are supported. The engine automatically tries fallback backend combinations if the primary one fails.
>
> **Android GPU**: The GPU backend requires OpenCL, which is **not available on most Samsung and Qualcomm devices**. Use `checkBackendSupport('gpu')` to check before loading. The engine will throw a clear error if GPU is unsupported.

### `sendMessage(message): Promise<string>`

Runs inference synchronously on a background thread. Returns the complete response.

### `sendMessageAsync(message, callback)`

Streaming generation. Callback signature: `(token: string, isDone: boolean) => void`.

### `sendMessageWithImage(message, imagePath): Promise<string>`

Send a message with an image (for vision models like Gemma 4 E2B).

### `sendMessageWithAudio(message, audioPath): Promise<string>`

Send a message with audio (for audio-capable models like Gemma 4 E2B).

### `getStats(): GenerationStats`

Returns performance metrics from the last inference call.

### `getMemoryUsage(): MemoryUsage`

Returns real OS-level memory usage.

### `getHistory(): Message[]`

Returns the conversation history.

### `resetConversation()`

Clears conversation context and starts a fresh session.

### `close()`

Releases all native resources. Call when the model is no longer needed.

### `deleteModel(fileName): Promise<void>`

Deletes a cached model file from the app's local storage.

### Utility Functions

```typescript
import {
  checkBackendSupport,
  checkMultimodalSupport,
  getRecommendedBackend,
} from "react-native-litert-lm";

// Check if GPU is supported on this device
const gpuWarning = checkBackendSupport("gpu");

// Check NPU support
const npuWarning = checkBackendSupport("npu"); // string | undefined

// Check multimodal support
const mmError = checkMultimodalSupport(); // string | undefined

// Get recommended backend
const backend = getRecommendedBackend(); // 'cpu'
```

## Requirements

| Dependency                 | Version       |
| -------------------------- | ------------- |
| React Native               | 0.76+         |
| react-native-nitro-modules | 0.35.0+       |
| Android API                | 26+ (ARM64)   |
| iOS                        | 15.0+ (ARM64) |
| LiteRT-LM Engine           | 0.12.0        |

## Platform Support

| Platform | Status   | Architecture | Backends                                          |
| -------- | -------- | ------------ | ------------------------------------------------- |
| Android  | ✅ Ready | arm64-v8a    | CPU (all devices), GPU (OpenCL devices only), NPU |
| iOS      | ✅ Ready | arm64        | CPU, GPU (Metal — always available)               |

### iOS Feature Matrix

| Feature                      | Status | Notes                                                  |
| ---------------------------- | ------ | ------------------------------------------------------ |
| Text inference (blocking)    | ✅     | Direct FFI using `dev.litert.engine` background queue  |
| Text inference (streaming)   | ✅     | Token-by-token callbacks                               |
| CPU inference                | ✅     | Safe fallback default                                  |
| GPU inference (Metal/MPS)    | ✅     | Supported via `backend: 'gpu'`                         |
| Model download with progress | ✅     | URLSession-based, cached in `Caches/`                  |
| Memory tracking              | ✅     | Real-time Resident Set Size (RSS) tracking             |
| Multi-turn conversation      | ✅     | Context retained across turns                          |
| Multimodal (image/audio)     | ✅     | Zero-copy `ArrayBuffer` mapping to FFI input buffers   |
| Speculative Decoding         | ✅     | Dynamic capabilities check during model pre-load       |
| Function / Tool Calling      | ✅     | Supported via JSON-encoded schema specification        |

### iOS Entitlements

Models larger than ~2 GB (like Gemma 4 E2B at 2.58 GB) require the **Extended Virtual Addressing** entitlement on iOS physical devices. Without it, iOS limits virtual memory to ~2 GB and the app will be killed by Jetsam.

Add to your app's `.entitlements` file:

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

> **Note:** This entitlement requires a **paid Apple Developer account** ($99/year). Gemma 3n E2B (~1.3 GB) works without it.

## iOS FFI Architecture & Integration

The library uses a highly optimized Swift Direct-FFI bridge that links directly with the pre-compiled C library `CLiteRTLM.xcframework`.

### Key Design Commitments

1. **JSI Thread Safety (User Rule #1)**:
   - The JSI/JS thread must never be blocked by native synchronous lock-waiting operations.
   - We dispatch all FFI calls to a serial background `dev.litert.engine` queue, executing callbacks asynchronously to prevent deadlocking JSI execution.

2. **Zero-Copy Memory Pipelines (User Rule #2)**:
   - Enforce the use of Nitro Modules' `ArrayBuffer` directly referencing native memory pointers (`ArrayBuffer.data`) when processing heavy media assets like images or audio.

3. **Manual FFI Resource Management (User Rule #3)**:
   - Raw pointers (`LiteRtLmEngine*`, `LiteRtLmConversation*`) are manually allocated and strictly deallocated inside Swift `deinit` and `close()` destructors to guarantee 0% memory leaks during prolonged inference sessions.

### Architecture Topology

```
┌──────────────────────────────────────────────────────────┐
│  React Native (TypeScript / JavaScript)                  │
├──────────────────────────────────────────────────────────┤
│  Nitro Modules JSI Bindings (`HybridLiteRTLMSpec`)       │
├─────────────────────────────┬────────────────────────────┤
│  Android (Kotlin)           │  iOS (Swift Direct FFI)    │
│  `HybridLiteRTLM.kt`        │  `HybridLiteRTLM.swift`    │
│  `litertlm-android` AAR     │  `CLiteRTLM.xcframework`   │
└─────────────────────────────┴────────────────────────────┘
```

#### Android Bridging
- Conforms fully to `HybridLiteRTLMSpec` using Kotlin.
- Incorporates Proguard keep rules to prevent dynamic JSI/JNI code stripping.
- Declares `<uses-native-library android:name="libOpenCL.so" android:required="false" />` to load dynamic OpenCL for GPU delegate acceleration on Android 12+ without throwing platform installer exceptions.

#### iOS Bridging
- Entirely written in native Swift (`HybridLiteRTLM.swift`) calling direct FFI.
- Avoids the upstream Swift SDK `actor` lock-blocking deadlocks by utilizing low-level C functions directly.
- Implements custom `getMemoryUsage` that queries the OS directly via `mach_task_basic_info` to get precise real-time Resident Set Size (RSS) metrics.

## Testing

The library includes a comprehensive multi-tier unit testing suite designed to run quickly on host machines (CI runners or local development environments) without requiring a physical test device.

### 1. JavaScript / TypeScript Layer (Jest)

The JS/TS layer uses Jest to validate the `useModel` hook, download progress callbacks, URL query scrubbing, file storage helpers, and the zero-copy native memory tracker buffer allocations.

* **Setup & Mocking**: Includes an active stub (`src/__mocks__/react-native-nitro-modules.ts`) that mocks the Nitro Modules `HybridObject` architecture.
* **How to run**:
  ```bash
  npm run test
  ```

### 2. Android Kotlin Layer (Robolectric)

The Android layer uses local JUnit Robolectric tests to run Android code on the JVM, sandboxing OS dependencies. It validates HTTPS schema constraints, path traversal mitigations, and initial telemetry states.

* **Setup & Mocking**: Uses a local shadow `Promise` implementation to test thread-asynchronous errors.
* **How to run**:
  ```bash
  cd example/android
  ./gradlew :react-native-litert-lm:testDebugUnitTest
  ```

### 3. iOS Swift Layer (XCTest)

The iOS layer leverages native XCTests integrated directly into CocoaPods via standard development test specs. It verifies FFI path traversal blocking, non-HTTPS download blocks, automatic `deinit` cleanup, and Mach-based telemetry bounds.

* **How to run**:
  1. Boot your preferred iOS simulator (e.g., iPhone 16 running iOS 18.6).
  2. Run the tests using `xcodebuild`:
     ```bash
     cd example/ios
     xcodebuild test -workspace LLMTest.xcworkspace -scheme react-native-litert-lm-Unit-Tests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 16'
     ```

### Security & Sanitization Protections Checked
Every test run automatically asserts:
- **Defense in depth for download boundaries**: Blocks non-HTTPS schemes at both JS model factory and low-level native layers.
- **Path Traversal protections**: Prevents directory traversal attacks (`..`, `/`, `\`) in download and deletion APIs.
- **Telemetry sanity**: Ensures zero-leak memory usage telemetry boundaries stay strictly linear.

## License

The code in this repository is licensed under the **[MIT License](LICENSE)**.

### ⚠️ AI Model Disclaimer

This library is an execution engine for on-device LLMs. The AI models themselves are **not** distributed with this package and have their own licenses:

- **Gemma (Google)**: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- **Llama 3 (Meta)**: [Llama 3.2 Community License](https://www.llama.com/llama3/license/)
- **Qwen (Alibaba)**: [Apache 2.0](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE)
- **Phi (Microsoft)**: [MIT License](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE)

By downloading and using these models, you agree to their respective licenses and acceptable use policies. The author of `react-native-litert-lm` takes no responsibility for model outputs or applications built with them.
