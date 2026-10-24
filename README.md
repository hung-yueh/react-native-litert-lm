# react-native-litert-lm

**Gemma 4 on iPhone and Android from React Native.** On-device inference with [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [Nitro Modules](https://github.com/mrousavy/nitro), with first-class **memory safety** so a 1–4 GB model can't silently OOM-kill your app.

One install:

```bash
npm install react-native-litert-lm react-native-nitro-modules
```

Five lines to a Gemma 4 answer:

```typescript
import { createLLM, GEMMA_4_E2B_IT } from "react-native-litert-lm";

const llm = createLLM();
await llm.loadModel(GEMMA_4_E2B_IT); // downloads 2.58 GB once, then cached
console.log(await llm.execute([{ type: "text", text: "What is the capital of France?" }]));
llm.unload(); // frees the engine; llm stays reusable
```

Check the JS layer on your Mac, no device or model needed:

```bash
git clone https://github.com/hung-yueh/react-native-litert-lm && cd react-native-litert-lm && npm install && npm test
```

📖 **[Documentation →](https://litert.dev)** — guides and the full API reference. Expo and bare setup: [Installation](#installation). iOS needs one [entitlement](#ios-entitlements) for models over ~2 GB.

## Highlights

- 🛡️ **Crash-free memory handling** — pre-flight estimation, live tracking, context forecasting, OS pressure warnings, budgets, and deterministic `unload()`. [See below.](#memory-handling)
- 🔍 **Knows a model before loading it** — `inspectModel()` reads a bundle's type, modalities, context limit and preferred backends without loading weights, and `loadModel()` uses it to pick a backend and refuse what can't run. [See below.](#know-the-model-before-loading-it)
- 🔎 **On-device embeddings** — EmbeddingGemma 2 turns text, images and audio into vectors for semantic search and RAG. [See below.](#embeddings)
- 🤖 **Agents** — the model calls your tools, `toolResultPart()` sends the results back, and the model answers from them. [See below.](#tool-calling-agents)
- 🧯 **Typed errors** — every failure is a `LiteRTLMError` with a stable `code` and the engine's own status. [See below.](#errors)
- ⚡ **Binary multimodal input** — pass image/audio as native `ArrayBuffer`s, no base64 heap blow-up (buffers are staged to temp files for the engine's file-based API).
- 🧩 **Typed streaming events** — `token` / `toolCall` / `thinking` events on both platforms, parsed from the engine's channel markers.
- 🎯 **Guaranteed structured output** — constrain any response to a JSON Schema or regex (constrained decoding, both platforms) — the output *cannot* come back malformed.
- 💬 **Multiple conversations, one engine** — independent chats without loading the model twice.
- 🏎️ **GPU acceleration** — Metal (iOS), OpenCL delegate (Android), with automatic CPU fallback.
- 🧠 **Speculative decoding** — multi-token prediction, per engine or per conversation.
- 📥 **Automatic model download** — HTTPS download with progress and local caching. An interrupted download continues where it stopped, and an optional `sha256` is checked before the file is used.

## Why not react-native-executorch or llama.cpp?

Both are good libraries. This one is built for a narrower job: running a multi-gigabyte model inside a phone app **without the OS killing it and without freezing the UI**. It runs any `.litertlm` bundle LiteRT-LM supports — Gemma 4 / 3n / 3, Qwen, Phi-4 Mini, LFM2.5 — plus EmbeddingGemma for on-device embeddings. Where it pulls ahead is memory and throughput, not model choice.

### Where it wins

- **Memory is managed, not hoped for.** `loadModel()` estimates weights + KV cache + overhead against real OS headroom and refuses loads that estimate says would be Jetsam- or LMK-killed (`forceLoad: true` overrides). After load: live RSS tracking, a context forecast that warns before the KV cache overflows, real OS pressure callbacks, budgets with thresholds, and deterministic `unload()`. The alternatives give you a model handle; the crash-prevention layer is yours to write.
- **Faster where phones are actually fast.** GPU on both platforms with automatic CPU fallback, plus LiteRT-LM's **NPU** path on Android (Qualcomm / MediaTek, with a bundle compiled for that chip). Speculative decoding raises decode throughput on models that ship a drafter; on iOS, `prefillChunkSize` caps peak activation memory on long prompts.
- **No bridge tax.** Nitro JSI: each streamed token reaches JS without serialization, and images and audio cross as raw `ArrayBuffer` bytes — no base64 (a 4 MB photo as base64 is 5+ MB of JS string plus copies). Nothing the JS thread calls waits on the engine: generation runs off-thread, and `close()` or a reset cancels a running reply instead of waiting it out.
- **One model, many contexts.** `createConversation()` keeps several independent chats on one loaded engine and switches between them by replaying the transcript; one generates at a time. Loading a second 2–4 GB engine is not an option on a phone.
- **Output you never repair.** Constrained decoding to a JSON Schema or regex on both platforms: the model can only emit tokens that fit, so there's no repair-and-retry loop. (Give it enough `maxOutputTokens` that a reply isn't cut off.)
- **Agents and search in the same package.** Tool calls arrive as typed events and `toolResultPart()` answers them; `createEmbedder()` turns text, images or audio into vectors for on-device search and RAG.

### Use cases it is built for

| Use case | What matters | What this library gives you |
| --- | --- | --- |
| Chat / assistant on mid-range Android with a 2–4 GB model | Not getting OOM-killed | Pre-flight check, pressure warnings, budgets, `unload()` |
| Structured extraction — receipts, forms, classification into your schema | Output that fits your schema | `responseSchema` / `responseRegex` constrained decoding |
| Voice memos and photos as input | No base64 on the way in | `ArrayBuffer` image / audio parts |
| Agentic apps with tool calls and visible reasoning | Typed events, not regex over text | `token` / `toolCall` / `thinking` events; `toolResultPart()` to answer calls |
| Long-document summarization (up to 128K context) | Knowing when the context will overflow | `getMemoryForecast()`, `getContextTokenCount()` |
| Assistant plus background summarizer in one app | Two conversations, one engine | `createConversation()` — one reply at a time, no second load |
| Semantic search / RAG over app content | Embeddings without a server | `createEmbedder()` with EmbeddingGemma 2 |
| Battery-sensitive workloads on NPU-equipped devices | Throughput per watt | `backend: 'npu'` on Android, with CPU fallback |

### Side by side

| | react-native-litert-lm | react-native-executorch | llama.cpp (llama.rn) |
| --- | --- | --- | --- |
| Engine | Google LiteRT-LM | PyTorch ExecuTorch | llama.cpp |
| Models | `.litertlm` — Gemma 4 / 3n / 3, Qwen, Phi-4 Mini, LFM2.5, any LiteRT-LM bundle | `.pte` from a curated export pipeline | GGUF — anything on Hugging Face |
| Accelerators | CPU, GPU (Metal / OpenCL), NPU (Android) | XNNPACK, Core ML, MLX, Vulkan | Metal; OpenCL on Adreno (limited quant types); Hexagon experimental |
| Memory safety | Pre-flight estimate, live tracking, context forecast, OS pressure warnings, budgets | Manual | Manual (`n_gpu_layers`, context size) |
| Multimodal input transfer | Raw `ArrayBuffer` bytes or a file path | — | File path or base64 |
| Streaming events (`token` / `toolCall` / `thinking`) | Typed, parsed by the library | Depends on export and chat template | Jinja chat templates |
| Structured output | JSON Schema / regex, both platforms | — | GBNF / JSON Schema grammars |
| Multiple chats on one loaded model | Yes (switched by transcript replay) | — | Manual session save / load |
| JS thread during generation | Never waits on the engine; close and reset cancel a running reply | Native modules | Native modules |
| Minimum platform | Nitro Modules 0.37, iOS 15.1, Android API 26 | RN 0.81 / Expo 54 (New Architecture), iOS 17, Android 13 | Varies with build |

**Choose react-native-executorch when** you also need non-LLM pipelines — OCR, speech-to-text, text-to-speech, object detection — from one dependency, and your app is already on the New Architecture.

**Choose llama.cpp when** you need the long tail of GGUF models and community fine-tunes, or full control over quantization and layer offloading, and are comfortable owning memory management yourself.

Competitor details are as of September 2026; check each project's docs for current status.

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

  const ask = async () => console.log(await model.execute([{ type: "text", text: "Hello!" }]));
  return <Button title="Generate" onPress={ask} />;
}
```

Prefer imperative control? Use `createLLM()`:

```typescript
import { createLLM } from "react-native-litert-lm";

const llm = createLLM();
await llm.loadModel("https://example.com/model.litertlm", { backend: "gpu" });
const reply = await llm.execute([{ type: "text", text: "What is the capital of France?" }]);
llm.unload(); // free the engine; llm stays reusable
```

## Know the model before loading it

A `.litertlm` bundle describes itself. `inspectModel()` reads that description without loading the weights, in milliseconds rather than the seconds and gigabytes a load costs:

```typescript
import { inspectModel, recommendBackend, getDeviceInfo } from "react-native-litert-lm";

const info = await inspectModel(path); // a local path, or a URL that is already downloaded
info.type;              // 'llm' | 'embedding' | 'unknown'
info.modalities;        // { text, vision, audio, video }
info.maxContextTokens;  // the bundle's limit
info.backends.text;     // backends it was built for, most preferred first
recommendBackend(info); // { backend, reason, candidates } for this device
getDeviceInfo();        // platform, SoC, memory, GPU availability
```

`loadModel()` runs the same checks for you, so most apps never call these directly:

- **The wrong kind of bundle** rejects with `MODEL_TYPE_MISMATCH`: an embedding model in `createLLM()`, or an LLM in `createEmbedder()`.
- **A bundle that needs a newer engine** rejects with `RUNTIME_TOO_OLD`.
- **With `backend` omitted**, the bundle's preferred backend that this device can run is used (CPU when the bundle declares none). A per-SoC NPU build on another phone rejects with `UNSUPPORTED`.
- **Image and audio encoders** load only if the bundle has them, on the backends it declares; `multimodal: false` keeps an engine text-only.
- **`maxContextTokens`** above the bundle's maximum is clamped, so the memory estimate and the engine agree.

`llm.getModelInfo()` and `useModel().modelInfo` return what the loaded bundle declared.

## Errors

Every failure is a `LiteRTLMError` with a stable `code`, so you branch on codes rather than parse messages. Failures from native code also carry the engine's status as `nativeStatus` (`{ code, name, phase, message }`, where `name` is a google.rpc code such as `RESOURCE_EXHAUSTED`).

| Code | Meaning |
| --- | --- |
| `MEMORY_ESTIMATE_EXCEEDED` | The pre-flight refused a load that would not fit (a `MemoryError` carrying the estimate) |
| `OUT_OF_MEMORY` | The engine ran out of memory |
| `MODEL_NOT_FOUND` / `INVALID_MODEL` | No file at the path, or not a usable bundle |
| `MODEL_TYPE_MISMATCH` / `RUNTIME_TOO_OLD` / `UNSUPPORTED` | The bundle can't run here ([see above](#know-the-model-before-loading-it)) |
| `MODEL_NOT_LOADED` | Inference before `loadModel()` |
| `ENGINE_INIT_FAILED` / `INVALID_CONFIG` | Engine creation, or a config value, failed |
| `INVALID_INPUT` | A bad part: a missing media file, an unreadable buffer, a malformed tool result |
| `INFERENCE_FAILED` / `INFERENCE_CANCELLED` / `EMBEDDING_FAILED` | Generation or embedding failed, or was cancelled |
| `DOWNLOAD_FAILED` | A model download failed |

```typescript
import { LiteRTLMError } from "react-native-litert-lm";

try {
  await llm.loadModel(url);
} catch (e) {
  if (e instanceof LiteRTLMError && e.code === "UNSUPPORTED") offerAnotherModel();
}
```

`useModel()` and `useEmbedder()` expose the code as `errorCode`.

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

`getMemoryForecast()` combines the engine's KV-cache token count with the cost model to warn *before* the context window runs out:

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

On Android the library also protects itself as a last resort: in a genuine memory emergency (`TRIM_MEMORY_RUNNING_CRITICAL` while foregrounded, or `TRIM_MEMORY_COMPLETE` when the cached app is next in line to be killed) it releases the engine automatically — the warning callback fires with `'critical'` and the instance stays reusable, so recover with `loadModel()`. Ordinary lifecycle events (screen lock, home button — `TRIM_MEMORY_UI_HIDDEN`) never release the engine.

### Tuning knobs

Every knob's memory impact, documented. `maxContextTokens` is the biggest lever.

| Knob | Effect | Platform |
| --- | --- | --- |
| `maxContextTokens` | KV-cache size — the biggest lever | both |
| `activationDataType: 'f16'` | ~halves activation memory (mainly GPU) | both |
| `maxVisionTokensPerImage` | caps the visual tokens, and memory, each image may use (Gemma 4: 70–1120) | both |
| `execute(…, { visualTokenBudget })` | per-message image budget, up to that cap | both |
| `prefillChunkSize` | caps peak prefill activation memory | iOS |
| `numThreads` | CPU memory-bandwidth pressure | both |
| `gpu: { metalResidencySet: true }` | keeps GPU weights resident between calls (iOS 18+) | iOS |
| `execute(…, { maxOutputTokens })` | per-message output cap | both |
| `loraPath` | one base model + small adapters | both |

### With the `useModel` hook

All of the above is reactive — `memoryEstimate`, `memoryForecast`, and `memoryWarning` are returned alongside `memorySummary`, updating automatically as you load and generate.

## Inference

### Streaming

```typescript
await llm.execute([{ type: "text", text: "Tell me a story" }], (token, done) => {
  process.stdout.write(token);
  if (done) console.log("\n— done —");
});
```

### Typed streaming events (tool calls & thinking)

`executeWithEvents()` turns the raw token stream into typed events. Tool calls
the model emits arrive as `toolCall` events on **both platforms**; setting
`streamToolCalls: true` additionally streams tool-call and reasoning tokens
*as they are generated* (iOS only — on Android a tool call surfaces once it is
complete, which is what most callers want anyway):

```typescript
await llm.loadModel(modelUrl, { tools }); // + streamToolCalls: true for token-level streaming on iOS

await llm.executeWithEvents([{ type: "text", text: "Weather in Tokyo?" }], (event) => {
  switch (event.type) {
    case "token":    ui.appendText(event.text); break;
    case "toolCall": toolBuffer += event.text; break;
    case "thinking": ui.showReasoning(event.text); break;
  }
  if (event.done) runTool(JSON.parse(toolBuffer));
});
```

Markers default to `<tool_call>…</tool_call>` / `<thinking>…</thinking>` and are configurable via `createLLM({ streamChannels })`. To read whole calls with parsed arguments and answer them, see [Tool calling](#tool-calling-agents).

### Multimodal (binary buffers)

Pass native-backed `ArrayBuffer`s directly — no base64 encoding. (Internally the engine's API is file-based, so buffers are staged to temp files that are cleaned up after inference.)

```typescript
const buf = await (await fetch(Image.resolveAssetSource(require("./photo.jpg")).uri)).arrayBuffer();

const reply = await llm.execute([
  { type: "image", imageBuffer: buf },
  { type: "text", text: "Describe this image." },
]);
```

Parts can also name a file: `{ type: "image", path }` or `{ type: "audio", path }`. Images and audio need a bundle with the matching encoder (e.g. Gemma 4 E2B); `inspectModel()` shows which it has, and sending media to an engine without one rejects with `UNSUPPORTED`.

### Tool calling (agents)

Declare tools when you load. When the model wants one, its reply carries the call; run the tool and send the result back with `toolResultPart()`, and the model continues from it:

```typescript
import { parseToolCalls, stripToolCalls, toolResultPart } from "react-native-litert-lm";

await llm.loadModel(GEMMA_4_E2B_IT, {
  tools: [{
    name: "get_current_weather",
    description: "Get the current weather for a location",
    parametersJson: JSON.stringify({
      type: "object",
      properties: { location: { type: "string" } },
      required: ["location"],
    }),
  }],
});

let reply = await llm.execute([{ type: "text", text: "Is it warmer in Paris or Rome?" }]);
for (let step = 0; step < 5; step++) {
  const calls = parseToolCalls(reply); // [{ name, arguments }]
  if (calls.length === 0) break;
  reply = await llm.execute(
    await Promise.all(calls.map(async (call) => toolResultPart(call.name, await tools[call.name](call.arguments)))),
  );
}
showAnswer(stripToolCalls(reply));
```

- **Answer every call of a reply in one `execute()`**; a message of tool results takes no other parts.
- **`getHistory()` keeps the exchange** as structured turns: the model's `toolCalls`, then one `'tool'` turn per result. Conversation switches replay it, so a restored conversation still knows what the tools returned.
- **Integers in tool arguments stay integers.**
- **It depends on the model's chat template.** Gemma 4 handles the round trip; LFM2.5 writes calls in its own syntax, which isn't parsed.

### Speculative decoding

Bundles with a multi-token-prediction drafter (Gemma 4 E2B ships one) can decode faster. `enableSpeculativeDecoding: true` turns it on for the engine, and `createConversation({ speculativeDecoding })` overrides it per conversation. A bundle without a drafter ignores the request with a warning. Measure on your target devices: on CPU it isn't always a win.

### Chat templates

Every bundle carries its own chat template. When a community conversion's template is missing or wrong, `chatTemplate` replaces it with a Jinja template of yours. Start from upstream's [`models/<family>/chat_template.jinja`](https://github.com/google-ai-edge/LiteRT-LM/tree/main/models); the engine adds the BOS token itself, so the template must not.

```typescript
await llm.loadModel(url, { chatTemplate: myJinjaTemplate });
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

Each conversation takes `systemPrompt` and `speculativeDecoding` overrides.

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

## Embeddings

`createEmbedder()` runs an embedding model beside an LLM, or on its own: semantic search, RAG retrieval, deduplication, clustering and classification, all on device. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) comes as a 165 MB text model and a 485 MB model that embeds images and audio into the same space.

```typescript
import { createEmbedder, EmbeddingTask, topK, EMBEDDING_GEMMA_2_TEXT_270M } from "react-native-litert-lm";

const embedder = createEmbedder();
await embedder.loadModel(EMBEDDING_GEMMA_2_TEXT_270M, { maxInputLength: 256 });

const options = { normalize: true, outputSize: 256 } as const; // Matryoshka: 768 → 256 dimensions
const index = await embedder.embedBatch(notes.map(EmbeddingTask.document), options);
const query = await embedder.embed(EmbeddingTask.searchQuery("when is my flight?"), options);
const best = topK(query, index, 3); // [{ index, score }], best first
```

- **Prompts matter.** EmbeddingGemma 2 was trained with task prompts: `EmbeddingTask.searchQuery` for queries and `EmbeddingTask.document` (a string, or `{ title, text }`) for what you search. `questionAnswering`, `factChecking`, `codeRetrieval`, `classification`, `clustering` and `sentenceSimilarity` cover the other tasks. Images and audio take no prompt; pass them as parts.
- **Vectors are `Float32Array`s** over native memory, with no JSON on the way. `cosineSimilarity`, `dotProduct`, `normalize` and `topK` work on them in plain JS, which is plenty for the thousands of vectors an on-device index holds.
- **`outputSize`** truncates to 512, 256 or 128 dimensions (close to lossless down to 256), and the engine re-normalizes after truncating. Compare only vectors of the same size.
- **`maxInputLength`** picks the smallest compiled input that fits (128–8192 tokens): less memory, faster calls.

`useEmbedder(urlOrPath, config)` manages download, load and cleanup like `useModel()`. The example app's **Search notes** sheet is a working semantic search over a dozen notes.

## Supported Models

Any `.litertlm` bundle runs if the engine knows its model family: Gemma 4, 3n and 3, FunctionGemma, Qwen 3 and 3.5, LFM2, MiniCPM, FastVLM and generic decoders, plus EmbeddingGemma 2 for embeddings. [litert-community](https://huggingface.co/litert-community) hosts hundreds of them. `inspectModel()` tells you what a bundle is before you load it, and [manifest resolution](#manifest-resolution) picks a repo's right variant for the device.

The exported constants are **public, no auth required**. Pass any of them to `useModel()` / `loadModel()`, or the embedding ones to `useEmbedder()` / `createEmbedder()`:

| Constant | Model | Size | Min RAM |
| --- | --- | --- | --- |
| `GEMMA_4_E2B_IT` | Gemma 4 E2B — text, image, audio; tools and thinking | 2.58 GB | 4 GB+ |
| `GEMMA_4_E4B_IT` | Gemma 4 E4B — higher quality | 3.65 GB | 6 GB+ |
| `LFM2_5_1_2B_INSTRUCT` | Liquid AI LFM2.5 1.2B Instruct — a fast small chat model, CPU | 736 MB | 3 GB+ |
| `EMBEDDING_GEMMA_2_TEXT_270M` | EmbeddingGemma 2 — text embeddings | 165 MB | 2 GB+ |
| `EMBEDDING_GEMMA_2_740M` | EmbeddingGemma 2 — text, image and audio embeddings | 485 MB | 3 GB+ |
| `GEMMA_3N_E2B_IT_INT4` _(deprecated)_ | Gemma 3n E2B (int4, multimodal) | ~3.66 GB | 6 GB+ |

Gemma 4 E2B, LFM2.5 and both EmbeddingGemma 2 models were verified end to end on the iOS simulator and an Android emulator for this release. LFM2.5 is under the [LFM Open License](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE), not Apache; check its commercial terms.

> **`GEMMA_3N_E2B_IT_INT4` is deprecated** — prefer `GEMMA_4_E2B_IT`: it is smaller (2.58 GB), adds audio, tool calling and thinking, and comes straight from the Hub. Gemma 3n is mirrored on `models.litert.dev` only because the upstream repo `google/gemma-3n-E2B-it-litert-lm` is gated (401 without a token and manual license approval). It still works, and the old `litert.dev/...` URL 301s to the mirror.

> **iOS:** models over ~2 GB need the [Extended Virtual Addressing entitlement](#ios-entitlements) — that includes the Gemma 4 and Gemma 3n models above. LFM2.5 and the embedding models fit without it.

## Manifest Resolution

Instead of hardcoding a URL and config, point `resolveFromManifest()` at a HuggingFace repo. It reads that repo's `litertlm_manifest.json` and picks the `.litertlm` variant, backend, sampler defaults, and stream channels that fit **this** device:

```typescript
import { resolveFromManifest, createLLM, GEMMA_4_E2B_IT } from 'react-native-litert-lm';

const resolution = await resolveFromManifest('litert-community/LFM2.5-1.2B-Instruct');
const llm = createLLM(resolution ? { streamChannels: resolution.streamChannels } : undefined);

if (resolution) {
  // config carries the manifest's backend + sampler defaults; your overrides win.
  await llm.loadModel(resolution.url, { ...resolution.config, temperature: 0.7 });
  resolution.notes.forEach((n) => console.warn(n)); // platform_notes + known_issues
} else {
  await llm.loadModel(GEMMA_4_E2B_IT); // no manifest — your normal path
}
```

`platform` defaults to the device OS. Two behaviours worth knowing:

- **It never throws.** You get `null` when the repo ships no manifest, the schema is unsupported, the fetch fails, or you abort it via `options.signal` — so layer it *in front of* your existing loading path rather than replacing it. Only unexpected cases log a warning; a missing manifest and an abort are silent.
- **`backend` is a filter, not a preference.** Request one and only variants listing it are considered; it returns `null` rather than quietly substituting a different backend.

Lower-level pieces are exported too, so you can drive the steps yourself: `fetchManifest`, `parseManifest`, `resolveVariant`, `resolutionFor`, `mergeStreamChannels`, `declaredChannels`, `thinkingMarkers`, `manifestFetchStatus`. Full reference on the [docs site](https://litert.dev/manifest.html).

## API Reference

**`createLLM(options?)`** → instance. Options: `enableMemoryTracking`, `maxMemorySnapshots` (default 256), `memoryBudget`, `streamChannels`.

**`loadModel(path, config?)`** → `Promise<void>`. `path` is a local path or HTTPS URL.

| Config | Default | Notes |
| --- | --- | --- |
| `backend` | the bundle's preference | `'cpu'` \| `'gpu'` \| `'npu'` (auto-fallback to CPU). Omitted: the bundle's preferred backend this device can run, else CPU |
| `systemPrompt` | — | System prompt |
| `temperature` / `topK` / `topP` | `0.7` / `40` / `0.95` | Sampling |
| `maxContextTokens` | `4096` | Total KV-cache budget (tokens) |
| `maxOutputTokens` | `1024` | Max tokens generated per response |
| `streamToolCalls` | `false` | Stream tool-call/thinking tokens mid-generation (iOS only; completed tool calls surface as typed events on both) |
| `enableStructuredOutput` | `false` | Initialize constrained decoding for per-message `responseSchema`/`responseRegex` |
| `thinking` | engine default | `{ enabled, tokenBudget }` reasoning controls (Gemma 4) |
| `tools` | — | Tool definitions — see [Tool calling](#tool-calling-agents) |
| `chatTemplate` | the bundle's | Jinja template that replaces the bundle's — see [Chat templates](#chat-templates) |
| `enableSpeculativeDecoding` | `false` | Multi-token prediction, if the bundle has a drafter |
| `forceLoad` | `false` | Skip the pre-flight memory check |
| memory tuning | — | `numThreads`, `prefillChunkSize`, `activationDataType`, `maxVisionTokensPerImage`, `gpu.metalResidencySet`, `loraPath` — see [Tuning knobs](#tuning-knobs) |
| `singleThreadedExecution` | `false` | One engine thread, for debugging and reproducibility (iOS) |

**Inference:** `execute(parts, onToken?, options?)`, `executeWithEvents(parts, onEvent, options?)`. The older `sendMessage*` family is deprecated: it still works, warns once, and goes in 1.0.

**Tools:** `parseToolCalls(reply | historyTurn)`, `stripToolCalls(reply)`, `toolResultPart(name, response)` — see [Tool calling](#tool-calling-agents).

**Inspection:** `inspectModel(path)`, `recommendBackend(info)`, `getDeviceInfo()`, `llm.getModelInfo()` — see [Know the model before loading it](#know-the-model-before-loading-it).

**Embeddings:** `createEmbedder()` → `loadModel`, `embed`, `embedBatch`, `getModelInfo`, `unload`, `close`; `useEmbedder(urlOrPath, config?)`; `EmbeddingTask`, `topK`, `cosineSimilarity`, `dotProduct`, `normalize` — see [Embeddings](#embeddings).

**Errors:** `LiteRTLMError` (`code`, `nativeStatus`), `MemoryError`, `isMemoryError(e)`, `toLiteRTLMError(e, phase)` — see [Errors](#errors).

**Conversations:** `createConversation(options?)` → handle with `execute`, `executeWithEvents`, `getHistory()`, `release()` — independent chats sharing one engine (see [Multiple conversations](#multiple-conversations-one-engine)).

**Memory:** `estimateMemory(inputs)`, `getMemoryUsage()`, `getMemoryForecast()`, `getContextTokenCount()`, `setMemoryWarningCallback(cb)` / `clearMemoryWarningCallback()`, `memoryTracker`.

**Lifecycle:** `getStats()`, `getHistory()`, `resetConversation()`, `unload()`, `close()`, `deleteModel(fileName)`.

**Deprecated utilities:** `checkBackendSupport(backend)`, `checkMultimodalSupport()` and `getRecommendedBackend()` guessed from the platform. Use `recommendBackend(await inspectModel(path))` and `getDeviceInfo()`, which know the bundle and the device. They warn once and go in 1.0; `setDeprecationWarningsEnabled(false)` silences the warnings.

**Manifest:** `resolveFromManifest(repo, options?)` → `Promise<ManifestResolution | null>`, plus `fetchManifest`, `parseManifest`, `resolveVariant`, `resolutionFor`, `mergeStreamChannels`, `declaredChannels`, `thinkingMarkers`, `manifestFetchStatus` — see [Manifest Resolution](#manifest-resolution).

## Requirements & Platform Support

| | |
| --- | --- |
| React Native | 0.76+ |
| react-native-nitro-modules | 0.37.1+ |
| LiteRT-LM engine | 0.18.0 |
| Android | API 26+, arm64-v8a — CPU (all), GPU (where OpenCL is present), NPU |
| iOS | 15.1+, arm64 — CPU, GPU (Metal; auto-fallback to CPU) |

> **Android GPU** requires the device to ship `libOpenCL.so`, which varies by vendor and SoC rather than by brand (present on a Galaxy S22 / Snapdragon 8 Gen 1, absent on plenty of other devices). `getDeviceInfo().gpuAvailable` reports whether this device has one, and `loadModel()` skips GPU without it; the engine auto-falls back to CPU either way.

### iOS Entitlements

Models over ~2 GB need **Extended Virtual Addressing** or iOS caps virtual memory at ~2 GB and Jetsam kills the app. With Expo, let the config plugin add it (`increasedMemoryLimit` also raises the app's memory limit on devices that allow it):

```json
{ "expo": { "plugins": [["react-native-litert-lm", { "ios": { "extendedVirtualAddressing": true, "increasedMemoryLimit": true } }]] } }
```

Without Expo, add it to your `.entitlements`:

```xml
<key>com.apple.developer.kernel.extended-virtual-addressing</key>
<true/>
```

Both entitlements require a paid Apple Developer account, which is why the plugin leaves them off unless asked.

## Architecture

Nitro Modules (JSI) bridges TypeScript to a per-platform native engine:

```
React Native (TypeScript)
        │  Nitro JSI bindings (HybridLiteRTLMSpec)
   ┌────┴─────────────────────┐
   iOS (Swift Direct FFI)     Android (Kotlin)
   CLiteRTLM.xcframework       litertlm-android AAR
```

- **iOS** — native Swift calling the C FFI directly; load, inference and teardown run on a serial `dev.litert.engine` queue, and nothing the JS thread calls waits on it: accessors like `getStats()` read a lock-protected snapshot, and `close()`, `unload()` and `resetConversation()` cancel a running message instead of waiting it out. Raw pointers are freed deterministically in `deinit`/`close()`/`unload()` for zero leaks. RSS read via `mach_task_basic_info`.
- **Android** — stateless Kotlin conforming to `HybridLiteRTLMSpec`, with Proguard keep rules and optional `libOpenCL.so` probing for the GPU delegate (with a CPU fallback chain when engine creation fails). Messages generate on worker threads; a conversation is closed only once the messages running on it have returned, and `close()` frees the engine on a background thread.
- **Embeddings** — a separate `Embedder` hybrid object with its own serial queue (iOS) or worker thread (Android), so an LLM and an embedder can run at the same time; `close()` returns at once and frees the engine after any in-flight call.

## Testing

Multi-tier suite that runs on CI without a device:

- **JS/TS (Jest):** `npm test` — memory estimator (golden values), forecast/budget logic, bundle pre-flight, error typing, stream-event and tool-call parsing, embedding helpers, ring-buffer tracker, hook & factory behavior, HTTPS guard.
- **Android (Robolectric):** `cd example/android && ./gradlew :react-native-litert-lm:testDebugUnitTest` — covers path-traversal/HTTPS guards, memory telemetry, and error paths. Requires a JDK 21+ test launcher (the litertlm-android AAR is built for Java 21); Gradle picks one up via toolchain auto-detection, or set `org.gradle.java.installations.paths`.
- **iOS (XCTest):** the example's `withLibraryTestSpecs` config plugin adds the library's test spec to the generated Podfile, so after `cd example && npx expo prebuild`, run `cd example/ios && xcodebuild test -workspace LLMTest.xcworkspace -scheme react-native-litert-lm-Unit-Tests -destination 'platform=iOS Simulator,name=iPhone 17'`.

The Jest tier also includes a **config-parity contract test** (every `LLMConfig` key must be forwarded by `useModel` or explicitly excluded) and a **device-baseline guardrail** that validates the memory cost model against peak-RSS numbers recorded in [`scripts/memory-baseline.json`](scripts/memory-baseline.json).

### Real-inference integration suites (opt-in)

Both platforms have a suite that loads an actual `.litertlm` bundle and asserts
on real generations — streaming, system prompt, schema/regex constrained
output, transcript replay, generation controls and tool-call events. Both skip
cleanly when no model is present, so CI stays device-free.

- **iOS** ([`ios/Tests/HybridLiteRTLMIntegrationTests.swift`](ios/Tests/HybridLiteRTLMIntegrationTests.swift)) — runs in the simulator:

  ```bash
  cd example/ios && TEST_RUNNER_LITERTLM_TEST_MODEL=$HOME/.litert-models/gemma-4-E2B-it.litertlm xcodebuild test -workspace LLMTest.xcworkspace -scheme react-native-litert-lm-Unit-Tests -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17'
  ```

- **Android** ([`android/src/androidTest/…/HybridLiteRTLMInstrumentedTest.kt`](android/src/androidTest/java/com/margelo/nitro/dev/litert/litertlm/HybridLiteRTLMInstrumentedTest.kt)) — needs a real device or emulator, since Robolectric cannot run the engine. Install the APK **before** pushing the model and drive it with `am instrument`: `connectedAndroidTest` uninstalls the test APK when it finishes, deleting the pushed model with it. Full command sequence is in the suite's header comment.

Both platforms also have an embedder suite (EmbeddingGemma 2, including cross-modal image and audio checks) and a **model verification battery** that any bundle can run before it earns a constant: facts, multi-turn recall, streaming, a system prompt, structured output, arithmetic, and a reported tool-call round trip. Point `TEST_RUNNER_LITERTLM_VERIFY_MODEL` at a bundle on iOS, or pass `-e verifyModel <file>` on Android.

On-device memory scenarios (OOM prevention, pressure simulation, peak-RSS regression budget) are documented in [`scripts/device-memory-scenarios.md`](scripts/device-memory-scenarios.md), and the scripted example-app integration pass lives in [`scripts/e2e-example-flow.md`](scripts/e2e-example-flow.md).

## Example App

`example/` is a full showcase app — Chat + Memory dashboard (pre-flight verdict, live RSS sparkline, context forecast, pressure warnings), typed streaming events, the tuning knobs, and a **Search notes** sheet: on-device semantic search with EmbeddingGemma 2. Run it with `npm run build`, then `cd example && npm install && npx expo prebuild --clean && npx expo run:ios`.

## License

Code is [MIT](LICENSE).

### ⚠️ AI Model Disclaimer

This library is an execution engine; the models are **not** distributed with it and carry their own licenses — [Gemma](https://ai.google.dev/gemma/terms) (including EmbeddingGemma), [LFM](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE), [Llama 3](https://www.llama.com/llama3/license/), [Qwen](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE), [Phi](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/main/LICENSE). By downloading a model you accept its license and acceptable-use policy. The author takes no responsibility for model outputs.
