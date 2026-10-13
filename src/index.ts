import { Platform } from "react-native";
import type { Backend } from "./specs/LiteRTLM.nitro";
import { warnDeprecated } from "./core/deprecations";
import {
  manifestPlatformFor,
  resolveFromManifest as resolveFromManifestCore,
  type ManifestResolution,
  type ResolveFromManifestOptions,
} from "./core/manifestResolver";

export type {
  LiteRTLM,
  LLMConfig,
  Message,
  Backend,
  Role,
  GenerationStats,
  MemoryUsage,
  /** New in v0.5: pass to execute() instead of individual send methods */
  MultimodalPart,
  PartType,
  ActivationDataType,
  MemoryWarningLevel,
  ToolDefinition,
  ToolCall,
} from "./specs/LiteRTLM.nitro";


// Re-export memory tracking utilities (uses NitroModules.createNativeArrayBuffer v0.35+)
export type {
  MemorySnapshot,
  MemoryTracker,
  MemoryTrackerSummary,
} from "./memoryTracker";
export { createMemoryTracker, createNativeBuffer } from "./memoryTracker";

// Everything react-native-free — memory estimation and forecasting, typed
// errors, stream-event parsing, manifest resolution, bundle model info,
// versions — also importable on its own as `react-native-litert-lm/core`.
// The two functions below that need the device (resolveFromManifest,
// recommendBackend) shadow their core versions with device-aware defaults.
export * from "./core";

/**
 * Fetch `<repo>`'s litertlm_manifest.json from the Hugging Face Hub and pick
 * the right .litertlm variant, backend, config, and stream channels for this
 * device. `platform` defaults to this device's OS; pass it to override.
 * Never throws: returns null when the repo ships no manifest, the manifest's
 * schema is unsupported, the fetch fails or is aborted (`options.signal`),
 * or no variant lists an explicitly requested backend — fall back to your
 * current loading path. Only the unexpected cases log a console.warn line;
 * a missing manifest and an abort are silent.
 *
 * @example
 * ```typescript
 * const resolution = await resolveFromManifest("litert-community/LFM2.5-1.2B-Instruct");
 * if (resolution) {
 *   const llm = createLLM({ streamChannels: resolution.streamChannels });
 *   await llm.loadModel(resolution.url, { ...resolution.config });
 * }
 * ```
 */
export function resolveFromManifest(
  repo: string,
  options: ResolveFromManifestOptions = {},
): Promise<ManifestResolution | null> {
  return resolveFromManifestCore(repo, {
    ...options,
    platform: options.platform ?? manifestPlatformFor(Platform.OS),
  });
}

// Bundle introspection (LiteRT-LM 0.18 model info): know what a .litertlm
// file is, what it accepts and where it runs — before loading any weights.
export { inspectModel, recommendBackend, getDeviceInfo } from "./modelInspector";

export type {
  LiteRTLMInstance,
  CreateLLMOptions,
  StreamEventCallback,
  ConversationHandle,
  ConversationOptions,
} from "./modelFactory";
export { ModelRegistry } from "./modelRegistry";

// On-device embeddings (LiteRT-LM EmbeddingEngine, 0.17+)
export { createEmbedder } from "./embedder";
export type { EmbedderInstance, EmbeddingInput } from "./embedder";
export type { EmbedderConfig, EmbeddingOptions } from "./specs/Embedder.nitro";
export type { ModelDownloadOptions } from "./modelRegistry";
export * from "./hooks";

/**
 * Creates a new LiteRT-LM inference engine instance.
 *
 * @example
 * ```typescript
 * import { createLLM } from 'react-native-litert-lm';
 *
 * const llm = createLLM();
 * await llm.loadModel('/path/to/gemma-3n-e2b.litertlm', {
 *   backend: 'gpu',
 *   temperature: 0.7,
 *   maxContextTokens: 4096,
 *   maxOutputTokens: 512
 * });
 *
 * // ── Unified entry point (recommended) ─────────────────────────────────────
 * // Blocking text
 * const response = await llm.execute([{ type: 'text', text: 'Hello!' }]);
 *
 * // Streaming text
 * await llm.execute(
 *   [{ type: 'text', text: 'Write me a haiku' }],
 *   (token, done) => { process.stdout.write(token); }
 * );
 *
 * // Multimodal (image path — auto-scaled to 1024px on both platforms)
 * const desc = await llm.execute([
 *   { type: 'text', text: 'Describe this image' },
 *   { type: 'image', path: '/path/to/photo.jpg' },
 * ]);
 *
 * // The older sendMessage* methods still work but are deprecated (removed in 1.0).
 *
 * llm.close();
 * ```
 */
export { createLLM } from "./modelFactory";

/**
 * Pre-defined model identifiers for common models.
 * Use with model download utilities or as reference.
 */
export const Models = {
  /** Gemma 4 E2B Instruct (2B parameters, latest generation) */
  GEMMA_4_E2B: "gemma-4-E2B-it-litert-lm",
  /** Gemma 4 E4B Instruct (4B parameters, higher quality) */
  GEMMA_4_E4B: "gemma-4-E4B-it-litert-lm",
  /** Gemma 3n E2B (2B parameters, efficient) */
  GEMMA_3N_E2B: "gemma-3n-E2B-it-litert-lm-preview",
  /** Gemma 3n E4B (4B parameters, higher quality) */
  GEMMA_3N_E4B: "gemma-3n-E4B-it-litert-lm-preview",
  /** Gemma 3 1B (smallest Gemma) */
  GEMMA_3_1B: "Gemma3-1B-IT_multi-prefill-seq_q4_ekv4096",
  /** Phi-4 Mini Instruct */
  PHI_4_MINI: "Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096",
  /** Qwen 2.5 1.5B Instruct */
  QWEN_2_5_1_5B: "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096",
  /** EmbeddingGemma 2 text encoder (270M, 165 MB) — use with createEmbedder() */
  EMBEDDING_GEMMA_2_TEXT_270M: "embeddinggemma-2-text-270m-litert-lm",
  /** EmbeddingGemma 2 text + image + audio encoder (740M, 485 MB) — use with createEmbedder() */
  EMBEDDING_GEMMA_2_740M: "embeddinggemma-2-740m-litert-lm",
  /** Liquid AI LFM2.5 1.2B Instruct (736 MB int4, CPU) — see {@link LFM2_5_1_2B_INSTRUCT} */
  LFM2_5_1_2B_INSTRUCT: "LFM2.5-1.2B-Instruct",
} as const;

export type ModelId = (typeof Models)[keyof typeof Models];

/**
 * Get the recommended backend for the current platform.
 * Returns 'cpu' as the safe default. GPU (Metal on iOS, GPU delegate on Android)
 * is faster but may not be available on all devices or model configurations.
 *
 * @deprecated Use `recommendBackend(await inspectModel(path))`, which knows
 * what the bundle supports and what this device can run. `loadModel()` already
 * does this when `backend` is omitted.
 * @returns The recommended backend ('cpu')
 *
 * @example
 * ```typescript
 * const backend = getRecommendedBackend();
 * llm.loadModel(path, { backend });
 * ```
 */
export function getRecommendedBackend(): Backend {
  warnDeprecated("getRecommendedBackend()", "recommendBackend(await inspectModel(path))");
  // CPU is the safe default — always available, broadly compatible.
  // GPU is faster but may fail on some models/devices.
  return "cpu";
}

/**
 * Check if a backend configuration is supported on the current platform.
 * Returns a warning message if the configuration may have issues.
 *
 * @deprecated Use `recommendBackend(info)` — its `candidates`, `compatible`
 * and `reason` account for the bundle and this device's GPU driver and SoC.
 *
 * @param backend The backend to check
 * @returns Warning message if there may be issues, undefined if OK
 *
 * @example
 * ```typescript
 * const warning = checkBackendSupport('npu');
 * if (warning) {
 *   console.warn(warning);
 * }
 * ```
 */
export function checkBackendSupport(backend: Backend): string | undefined {
  warnDeprecated("checkBackendSupport()", "recommendBackend(info).candidates");
  if (backend === "gpu") {
    if (Platform.OS === "android") {
      // LiteRT-LM GPU delegate requires OpenCL, which is unavailable
      // on most Samsung/Qualcomm devices. Only Pixel devices reliably expose it.
      return "GPU backend requires OpenCL support, which is unavailable on most Samsung and Qualcomm devices. Will automatically fall back to CPU if unavailable.";
    }
    // iOS always supports GPU via Metal
    return undefined;
  }

  if (backend === "npu") {
    if (Platform.OS === "android") {
      return "NPU backend requires compatible hardware (Qualcomm Hexagon, MediaTek APU, etc.). Will automatically fall back to CPU if unavailable.";
    }
    if (Platform.OS === "ios") {
      return "NPU (Neural Engine) is not yet supported on iOS. Mapped to GPU (Metal) internally.";
    }
  }

  return undefined;
}

/**
 * Check if multimodal features (image/audio) are supported on the current platform.
 * Returns an error message if not supported, undefined if OK.
 *
 * @deprecated Always returned undefined. Use `(await inspectModel(path)).modalities`,
 * which reports what the bundle actually accepts.
 *
 * Both iOS (v0.12.0 CLiteRTLM xcframework) and Android (LiteRT-LM SDK) ship the
 * vision/audio executor ops, so there is no platform-level block. Whether a
 * given call succeeds depends on the **loaded model**: only multimodal models
 * (e.g. Gemma 3n) bundle the vision/audio executors. Pass `multimodal: true` to
 * `loadModel` for such models, or rely on filename sniffing ("3n"/"gemma3").
 *
 * @returns Error message if multimodal is not supported, undefined if OK
 *
 * @example
 * ```typescript
 * const error = checkMultimodalSupport();
 * if (error) {
 *   console.warn(error);
 *   // Fall back to text-only
 * } else {
 *   llm.sendMessageWithImage('Describe this', imagePath);
 * }
 * ```
 */
export function checkMultimodalSupport(): string | undefined {
  warnDeprecated("checkMultimodalSupport()", "(await inspectModel(path)).modalities");
  // Supported on both platforms with a multimodal model loaded.
  return undefined;
}

/**
 * Download URL for the Gemma 3n E2B IT INT4 model (~3.66 GB as served).
 * Public mirror — no authentication required.
 *
 * @deprecated Prefer {@link GEMMA_4_E2B_IT}. It is smaller (2.58 GB), adds
 * audio input, tool calling and thinking, and is served directly from the
 * Hugging Face Hub. Gemma 3n is mirrored here only because the upstream repo
 * `google/gemma-3n-E2B-it-litert-lm` is gated and returns 401 without an
 * access token and manual license approval.
 *
 * Still fully supported. The former `https://litert.dev/...` URL now 301s to
 * this one, so apps already shipped against it keep working unchanged.
 *
 * Note: on iOS this exceeds the ~2 GB threshold, so the Extended Virtual
 * Addressing entitlement is required (same as Gemma 4).
 */
export const GEMMA_3N_E2B_IT_INT4 =
  "https://models.litert.dev/gemma-3n-E2B-it-int4.litertlm";

/**
 * Download URL for the Gemma 4 E2B IT model (2.58 GB).
 * Public — no HuggingFace account required.
 */
export const GEMMA_4_E2B_IT =
  "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm";

/**
 * Download URL for the Gemma 4 E4B IT model (3.65 GB).
 * Higher quality than E2B but requires more device memory.
 * Public — no HuggingFace account required.
 */
export const GEMMA_4_E4B_IT =
  "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/resolve/main/gemma-4-E4B-it.litertlm";

/**
 * Download URL for EmbeddingGemma 2 Text 270M (165 MB): text-only, 768-dim
 * vectors with Matryoshka truncation to 512/256/128, 100+ languages and code.
 * Public — no Hugging Face account required. Load with `createEmbedder()`.
 */
export const EMBEDDING_GEMMA_2_TEXT_270M =
  "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm";

/**
 * Download URL for EmbeddingGemma 2 740M (485 MB): text, images and audio in
 * one 768-dim space (cross-modal search). Public — no Hugging Face account
 * required. Load with `createEmbedder()`; per-SoC NPU variants live in the
 * same repository.
 */
export const EMBEDDING_GEMMA_2_740M =
  "https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/main/embeddinggemma-2-740m.litertlm";

/**
 * Download URL for Liquid AI's LFM2.5 1.2B Instruct (736 MB, int4) — a small,
 * fast chat model, verified on device (facts, multi-turn, system prompts,
 * structured output). Runs on CPU: the Metal GPU engine fails for this model
 * family on iOS; on Android, `resolveFromManifest("litert-community/LFM2.5-1.2B-Instruct")`
 * picks the GPU re-export where it helps. Its tool calls use LFM's own
 * `<|tool_call_start|>` syntax, which `parseToolCalls()` doesn't read.
 * License: LFM Open License v1.0 (not Apache) — check its commercial terms.
 * Public — no Hugging Face account required.
 */
export const LFM2_5_1_2B_INSTRUCT =
  "https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/main/LFM2.5-1.2B-Instruct_int4.litertlm";
