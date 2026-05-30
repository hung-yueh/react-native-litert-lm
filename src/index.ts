import { NitroModules } from "react-native-nitro-modules";
import { Platform } from "react-native";
import type {
  LiteRTLM,
  LLMConfig,
  Message,
  Backend,
  Role,
  GenerationStats,
  MemoryUsage,
} from "./specs/LiteRTLM.nitro";

export type {
  LiteRTLM,
  LLMConfig,
  Message,
  Backend,
  Role,
  GenerationStats,
  MemoryUsage,
} from "./specs/LiteRTLM.nitro";


// Re-export memory tracking utilities (uses NitroModules.createNativeArrayBuffer v0.35+)
export type {
  MemorySnapshot,
  MemoryTracker,
  MemoryTrackerSummary,
} from "./memoryTracker";
export { createMemoryTracker, createNativeBuffer } from "./memoryTracker";

export type { LiteRTLMInstance } from "./modelFactory";
export * from "./hooks";

/**
 * Creates a new LiteRT-LM inference engine instance.
 *
 * @example
 * ```typescript
 * import { createLLM } from 'react-native-litert-lm';
 *
 * // Basic usage with Gemma 3n
 * const llm = createLLM();
 * llm.loadModel('/path/to/gemma-3n-e2b.litertlm', {
 *   backend: 'gpu',
 *   temperature: 0.7,
 *   maxTokens: 512
 * });
 *
 * // Simple text generation
 * const response = llm.sendMessage('Hello, how are you?');
 * console.log(response);
 *
 * // Streaming generation
 * llm.sendMessageAsync('Tell me about React Native', (token, done) => {
 *   process.stdout.write(token);
 *   if (done) console.log('\n--- Done ---');
 * });
 *
 * // Check stats
 * const stats = llm.getStats();
 * console.log(`Generated at ${stats.tokensPerSecond} tokens/sec`);
 *
 * // Cleanup
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
} as const;

export type ModelId = (typeof Models)[keyof typeof Models];

/**
 * Get the recommended backend for the current platform.
 * Returns 'cpu' as the safe default. GPU (Metal on iOS, GPU delegate on Android)
 * is faster but may not be available on all devices or model configurations.
 *
 * @returns The recommended backend ('cpu')
 *
 * @example
 * ```typescript
 * const backend = getRecommendedBackend();
 * llm.loadModel(path, { backend });
 * ```
 */
export function getRecommendedBackend(): Backend {
  // CPU is the safe default — always available, broadly compatible.
  // GPU is faster but may fail on some models/devices.
  return "cpu";
}

/**
 * Check if a backend configuration is supported on the current platform.
 * Returns a warning message if the configuration may have issues.
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
  if (backend === "gpu") {
    if (Platform.OS === "android") {
      // LiteRT-LM GPU delegate requires OpenCL, which is unavailable
      // on most Samsung/Qualcomm devices. Only Pixel devices reliably expose it.
      return "GPU backend requires OpenCL support, which is unavailable on most Samsung and Qualcomm devices.";
    }
    // iOS always supports GPU via Metal
    return undefined;
  }

  if (backend === "npu") {
    if (Platform.OS === "android") {
      return "NPU backend requires compatible hardware (Qualcomm Hexagon, MediaTek APU, etc.). Will fall back to GPU if unavailable.";
    }
    if (Platform.OS === "ios") {
      return "NPU (Neural Engine) is not yet supported on iOS. Use 'gpu' (Metal) or 'cpu' instead.";
    }
  }

  return undefined;
}

/**
 * Check if multimodal features (image/audio) are supported on the current platform.
 * Returns an error message if not supported, undefined if OK.
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
  // Supported on both platforms with a multimodal model loaded.
  return undefined;
}

/**
 * Download URL for the Gemma 3n E2B IT INT4 model (~1.3 GB).
 * Public — hosted on litert.dev, no authentication required.
 */
export const GEMMA_3N_E2B_IT_INT4 =
  "https://litert.dev/gemma-3n-E2B-it-int4.litertlm";

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
