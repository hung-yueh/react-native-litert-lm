import type { HybridObject } from "react-native-nitro-modules";
import type { Backend } from "./LiteRTLM.nitro";

/** What a `.litertlm` bundle is for. */
export type ModelType = "llm" | "embedding" | "unknown";

/** An input modality a bundle may accept. */
export type Modality = "text" | "vision" | "audio" | "video";

/** The NPU vendor a bundle (or one of its modalities) was compiled for. */
export type NpuBrand =
  | "unknown"
  | "qualcomm"
  | "googleTensor"
  | "mediatek"
  | "intel"
  | "samsung";

/** Raw per-modality facts read from a bundle header. */
export interface NativeModalityInfo {
  modality: Modality;
  /** Whether the bundle accepts this input modality */
  supported: boolean;
  /** Backends the bundle declares for this modality, most preferred first (empty = undeclared) */
  backends: Backend[];
  /** NPU vendor the modality was compiled for ("unknown" when not NPU-specific) */
  npuBrand: NpuBrand;
  /** Target SoC of an NPU build, e.g. "SM8750" */
  socName?: string;
}

/**
 * Raw bundle metadata as the engine reports it. `inspectModel()` normalizes
 * this into {@link ModelInfo} — prefer that.
 */
export interface NativeModelInfo {
  modelType: ModelType;
  fileSizeBytes: number;
  /** Maximum context the bundle supports (0 = not declared) */
  maxContextTokens: number;
  /** True when the context size is configurable up to `maxContextTokens` */
  isDynamicContext: boolean;
  /** Minimum LiteRT-LM runtime version the bundle needs, when declared */
  minRuntimeVersion?: string;
  modalities: NativeModalityInfo[];
  supportsThinking: boolean;
  supportsFunctionCalling: boolean;
  hasSpeculativeDecoding: boolean;
  /** Raw LiteRtLmSamplerType: 0 unspecified, 1 top-k, 2 top-p, 3 greedy */
  samplerType: number;
  samplerTemperature: number;
  samplerTopK: number;
  samplerTopP: number;
  /** Maximum visual tokens per image (-1 = no vision or undeclared) */
  maxVisionTokenBudget: number;
  /** Visual-token budgets the vision encoder was compiled for */
  visionSignatureSelection: number[];
  /** Embedding width for embedding bundles (-1 otherwise) */
  embeddingDimension: number;
  /** Input lengths (tokens) the embedding encoder was compiled for */
  embeddingSignatureSelection: number[];
}

/** Facts about the device that decide which bundle variant and backend fit. */
export interface NativeDeviceInfo {
  /** "ios" or "android" */
  platform: string;
  osVersion: string;
  /** Hardware model: "iPhone17,1", "Pixel 9 Pro", … */
  model: string;
  /** Android 12+: Build.SOC_MODEL, e.g. "SM8750" */
  socModel?: string;
  /** Android 12+: Build.SOC_MANUFACTURER, e.g. "QTI" */
  socManufacturer?: string;
  totalMemoryBytes: number;
  /** Whether the GPU backend's driver is present (OpenCL on Android; Metal on iOS) */
  gpuAvailable: boolean;
  isEmulator: boolean;
}

/**
 * Reads `.litertlm` bundle metadata without creating an engine, and describes the
 * device. Backs `inspectModel()` and `recommendBackend()`.
 */
export interface ModelInspector
  extends HybridObject<{ ios: "swift"; android: "kotlin" }> {
  /** Read a local bundle's metadata without creating an engine (cached per file version). */
  inspect(modelPath: string): Promise<NativeModelInfo>;
  /** Describe this device. */
  getDeviceInfo(): NativeDeviceInfo;
}
