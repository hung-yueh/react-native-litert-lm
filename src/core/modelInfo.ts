/**
 * Bundle metadata, backend choice and load pre-flight — pure functions over
 * what the engine reads from a `.litertlm` header (LiteRT-LM 0.18 model info).
 * The capability facts come from the bundle itself, never from its file name.
 */
import type { Backend, LLMConfig } from "../specs/LiteRTLM.nitro";
import type {
  Modality,
  ModelType,
  NativeDeviceInfo,
  NativeModelInfo,
  NpuBrand,
} from "../specs/ModelInspector.nitro";
import { LiteRTLMError } from "./errors";
import { resolveMaxContextTokens } from "./memoryEstimator";
import { LITERT_LM_VERSION, compareVersions } from "./version";

export type { Modality, ModelType, NpuBrand };

/** The default sampler a bundle declares. */
export type SamplerType = "topK" | "topP" | "greedy";

/** The NPU a bundle (or one of its modalities) was compiled for. */
export interface NpuTarget {
  brand: NpuBrand;
  /** Target SoC, e.g. "SM8750"; null when not declared */
  socName: string | null;
}

/** Everything a `.litertlm` bundle declares about itself, read without loading weights. */
export interface ModelInfo {
  /** The local file that was inspected */
  path: string;
  fileSizeBytes: number;
  /** `'llm'` for generation, `'embedding'` for {@link createEmbedder}, `'unknown'` for older bundles */
  type: ModelType;
  /** Input modalities the bundle accepts */
  modalities: Record<Modality, boolean>;
  /** Largest context the bundle supports; 0 when undeclared */
  maxContextTokens: number;
  /** True when the context is configurable up to `maxContextTokens`; false = fixed */
  dynamicContext: boolean;
  /** Oldest LiteRT-LM runtime that can run the bundle, when declared */
  minRuntimeVersion: string | null;
  /** Backends the bundle declares per modality, most preferred first (empty = undeclared) */
  backends: Record<Modality, Backend[]>;
  /** NPU targets, for modalities compiled for a specific NPU */
  npu: Partial<Record<Modality, NpuTarget>>;
  /** Generation capabilities (LLM bundles) */
  llm?: {
    supportsThinking: boolean;
    supportsFunctionCalling: boolean;
    /** Ships a multi-token-prediction drafter (`enableSpeculativeDecoding`) */
    speculativeDecoding: boolean;
    /** The author's intended sampler, when declared */
    samplerDefaults: { type: SamplerType; temperature: number; topK: number; topP: number } | null;
    /** Largest visual-token budget per image; null without vision */
    maxVisionTokenBudget: number | null;
    /** Visual-token budgets the vision encoder was compiled for */
    visionTokenOptions: number[];
  };
  /** Embedding capabilities (embedding bundles) */
  embedding?: {
    /** Output width before any `outputSize` truncation; null when undeclared */
    dimension: number | null;
    /** Input lengths (tokens) the encoder was compiled for */
    inputLengthOptions: number[];
    maxVisionTokenBudget: number | null;
    visionTokenOptions: number[];
  };
}

/** The device facts backend selection depends on. */
export interface DeviceInfo extends Omit<NativeDeviceInfo, "platform"> {
  platform: "ios" | "android";
}

const MODALITIES: Modality[] = ["text", "vision", "audio", "video"];
const SAMPLERS: Record<number, SamplerType> = { 1: "topK", 2: "topP", 3: "greedy" };

const orNull = (n: number): number | null => (n >= 0 ? n : null);

/** Normalize the native reading into {@link ModelInfo}. */
export function normalizeModelInfo(path: string, native: NativeModelInfo): ModelInfo {
  const byModality = new Map(native.modalities.map((m) => [m.modality, m]));
  const modalities = {} as Record<Modality, boolean>;
  const backends = {} as Record<Modality, Backend[]>;
  const npu: Partial<Record<Modality, NpuTarget>> = {};
  for (const modality of MODALITIES) {
    const m = byModality.get(modality);
    modalities[modality] = m?.supported ?? false;
    backends[modality] = m ? [...m.backends] : [];
    if (m && (m.npuBrand !== "unknown" || m.socName)) {
      npu[modality] = { brand: m.npuBrand, socName: m.socName ?? null };
    }
  }

  const info: ModelInfo = {
    path,
    fileSizeBytes: native.fileSizeBytes,
    type: native.modelType,
    modalities,
    maxContextTokens: Math.max(0, native.maxContextTokens),
    dynamicContext: native.isDynamicContext,
    minRuntimeVersion: native.minRuntimeVersion ?? null,
    backends,
    npu,
  };

  const samplerType = SAMPLERS[native.samplerType];
  if (native.modelType !== "embedding") {
    info.llm = {
      supportsThinking: native.supportsThinking,
      supportsFunctionCalling: native.supportsFunctionCalling,
      speculativeDecoding: native.hasSpeculativeDecoding,
      samplerDefaults: samplerType
        ? {
            type: samplerType,
            temperature: native.samplerTemperature,
            topK: native.samplerTopK,
            topP: native.samplerTopP,
          }
        : null,
      maxVisionTokenBudget: modalities.vision ? orNull(native.maxVisionTokenBudget) : null,
      visionTokenOptions: [...native.visionSignatureSelection],
    };
  }
  if (native.modelType === "embedding") {
    info.embedding = {
      dimension: orNull(native.embeddingDimension),
      inputLengthOptions: [...native.embeddingSignatureSelection],
      maxVisionTokenBudget: modalities.vision ? orNull(native.maxVisionTokenBudget) : null,
      visionTokenOptions: [...native.visionSignatureSelection],
    };
  }
  return info;
}

/** Result of {@link recommendBackend}. */
export interface BackendRecommendation {
  /** The backend to load with */
  backend: Backend;
  /** Every usable backend for this bundle on this device, best first */
  candidates: Backend[];
  /** False when the bundle targets hardware this device lacks (e.g. another SoC's NPU) */
  compatible: boolean;
  /** Why — log it, or show it in a model picker */
  reason: string;
}

/** Options for {@link recommendBackend}. */
export interface RecommendBackendOptions {
  /**
   * `'speed'` (default) follows the bundle's own preference order;
   * `'compatibility'` puts CPU first whenever the bundle allows it.
   */
  prefer?: "speed" | "compatibility";
  /** This device. Without it, GPU is assumed present and SoC-specific NPU builds are skipped. */
  device?: DeviceInfo;
}

const sameSoc = (a: string, b: string) => a.trim().toUpperCase() === b.trim().toUpperCase();

/**
 * Pick the backend for a bundle on this device, from what the bundle declares
 * (`ModelInfo.backends.text`, most preferred first) filtered by what the
 * device can run: no GPU driver drops `'gpu'`, an NPU build for another SoC
 * drops `'npu'`, and iOS has no NPU backend. A bundle that declares nothing
 * gets CPU — the safe default this library has always used.
 */
export function recommendBackend(
  info: ModelInfo,
  options: RecommendBackendOptions = {},
): BackendRecommendation {
  const { device, prefer = "speed" } = options;
  const declared = info.backends.text;
  if (declared.length === 0) {
    return {
      backend: "cpu",
      candidates: ["cpu"],
      compatible: true,
      reason: "The bundle declares no backend constraints; CPU is the safe default.",
    };
  }

  const dropped: string[] = [];
  const npuTarget = info.npu.text;
  let candidates = declared.filter((backend) => {
    if (backend === "gpu" && device && !device.gpuAvailable) {
      dropped.push(`gpu (no ${device.platform === "android" ? "OpenCL" : "Metal"} driver)`);
      return false;
    }
    if (backend === "npu") {
      if (device?.platform === "ios") {
        dropped.push("npu (no NPU backend on iOS)");
        return false;
      }
      if (npuTarget?.socName) {
        if (!device?.socModel || !sameSoc(device.socModel, npuTarget.socName)) {
          dropped.push(`npu (built for ${npuTarget.socName}, device is ${device?.socModel ?? "unknown"})`);
          return false;
        }
      }
    }
    return true;
  });

  if (prefer === "compatibility" && candidates.includes("cpu")) {
    candidates = ["cpu", ...candidates.filter((b) => b !== "cpu")];
  }

  if (candidates.length === 0) {
    return {
      backend: declared[0],
      candidates: [],
      compatible: false,
      reason:
        `This bundle only runs on ${declared.join(", ")}, which this device cannot provide: ` +
        `${dropped.join("; ")}. Pick another variant of the model.`,
    };
  }
  const why = dropped.length ? ` Skipped ${dropped.join("; ")}.` : "";
  return {
    backend: candidates[0],
    candidates,
    compatible: true,
    reason: `The bundle prefers ${declared.join(" > ")}; ${candidates[0]} is the best this device can run.${why}`,
  };
}

/**
 * 0.7's default for `multimodal` when unset: on for Gemma 3n / Gemma 3 file
 * names. Native code applies it only when a bundle's metadata can't be read
 * (readable bundles load the encoders they declare); deprecated, gone in 1.0.
 */
export function legacyMultimodalHint(modelPath: string): boolean {
  const name = (modelPath.split("/").pop() ?? "").toLowerCase();
  return name.includes("3n") || name.includes("gemma3");
}

/**
 * Refuse a bundle of the wrong type (`MODEL_TYPE_MISMATCH`) or one needing a
 * newer runtime than this package ships (`RUNTIME_TOO_OLD`).
 */
function assertLoadable(
  info: ModelInfo,
  expectType: Exclude<ModelType, "unknown">,
  runtimeVersion: string = LITERT_LM_VERSION,
): void {
  if (info.type !== "unknown" && info.type !== expectType) {
    const hint =
      info.type === "embedding"
        ? " Load it with createEmbedder() instead."
        : " Load it with createLLM() instead.";
    throw new LiteRTLMError(
      "MODEL_TYPE_MISMATCH",
      `${info.path} is ${info.type === "embedding" ? "an embedding" : "an LLM"} bundle.${hint}`,
    );
  }
  if (info.minRuntimeVersion && compareVersions(info.minRuntimeVersion, runtimeVersion) > 0) {
    throw new LiteRTLMError(
      "RUNTIME_TOO_OLD",
      `${info.path} needs LiteRT-LM ${info.minRuntimeVersion} or newer; this package ships ${runtimeVersion}.`,
    );
  }
}

/**
 * The backend to fill in when the caller chose none. A bundle this device
 * cannot run at all (an NPU build for another SoC) fails here with the
 * reason, rather than as an opaque engine error after every fallback.
 */
function compatibleRecommendation(info: ModelInfo, device?: DeviceInfo): BackendRecommendation {
  const rec = recommendBackend(info, { device });
  if (!rec.compatible) {
    throw new LiteRTLMError("UNSUPPORTED", `${info.path}: ${rec.reason}`);
  }
  return rec;
}

/** Result of {@link preflightLoad}. */
export interface PreflightResult {
  /** The config to load with: backend chosen, context clamped */
  config: LLMConfig;
  /** Whether `config` differs from the caller's */
  changed: boolean;
  /** Adjustments that were made, worth logging */
  notes: string[];
}

/** Options for {@link preflightLoad}. */
export interface PreflightOptions {
  device?: DeviceInfo;
  /** The model type the caller is loading (default `'llm'`) */
  expectType?: Exclude<ModelType, "unknown">;
  /** Runtime to check `minRuntimeVersion` against (default: this package's engine) */
  runtimeVersion?: string;
}

/**
 * Check a bundle against a load config before any weights are touched, and
 * resolve what the caller left open. Throws `MODEL_TYPE_MISMATCH` (an
 * embedding bundle passed to createLLM, or vice versa), `RUNTIME_TOO_OLD`, and
 * — when no backend was given — `UNSUPPORTED` for a bundle this device can't run.
 * Fills `backend` from {@link recommendBackend} when omitted, and clamps the
 * context (explicit or the 4096 default) to the bundle's declared maximum.
 */
export function preflightLoad(
  info: ModelInfo,
  config: LLMConfig | undefined,
  options: PreflightOptions = {},
): PreflightResult {
  assertLoadable(info, options.expectType ?? "llm", options.runtimeVersion);

  const notes: string[] = [];
  const resolved: LLMConfig = { ...config };
  let changed = false;
  if (resolved.backend === undefined) {
    // CPU is also the native default, so only an upgrade is written back.
    const rec = compatibleRecommendation(info, options.device);
    if (rec.backend !== "cpu") {
      resolved.backend = rec.backend;
      changed = true;
      notes.push(`Using the ${rec.backend} backend: ${rec.reason}`);
    }
  }
  // A context above the bundle's maximum can't be honored either way: a
  // fixed-context bundle gets clamped by the engine mid-session, and a
  // dynamic one is only configurable up to its maximum. Clamp up front so the
  // memory estimate and the engine agree.
  const effective = resolveMaxContextTokens(config);
  if (info.maxContextTokens > 0 && effective > info.maxContextTokens) {
    resolved.maxContextTokens = info.maxContextTokens;
    changed = true;
    const explicit = config?.maxContextTokens !== undefined || config?.maxTokens !== undefined;
    if (explicit) {
      notes.push(
        `maxContextTokens ${effective} exceeds this bundle's maximum of ` +
          `${info.maxContextTokens}; using ${info.maxContextTokens}.`,
      );
    }
  }
  if (config?.multimodal === true && !info.modalities.vision && !info.modalities.audio) {
    notes.push("multimodal: true, but this bundle has no vision or audio encoder — loading text-only.");
  }
  return { config: resolved, changed, notes };
}
