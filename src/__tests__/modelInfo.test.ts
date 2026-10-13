/**
 * Bundle introspection: normalization of the native model-info reading,
 * device-aware backend recommendation, the load pre-flight, and their wiring
 * into inspectModel() / createLLM().loadModel().
 */
import {
  normalizeModelInfo,
  recommendBackend,
  preflightLoad,
  legacyMultimodalHint,
  type DeviceInfo,
} from "../core/modelInfo";
import { LITERT_LM_VERSION, compareVersions } from "../core/version";
import {
  warnDeprecated,
  setDeprecationWarningsEnabled,
  resetDeprecationWarnings,
} from "../core/deprecations";
import { LiteRTLMError } from "../core/errors";
import { inspectModel } from "../modelInspector";
import { createLLM } from "../modelFactory";
import {
  mockLiteRTLM,
  mockModelInspector,
  mockModelStore,
  mockNativeModelInfo,
} from "../__mocks__/react-native-nitro-modules";

jest.mock("react-native-nitro-modules");

type NativeInfo = ReturnType<typeof mockNativeModelInfo>;
type Backend = "cpu" | "gpu" | "npu";

function native(patch: Omit<Partial<NativeInfo>, "modalities"> & {
  modalities?: Partial<Record<"text" | "vision" | "audio" | "video", {
    supported?: boolean;
    backends?: Backend[];
    npuBrand?: string;
    socName?: string;
  }>>;
} = {}): NativeInfo {
  const base = mockNativeModelInfo();
  const { modalities, ...rest } = patch;
  return {
    ...base,
    ...rest,
    modalities: base.modalities.map((m) => ({ ...m, ...(modalities?.[m.modality] ?? {}) })) as NativeInfo["modalities"],
  };
}

/** Shaped like the Gemma 4 E2B bundle: multimodal, GPU-preferred, thinking + tools. */
const gemmaLike = () =>
  native({
    maxContextTokens: 32768,
    isDynamicContext: true,
    supportsThinking: true,
    supportsFunctionCalling: true,
    samplerType: 1,
    samplerTemperature: 1,
    samplerTopK: 64,
    samplerTopP: 0.95,
    maxVisionTokenBudget: 1120,
    visionSignatureSelection: [70, 140, 280, 560, 1120],
    modalities: {
      text: { supported: true, backends: ["gpu", "cpu"] },
      vision: { supported: true, backends: ["gpu", "cpu"] },
      audio: { supported: true, backends: ["cpu"] },
    },
  });

/** Shaped like embeddinggemma-2-740m_Qualcomm_SM8750.litertlm. */
const npuEmbedder = () =>
  native({
    modelType: "embedding" as any,
    embeddingDimension: 768,
    embeddingSignatureSelection: [128, 256, 512, 1024, 2048, 8192],
    modalities: {
      text: { supported: true, backends: ["npu"], npuBrand: "qualcomm", socName: "SM8750" },
    },
  });

const ios: DeviceInfo = {
  platform: "ios",
  osVersion: "26.1",
  model: "iPhone18,1",
  totalMemoryBytes: 8e9,
  gpuAvailable: true,
  isEmulator: false,
};
const pixel: DeviceInfo = {
  platform: "android",
  osVersion: "16",
  model: "Google Pixel 9",
  socModel: "Tensor G4",
  totalMemoryBytes: 12e9,
  gpuAvailable: true,
  isEmulator: false,
};
const galaxy: DeviceInfo = { ...pixel, model: "samsung SM-S931U", socModel: "SM8750", gpuAvailable: false };

describe("normalizeModelInfo", () => {
  it("normalizes an LLM bundle", () => {
    const info = normalizeModelInfo("/m/gemma.litertlm", gemmaLike());
    expect(info.type).toBe("llm");
    expect(info.modalities).toEqual({ text: true, vision: true, audio: true, video: false });
    expect(info.backends.text).toEqual(["gpu", "cpu"]);
    expect(info.backends.audio).toEqual(["cpu"]);
    expect(info.maxContextTokens).toBe(32768);
    expect(info.dynamicContext).toBe(true);
    expect(info.llm).toEqual({
      supportsThinking: true,
      supportsFunctionCalling: true,
      speculativeDecoding: false,
      samplerDefaults: { type: "topK", temperature: 1, topK: 64, topP: 0.95 },
      maxVisionTokenBudget: 1120,
      visionTokenOptions: [70, 140, 280, 560, 1120],
    });
    expect(info.embedding).toBeUndefined();
    expect(info.npu).toEqual({});
  });

  it("reports undeclared fields as absent, not as zeros", () => {
    const info = normalizeModelInfo("/m/old.litertlm", native());
    expect(info.llm?.samplerDefaults).toBeNull();
    expect(info.llm?.maxVisionTokenBudget).toBeNull();
    expect(info.minRuntimeVersion).toBeNull();
    expect(info.maxContextTokens).toBe(0);
  });

  it("normalizes an NPU embedding bundle", () => {
    const info = normalizeModelInfo("/m/egemma.litertlm", npuEmbedder());
    expect(info.type).toBe("embedding");
    expect(info.llm).toBeUndefined();
    expect(info.embedding).toEqual({
      dimension: 768,
      inputLengthOptions: [128, 256, 512, 1024, 2048, 8192],
      maxVisionTokenBudget: null,
      visionTokenOptions: [],
    });
    expect(info.npu.text).toEqual({ brand: "qualcomm", socName: "SM8750" });
  });
});

describe("recommendBackend", () => {
  const gemma = normalizeModelInfo("/m/gemma.litertlm", gemmaLike());

  it("defaults to CPU when the bundle declares nothing", () => {
    const rec = recommendBackend(normalizeModelInfo("/m/x.litertlm", native()), { device: ios });
    expect(rec).toMatchObject({ backend: "cpu", candidates: ["cpu"], compatible: true });
  });

  it("follows the bundle's preference when the device can run it", () => {
    expect(recommendBackend(gemma, { device: ios })).toMatchObject({
      backend: "gpu",
      candidates: ["gpu", "cpu"],
    });
  });

  it("drops GPU without a driver", () => {
    const rec = recommendBackend(gemma, { device: galaxy });
    expect(rec.backend).toBe("cpu");
    expect(rec.reason).toContain("no OpenCL driver");
  });

  it("puts CPU first when asked for compatibility", () => {
    expect(recommendBackend(gemma, { device: ios, prefer: "compatibility" }).candidates).toEqual([
      "cpu",
      "gpu",
    ]);
  });

  it("uses an NPU build only on its SoC", () => {
    const egemma = normalizeModelInfo("/m/egemma.litertlm", npuEmbedder());
    expect(recommendBackend(egemma, { device: galaxy })).toMatchObject({
      backend: "npu",
      compatible: true,
    });
    const elsewhere = recommendBackend(egemma, { device: pixel });
    expect(elsewhere.compatible).toBe(false);
    expect(elsewhere.candidates).toEqual([]);
    expect(elsewhere.reason).toContain("built for SM8750, device is Tensor G4");
  });

  it("never picks NPU on iOS", () => {
    const egemma = normalizeModelInfo("/m/egemma.litertlm", npuEmbedder());
    expect(recommendBackend(egemma, { device: ios }).compatible).toBe(false);
  });

  it("without device facts keeps GPU but skips SoC-bound NPU builds", () => {
    expect(recommendBackend(gemma).backend).toBe("gpu");
    const egemma = normalizeModelInfo("/m/egemma.litertlm", npuEmbedder());
    expect(recommendBackend(egemma).compatible).toBe(false);
  });
});

describe("preflightLoad", () => {
  const gemma = normalizeModelInfo("/m/gemma.litertlm", gemmaLike());

  it("refuses an embedding bundle in createLLM", () => {
    const egemma = normalizeModelInfo("/m/egemma.litertlm", npuEmbedder());
    expect(() => preflightLoad(egemma, undefined)).toThrow(
      expect.objectContaining({ code: "MODEL_TYPE_MISMATCH", message: expect.stringContaining("createEmbedder()") }),
    );
  });

  it("allows bundles of unknown type (older conversions)", () => {
    const old = normalizeModelInfo("/m/old.litertlm", native({ modelType: "unknown" as any }));
    expect(() => preflightLoad(old, undefined)).not.toThrow();
  });

  it("refuses a bundle that needs a newer runtime", () => {
    const future = normalizeModelInfo("/m/f.litertlm", native({ minRuntimeVersion: "0.19.0" }));
    expect(() => preflightLoad(future, undefined, { runtimeVersion: "0.18.0" })).toThrow(
      expect.objectContaining({ code: "RUNTIME_TOO_OLD" }),
    );
    const ok = normalizeModelInfo("/m/ok.litertlm", native({ minRuntimeVersion: "0.18.0" }));
    expect(() => preflightLoad(ok, undefined, { runtimeVersion: "0.18.0" })).not.toThrow();
  });

  it("fills the backend the bundle prefers, and only then", () => {
    const pf = preflightLoad(gemma, { temperature: 0.2 }, { device: ios });
    expect(pf.config).toEqual({ temperature: 0.2, backend: "gpu" });
    expect(pf.changed).toBe(true);
    expect(pf.notes[0]).toContain("Using the gpu backend");

    const explicit = preflightLoad(gemma, { backend: "cpu" }, { device: ios });
    expect(explicit.changed).toBe(false);
    expect(explicit.config.backend).toBe("cpu");

    const undeclared = preflightLoad(normalizeModelInfo("/m/x", native()), undefined, { device: ios });
    expect(undeclared.changed).toBe(false);
    expect(undeclared.config.backend).toBeUndefined();
  });

  it("refuses a bundle this device can't run, unless a backend was chosen", () => {
    const npuOnlyLlm = normalizeModelInfo(
      "/m/qwen_SM8750.litertlm",
      native({ modalities: { text: { supported: true, backends: ["npu"], npuBrand: "qualcomm", socName: "SM8750" } } }),
    );
    expect(() => preflightLoad(npuOnlyLlm, undefined, { device: pixel })).toThrow(
      expect.objectContaining({ code: "UNSUPPORTED", message: expect.stringContaining("built for SM8750") }),
    );
    // An explicit choice is the caller's call.
    expect(preflightLoad(npuOnlyLlm, { backend: "cpu" }, { device: pixel }).changed).toBe(false);
  });

  it("clamps the context to the bundle's maximum", () => {
    const small = normalizeModelInfo("/m/s.litertlm", native({ maxContextTokens: 2048 }));
    const explicit = preflightLoad(small, { maxContextTokens: 8192 });
    expect(explicit.config.maxContextTokens).toBe(2048);
    expect(explicit.notes[0]).toContain("exceeds this bundle's maximum of 2048");

    // The implicit 4096 default is clamped silently
    const implicit = preflightLoad(small, undefined);
    expect(implicit.config.maxContextTokens).toBe(2048);
    expect(implicit.notes).toEqual([]);

    // Legacy maxTokens counts as a request
    expect(preflightLoad(small, { maxTokens: 4096 }).notes).toHaveLength(1);

    // Within the limit: untouched
    expect(preflightLoad(gemma, { backend: "cpu", maxContextTokens: 8192 }).changed).toBe(false);
  });
});

describe("multimodal encoders", () => {
  it("notes multimodal: true on a text-only bundle", () => {
    const textOnly = normalizeModelInfo("/m/t.litertlm", native());
    expect(preflightLoad(textOnly, { multimodal: true }).notes[0]).toContain("no vision or audio encoder");
  });

  it("recognizes 0.7's file-name default (deprecated)", () => {
    expect(legacyMultimodalHint("/cache/gemma-3n-E2B-it-int4.litertlm")).toBe(true);
    expect(legacyMultimodalHint("/cache/Gemma3-1B-IT.litertlm")).toBe(true);
    expect(legacyMultimodalHint("/cache/gemma-4-E2B-it.litertlm")).toBe(false);
    expect(legacyMultimodalHint("/3n-models/qwen.litertlm")).toBe(false);
  });

  it("warns only when an unreadable bundle falls back to the file-name default", async () => {
    resetDeprecationWarnings();
    setDeprecationWarningsEnabled(true);
    const warn = jest.spyOn(console, "warn").mockImplementation(() => {});
    const fileNameWarning = () =>
      warn.mock.calls.some((c) => String(c[0]).includes("from the model's file name"));
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);

    // Readable metadata: the bundle decides, no warning.
    mockModelInspector.inspect.mockResolvedValueOnce(
      native({ modalities: { vision: { supported: true }, audio: { supported: true } } }),
    );
    await createLLM().loadModel("/m/gemma-3n-E2B-it-int4.litertlm");
    expect(fileNameWarning()).toBe(false);

    // Unreadable metadata + a Gemma 3n name: the deprecated fallback applies.
    mockModelInspector.inspect.mockRejectedValueOnce(new Error("[LiteRTLM:inspect:DATA_LOSS] old"));
    await createLLM().loadModel("/m/gemma-3n-E2B-it-int4.litertlm");
    expect(fileNameWarning()).toBe(true);

    // An explicit choice never warns.
    warn.mockClear();
    resetDeprecationWarnings();
    mockModelInspector.inspect.mockRejectedValueOnce(new Error("[LiteRTLM:inspect:DATA_LOSS] old"));
    await createLLM().loadModel("/m/gemma-3n-E2B-it-int4.litertlm", { multimodal: true });
    expect(fileNameWarning()).toBe(false);
    warn.mockRestore();
  });
});

describe("version", () => {
  it("compares dotted versions", () => {
    expect(compareVersions("0.18.0", "0.17.1")).toBeGreaterThan(0);
    expect(compareVersions("v0.18", "0.18.0")).toBe(0);
    expect(compareVersions("0.9.0", "0.18.0")).toBeLessThan(0);
    expect(compareVersions("0.18.0-rc1", "0.18.0")).toBe(0);
  });

  it("matches the engine pinned in package.json", () => {
    expect(LITERT_LM_VERSION).toBe(require("../../package.json").litertLm.version);
  });
});

describe("deprecations", () => {
  let warn: jest.SpyInstance;
  beforeEach(() => {
    resetDeprecationWarnings();
    setDeprecationWarningsEnabled(true);
    warn = jest.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("warns once per API", () => {
    warnDeprecated("old()", "new()");
    warnDeprecated("old()", "new()");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toContain("old() is deprecated and will be removed in 1.0. Use new() instead.");
  });

  it("can be silenced", () => {
    setDeprecationWarningsEnabled(false);
    warnDeprecated("old()", "new()");
    expect(warn).not.toHaveBeenCalled();
    setDeprecationWarningsEnabled(true);
  });

  it("legacy inference methods warn once and still work", async () => {
    const llm = createLLM();
    await llm.sendMessage("a");
    await llm.sendMessage("b");
    const calls = warn.mock.calls.filter((c) => String(c[0]).includes("llm.sendMessage()"));
    expect(calls).toHaveLength(1);
  });

  it("LLMConfig.maxTokens warns", async () => {
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
    await createLLM().loadModel("/local/m.litertlm", { maxTokens: 1024 });
    expect(warn.mock.calls.some((c) => String(c[0]).includes("LLMConfig.maxTokens"))).toBe(true);
  });
});

describe("inspectModel", () => {
  beforeEach(() => jest.clearAllMocks());

  it("normalizes a local bundle", async () => {
    mockModelInspector.inspect.mockResolvedValueOnce(gemmaLike());
    const info = await inspectModel("file:///m/gemma.litertlm");
    expect(mockModelInspector.inspect).toHaveBeenCalledWith("/m/gemma.litertlm");
    expect(info.path).toBe("/m/gemma.litertlm");
    expect(info.backends.text).toEqual(["gpu", "cpu"]);
  });

  it("resolves a downloaded URL to its cached file, and never downloads", async () => {
    mockModelStore.isCached.mockReturnValueOnce(true);
    await inspectModel("https://example.com/models/gemma.litertlm");
    expect(mockModelInspector.inspect).toHaveBeenCalledWith("/mock/path/gemma.litertlm");

    mockModelStore.isCached.mockReturnValueOnce(false);
    await expect(inspectModel("https://example.com/models/other.litertlm")).rejects.toMatchObject({
      code: "MODEL_NOT_FOUND",
    });
    expect(mockModelStore.downloadFile).not.toHaveBeenCalled();
  });

  it("types native failures", async () => {
    mockModelInspector.inspect.mockRejectedValueOnce(
      new Error("[LiteRTLM:inspect:INVALID_ARGUMENT] Not a readable LiteRT-LM bundle: Invalid magic number"),
    );
    await expect(inspectModel("/m/junk.litertlm")).rejects.toMatchObject({ code: "INVALID_MODEL" });
  });
});

describe("createLLM().loadModel runs the bundle pre-flight", () => {
  let warn: jest.SpyInstance;
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
    warn = jest.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("refuses an embedding bundle before touching the engine", async () => {
    mockModelInspector.inspect.mockResolvedValueOnce(npuEmbedder());
    const err = await createLLM().loadModel("/m/egemma.litertlm").catch((e) => e);
    expect(err).toBeInstanceOf(LiteRTLMError);
    expect(err.code).toBe("MODEL_TYPE_MISMATCH");
    expect(mockLiteRTLM.loadModel).not.toHaveBeenCalled();
  });

  it("loads with the backend the bundle prefers and exposes the model info", async () => {
    mockModelInspector.inspect.mockResolvedValueOnce(gemmaLike());
    const llm = createLLM();
    await llm.loadModel("/m/gemma.litertlm", { temperature: 0.5 });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledWith("/m/gemma.litertlm", {
      temperature: 0.5,
      backend: "gpu",
    });
    expect(llm.getModelInfo()?.llm?.supportsThinking).toBe(true);
    await llm.unload();
    expect(llm.getModelInfo()).toBeNull();
  });

  it("passes the caller's config through untouched when nothing needs resolving", async () => {
    const config = { backend: "cpu" as const };
    await createLLM().loadModel("/m/x.litertlm", config);
    expect(mockLiteRTLM.loadModel.mock.calls[0][1]).toBe(config);
  });

  it("falls back to a plain load when the metadata can't be read", async () => {
    mockModelInspector.inspect.mockRejectedValueOnce(
      new Error("[LiteRTLM:inspect:DATA_LOSS] truncated metadata"),
    );
    const llm = createLLM();
    await llm.loadModel("/m/old.litertlm");
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledWith("/m/old.litertlm", undefined);
    expect(llm.getModelInfo()).toBeNull();
    expect(warn.mock.calls.some((c) => String(c[0]).includes("loading without the bundle pre-flight"))).toBe(true);
  });

  it("fails fast on a missing file", async () => {
    mockModelInspector.inspect.mockRejectedValueOnce(
      new Error("[LiteRTLM:inspect:NOT_FOUND:MODEL_NOT_FOUND] Model file not found: /m/none.litertlm"),
    );
    const err = await createLLM().loadModel("/m/none.litertlm").catch((e) => e);
    expect(err.code).toBe("MODEL_NOT_FOUND");
    expect(mockLiteRTLM.loadModel).not.toHaveBeenCalled();
  });
});
