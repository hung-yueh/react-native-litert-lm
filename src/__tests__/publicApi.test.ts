/**
 * The public entry point: every documented export is present with the right
 * kind, and the deprecated helpers keep working while warning once.
 * Guards against an accidental export removal in a refactor.
 */
jest.mock("react-native", () => ({ Platform: { OS: "android" } }));
jest.mock("react-native-nitro-modules");

import * as api from "../index";
import * as core from "../core";
import { resetDeprecationWarnings } from "../core/deprecations";

const FUNCTIONS = [
  // inference + lifecycle
  "createLLM",
  "useModel",
  // embeddings
  "createEmbedder",
  "useEmbedder",
  "cosineSimilarity",
  "dotProduct",
  "normalize",
  "topK",
  "preflightEmbedder",
  // bundle introspection
  "inspectModel",
  "recommendBackend",
  "getDeviceInfo",
  "preflightLoad",
  "compareVersions",
  // memory
  "estimateMemory",
  "estimateKvCacheBytesPerToken",
  "forecastMemory",
  "evaluateBudget",
  "createMemoryTracker",
  "createNativeBuffer",
  // errors
  "isMemoryError",
  "toLiteRTLMError",
  // tool calling
  "parseToolCalls",
  "stripToolCalls",
  "toolResultPart",
  // streaming events
  "createStreamEventParser",
  // manifest resolution
  "resolveFromManifest",
  "resolutionFor",
  "resolveVariant",
  "mergeStreamChannels",
  "parseManifest",
  "fetchManifest",
  "manifestFetchStatus",
  "thinkingMarkers",
  "declaredChannels",
  // misc
  "setDeprecationWarningsEnabled",
  // deprecated, still exported until 1.0
  "getRecommendedBackend",
  "checkBackendSupport",
  "checkMultimodalSupport",
] as const;

const CLASSES = ["LiteRTLMError", "MemoryError"] as const;

describe("public API", () => {
  it.each(FUNCTIONS)("exports %s()", (name) => {
    expect(typeof (api as Record<string, unknown>)[name]).toBe("function");
  });

  it.each(CLASSES)("exports class %s", (name) => {
    const Ctor = (api as Record<string, any>)[name];
    expect(typeof Ctor).toBe("function");
    expect(Ctor.prototype).toBeInstanceOf(Error);
  });

  it("exports the constants", () => {
    expect(api.LITERT_LM_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
    expect(api.ESTIMATOR_CONSTANTS.SAFE_FRACTION).toBeGreaterThan(0);
    expect(api.DEFAULT_CHANNELS.length).toBeGreaterThan(0);
    expect(api.ModelRegistry.resolveModel).toBeInstanceOf(Function);
    expect(Object.values(api.Models)).toContain("gemma-4-E2B-it-litert-lm");
    expect(api.GEMMA_4_E2B_IT).toMatch(/^https:\/\//);
    expect(api.GEMMA_4_E4B_IT).toMatch(/^https:\/\//);
    expect(api.GEMMA_3N_E2B_IT_INT4).toMatch(/^https:\/\//);
    expect(api.EMBEDDING_GEMMA_2_TEXT_270M).toMatch(/embeddinggemma-2-text-270m\.litertlm$/);
    expect(api.EMBEDDING_GEMMA_2_740M).toMatch(/embeddinggemma-2-740m\.litertlm$/);
    expect(api.LFM2_5_1_2B_INSTRUCT).toMatch(/LFM2\.5-1\.2B-Instruct_int4\.litertlm$/);
    expect(api.EmbeddingTask.searchQuery("q")).toBe("task: search result | query: q");
    expect(api.EmbeddingTask.document("d")).toBe("title: none | text: d");
  });

  it("re-exports the react-native-free core", () => {
    for (const name of Object.keys(core)) {
      // index.ts wraps resolveFromManifest/recommendBackend with device defaults
      expect((api as Record<string, unknown>)[name]).toBeDefined();
    }
  });
});

describe("deprecated helpers", () => {
  let warn: jest.SpyInstance;
  beforeEach(() => {
    resetDeprecationWarnings();
    api.setDeprecationWarningsEnabled(true);
    warn = jest.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  it("keep their old behavior and warn once each", () => {
    expect(api.getRecommendedBackend()).toBe("cpu");
    expect(api.getRecommendedBackend()).toBe("cpu");
    expect(api.checkBackendSupport("gpu")).toContain("OpenCL");
    expect(api.checkBackendSupport("npu")).toContain("NPU");
    expect(api.checkBackendSupport("cpu")).toBeUndefined();
    expect(api.checkMultimodalSupport()).toBeUndefined();
    const messages = warn.mock.calls.map((c) => String(c[0]));
    expect(messages.filter((m) => m.includes("getRecommendedBackend()"))).toHaveLength(1);
    expect(messages.filter((m) => m.includes("checkBackendSupport()"))).toHaveLength(1);
    expect(messages.filter((m) => m.includes("checkMultimodalSupport()"))).toHaveLength(1);
  });
});
