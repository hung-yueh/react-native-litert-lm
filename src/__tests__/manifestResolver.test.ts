import {
  declaredChannels,
  fetchManifest,
  mergeStreamChannels,
  parseManifest,
  resolutionFor,
  resolveFromManifest,
  resolveVariant,
  thinkingMarkers,
  type Manifest,
} from "../manifestResolver";
import { DEFAULT_CHANNELS } from "../streamEvents";

/**
 * Trimmed from the live litert-community/LFM2.5-1.2B-Instruct manifest: a
 * cpu-only int4 (smaller, android-recommended), a cpu+gpu re-export
 * (android-gpu-recommended), and a cpu-only int8 (macos-recommended). The
 * backend-loss shape from #23 lives in the first two: both carry an android
 * recommendation and the cpu-only file is smaller.
 */
const lfmLike = (): Manifest =>
  parseManifest({
    manifest_schema: "0.1.0",
    repo: "litert-community/LFM2.5-1.2B-Instruct",
    generated: "2026-08-24",
    model: {
      display_name: "LFM2.5-1.2B-Instruct",
      context_length: 4096,
      capabilities: {
        thinking: { declared: true, channel: { start: "<think>", end: "</think>" } },
      },
      session_defaults: { max_output_tokens_min: 2048, temperature: 0.3, top_k: 40 },
    },
    variants: [
      {
        file: "LFM2.5-1.2B-Instruct_int4.litertlm",
        sha256: "a".repeat(64),
        size_bytes: 736_015_744,
        quantization: "int4",
        backends: ["cpu"],
        default_backend: "cpu",
        recommended: [
          { platform: "android", device_class: "midrange", backend: "cpu", reason: "int4 decodes faster" },
          { platform: "ios", backend: "cpu", reason: "verified path" },
        ],
      },
      {
        file: "LFM2.5-1.2B-Instruct_int4_gpu.litertlm",
        sha256: "b".repeat(64),
        size_bytes: 736_220_768,
        quantization: "int4 gpu re-export",
        backends: ["cpu", "gpu"],
        default_backend: "gpu",
        recommended: [
          { platform: "android", device_class: "midrange-2023+", backend: "gpu", reason: "3-6x prefill" },
          { platform: "macos", backend: "gpu", reason: "~11x prefill" },
        ],
        known_issues: ["iPhone Metal fails on this family — use CPU on iOS"],
      },
      {
        file: "LFM2.5-1.2B-Instruct_int8.litertlm",
        sha256: "c".repeat(64),
        size_bytes: 1_247_091_440,
        quantization: "int8",
        backends: ["cpu"],
        default_backend: "cpu",
        recommended: [{ platform: "macos", backend: "cpu", reason: "highest fidelity" }],
      },
    ],
  });

describe("resolveVariant", () => {
  it("keeps an explicit gpu request across the variant pick (#23 repro)", () => {
    const r = resolveVariant(lfmLike(), { platform: "android", backend: "gpu" });
    expect(r?.file).toBe("LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
    expect(r?.backend).toBe("gpu");
  });

  it("returns null when no variant lists the requested backend", () => {
    expect(resolveVariant(lfmLike(), { backend: "npu" })).toBeNull();
    expect(resolveVariant(lfmLike(), { platform: "android", backend: "npu" })).toBeNull();
  });

  it("counts only recommendations naming the requested backend", () => {
    // The gpu re-export carries a macos/gpu recommendation; for a cpu request
    // the macos/cpu-recommended int8 must win, not the gpu file on cpu.
    const r = resolveVariant(lfmLike(), { platform: "macos", backend: "cpu" });
    expect(r?.file).toBe("LFM2.5-1.2B-Instruct_int8.litertlm");
    expect(r?.backend).toBe("cpu");
  });

  it("follows platform recommendations when no backend is requested", () => {
    const r = resolveVariant(lfmLike(), { platform: "android", deviceClass: "midrange-2023+" });
    expect(r?.file).toBe("LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
    expect(r?.backend).toBe("gpu");
  });

  it("notes an unmatched deviceClass in reason instead of silently upgrading", () => {
    const r = resolveVariant(lfmLike(), { platform: "android", deviceClass: "budget-2019" });
    expect(r?.reason).toContain("no budget-2019 entry; using the midrange recommendation");
  });

  it("builds URLs from the fetched revision", () => {
    const m = lfmLike();
    m.revision = "abc123";
    expect(resolveVariant(m)?.url).toContain("/resolve/abc123/");
    expect(resolveVariant(m, { revision: "deadbeef" })?.url).toContain("/resolve/deadbeef/");
    expect(resolveVariant(lfmLike())?.url).toContain("/resolve/main/");
  });
});

describe("parseManifest", () => {
  it("rejects non-0.1 schemas and empty backends", () => {
    expect(() => parseManifest({})).toThrow(/not a litertlm_manifest/);
    const m = lfmLike() as unknown as { manifest_schema: string };
    expect(() => parseManifest({ ...lfmLike(), manifest_schema: "0.2.0" })).toThrow(
      /unsupported manifest_schema/,
    );
    expect(m).toBeDefined();
    expect(() =>
      parseManifest({
        manifest_schema: "0.1.0",
        repo: "t/x",
        generated: "2026-08-27",
        model: { display_name: "X" },
        variants: [{ file: "a.litertlm", quantization: "q", backends: [] }],
      }),
    ).toThrow(/no backends/);
  });
});

describe("mergeStreamChannels", () => {
  it("keeps the default toolCall channel when the manifest declares only thinking", () => {
    const channels = mergeStreamChannels(lfmLike());
    expect(channels).toContainEqual(
      DEFAULT_CHANNELS.find((c) => c.type === "toolCall"),
    );
    expect(channels).toContainEqual({ type: "thinking", start: "<think>", end: "</think>" });
  });

  it("lets a 0.1.1 declared tool-call channel override the default markers", () => {
    const m = lfmLike();
    m.model.capabilities = {
      ...m.model.capabilities,
      channels: [
        { name: "thought", start: "<think>", end: "</think>", is_reasoning: true },
        { name: "tool_call", start: "<function>", end: "</function>" },
      ],
    };
    const channels = mergeStreamChannels(m);
    expect(channels).toContainEqual({ type: "toolCall", start: "<function>", end: "</function>" });
    expect(channels).toContainEqual({ type: "thinking", start: "<think>", end: "</think>" });
    expect(declaredChannels(m)).toHaveLength(2);
  });

  it("returns the defaults untouched for a manifest declaring nothing", () => {
    const m = lfmLike();
    m.model.capabilities = {};
    expect(mergeStreamChannels(m)).toEqual(DEFAULT_CHANNELS);
    expect(thinkingMarkers(m)).toBeUndefined();
  });
});

describe("resolutionFor", () => {
  it("maps session_defaults into an LLMConfig fragment", () => {
    const r = resolutionFor(lfmLike(), { platform: "android", backend: "gpu" });
    expect(r).not.toBeNull();
    expect(r?.backend).toBe("gpu");
    expect(r?.config).toEqual({
      backend: "gpu",
      maxOutputTokens: 2048,
      temperature: 0.3,
      topK: 40,
    });
    expect(r?.streamChannels.map((c) => c.type).sort()).toEqual(["thinking", "toolCall"]);
    expect(r?.sha256).toBe("b".repeat(64));
    expect(r?.notes).toContain("iPhone Metal fails on this family — use CPU on iOS");
    expect(r?.contextLength).toBe(4096);
  });

  it("propagates null for an unsupported explicit backend", () => {
    expect(resolutionFor(lfmLike(), { backend: "npu" })).toBeNull();
  });
});

describe("resolveFromManifest / fetchManifest", () => {
  const manifestBody = JSON.stringify(lfmLike());
  let fetchSpy: jest.SpyInstance;

  beforeEach(() => {
    fetchSpy = jest
      .spyOn(globalThis, "fetch")
      .mockResolvedValue({ ok: true, text: () => Promise.resolve(manifestBody) } as Response);
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("fetches, resolves, and pins URLs to the fetched revision", async () => {
    const r = await resolveFromManifest("litert-community/LFM2.5-1.2B-Instruct", {
      platform: "android",
      backend: "gpu",
      revision: "abc123",
    });
    expect(fetchSpy).toHaveBeenCalledWith(
      "https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/abc123/litertlm_manifest.json",
    );
    expect(r?.file).toBe("LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
    expect(r?.url).toContain("/resolve/abc123/");
  });

  it("throws with the URL and status when the manifest is missing", async () => {
    fetchSpy.mockResolvedValue({ ok: false, status: 404 } as Response);
    await expect(fetchManifest("litert-community/NoManifest")).rejects.toThrow(/HTTP 404/);
  });
});
