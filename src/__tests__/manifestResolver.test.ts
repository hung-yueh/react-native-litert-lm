import {
  declaredChannels,
  fetchManifest,
  manifestFetchStatus,
  manifestPlatformFor,
  mergeStreamChannels,
  parseManifest,
  resolutionFor,
  resolveFromManifest,
  resolveVariant,
  thinkingMarkers,
  type Manifest,
} from "../manifestResolver";
import { createStreamEventParser, DEFAULT_CHANNELS } from "../streamEvents";

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
      session_defaults: {
        max_output_tokens_min: 2048,
        temperature: 0.3,
        top_k: 40,
        notes: "Reasoning model: keep maxOutputTokens ≥ 2048 so it can finish thinking.",
      },
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
    expect(() => parseManifest({ ...lfmLike(), manifest_schema: "0.2.0" })).toThrow(
      /unsupported manifest_schema/,
    );
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

  it("rejects a manifest without a repo instead of building undefined URLs", () => {
    const { repo: _repo, ...withoutRepo } = lfmLike();
    expect(() => parseManifest(withoutRepo)).toThrow(/no repo/);
  });

  it("rejects a variant without a file instead of a later TypeError", () => {
    expect(() =>
      parseManifest({
        manifest_schema: "0.1.0",
        repo: "t/x",
        generated: "2026-08-27",
        model: { display_name: "X" },
        variants: [{ quantization: "q", backends: ["cpu"] }],
      }),
    ).toThrow(/has no file/);
  });
});

describe("resolveVariant on hand-built manifests", () => {
  it("never fabricates a backend for a variant listing none", () => {
    // Bypasses parseManifest deliberately: the type technically permits
    // backends: [], and the resolver must skip such variants, not pick cpu.
    const hand: Manifest = {
      ...lfmLike(),
      variants: [
        { file: "broken.litertlm", quantization: "q", backends: [] },
        { file: "ok.litertlm", quantization: "q", backends: ["gpu"] },
      ],
    };
    const r = resolveVariant(hand);
    expect(r?.file).toBe("ok.litertlm");
    expect(r?.backend).toBe("gpu");

    hand.variants = [{ file: "broken.litertlm", quantization: "q", backends: [] }];
    expect(resolveVariant(hand)).toBeNull();
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

  it("keeps capabilities.thinking markers when channels declares only tool_call", () => {
    // A 0.1.1 manifest may declare a partial channel list; the thinking
    // markers in capabilities must still land or reasoning leaks into tokens.
    const m = lfmLike();
    m.model.capabilities = {
      ...m.model.capabilities,
      channels: [{ name: "tool_call", start: "<function>", end: "</function>" }],
    };
    const channels = mergeStreamChannels(m);
    expect(channels).toContainEqual({ type: "toolCall", start: "<function>", end: "</function>" });
    expect(channels).toContainEqual({ type: "thinking", start: "<think>", end: "</think>" });
  });

  it("returns the defaults untouched for a manifest declaring nothing", () => {
    const m = lfmLike();
    m.model.capabilities = {};
    expect(mergeStreamChannels(m)).toEqual(DEFAULT_CHANNELS);
    expect(thinkingMarkers(m)).toBeUndefined();
  });

  // From the #25 review: the whitespace-carrying markers must meet the real
  // parser, which is where the original bug lived.
  it("parses a manifest's whitespace-carrying markers end to end", () => {
    const qwenLike = parseManifest({
      manifest_schema: "0.1.0",
      repo: "litert-community/Qwen3-4B-Thinking-2507",
      generated: "2026-08-24",
      model: {
        display_name: "Qwen3-4B-Thinking",
        capabilities: {
          thinking: { declared: true, channel: { start: "<think>\n", end: "\n</think>" } },
        },
      },
      variants: [{ file: "model.litertlm", quantization: "int4", backends: ["cpu"] }],
    });
    const parser = createStreamEventParser(resolutionFor(qwenLike, {})!.streamChannels);
    const events = [
      ...parser.push("<think>\nlet me reason"),
      ...parser.push(" about this\n</think>The answer is 4."),
      ...parser.finish(),
    ];
    expect(events.filter((e) => e.type === "thinking").map((e) => e.text).join(""))
      .toBe("let me reason about this");
    expect(events.filter((e) => e.type === "token").map((e) => e.text).join(""))
      .toBe("The answer is 4.");
  });
});

describe("manifestPlatformFor", () => {
  it("forwards every OS a manifest can name and drops the rest", () => {
    // react-native-macos / -windows must keep their recommendations (#25 review).
    for (const os of ["android", "ios", "macos", "windows"]) {
      expect(manifestPlatformFor(os)).toBe(os);
    }
    expect(manifestPlatformFor("web")).toBeUndefined();
    expect(manifestPlatformFor("native")).toBeUndefined();
  });

  it("changes which variant wins for a macos caller", () => {
    // No platform → smallest file; macos → the gpu re-export it recommends.
    expect(resolutionFor(lfmLike(), { platform: manifestPlatformFor("web") })?.file).toBe(
      "LFM2.5-1.2B-Instruct_int4.litertlm",
    );
    const mac = resolutionFor(lfmLike(), { platform: manifestPlatformFor("macos") });
    expect(mac?.file).toBe("LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
    expect(mac?.reason).toContain("recommended for macos");
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

  it("surfaces session_defaults.notes after the variant's notes, and the raw object", () => {
    const m = lfmLike();
    const r = resolutionFor(m, { platform: "android", backend: "gpu" });
    expect(r?.notes).toEqual([
      "iPhone Metal fails on this family — use CPU on iOS",
      "Reasoning model: keep maxOutputTokens ≥ 2048 so it can finish thinking.",
    ]);
    // The raw object rides along so a key a later 0.1.x adds needs no re-fetch.
    expect(r?.sessionDefaults).toBe(m.model.session_defaults);

    m.model.session_defaults = { notes: "" };
    expect(resolutionFor(m, { platform: "android", backend: "gpu" })?.notes).toEqual([
      "iPhone Metal fails on this family — use CPU on iOS",
    ]);
    delete m.model.session_defaults;
    const bare = resolutionFor(m, { platform: "android", backend: "gpu" });
    expect(bare?.sessionDefaults).toBeUndefined();
    expect(bare?.notes).toEqual(["iPhone Metal fails on this family — use CPU on iOS"]);
  });

  it("propagates null for an unsupported explicit backend", () => {
    expect(resolutionFor(lfmLike(), { backend: "npu" })).toBeNull();
  });
});

describe("resolveFromManifest / fetchManifest", () => {
  const manifestBody = JSON.stringify(lfmLike());
  const okResponse = (body: string) =>
    ({ ok: true, text: () => Promise.resolve(body) }) as Response;
  let fetchSpy: jest.SpyInstance;
  let warnSpy: jest.SpyInstance;

  beforeEach(() => {
    fetchSpy = jest.spyOn(globalThis, "fetch").mockResolvedValue(okResponse(manifestBody));
    warnSpy = jest.spyOn(console, "warn").mockImplementation(() => {});
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    warnSpy.mockRestore();
  });

  it("fetches, resolves, and pins URLs to the fetched revision", async () => {
    const r = await resolveFromManifest("litert-community/LFM2.5-1.2B-Instruct", {
      platform: "android",
      backend: "gpu",
      revision: "abc123",
    });
    expect(fetchSpy).toHaveBeenCalledWith(
      "https://huggingface.co/litert-community/LFM2.5-1.2B-Instruct/resolve/abc123/litertlm_manifest.json",
      { signal: undefined },
    );
    expect(r?.file).toBe("LFM2.5-1.2B-Instruct_int4_gpu.litertlm");
    expect(r?.url).toContain("/resolve/abc123/");
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("builds URLs from the repo it fetched from, not the manifest's own claim", async () => {
    // A fork's copied manifest still names the origin repo; downloads must
    // follow the fork the caller actually asked for.
    const r = await resolveFromManifest("myorg/LFM2.5-fork");
    expect(r?.url).toContain("https://huggingface.co/myorg/LFM2.5-fork/resolve/");
    expect(r?.url).not.toContain("litert-community");

    const m = await fetchManifest("myorg/LFM2.5-fork");
    expect(m.repo).toBe("myorg/LFM2.5-fork");
  });

  it("fetchManifest itself still throws, carrying the URL and status", async () => {
    fetchSpy.mockResolvedValue({ ok: false, status: 404 } as Response);
    const missing = fetchManifest("litert-community/NoManifest");
    await expect(missing).rejects.toThrow(/no litertlm_manifest\.json at .*NoManifest.*HTTP 404/);
    expect(manifestFetchStatus(await missing.catch((e) => e))).toBe(404);

    // A gated repo is not a missing manifest — say what happened.
    fetchSpy.mockResolvedValue({ ok: false, status: 403 } as Response);
    const gated = fetchManifest("t/x");
    await expect(gated).rejects.toThrow(/^HTTP 403 fetching /);
    expect(manifestFetchStatus(await gated.catch((e) => e))).toBe(403);
    expect(manifestFetchStatus(new Error("plain"))).toBeUndefined();
    expect(manifestFetchStatus("boom")).toBeUndefined();
  });

  // The null cases agreed in #23/#25 — missing manifest, unknown schema,
  // network failure, unresolvable variant. Most Hub repos ship no manifest
  // yet, so the common path must not need a try/catch at the call site —
  // and must not put a LogBox warning on every dev launch either.
  it("returns null, silently, when the repo has no manifest", async () => {
    fetchSpy.mockResolvedValue({ ok: false, status: 404 } as Response);
    await expect(resolveFromManifest("litert-community/NoManifest")).resolves.toBeNull();
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("returns null, with one warning, for any other HTTP status", async () => {
    fetchSpy.mockResolvedValue({ ok: false, status: 403 } as Response);
    await expect(resolveFromManifest("t/x")).resolves.toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toMatch(/resolveFromManifest\(t\/x\): HTTP 403 fetching /);
  });

  it("returns null when the manifest's schema is newer than this reader", async () => {
    fetchSpy.mockResolvedValue(
      okResponse(JSON.stringify({ ...lfmLike(), manifest_schema: "0.2.0" })),
    );
    await expect(resolveFromManifest("t/x")).resolves.toBeNull();
    expect(warnSpy.mock.calls[0][0]).toMatch(/unsupported manifest_schema 0\.2\.0/);
  });

  it("returns null when the fetch fails or the body is not a manifest", async () => {
    fetchSpy.mockRejectedValue(new TypeError("Network request failed"));
    await expect(resolveFromManifest("t/x")).resolves.toBeNull();
    expect(warnSpy.mock.calls[0][0]).toMatch(/Network request failed/);

    fetchSpy.mockResolvedValue(okResponse("<html>not json</html>"));
    await expect(resolveFromManifest("t/x")).resolves.toBeNull();

    fetchSpy.mockRejectedValue("boom");
    await expect(resolveFromManifest("t/x")).resolves.toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(3);
    expect(warnSpy.mock.calls[2][0]).toMatch(/boom/);
  });

  it("forwards an AbortSignal and treats an aborted fetch as null, silently", async () => {
    const controller = new AbortController();
    await resolveFromManifest("t/x", { signal: controller.signal });
    expect(fetchSpy).toHaveBeenLastCalledWith(expect.any(String), { signal: controller.signal });

    // The caller asked for the abort (a startup timeout, an unmount) — no warning.
    fetchSpy.mockRejectedValue(Object.assign(new Error("Aborted"), { name: "AbortError" }));
    controller.abort();
    await expect(resolveFromManifest("t/x", { signal: controller.signal })).resolves.toBeNull();
    expect(warnSpy).not.toHaveBeenCalled();

    // A rejection while the signal is still live is a real failure, and warns.
    await expect(
      resolveFromManifest("t/x", { signal: new AbortController().signal }),
    ).resolves.toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(1);
  });

  it("still returns null, silently, for an unsupported explicit backend", async () => {
    await expect(resolveFromManifest("t/x", { backend: "npu" })).resolves.toBeNull();
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
