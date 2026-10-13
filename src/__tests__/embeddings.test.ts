/**
 * On-device embeddings: the pure helpers (task prefixes, vector math), the
 * embedding pre-flight, createEmbedder() over the native mock, and useEmbedder.
 */
(global as any).IS_REACT_ACT_ENVIRONMENT = true;
jest.mock("react-native", () => ({ Platform: { OS: "android" } }));
jest.mock("react-native-nitro-modules");

import React from "react";
import TestRenderer from "react-test-renderer";
import {
  EmbeddingTask,
  cosineSimilarity,
  dotProduct,
  normalize,
  topK,
} from "../core/embeddings";
import { normalizeModelInfo, preflightEmbedder, type DeviceInfo } from "../core/modelInfo";
import { LiteRTLMError, MemoryError } from "../core/errors";
import { createEmbedder } from "../embedder";
import { useEmbedder } from "../hooks";
import {
  mockEmbedder,
  mockModelInspector,
  mockModelStore,
  mockNativeModelInfo,
} from "../__mocks__/react-native-nitro-modules";

const GIB = 1024 * 1024 * 1024;

/** Shaped like embeddinggemma-2-text-270m: text-only, 768 dims, six input lengths. */
const embeddingBundle = (patch: Record<string, unknown> = {}) => ({
  ...mockNativeModelInfo(),
  modelType: "embedding" as any,
  embeddingDimension: 768,
  embeddingSignatureSelection: [128, 256, 512, 1024, 2048, 8192],
  ...patch,
});

describe("EmbeddingTask", () => {
  // The prompts in google/embeddinggemma-2's config_sentence_transformers.json
  it("builds the model card's query prompts from trimmed input", () => {
    expect(EmbeddingTask.searchQuery("  best ramen near me \n")).toBe(
      "task: search result | query: best ramen near me",
    );
    expect(EmbeddingTask.questionAnswering("why is the sky blue?")).toBe(
      "task: question answering | query: why is the sky blue?",
    );
    expect(EmbeddingTask.factChecking("x")).toBe("task: fact checking | query: x");
    expect(EmbeddingTask.codeRetrieval("x")).toBe("task: code retrieval | query: x");
  });

  it("builds symmetric prompts", () => {
    expect(EmbeddingTask.classification("x")).toBe("task: classification | query: x");
    expect(EmbeddingTask.clustering("x")).toBe("task: clustering | query: x");
    expect(EmbeddingTask.sentenceSimilarity("x")).toBe("task: sentence similarity | query: x");
  });

  it("formats documents with their title, or 'none'", () => {
    expect(EmbeddingTask.document(" Ramen Ichiraku ")).toBe("title: none | text: Ramen Ichiraku");
    expect(EmbeddingTask.document({ title: " Trip ", text: "Flight on May 4." })).toBe(
      "title: Trip | text: Flight on May 4.",
    );
    expect(EmbeddingTask.document({ title: "  ", text: "x" })).toBe("title: none | text: x");
    expect(EmbeddingTask.document({ title: "utils.ts", text: "export {}" })).toBe(
      "title: utils.ts | text: export {}",
    );
  });

  it("maps straight over arrays (the index is not taken for a title)", () => {
    expect(["a", "b"].map(EmbeddingTask.document)).toEqual([
      "title: none | text: a",
      "title: none | text: b",
    ]);
    expect(["q"].map(EmbeddingTask.searchQuery)).toEqual(["task: search result | query: q"]);
  });
});

describe("vector math", () => {
  const a = new Float32Array([1, 0, 0]);
  const b = new Float32Array([0, 1, 0]);

  it("computes dot products and cosine similarity", () => {
    expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32);
    expect(cosineSimilarity(a, a)).toBeCloseTo(1);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0);
    expect(cosineSimilarity(a, new Float32Array([-2, 0, 0]))).toBeCloseTo(-1);
    expect(cosineSimilarity(a, new Float32Array(3))).toBe(0);
  });

  it("refuses vectors of different sizes", () => {
    expect(() => cosineSimilarity(a, new Float32Array(4))).toThrow(RangeError);
    expect(() => dotProduct([1], [1, 2])).toThrow(/same model and outputSize/);
  });

  it("normalizes to unit length without touching the input", () => {
    const v = new Float32Array([3, 4]);
    const n = normalize(v);
    expect(Array.from(n)).toEqual([0.6000000238418579, 0.800000011920929]);
    expect(Array.from(v)).toEqual([3, 4]);
    expect(Array.from(normalize([0, 0]))).toEqual([0, 0]);
  });

  it("ranks candidates by similarity", () => {
    const hits = topK(a, [b, new Float32Array([1, 1, 0]), a], 2);
    expect(hits.map((h) => h.index)).toEqual([2, 1]);
    expect(hits[0].score).toBeCloseTo(1);
    expect(topK(a, [b], 0)).toEqual([]);
    expect(topK(a, [b], 5)).toHaveLength(1);
  });
});

describe("preflightEmbedder", () => {
  const galaxy: DeviceInfo = {
    platform: "android",
    osVersion: "16",
    model: "samsung SM-S931U",
    socModel: "SM8750",
    totalMemoryBytes: 12e9,
    gpuAvailable: false,
    isEmulator: false,
  };

  it("refuses an LLM bundle", () => {
    const llm = normalizeModelInfo("/m/gemma.litertlm", mockNativeModelInfo());
    expect(() => preflightEmbedder(llm, undefined)).toThrow(
      expect.objectContaining({ code: "MODEL_TYPE_MISMATCH", message: expect.stringContaining("createLLM()") }),
    );
  });

  it("runs a per-SoC NPU build on its NPU, and refuses it elsewhere", () => {
    const npu = normalizeModelInfo(
      "/m/egemma_SM8750.litertlm",
      embeddingBundle({
        modalities: mockNativeModelInfo().modalities.map((m) =>
          m.modality === "text"
            ? { ...m, backends: ["npu"], npuBrand: "qualcomm", socName: "SM8750" }
            : m,
        ),
      }),
    );
    expect(preflightEmbedder(npu, undefined, { device: galaxy }).config.backend).toBe("npu");
    expect(() =>
      preflightEmbedder(npu, undefined, { device: { ...galaxy, socModel: "Tensor G4" } }),
    ).toThrow(expect.objectContaining({ code: "UNSUPPORTED" }));
  });

  it("caps maxInputLength at the longest compiled input", () => {
    const info = normalizeModelInfo("/m/e.litertlm", embeddingBundle());
    const pf = preflightEmbedder(info, { maxInputLength: 32768 });
    expect(pf.config.maxInputLength).toBe(8192);
    expect(pf.notes[0]).toContain("longest input of 8192");
    expect(preflightEmbedder(info, { maxInputLength: 512 }).changed).toBe(false);
  });
});

describe("createEmbedder", () => {
  let warn: jest.SpyInstance;
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
    mockModelInspector.inspect.mockResolvedValue(embeddingBundle());
    warn = jest.spyOn(console, "warn").mockImplementation(() => {});
  });
  afterEach(() => {
    warn.mockRestore();
    mockModelInspector.inspect.mockImplementation(async () => mockNativeModelInfo());
  });

  it("loads an embedding bundle and exposes its info", async () => {
    const embedder = createEmbedder();
    await embedder.loadModel("/m/egemma.litertlm", { maxInputLength: 512 });
    expect(mockEmbedder.loadModel).toHaveBeenCalledWith("/m/egemma.litertlm", { maxInputLength: 512 });
    expect(embedder.getModelInfo()?.embedding?.dimension).toBe(768);
    await embedder.unload();
    expect(embedder.getModelInfo()).toBeNull();
  });

  it("refuses an LLM bundle before touching the engine", async () => {
    mockModelInspector.inspect.mockResolvedValueOnce(mockNativeModelInfo());
    const err = await createEmbedder().loadModel("/m/gemma.litertlm").catch((e) => e);
    expect(err).toBeInstanceOf(LiteRTLMError);
    expect(err.code).toBe("MODEL_TYPE_MISMATCH");
    expect(mockEmbedder.loadModel).not.toHaveBeenCalled();
  });

  it("refuses a load that won't fit, unless forced", async () => {
    mockModelStore.getFileSizeBytes.mockReturnValue(8 * GIB);
    const err = await createEmbedder().loadModel("/m/huge.litertlm").catch((e) => e);
    expect(err).toBeInstanceOf(MemoryError);
    expect(mockEmbedder.loadModel).not.toHaveBeenCalled();
    await createEmbedder().loadModel("/m/huge.litertlm", { forceLoad: true });
    expect(mockEmbedder.loadModel).toHaveBeenCalled();
  });

  it("embeds text and parts into Float32Arrays over the native buffer", async () => {
    const embedder = createEmbedder();
    const v = await embedder.embed("hi", { normalize: true, outputSize: 4 });
    expect(v).toBeInstanceOf(Float32Array);
    expect(v).toHaveLength(4);
    expect(mockEmbedder.embed).toHaveBeenCalledWith([{ type: "text", text: "hi" }], {
      normalize: true,
      outputSize: 4,
    });

    await embedder.embed([{ type: "image", path: "file:///photos/a.jpg" }]);
    expect(mockEmbedder.embed.mock.calls[1][0]).toEqual([{ type: "image", path: "/photos/a.jpg" }]);
  });

  it("embeds a batch in one native call", async () => {
    const embedder = createEmbedder();
    const vectors = await embedder.embedBatch(["a", "bb", [{ type: "text", text: "ccc" }]]);
    expect(mockEmbedder.embedBatch).toHaveBeenCalledTimes(1);
    expect(vectors.map((v) => v[0])).toEqual([1, 2, 3]);
    expect(await embedder.embedBatch([])).toEqual([]);
    expect(mockEmbedder.embedBatch).toHaveBeenCalledTimes(1);
  });

  it("types native failures", async () => {
    mockEmbedder.embed.mockRejectedValueOnce(
      new Error("[LiteRTLM:embed:NOT_FOUND:INVALID_INPUT] Image file not found: /x.jpg"),
    );
    const err = await createEmbedder().embed([{ type: "image", path: "/x.jpg" }]).catch((e) => e);
    expect(err.code).toBe("INVALID_INPUT");

    mockEmbedder.loadModel.mockRejectedValueOnce(new Error("[LiteRTLM:load:UNIMPLEMENTED] no NPU dispatch"));
    const loadErr = await createEmbedder().loadModel("/m/e.litertlm").catch((e) => e);
    expect(loadErr.code).toBe("UNSUPPORTED");

    mockEmbedder.embedBatch.mockRejectedValueOnce(new Error("boom"));
    const batchErr = await createEmbedder().embedBatch(["a"]).catch((e) => e);
    expect(batchErr.code).toBe("EMBEDDING_FAILED");

    // A bridge that throws synchronously is typed the same way
    mockEmbedder.embed.mockImplementationOnce(() => {
      throw new Error("[LiteRTLM:embed:INVALID_ARGUMENT:INVALID_INPUT] Image part needs a path or a buffer");
    });
    const syncErr = await createEmbedder().embed([{ type: "image" }]).catch((e) => e);
    expect(syncErr).toBeInstanceOf(LiteRTLMError);
    expect(syncErr.code).toBe("INVALID_INPUT");
  });

  it("passes through readiness, memory and close", () => {
    const embedder = createEmbedder();
    expect(embedder.isReady()).toBe(true);
    expect(embedder.getMemoryUsage().availableMemoryBytes).toBe(4000000);
    embedder.close();
    expect(mockEmbedder.close).toHaveBeenCalled();
  });
});

describe("useEmbedder", () => {
  function renderHook<R>(callback: () => R) {
    const result = { current: null as unknown as R };
    const Probe = () => {
      result.current = callback();
      return null;
    };
    let renderer!: TestRenderer.ReactTestRenderer;
    TestRenderer.act(() => {
      renderer = TestRenderer.create(React.createElement(Probe));
    });
    return { result, unmount: () => TestRenderer.act(() => renderer.unmount()) };
  }

  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
    mockModelInspector.inspect.mockResolvedValue(embeddingBundle());
  });
  afterEach(() => mockModelInspector.inspect.mockImplementation(async () => mockNativeModelInfo()));

  it("loads, embeds, and closes on unmount", async () => {
    const { result, unmount } = renderHook(() => useEmbedder("/m/egemma.litertlm", { maxInputLength: 256 }));
    await TestRenderer.act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(result.current.isReady).toBe(true);
    expect(result.current.modelInfo?.embedding?.dimension).toBe(768);
    const v = await result.current.embed("hello");
    expect(v).toBeInstanceOf(Float32Array);
    expect((await result.current.embedBatch(["a", "b"])).length).toBe(2);
    unmount();
    expect(mockEmbedder.close).toHaveBeenCalled();
  });

  it("reports a typed load failure", async () => {
    mockModelInspector.inspect.mockResolvedValue(mockNativeModelInfo());
    const { result } = renderHook(() => useEmbedder("/m/gemma.litertlm"));
    await TestRenderer.act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(result.current.isReady).toBe(false);
    expect(result.current.errorCode).toBe("MODEL_TYPE_MISMATCH");
    await expect(result.current.embed("x")).rejects.toMatchObject({ code: "MODEL_NOT_LOADED" });
  });
});
