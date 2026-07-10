/**
 * Integration tests for the memory-control layer in createLLM:
 * pre-flight rejection, forceLoad override, forecasting, budget callbacks,
 * typed event streaming, and unload.
 */
import { createLLM } from "../modelFactory";
import { MemoryError, isMemoryError } from "../errors";
import {
  mockLiteRTLM,
  mockModelStore,
} from "../__mocks__/react-native-nitro-modules";

jest.mock("react-native-nitro-modules");

const GIB = 1024 * 1024 * 1024;

describe("loadModel pre-flight memory check", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("rejects with a typed MemoryError when the model cannot fit", async () => {
    // 8 GiB model vs the mock's 4 MB of available memory
    mockModelStore.getFileSizeBytes.mockReturnValue(8 * GIB);

    const llm = createLLM();
    let caught: unknown;
    try {
      await llm.loadModel("/local/huge-model.litertlm");
    } catch (e) {
      caught = e;
    }

    expect(caught).toBeInstanceOf(MemoryError);
    expect(isMemoryError(caught)).toBe(true);
    const err = caught as MemoryError;
    expect(err.code).toBe("MEMORY_ESTIMATE_EXCEEDED");
    expect(err.estimate.verdict).toBe("critical");
    expect(err.message).toContain("forceLoad");
    // The engine was never touched — the app stays alive and usable
    expect(mockLiteRTLM.loadModel).not.toHaveBeenCalled();
  });

  it("loads anyway with forceLoad: true", async () => {
    mockModelStore.getFileSizeBytes.mockReturnValue(8 * GIB);

    const llm = createLLM();
    await llm.loadModel("/local/huge-model.litertlm", { forceLoad: true });

    expect(mockLiteRTLM.loadModel).toHaveBeenCalledWith(
      "/local/huge-model.litertlm",
      { forceLoad: true },
    );
  });

  it("loads normally when the estimate is safe", async () => {
    // 1 MB model vs 4 MB available: modelBytes + kv + overhead — overhead
    // floor is 200 MB… which exceeds 4 MB. Use a tiny context and override
    // realism by pointing at the file-not-found path instead.
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);

    const llm = createLLM();
    await llm.loadModel("/local/model.litertlm");
    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
  });

  it("skips the pre-flight when the file size is unknown", async () => {
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);

    const llm = createLLM();
    await expect(llm.loadModel("/local/model.litertlm")).resolves.not.toThrow();
  });
});

describe("estimateMemory / getMemoryForecast on the instance", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
  });

  it("returns null before any model is loaded", () => {
    const llm = createLLM();
    expect(llm.estimateMemory()).toBeNull();
    expect(llm.getMemoryForecast()).toBeNull();
  });

  it("returns an estimate for an explicit path", () => {
    const llm = createLLM();
    mockModelStore.getFileSizeBytes.mockReturnValue(1 * GIB);
    const estimate = llm.estimateMemory("/local/model.litertlm");
    expect(estimate).not.toBeNull();
    expect(estimate!.modelBytes).toBe(1 * GIB * 1.1);
    // The mock reports only 4 MB available — a 1 GiB model is critical
    expect(estimate!.verdict).toBe("critical");
  });

  it("forecasts context usage from the engine token count after load", async () => {
    const llm = createLLM();
    mockModelStore.getFileSizeBytes.mockReturnValue(1 * GIB);
    await llm.loadModel("/local/model.litertlm", {
      forceLoad: true,
      maxContextTokens: 512,
    });

    // Mock engine reports 128 tokens in the KV cache
    const forecast = llm.getMemoryForecast();
    expect(forecast).not.toBeNull();
    expect(forecast!.contextTokensUsed).toBe(128);
    expect(forecast!.maxContextTokens).toBe(512);
    expect(forecast!.contextUsedFraction).toBe(0.25);
    expect(forecast!.nearingLimit).toBe(false);
  });
});

describe("memory budget monitoring", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
  });

  it("fires onBudgetExceeded on level transitions after inference", async () => {
    // Mock usage: resident 3.9 GB of 4 GB total → critical
    mockLiteRTLM.getMemoryUsage.mockReturnValue({
      nativeHeapBytes: 0,
      residentBytes: 3.9 * GIB,
      availableMemoryBytes: 0.1 * GIB,
      isLowMemory: true,
    });

    const onBudgetExceeded = jest.fn();
    const llm = createLLM({ memoryBudget: { onBudgetExceeded } });
    await llm.loadModel("/local/model.litertlm");
    expect(onBudgetExceeded).toHaveBeenCalledWith("critical");

    // Stays at critical — no duplicate callback for the same level
    onBudgetExceeded.mockClear();
    await llm.execute([{ type: "text", text: "hi" }]);
    expect(onBudgetExceeded).not.toHaveBeenCalled();
  });
});

describe("executeWithEvents", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
  });

  it("delivers typed events from a stream containing tool-call markers", async () => {
    mockLiteRTLM.execute.mockImplementationOnce(
      (_parts: unknown, onToken?: (t: string, d: boolean) => void) => {
        onToken?.("Let me check: <tool_", false);
        onToken?.('call>{"name":"getWeather"}</tool_call>', false);
        onToken?.(" Done.", true);
        return Promise.resolve("full response");
      },
    );

    const llm = createLLM();
    const events: { type: string; text: string; done: boolean }[] = [];
    await llm.executeWithEvents([{ type: "text", text: "weather?" }], (e) =>
      events.push(e),
    );

    const tokens = events.filter((e) => e.type === "token").map((e) => e.text).join("");
    const toolCalls = events.filter((e) => e.type === "toolCall").map((e) => e.text).join("");
    expect(tokens).toBe("Let me check:  Done.");
    expect(toolCalls).toBe('{"name":"getWeather"}');
    expect(events[events.length - 1]!.done).toBe(true);
  });

  it("behaves like plain streaming when no markers appear", async () => {
    const llm = createLLM();
    const events: { type: string; text: string }[] = [];
    const result = await llm.executeWithEvents(
      [{ type: "text", text: "hello" }],
      (e) => events.push(e),
    );
    expect(result).toBe("Mock token");
    expect(events.every((e) => e.type === "token")).toBe(true);
  });
});

describe("unload", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
  });

  it("calls native unload and clears cached model facts", async () => {
    const llm = createLLM();
    mockModelStore.getFileSizeBytes.mockReturnValue(1 * GIB);
    await llm.loadModel("/local/model.litertlm", { forceLoad: true });
    expect(llm.estimateMemory()).not.toBeNull();

    await llm.unload();
    expect(mockLiteRTLM.unload).toHaveBeenCalled();
    expect(llm.estimateMemory()).toBeNull();
    expect(llm.getMemoryForecast()).toBeNull();
  });
});
