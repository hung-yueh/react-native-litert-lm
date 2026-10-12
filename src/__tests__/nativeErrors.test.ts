/**
 * Native failures → typed LiteRTLMError: the tag protocol shared by Swift and
 * Kotlin, the per-phase status mapping, and the createLLM / ModelRegistry
 * wiring that applies it to every native rejection.
 */
import {
  parseNativeStatus,
  errorCodeForStatus,
  toLiteRTLMError,
  CANONICAL_STATUS_NAMES,
} from "../core/nativeError";
import { LiteRTLMError, MemoryError, isMemoryError } from "../core/errors";
import { createLLM } from "../modelFactory";
import { ModelRegistry } from "../modelRegistry";
import { mockLiteRTLM, mockModelStore } from "../__mocks__/react-native-nitro-modules";

jest.mock("react-native-nitro-modules");

const GIB = 1024 * 1024 * 1024;

describe("parseNativeStatus", () => {
  it("reads an iOS-style tag (Swift description, no prefix)", () => {
    const parsed = parseNativeStatus(
      "[LiteRTLM:load:INVALID_ARGUMENT] Failed to create the engine (cpu): unsupported activation type",
    );
    expect(parsed).toEqual({
      status: {
        code: 3,
        name: "INVALID_ARGUMENT",
        phase: "load",
        message: "Failed to create the engine (cpu): unsupported activation type",
      },
      code: undefined,
    });
  });

  it("reads an Android-style tag behind the Java exception class", () => {
    const parsed = parseNativeStatus(
      "com.margelo.nitro.dev.litert.litertlm.LiteRTLMNativeException: [LiteRTLM:execute:RESOURCE_EXHAUSTED] KV cache full",
    );
    expect(parsed?.status).toMatchObject({ code: 8, name: "RESOURCE_EXHAUSTED", phase: "execute" });
    expect(parsed?.status.message).toBe("KV cache full");
  });

  it("carries an explicit error code for wrapper-detected failures", () => {
    const parsed = parseNativeStatus(
      "[LiteRTLM:execute:FAILED_PRECONDITION:MODEL_NOT_LOADED] No model loaded. Call loadModel() first.",
    );
    expect(parsed?.code).toBe("MODEL_NOT_LOADED");
  });

  it("ignores an unknown explicit code and an unknown status name", () => {
    const parsed = parseNativeStatus("[LiteRTLM:load:WEIRD_STATUS:NOT_A_CODE] boom");
    expect(parsed?.code).toBeUndefined();
    expect(parsed?.status).toMatchObject({ code: 2, name: "UNKNOWN" });
  });

  it("returns null without a tag", () => {
    expect(parseNativeStatus("plain JS failure")).toBeNull();
  });

  it("indexes status names by their google.rpc.Code value", () => {
    expect(CANONICAL_STATUS_NAMES[0]).toBe("OK");
    expect(CANONICAL_STATUS_NAMES[8]).toBe("RESOURCE_EXHAUSTED");
    expect(CANONICAL_STATUS_NAMES[16]).toBe("UNAUTHENTICATED");
  });
});

describe("errorCodeForStatus", () => {
  it.each([
    ["RESOURCE_EXHAUSTED", "load", "OUT_OF_MEMORY"],
    ["RESOURCE_EXHAUSTED", "execute", "OUT_OF_MEMORY"],
    ["CANCELLED", "execute", "INFERENCE_CANCELLED"],
    ["UNIMPLEMENTED", "load", "UNSUPPORTED"],
    ["NOT_FOUND", "load", "MODEL_NOT_FOUND"],
    ["NOT_FOUND", "inspect", "MODEL_NOT_FOUND"],
    ["NOT_FOUND", "execute", "INVALID_INPUT"],
    ["INVALID_ARGUMENT", "load", "INVALID_CONFIG"],
    ["OUT_OF_RANGE", "execute", "INVALID_INPUT"],
    ["FAILED_PRECONDITION", "embed", "INVALID_INPUT"],
    ["INVALID_ARGUMENT", "inspect", "INVALID_MODEL"],
    ["INTERNAL", "load", "ENGINE_INIT_FAILED"],
    ["UNKNOWN", "execute", "INFERENCE_FAILED"],
    ["DATA_LOSS", "inspect", "INVALID_MODEL"],
    ["INTERNAL", "embed", "EMBEDDING_FAILED"],
  ] as const)("%s during %s → %s", (status, phase, code) => {
    expect(errorCodeForStatus(status, phase)).toBe(code);
  });
});

describe("toLiteRTLMError", () => {
  it("builds a typed error with the native status and a clean message", () => {
    const raw = new Error("[LiteRTLM:load:UNIMPLEMENTED] NPU dispatch library not found");
    const e = toLiteRTLMError(raw, "load");
    expect(e).toBeInstanceOf(LiteRTLMError);
    expect(e.code).toBe("UNSUPPORTED");
    expect(e.message).toBe("NPU dispatch library not found");
    expect(e.nativeStatus?.name).toBe("UNIMPLEMENTED");
    expect(e.cause).toBe(raw);
  });

  it("prefers the explicit code over the status mapping", () => {
    const e = toLiteRTLMError(
      new Error("[LiteRTLM:execute:FAILED_PRECONDITION:INVALID_CONFIG] needs enableStructuredOutput"),
      "execute",
    );
    expect(e.code).toBe("INVALID_CONFIG");
  });

  it("passes typed errors through unchanged", () => {
    const typed = new LiteRTLMError("MODEL_NOT_FOUND", "nope");
    expect(toLiteRTLMError(typed, "load")).toBe(typed);
  });

  it("gives untagged errors the phase default and strips the Java class prefix", () => {
    const e = toLiteRTLMError(new Error("java.lang.RuntimeException: something odd"), "execute");
    expect(e.code).toBe("INFERENCE_FAILED");
    expect(e.message).toBe("something odd");
    expect(e.nativeStatus).toBeUndefined();
  });

  it("unwraps an NSError description", () => {
    const e = toLiteRTLMError(
      new Error('Error Domain=LiteRTLM.ModelStore Code=503 "HTTP 503" UserInfo={NSLocalizedDescription=HTTP 503}'),
      "load",
    );
    expect(e.message).toBe("HTTP 503");
  });

  it("accepts non-Error rejections", () => {
    expect(toLiteRTLMError("string failure", "embed").code).toBe("EMBEDDING_FAILED");
  });
});

describe("createLLM types native rejections", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockModelStore.getFileSizeBytes.mockReturnValue(-1);
  });

  it("loadModel rejects with the mapped code", async () => {
    mockLiteRTLM.loadModel.mockRejectedValueOnce(
      new Error("[LiteRTLM:load:NOT_FOUND:MODEL_NOT_FOUND] Model file not found: /x.litertlm"),
    );
    const llm = createLLM();
    const err = await llm.loadModel("/x.litertlm").catch((e) => e);
    expect(err).toBeInstanceOf(LiteRTLMError);
    expect(err.code).toBe("MODEL_NOT_FOUND");
    expect(err.message).toBe("Model file not found: /x.litertlm");
  });

  it("an engine OOM during load is a MemoryError carrying the estimate", async () => {
    mockModelStore.getFileSizeBytes.mockReturnValue(1 * GIB);
    mockLiteRTLM.loadModel.mockRejectedValueOnce(
      new Error("[LiteRTLM:load:RESOURCE_EXHAUSTED] Failed to allocate tensor arena"),
    );
    const llm = createLLM();
    // forceLoad skips the pre-flight refusal so the engine itself fails
    const err = await llm.loadModel("/m.litertlm", { forceLoad: true }).catch((e) => e);
    expect(err).toBeInstanceOf(MemoryError);
    expect(isMemoryError(err)).toBe(true);
    expect(err.code).toBe("OUT_OF_MEMORY");
    expect(err.estimate.maxContextTokens).toBe(4096);
    expect(err.nativeStatus.name).toBe("RESOURCE_EXHAUSTED");
  });

  it("an OOM without an estimate stays a plain OUT_OF_MEMORY error", async () => {
    mockLiteRTLM.loadModel.mockRejectedValueOnce(
      new Error("[LiteRTLM:load:RESOURCE_EXHAUSTED] Failed to allocate"),
    );
    const llm = createLLM();
    const err = await llm.loadModel("/m.litertlm").catch((e) => e);
    expect(err.code).toBe("OUT_OF_MEMORY");
    expect(err).not.toBeInstanceOf(MemoryError);
    expect(isMemoryError(err)).toBe(false);
  });

  it("execute rejects with a typed error (blocking and streaming)", async () => {
    const llm = createLLM();
    mockLiteRTLM.execute.mockRejectedValueOnce(
      new Error("[LiteRTLM:execute:INVALID_ARGUMENT] Invalid JSON schema"),
    );
    const blocking = await llm.execute([{ type: "text", text: "hi" }]).catch((e) => e);
    expect(blocking.code).toBe("INVALID_INPUT");

    mockLiteRTLM.execute.mockRejectedValueOnce(new Error("[LiteRTLM:execute:CANCELLED] cancelled"));
    const streaming = await llm
      .execute([{ type: "text", text: "hi" }], () => {})
      .catch((e) => e);
    expect(streaming.code).toBe("INFERENCE_CANCELLED");
  });

  it("legacy methods routed through execute are typed too", async () => {
    const llm = createLLM();
    mockLiteRTLM.execute.mockRejectedValueOnce(
      new Error("[LiteRTLM:execute:FAILED_PRECONDITION:MODEL_NOT_LOADED] No model loaded."),
    );
    const err = await llm.sendMessage("hi").catch((e: unknown) => e);
    expect((err as LiteRTLMError).code).toBe("MODEL_NOT_LOADED");
  });

  it("resetConversation rethrows a typed error", () => {
    const llm = createLLM();
    mockLiteRTLM.resetConversation.mockImplementationOnce(() => {
      throw new Error("[LiteRTLM:load:INVALID_ARGUMENT] Invalid initial messages");
    });
    expect(() => llm.resetConversation("[{bad")).toThrow(
      expect.objectContaining({ code: "INVALID_CONFIG" }),
    );
  });
});

describe("ModelRegistry download errors", () => {
  beforeEach(() => jest.clearAllMocks());

  it("a failed download is DOWNLOAD_FAILED with the cause kept", async () => {
    const cause = new Error("HTTP 503");
    mockModelStore.downloadFile.mockRejectedValueOnce(cause);
    const err = await ModelRegistry.resolveModel("https://example.com/m.litertlm").catch((e) => e);
    expect(err).toBeInstanceOf(LiteRTLMError);
    expect(err.code).toBe("DOWNLOAD_FAILED");
    expect(err.message).toContain("HTTP 503");
    expect(err.cause).toBe(cause);
  });

  it("an insecure URL is INVALID_INPUT", async () => {
    const err = await ModelRegistry.resolveModel("http://example.com/m.litertlm").catch((e) => e);
    expect(err.code).toBe("INVALID_INPUT");
  });
});
