import type { MultimodalPart } from "../specs/LiteRTLM.nitro";

function streamMockTokens(
  parts: MultimodalPart[],
  onToken: (token: string, done: boolean) => void,
): void {
  const hasImage = parts.some((p) => p.type === "image");
  const hasAudio = parts.some((p) => p.type === "audio");
  if (hasImage) {
    onToken("Mock vision ", false);
    onToken("token", true);
  } else if (hasAudio) {
    onToken("Mock audio ", false);
    onToken("token", true);
  } else {
    onToken("Mock ", false);
    onToken("token", true);
  }
}

function mockExecuteResponse(parts: MultimodalPart[]): string {
  const hasImage = parts.some((p) => p.type === "image");
  const hasAudio = parts.some((p) => p.type === "audio");
  return hasImage
    ? "Mock vision token"
    : hasAudio
      ? "Mock audio token"
      : "Mock token";
}

const mockExecute = jest.fn(
  (parts: MultimodalPart[], onToken?: (token: string, done: boolean) => void): Promise<string> => {
    if (onToken) {
      streamMockTokens(parts, onToken);
    }
    return Promise.resolve(mockExecuteResponse(parts));
  },
);

export const mockLiteRTLM = {
  isReady: jest.fn(() => false),
  loadModel: jest.fn().mockResolvedValue(undefined),
  execute: mockExecute,
  sendMessage: jest.fn((message: string) =>
    mockExecute([{ type: "text", text: message }]),
  ),
  sendMessageWithImage: jest.fn((message: string, imagePath: string) =>
    mockExecute([
      { type: "text", text: message },
      { type: "image", path: imagePath },
    ]),
  ),
  downloadModel: jest.fn(async (url: string, fileName: string, onProgress?: (progress: number) => void) => {
    onProgress?.(1.0);
    return "/mock/path/model.litertlm";
  }),
  deleteModel: jest.fn().mockResolvedValue(undefined),
  sendMessageWithAudio: jest.fn((message: string, audioPath: string) =>
    mockExecute([
      { type: "text", text: message },
      { type: "audio", path: audioPath },
    ]),
  ),
  sendMultimodalMessage: jest.fn((parts: MultimodalPart[]) => mockExecute(parts)),
  sendMessageAsync: jest.fn((msg: string, onToken: (token: string, done: boolean) => void) =>
    mockExecute([{ type: "text", text: msg }], onToken).then(() => {}),
  ),
  sendMessageWithImageAsync: jest.fn(
    (msg: string, imagePath: string, onToken: (token: string, done: boolean) => void) =>
      mockExecute(
        [{ type: "text", text: msg }, { type: "image", path: imagePath }],
        onToken,
      ).then(() => {}),
  ),
  sendMessageWithAudioAsync: jest.fn(
    (msg: string, audioPath: string, onToken: (token: string, done: boolean) => void) =>
      mockExecute(
        [{ type: "text", text: msg }, { type: "audio", path: audioPath }],
        onToken,
      ).then(() => {}),
  ),
  getHistory: jest.fn(() => []),
  resetConversation: jest.fn(),
  getStats: jest.fn(() => ({
    promptTokens: 10,
    completionTokens: 20,
    totalTokens: 30,
    timeToFirstToken: 5,
    totalTime: 50,
    tokensPerSecond: 400,
  })),
  countTokens: jest.fn(() => -1),
  getMemoryUsage: jest.fn(() => ({
    nativeHeapBytes: 1000000,
    residentBytes: 2000000,
    availableMemoryBytes: 4000000,
    isLowMemory: false,
  })),
  getContextTokenCount: jest.fn(() => 128),
  unload: jest.fn().mockResolvedValue(undefined),
  setMemoryWarningCallback: jest.fn(),
  clearMemoryWarningCallback: jest.fn(),
  close: jest.fn(),
};

export const mockModelStore = {
  isCached: jest.fn((fileName: string) => false),
  getFilePath: jest.fn((fileName: string) => `/mock/path/${fileName}`),
  // -1 = "file not found": legacy tests skip the pre-flight memory check.
  // Estimator behavior is covered directly in memoryEstimator.test.ts.
  getFileSizeBytes: jest.fn((_absolutePath: string) => -1),
  listCachedFiles: jest.fn(() => []),
  deleteFile: jest.fn((fileName: string) => {
    return mockLiteRTLM.deleteModel(fileName);
  }),
  downloadFile: jest.fn(
    async (
      url: string,
      fileName: string,
      headersJson: string,
      sha256: string,
      onProgress: (progress: number) => void,
    ) => mockLiteRTLM.downloadModel(url, fileName, onProgress),
  ),
};

/** A bundle that declares nothing: an LLM with no backend or context constraints. */
export const mockNativeModelInfo = () => ({
  modelType: "llm" as const,
  fileSizeBytes: 1024,
  maxContextTokens: 0,
  isDynamicContext: false,
  minRuntimeVersion: undefined as string | undefined,
  modalities: (["text", "vision", "audio", "video"] as const).map((modality) => ({
    modality,
    supported: modality === "text",
    backends: [] as ("cpu" | "gpu" | "npu")[],
    npuBrand: "unknown" as const,
    socName: undefined as string | undefined,
  })),
  supportsThinking: false,
  supportsFunctionCalling: false,
  hasSpeculativeDecoding: false,
  samplerType: 0,
  samplerTemperature: 0,
  samplerTopK: 0,
  samplerTopP: 0,
  maxVisionTokenBudget: -1,
  visionSignatureSelection: [] as number[],
  embeddingDimension: -1,
  embeddingSignatureSelection: [] as number[],
});

export const mockModelInspector = {
  inspect: jest.fn(async (_modelPath: string) => mockNativeModelInfo()),
  getDeviceInfo: jest.fn(() => ({
    platform: "ios",
    osVersion: "26.1",
    model: "iPhone18,1",
    socModel: undefined as string | undefined,
    socManufacturer: undefined as string | undefined,
    totalMemoryBytes: 8 * 1024 * 1024 * 1024,
    gpuAvailable: true,
    isEmulator: false,
  })),
};

/** Deterministic fake vectors: [len, first char code, 1, 0, ...] as float32. */
const fakeVector = (parts: MultimodalPart[], size = 8): ArrayBuffer => {
  const text = parts.map((p) => p.text ?? p.type).join(" ");
  const v = new Float32Array(size);
  v[0] = text.length;
  v[1] = text.charCodeAt(0) || 0;
  v[2] = 1;
  return v.buffer;
};

export const mockEmbedder = {
  loadModel: jest.fn().mockResolvedValue(undefined),
  embed: jest.fn(async (parts: MultimodalPart[], options?: { outputSize?: number }) =>
    fakeVector(parts, options?.outputSize ?? 8),
  ),
  embedBatch: jest.fn(async (inputs: MultimodalPart[][], options?: { outputSize?: number }) =>
    inputs.map((parts) => fakeVector(parts, options?.outputSize ?? 8)),
  ),
  isReady: jest.fn(() => true),
  getMemoryUsage: jest.fn(() => ({
    nativeHeapBytes: 1000000,
    residentBytes: 2000000,
    availableMemoryBytes: 4000000,
    isLowMemory: false,
  })),
  unload: jest.fn().mockResolvedValue(undefined),
  close: jest.fn(),
};

export const NitroModules = {
  createHybridObject: jest.fn((name: string) => {
    if (name === "LiteRTLM") {
      return mockLiteRTLM;
    }
    if (name === "ModelStore") {
      return mockModelStore;
    }
    if (name === "ModelInspector") {
      return mockModelInspector;
    }
    if (name === "Embedder") {
      return mockEmbedder;
    }
    throw new Error(`Mock not implemented for hybrid object: ${name}`);
  }),
  createNativeArrayBuffer: jest.fn((size: number) => {
    return new ArrayBuffer(size);
  }),
};
