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
      onProgress: (progress: number) => void,
    ) => mockLiteRTLM.downloadModel(url, fileName, onProgress),
  ),
};

export const NitroModules = {
  createHybridObject: jest.fn((name: string) => {
    if (name === "LiteRTLM") {
      return mockLiteRTLM;
    }
    if (name === "ModelStore") {
      return mockModelStore;
    }
    throw new Error(`Mock not implemented for hybrid object: ${name}`);
  }),
  createNativeArrayBuffer: jest.fn((size: number) => {
    return new ArrayBuffer(size);
  }),
};
