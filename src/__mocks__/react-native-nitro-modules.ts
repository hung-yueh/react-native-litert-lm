export const mockLiteRTLM = {
  isReady: jest.fn(() => false),
  loadModel: jest.fn().mockResolvedValue(undefined),
  sendMessage: jest.fn().mockResolvedValue("Mock response"),
  sendMessageWithImage: jest.fn().mockResolvedValue("Mock vision response"),
  downloadModel: jest.fn(async (url, fileName, onProgress) => {
    onProgress?.(1.0);
    return "/mock/path/model.litertlm";
  }),
  deleteModel: jest.fn().mockResolvedValue(undefined),
  sendMessageWithAudio: jest.fn().mockResolvedValue("Mock audio response"),
  sendMultimodalMessage: jest.fn().mockResolvedValue("Mock multimodal response"),
  sendMessageAsync: jest.fn((msg, onToken) => {
    onToken("Mock ", false);
    onToken("token", true);
    return Promise.resolve();
  }),
  sendMessageWithImageAsync: jest.fn((msg, imagePath, onToken) => {
    onToken("Mock vision ", false);
    onToken("token", true);
    return Promise.resolve();
  }),
  sendMessageWithAudioAsync: jest.fn((msg, audioPath, onToken) => {
    onToken("Mock audio ", false);
    onToken("token", true);
    return Promise.resolve();
  }),
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
  close: jest.fn(),
};

export const NitroModules = {
  createHybridObject: jest.fn((name: string) => {
    if (name === "LiteRTLM") {
      return mockLiteRTLM;
    }
    throw new Error(`Mock not implemented for hybrid object: ${name}`);
  }),
  createNativeArrayBuffer: jest.fn((size: number) => {
    return new ArrayBuffer(size);
  }),
};
