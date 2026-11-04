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

export const mockMetricModel = {
  isReady: jest.fn(() => false),
  loadModel: jest.fn().mockResolvedValue(undefined),
  evaluate: jest.fn(async () => {
    const rawValues = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
    const scores: Record<string, number> = {
      Warmth: 0.5,
      Sentiment: 0.5,
      Directness: 0.5,
      "Passive-Aggressive": 0.5,
      Chemistry: 0.5,
      "Professional Impression": 0.5,
      Urgency: 0.5,
      Formality: 0.5,
      Tension: 0.5,
      Empathy: 0.5,
      Clarity: 0.5,
      Assertiveness: 0.5,
      Playfulness: 0.5,
    };
    return {
      scores,
      values,
      rawValues,
      latencyMs: 1.5,
    };
  }),
  unload: jest.fn().mockResolvedValue(undefined),
  close: jest.fn(),
};

export const mockLayaModel = {
  isReady: jest.fn(() => false),
  loadModel: jest.fn().mockResolvedValue(undefined),
  decide: jest.fn(async (request: any) => {
    const qtype = (request.type || "choice").toLowerCase();
    let choice: string | undefined;
    let score: number | undefined;
    let noul: number | undefined;
    let decision: boolean | undefined;
    const probabilities: Record<string, number> = {};

    if (qtype === "choice") {
      let opts = request.options;
      if (!opts && request.criteria) {
        try {
          const parsed = JSON.parse(request.criteria);
          if (Array.isArray(parsed)) opts = parsed;
          else if (typeof parsed === "object") opts = Object.keys(parsed);
        } catch (_) {}
      }
      opts = opts && opts.length > 0 ? opts : ["default"];
      choice = opts[0];
      const p = 1.0 / opts.length;
      opts.forEach((o: string) => {
        probabilities[o] = p;
      });
    } else if (qtype === "score") {
      score = 2.0;
      probabilities["0"] = 0.1;
      probabilities["1"] = 0.2;
      probabilities["2"] = 0.4;
      probabilities["3"] = 0.3;
    } else if (qtype === "noul") {
      noul = 0.95;
      decision = true;
      probabilities["false"] = 0.05;
      probabilities["true"] = 0.95;
    }

    return {
      type: request.type,
      choice,
      score,
      noul,
      decision,
      probabilities,
      confidence: 0.95,
      actProbability: 1.0,
      latencyMs: 12.5,
    };
  }),
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
    if (name === "MetricModel") {
      return mockMetricModel;
    }
    if (name === "LayaModel") {
      return mockLayaModel;
    }
    throw new Error(`Mock not implemented for hybrid object: ${name}`);
  }),
  createNativeArrayBuffer: jest.fn((size: number) => {
    return new ArrayBuffer(size);
  }),
};
