import { createLLM } from '../modelFactory';
import { mockLiteRTLM } from '../__mocks__/react-native-nitro-modules';

describe('modelFactory Security & Proxy Unit Tests', () => {
  let llm: ReturnType<typeof createLLM>;

  beforeEach(() => {
    jest.clearAllMocks();
    llm = createLLM({ enableMemoryTracking: true });
  });

  it('should block insecure HTTP downloads', async () => {
    await expect(llm.loadModel('http://example.com/model.litertlm'))
      .rejects.toThrow('Insecure HTTP URLs are not allowed for model downloads');
  });

  it('should allow secure HTTPS downloads and strip query parameters', async () => {
    await llm.loadModel('https://example.com/model.litertlm?token=123');

    expect(mockLiteRTLM.downloadModel).toHaveBeenCalledWith(
      'https://example.com/model.litertlm?token=123',
      'model.litertlm',
      expect.any(Function)
    );
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledWith('/mock/path/model.litertlm', undefined);
  });

  it('should throw an error for invalid model URL', async () => {
    await expect(llm.loadModel('https://example.com/'))
      .rejects.toThrow('Invalid model URL: https://example.com/');
  });

  it('should successfully proxy sendMessage and record memory metrics', async () => {
    const response = await llm.sendMessage("Test prompt");

    expect(response).toBe("Mock token");
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith(
      [{ type: "text", text: "Test prompt" }],
      undefined,
      undefined,
    );
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
    expect(llm.memoryTracker?.getSnapshotCount()).toBe(1); // sendMessage records one
  });

  it('should successfully proxy resetConversation and record memory metrics', async () => {
    await llm.resetConversation();

    expect(mockLiteRTLM.resetConversation).toHaveBeenCalled();
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
  });

  it('should successfully proxy sendMessageAsync and record memory metrics when done', async () => {
    const onToken = jest.fn();
    await llm.sendMessageAsync("Async prompt", onToken);

    expect(onToken).toHaveBeenCalledWith("Mock ", false);
    expect(onToken).toHaveBeenCalledWith("token", true);
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith(
      [{ type: "text", text: "Async prompt" }],
      expect.any(Function),
      undefined,
    );
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
  });

  it('should successfully proxy sendMessageWithImageAsync and record memory metrics when done', async () => {
    const onToken = jest.fn();
    await llm.sendMessageWithImageAsync("Vision prompt", "/path/to/image.jpg", onToken);

    expect(onToken).toHaveBeenCalledWith("Mock vision ", false);
    expect(onToken).toHaveBeenCalledWith("token", true);
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith(
      [
        { type: "text", text: "Vision prompt" },
        { type: "image", path: "/path/to/image.jpg" },
      ],
      expect.any(Function),
      undefined,
    );
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
  });

  it('should successfully proxy sendMessageWithAudioAsync and record memory metrics when done', async () => {
    const onToken = jest.fn();
    await llm.sendMessageWithAudioAsync("Audio prompt", "/path/to/audio.wav", onToken);

    expect(onToken).toHaveBeenCalledWith("Mock audio ", false);
    expect(onToken).toHaveBeenCalledWith("token", true);
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith(
      [
        { type: "text", text: "Audio prompt" },
        { type: "audio", path: "/path/to/audio.wav" },
      ],
      expect.any(Function),
      undefined,
    );
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
  });

  it('should successfully access memoryTracker and getSnapshots when memory tracking is enabled', () => {
    expect(llm.memoryTracker).toBeDefined();
    expect(llm.memoryTracker?.getCapacity()).toBe(256);
  });

  it('should not initialize memoryTracker when enableMemoryTracking option is false', () => {
    const untrackedLLM = createLLM({ enableMemoryTracking: false });
    expect(untrackedLLM.memoryTracker).toBeUndefined();
  });

  it('should forward maxContextTokens and maxOutputTokens config to native loadModel', async () => {
    const config = {
      backend: 'cpu' as const,
      maxContextTokens: 8192,
      maxOutputTokens: 2048,
    };

    await llm.loadModel('/local/path/model.litertlm', config);

    expect(mockLiteRTLM.loadModel).toHaveBeenCalledWith(
      '/local/path/model.litertlm',
      config
    );
  });

  it('should forward legacy maxTokens config to native loadModel for backward compat', async () => {
    const config = {
      backend: 'cpu' as const,
      maxTokens: 512,
    };

    await llm.loadModel('/local/path/model.litertlm', config);

    expect(mockLiteRTLM.loadModel).toHaveBeenCalledWith(
      '/local/path/model.litertlm',
      config
    );
  });

  it('should forward execute call exactly to native execute', async () => {
    const parts = [
      { type: 'image' as const, path: '/path/to/image.jpg', imageBuffer: new ArrayBuffer(10) }
    ];
    await llm.execute(parts);
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith(parts, undefined, undefined);
  });

  it('should strip file:// prefix from paths in execute parts before forwarding to native execute', async () => {
    const parts = [
      { type: 'image' as const, path: 'file:///path/to/image.jpg', imageBuffer: new ArrayBuffer(10) }
    ];
    await llm.execute(parts);
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith([
      { type: 'image' as const, path: '/path/to/image.jpg', imageBuffer: expect.any(ArrayBuffer) }
    ], undefined, undefined);
  });
});
