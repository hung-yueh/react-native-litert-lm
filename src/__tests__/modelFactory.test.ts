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

    expect(response).toBe("Mock response");
    expect(mockLiteRTLM.sendMessage).toHaveBeenCalledWith("Test prompt");
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
    expect(mockLiteRTLM.sendMessageAsync).toHaveBeenCalled();
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
  });

  it('should successfully proxy sendMessageWithImageAsync and record memory metrics when done', async () => {
    const onToken = jest.fn();
    await llm.sendMessageWithImageAsync("Vision prompt", "/path/to/image.jpg", onToken);

    expect(onToken).toHaveBeenCalledWith("Mock vision ", false);
    expect(onToken).toHaveBeenCalledWith("token", true);
    expect(mockLiteRTLM.sendMessageWithImageAsync).toHaveBeenCalledWith(
      "Vision prompt",
      "/path/to/image.jpg",
      expect.any(Function)
    );
    expect(mockLiteRTLM.getMemoryUsage).toHaveBeenCalled();
  });

  it('should successfully proxy sendMessageWithAudioAsync and record memory metrics when done', async () => {
    const onToken = jest.fn();
    await llm.sendMessageWithAudioAsync("Audio prompt", "/path/to/audio.wav", onToken);

    expect(onToken).toHaveBeenCalledWith("Mock audio ", false);
    expect(onToken).toHaveBeenCalledWith("token", true);
    expect(mockLiteRTLM.sendMessageWithAudioAsync).toHaveBeenCalledWith(
      "Audio prompt",
      "/path/to/audio.wav",
      expect.any(Function)
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
});
