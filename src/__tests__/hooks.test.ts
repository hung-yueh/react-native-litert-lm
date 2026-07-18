// Configure React act environment
(global as any).IS_REACT_ACT_ENVIRONMENT = true;

import { useModel } from '../hooks';
import { mockLiteRTLM } from '../__mocks__/react-native-nitro-modules';
import TestRenderer from 'react-test-renderer';
import React from 'react';

// Mock react-native
jest.mock('react-native', () => ({
  Platform: {
    OS: 'ios',
    select: jest.fn((dict) => dict.ios),
  },
}));

// Helper to render and test hooks using react-test-renderer
function renderHook<P, R>(callback: (props: P) => R, initialProps?: P) {
  const result = { current: null as unknown as R };
  
  const TestComponent = ({ props }: { props: P }) => {
    result.current = callback(props);
    return null;
  };

  let renderer: TestRenderer.ReactTestRenderer;
  TestRenderer.act(() => {
    renderer = TestRenderer.create(React.createElement(TestComponent, { props: initialProps as P }));
  });

  const rerender = (newProps: P) => {
    TestRenderer.act(() => {
      renderer.update(React.createElement(TestComponent, { props: newProps }));
    });
  };

  const unmount = () => {
    TestRenderer.act(() => {
      renderer.unmount();
    });
  };

  return { result, rerender, unmount };
}

describe('useModel React Hook Unit Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should initialize with correct default state and call loadModel automatically when autoLoad is true', async () => {
    let hookResult: any;
    
    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { autoLoad: true }));
    });

    expect(hookResult.result.current.isReady).toBe(true);
    expect(hookResult.result.current.isGenerating).toBe(false);
    expect(hookResult.result.current.downloadProgress).toBe(1); // loadModel completed
    expect(hookResult.result.current.error).toBeNull();
    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
  });

  it('should not call loadModel automatically when autoLoad is false', async () => {
    let hookResult: any;
    
    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { autoLoad: false }));
    });

    expect(hookResult.result.current.isReady).toBe(false);
    expect(mockLiteRTLM.loadModel).not.toHaveBeenCalled();

    // Call load manually
    await TestRenderer.act(async () => {
      await hookResult.result.current.load();
    });

    expect(hookResult.result.current.isReady).toBe(true);
    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
  });

  it('should handle model load failure gracefully', async () => {
    mockLiteRTLM.loadModel.mockRejectedValueOnce(new Error("Model load failed"));
    let hookResult: any;

    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { autoLoad: true }));
    });

    expect(hookResult.result.current.isReady).toBe(false);
    expect(hookResult.result.current.error).toBe("Model load failed");
  });

  it('should generate text successfully and trigger memory summary update', async () => {
    let hookResult: any;
    
    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { 
        autoLoad: true,
        enableMemoryTracking: true 
      }));
    });

    let response = "";
    await TestRenderer.act(async () => {
      response = await hookResult.result.current.generate("Test prompt");
    });

    expect(response).toBe("Mock token");
    expect(mockLiteRTLM.execute).toHaveBeenCalledWith(
      [{ type: "text", text: "Test prompt" }],
      undefined,
      undefined,
    );
    expect(hookResult.result.current.memorySummary).toBeDefined();
  });

  it('should reset conversation correctly', async () => {
    let hookResult: any;
    
    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { autoLoad: true }));
    });

    hookResult.result.current.reset();
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalled();
  });

  it('should delete model file correctly', async () => {
    let hookResult: any;
    
    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { autoLoad: true }));
    });

    await TestRenderer.act(async () => {
      await hookResult.result.current.deleteModel();
    });

    expect(mockLiteRTLM.deleteModel).toHaveBeenCalledWith('model.litertlm');
    expect(hookResult.result.current.isReady).toBe(false);
    expect(hookResult.result.current.downloadProgress).toBe(0);
  });

  it('should call close on unmount', async () => {
    let hookResult: any;
    
    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', { autoLoad: false }));
    });

    hookResult.unmount();
    expect(mockLiteRTLM.close).toHaveBeenCalled();
  });

  it('should pass maxContextTokens and maxOutputTokens through to loadModel', async () => {
    let hookResult: any;

    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', {
        autoLoad: true,
        maxContextTokens: 8192,
        maxOutputTokens: 2048,
      }));
    });

    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
    const callArgs = mockLiteRTLM.loadModel.mock.calls[0];
    const config = callArgs[1];
    expect(config).toMatchObject({
      maxContextTokens: 8192,
      maxOutputTokens: 2048,
    });
  });

  it('should forward streamToolCalls and tuning knobs to loadModel', async () => {
    let hookResult: any;

    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', {
        autoLoad: true,
        streamToolCalls: true,
        toolCallChannelName: 'my_tool_call',
        numThreads: 4,
        prefillChunkSize: 256,
        activationDataType: 'f16',
        loraPath: '/lora/adapter.bin',
        audioLoraPath: '/lora/audio.bin',
        loraRank: 16,
        forceLoad: true,
      }));
    });

    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
    const config = mockLiteRTLM.loadModel.mock.calls[0][1];
    expect(config).toMatchObject({
      streamToolCalls: true,
      toolCallChannelName: 'my_tool_call',
      numThreads: 4,
      prefillChunkSize: 256,
      activationDataType: 'f16',
      loraPath: '/lora/adapter.bin',
      audioLoraPath: '/lora/audio.bin',
      loraRank: 16,
      forceLoad: true,
    });
  });

  it('should pass legacy maxTokens through when new fields are not set', async () => {
    let hookResult: any;

    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', {
        autoLoad: true,
        maxTokens: 512,
      }));
    });

    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
    const callArgs = mockLiteRTLM.loadModel.mock.calls[0];
    const config = callArgs[1];
    expect(config).toMatchObject({ maxTokens: 512 });
    expect(config.maxContextTokens).toBeUndefined();
    expect(config.maxOutputTokens).toBeUndefined();
  });

  it('should pass both new fields alongside legacy maxTokens when all are set', async () => {
    let hookResult: any;

    await TestRenderer.act(async () => {
      hookResult = renderHook(() => useModel('https://example.com/model.litertlm', {
        autoLoad: true,
        maxTokens: 512,
        maxContextTokens: 4096,
        maxOutputTokens: 1024,
      }));
    });

    expect(mockLiteRTLM.loadModel).toHaveBeenCalled();
    const callArgs = mockLiteRTLM.loadModel.mock.calls[0];
    const config = callArgs[1];
    expect(config).toMatchObject({
      maxTokens: 512,
      maxContextTokens: 4096,
      maxOutputTokens: 1024,
    });
  });
});

describe('useModel lifecycle races', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('closes the instance when unmounted while a load is in flight', async () => {
    let resolveLoad!: () => void;
    mockLiteRTLM.loadModel.mockImplementationOnce(
      () => new Promise<void>((resolve) => { resolveLoad = resolve; }),
    );

    let hookResult: any;
    await TestRenderer.act(async () => {
      hookResult = renderHook(() =>
        useModel('https://example.com/model.litertlm', { autoLoad: true }),
      );
    });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledTimes(1);
    expect(hookResult.result.current.isReady).toBe(false);

    await TestRenderer.act(async () => {
      hookResult.unmount();
    });
    expect(mockLiteRTLM.close).toHaveBeenCalled();

    // Late resolution after unmount must not throw or crash the renderer.
    await TestRenderer.act(async () => {
      resolveLoad();
    });
  });

  it('reloads when the config changes', async () => {
    let hookResult: any;
    await TestRenderer.act(async () => {
      hookResult = renderHook(
        (props: { temperature: number }) =>
          useModel('https://example.com/model.litertlm', {
            autoLoad: true,
            temperature: props.temperature,
          }),
        { temperature: 0.5 },
      );
    });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledTimes(1);
    expect(mockLiteRTLM.loadModel.mock.calls[0][1]).toMatchObject({ temperature: 0.5 });

    await TestRenderer.act(async () => {
      hookResult.rerender({ temperature: 0.9 });
    });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledTimes(2);
    expect(mockLiteRTLM.loadModel.mock.calls[1][1]).toMatchObject({ temperature: 0.9 });
  });

  it('does not reload when an equivalent inline config is re-rendered', async () => {
    let hookResult: any;
    await TestRenderer.act(async () => {
      hookResult = renderHook(
        (props: { tick: number }) => {
          // Inline object + inline tools array on every render — the hook's
          // primitive destructuring / toolsKey must keep this stable.
          void props.tick;
          return useModel('https://example.com/model.litertlm', {
            autoLoad: true,
            temperature: 0.5,
            tools: [{ name: 't', description: 'd', parametersJson: '{}' }],
          });
        },
        { tick: 1 },
      );
    });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledTimes(1);

    await TestRenderer.act(async () => {
      hookResult.rerender({ tick: 2 });
    });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledTimes(1);
  });

  it('recovers from a failed load when load() is retried', async () => {
    mockLiteRTLM.loadModel.mockRejectedValueOnce(new Error('boom'));

    let hookResult: any;
    await TestRenderer.act(async () => {
      hookResult = renderHook(() =>
        useModel('https://example.com/model.litertlm', { autoLoad: true }),
      );
    });
    expect(hookResult.result.current.error).toBe('boom');
    expect(hookResult.result.current.isReady).toBe(false);

    await TestRenderer.act(async () => {
      await hookResult.result.current.load();
    });
    expect(hookResult.result.current.error).toBeNull();
    expect(hookResult.result.current.isReady).toBe(true);
  });

  it('supports calling load() repeatedly without autoLoad', async () => {
    let hookResult: any;
    await TestRenderer.act(async () => {
      hookResult = renderHook(() =>
        useModel('https://example.com/model.litertlm', { autoLoad: false }),
      );
    });
    expect(mockLiteRTLM.loadModel).not.toHaveBeenCalled();

    await TestRenderer.act(async () => {
      await hookResult.result.current.load();
    });
    await TestRenderer.act(async () => {
      await hookResult.result.current.load();
    });
    expect(mockLiteRTLM.loadModel).toHaveBeenCalledTimes(2);
    expect(hookResult.result.current.isReady).toBe(true);
  });
});
