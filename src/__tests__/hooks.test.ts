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
  let result = { current: null as unknown as R };
  
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
    expect(mockLiteRTLM.sendMessageAsync).toHaveBeenCalled();
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

  describe('nativeConfig field forwarding', () => {
    async function loadWithConfig(config: any) {
      let hookResult: any;
      await TestRenderer.act(async () => {
        hookResult = renderHook(() => useModel('https://example.com/model.litertlm', config));
      });
      // loadModel is called as: loadModel(pathOrUrl, nativeConfig, onProgress)
      const [, nativeConfig] = mockLiteRTLM.loadModel.mock.calls[0];
      return nativeConfig;
    }

    it('should forward validate to nativeConfig', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true, validate: true });
      expect(nativeConfig).toHaveProperty('validate', true);
    });

    it('should forward multimodal to nativeConfig', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true, multimodal: true });
      expect(nativeConfig).toHaveProperty('multimodal', true);
    });

    it('should forward enableSpeculativeDecoding to nativeConfig', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true, enableSpeculativeDecoding: true });
      expect(nativeConfig).toHaveProperty('enableSpeculativeDecoding', true);
    });

    it('should forward tools array to nativeConfig', async () => {
      const tools = [{ name: 'get_weather', description: 'Gets weather', parametersJson: '{}' }];
      const nativeConfig = await loadWithConfig({ autoLoad: true, tools });
      expect(nativeConfig).toHaveProperty('tools');
      expect(nativeConfig.tools).toEqual(tools);
    });

    it('should omit validate from nativeConfig when not provided', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true });
      expect(nativeConfig).not.toHaveProperty('validate');
    });

    it('should omit multimodal from nativeConfig when not provided', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true });
      expect(nativeConfig).not.toHaveProperty('multimodal');
    });

    it('should omit enableSpeculativeDecoding from nativeConfig when not provided', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true });
      expect(nativeConfig).not.toHaveProperty('enableSpeculativeDecoding');
    });

    it('should omit tools from nativeConfig when not provided', async () => {
      const nativeConfig = await loadWithConfig({ autoLoad: true });
      expect(nativeConfig).not.toHaveProperty('tools');
    });

    it('should not reload the model when tools array is a new reference with the same content (toolsKey stability)', async () => {
      const tools = [{ name: 'get_weather', description: 'Gets weather', parametersJson: '{}' }];
      let hookResult: any;

      await TestRenderer.act(async () => {
        hookResult = renderHook(
          (props: any) => useModel('https://example.com/model.litertlm', props),
          { autoLoad: true, tools },
        );
      });

      const callCountAfterMount = mockLiteRTLM.loadModel.mock.calls.length;

      // Re-render with a new array reference that has identical content
      await TestRenderer.act(async () => {
        hookResult.rerender({
          autoLoad: true,
          tools: [{ name: 'get_weather', description: 'Gets weather', parametersJson: '{}' }],
        });
      });

      expect(mockLiteRTLM.loadModel.mock.calls.length).toBe(callCountAfterMount);
    });

    it('should reload the model when tools content actually changes', async () => {
      const tools = [{ name: 'get_weather', description: 'Gets weather', parametersJson: '{}' }];
      let hookResult: any;

      await TestRenderer.act(async () => {
        hookResult = renderHook(
          (props: any) => useModel('https://example.com/model.litertlm', props),
          { autoLoad: true, tools },
        );
      });

      const callCountAfterMount = mockLiteRTLM.loadModel.mock.calls.length;

      // Re-render with different tools content
      await TestRenderer.act(async () => {
        hookResult.rerender({
          autoLoad: true,
          tools: [{ name: 'search', description: 'Searches the web', parametersJson: '{}' }],
        });
      });

      expect(mockLiteRTLM.loadModel.mock.calls.length).toBeGreaterThan(callCountAfterMount);
    });
  });
});
