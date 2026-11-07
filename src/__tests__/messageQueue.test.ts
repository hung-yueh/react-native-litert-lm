import { createLLM } from '../modelFactory';
import { mockLiteRTLM, mockModelStore } from '../__mocks__/react-native-nitro-modules';
import type { MultimodalPart } from '../specs/LiteRTLM.nitro';
import type { StreamEvent } from '../core/streamEvents';

type TokenCallback = (token: string, done: boolean) => void;

const mockExecute = mockLiteRTLM.execute as jest.Mock;
const mockReset = mockLiteRTLM.resetConversation as jest.Mock;
const mockUnload = mockLiteRTLM.unload as jest.Mock;
const mockClose = mockLiteRTLM.close as jest.Mock;
const mockLoad = mockLiteRTLM.loadModel as jest.Mock;
const mockGetHistory = mockLiteRTLM.getHistory as jest.Mock;

// The mocks' own behavior, put back after each test
const defaults = [mockExecute, mockReset, mockUnload, mockClose, mockLoad].map(
  (mock) => [mock, mock.getMockImplementation()] as const,
);

const text = (t: string): MultimodalPart[] => [{ type: 'text', text: t }];
/** Let every pending promise callback run. */
const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * A fake engine whose messages generate until the test ends them. Like the
 * real one, resetConversation(), loadModel(), unload() and close() cancel
 * the message generating: it rejects with the native CANCELLED status.
 */
function holdGenerations() {
  type Generation = { text: string; finish: (reply?: string) => void; fail: (message: string) => void };
  const generating: Generation[] = [];
  /** Messages that reached native, in order */
  const started: string[] = [];
  let mostAtOnce = 0;

  mockExecute.mockImplementation(
    (parts: MultimodalPart[], onToken?: TokenCallback) =>
      new Promise<string>((resolve, reject) => {
        const message = parts[0]?.text ?? '';
        const end = () => {
          generating.splice(generating.indexOf(generation), 1);
        };
        const generation: Generation = {
          text: message,
          finish: (reply = `reply to ${message}`) => {
            end();
            onToken?.(reply, true);
            resolve(reply);
          },
          fail: (error) => {
            end();
            reject(new Error(error));
          },
        };
        started.push(message);
        generating.push(generation);
        mostAtOnce = Math.max(mostAtOnce, generating.length);
      }),
  );
  const cancel = () => {
    for (const generation of [...generating]) generation.fail('[LiteRTLM:execute:CANCELLED] Cancelled');
  };
  mockReset.mockImplementation(cancel);
  mockUnload.mockImplementation(async () => cancel());
  mockClose.mockImplementation(cancel);
  mockLoad.mockImplementation(async () => cancel());

  const find = (message: string) => {
    const generation = generating.find((g) => g.text === message);
    if (!generation) {
      throw new Error(`'${message}' is not generating (generating: ${generating.map((g) => g.text).join(', ') || 'nothing'})`);
    }
    return generation;
  };
  return {
    started,
    get mostAtOnce() {
      return mostAtOnce;
    },
    finish: (message: string, reply?: string) => find(message).finish(reply),
    fail: (message: string, error: string) => find(message).fail(error),
  };
}

describe('inference queue', () => {
  let llm: ReturnType<typeof createLLM>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetHistory.mockReturnValue([]);
    llm = createLLM();
  });

  afterEach(() => {
    for (const [mock, implementation] of defaults) {
      if (implementation) mock.mockImplementation(implementation);
      else mock.mockReset();
    }
  });

  it('runs overlapping execute() calls one at a time, each resolving with its own reply', async () => {
    const engine = holdGenerations();
    const a = llm.execute(text('A'));
    const b = llm.execute(text('B'));
    await flush();
    expect(engine.started).toEqual(['A']); // B waits for A

    engine.finish('A', 'reply A');
    await flush();
    expect(engine.started).toEqual(['A', 'B']);
    engine.finish('B', 'reply B');

    await expect(a).resolves.toBe('reply A');
    await expect(b).resolves.toBe('reply B');
    expect(engine.mostAtOnce).toBe(1);
  });

  it("starts a message in the caller's tick when nothing is generating", () => {
    const engine = holdGenerations();
    void llm.execute(text('A'));
    expect(engine.started).toEqual(['A']);
    engine.finish('A');
  });

  it('does not start a blocking message while a streamed one is still generating', async () => {
    const engine = holdGenerations();
    const tokens: string[] = [];
    const streamed = llm.execute(text('A'), (token) => tokens.push(token));
    const blocking = llm.execute(text('B'));
    await flush();
    expect(engine.started).toEqual(['A']);

    engine.finish('A', 'streamed reply');
    await expect(streamed).resolves.toBe('streamed reply');
    await flush();
    engine.finish('B', 'blocking reply');
    await expect(blocking).resolves.toBe('blocking reply');
    expect(tokens).toEqual(['streamed reply']);
    expect(engine.mostAtOnce).toBe(1);
  });

  it('queues executeWithEvents() and the deprecated send methods with execute()', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    const engine = holdGenerations();
    const events: StreamEvent[] = [];
    const a = llm.executeWithEvents(text('A'), (event) => events.push(event));
    const b = llm.sendMessage('B');
    const c = llm.sendMessageAsync('C', () => {});
    await flush();
    expect(engine.started).toEqual(['A']);

    engine.finish('A', 'events reply');
    await flush();
    engine.finish('B');
    await flush();
    engine.finish('C');

    await expect(a).resolves.toBe('events reply');
    await expect(b).resolves.toBe('reply to B');
    await expect(c).resolves.toBeUndefined();
    expect(events.map((event) => event.text).join('')).toBe('events reply');
    expect(events[events.length - 1]?.done).toBe(true);
    expect(engine.started).toEqual(['A', 'B', 'C']);
    expect(engine.mostAtOnce).toBe(1);
    warn.mockRestore();
  });

  it('runs the next message after one fails', async () => {
    const engine = holdGenerations();
    const a = llm.execute(text('A'));
    const b = llm.execute(text('B'));
    engine.fail('A', '[LiteRTLM:execute:INTERNAL] Decode failed');
    await expect(a).rejects.toMatchObject({ code: 'INFERENCE_FAILED' });

    await flush();
    engine.finish('B');
    await expect(b).resolves.toBe('reply to B');
  });

  describe('cancelling', () => {
    it('resetConversation() cancels the message generating and the ones waiting behind it', async () => {
      const engine = holdGenerations();
      const onToken = jest.fn();
      const a = llm.execute(text('A'));
      const b = llm.execute(text('B'));
      const c = llm.execute(text('C'), onToken);
      llm.resetConversation();

      await expect(a).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      await expect(b).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      await expect(c).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      expect(engine.started).toEqual(['A']); // B and C never reached native
      expect(onToken).not.toHaveBeenCalled();
    });

    it('runs a message sent after the reset, once the cancelled one has returned', async () => {
      const engine = holdGenerations();
      // The cancel lands later, as a prefill runs to its end before it stops
      mockReset.mockImplementation(() => {});
      const a = llm.execute(text('A'));
      llm.resetConversation();
      const b = llm.execute(text('B'));
      await flush();
      expect(engine.started).toEqual(['A']); // B waits until A has returned

      engine.fail('A', '[LiteRTLM:execute:CANCELLED] Cancelled');
      await expect(a).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      await flush();
      expect(engine.started).toEqual(['A', 'B']);
      engine.finish('B');
      await expect(b).resolves.toBe('reply to B');
    });

    it('unload() cancels the waiting messages, and messages run again after the next load', async () => {
      const engine = holdGenerations();
      const a = llm.execute(text('A'));
      const b = llm.execute(text('B'));
      await llm.unload();
      await expect(a).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      await expect(b).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      expect(engine.started).toEqual(['A']);

      await llm.loadModel('/models/model.litertlm', { backend: 'cpu' });
      const c = llm.execute(text('C'));
      engine.finish('C');
      await expect(c).resolves.toBe('reply to C');
    });

    it('close() cancels the waiting messages, and no message reaches native afterwards', async () => {
      const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
      const engine = holdGenerations();
      const a = llm.execute(text('A'));
      const b = llm.execute(text('B'));
      llm.close();
      await expect(a).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      await expect(b).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });

      await expect(llm.execute(text('C'))).rejects.toMatchObject({ code: 'MODEL_NOT_LOADED' });
      await expect(llm.executeWithEvents(text('D'), () => {})).rejects.toMatchObject({ code: 'MODEL_NOT_LOADED' });
      await expect(llm.sendMessage('E')).rejects.toMatchObject({ code: 'MODEL_NOT_LOADED' });
      expect(engine.started).toEqual(['A']);
      warn.mockRestore();
    });

    it('loadModel() cancels the waiting messages when it replaces the engine', async () => {
      const engine = holdGenerations();
      const a = llm.execute(text('A'));
      const b = llm.execute(text('B'));
      await llm.loadModel('/models/other.litertlm', { backend: 'cpu' });
      await expect(a).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      await expect(b).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      expect(engine.started).toEqual(['A']);
    });

    it('a load refused before it reaches the engine cancels nothing', async () => {
      const engine = holdGenerations();
      // 8 GiB model vs the mock's 4 MB of available memory: refused up front
      mockModelStore.getFileSizeBytes.mockReturnValueOnce(8 * 1024 ** 3);
      const a = llm.execute(text('A'));
      const b = llm.execute(text('B'));
      await expect(llm.loadModel('/models/huge.litertlm', { backend: 'cpu' })).rejects.toMatchObject({
        code: 'MEMORY_ESTIMATE_EXCEEDED',
      });

      engine.finish('A');
      await expect(a).resolves.toBe('reply to A');
      await flush();
      engine.finish('B');
      await expect(b).resolves.toBe('reply to B');
    });
  });

  describe('with conversation handles', () => {
    it("a reset cancels the default conversation's waiting messages, not a handle's", async () => {
      const engine = holdGenerations();
      const conv = llm.createConversation();
      const a = llm.execute(text('A'));
      const b = conv.execute(text('B'));
      const c = llm.execute(text('C'));
      llm.resetConversation();
      await expect(a).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });

      await flush();
      expect(engine.started).toEqual(['A', 'B']); // B runs in its own conversation
      engine.finish('B', 'reply B');
      await expect(b).resolves.toBe('reply B');
      await expect(c).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      expect(engine.started).toEqual(['A', 'B']);
    });

    it.each(['unload', 'close'] as const)('%s() cancels the waiting messages of every conversation', async (teardown) => {
      const engine = holdGenerations();
      const conv = llm.createConversation();
      const onEvent = jest.fn();
      const a = conv.execute(text('A'));
      const b = conv.executeWithEvents(text('B'), onEvent);
      const c = llm.execute(text('C'));
      await llm[teardown]();

      for (const message of [a, b, c]) {
        await expect(message).rejects.toMatchObject({ code: 'INFERENCE_CANCELLED' });
      }
      expect(engine.started).toEqual(['A']);
      expect(onEvent).not.toHaveBeenCalled();
    });

    it('release() on a closed instance makes no native call', async () => {
      const conv = llm.createConversation();
      await conv.execute(text('A')); // conv now holds the native context
      llm.close();
      mockReset.mockClear();
      mockGetHistory.mockClear();

      await conv.release();
      expect(mockReset).not.toHaveBeenCalled();
      expect(mockGetHistory).not.toHaveBeenCalled();
    });

    it('a call on a released handle rejects without switching the native context', async () => {
      const conv = llm.createConversation();
      await conv.execute(text('A'));
      await conv.release();
      mockReset.mockClear();

      await expect(conv.execute(text('B'))).rejects.toThrow(/has been released/);
      expect(mockReset).not.toHaveBeenCalled();
      expect(mockExecute).toHaveBeenCalledTimes(1);
    });
  });
});
