import { createLLM } from '../modelFactory';
import { mockLiteRTLM } from '../__mocks__/react-native-nitro-modules';

const mockGetHistory = mockLiteRTLM.getHistory as jest.Mock;
const mockExecute = mockLiteRTLM.execute as jest.Mock;

describe('multi-conversation manager', () => {
  let llm: ReturnType<typeof createLLM>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetHistory.mockReturnValue([]);
    llm = createLLM();
  });

  it('does not touch resetConversation when no conversations are created', async () => {
    await llm.execute([{ type: 'text', text: 'hi' }]);
    expect(mockLiteRTLM.resetConversation).not.toHaveBeenCalled();
  });

  it('activates a conversation on first execute (empty transcript, no seed)', async () => {
    const conv = llm.createConversation();
    await conv.execute([{ type: 'text', text: 'hi' }]);

    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledTimes(1);
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, undefined, undefined);
    expect(mockLiteRTLM.execute).toHaveBeenCalledTimes(1);
  });

  it('passes the per-conversation systemPrompt override on switch', async () => {
    const conv = llm.createConversation({ systemPrompt: 'You are pirate.' });
    await conv.execute([{ type: 'text', text: 'hi' }]);

    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, 'You are pirate.', undefined);
  });

  it('passes the per-conversation speculative decoding switch', async () => {
    const conv = llm.createConversation({ speculativeDecoding: true });
    await conv.execute([{ type: 'text', text: 'hi' }]);

    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, undefined, true);
  });

  it('passes speculative decoding through a manual resetConversation()', () => {
    llm.resetConversation(undefined, 'Be brief.', true);
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(undefined, 'Be brief.', true);
  });

  it('drops speculative decoding for a loaded bundle without a drafter', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    await llm.loadModel('/m/plain.litertlm', { backend: 'cpu' }); // the mock bundle declares no drafter
    const conv = llm.createConversation({ speculativeDecoding: true });
    await conv.execute([{ type: 'text', text: 'hi' }]);

    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(undefined, undefined, undefined);
    expect(warn.mock.calls.some((c) => String(c[0]).includes('no multi-token-prediction drafter'))).toBe(true);
    warn.mockRestore();
  });

  it('does not reset again for consecutive executes on the same conversation', async () => {
    const conv = llm.createConversation();
    await conv.execute([{ type: 'text', text: 'one' }]);
    await conv.execute([{ type: 'text', text: 'two' }]);

    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledTimes(1);
    expect(mockLiteRTLM.execute).toHaveBeenCalledTimes(2);
  });

  it('replays the saved transcript when switching back to a conversation', async () => {
    const historyA = [
      { role: 'user', content: 'question A' },
      { role: 'model', content: 'answer A' },
    ];
    const convA = llm.createConversation();
    const convB = llm.createConversation();

    await convA.execute([{ type: 'text', text: 'question A' }]);

    // Switching to B snapshots A's native history…
    mockGetHistory.mockReturnValueOnce(historyA);
    await convB.execute([{ type: 'text', text: 'question B' }]);
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(undefined, undefined, undefined);

    // …and switching back to A replays that snapshot.
    mockGetHistory.mockReturnValueOnce([
      { role: 'user', content: 'question B' },
    ]);
    await convA.execute([{ type: 'text', text: 'follow-up A' }]);
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(
      JSON.stringify(historyA),
      undefined,
      undefined,
    );
  });

  it('treats top-level execute as the default conversation once handles exist', async () => {
    const conv = llm.createConversation();
    await conv.execute([{ type: 'text', text: 'in conversation' }]);

    mockGetHistory.mockReturnValueOnce([
      { role: 'user', content: 'in conversation' },
    ]);
    await llm.execute([{ type: 'text', text: 'top level' }]);

    // Default conversation had no prior transcript → plain reset.
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(undefined, undefined, undefined);
    expect(mockLiteRTLM.execute).toHaveBeenCalledTimes(2);
  });

  it('getHistory returns the live native history when active, snapshot otherwise', async () => {
    const historyA = [{ role: 'user', content: 'a' }];
    const convA = llm.createConversation();
    const convB = llm.createConversation();

    await convA.execute([{ type: 'text', text: 'a' }]);
    mockGetHistory.mockReturnValue(historyA);
    expect(convA.getHistory()).toEqual(historyA); // live

    await convB.execute([{ type: 'text', text: 'b' }]);
    mockGetHistory.mockReturnValue([]);
    expect(convA.getHistory()).toEqual(historyA); // snapshot survives the switch
  });

  it('release() hands the context back to the default conversation and rejects further use', async () => {
    const conv = llm.createConversation();
    await conv.execute([{ type: 'text', text: 'hi' }]);

    await conv.release();
    // Releasing the active conversation switches back to default.
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledTimes(2);

    await expect(conv.execute([{ type: 'text', text: 'again' }])).rejects.toThrow(
      /has been released/,
    );
    expect(() => conv.getHistory()).toThrow(/has been released/);
  });

  it('serializes interleaved executes so switches never overlap generations', async () => {
    const order: string[] = [];
    mockExecute.mockImplementation(async (parts: Array<{ text?: string }>) => {
      order.push(parts[0]?.text ?? '');
      return 'ok';
    });

    const convA = llm.createConversation();
    const convB = llm.createConversation();
    await Promise.all([
      convA.execute([{ type: 'text', text: '1-A' }]),
      convB.execute([{ type: 'text', text: '2-B' }]),
      convA.execute([{ type: 'text', text: '3-A' }]),
    ]);

    expect(order).toEqual(['1-A', '2-B', '3-A']);
    // Three switches: →A, →B, →A (no dedup possible with interleaving).
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledTimes(3);
  });

  it('loadModel clears saved transcripts and the active conversation', async () => {
    const conv = llm.createConversation();
    await conv.execute([{ type: 'text', text: 'hi' }]);

    await llm.loadModel('https://example.com/model.litertlm');

    mockLiteRTLM.resetConversation.mockClear();
    await conv.execute([{ type: 'text', text: 'after reload' }]);
    // Fresh manager state: the conversation re-activates with no seed.
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, undefined, undefined);
  });

  describe('native failures and manual resets', () => {
    const mockReset = mockLiteRTLM.resetConversation as jest.Mock;
    /** What Android throws when no model is loaded (or an unload is under way). */
    const notLoaded = () => {
      throw new Error('[LiteRTLM:execute:FAILED_PRECONDITION:MODEL_NOT_LOADED] No model loaded. Call loadModel() first.');
    };

    it('rejects a switch that native refuses with a typed error, and switches again next time', async () => {
      const conv = llm.createConversation();
      mockReset.mockImplementationOnce(notLoaded);
      await expect(conv.execute([{ type: 'text', text: 'hi' }])).rejects.toMatchObject({ code: 'MODEL_NOT_LOADED' });
      expect(mockExecute).not.toHaveBeenCalled();

      await conv.execute([{ type: 'text', text: 'again' }]);
      expect(mockReset).toHaveBeenCalledTimes(2);
      expect(mockExecute).toHaveBeenCalledTimes(1);
    });

    it("doesn't save what a failed switch left behind as the outgoing conversation's transcript", async () => {
      const historyA = [
        { role: 'user', content: 'question A' },
        { role: 'model', content: 'answer A' },
      ];
      const convA = llm.createConversation();
      const convB = llm.createConversation();
      await convA.execute([{ type: 'text', text: 'question A' }]);

      // The switch to B saves A's turns; native then fails, after replacing
      // its transcript (as Android does)
      mockGetHistory.mockReturnValueOnce(historyA);
      mockReset.mockImplementationOnce(notLoaded);
      await expect(convB.execute([{ type: 'text', text: 'question B' }])).rejects.toMatchObject({
        code: 'MODEL_NOT_LOADED',
      });

      // The retry saves nothing over A's transcript…
      mockGetHistory.mockReturnValue([{ role: 'user', content: 'what native left' }]);
      await convB.execute([{ type: 'text', text: 'question B' }]);
      // …so switching back to A replays A's own turns
      await convA.execute([{ type: 'text', text: 'follow-up A' }]);
      expect(mockReset).toHaveBeenLastCalledWith(JSON.stringify(historyA), undefined, undefined);
    });

    it('release() releases the handle even when native refuses the hand-back', async () => {
      const conv = llm.createConversation();
      await conv.execute([{ type: 'text', text: 'hi' }]); // conv holds the context
      mockReset.mockImplementationOnce(notLoaded); // e.g. an unload under way on Android

      await expect(conv.release()).resolves.toBeUndefined();
      expect(() => conv.getHistory()).toThrow(/has been released/);

      // Nothing holds the context: the default conversation's next message replays it
      mockReset.mockClear();
      await llm.execute([{ type: 'text', text: 'top level' }]);
      expect(mockReset).toHaveBeenCalledWith(undefined, undefined, undefined);
    });

    it('a manual reset keeps the turns of the conversation that held the context', async () => {
      const turns = [
        { role: 'user', content: 'question' },
        { role: 'model', content: 'answer' },
      ];
      const conv = llm.createConversation();
      await conv.execute([{ type: 'text', text: 'question' }]);

      mockGetHistory.mockReturnValue(turns);
      llm.resetConversation(); // the default conversation takes the context
      mockGetHistory.mockReturnValue([]);
      expect(conv.getHistory()).toEqual(turns);

      await conv.execute([{ type: 'text', text: 'follow-up' }]);
      expect(mockReset).toHaveBeenLastCalledWith(JSON.stringify(turns), undefined, undefined);
    });

    it.each([
      ['user turn', [{ role: 'user', content: 'in flight' }]],
      [
        'tool results',
        [
          { role: 'tool', content: '{"temp":21}', toolName: 'weather' },
          { role: 'tool', content: '{"rain":false}', toolName: 'rain' },
        ],
      ],
    ])("leaves out the %s of the handle's message that a manual reset cancels", async (_, unanswered) => {
      const answered = [
        { role: 'user', content: 'question' },
        { role: 'model', content: 'answer' },
      ];
      const conv = llm.createConversation();
      await conv.execute([{ type: 'text', text: 'question' }]);

      // Android has written the turns of the message still generating
      mockGetHistory.mockReturnValue([...answered, ...unanswered]);
      llm.resetConversation();
      mockGetHistory.mockReturnValue([]);
      expect(conv.getHistory()).toEqual(answered);
    });

    it('switching back to the default conversation keeps the options of its last manual reset', async () => {
      llm.resetConversation(undefined, 'You are a pirate.', true);
      const conv = llm.createConversation();
      await conv.execute([{ type: 'text', text: 'in conversation' }]);

      await llm.execute([{ type: 'text', text: 'top level' }]);
      expect(mockReset).toHaveBeenLastCalledWith(undefined, 'You are a pirate.', true);
    });

    it("a load forgets the default conversation's reset options", async () => {
      llm.resetConversation(undefined, 'You are a pirate.');
      await llm.loadModel('https://example.com/model.litertlm');
      const conv = llm.createConversation();
      await conv.execute([{ type: 'text', text: 'in conversation' }]);

      await llm.execute([{ type: 'text', text: 'top level' }]);
      expect(mockReset).toHaveBeenLastCalledWith(undefined, undefined, undefined);
    });
  });
});
