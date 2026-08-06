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
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, undefined);
    expect(mockLiteRTLM.execute).toHaveBeenCalledTimes(1);
  });

  it('passes the per-conversation systemPrompt override on switch', async () => {
    const conv = llm.createConversation({ systemPrompt: 'You are pirate.' });
    await conv.execute([{ type: 'text', text: 'hi' }]);

    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, 'You are pirate.');
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
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(undefined, undefined);

    // …and switching back to A replays that snapshot.
    mockGetHistory.mockReturnValueOnce([
      { role: 'user', content: 'question B' },
    ]);
    await convA.execute([{ type: 'text', text: 'follow-up A' }]);
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(
      JSON.stringify(historyA),
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
    expect(mockLiteRTLM.resetConversation).toHaveBeenLastCalledWith(undefined, undefined);
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
    expect(mockLiteRTLM.resetConversation).toHaveBeenCalledWith(undefined, undefined);
  });
});
