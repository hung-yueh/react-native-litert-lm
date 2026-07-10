/**
 * Tests for the incremental stream-event parser that turns raw LiteRT-LM
 * token streams into typed token / toolCall / thinking events.
 */
import {
  createStreamEventParser,
  DEFAULT_CHANNELS,
  type StreamEvent,
} from "../streamEvents";

/** Run a whole stream through the parser and collect every event. */
function collect(chunks: string[]): StreamEvent[] {
  const parser = createStreamEventParser();
  const events: StreamEvent[] = [];
  for (const chunk of chunks) {
    events.push(...parser.push(chunk));
  }
  events.push(...parser.finish());
  return events;
}

/** Re-join events of one type. */
function textOf(events: StreamEvent[], type: string): string {
  return events
    .filter((e) => e.type === type)
    .map((e) => e.text)
    .join("");
}

describe("createStreamEventParser — plain streams", () => {
  it("passes plain tokens through unchanged", () => {
    const events = collect(["Hello ", "world"]);
    expect(textOf(events, "token")).toBe("Hello world");
    expect(events.filter((e) => e.type !== "token")).toHaveLength(0);
  });

  it("emits exactly one done event, last", () => {
    const events = collect(["Hi"]);
    expect(events.filter((e) => e.done)).toHaveLength(1);
    expect(events[events.length - 1]).toEqual({
      type: "token",
      text: "",
      done: true,
    });
  });

  it("does not treat a lone '<' or non-marker tags as channels", () => {
    const events = collect(["1 < 2 ", "and <b>bold</b>"]);
    expect(textOf(events, "token")).toBe("1 < 2 and <b>bold</b>");
  });
});

describe("createStreamEventParser — channel extraction", () => {
  it("splits tool-call content out of the stream", () => {
    const events = collect([
      'Sure: <tool_call>{"name":"getWeather"}</tool_call> done',
    ]);
    expect(textOf(events, "token")).toBe("Sure:  done");
    expect(textOf(events, "toolCall")).toBe('{"name":"getWeather"}');
  });

  it("splits thinking content out of the stream", () => {
    const events = collect([
      "<thinking>step 1: consider</thinking>The answer is 42.",
    ]);
    expect(textOf(events, "thinking")).toBe("step 1: consider");
    expect(textOf(events, "token")).toBe("The answer is 42.");
  });

  it("handles markers split across chunk boundaries", () => {
    const events = collect([
      "call: <tool",
      "_call>{\"a\":",
      "1}</tool_",
      "call> after",
    ]);
    expect(textOf(events, "token")).toBe("call:  after");
    expect(textOf(events, "toolCall")).toBe('{"a":1}');
  });

  it("handles one-character-at-a-time streaming", () => {
    const raw = "x<tool_call>y</tool_call>z";
    const events = collect(raw.split(""));
    expect(textOf(events, "token")).toBe("xz");
    expect(textOf(events, "toolCall")).toBe("y");
  });

  it("handles multiple channels interleaved in one stream", () => {
    const events = collect([
      "<thinking>hm</thinking>a<tool_call>t1</tool_call>b<tool_call>t2</tool_call>",
    ]);
    expect(textOf(events, "thinking")).toBe("hm");
    expect(textOf(events, "token")).toBe("ab");
    expect(textOf(events, "toolCall")).toBe("t1t2");
  });

  it("flushes an unterminated channel as channel content on finish", () => {
    const events = collect(["<tool_call>{\"partial\":true}"]);
    expect(textOf(events, "toolCall")).toBe('{"partial":true}');
    expect(events[events.length - 1]!.done).toBe(true);
  });

  it("does not emit held-back partial markers as text mid-stream", () => {
    const parser = createStreamEventParser();
    const first = parser.push("text <tool");
    // "<tool" could become "<tool_call>" — must not leak into token text yet
    expect(first.map((e) => e.text).join("")).toBe("text ");
    const rest = [...parser.push("box>"), ...parser.finish()];
    // "<toolbox>" was not a marker after all — text is preserved
    expect(rest.map((e) => e.text).join("")).toBe("<toolbox>");
  });
});

describe("createStreamEventParser — custom channels", () => {
  it("supports custom marker definitions", () => {
    const parser = createStreamEventParser([
      { type: "thinking", start: "<start_of_thought>", end: "<end_of_thought>" },
    ]);
    const events = [
      ...parser.push("<start_of_thought>deep<end_of_thought>shallow"),
      ...parser.finish(),
    ];
    expect(textOf(events, "thinking")).toBe("deep");
    expect(textOf(events, "token")).toBe("shallow");
  });

  it("exports sane defaults", () => {
    expect(DEFAULT_CHANNELS.map((c) => c.type).sort()).toEqual([
      "thinking",
      "toolCall",
    ]);
  });

  it("ignores pushes after finish", () => {
    const parser = createStreamEventParser();
    parser.push("a");
    parser.finish();
    expect(parser.push("b")).toEqual([]);
    expect(parser.finish()).toEqual([]);
  });
});
