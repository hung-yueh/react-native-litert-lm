/**
 * Tool-calling helpers: reading the model's calls (from a response or a
 * history turn) and building the tool-result parts that answer them.
 */
import { parseToolCalls, stripToolCalls, toolResultPart } from "../core/tools";
import type { Message } from "../specs/LiteRTLM.nitro";

const reply =
  'Let me check both. <tool_call>{"name":"get_weather","arguments":{"city":"Paris","days":3}}</tool_call>' +
  '<tool_call>{"name":"get_weather","arguments":{"city":"Rome"}}</tool_call>';

describe("parseToolCalls", () => {
  it("reads every call in a response, integers intact", () => {
    expect(parseToolCalls(reply)).toEqual([
      { name: "get_weather", arguments: { city: "Paris", days: 3 } },
      { name: "get_weather", arguments: { city: "Rome" } },
    ]);
    expect(Number.isInteger(parseToolCalls(reply)[0].arguments.days)).toBe(true);
  });

  it("returns nothing for a plain answer", () => {
    expect(parseToolCalls("It is 18°C in Paris.")).toEqual([]);
  });

  it("skips malformed or nameless calls and defaults missing arguments", () => {
    expect(
      parseToolCalls(
        '<tool_call>{"name":</tool_call><tool_call>{"arguments":{}}</tool_call>' +
          '<tool_call>{"name":"now"}</tool_call><tool_call>{"name":"x"',
      ),
    ).toEqual([{ name: "now", arguments: {} }]);
  });

  it("reads a model turn from getHistory()", () => {
    const turn: Message = {
      role: "model",
      content: "Let me check.",
      toolCalls: [
        { name: "get_weather", argumentsJson: '{"city":"Paris"}' },
        { name: "broken", argumentsJson: "{" },
      ],
    };
    expect(parseToolCalls(turn)).toEqual([
      { name: "get_weather", arguments: { city: "Paris" } },
      { name: "broken", arguments: {} },
    ]);
    expect(parseToolCalls({ role: "user", content: "hi" })).toEqual([]);
  });
});

describe("stripToolCalls", () => {
  it("leaves the text a user should see", () => {
    expect(stripToolCalls(reply)).toBe("Let me check both.");
    expect(stripToolCalls("No tools here.")).toBe("No tools here.");
    expect(stripToolCalls("Unclosed <tool_call>{")).toBe("Unclosed <tool_call>{");
  });
});

describe("toolResultPart", () => {
  it("serializes any JSON value", () => {
    expect(toolResultPart("get_weather", { city: "Paris", temperature: 18 })).toEqual({
      type: "toolResult",
      name: "get_weather",
      responseJson: '{"city":"Paris","temperature":18}',
    });
    expect(toolResultPart("echo", "sunny").responseJson).toBe('"sunny"');
    expect(toolResultPart("noop", undefined).responseJson).toBe("null");
  });
});
