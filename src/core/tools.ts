/**
 * Tool calling — both halves of an agent loop: read the tools the model wants
 * to run, and send their results back. Pure and react-native-free.
 *
 * Declare tools in `LLMConfig.tools`. When the model calls one, the response
 * (and the streamed `toolCall` events) carry `<tool_call>{json}</tool_call>`
 * markers; {@link parseToolCalls} turns them into objects. Answer with one
 * {@link toolResultPart} per call, all in a single `execute()`, and the model
 * continues from the results.
 *
 * @example
 * ```ts
 * let reply = await llm.execute([{ type: "text", text: "Is it warmer in Paris or Rome?" }]);
 * for (let step = 0; step < 5; step++) {
 *   const calls = parseToolCalls(reply);
 *   if (calls.length === 0) break;
 *   reply = await llm.execute(
 *     await Promise.all(calls.map(async (call) => toolResultPart(call.name, await tools[call.name](call.arguments)))),
 *   );
 * }
 * console.log(stripToolCalls(reply));
 * ```
 */
import type { Message, MultimodalPart } from "../specs/LiteRTLM.nitro";

/** A tool call with its arguments parsed. */
export interface ToolCallRequest {
  /** The tool the model wants to run */
  name: string;
  /** Its arguments, as the model wrote them (integers stay integers) */
  arguments: Record<string, unknown>;
}

const CALL_START = "<tool_call>";
const CALL_END = "</tool_call>";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

function toRequest(name: unknown, args: unknown): ToolCallRequest | null {
  if (typeof name !== "string" || name.length === 0) return null;
  return { name, arguments: isRecord(args) ? args : {} };
}

/** Visit each `<tool_call>…</tool_call>` marker: its body, and its span in `text`. */
function eachMarker(text: string, visit: (body: string, start: number, end: number) => void): void {
  let from = 0;
  for (;;) {
    const start = text.indexOf(CALL_START, from);
    if (start < 0) return;
    const end = text.indexOf(CALL_END, start + CALL_START.length);
    if (end < 0) return;
    visit(text.slice(start + CALL_START.length, end), start, end + CALL_END.length);
    from = end + CALL_END.length;
  }
}

/**
 * The tool calls in a model response: the string `execute()` resolves with,
 * or a `'model'` turn from `getHistory()`. A call whose JSON doesn't parse is
 * skipped; its marker stays in the text.
 */
export function parseToolCalls(source: string | Message): ToolCallRequest[] {
  const calls: ToolCallRequest[] = [];
  if (typeof source !== "string") {
    for (const { name, argumentsJson } of source.toolCalls ?? []) {
      let args: unknown;
      try {
        args = JSON.parse(argumentsJson);
      } catch {
        args = {};
      }
      const call = toRequest(name, args);
      if (call) calls.push(call);
    }
    return calls;
  }
  eachMarker(source, (body) => {
    try {
      const parsed: unknown = JSON.parse(body);
      const call = isRecord(parsed) ? toRequest(parsed.name, parsed.arguments) : null;
      if (call) calls.push(call);
    } catch {
      // malformed — leave it for the caller to see in the text
    }
  });
  return calls;
}

/** A response without its tool-call markers — the text to show a user. */
export function stripToolCalls(response: string): string {
  let out = "";
  let last = 0;
  eachMarker(response, (_body, start, end) => {
    out += response.slice(last, start);
    last = end;
  });
  return (out + response.slice(last)).trim();
}

/**
 * A tool's result, to send back with `execute()`. Send one part per call the
 * model made, all in one `execute()` — a message of tool results takes no
 * other parts. `response` is any JSON value: an object reaches the model
 * field by field; anything else is wrapped as `{value: …}` by Gemma's template.
 */
export function toolResultPart(name: string, response: unknown): MultimodalPart {
  return { type: "toolResult", name, responseJson: JSON.stringify(response) ?? "null" };
}
