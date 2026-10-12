/**
 * Incremental parser that splits a raw LLM token stream into typed events.
 *
 * With `streamToolCalls` enabled (LiteRT-LM v0.14+), tool-call — and, for
 * thinking models, reasoning — content arrives inline in the token stream,
 * wrapped in channel markers (e.g. `<tool_call>…</tool_call>`). This parser
 * turns that interleaved stream into typed events so apps never have to
 * regex the raw text.
 *
 * Pure and incremental: feed it chunks as they arrive; markers split across
 * chunk boundaries are handled by buffering the longest possible marker prefix.
 */

/** Kind of streamed content. */
export type StreamEventType = "token" | "toolCall" | "thinking";

/** A typed streaming event. */
export interface StreamEvent {
  /** What this chunk is: plain response text, tool-call content, or reasoning */
  type: StreamEventType;
  /** The text content of this chunk */
  text: string;
  /** True exactly once, on the final event of the stream */
  done: boolean;
}

/** A channel definition: content between `start` and `end` gets `type`. */
export interface StreamChannel {
  type: Exclude<StreamEventType, "token">;
  start: string;
  end: string;
}

/** Default channel markers (override via `createLLM({ streamChannels })`). */
export const DEFAULT_CHANNELS: StreamChannel[] = [
  { type: "toolCall", start: "<tool_call>", end: "</tool_call>" },
  { type: "thinking", start: "<thinking>", end: "</thinking>" },
];

export interface StreamEventParser {
  /**
   * Feed a raw chunk; returns zero or more complete events.
   * Text that could be the start of a channel marker is held back until
   * disambiguated by the next chunk (or by `finish()`).
   */
  push(chunk: string): StreamEvent[];
  /**
   * Flush any held-back text and emit the final `done` event.
   * An unterminated channel is flushed as content of that channel.
   */
  finish(): StreamEvent[];
}

/**
 * Create an incremental stream-event parser.
 */
export function createStreamEventParser(
  channels: StreamChannel[] = DEFAULT_CHANNELS,
): StreamEventParser {
  /** Channel we are currently inside, or null for plain tokens. */
  let active: StreamChannel | null = null;
  /** Unprocessed tail (may contain a partial marker). */
  let buffer = "";
  let finished = false;

  /** Longest marker length — upper bound on how much tail we must hold back. */
  const maxMarkerLen = channels.reduce(
    (m, c) => Math.max(m, c.start.length, c.end.length),
    0,
  );

  /** Length of the longest suffix of `text` that is a proper prefix of `marker`. */
  const partialSuffixLen = (text: string, marker: string): number => {
    const maxLen = Math.min(text.length, marker.length - 1);
    for (let len = maxLen; len > 0; len--) {
      if (text.endsWith(marker.slice(0, len))) return len;
    }
    return 0;
  };

  /** How many trailing chars might be an incomplete marker and must be held. */
  const holdbackLen = (text: string): number => {
    if (maxMarkerLen === 0) return 0;
    const markers = active
      ? [active.end]
      : channels.map((c) => c.start);
    let hold = 0;
    for (const m of markers) {
      hold = Math.max(hold, partialSuffixLen(text, m));
    }
    return hold;
  };

  const emit = (
    events: StreamEvent[],
    type: StreamEventType,
    text: string,
    done = false,
  ) => {
    if (text.length > 0 || done) events.push({ type, text, done });
  };

  const process = (flush: boolean): StreamEvent[] => {
    const events: StreamEvent[] = [];

    // Repeatedly scan for the next relevant marker in the buffer
    for (;;) {
      if (active === null) {
        // Look for the earliest channel-start marker
        let foundIdx = -1;
        let foundChannel: StreamChannel | null = null;
        for (const c of channels) {
          const idx = buffer.indexOf(c.start);
          if (idx !== -1 && (foundIdx === -1 || idx < foundIdx)) {
            foundIdx = idx;
            foundChannel = c;
          }
        }
        if (foundChannel === null) break;
        emit(events, "token", buffer.slice(0, foundIdx));
        buffer = buffer.slice(foundIdx + foundChannel.start.length);
        active = foundChannel;
      } else {
        const idx = buffer.indexOf(active.end);
        if (idx === -1) break;
        emit(events, active.type, buffer.slice(0, idx));
        buffer = buffer.slice(idx + active.end.length);
        active = null;
      }
    }

    // Emit what we can, holding back a possible partial marker
    const hold = flush ? 0 : holdbackLen(buffer);
    const emittable = buffer.slice(0, buffer.length - hold);
    buffer = buffer.slice(buffer.length - hold);
    emit(events, active ? active.type : "token", emittable);

    return events;
  };

  return {
    push(chunk: string): StreamEvent[] {
      if (finished) return [];
      buffer += chunk;
      return process(false);
    },

    finish(): StreamEvent[] {
      if (finished) return [];
      finished = true;
      const events = process(true);
      events.push({ type: "token", text: "", done: true });
      return events;
    },
  };
}
