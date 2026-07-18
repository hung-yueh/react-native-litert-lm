/**
 * Contract test: every key declared on the LLMConfig interface must be
 * handled by the useModel hook — either forwarded into the nativeConfig it
 * passes to loadModel, or listed here as deliberately hook-unsupported.
 *
 * This exists because useModel once silently dropped `streamToolCalls` and
 * every tuning knob (fixed in v0.5.x): the hook rebuilds its config from a
 * hand-maintained whitelist for dependency stability, so a key added to the
 * spec is invisible to hook users unless someone remembers to forward it.
 * This test is that someone.
 */
import * as fs from "fs";
import * as path from "path";

const SPEC_PATH = path.join(__dirname, "..", "specs", "LiteRTLM.nitro.ts");
const HOOKS_PATH = path.join(__dirname, "..", "hooks.ts");

/**
 * Keys of LLMConfig that useModel intentionally does NOT forward.
 * Add a key here only with a reason.
 */
const HOOK_UNSUPPORTED_KEYS: Record<string, string> = {
  // (none currently)
};

/** Extract top-level property names of `interface LLMConfig { ... }`. */
function parseLLMConfigKeys(source: string): string[] {
  const start = source.indexOf("export interface LLMConfig");
  expect(start).toBeGreaterThan(-1);
  const open = source.indexOf("{", start);
  // Walk to the matching closing brace.
  let depth = 0;
  let end = open;
  for (let i = open; i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}") depth--;
    if (depth === 0) {
      end = i;
      break;
    }
  }
  const body = source.slice(open + 1, end);
  const keys: string[] = [];
  // Property lines look like `  name?: type;` or `  name: type;`
  // (strip comments first so documented examples don't match).
  const noComments = body
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/\/\/[^\n]*/g, "");
  const re = /^\s*([A-Za-z_][A-Za-z0-9_]*)\??\s*:/gm;
  let m: RegExpExecArray | null;
  while ((m = re.exec(noComments)) !== null) {
    keys.push(m[1]);
  }
  return keys;
}

describe("LLMConfig ↔ useModel parity", () => {
  const specSource = fs.readFileSync(SPEC_PATH, "utf8");
  const hooksSource = fs.readFileSync(HOOKS_PATH, "utf8");
  const keys = parseLLMConfigKeys(specSource);

  it("parses a plausible LLMConfig surface", () => {
    // Sanity: the parser found the fields we know exist.
    expect(keys).toEqual(
      expect.arrayContaining(["backend", "maxContextTokens", "streamToolCalls"]),
    );
    expect(keys.length).toBeGreaterThanOrEqual(10);
  });

  it.each(keys)("useModel forwards or explicitly excludes '%s'", (key) => {
    if (key in HOOK_UNSUPPORTED_KEYS) {
      return; // deliberately unsupported, documented above
    }
    // The hook must read the key from its config (`config?.<key>`) AND place
    // it into the nativeConfig object literal (`{ <key> }` shorthand or
    // `<key>:`). Requiring both catches "read but never forwarded" too.
    const reads = new RegExp(`config\\?\\.${key}\\b`).test(hooksSource);
    const forwards = new RegExp(`[{\\s(&]${key}\\s*[},:]`).test(hooksSource);
    if (!reads || !forwards) {
      throw new Error(
        `LLMConfig.${key} is not handled by useModel. Forward it in the ` +
          `nativeConfig builder in src/hooks.ts (and its dependency array), ` +
          `or add it to HOOK_UNSUPPORTED_KEYS in this test with a reason.`,
      );
    }
  });
});
