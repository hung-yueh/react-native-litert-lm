/**
 * litertlm_manifest.json resolver — pick the right .litertlm file, backend,
 * and stream channels for a device from a model repo's deployment manifest.
 *
 * Repos converted for LiteRT-LM ship a `litertlm_manifest.json` at their root
 * describing every variant (backends verified to generate, per-platform
 * recommendations with evidence, sha256/size, thinking markers, session
 * defaults). This module reads it so apps stop hardcoding file names and
 * channel markers per model.
 *
 * Spec: https://github.com/john-rocky/hf-to-litertlm/blob/main/manifest/SCHEMA.md
 * The core reader (parse/resolve) is vendored from the reference TypeScript
 * reader (john-rocky/hf-to-litertlm `readers/ts`, v0.2.0) with `resolve`
 * renamed to `resolveVariant`, plus an optional `signal` and a
 * status-carrying error on `fetchManifest`; it is dependency-free and,
 * apart from `fetchManifest`, IO-free. The
 * react-native layer (`resolveFromManifest`, `mergeStreamChannels`) is at the
 * bottom. This module deliberately does not import react-native — `index.ts`
 * injects `Platform.OS` as the default platform.
 *
 * https://github.com/hung-yueh/react-native-litert-lm/issues/23
 */

import type { Backend, LLMConfig } from "./specs/LiteRTLM.nitro";
import { DEFAULT_CHANNELS, type StreamChannel } from "./streamEvents";

// ─── Vendored reference reader (types) ──────────────────────────────────────

/** Platform key used by manifest `recommended` rows (superset of RN's OS). */
export type ManifestPlatform = "android" | "ios" | "macos" | "windows" | "linux";

export interface ThinkingChannel {
  start: string;
  end: string;
}

/** One entry of the bundle's declared channel set (manifest 0.1.1+). */
export interface DeclaredChannel {
  name: string;
  start: string;
  end: string;
  is_reasoning?: boolean;
}

export interface ManifestCapabilities {
  vision?: boolean;
  audio?: boolean;
  thinking?: { declared: boolean; channel?: ThinkingChannel };
  /** Full bundle-declared channel set (0.1.1+); `thinking` mirrors the first entry. */
  channels?: DeclaredChannel[];
}

export interface ManifestRecommendation {
  platform: ManifestPlatform;
  device_class?: string;
  backend: Backend;
  reason?: string;
}

export interface ManifestVariant {
  file: string;
  sha256?: string;
  size_bytes?: number;
  quantization: string;
  /** Backends this file is verified to GENERATE on — not merely load. */
  backends: Backend[];
  default_backend?: Backend;
  min_runtime_version?: string;
  recommended?: ManifestRecommendation[];
  requirements?: { peak_ram_mb?: number; platform_notes?: string[] };
  known_issues?: string[];
  sections?: {
    type: string;
    size_bytes?: number;
    model_type?: string;
    backend_constraint?: string;
  }[];
}

export interface Manifest {
  manifest_schema: string;
  repo: string;
  generated: string;
  /**
   * Not part of the file: the revision the manifest was fetched at, stamped by
   * fetchManifest() so resolveVariant() URLs follow a pinned fetch.
   */
  revision?: string;
  model: {
    display_name: string;
    base_model?: string;
    architecture?: string;
    parameters_b?: number;
    license?: string;
    context_length?: number;
    capabilities?: ManifestCapabilities;
    session_defaults?: Record<string, unknown>;
  };
  variants: ManifestVariant[];
}

export interface ResolveVariantOptions {
  platform?: ManifestPlatform;
  /**
   * Explicit backend request. A filter, not a preference: only variants
   * listing it are considered, and resolveVariant() returns null when none
   * does — it never substitutes a backend the caller didn't ask for.
   */
  backend?: Backend;
  deviceClass?: string;
  /** Repo revision for the download URL; overrides the manifest's fetched revision (default "main"). */
  revision?: string;
}

export interface VariantResolution {
  /** File name inside the repo. */
  file: string;
  /** Direct download URL (huggingface.co/<repo>/resolve/<revision>/<file>). */
  url: string;
  backend: Backend;
  variant: ManifestVariant;
  sessionDefaults?: Record<string, unknown>;
  capabilities?: ManifestCapabilities;
  thinkingChannel?: ThinkingChannel;
  contextLength?: number;
  /** platform_notes + known_issues of the chosen variant — surface these to the developer. */
  notes: string[];
  /** Why this variant/backend was chosen (for logs). */
  reason: string;
}

// ─── Vendored reference reader (implementation) ─────────────────────────────

export function parseManifest(input: string | object): Manifest {
  const m = (typeof input === "string" ? JSON.parse(input) : input) as Manifest;
  if (
    !m ||
    typeof m !== "object" ||
    !m.manifest_schema ||
    !Array.isArray(m.variants) ||
    m.variants.length === 0
  ) {
    throw new Error("not a litertlm_manifest.json (missing manifest_schema or variants)");
  }
  if (!/^0\.1\./.test(m.manifest_schema)) {
    throw new Error(`unsupported manifest_schema ${m.manifest_schema} (reader supports 0.1.x)`);
  }
  for (const v of m.variants) {
    if (!Array.isArray(v.backends) || v.backends.length === 0) {
      throw new Error(`variant ${v?.file ?? "?"} lists no backends (schema requires minItems: 1)`);
    }
  }
  return m;
}

/**
 * Thrown by {@link fetchManifest} for a non-2xx response. `status` tells a
 * missing manifest (404) from a gated repo (401/403) or a Hub outage (5xx).
 */
export interface ManifestFetchError extends Error {
  status: number;
}

/** The HTTP status carried by a {@link ManifestFetchError}, or undefined for any other error. */
export function manifestFetchStatus(e: unknown): number | undefined {
  const status = e instanceof Error ? (e as Partial<ManifestFetchError>).status : undefined;
  return typeof status === "number" ? status : undefined;
}

/**
 * Fetch <repo>'s manifest from the Hugging Face Hub. resolveVariant() URLs
 * follow the revision fetched here. Throws on a non-2xx response (a
 * {@link ManifestFetchError} carrying the status), an unsupported schema, or
 * a failed/aborted fetch — {@link resolveFromManifest} turns all of those
 * into null; call this directly when you want the error. `signal` aborts
 * the fetch.
 */
export async function fetchManifest(
  repo: string,
  revision = "main",
  signal?: AbortSignal,
): Promise<Manifest> {
  const url = `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/litertlm_manifest.json`;
  const res = await fetch(url, { signal });
  if (!res.ok) {
    const message =
      res.status === 404
        ? `no litertlm_manifest.json at ${url} (HTTP 404)`
        : `HTTP ${res.status} fetching ${url}`;
    throw Object.assign(new Error(message), { status: res.status }) as ManifestFetchError;
  }
  const m = parseManifest(await res.text());
  m.revision = revision;
  return m;
}

interface ScoredVariant {
  variant: ManifestVariant;
  backend: Backend;
  score: number;
  reason: string;
}

/**
 * Pick the variant + backend for a device. Manifest v0.1 algorithm,
 * deterministic — identical to the reference readers:
 *
 * 1. An explicit `backend` request is a filter, not a score: only variants
 *    listing it compete, the result keeps that backend, and resolveVariant()
 *    returns null when no variant lists it — it never substitutes another
 *    backend.
 * 2. A variant with a `recommended` entry matching the requested platform
 *    wins; matching device_class too ranks higher. When a backend was
 *    requested, only recommendations naming that backend count.
 * 3. Otherwise the smallest variant (by size_bytes), on the requested backend
 *    (else its default_backend, else the first listed backend).
 *
 * Ties break toward the smaller file. The resolver never returns a backend
 * absent from the variant's verified `backends` list.
 */
export function resolveVariant(
  manifest: Manifest,
  opts: ResolveVariantOptions = {},
): VariantResolution | null {
  const requested = opts.backend;
  const candidates = requested
    ? manifest.variants.filter((v) => v.backends.includes(requested))
    : manifest.variants;
  if (candidates.length === 0) {
    return null;
  }

  const scored: ScoredVariant[] = candidates.map((v) => {
    let score = 0;
    let backend: Backend | undefined = requested;
    let reason = requested
      ? `supports requested backend ${requested}`
      : "fallback: smallest variant";
    if (opts.platform && v.recommended) {
      const recs = v.recommended.filter(
        (r) =>
          r.platform === opts.platform &&
          v.backends.includes(r.backend) &&
          (!requested || r.backend === requested),
      );
      const classRec = opts.deviceClass
        ? recs.find((r) => r.device_class === opts.deviceClass)
        : undefined;
      const rec = classRec ?? recs.find((r) => !r.device_class || !opts.deviceClass) ?? recs[0];
      if (rec) {
        score = classRec ? 300 : 200;
        backend = requested ?? rec.backend;
        const classNote =
          !classRec && opts.deviceClass && rec.device_class
            ? ` (no ${opts.deviceClass} entry; using the ${rec.device_class} recommendation)`
            : "";
        reason =
          `recommended for ${opts.platform}${classRec ? `/${opts.deviceClass}` : ""}` +
          `${classNote}: ${rec.reason ?? ""}`.trimEnd();
      }
    }
    if (!backend) {
      backend =
        v.default_backend && v.backends.includes(v.default_backend)
          ? v.default_backend
          : (v.backends[0] ?? "cpu");
    }
    return { variant: v, backend, score, reason };
  });

  scored.sort(
    (a, b) =>
      b.score - a.score || (a.variant.size_bytes ?? Infinity) - (b.variant.size_bytes ?? Infinity),
  );
  const best = scored[0];
  const v = best.variant;
  const caps = manifest.model.capabilities;
  const revision = encodeURIComponent(opts.revision ?? manifest.revision ?? "main");
  return {
    file: v.file,
    // Encode per path segment so a repo that nests variants in subfolders
    // (file containing '/') keeps its structure — '%2F' would 404.
    url: `https://huggingface.co/${manifest.repo}/resolve/${revision}/${v.file
      .split("/")
      .map(encodeURIComponent)
      .join("/")}`,
    backend: best.backend,
    variant: v,
    sessionDefaults: manifest.model.session_defaults,
    capabilities: caps,
    thinkingChannel: caps?.thinking?.declared ? caps.thinking.channel : undefined,
    contextLength: manifest.model.context_length,
    notes: [...(v.requirements?.platform_notes ?? []), ...(v.known_issues ?? [])],
    reason: best.reason,
  };
}

/** The model's declared thinking markers (exact strings, whitespace included). */
export function thinkingMarkers(manifest: Manifest): ThinkingChannel | undefined {
  const t = manifest.model.capabilities?.thinking;
  return t?.declared ? t.channel : undefined;
}

/** The bundle's full declared channel set (manifest 0.1.1+); empty for 0.1.0 manifests. */
export function declaredChannels(manifest: Manifest): DeclaredChannel[] {
  return manifest.model.capabilities?.channels ?? [];
}

// ─── react-native layer ─────────────────────────────────────────────────────

/**
 * Map a manifest-declared channel to a StreamEventType, or null when this
 * package has no event type for it yet.
 */
function streamTypeFor(c: DeclaredChannel): StreamChannel["type"] | null {
  if (c.is_reasoning || c.name === "thought" || c.name === "thinking") {
    return "thinking";
  }
  if (c.name.includes("tool")) {
    return "toolCall";
  }
  return null;
}

/**
 * Merge the manifest's declared channels OVER the package defaults, by type.
 *
 * `createLLM({ streamChannels })` replaces the whole array, so handing it only
 * the manifest's thinking markers would silently drop tool-call parsing
 * (#23). This merge keeps every default channel the manifest doesn't
 * redeclare — the default toolCall entry survives a manifest that declares
 * only thinking.
 */
export function mergeStreamChannels(
  manifest: Manifest,
  defaults: StreamChannel[] = DEFAULT_CHANNELS,
): StreamChannel[] {
  const merged = new Map<StreamChannel["type"], StreamChannel>();
  for (const c of defaults) {
    merged.set(c.type, c);
  }
  const declared = declaredChannels(manifest);
  for (const c of declared) {
    const type = streamTypeFor(c);
    if (type) {
      merged.set(type, { type, start: c.start, end: c.end });
    }
  }
  if (declared.length === 0) {
    // 0.1.0 manifests declare thinking markers only.
    const t = thinkingMarkers(manifest);
    if (t) {
      merged.set("thinking", { type: "thinking", start: t.start, end: t.end });
    }
  }
  return [...merged.values()];
}

/** react-native `Platform.OS` values that are also manifest `recommended[].platform` keys. */
const MANIFEST_PLATFORMS = new Set<string>([
  "android",
  "ios",
  "macos",
  "windows",
] satisfies ManifestPlatform[]);

/**
 * The manifest platform for a react-native `Platform.OS` value, or undefined
 * for one no manifest names (`web`, `native`). index.ts injects this as the
 * default `platform`, so react-native-macos and -windows keep their
 * recommendations instead of silently falling back to the smallest file.
 */
export function manifestPlatformFor(os: string): ManifestPlatform | undefined {
  return MANIFEST_PLATFORMS.has(os) ? (os as ManifestPlatform) : undefined;
}

/** Options for {@link resolveFromManifest}. */
export interface ResolveFromManifestOptions extends ResolveVariantOptions {
  /** Hub revision to fetch and pin (default "main"); download URLs follow it. */
  revision?: string;
  /**
   * Aborts the manifest fetch — pass an AbortController's signal so a
   * startup-time resolve can't hang on a bad connection. An aborted fetch
   * resolves to null, silently.
   */
  signal?: AbortSignal;
}

/** Everything an app needs to load the chosen variant. */
export interface ManifestResolution {
  /** Direct download URL of the chosen .litertlm — feed to `loadModel` / ModelRegistry. */
  url: string;
  file: string;
  backend: Backend;
  /**
   * LLMConfig fragment derived from the manifest: chosen backend plus mapped
   * `session_defaults` (sampler hints; `max_output_tokens_min` applied as a
   * floor). Spread it under your own overrides:
   * `{ ...resolution.config, ...userConfig }`.
   */
  config: LLMConfig;
  /**
   * Complete channel array for `createLLM({ streamChannels })` — the
   * manifest's channels merged over DEFAULT_CHANNELS, so passing it can never
   * silently drop tool-call parsing.
   */
  streamChannels: StreamChannel[];
  /** Verify the download against these (from HF LFS metadata). */
  sha256?: string;
  sizeBytes?: number;
  contextLength?: number;
  /**
   * The manifest's raw `session_defaults` — an open object (SCHEMA.md
   * declares `max_output_tokens_min`, `temperature`, `top_k`, `top_p`,
   * `notes`). `config` maps the keys it knows; any key a later 0.1.x adds is
   * reachable here without a re-fetch.
   */
  sessionDefaults?: Record<string, unknown>;
  /**
   * Worth surfacing to the developer: the chosen variant's platform_notes
   * and known_issues, then the model's `session_defaults.notes` (curated
   * guidance) when the manifest carries one.
   */
  notes: string[];
  /** Why this variant/backend was chosen (for logs). */
  reason: string;
}

/**
 * Their `maxOutputTokens` floor semantics: `session_defaults.max_output_tokens_min`
 * is a FLOOR (never a cap) — e.g. 2048 for reasoning models that need room to
 * think before answering. Applied as max(packageDefault, floor) here; caller
 * overrides win via `{ ...config, ...userConfig }`, so apply
 * max(userValue, floor) yourself if you take user values.
 */
const DEFAULT_MAX_OUTPUT_TOKENS = 1024;

/**
 * Synchronous core of {@link resolveFromManifest} for an already-fetched
 * manifest. Returns null when an explicitly requested backend is listed by no
 * variant — fall back to your current (non-manifest) path in that case.
 */
export function resolutionFor(
  manifest: Manifest,
  opts: ResolveFromManifestOptions = {},
): ManifestResolution | null {
  const r = resolveVariant(manifest, opts);
  if (!r) {
    return null;
  }

  const config: LLMConfig = { backend: r.backend };
  const sd = r.sessionDefaults ?? {};
  if (typeof sd.max_output_tokens_min === "number") {
    config.maxOutputTokens = Math.max(DEFAULT_MAX_OUTPUT_TOKENS, sd.max_output_tokens_min);
  }
  if (typeof sd.temperature === "number") {
    config.temperature = sd.temperature;
  }
  if (typeof sd.top_k === "number") {
    config.topK = sd.top_k;
  }
  if (typeof sd.top_p === "number") {
    config.topP = sd.top_p;
  }
  // Two knobs deliberately NOT derived from the manifest (agreed in the #25
  // review; not final):
  // - `context_length` does not seed `maxContextTokens`. They look like the
  //   same number but are different knobs: the manifest value is what the
  //   model can address, the package default is a memory ceiling, and quietly
  //   raising it to 128K on a phone is the OOM the pre-flight check exists to
  //   catch. `contextLength` is on the resolution for apps that want more,
  //   knowingly.
  // - `streamToolCalls` stays off even when the manifest declares a tool
  //   channel. A declared channel says the model *can* emit tool calls, not
  //   that this app wants to handle them, and switching engine behaviour on
  //   as a side effect of "resolve a URL" would surprise a caller. Revisit
  //   once the bundle-header read lands and the signal is local, not fetched.

  return {
    url: r.url,
    file: r.file,
    backend: r.backend,
    config,
    streamChannels: mergeStreamChannels(manifest),
    sha256: r.variant.sha256,
    sizeBytes: r.variant.size_bytes,
    contextLength: r.contextLength,
    sessionDefaults: r.sessionDefaults,
    notes: typeof sd.notes === "string" && sd.notes !== "" ? [...r.notes, sd.notes] : r.notes,
    reason: r.reason,
  };
}

/**
 * Fetch `<repo>`'s manifest from the Hugging Face Hub and resolve the right
 * variant for this device.
 *
 * ```ts
 * const resolution = await resolveFromManifest("litert-community/LFM2.5-1.2B-Instruct");
 * if (resolution) {
 *   const llm = createLLM({ streamChannels: resolution.streamChannels });
 *   await llm.loadModel(resolution.url, { ...resolution.config });
 * }
 * ```
 *
 * `platform` defaults to this device's OS when called through the package
 * export (index.ts injects `Platform.OS`).
 *
 * Never throws. Returns null whenever the manifest can't help — the repo
 * ships none (HTTP 404, the common case on the Hub today), its schema is
 * newer than this reader supports, the fetch fails or is aborted, or no
 * variant lists an explicitly requested backend — so callers fall back to
 * their current (non-manifest) loading path without a try/catch.
 *
 * The expected outcomes are silent: no manifest, an abort the caller asked
 * for, an unlisted backend. Anything else (unsupported schema, a body that
 * is not a manifest, a network failure, a non-404 HTTP status) is reported
 * with one console.warn line naming the reason — on React Native, LogBox
 * shows console.warn in dev, which is too loud for the common path.
 */
export async function resolveFromManifest(
  repo: string,
  opts: ResolveFromManifestOptions = {},
): Promise<ManifestResolution | null> {
  try {
    const manifest = await fetchManifest(repo, opts.revision ?? "main", opts.signal);
    return resolutionFor(manifest, opts);
  } catch (e) {
    if (opts.signal?.aborted || manifestFetchStatus(e) === 404) {
      return null;
    }
    const reason = e instanceof Error ? e.message : String(e);
    console.warn(`resolveFromManifest(${repo}): ${reason} — returning null`);
    return null;
  }
}
