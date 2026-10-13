/**
 * The LiteRT-LM engine version this package is built against
 * (`litertLm.version` in package.json; a unit test keeps the two in sync).
 * Compare a bundle's `minRuntimeVersion` against it before loading.
 */
export const LITERT_LM_VERSION = "0.18.0";

/**
 * Compare dotted numeric versions ("0.17.0" vs "0.18"); a leading "v" and any
 * pre-release suffix ("-rc1") are ignored. Returns <0, 0 or >0 like a sort
 * comparator.
 */
export function compareVersions(a: string, b: string): number {
  const parse = (v: string) =>
    v
      .trim()
      .replace(/^v/i, "")
      .split(/[-+]/)[0]
      .split(".")
      .map((n) => Number.parseInt(n, 10) || 0);
  const pa = parse(a);
  const pb = parse(b);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] ?? 0) - (pb[i] ?? 0);
    if (d !== 0) return d;
  }
  return 0;
}
