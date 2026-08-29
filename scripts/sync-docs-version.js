#!/usr/bin/env node
/**
 * Sync the docs site's version badge with package.json.
 *
 * The badge is hand-written in every website/*.html page, so it silently drifts
 * after a release. release-it runs this from its `after:bump` hook, between the
 * version bump and the release commit, so the site ships with the right number.
 *
 * Safe to run manually at any time; prints what it changed and nothing else.
 */
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const { version } = require(path.join(root, "package.json"));
const siteDir = path.join(root, "website");

if (!fs.existsSync(siteDir)) {
  console.log("sync-docs-version: no website/ directory — nothing to do");
  process.exit(0);
}

// Matches the badge markup in the top bar: <span class="ver">v0.6.1</span>
const BADGE = /(<span class="ver">v)(\d+\.\d+\.\d+)(<\/span>)/g;

let changed = 0;
let scanned = 0;
const stale = [];

for (const file of fs.readdirSync(siteDir).filter((f) => f.endsWith(".html"))) {
  const full = path.join(siteDir, file);
  const before = fs.readFileSync(full, "utf8");
  if (!BADGE.test(before)) continue;
  BADGE.lastIndex = 0;
  scanned++;

  const after = before.replace(BADGE, (_m, open, found, close) => {
    if (found !== version) stale.push(`${file} (${found})`);
    return `${open}${version}${close}`;
  });

  if (after !== before) {
    fs.writeFileSync(full, after);
    changed++;
  }
}

if (changed === 0) {
  console.log(`sync-docs-version: all ${scanned} pages already at v${version}`);
} else {
  console.log(
    `sync-docs-version: updated ${changed}/${scanned} pages to v${version}` +
      (stale.length ? ` — was ${[...new Set(stale.map((s) => s.split("(")[1]))].join(", ").replace(/\)/g, "")}` : ""),
  );
}
