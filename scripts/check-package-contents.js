#!/usr/bin/env node
/**
 * check-package-contents.js
 *
 * CI guardrail. Lists what `npm pack` would publish and fails when a file the
 * package refers to at install or build time is missing from it. 0.8.1 reached
 * npm without android/consumer-rules.pro: android/build.gradle names it, but
 * the "files" list in package.json didn't, so the release build of every
 * Android app that installed the package failed (#31).
 *
 * References checked:
 *   - android/build.gradle: consumerProguardFiles, the CMake path, `apply from`
 *   - the podspec, and the files it `load`s
 *   - the entry points of package.json and of every nested package.json shipped
 *   - the files the postinstall script runs, and the relative requires of
 *     shipped JavaScript (tests and mocks aside)
 *
 * lib/ and nitrogen/generated/ are build output: run `npx nitrogen` and
 * `npm run build` first, as CI does.
 */

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');

function log(msg) {
  console.log(`[react-native-litert-lm] ${msg}`);
}

function packedFiles() {
  const out = execFileSync('npm', ['pack', '--dry-run', '--json', '--ignore-scripts'], {
    cwd: root,
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'inherit'],
  });
  return new Set(JSON.parse(out)[0].files.map((f) => f.path));
}

/** The packed path a reference names; entry points and requires may leave out the extension or name a directory. */
function findPacked(packed, file) {
  const candidates = ['', '.js', '.json', '.ts', '.tsx', '.d.ts', '/index.js', '/index.ts', '/index.tsx'];
  return candidates.map((ext) => file + ext).find((candidate) => packed.has(candidate));
}

function quoted(text) {
  return [...text.matchAll(/['"]([^'"]+)['"]/g)].map((m) => m[1]);
}

/** Every file the package refers to, as { file, from }, with `file` relative to the package root. */
function references(packed) {
  const refs = [];
  const add = (from, dir, names) => {
    for (const name of names) refs.push({ file: path.posix.normalize(path.posix.join(dir, name)), from });
  };

  const gradle = fs.readFileSync(path.join(root, 'android/build.gradle'), 'utf8');
  for (const m of gradle.matchAll(/consumerProguardFiles\s+([^\n]+)/g)) {
    add('android/build.gradle consumerProguardFiles', 'android', quoted(m[1]));
  }
  for (const m of gradle.matchAll(/\bpath\s+['"]([^'"]+)['"]/g)) add('android/build.gradle cmake path', 'android', [m[1]]);
  for (const m of gradle.matchAll(/apply from:\s*['"]([^'"]+)['"]/g)) add('android/build.gradle apply from', 'android', [m[1]]);

  for (const podspec of fs.readdirSync(root).filter((f) => f.endsWith('.podspec'))) {
    add('React Native autolinking', '.', [podspec]);
    const text = fs.readFileSync(path.join(root, podspec), 'utf8');
    for (const m of text.matchAll(/^\s*load\s+['"]([^'"]+)['"]/gm)) add(`${podspec} load`, '.', [m[1]]);
  }

  for (const manifest of [...packed].filter((p) => p === 'package.json' || p.endsWith('/package.json'))) {
    const json = JSON.parse(fs.readFileSync(path.join(root, manifest), 'utf8'));
    const dir = path.posix.dirname(manifest);
    for (const field of ['main', 'module', 'types', 'react-native', 'source']) {
      if (typeof json[field] === 'string') add(`${manifest} "${field}"`, dir, [json[field]]);
    }
  }

  const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const postinstall = pkg.scripts?.postinstall ?? '';
  add('package.json postinstall', '.', postinstall.split(/\s+/).filter((t) => /\.(c?js|sh)$/.test(t)));

  for (const file of [...packed].filter((p) => /\.c?js$/.test(p) && !/(^|\/)__(tests|mocks)__\//.test(p))) {
    const text = fs.readFileSync(path.join(root, file), 'utf8');
    for (const m of text.matchAll(/require\(\s*['"](\.{1,2}\/[^'"]+)['"]\s*\)/g)) {
      add(`${file} require`, path.posix.dirname(file), [m[1]]);
    }
  }
  return refs;
}

function hint(file) {
  if (file.startsWith('lib/')) return 'build output: run `npm run build` first';
  if (file.startsWith('nitrogen/generated/')) return 'codegen output: run `npx nitrogen` first';
  if (fs.existsSync(path.join(root, file))) return 'it exists, but the "files" list in package.json leaves it out';
  return 'no such file';
}

function main() {
  const packed = packedFiles();
  const refs = references(packed);
  const missing = refs.filter((ref) => !findPacked(packed, ref.file));

  if (missing.length > 0) {
    console.error('[react-native-litert-lm] ERROR: the npm package would be missing files it refers to:');
    for (const ref of missing) console.error(`  ${ref.file}  (named by ${ref.from}; ${hint(ref.file)})`);
    process.exit(1);
  }
  log(`Package contents verified ✓ (${packed.size} files; all ${refs.length} references are packed)`);
}

main();
