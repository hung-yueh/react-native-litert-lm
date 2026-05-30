#!/usr/bin/env node
/**
 * postinstall.js
 *
 * Downloads prebuilt LiteRT-LM iOS frameworks from this package's GitHub
 * releases when consumers run `npm install react-native-litert-lm`.
 *
 * The framework is intentionally NOT shipped inside the npm tarball (it is
 * ~40MB and irrelevant to Android-only consumers). The asset name, tag, and
 * URL come from scripts/framework-source.js so they stay in lockstep with the
 * release-time guardrail (scripts/check-framework-release.js).
 *
 * Skips download if:
 *   - Not on macOS (iOS builds require macOS)
 *   - Frameworks already exist
 *   - SKIP_IOS_FRAMEWORK_DOWNLOAD=1 (e.g. Android-only / CI builds)
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const https = require('https');

const { ASSET_URL, FRAMEWORKS_DIR, FRAMEWORK_TAG } = require('./framework-source');

/** ZIP local-file-header magic ("PK\x03\x04"). Guards against truncated downloads / HTML error pages. */
const ZIP_MAGIC = Buffer.from([0x50, 0x4b, 0x03, 0x04]);
/** A valid framework zip is well over this; anything smaller is certainly an error page. */
const MIN_VALID_BYTES = 1024 * 1024; // 1 MB

function log(msg) {
  console.log(`[react-native-litert-lm] ${msg}`);
}

function shouldSkip() {
  // Skip if not macOS
  if (process.platform !== 'darwin') {
    log('Skipping iOS framework download (not macOS).');
    return true;
  }

  // Skip if explicitly disabled
  if (process.env.SKIP_IOS_FRAMEWORK_DOWNLOAD === '1') {
    log('Skipping iOS framework download (SKIP_IOS_FRAMEWORK_DOWNLOAD=1).');
    return true;
  }

  // Skip if frameworks already exist
  if (fs.existsSync(FRAMEWORKS_DIR) && fs.readdirSync(FRAMEWORKS_DIR).length > 0) {
    log('iOS frameworks already present, skipping download.');
    return true;
  }

  return false;
}

function downloadFile(url, destPath, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) {
      return reject(new Error('Too many redirects'));
    }

    const protocol = url.startsWith('https') ? https : require('http');

    protocol.get(url, { headers: { 'User-Agent': 'react-native-litert-lm' } }, (res) => {
      // Follow redirects
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        return downloadFile(res.headers.location, destPath, maxRedirects - 1)
          .then(resolve)
          .catch(reject);
      }

      if (res.statusCode !== 200) {
        return reject(new Error(`HTTP ${res.statusCode} downloading ${url}`));
      }

      const file = fs.createWriteStream(destPath);
      res.pipe(file);
      file.on('finish', () => {
        file.close(resolve);
      });
      file.on('error', reject);
    }).on('error', reject);
  });
}

/**
 * Verify the downloaded file is a non-trivial ZIP before we trust it.
 * Catches GitHub HTML error pages and truncated downloads that would
 * otherwise fail cryptically at `unzip` or, worse, link a corrupt framework.
 */
function assertValidZip(zipPath) {
  const { size } = fs.statSync(zipPath);
  if (size < MIN_VALID_BYTES) {
    throw new Error(`downloaded asset is only ${size} bytes — expected a multi-MB framework zip (likely an error page or truncated download)`);
  }

  const header = Buffer.alloc(4);
  const fd = fs.openSync(zipPath, 'r');
  try {
    fs.readSync(fd, header, 0, 4, 0);
  } finally {
    fs.closeSync(fd);
  }
  if (!header.equals(ZIP_MAGIC)) {
    throw new Error('downloaded asset is not a valid ZIP (bad magic bytes)');
  }
}

async function main() {
  if (shouldSkip()) return;

  log(`Downloading iOS frameworks (engine ${FRAMEWORK_TAG}) from: ${ASSET_URL}`);

  const tmpZip = path.join(path.dirname(FRAMEWORKS_DIR), '.ios-frameworks-tmp.zip');

  try {
    await downloadFile(ASSET_URL, tmpZip);
    assertValidZip(tmpZip);

    // Extract
    fs.mkdirSync(FRAMEWORKS_DIR, { recursive: true });
    execSync(`unzip -o -q "${tmpZip}" -d "${FRAMEWORKS_DIR}"`, { stdio: 'inherit' });

    // Cleanup
    fs.unlinkSync(tmpZip);

    log('iOS frameworks installed successfully.');
  } catch (err) {
    // Cleanup partial download
    try { fs.unlinkSync(tmpZip); } catch {}

    log(`Error: Could not download iOS frameworks: ${err.message}`);
    log('iOS builds will not work until frameworks are available.');
    log('Run: ./scripts/download-ios-frameworks.sh to download manually.');

    // Fail fast on macOS so users discover the problem now, not at Xcode link time.
    if (process.platform === 'darwin') {
      log('Set SKIP_IOS_FRAMEWORK_DOWNLOAD=1 to suppress this error (e.g. Android-only builds).');
      process.exit(1);
    }
  }
}

main();
