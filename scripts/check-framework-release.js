#!/usr/bin/env node
/**
 * check-framework-release.js
 *
 * Release-time guardrail. Verifies the prebuilt iOS framework asset actually
 * exists at the URL that consumers' postinstall will fetch, BEFORE this package
 * is published to npm. Wired into `prepublishOnly` so a missing / mis-named /
 * mis-tagged asset fails the maintainer's publish instead of silently breaking
 * every downstream macOS `npm install`.
 *
 * Uses the same source of truth (scripts/framework-source.js) as postinstall,
 * so the URL checked here is exactly the URL fetched there.
 *
 * Skips (with a warning, non-fatal) when SKIP_FRAMEWORK_RELEASE_CHECK=1, for
 * local dry-runs where the release asset isn't uploaded yet.
 */

const https = require('https');
const { ASSET_URL, ASSET_NAME, FRAMEWORK_TAG } = require('./framework-source');

function log(msg) {
  console.log(`[react-native-litert-lm] ${msg}`);
}

/** Resolve to the final status code, following redirects (GitHub release assets 302 to a CDN). */
function headStatus(url, maxRedirects = 5) {
  return new Promise((resolve, reject) => {
    if (maxRedirects <= 0) return reject(new Error('Too many redirects'));

    https
      .request(url, { method: 'HEAD', headers: { 'User-Agent': 'react-native-litert-lm' } }, (res) => {
        const { statusCode, headers } = res;
        res.resume(); // drain
        if (statusCode >= 300 && statusCode < 400 && headers.location) {
          return headStatus(headers.location, maxRedirects - 1).then(resolve).catch(reject);
        }
        resolve(statusCode);
      })
      .on('error', reject)
      .end();
  });
}

async function main() {
  if (process.env.SKIP_FRAMEWORK_RELEASE_CHECK === '1') {
    log('Skipping iOS framework release check (SKIP_FRAMEWORK_RELEASE_CHECK=1).');
    return;
  }

  log(`Verifying iOS framework release asset is published: ${ASSET_URL}`);

  let status;
  try {
    status = await headStatus(ASSET_URL);
  } catch (err) {
    console.error(`[react-native-litert-lm] ERROR: could not reach the release asset URL: ${err.message}`);
    process.exit(1);
  }

  if (status !== 200) {
    console.error(
      `[react-native-litert-lm] ERROR: iOS framework asset "${ASSET_NAME}" is not available ` +
        `(HTTP ${status}) under release tag "${FRAMEWORK_TAG}".\n` +
        `  Publishing now would break every macOS \`npm install\`.\n` +
        `  Upload "${ASSET_NAME}" to the "${FRAMEWORK_TAG}" GitHub release, then retry,\n` +
        `  or set SKIP_FRAMEWORK_RELEASE_CHECK=1 to bypass (not recommended).`
    );
    process.exit(1);
  }

  log('iOS framework release asset verified ✓');
}

main();
