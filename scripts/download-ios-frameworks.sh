#!/bin/bash
# download-ios-frameworks.sh
# Downloads the official prebuilt LiteRT-LM iOS framework (CLiteRTLM.xcframework)
# directly from the google-ai-edge/LiteRT-LM releases.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Resolve the asset URL + output dir from the shared single source of truth
# (scripts/framework-source.js) so this manual path can't drift from postinstall.
RELEASE_URL="$(node -e "console.log(require('$SCRIPT_DIR/framework-source').ASSET_URL)")"
OUTPUT_DIR="$(node -e "console.log(require('$SCRIPT_DIR/framework-source').FRAMEWORKS_DIR)")"
PINNED_TAG="$(node -e "console.log(require('$SCRIPT_DIR/framework-source').FRAMEWORK_TAG)")"
STAMP_FILE="$OUTPUT_DIR/.engine-tag"

# Skip only if the framework present matches the pinned engine tag
if [ -d "$OUTPUT_DIR/CLiteRTLM.xcframework" ]; then
  if [ "$(cat "$STAMP_FILE" 2>/dev/null)" = "$PINNED_TAG" ]; then
    echo "[LiteRT-LM] iOS CLiteRTLM.xcframework ($PINNED_TAG) already present, skipping download."
    exit 0
  fi
  echo "[LiteRT-LM] Installed framework is not $PINNED_TAG — replacing it."
fi

echo "[LiteRT-LM] Downloading prebuilt iOS engine from Google's release:"
echo "   ${RELEASE_URL}"

mkdir -p "$OUTPUT_DIR"
TMP_ZIP="$PROJECT_ROOT/.ios-frameworks-tmp.zip"

if curl -fsSL -o "$TMP_ZIP" "$RELEASE_URL"; then
  echo "[LiteRT-LM] Download successful, extracting to $OUTPUT_DIR..."
  rm -rf "$OUTPUT_DIR/CLiteRTLM.xcframework"
  unzip -o -q "$TMP_ZIP" -d "$OUTPUT_DIR"
  echo "$PINNED_TAG" > "$STAMP_FILE"
  rm -f "$TMP_ZIP"
  echo "[LiteRT-LM] ✅ iOS frameworks successfully installed."
  exit 0
else
  rm -f "$TMP_ZIP"
  echo "Error: Failed to download LiteRT-LM iOS framework from ${RELEASE_URL}"
  exit 1
fi
