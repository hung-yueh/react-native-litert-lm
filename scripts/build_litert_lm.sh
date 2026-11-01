#!/bin/bash
#
# build_litert_lm.sh
# Build LiteRT-LM libraries for React Native integration
#
# Prerequisites:
#   - Bazel 7.6.1 (via Bazelisk recommended)
#   - Android NDK r28b or newer
#   - Xcode Command Line Tools (macOS)
#
# Usage: ./build_litert_lm.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITERT_LM_DIR="${SCRIPT_DIR}/LiteRT-LM"
OUTPUT_DIR="${SCRIPT_DIR}/libs"

echo "============================================"
echo "  LiteRT-LM Build Script for React Native  "
echo "============================================"

# Check prerequisites
check_prerequisites() {
    echo ""
    echo "Checking prerequisites..."
    
    # Check Bazel
    if ! command -v bazel &> /dev/null && ! command -v bazelisk &> /dev/null; then
        echo "❌ Bazel not found. Install via:"
        echo "   brew install bazelisk          # macOS"
        echo "   npm install -g @bazel/bazelisk # Linux / Node"
        exit 1
    fi
    echo "✅ Bazel found"
    
    # Check Android NDK
    if [ -z "$ANDROID_NDK_HOME" ]; then
        for search_dir in "$ANDROID_HOME/ndk" "$HOME/Android/Sdk/ndk"; do
            if [ -d "$search_dir" ]; then
                LATEST_NDK=$(find "$search_dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -V | tail -n 1)
                if [ -n "$LATEST_NDK" ] && [ -d "$LATEST_NDK" ]; then
                    export ANDROID_NDK_HOME="$LATEST_NDK"
                    break
                fi
            fi
        done
    fi
    if [ -z "$ANDROID_NDK_HOME" ]; then
        echo "❌ ANDROID_NDK_HOME not set. Install NDK r28b+ and set:"
        echo "   export ANDROID_NDK_HOME=/path/to/ndk"
        exit 1
    fi
    echo "✅ Android NDK: $ANDROID_NDK_HOME"
    
    # Check Xcode (macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! xcode-select -p &> /dev/null; then
            echo "❌ Xcode Command Line Tools not installed. Run:"
            echo "   xcode-select --install"
            exit 1
        fi
        echo "✅ Xcode Command Line Tools found"
    fi
}

# Clone LiteRT-LM if not exists
clone_repo() {
    if [ ! -d "$LITERT_LM_DIR" ]; then
        echo ""
        echo "Cloning LiteRT-LM repository..."
        git clone https://github.com/google-ai-edge/LiteRT-LM.git "$LITERT_LM_DIR"
        cd "$LITERT_LM_DIR"
        git fetch --tags
        # Checkout latest stable release
        LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "main")
        echo "Checking out: $LATEST_TAG"
        git checkout "$LATEST_TAG"
    else
        echo ""
        echo "LiteRT-LM directory exists, using existing checkout"
    fi
}

# Build for Android arm64
build_android_arm64() {
    echo ""
    echo "Building for Android arm64-v8a..."
    cd "$LITERT_LM_DIR"
    
    bazel build --config=android_arm64 \
        //runtime/engine:litert_lm_jni \
        //runtime/engine:LiteRtGpuAccelerator \
        //runtime/engine:LiteRtTopKSampler \
        || {
            echo "⚠️  JNI target may not exist, trying alternative..."
            bazel build --config=android_arm64 //runtime/engine:litert_lm_main
        }
    
    # Copy outputs
    mkdir -p "${OUTPUT_DIR}/arm64-v8a"
    find bazel-bin -name "*.so" -exec cp {} "${OUTPUT_DIR}/arm64-v8a/" \;
    echo "✅ Android arm64-v8a libraries built"
}

# Build for Android armeabi-v7a
build_android_arm32() {
    echo ""
    echo "Building for Android armeabi-v7a..."
    cd "$LITERT_LM_DIR"
    
    bazel build --config=android_arm \
        //runtime/engine:litert_lm_jni \
        || echo "⚠️  arm32 build skipped (optional)"
    
    # Copy outputs
    mkdir -p "${OUTPUT_DIR}/armeabi-v7a"
    find bazel-bin -name "*.so" -exec cp {} "${OUTPUT_DIR}/armeabi-v7a/" \;
}

# Build for macOS (for iOS simulator and local testing)
build_macos() {
    echo ""
    echo "Building for macOS (local testing)..."
    cd "$LITERT_LM_DIR"
    
    bazel build //runtime/engine:litert_lm_main
    
    mkdir -p "${OUTPUT_DIR}/macos"
    cp bazel-bin/runtime/engine/litert_lm_main "${OUTPUT_DIR}/macos/" || true
    find bazel-bin -name "*.dylib" -exec cp {} "${OUTPUT_DIR}/macos/" \;
    echo "✅ macOS binaries built"
}

# Print summary
print_summary() {
    echo ""
    echo "============================================"
    echo "  Build Complete!"
    echo "============================================"
    echo ""
    echo "Libraries are in: ${OUTPUT_DIR}"
    echo ""
    ls -la "${OUTPUT_DIR}/" 2>/dev/null || true
    echo ""
    echo "Next steps:"
    echo "1. Copy libs to react-native-litert-lm/android/libs/"
    echo "2. Uncomment LiteRT-LM linking in CMakeLists.txt"
    echo "3. Build your React Native app"
}

# Main
main() {
    check_prerequisites
    clone_repo
    build_android_arm64
    # build_android_arm32  # Optional
    if [[ "$OSTYPE" == "darwin"* ]]; then
        build_macos
    fi
    print_summary
}

main "$@"
