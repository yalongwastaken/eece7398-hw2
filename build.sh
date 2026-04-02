#!/usr/bin/env bash
# @file    build.sh
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Clone and build llama.cpp with Metal backend for macOS Apple Silicon.
# @usage   bash build.sh

set -e

LLAMA_DIR="llama.cpp"

# clone if not already present
if [ ! -d "$LLAMA_DIR" ]; then
    echo "[build] cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
else
    echo "[build] llama.cpp already cloned, pulling latest..."
    git -C "$LLAMA_DIR" pull
fi

cd "$LLAMA_DIR"
mkdir -p build && cd build

SDK=$(xcrun --show-sdk-path)
echo "[build] building with Metal (SDK: $SDK)..."
cmake .. \
    -DGGML_METAL=ON \
    -DGGML_CCACHE=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_SYSROOT="$SDK"

JOBS="$(sysctl -n hw.logicalcpu)"
cmake --build . --config Release -j"$JOBS"

echo "[build] done. binary at llama.cpp/build/bin/llama-cli"