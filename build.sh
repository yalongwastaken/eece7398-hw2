#!/usr/bin/env bash
# @file    build.sh
# @author  Anthony
# @brief   clone and build llama.cpp with CUDA (Explorer) or Metal (macOS)
# @usage   bash build.sh

set -e

OS="$(uname)"
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

if [ "$OS" = "Darwin" ]; then
    echo "[build] macOS detected — building with Metal..."
    SDK=$(xcrun --show-sdk-path)
    CC=$(xcrun -f clang)
    CXX=$(xcrun -f clang++)
    cmake .. \
        -DGGML_METAL=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CCACHE=OFF \
        -DCMAKE_OSX_SYSROOT="$SDK" \
        -DCMAKE_C_COMPILER="$CC" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_FLAGS="-isysroot $SDK -stdlib=libc++" \
        -DCMAKE_C_FLAGS="-isysroot $SDK"
else
    echo "[build] Linux detected — building with CUDA..."
    cmake .. \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"
fi

if [ "$OS" = "Darwin" ]; then
    JOBS="$(sysctl -n hw.logicalcpu)"
else
    JOBS="$(nproc)"
fi
cmake --build . --config Release -j"$JOBS"
echo "[build] done. binary at llama.cpp/build/bin/llama-cli"