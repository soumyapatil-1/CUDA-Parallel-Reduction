#!/usr/bin/env bash
# Build (and optionally run) the Parallel Reduction project on Linux.
# Usage:  bash scripts/build_linux.sh          -> build only
#         bash scripts/build_linux.sh --run    -> build, then run the full benchmark
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p build results

if ! command -v nvcc >/dev/null 2>&1; then
    for d in /usr/local/cuda/bin /opt/cuda/bin; do
        [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
    done
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found. Install the CUDA Toolkit or add it to PATH."; exit 1; }

echo "== nvcc version =="; nvcc --version | tail -n 2
SRC=src/parallel_reduction.cu
OUT=build/parallel_reduction
FLAGS="-O3 -std=c++17 -lineinfo"

# 1) Let nvcc detect the installed GPU (CUDA 11.6+).
if nvcc $FLAGS -arch=native "$SRC" -o "$OUT" 2>/dev/null; then
    echo "Built with -arch=native"
else
    # 2) Ask the driver for the compute capability (e.g. 8.6 -> sm_86).
    CC=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .')"
    fi
    if [ -n "$CC" ]; then
        echo "Building for sm_$CC (from nvidia-smi)"
        nvcc $FLAGS -gencode "arch=compute_${CC},code=[sm_${CC},compute_${CC}]" "$SRC" -o "$OUT"
    else
        # 3) Toolkit default architecture + PTX (JIT-compiled by the driver).
        echo "Could not detect GPU architecture; using nvcc default target."
        nvcc $FLAGS "$SRC" -o "$OUT"
    fi
fi
echo "Build OK -> $OUT"

if [ "${1:-}" = "--run" ]; then
    shift
    "./$OUT" "$@" | tee results/run_log.txt
fi
