#!/usr/bin/env bash
# Build the throwaway logprob_topk prototype. Requires the flake devShell (nvcc, CUDA 13.1).
set -euo pipefail
cd "$(dirname "$0")"

nvcc --version | tail -n 1
nvcc -O3 -std=c++17 -arch=sm_120a gather_prototype.cu -o gather_prototype
echo "built: $(pwd)/gather_prototype"
