#!/usr/bin/env bash
# Build the wrapper-grammar prototype probe (ticket #96) against a configured build/ tree.
#   SRC=/path/to/ninfer-yarn B=$SRC/build OUT=/tmp/opencode/proto-wrapper-cpu \
#       nix develop -c bash build.sh
set -euo pipefail
SPIKE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$SPIKE/../.." && pwd)}
B=${B:-$SRC/build}
OUT=${OUT:-/tmp/opencode/proto-wrapper-cpu}
CUDA_LINK=$(grep -oE '\-L[^ ]+|/[^ ]*libcudart\.so|/[^ ]*stubs/libcuda\.so' \
  "$B/tests/CMakeFiles/ninfer_grammar_test.dir/link.txt" | sort -u | tr '\n' ' ')
mkdir -p "$OUT"

g++ -O3 -DNDEBUG -std=gnu++20 \
  -I"$SRC/include" -I"$SRC/src" \
  -I"$SRC/third_party/llama-grammar" -I"$SRC/third_party/llama-grammar/compat" \
  -I"$SRC/third_party/llama-chat/common" -I"$SRC/third_party/llama-chat/compat" \
  -I"$SRC/third_party/llama-jinja" -isystem "$SRC/third_party" \
  -c "$SPIKE/wrapper_probe.cpp" -o "$OUT/wrapper_probe.o"

g++ "$OUT/wrapper_probe.o" -o "$OUT/wrapper_probe" \
  "$B/src/models/libninfer_grammar.a" \
  "$B/third_party/llama-grammar/libninfer_llama_grammar.a" \
  "$B/third_party/llama-chat/libninfer_llama_chat.a" \
  "$B/src/models/libninfer_model_loading.a" \
  "$B/third_party/llama-jinja/libninfer_jinja.a" \
  "$B/src/artifact/libninfer_artifact.a" \
  "$B/src/text/libninfer_text.a" \
  "$B/src/ops/libninfer_ops.a" \
  "$B/src/ops/libninfer_nvfp4_non_rdc.a" \
  "$B/src/core/libninfer_core.a" \
  $CUDA_LINK \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl

echo "built $OUT/wrapper_probe"
