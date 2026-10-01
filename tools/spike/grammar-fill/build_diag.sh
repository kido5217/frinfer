#!/usr/bin/env bash
# Build the diag-instrumented probe: applies diag.patch to a private copy of the module source
# and compiles it with NINFER_GRAMMAR_DIAG, so master is never modified.
#
#   SRC=/path/to/ninfer-yarn B=/path/to/ninfer-yarn/build OUT=/tmp/opencode/grammar-fill \
#       nix develop -c bash build_diag.sh
set -euo pipefail
SPIKE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-$(cd "$SPIKE/../../.." && pwd)}
B=${B:-$SRC/build}
OUT=${OUT:-/tmp/opencode/grammar-fill}
DIAG=$OUT/diag-src
CUDA_LINK=$(grep -oE '\-L[^ ]+|/[^ ]*libcudart\.so|/[^ ]*stubs/libcuda\.so' \
  "$B/tests/CMakeFiles/ninfer_grammar_test.dir/link.txt" | sort -u | tr '\n' ' ')
mkdir -p "$OUT"

rm -rf "$DIAG"
mkdir -p "$DIAG/src/models/qwen3_5/frontend/grammar"
cp "$SRC/src/models/qwen3_5/frontend/grammar/grammar.cpp" \
  "$DIAG/src/models/qwen3_5/frontend/grammar/grammar.cpp"
cp "$SRC/src/models/qwen3_5/frontend/grammar/grammar.h" \
  "$DIAG/src/models/qwen3_5/frontend/grammar/grammar.h"
patch -s -p1 -d "$DIAG" < "$SPIKE/diag.patch"

FLAGS=(-O2 -g -fno-omit-frame-pointer -DNDEBUG -DNINFER_GRAMMAR_DIAG -std=gnu++20
  -I"$DIAG/src"
  -I"$SRC/include" -I"$SRC/src"
  -I"$SRC/third_party/llama-grammar" -I"$SRC/third_party/llama-grammar/compat"
  -I"$SRC/third_party/llama-chat/common" -I"$SRC/third_party/llama-chat/compat"
  -I"$SRC/third_party/llama-jinja" -isystem "$SRC/third_party")

g++ "${FLAGS[@]}" -c "$SPIKE/probe.cpp" -o "$OUT/probe-diag.o"
g++ "${FLAGS[@]}" -c "$DIAG/src/models/qwen3_5/frontend/grammar/grammar.cpp" \
  -o "$OUT/grammar-diag.o"
g++ "${FLAGS[@]}" -c "$SRC/src/models/qwen3_5/frontend/grammar/tokenizer_vocabulary.cpp" \
  -o "$OUT/tokenizer_vocabulary-diag.o"

g++ "$OUT/probe-diag.o" "$OUT/grammar-diag.o" "$OUT/tokenizer_vocabulary-diag.o" -o "$OUT/probe-diag" \
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

echo "built $OUT/probe-diag"
