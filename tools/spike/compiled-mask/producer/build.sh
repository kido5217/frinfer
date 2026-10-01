#!/usr/bin/env bash
# Build the trie-producer probe (ticket #57, route (a)).
#
#   SRC=/tmp/opencode/trie-prototype/ninfer B=/home/kido/network/projects/ninfer-yarn/build \
#       nix develop -c bash build_trie.sh
#
# Compiles a private patched copy of the grammar module (probe accessor only;
# the repository is never modified) plus the trie producer and driver, and links
# against the existing build tree's archives.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-/tmp/opencode/trie-prototype/ninfer}
B=${B:-/home/kido/network/projects/ninfer-yarn/build}
OUT=${OUT:-$HERE/../work}
DIAG=$OUT/probe-src
CUDA_LINK=$(grep -oE '\-L[^ ]+|/[^ ]*libcudart\.so|/[^ ]*stubs/libcuda\.so' \
  "$B/tests/CMakeFiles/ninfer_grammar_test.dir/link.txt" | sort -u | tr '\n' ' ')
mkdir -p "$OUT"

rm -rf "$DIAG"
mkdir -p "$DIAG/src/models/qwen3_5/frontend/grammar"
cp "$SRC/src/models/qwen3_5/frontend/grammar/grammar.cpp" \
  "$DIAG/src/models/qwen3_5/frontend/grammar/grammar.cpp"
cp "$SRC/src/models/qwen3_5/frontend/grammar/grammar.h" \
  "$DIAG/src/models/qwen3_5/frontend/grammar/grammar.h"
patch -s -p1 -d "$DIAG" < "$HERE/grammar-probe.patch"

FLAGS=(-O3 -DNDEBUG -DNINFER_GRAMMAR_PROBE -std=gnu++20
  -I"$DIAG/src"
  -I"$SRC/include" -I"$SRC/src"
  -I"$SRC/third_party/llama-grammar" -I"$SRC/third_party/llama-grammar/compat"
  -I"$SRC/third_party/llama-chat/common" -I"$SRC/third_party/llama-chat/compat"
  -I"$SRC/third_party/llama-jinja" -isystem "$SRC/third_party")

g++ "${FLAGS[@]}" -c "$HERE/trie_probe.cpp" -o "$OUT/trie_probe.o"
g++ "${FLAGS[@]}" -c "$HERE/trie_producer.cpp" -o "$OUT/trie_producer.o"
g++ "${FLAGS[@]}" -c "$DIAG/src/models/qwen3_5/frontend/grammar/grammar.cpp" \
  -o "$OUT/grammar-probe.o"

g++ "$OUT/trie_probe.o" "$OUT/trie_producer.o" "$OUT/grammar-probe.o" -o "$OUT/trie_probe" \
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

echo "built $OUT/trie_probe"
