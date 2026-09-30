#!/usr/bin/env bash
# Builds the mask-cost spike harness and the JSON-schema->GBNF driver.
#
# Usage: tools/spike/mask-cost/build.sh [build-dir]
# Default build dir: tools/spike/mask-cost/build (ignored; use a /tmp path to
# keep the worktree clean). Run inside the repo's nix devShell:
#   nix develop -c bash tools/spike/mask-cost/build.sh /tmp/opencode/mask-cost-build
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
out="${1:-$here/build}"
mkdir -p "$out"

cxx="${CXX:-g++}"
flags=(-std=c++17 -O3 -DNDEBUG -Wall -Wextra)

echo "== mask-cost-harness =="
"$cxx" "${flags[@]}" \
    -I"$here/shim" -I"$here/upstream" \
    "$here/upstream/llama-grammar.cpp" \
    "$here/harness.cpp" \
    -o "$out/mask-cost-harness"

echo "== gbnf_from_schema =="
"$cxx" "${flags[@]}" \
    -I"$repo/third_party/llama-chat/common" \
    -I"$repo/third_party/llama-chat/compat" \
    -I"$repo/third_party/llama-jinja" \
    -I"$repo/third_party" \
    "$here/gbnf_from_schema.cpp" \
    "$repo/third_party/llama-chat/common/json-schema-to-grammar.cpp" \
    "$repo/third_party/llama-chat/common/json-schema.cpp" \
    "$repo/third_party/llama-chat/common/json.cpp" \
    "$repo/third_party/llama-chat/common/trie.cpp" \
    "$repo/third_party/llama-chat/common/unicode.cpp" \
    "$repo/third_party/llama-chat/compat/common_helpers.cpp" \
    -o "$out/gbnf_from_schema"

echo "built: $out/mask-cost-harness, $out/gbnf_from_schema"
