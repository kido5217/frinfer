#!/usr/bin/env bash
# Verify that the maintained llama-grammar copies are exactly the pristine upstream files at
# the recorded baseline plus the documented patch set in patches/ (see README.frinfer.md).
#
# Usage: [LLAMA_CPP_REF=<path to a llama.cpp clone>] third_party/llama-grammar/verify-vendor.sh
# The reference must contain the baseline commit; a blob-filtered clone works.
set -euo pipefail

BASELINE=05af0d2b1398394cfa67e1918fee7feabccaa9bc
REF=${LLAMA_CPP_REF:-/tmp/opencode/llamacpp-chat-ref}
VENDOR=$(cd "$(dirname "$0")" && pwd)
FILES=(llama-grammar.h llama-grammar.cpp)

if ! git -C "$REF" cat-file -e "$BASELINE^{commit}" 2>/dev/null; then
    echo "reference clone $REF does not contain $BASELINE" >&2
    echo "set LLAMA_CPP_REF to a llama.cpp clone that does" >&2
    exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for file in "${FILES[@]}"; do
    git -C "$REF" show "$BASELINE:src/$file" > "$work/$file"
done

for patch in "$VENDOR"/patches/*.patch; do
    (cd "$work" && patch -p1 -s -i "$patch")
done

for file in "${FILES[@]}"; do
    if ! cmp -s "$work/$file" "$VENDOR/$file"; then
        echo "drift: $file does not match baseline+patches" >&2
        exit 1
    fi
done

echo "llama-grammar vendor matches $BASELINE plus $(ls "$VENDOR"/patches/*.patch | wc -l) patch(es)"
