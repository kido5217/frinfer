#!/usr/bin/env bash
# Prototype launch (ticket #97): worktree-built serve (constrained-mode close rule),
# deployed artifact + flags verbatim from nixos-configs/users/kido/llm.nix
# (`ninfer-yarn-qwen38-27b`), request log redirected to the scratch e2e dir.
set -euo pipefail
E2E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$E2E/../.." && pwd)
RUN=/tmp/opencode/proto-close-rule-e2e
mkdir -p "$RUN"
ART=/home/kido/trash/ai/models/hf/hub/models--neroued--Qwen3.8-27B-nvfp4-NInfer/snapshots/f0b43ad436b9fa8142c6ed6647c470a6fe409484/qwen3_8_27b_nvfp4.ninfer
TEMPLATE=/home/kido/network/projects/nixos-configs/users/kido/chat-template/chat_template.jinja

LD_LIBRARY_PATH=/run/opengl-driver/lib \
  "$SRC/build/apps/ninfer-yarn-serve" "$ART" \
  --host 127.0.0.1 --port 8827 --model-id Qwen3.8-27B \
  --max-context 446902 --default-max-tokens 32768 \
  --kv-dtype nvfp4 --max-concurrency 4 --device-state-slots 0 \
  --host-state-slots 32 --host-kv-mib 24576 \
  --request-log-jsonl "$RUN/request-log.jsonl" \
  --max-pending-requests 64 --pending-timeout-ms 600000 \
  --spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision \
  --chat-template "$TEMPLATE" \
  --max-long-anchors-per-continuation 4 \
  --max-shared-prefixes 0 \
  --media-cache-mib 2048
