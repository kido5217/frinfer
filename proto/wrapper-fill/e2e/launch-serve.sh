#!/usr/bin/env bash
# Prototype launch (ticket #96): worktree-built serve, deployed artifact + flags verbatim,
# mask engaged from token 0 (NINFER_PROTO_WRAPPER=1), request log redirected to this dir.
set -euo pipefail
E2E=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$E2E/../../.." && pwd)
ART=/home/kido/trash/ai/models/hf/hub/models--neroued--Qwen3.8-27B-nvfp4-NInfer/snapshots/f0b43ad436b9fa8142c6ed6647c470a6fe409484/qwen3_8_27b_nvfp4.ninfer
TEMPLATE=/home/kido/network/projects/nixos-configs/users/kido/chat-template/chat_template.jinja

NINFER_PROTO_WRAPPER=1 LD_LIBRARY_PATH=/run/opengl-driver/lib \
  "$SRC/build/apps/ninfer-yarn-serve" "$ART" \
  --host 127.0.0.1 --port 8827 --model-id Qwen3.8-27B \
  --max-context 446902 --default-max-tokens 32768 \
  --kv-dtype nvfp4 --max-concurrency 4 --device-state-slots 0 \
  --host-state-slots 32 --host-kv-mib 24576 \
  --request-log-jsonl "$E2E/request-log.jsonl" \
  --max-pending-requests 64 --pending-timeout-ms 600000 \
  --spec mtp --draft-tokens 4 --lm-head-draft --preserve-thinking --vision \
  --chat-template "$TEMPLATE" \
  --max-long-anchors-per-continuation 4 \
  --max-shared-prefixes 0 \
  --media-cache-mib 2048
