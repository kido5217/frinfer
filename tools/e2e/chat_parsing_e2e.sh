#!/usr/bin/env bash
# Acceptance gate for the ported Qwen3.5/froggeric chat parsing (map #25, ticket #34, gate #32).
#
# Runs the real product path end to end: `ninfer-yarn-serve` on the neroued Qwen3.8-27B NVFP4
# artifact with the fetched froggeric template, driven by `opencode run --format json`. It asserts
# that the quoted `</think>` stays in the reasoning channel, that no tool-call markup is routed into
# reasoning, and that the model's `shell` call arrives structured - present once, not lost or
# duplicated. Marker words in the visible answer are recorded as evidence, not failures: the corpus
# documents quoted markers and flushed (demoted) candidate regions as legal content
# (tool-call-marker-quoted-without-call, tool-call-quoted-marker-before-real-call).
# Evidence (serve log, request log, opencode JSONL, template digest, decode rate) is written to
# --evidence.
#
# Run this on the host with the real CUDA driver - the flake devShell ships a stub driver, so the
# server cannot run inside a plain `nix develop`. When present, the host driver directory
# (DRIVER_LIB_DIR, default /run/opengl-driver/lib) is prepended to LD_LIBRARY_PATH for the server.
# Build the binary first:
#   nix develop -c cmake --build build --target ninfer-yarn-serve
#
# The opencode provider is not discoverable from the repository or the user configuration: pass
# --model <provider>/<model-id> explicitly. Its provider must be an OpenAI-compatible provider whose
# settings.baseURL is http://HOST:PORT/v1 and whose model key equals the server's public model id
# (verified below against GET /v1/models; see opencode.ai/v2/docs/providers and docs/serving.md).
#
# Usage: tools/e2e/chat_parsing_e2e.sh --model <provider/model> [options]
set -euo pipefail

SERVE_BIN="build/apps/ninfer-yarn-serve"
ARTIFACT="/home/kido/trash/ai/models/hf/hub/models--neroued--Qwen3.8-27B-nvfp4-NInfer/snapshots/f0b43ad436b9fa8142c6ed6647c470a6fe409484/qwen3_8_27b_nvfp4.ninfer"
TEMPLATE="/tmp/opencode/froggeric_chat_template.jinja"
HOST="127.0.0.1"
PORT=8080
API_KEY=""
MODEL=""
EXPECT_TEMPLATE_SHA256=""
SKIP_MODEL_ID_CHECK=0
MIN_FREE_MIB=24000
MAX_FOREIGN_MIB=1536
ALLOW_FOREIGN_PIDS=()
STARTUP_TIMEOUT=900
OPENCODE_BIN="opencode"
OPENCODE_TIMEOUT=900
OPENCODE_STANDALONE=1
EVIDENCE=""
WORKDIR=""
PROMPT=""
SERVE_EXTRA_ARGS=""
OPENCODE_EXTRA_ARGS=()
# Canonical serving profile for Qwen3.8-27B NVFP4 (docs/serving.md quickstart), single request.
SERVE_ARGS=(--max-context 240000 --kv-capacity 240000 --max-concurrency 1 --kv-dtype fp8
  --spec mtp --draft-tokens 3 --lm-head-draft --preserve-thinking)
TOOL_NAME="shell"
TOOL_COMMAND_MARKER="chat-parsing-e2e"
DRIVER_LIB_DIR="${DRIVER_LIB_DIR:-/run/opengl-driver/lib}"

usage() {
  awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"
  cat <<'EOF'

Required:
  --model <provider>/<model-id>   opencode model id for the locally served provider

Optional:
  --artifact PATH                 .ninfer artifact (default: the neroued Qwen3.8-27B NVFP4 snapshot)
  --serve-bin PATH                ninfer-yarn-serve binary (default: build/apps/ninfer-yarn-serve)
  --template PATH                 chat template handed to the server (default: /tmp/opencode/froggeric_chat_template.jinja)
  --expect-template-sha256 HEX    fail unless the template file has this sha256
  --host H / --port N             server address (default: 127.0.0.1:8080)
  --api-key KEY                   server bearer key (also exported as NINFER_API_KEY for opencode)
  --serve-args "ARGS"             extra server flags appended last (later flags win)
  --min-free-mib N                GPU free-memory gate (default: 24000)
  --max-foreign-mib N             fail when another process holds more than this (default: 1536)
  --allow-foreign-pid PID         tolerate this compute pid regardless of usage (repeatable)
  --startup-timeout SECONDS       server readiness timeout (default: 900)
  --opencode-bin PATH             opencode executable (default: opencode)
  --opencode-timeout SECONDS      run timeout (default: 900)
  --no-standalone                 use the shared background service instead of --standalone
  --extra-opencode-arg ARG        extra opencode run argument (repeatable)
  --skip-model-id-check           do not require <model-id> == GET /v1/models id
  --prompt TEXT                   override the elicitation prompt
  --tool-name NAME                opencode tool id the prompt asks for (default: shell)
  --workdir PATH                  directory opencode runs in (default: the evidence directory)
  --evidence DIR                  evidence directory (default: /tmp/opencode/e2e/chat_parsing-<stamp>)
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --artifact) ARTIFACT="$2"; shift 2 ;;
    --serve-bin) SERVE_BIN="$2"; shift 2 ;;
    --template) TEMPLATE="$2"; shift 2 ;;
    --expect-template-sha256) EXPECT_TEMPLATE_SHA256="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --serve-args) SERVE_EXTRA_ARGS="$2"; shift 2 ;;
    --min-free-mib) MIN_FREE_MIB="$2"; shift 2 ;;
    --max-foreign-mib) MAX_FOREIGN_MIB="$2"; shift 2 ;;
    --allow-foreign-pid) ALLOW_FOREIGN_PIDS+=("$2"); shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
    --opencode-bin) OPENCODE_BIN="$2"; shift 2 ;;
    --opencode-timeout) OPENCODE_TIMEOUT="$2"; shift 2 ;;
    --no-standalone) OPENCODE_STANDALONE=0; shift ;;
    --extra-opencode-arg) OPENCODE_EXTRA_ARGS+=("$2"); shift 2 ;;
    --skip-model-id-check) SKIP_MODEL_ID_CHECK=1; shift ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --tool-name) TOOL_NAME="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --evidence) EVIDENCE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

[[ -n "$MODEL" ]] || die "--model <provider>/<model-id> is required (see --help)"
[[ "$MODEL" == */* ]] || die "--model must be <provider>/<model-id>, got: $MODEL"

for tool in python3 curl nvidia-smi sha256sum; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool is required but not on PATH"
done
command -v "$OPENCODE_BIN" >/dev/null 2>&1 || die "opencode executable not found: $OPENCODE_BIN"
[[ -x "$SERVE_BIN" ]] || die "serve binary not found or not executable: $SERVE_BIN (build it: nix develop -c cmake --build build --target ninfer-yarn-serve)"
[[ -f "$ARTIFACT" ]] || die "artifact not found: $ARTIFACT"

if [[ -z "$EVIDENCE" ]]; then
  EVIDENCE="/tmp/opencode/e2e/chat_parsing-$(date -u +%Y%m%d-%H%M%S)"
fi
mkdir -p "$EVIDENCE"
if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$EVIDENCE/workdir"
  mkdir -p "$WORKDIR"
fi
if [[ -z "$PROMPT" ]]; then
  # The prompt never spells out the marker literals: marker text in the visible answer is then the
  # model's own doing, and the assertion block below knows what the corpus considers legal content.
  PROMPT="You are a parser boundary check. In your thinking, quote your chat template's reasoning-close marker exactly once inside double quotes, and note in that same sentence that a quoted marker must not end the reasoning. Then call the ${TOOL_NAME} tool once with the command: echo ${TOOL_COMMAND_MARKER}-ok. Your visible reply must be exactly: DONE"
fi
printf '%s\n' "$PROMPT" >"$EVIDENCE/prompt.txt"

# --- Template -------------------------------------------------------------------------------
if [[ ! -f "$TEMPLATE" ]]; then
  echo "template missing; fetching $TEMPLATE"
  nix develop -c python3.13 -B "$(dirname "$0")/../fetch_froggeric_template.py" "$TEMPLATE"
fi
TEMPLATE_SHA256="$(sha256sum "$TEMPLATE" | cut -d' ' -f1)"
TEMPLATE_BYTES="$(wc -c <"$TEMPLATE" | tr -d ' ')"
if [[ -n "$EXPECT_TEMPLATE_SHA256" && "$TEMPLATE_SHA256" != "$EXPECT_TEMPLATE_SHA256" ]]; then
  die "template $TEMPLATE sha256=$TEMPLATE_SHA256, expected $EXPECT_TEMPLATE_SHA256"
fi
echo "template $TEMPLATE sha256=$TEMPLATE_SHA256 bytes=$TEMPLATE_BYTES"

# --- GPU gate -------------------------------------------------------------------------------
{
  nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
} >"$EVIDENCE/gpu.txt" 2>&1 || die "nvidia-smi failed - the real CUDA driver is required"
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
[[ "$FREE_MIB" =~ ^[0-9]+$ ]] || die "could not read GPU free memory from nvidia-smi"
if ((FREE_MIB < MIN_FREE_MIB)); then
  die "5090 busy: ${FREE_MIB} MiB free, need >= ${MIN_FREE_MIB} MiB (see $EVIDENCE/gpu.txt; free the card for #32)"
fi
BUSY_PIDS=()
while IFS=, read -r pid name used; do
  pid="${pid// /}"; name="${name# }"; used="${used// /}"
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  used_mib="${used% MiB}"
  [[ "$used_mib" =~ ^[0-9]+$ ]] || continue
  allowed=0
  for allow in ${ALLOW_FOREIGN_PIDS[@]+"${ALLOW_FOREIGN_PIDS[@]}"}; do
    if [[ "$pid" == "$allow" ]]; then
      allowed=1
    fi
  done
  if ((allowed == 0 && used_mib > MAX_FOREIGN_MIB)); then
    BUSY_PIDS+=("$pid:$name:${used_mib}MiB")
  fi
done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if ((${#BUSY_PIDS[@]} > 0)); then
  die "5090 busy: foreign compute processes over ${MAX_FOREIGN_MIB} MiB: ${BUSY_PIDS[*]} (free the card for #32 or raise --max-foreign-mib)"
fi
echo "gpu free=${FREE_MIB}MiB gate ok"

# --- Server ---------------------------------------------------------------------------------
SERVE_LOG="$EVIDENCE/serve.log"
SERVE_PID=""
cleanup() {
  if [[ -n "$SERVE_PID" ]] && kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$SERVE_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# The real driver must win over any stub libcuda the surrounding toolchain exposes (the flake
# devShell ships one; see the header note).
if [[ -e "$DRIVER_LIB_DIR/libcuda.so.1" && ":${LD_LIBRARY_PATH:-}:" != *":$DRIVER_LIB_DIR:"* ]]; then
  export LD_LIBRARY_PATH="$DRIVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "serve: prepended host driver dir $DRIVER_LIB_DIR to LD_LIBRARY_PATH"
fi

SERVE_CMD=("$SERVE_BIN" "$ARTIFACT" --host "$HOST" --port "$PORT" --chat-template "$TEMPLATE"
  --request-log-jsonl "$EVIDENCE/serve-requests.jsonl")
if [[ -n "$API_KEY" ]]; then
  SERVE_CMD+=(--api-key "$API_KEY")
  export NINFER_API_KEY="$API_KEY"
fi
SERVE_CMD+=("${SERVE_ARGS[@]}")
if [[ -n "$SERVE_EXTRA_ARGS" ]]; then
  # --serve-args is appended last so it can override the profile defaults.
  read -r -a EXTRA <<<"$SERVE_EXTRA_ARGS"
  SERVE_CMD+=("${EXTRA[@]}")
fi
printf '%s\n' "${SERVE_CMD[@]}" >"$EVIDENCE/serve-command.txt"
echo "starting serve (log: $SERVE_LOG)"
"${SERVE_CMD[@]}" >"$SERVE_LOG" 2>&1 &
SERVE_PID=$!

BASE_URL="http://$HOST:$PORT"
DEADLINE=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "$BASE_URL/health" 2>/dev/null | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    die "serve exited during startup; see $SERVE_LOG"
  fi
  if ((SECONDS >= DEADLINE)); then
    die "serve not healthy after ${STARTUP_TIMEOUT}s; see $SERVE_LOG"
  fi
  sleep 1
done
echo "serve healthy on $BASE_URL"

AUTH_HEADER=()
if [[ -n "$API_KEY" ]]; then
  AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi
MODELS_JSON="$(curl -fsS "${AUTH_HEADER[@]}" "$BASE_URL/v1/models")" ||
  die "GET /v1/models failed"
PUBLIC_MODEL_ID="$(printf '%s' "$MODELS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" ||
  die "could not read the public model id from: $MODELS_JSON"
printf '%s\n' "$MODELS_JSON" >"$EVIDENCE/serve-models.json"
REQUESTED_MODEL_ID="${MODEL#*/}"
if ((SKIP_MODEL_ID_CHECK == 0)) && [[ "$REQUESTED_MODEL_ID" != "$PUBLIC_MODEL_ID" ]]; then
  die "model id mismatch: the serve exposes '$PUBLIC_MODEL_ID' but --model asks for '$REQUESTED_MODEL_ID'. Point the opencode provider model key at the server id (or pass --skip-model-id-check if the provider rewrites it)."
fi
echo "serve model id=$PUBLIC_MODEL_ID requested=$REQUESTED_MODEL_ID"

# --- opencode -------------------------------------------------------------------------------
OPENCODE_ARGS=(run --format json --auto --model "$MODEL" --thinking)
if ((OPENCODE_STANDALONE == 1)); then
  OPENCODE_ARGS+=(--standalone)
fi
OPENCODE_ARGS+=("${OPENCODE_EXTRA_ARGS[@]}" "$PROMPT")
printf '%s\n' "${OPENCODE_ARGS[@]}" >"$EVIDENCE/opencode-command.txt"
echo "driving: $OPENCODE_BIN ${OPENCODE_ARGS[*]:0:6} ..."
set +e
(cd "$WORKDIR" && timeout -k 5s "$OPENCODE_TIMEOUT" "$OPENCODE_BIN" "${OPENCODE_ARGS[@]}") \
  >"$EVIDENCE/opencode.jsonl" 2>"$EVIDENCE/opencode.stderr"
OPENCODE_RC=$?
set -e
echo "opencode exit=$OPENCODE_RC (jsonl: $EVIDENCE/opencode.jsonl)"
if ((OPENCODE_RC != 0)); then
  tail -5 "$EVIDENCE/opencode.stderr" >&2 || true
  die "opencode run failed with exit $OPENCODE_RC"
fi

# --- Assertions -----------------------------------------------------------------------------
python3 - "$EVIDENCE/opencode.jsonl" "$TOOL_NAME" "$TOOL_COMMAND_MARKER" "$EVIDENCE/assertions-warnings.txt" <<'PY'
import json
import sys

path, tool_name, marker = sys.argv[1], sys.argv[2], sys.argv[3]
events, unparsed = [], []
with open(path, encoding="utf-8") as source:
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            unparsed.append(line)

texts = [
    event["part"]["text"]
    for event in events
    if event.get("type") == "text" and isinstance(event.get("part"), dict)
    and isinstance(event["part"].get("text"), str)
]
reasoning = [
    event["part"]["text"]
    for event in events
    if event.get("type") == "reasoning" and isinstance(event.get("part"), dict)
    and isinstance(event["part"].get("text"), str)
]
tools = {}
for event in events:
    if event.get("type") != "tool_use" or not isinstance(event.get("part"), dict):
        continue
    part, state = event["part"], event["part"].get("state") or {}
    if state.get("status") == "completed":
        tools[part.get("id")] = (part.get("tool"), state.get("input"))
visible = "\n".join(texts)
visible_reasoning = "\n".join(reasoning)

failures, warnings = [], []
# Visible content is free text: the corpus documents that a quoted marker and a flushed (demoted)
# candidate region may legally appear in the answer (tool-call-marker-quoted-without-call,
# tool-call-quoted-marker-before-real-call). Marker words in the answer are recorded, not failed.
markers = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "<parameter="]
in_visible = [name for name in markers if name in visible]
if in_visible:
    warnings.append(f"marker text in visible content (contract-legal, recorded): {in_visible}")
if ("<tool_call>" in visible_reasoning or "<function=" in visible_reasoning
        or "<parameter=" in visible_reasoning):
    failures.append("tool-call markup appeared in the reasoning channel")
if not reasoning:
    failures.append(
        "no reasoning events captured: the quoted-</think> boundary was not exercised "
        "(provider must surface reasoning_content; --thinking is already passed)"
    )
elif "</think>" not in visible_reasoning:
    failures.append("the model did not quote </think> in its reasoning; rerun with a stricter prompt")
matching = [
    (part_id, tool, data)
    for part_id, (tool, data) in tools.items()
    if tool == tool_name and marker in json.dumps(data, ensure_ascii=False)
]
all_tools = {part_id: (tool, data) for part_id, (tool, data) in tools.items()}
if len(matching) != 1:
    failures.append(
        f"expected exactly one completed {tool_name} call containing '{marker}', "
        f"saw {len(matching)}; all completed tool calls: {all_tools}"
    )

print(f"events={len(events)} unparsed={len(unparsed)} text_parts={len(texts)} "
      f"reasoning_parts={len(reasoning)} completed_tool_calls={len(tools)}")
if matching:
    print(f"tool call: {matching[0][0]} {matching[0][1]} input={json.dumps(matching[0][2], ensure_ascii=False)}")
if visible:
    print(f"visible text: {visible[:200]!r}")
if unparsed:
    print(f"warning: {len(unparsed)} non-JSON line(s) on stdout, first: {unparsed[0][:120]!r}")
for warning in warnings:
    print(f"WARNING: {warning}")
if warnings:
    with open(sys.argv[4], "w", encoding="utf-8") as sink:
        sink.write("\n".join(warnings) + "\n")
if failures:
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    raise SystemExit(1)
print("assertions ok: structured call present once; reasoning channel clean")
PY

# --- Evidence summary -----------------------------------------------------------------------
DECODE_RATE="$(grep -o '| decode [^|]*' "$SERVE_LOG" | tail -1 | sed 's/^| decode //;s/ *$//' || true)"
[[ -n "$DECODE_RATE" ]] || DECODE_RATE="n/a"
{
  echo "artifact=$ARTIFACT"
  echo "serve_bin=$SERVE_BIN"
  echo "serve_command=${SERVE_CMD[*]}"
  echo "model=$MODEL"
  echo "template=$TEMPLATE"
  echo "template_sha256=$TEMPLATE_SHA256"
  echo "template_bytes=$TEMPLATE_BYTES"
  echo "gpu_free_mib=$FREE_MIB"
  echo "decode_rate=$DECODE_RATE"
  echo "assertions_warnings=$(if [[ -f "$EVIDENCE/assertions-warnings.txt" ]]; then tr '\n' ' ' <"$EVIDENCE/assertions-warnings.txt"; else echo none; fi)"
  echo "opencode_exit=$OPENCODE_RC"
  echo "evidence=$EVIDENCE"
} >"$EVIDENCE/summary.txt"
echo "decode rate (last request): $DECODE_RATE"
cat "$EVIDENCE/summary.txt"
echo "PASS: chat parsing e2e"
