#!/usr/bin/env bash
set -euo pipefail

# GPQA-Diamond context-window comparison (map kido5217/ninfer-yarn#129):
# run the same artifact and serving flags at the native context window
# (262,144) and the deployed YaRN window (446,902); only --max-context
# differs. nvfp4 KV is required for the 446,902 arm to fit the RTX 5090.
# Concurrency is one so the per-request 245,760-token output budget always
# fits the shared KV. Outputs land under profiles/eval/... per tier.

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
server_bin="${repo_dir}/build/apps/ninfer-yarn-serve"
artifact_hub="${NINFER_ARTIFACT_HUB:-/home/kido/trash/ai/models/hf/hub}"
artifact="${NINFER_EVAL_ARTIFACT:-${artifact_hub}/models--neroued--Qwen3.8-27B-nvfp4-NInfer/snapshots/f0b43ad436b9fa8142c6ed6647c470a6fe409484/qwen3_8_27b_nvfp4.ninfer}"
config="${repo_dir}/eval/configs/qwen3_8_27b_nvfp4_gpqa_windows.yaml"
eval_python="${repo_dir}/eval/.venv/bin/python"
campaign_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_root="${repo_dir}/profiles/eval/qwen3_8_27b_nvfp4_gpqa_windows_${campaign_stamp}"

usage() {
    echo "usage: $0 [--plan] [native|yarn]" >&2
    echo "       with no step argument, native runs followed by yarn" >&2
}

# Emits tab-separated fields: suite, max-context, description.
step_config() {
    case "$1" in
        native)
            printf 'native\t262144\t%s\n' "GPQA-Diamond at the native context window"
            ;;
        yarn)
            printf 'yarn\t446902\t%s\n' "GPQA-Diamond at the deployed YaRN context window"
            ;;
        *)
            return 2
            ;;
    esac
}

plan_only=0
if [[ "${1:-}" == "--plan" ]]; then
    plan_only=1
    shift
fi
if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi
case "${1:-}" in
    "")
        steps=(native yarn)
        ;;
    native | yarn)
        steps=("$1")
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ "${plan_only}" -eq 1 ]]; then
    for step in "${steps[@]}"; do
        IFS=$'\t' read -r suite _ _ <<<"$(step_config "${step}")"
        PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval plan \
            --config "${config}" --suite "${suite}" --check-runtime
    done
    exit 0
fi

for required_file in "${server_bin}" "${artifact}" "${config}" "${eval_python}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "missing required file: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "${server_bin}" || ! -x "${eval_python}" ]]; then
    echo "ninfer-yarn-serve and the evaluation Python must be executable" >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required for the server health check" >&2
    exit 1
fi
if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
    echo "port 18080 already has a healthy service; stop it before running this campaign" >&2
    exit 1
fi

mkdir -p -- "${output_root}"
echo "campaign output: ${output_root}"

server_pid=""
cleanup_server() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM "${server_pid}"
        wait "${server_pid}" || true
    fi
    server_pid=""
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_tier() {
    local tier="$1"
    local suite="$2"
    local max_context="$3"
    local description="$4"
    local tier_dir="${output_root}/${tier}"
    local server_log="${tier_dir}/server.log"
    local request_log="${tier_dir}/server.requests.jsonl"

    mkdir -p -- "${tier_dir}"
    echo "starting tier=${tier} context=${max_context} (${description})"

    "${server_bin}" "${artifact}" \
        --host 127.0.0.1 \
        --port 18080 \
        --model-id qwen3.8-27b \
        --max-context "${max_context}" \
        --max-concurrency 1 \
        --max-pending-requests 1 \
        --pending-timeout-ms 86400000 \
        --prefill-chunk 1024 \
        --kv-dtype nvfp4 \
        --spec mtp \
        --draft-tokens 4 \
        --lm-head-draft \
        --request-log-jsonl "${request_log}" \
        >"${server_log}" 2>&1 &
    server_pid=$!

    local ready=0
    for ((attempt = 1; attempt <= 180; ++attempt)); do
        if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:18080/health >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            wait "${server_pid}" || true
            echo "ninfer-yarn-serve exited before becoming ready; see ${server_log}" >&2
            exit 1
        fi
        sleep 1
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "ninfer-yarn-serve did not become ready within 180 seconds; see ${server_log}" >&2
        exit 1
    fi

    echo "running suite=${suite}; server_log=${server_log}; request_log=${request_log}"
    PYTHONPATH="${repo_dir}/eval" "${eval_python}" -m ninfer_eval run \
        --config "${config}" --suite "${suite}"

    cleanup_server
    echo "completed tier=${tier}"
}

for step in "${steps[@]}"; do
    IFS=$'\t' read -r suite max_context description <<<"$(step_config "${step}")"
    run_tier "${step}" "${suite}" "${max_context}" "${description}"
done

echo "campaign completed: ${output_root}"
