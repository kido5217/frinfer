#!/usr/bin/env bash
# Full run matrix for the throwaway logprob_topk prototype.
# Builds, generates inputs for every case/shape, runs the FP64 oracle, benchmarks,
# and writes results/summary.md.
#
# Run from anywhere as:
#   nix develop /home/kido/network/projects/frinfer -c bash tools/spike/logprobs-evidence/run.sh
set -euo pipefail
cd "$(dirname "$0")"

# On NixOS the CUDA runtime dlopens libcuda.so.1 from /run/opengl-driver/lib,
# which is not on the devShell's search path; add it if present.
if [ -d /run/opengl-driver/lib ]; then
  export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
fi

bash build.sh

RESULTS="results"
DATA="$RESULTS/data"
rm -rf "$RESULTS"
mkdir -p "$DATA"

run_case() {
  local spec="$1"          # e.g. plain-t1:r1
  local tag="${spec//:/_}" # file-safe
  local dir="$DATA/$tag"
  mkdir -p "$dir"
  timeout -k 5s 120s ./gather_prototype --gen "$spec" "$dir" < /dev/null | tee "$dir/gen.txt"
  # oracle.py exits nonzero on a mismatch; keep collecting the rest of the matrix.
  timeout -k 5s 120s python3.13 oracle.py "$dir" < /dev/null | tee "$dir/oracle.txt" || true
  timeout -k 5s 120s ./gather_prototype --bench "$spec" < /dev/null | tee "$dir/bench.txt"
}

# V=151936 for every case; rows {1,8}; the speculative case is rows=8 with k+1=5.
for base in plain-t1 plain-t07 greedy penalty masked mask-penalty; do
  for r in 1 8; do
    run_case "${base}:r${r}"
  done
done
run_case "spec-k4:r8w5"

python3.13 summarize.py "$RESULTS" | tee "$RESULTS/summary.md"
echo "---"
echo "results written to $(pwd)/$RESULTS/summary.md"
