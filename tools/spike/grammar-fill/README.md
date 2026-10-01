# Grammar fill probe (wayfinder ticket #57)

Measures the per-state mask-production cost of the implemented grammar module
(`src/models/qwen3_5/frontend/grammar/`, `ninfer_grammar`) on the served 248,077-token
vocabulary: fixture walks (G1–G4), fills vs distinct masks, cache behavior, and phase attribution
inside a fill. The evidence record is `docs/research/grammar-fill-cost.md` on branch
`research/grammar-fill-cost`; the escalation decision it serves is
`docs/maintainer/grammar-mask-production.md` (master).

## Prerequisites

- A configured build tree with tests enabled (`build/`), so the probe can link
  `libninfer_grammar.a` and the `ninfer_grammar_test` link line.
- The tokenizer snapshot the served artifact names:
  `models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Everything runs inside the repository devShell (`nix develop`), from the repository root.

## Build and run

```bash
SNAP=$HOME/trash/ai/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
OUT=/tmp/opencode/grammar-fill
nix develop -c bash tools/spike/grammar-fill/build.sh          # -> $OUT/probe

$OUT/probe --snapshot "$SNAP" --schema tools/spike/grammar-fill/diary.json \
    --mode lowest --steps 20000 | tee tools/spike/grammar-fill/results/diary.txt
$OUT/probe --snapshot "$SNAP" --grammar tools/spike/grammar-fill/grammars/g2_bounded_line.gbnf \
    --mode lowest --steps 20000
```

`build.sh` and `build_diag.sh` resolve the repository from their own location; override `SRC`,
`B` (build tree) or `OUT` (work directory, default `/tmp/opencode/grammar-fill`) as needed.

`build_diag.sh` applies `diag.patch` to a private copy of `grammar.cpp`/`grammar.h` and compiles it
with `-DNINFER_GRAMMAR_DIAG`; master is never modified. The diag probe adds `--diag PATH`, which
writes the per-call JSONL (state key words, served row id, mask hash) plus `PATH.rules` (grammar
rules and counter-normalization representatives) and `PATH.phases` (µs accumulators for key build,
candidate reset, engine apply, row packing, dedup, total).

## Layout

| Path | What |
|---|---|
| `probe.cpp` | the walk driver (loads the production tokenizer, compiles a fixture, walks it) |
| `build.sh`, `build_diag.sh` | builds against a configured `build/` tree |
| `diag.patch` | `NINFER_GRAMMAR_DIAG` instrumentation against master (`e5d41e2e`) |
| `diary.json` | the G4 diary JSON schema (upstream `Neroued/ninfer` #33's acceptance subset) |
| `grammars/g1..g3` | the mask-cost spike's GBNF fixtures |
| `results/` | captured runs: `diary.txt`, `diary-diag.txt` + `.jsonl{,.rules,.phases}`, `g1..g3.txt`, `analysis.txt` |
| `analysis/` | the scripts behind the numbers (stack statistics, fold simulation, context counterexample, long-piece scan) + `dump_vocab.py` (from the mask-cost spike) |

## Analysis scripts

They read `results/diary-diag.jsonl{,.rules}` from this directory and a vocabulary dump at
`$VOCAB` (default `/tmp/opencode/grammar-fill/vocab.tsv`):

```bash
OUT=/tmp/opencode/grammar-fill
nix develop -c python3 tools/spike/grammar-fill/analysis/dump_vocab.py "$SNAP/tokenizer.json" \
    --tokenizer-config "$SNAP/tokenizer_config.json" --out "$OUT/vocab.tsv"
VOCAB=$OUT/vocab.tsv python3 tools/spike/grammar-fill/analysis/stack_stats.py
VOCAB=$OUT/vocab.tsv python3 tools/spike/grammar-fill/analysis/simulate_fold.py
python3 tools/spike/grammar-fill/analysis/check_fold3.py
python3 tools/spike/grammar-fill/analysis/long_tokens.py "$SNAP"
```

`analysis.txt` in `results/` is the captured output of those runs.

## Notes

- `--mode lowest` reproduces the deterministic maximal walk used for the budget statement (picks
  the lowest-id allowed token); `--mode random --seed N` is the seeded alternative.
- `--dump N` prints the first N accepted pieces with per-step fill/hit and mask hash.
- The probe applies no EOG handling beyond the module's own rows; walks stop when no non-EOG token
  is allowed. G3's chain is unbounded, so its walk runs to the step cap.
