# Compiled-mask producer prototype (ticket #59)

Throwaway verification harness — evidence only, no product change. Companion doc:
[docs/research/compiled-mask-producer.md](../../docs/research/compiled-mask-producer.md).

## What it is

An in-house token-trie producer for per-state mask rows, walking NInfer's vendored
`llama-grammar` engine, verified differentially against the engine's own row walk
(module v1, `src/models/qwen3_5/frontend/grammar/`). Three cost layers:

1. **Class-counter fast path** — per-(body, C-placement) counter tables (per-token round
   thresholds + continuation handoff correction + partial thresholds); a fill is a
   memcpy + threshold compare.
2. **Exact trie fallback** — bulk / descent / exclusion / tail.
3. **Producer-owned state-shape row cache** — 128-bit content hash over
   (root-position content, can_end, pending partial, ablation switches) → 31 KB row;
   adaptive store (rows copied only for repeated shapes). `TRIE_SHAPE_CACHE_OFF=1`
   disables it — differential-oracle runs must use it.

## Build

Links against a configured `build/` tree's archives and a clean repo checkout:

```bash
SRC=<repo-checkout> B=<build-dir> nix develop -c bash tools/spike/compiled-mask/producer/build.sh
```

- `SRC` — a checkout of this repo (unmodified; `grammar-probe.patch` and
  `grammar-nocache.patch` are applied to a private copy, never the tree).
- `B` — the build dir with the `ninfer_*.a` archives (same tree as the dev flake build).
- Output: `tools/spike/compiled-mask/work/trie_probe`.

`grammar-nocache.patch` adds a `GRAMMAR_ROW_NOCACHE` env bypass to the module's row
cache (semantics test only: fresh applies vs cached rows).

## Run

Fixtures are the #57 acceptance fixtures (`../grammar-fill/grammars/`, `../grammar-fill/diary.json`):

```bash
SNAP=$HOME/trash/ai/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
P=tools/spike/compiled-mask/work/trie_probe
G=tools/spike/grammar-fill

# Pass 1 — correctness (fresh differential oracle; producer shape cache OFF)
env TRIE_SHAPE_CACHE_OFF=1 $P --snapshot "$SNAP" --grammar $G/grammars/g1_literal_alternation.gbnf --mode lowest --steps 20000
env TRIE_SHAPE_CACHE_OFF=1 $P --snapshot "$SNAP" --grammar $G/grammars/g2_bounded_line.gbnf --mode lowest --steps 20000
env TRIE_SHAPE_CACHE_OFF=1 $P --snapshot "$SNAP" --grammar $G/grammars/g3_key_value_record.gbnf --mode lowest --steps 20000
env TRIE_SHAPE_CACHE_OFF=1 $P --snapshot "$SNAP" --schema $G/diary.json --mode lowest --steps 20000

# Pass 2 — cost (shape cache ON = serving configuration)
$P --snapshot "$SNAP" --grammar $G/grammars/g2_bounded_line.gbnf --mode lowest --steps 20000
$P --snapshot "$SNAP" --grammar $G/grammars/g3_key_value_record.gbnf --mode lowest --steps 20000
$P --snapshot "$SNAP" --schema $G/diary.json --mode lowest --steps 20000

# Semantics checks
env GRAMMAR_ROW_NOCACHE=1 $P --snapshot "$SNAP" --grammar $G/grammars/g2_bounded_line.gbnf --mode lowest --steps 240   # fresh applies
$P --snapshot "$SNAP" --grammar $G/grammars/g2_bounded_line.gbnf --mode lowest --steps 400                              # termination past max
$P --snapshot "$SNAP" --grammar $G/grammars/g2_bounded_line.gbnf --mode lowest --steps 20000 --no-fast                   # exact-fallback mode
```

`results/` holds the recorded runs (2026-10-02):

| File | Run |
|---|---|
| `p1-g1.txt`, `p1-g2.txt`, `p1-g3.txt`, `p1-diary2.txt` | pass 1 (correctness, cache off) |
| `p2b-g1.txt`, `p2b-g2.txt`, `p2b-g3.txt`, `p2b-diary.txt` | pass 2 (cost, cache on, seen-set store) |
| `g2-nocache.txt` | G2 vs fresh no-cache engine applies |
| `g2-400.txt` | G2 walk past max=240 (termination check) |

## Debug environment variables (producer/probe)

`TRIE_SHAPE_CACHE_OFF` (cache off), `TRIE_DEBUG_REC` (recognition reasons + raw engine
stacks), `TRIE_DEBUG_STK` (root positions / prepared / class tables at a fill),
`TRIE_DEBUG_ROW` (producer row dump at fills 15/16), `TRIE_DEBUG_REFROW` (reference row
hash at steps < 8), `TRIE_DEBUG_FILL` (fast-fill internals for id 11611),
`TRIE_DEBUG_REC` budget: first 8 failures.
