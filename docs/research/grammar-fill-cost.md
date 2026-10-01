# Grammar fill cost — per-state mask production on the real 248 k vocabulary

- **Date:** 2026-10-01
- **Branch:** `research/grammar-fill-cost` (throwaway; no PR, no product change — measurement and
  decision evidence only)
- **Ticket:** [Fix: mask-fill cost and retention — #57](https://github.com/kido5217/ninfer-yarn/issues/57)
- **Companions:** [mask-cost spike](../../tools/spike/mask-cost/README.md)
  (`docs/research/mask-cost-spike.md` on `research/mask-cost-spike`), the sources doc on
  `research/constrained-decoding-sources`, the compiler's own inputs
  (`tools/spike/mask-cost/grammars/g4_diary_schema.json`).

## 1. Question

Ticket #57 asked whether the three identified levers — (1) per-chain observable-position folding,
(2) a precomputed-codepoint / transition walk, (3) a row-retention policy — can bring mask
production within the serving budget (0.4 ms/step, the 10 % line of the 4 ms/token decode budget),
or whether the documented xgrammar-style compiled-mask escalation is required.

This measures the **implemented module** (`src/models/qwen3_5/frontend/grammar/`, `ninfer_grammar`)
on the served 248,077-token vocabulary, drives the acceptance-style fixture walks, and attributes
the fill cost. It does not measure xgrammar (not built here).

## 2. Method

### 2.1 Harness

`tools/spike/grammar-fill/probe.cpp` links the project's `ninfer_grammar` against a configured
`build/` tree, loads the tokenizer snapshot through the production `Tokenizer` →
`TokenizerVocabulary` path, compiles a fixture (GBNF file or JSON Schema through the vendored
converter), and walks the grammar: at each step it takes `row_for(state)` (timed), picks the
lowest-id allowed non-EOG token (deterministic, maximal walk), and `accept`s it. `--diag` writes a
per-call JSONL (state key words, served row id and mask hash) through the `diag.patch`
instrumentation, plus phase accumulators.

```bash
SNAP=$HOME/trash/ai/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
nix develop -c bash tools/spike/grammar-fill/build.sh
tools/spike/grammar-fill/probe --snapshot "$SNAP" --schema tools/spike/grammar-fill/diary.json \
    --mode lowest --steps 20000 | tee tools/spike/grammar-fill/results/diary.txt
```

The diag build follows the same shape (`build_diag.sh`; applies `diag.patch` to a copy of
`grammar.cpp` + an overlay `grammar.h`).

### 2.2 Vocabulary

The tokenizer snapshot named by the served artifact's manifest
(`models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), loaded through
the production frontend tokenizer. N = **248,077**; mask row = 31,012 B; **max piece = 128 bytes**
(85 pieces > 64 B, 360 > 32 B, 6 882 > 16 B). The module's fold limit is
`max_piece_bytes + 2 = 130`.

### 2.3 Fixtures

The mask-cost spike's G1/G2/G3 GBNF files and the diary JSON schema (G4) from upstream
`Neroued/ninfer`'s #33 acceptance subset, walked to exhaustion (G3's chain is unbounded, capped
at 20 000 steps).

## 3. Results

### 3.1 Fixture walks (lowest-token walk, first walk on a fresh grammar)

| Fixture | steps | fills | distinct masks | hits | fill time | ms/step |
|---|---|---|---|---|---|---|
| G1 `"да" \| "нет"` | 4 | 5 | 5 | 0 | 84 ms | 21.0 |
| G2 `[^\n]{3,240}` | 240 | 134 | 94 | 107 | 3.55 s | **14.8** |
| G3 key=value record | 20 000 (cap) | 23 | 16 | 19 977 | 0.45 s | 0.02 |
| G4 diary schema | 4 226 | 1 957 | 409 | 2 270 | **67.1 s** | **15.9** |

The budget line is 0.4 ms/step. Even the one-line G2 grammar exceeds it 37×; the diary walk
exceeds it 40×. Cached rows are cheap (hit p50 0.4 µs), so the entire cost is in fills.

### 3.2 Where a fill goes (diary walk, full run)

Engine-side phases (`diag.patch` accumulators, 4 227 `row_for` calls, 1 957 fills):

| Phase | total | accrual |
|---|---|---|
| `llama_grammar_apply_impl` (the full-vocabulary walk) | 66.69 s | 34.1 ms per fill |
| candidate-array reset | 0.26 s | 134 µs per fill |
| row bit-pack (248 077 logits → 7 753 words) | 0.67 s | 341 µs per fill |
| state-key build + hash | 0.002 s | 0.56 µs per **call** (hits included) |
| content hash + dedup compare | 0.06 s | 31 µs per fill |

The engine walk is **98.5 %** of the fill; the module's own work (keys, reset, row packing,
dedup) is 1.5 %. The median `row_for` call is a cache hit (p50 0.4–0.5 µs); the fills are the
long tail (p90 36 ms, max 60 ms).

Within the engine walk, the mask-cost spike's attribution on the same baseline measured
8.85–9.21 ms of per-candidate `decode_utf8` + allocation (one heap allocation per piece: 248 075
per fill) and 0.64–14.48 ms reject walk. Removing the decode stage entirely therefore leaves
≈ 25 ms per fill (34.1 ms − 9.2 ms).

### 3.3 Why the fills repeat but cannot be folded away

1 957 fills produced only 409 distinct mask contents: **1 548 fills recomputed a row that already
existed**. The dominant pattern is a nested-counter product. In one group, **180 distinct state
keys share one mask**, differing along two chain axes:

- the `facts` item chain `("," space string){0,13}` (rules 9–19, positions 3–13 plus the
  first-item context), 12 states, and
- a `string` char chain near its top (rules 118–132, positions 98–112), 15 positions.

The 180 keys are a complete 12 × 15 Cartesian grid — every item state × char position holds
exactly one key, all mapping to the same row. The same shape recurs across the
walk (87 masks × 12 fills, 3 × 48, one × 84, one × 180).

The positions are not *universally* equivalent, which is what a canonical key would need:

- **The sound bound is physical.** A token can consume at most as many rounds as its piece has
  codepoints; with `max_piece_bytes = 128` the existing fold (positions > 130 → 130) is the only
  context-free fold that is provably safe. Per-body refinement does not help the dominant chains:
  the char-chain's true reach measured over the vocabulary is 128 (token `55036`, 128 spaces; then
  113, 114, 97/98 …), so its threshold is 128 versus the global 130.
- **The measured equivalence is context-dependent.** In the deep-string context
  (`root`/`string`/`char` + item chain at link position), item positions 3–13 share mask
  `719bc5c6…`, but at the walk's final step (same context shape, call 4088) item position 2 has a
  *different* mask (`6170f697…`). Any context-free canonical key folding those positions would
  return a wrong row there (a position can also be observable through a token that enters a round
  partially, not only through complete traversals). A simulation folding item positions to one
  class collapses 1 957 fills to 602 but merges 92 masks — it is unsound, not an optimization.
  Computing the true equivalence would require the masks themselves — i.e. the walk.

### 3.4 Alternatives measured or bounded

- **Per-stack mask caching** (reuse a row per stack instead of per state, union at lookup): the
  walk's 1 957 states contain **3 888 distinct stacks** (12 610 instances; 3 867 appear once) —
  strictly worse than caching per state.
- **Retention** (lever 3) is a safety fix, not a cost fix: 409 stored rows = 12.1 MiB after one
  diary walk, and the state→row map grows per visited state (one key per step); unbounded today.
- **Floor projection.** With perfect (unattainable) folding the diary walk still pays 409 ×
  34.1 ms ≈ 13.9 s; with the decode stage eliminated as well ≈ 409 × 25 ms ≈ 10.2 s. The budget is
  **1.69 s** (0.4 ms × 4 226). The per-state full-vocabulary walk cannot reach it.

## 4. Conclusion

The levers cannot close the gap, and the gap is not a constant factor: it is the per-state
O(vocabulary) walk itself. The escalation named in wayfinder map #45's contingency is triggered:
mask production must move to compiled/adaptive masks (xgrammar-style vocabulary-trie
classification with bulk subtree acceptance and per-state caching) behind the existing module
seam. The decision is recorded in `docs/maintainer/grammar-mask-production.md` (master) and the
ticket resolution; the design of the replacement is its own ticket.

## 5. Caveats

- The walk is the deterministic lowest-allowed-token walk (the stress case the design ticket
  referenced); a model-sampled generation visits fewer states per token budget, but the fixtures
  are the acceptance bar.
- G3's walk is cyclic and capped at 20 000 steps (23 fills); its ms/step is not representative of
  a terminating walk.
- The diag instrumentation was compiled into a copy of the module (`-O2 -g`); the production,
  uninstrumented run (`results/diary.txt`) matches within run-to-run variance (67.1 s vs 67.6 s
  diag).
- No xgrammar build was in scope; its per-state cost is a projection from its documented design,
  to be measured by the replacement's own ticket.
- Host: AMD Ryzen 7 5800X3D, 128 GiB RAM, NixOS; devShell GCC 15.3; single-threaded.

## Appendix — raw artifacts

`tools/spike/grammar-fill/` contains the probe, both build scripts, `diag.patch`, the diary schema,
the G1–G3 grammars, and `results/`: the four fixture runs, the diary diag JSONL (+ `.rules`,
`.phases`), and `analysis.txt` (stack statistics, the fold simulation, the context counterexample,
the vocabulary long-piece scan). `analysis/` holds the scripts that produce it.
