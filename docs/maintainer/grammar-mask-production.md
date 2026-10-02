# Grammar mask production

How constrained decoding produces its per-state token masks — the serving budget, the measured
cost of the current producer, and the decision to replace it with compiled/adaptive masks. The
grammar runtime itself is covered by
[vendored llama.cpp grammar runtime](llama-grammar-vendor.md); the effort is wayfinder map
[#45](https://github.com/kido5217/ninfer-yarn/issues/45) (design
[#47](https://github.com/kido5217/ninfer-yarn/issues/47), injection
[#49](https://github.com/kido5217/ninfer-yarn/issues/49), fill-cost
[#57](https://github.com/kido5217/ninfer-yarn/issues/57)).

## The budget

The served configuration decodes at ≈ 250 tok/s, i.e. 4 ms/token; the mask work allowed per step
is the 10 % line, **0.4 ms/step**. Device rows are a Program-owned persistent buffer of 128 rows ×
31,012 B (column-major over the widest decode layout, `[16, 8]`; MTP's `[6, 8]` slices it), each a
dense bitmask over the exact 248,077-token domain;
inactive rows are all-ones and constrained rows are refreshed per round
([#47](https://github.com/kido5217/ninfer-yarn/issues/47),
[#49](https://github.com/kido5217/ninfer-yarn/issues/49)).

## The current producer (module v1)

`CompiledGrammar::row_for` produces a row on a state's first visit by running the vendored engine's
full-vocabulary walk (`llama_grammar_apply_impl`) over all 248,077 candidates, caching the row by
counter-normalized state key with content dedup, shared by grammar identity. Cached lookups are
sub-microsecond; everything rests on the fill cost. The acceptance fixtures are the G1–G4 walks
(the diary JSON schema from upstream `Neroued/ninfer` #33's subset, and the mask-cost spike's
G1–G3 grammars).

## Measured (2026-10-01, real vocabulary, lowest-allowed-token walks)

| Fixture | steps | fills | distinct masks | first-walk time | ms/step |
|---|---|---|---|---|---|
| G1 literal alternation | 4 | 5 | 5 | 84 ms | 21.0 |
| G2 `[^\n]{3,240}` | 240 | 134 | 94 | 3.55 s | 14.8 |
| G3 key=value record | 20 000 (cap) | 23 | 16 | 0.45 s | 0.02 |
| G4 diary schema | 4 226 | 1 957 | 409 | 67.1 s | **15.9** |

Against the 1.69 s budget (0.4 ms × 4 226), the diary first walk is 40× over. Phase attribution
over that walk: the engine walk is **98.5 %** of the fill (34.1 ms/fill), the module's own work is
1.5 % (per fill: candidate reset 134 µs, row packing 341 µs, dedup 31 µs; state keys 0.56 µs per
call). Inside the engine walk the mask-cost spike's attribution holds (`decode_utf8` + one
allocation per candidate: 8.85–9.21 ms; reject walk 0.64–14.48 ms), so removing the decode stage
entirely still leaves ≈ 25 ms per fill.

## Why the levers cannot close the gap

- **Folding.** 1 957 fills produce only 409 distinct rows — 1 548 fills recompute content that
  exists — but the repeated states are context-dependent equivalences, not canonicalizable ones.
  The dominant shape is a nested-counter product: one group of 180 keys — a complete 12 item-state
  × 15 char-position Cartesian grid — shares one mask. The same position pair is *observable* in one
  context and not in another (item position 2 vs 3–13 differ at the walk's last step), and a
  partially consumed round is observable too, so no context-free key can fold them without returning
  a wrong row; the only provably safe context-free fold is the physical bound already used (no token
  consumes more rounds than its piece has codepoints; limit = max piece bytes + 2 = 130).
- **Precomputed codepoints.** Eliminating the per-candidate decode/alloc leaves ≈ 25 ms/fill; the
  diary floor is then ≈ 409 × 25 ms ≈ 10.2 s even with perfect (unattainable) folding.
- **Per-stack caching.** The walk's 1 957 states contain 3 888 distinct stacks — worse than
  per-state caching.
- **Retention** is required for safety (409 rows = 12.1 MiB; the state→row map grows one entry per
  visited state; both unbounded today) but buys no speed.

Evidence and harness: `research/grammar-fill-cost` (`docs/research/grammar-fill-cost.md`,
`tools/spike/grammar-fill/`), plus the mask-cost spike on `research/mask-cost-spike`.

## Decision (2026-10-01, ticket #57)

The per-state full-vocabulary walk cannot meet the serving budget on the acceptance fixtures.
Mask production escalates to **compiled/adaptive masks (xgrammar-style) behind the existing module
seam**: a vocabulary trie of sorted decoded pieces classified per grammar state, with bulk
subtree acceptance where the state is permissive and token-boundary walks only where uncertain,
cached per state like today. The route (in-house classifier over the vendored engine vs vendoring
xgrammar vs hybrid) and the implementation are decided by the follow-up design ticket; this
document fixes the requirements they must meet.

## Requirements for the replacement

1. **Same seam.** `CompiledGrammar` compile/`compile_from_schema`, `GrammarState` accept/can_end/
   cheap copies, and `row_for` returning a dense `uint32` bitmask over the exact token domain,
   with EOG bits set exactly where the grammar can end. The fixed-shape device layout (128 rows,
   sized by the widest target-verify column domain), per-column MTP masks and committed-prefix
   advance from #49 are unchanged.
2. **Budget, measured not projected.** The G1–G4 fixture walks — the diary's 4 226-step walk above
   all — must fit 0.4 ms/step (first walk included) on this host and vocabulary.
3. **Bounded retention.** The row and state map must have an explicit bound and eviction policy,
   with the row view lifetime contract stated.
4. **Contract preserved.** The `grammar` request field stays the GBNF dialect pinned by
   [#48](https://github.com/kido5217/ninfer-yarn/issues/48) unless the owner migrates it; the JSON
   Schema path and fail-closed taxonomy are unchanged.
5. **Host-only and single-threaded** on the serving worker unless explicitly re-decided.
6. **Correctness bar unchanged.** The differential oracle (every vocabulary token replayed through
   the engine's validate/accept at the states the producer emitted rows for) plus the ported
   vectors; a producer swap moves vectors only deliberately.

The replacement landed and was measured: the token-trie producer
([#79](https://github.com/kido5217/ninfer-yarn/issues/79), PR
[#81](https://github.com/kido5217/ninfer-yarn/pull/81)) passed the end-to-end matrix
([#53](https://github.com/kido5217/ninfer-yarn/issues/53)) and the cost measurement
([#54](https://github.com/kido5217/ninfer-yarn/issues/54)), and serve enforcement shipped
([#52](https://github.com/kido5217/ninfer-yarn/issues/52), PR
[#85](https://github.com/kido5217/ninfer-yarn/pull/85)). Constrained decoding is enabled in
serving; the requirements above remain the contract for any future producer swap.
