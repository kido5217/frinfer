# Compiled-mask producer — in-house token-trie mask production (design #59)

- **Date:** 2026-10-02
- **Branch:** `research/compiled-mask-producer` (throwaway; no PR, no product change —
  design verification evidence only)
- **Ticket:** [Design: compiled-mask producer (xgrammar-style escalation) — #59](https://github.com/kido5217/ninfer-yarn/issues/59)
- **Companions:** [grammar fill-cost evidence](grammar-fill-cost.md)
  (`research/grammar-fill-cost` @ e7fb4ec8), the mask-cost spike
  (`research/mask-cost-spike` @ 821381cc), harness at
  [`tools/spike/compiled-mask/`](../../tools/spike/compiled-mask/README.md).

## 1. Question

Ticket #57 escalated: the module's per-state full-vocabulary walk (14.8–15.9 ms/step on
G2/diary, 2.9×–6.5× the 4 ms/token decode budget; 98.5 % of a fill is the engine walk)
cannot hold the 0.4 ms/step line, and the xgrammar vendor route was rejected
(dialect mismatch with the vendored engine + cost; recorded on #59). #59 asks for the
sanctioned replacement: **xgrammar-style compiled masks produced in-house** — precomputed
per-state masks whose per-step cost is a bit test, not a full walk.

This doc records the verified design: an in-house **token-trie producer** walking NInfer's
vendored `llama-grammar` engine, with three cost layers, verified differentially against
the engine's own row walk on the real 248,077-token vocabulary.

## 2. Design

The producer owns a **token trie** over the vocabulary's pieces (248,077 tokens; 247,130
registered pieces; row = 31,012 B). A *fill* (one state → one mask row) is:

1. **Class-counter fast path** (stack-based counters — the diary-class string/line
   counters). The state's counter chain is recognized (`recognize_chain`); a per
   (body, C-placement) **counter table** is looked up or built:
   - **body thresholds** — a DFS over the chain region computes, per token, the minimum
     rounds its piece consumes (`min_k_region_` / `min_k_boundary_`); bits land in
     bucket `b` = "allowed with ≤ b rounds remaining", prefix-OR'd downward;
   - **continuation correction** — tokens that hand off to the continuation `C` below the
     chain get the C-acceptance boundary re-scanned (trie descent with a built child
     index; `O(log degree)`);
   - **partial thresholds** — per pending-UTF-8 piece, the round threshold for a
     mid-codepoint tail.
   A fill is then a memcpy of the bucket `min(p, table_max)` + a threshold compare per
   partial (no descent). Diary: 4136/4227 fills.
2. **Exact trie fallback** — everything else (bulk subtree acceptance with a capacity
   witness, per-node descent, negated-class exclusion re-judgment, the 781-piece partial
   tail). Token terminals inside a counter frame body abort the table build by design
   (G3-class); root counters (G2-class) are structurally outside the fast path (no frame
   ref in the stack — §4).
3. **Producer-owned state-shape row cache** — the row is a *pure function* of
   (root-position content, can_end, pending partial, ablation switches) — verified
   (§3.3). A 128-bit content hash (FNV×2) keys a row store; the row (31 KB) is copied
   only for **repeated** shapes (first occurrences record the 16-byte signature only).
   Bounds: 8192 seen signatures (128 KB), 1024 rows (≤ 31 MB). `TRIE_SHAPE_CACHE_OFF=1`
   disables it.

## 3. Verification

### 3.1 Correctness (fresh differential oracle)

Producer shape cache **off** (a cached row would mask later-fill bugs), every row
memcmp'd against the vendored engine's `row_for(state)` — 20k-step walks, lowest mode,
host: Ryzen 7 5800X3D, GCC -O3:

| Fixture | States | Mismatches | Fast fills | Tables |
|---|---|---|---|---|
| G1 (literal alternation) | 5 | **0** | 0 (no counters by shape) | 0 |
| G2 (`[^\n]{3,240}`) | 241 | **0** | 0 (root counter) | 0 |
| G3 (key/value record) | 20,000 | **0** | 0 (build aborts by design) | 0 |
| Diary (JSON Schema) | 4,227 (walk completes) | **0** | 4136 / 90 fallback | 17, 59.45 MiB |

`results/p1-*.txt`. The `--no-fast` mode (fast-path machinery never executes) is exact on
the same walks — the two paths agree state-for-state.

### 3.2 Cost (producer `produce()` over all fills, shape cache on)

| Fixture | Before cache | After cache | Bar (0.4 ms/step) |
|---|---|---|---|
| G1 | 2.28 ms mean | **p50 0.59 ms** (10 ms one-time class structure) | at the line (fallback; no counters by shape) |
| G2 | 3.96 ms mean | **0.085 ms** (235/241 shape hits) | pass |
| G3 | 2.34 ms mean | **0.005 ms** (19975/20000 hits) | pass |
| Diary | 0.28 ms steady | **0.261 ms steady / p50 0.125 ms** | pass |

One-time work excluded from the steady-state line: 17 counter-table builds, 929 ms
(59.45 MiB retained) on the diary; 10 ms class-structure build on G1. `results/p2b-*.txt`.

**Metric correction:** the probe's fill percentiles were previously gated on the
*module's* row-cache fills — the "G3 0.0036 ms/step" reported to #57 was module fill
time ÷ states, not the producer. The gate is removed; all producer numbers here cover
every `produce()` call. G3's true pre-cache producer cost was 2.34 ms/step (781
`judge_linear` tail calls per fill).

### 3.3 Engine semantics found (G2, `root ::= [^\n]{3,240}`)

Verified against fresh no-cache engine applies (`GRAMMAR_ROW_NOCACHE`,
`results/g2-nocache.txt`: 240 states, 0 mismatches) and the 400-step termination walk
(`results/g2-400.txt`: terminates at 241 states):

- **Counter frames are not in the stack.** The engine keeps a single self-referential
  rule with internal repeat state; the state's stack is invariant (`[CNOT10]`, one
  element) across all counter positions. Diary-class counters (inside objects) are
  stack-based chains of distinct frame rules — the position is stack-visible there.
- **The counter counts pieces (tokens), not characters.** A 128-codepoint piece is
  allowed at k=200 (fresh applies: row identical across k); per-piece budget is
  unenforced — bounded overrun ≤ 127 chars past max. *Spec-deviation note for the
  vendored engine; out of scope here.* The producer reproduces these exact semantics.
- **min** is enforced via an ε/EOG empty-stack marker appearing at k≥3 (stack-visible →
  EOG bits); **max** by termination at 240 pieces.
- **The module's row cache is sound:** the row is a pure function of the state, so the
  lossy key (canonical words collapsing content-identical positions) only ever serves
  identical rows (fresh no-cache applies confirm). This is the property that licenses
  the producer's own shape cache (§2.3).
- G2 root counters are structurally outside the fast path (no frame ref in the stack) —
  covered by the exact fallback + shape cache.

### 3.4 Three correctness fixes (evidenced)

On the class-counter fast path (each took a measurable mismatch class to zero):

1. **`build_child_index()` never called** — the correction scan's mid-piece descent died
   at depth 1 (empty child index → `kNoEdge`); handoff thresholds collapsed to 0xFF
   (4124 diary mismatches).
2. **Threshold clamping** — out-of-range tokens were clamped into the top bucket instead
   of skipped (over-allow at `p == table_max`).
3. **`KSet::empty_k` default `0` → `0xFF` sentinel** — the `threshold_visit` min-merge
   `if (frag.empty_k < next.empty_k)` can never fire from 0, so every boundary-carrying
   KSet carried `empty_k=0` → `min_k_boundary_[node]=0` → `body_thr=0` for many tokens →
   over-allow at every `p` below the piece's true need. Took 20-step diary mismatches
   8 → 0.

## 4. Known limitations (design notes)

- **G3 table-build aborts** are by design (literal tokens inside the counter frame body;
  the class table deliberately doesn't model them — exact fallback + shape cache close
  the cost: 0.005 ms/step).
- **Chain undercount** — the chain walk breaks at the chain shape-change point (~6 frames
  in the diary string chain); `p` and body thresholds shift by the same delta, so the
  relative comparison stays consistent (benign as-is; a port should count frames from
  the normalized structure, not the shape walk).
- **G1** has no counters by shape (literal alternation) — exact fallback; its 10 ms
  class-structure build is one-time (compile-time placement candidate).
- The shape cache is useless for unique-per-step walks (diary: frame-ref rule ids change
  per step — 1 hit in 4225; it pays signature-only there, no row copies).

## 5. Open decisions (posted on #59, 2026-10-02)

- **D1 — route:** confirm the in-house trie producer (evidence above; xgrammar vendor
  port remains rejected).
- **D2 — cost-closure:** steady-state passes the bar on all fixtures; one-time work
  (929 ms / 59.4 MiB diary builds; 10 ms G1 class structure) → compile-time placement
  (D2a) so serving walks are pure steady-state — vs first-walk-in-serving.
- **D3 — state key + bounds:** 128-bit content hash over (root-position content, can_end,
  pending partial, switches); 1024 rows ≤ 31 MB + 8192 signatures; counter tables per
  (body, C-placement) (17 keys / 59.4 MiB on the diary).
- **D4 — validation + rollout:** differential oracle on the G1–G4 fixture walks
  (producer cache off for differential runs, on for cost runs); rollout behind the
  existing constraint seam with the #52 guards accompanying the enablement.

Next: the production port ([#79](https://github.com/kido5217/ninfer-yarn/issues/79)),
then the #53 verification matrix and #54 measurement.
