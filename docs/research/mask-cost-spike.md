# Mask-cost spike — llama.cpp GBNF per-step mask production at 248 k tokens

- **Date:** 2026-09-30
- **Branch:** `research/mask-cost-spike` (throwaway; no PR, no product change — measurement only)
- **Companion:** `docs/research/constrained-decoding-sources.md` §1.2 (mask production),
  §1.4 (spec decode), §4.2 (NInfer integration points), §4.3 (performance approach comparison).
- **Artifacts:** harness + fixtures + captured run under `tools/spike/mask-cost/`
  (`results/run-200.{json,txt}`).

## 1. Question

The design decision (adapt llama.cpp's `src/llama-grammar.{h,cpp}` runtime, MIT) is already made.
This spike answers the follow-up question before the mask strategy locks: what does **per-step mask
production** cost over this model's ~248 k-token vocabulary? llama.cpp scans the full candidate
array each time `llama_grammar_apply_impl` runs, with a sample-then-validate fast path that pays the
scan only on rejection (sources doc §1.2/§1.3). Does the measured cost argue for a naive per-step
fill port, or for xgrammar-style precomputed/adaptive masks?

## 2. Method

### 2.1 Engine under test

llama.cpp at the vendored chat-parsing baseline **`05af0d2b1398394cfa67e1918fee7feabccaa9bc`**
(read-only clone at `/tmp/opencode/llamacpp-chat-ref`), from `src/`:

| File | sha256 | Lines |
|---|---|---|
| `llama-grammar.cpp` | `97d76071a2ed1b0c4ce827f002f554b12a25593776adb525dde33ac4cb1eb1e9` | 1530 |
| `llama-grammar.h` | `db730e5aff77f96274aba4a43670f036400d89c7ed44b6e98d3ced41c7b9d193` | 194 |

Both are committed **byte-identical** as `tools/spike/mask-cost/upstream/`. Everything timed is an
engine function: `llama_grammar_init_impl`, `llama_grammar_apply_impl`, `llama_grammar_accept_token`;
the cost attribution additionally uses the exported `llama_grammar_reject_candidates_for_stack` and
`llama_grammar_get_{rules,stacks}`.

Adaptation lives only in `tools/spike/mask-cost/shim/` (no edit to the vendored files):

- `llama.h` — `llama_token` typedef and `llama_token_data(_array)` shaped as `include/llama.h`
  at the baseline (`:69`, `:230-243`);
- `llama-impl.h` — `LLAMA_LOG_{ERROR,WARN,DEBUG}` → stderr, `GGML_ASSERT`/`GGML_ABORT` → abort
  (same macro bodies as the in-tree `third_party/llama-chat/compat/ggml.h`);
- `llama-vocab.h` — `token_to_piece(id)` (from the piece dump), `is_eog(id)`, and a `tokenize()`
  stub that aborts (reachable only through `<token>` literals; the fixtures don't use them —
  sources doc §1.5);
- `llama-sampler.h` — empty (the upstream include is unused).

### 2.2 Vocabulary

`dump_vocab.py` mirrors `src/models/qwen3_5/frontend/tokenizer.cpp`: `model.vocab` ids are decoded
to raw bytes through the GPT-2 byte↔unicode alphabet (`decode_byte_level_token`), `added_tokens`
(tokenizer.json, plus `added_tokens_decoder` in tokenizer_config.json) keep their literal text.

- Source: `models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/`
  — the revision the served artifact's manifest names as its source base
  (`artifact-manifest.json` → `source.base.revision`). The `.ninfer` embeds 6 resources
  (12.8 MiB) but is not shippable through this spike, so the snapshot tokenizer (revision-pinned)
  is used.
- **N = 248 077 ids** (248 044 byte-level BPE + 33 added tokens), matching the token domain in
  the sources doc §4.2 item 8.
- **Mask row = ceil(248077/32)·4 = 31 012 bytes.**
- **EOG approximated** as `generation_config.json` `eos_token_id` = {248044, 248046}.

### 2.3 Fixture grammars and states

| # | Grammar | File | Rules | Bytes |
|---|---|---|---|---|
| G1 | literal alternation (`"да" \| "нет"`) | `grammars/g1_literal_alternation.gbnf` | 1 | 27 |
| G2 | bounded line `[^\n]{3,240}` | `grammars/g2_bounded_line.gbnf` | 238 | 22 |
| G3 | key=value record (acceptance fixture) | `grammars/g3_key_value_record.gbnf` | 11 | 393 |
| G4 | diary JSON schema → GBNF | `grammars/g4_diary_schema.gbnf` | 277 | 946 |

G4 was generated with the in-tree vendored converter (same llama.cpp baseline) via the tiny driver
`gbnf_from_schema.cpp`:

```bash
/path/gbnf_from_schema tools/spike/mask-cost/grammars/g4_diary_schema.json \
    > tools/spike/mask-cost/grammars/g4_diary_schema.gbnf
```

The schema and the exact emitted grammar are committed alongside it.

States per fixture — mid/near are reached by feeding the prefix as the grammar's own token pieces
(greedy longest-match over the piece dump; raw `accept_str` fallback for bytes with no piece):

| # | fresh | mid prefix | near prefix | continuations (fresh / mid / near) |
|---|---|---|---|---|
| G1 | — | `н` | `не` | `нет` / `ет` / `т` |
| G2 | — | 120 × `x` | 238 × `x` | `Hello world! The quick brown fox.` / same / `xy` |
| G3 | — | `act=base;base=forward;base_amount=0.5;` | `act=view;say=All clear;tone=calm` | `act=base;say=Ready;tone=calm` / `say=Ready;tone=calm` / ` for now` |
| G4 | — | `{"title": "Day one", "text": "The park` | full JSON minus closing `}` | minimal doc / ` was quiet"` / `}` |

### 2.4 Protocol

- Single-threaded; `std::chrono::steady_clock` (monotonic); warmups discarded; median/min/max/mean.
- Iterations (warmup): init 200 (10); full fill and attribution 200 (5); validate fast path and
  mask-row floor 1000 (50); accept 200 per state (20).
- Inputs reset **outside** the timed region:
  - fill: memcpy of a pristine candidate array (ids `0..N-1`, `logit=0`, `p=0`) over the working
    array — one 2.98 MiB memcpy per iteration, untimed;
  - validate: rebuild the one-element `{id, 1.0f, 0.0f}` array;
  - accept: per-iteration `llama_grammar_clone_impl` of the state, clone excluded from timing;
  - init: free of the previous grammar outside the timed region.
- The fast-path validate mirrors `common/sampling.cpp:648-655` exactly: one-element
  `llama_token_data_array`, `logit != -INFINITY` means accepted.
- Run (captured as `results/run-200.{json,txt}`):

```bash
nix develop -c bash tools/spike/mask-cost/build.sh /tmp/opencode/mask-cost-build
SNAP=$HOME/trash/ai/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
python3 tools/spike/mask-cost/dump_vocab.py "$SNAP/tokenizer.json" \
    --tokenizer-config "$SNAP/tokenizer_config.json" --out /tmp/opencode/mask-cost-build/vocab.tsv
/tmp/opencode/mask-cost-build/mask-cost-harness \
    --vocab /tmp/opencode/mask-cost-build/vocab.tsv \
    --grammars tools/spike/mask-cost/grammars --eog 248044,248046 --iters 200 \
    --json tools/spike/mask-cost/results/run-200.json | tee tools/spike/mask-cost/results/run-200.txt
```

- Host: **AMD Ryzen 7 5800X3D** (8 cores/16 threads, 96 MiB L3, max 4553 MHz), 128 GiB RAM,
  NixOS; devShell GCC 15.3.0, binutils 2.46; `-O3 -DNDEBUG`; loadavg 0.71 at run start.

## 3. Results

**Budget reference:** the served configuration decodes ≈250 tok/s ⇒ 4 ms/token. The task's
strategic line is 0.4 ms/step (10%). All numbers are medians over 200–1000 iterations; the raw
min/max are in `results/run-200.json`.

### 3.1 Compile (once per request) and mask-row floor

| Grammar | rules | bytes | init median µs | init min | init max |
|---|---|---|---|---|---|
| G1 | 1 | 27 | 0.47 | 0.46 | 0.70 |
| G2 | 238 | 22 | 54.77 | 52.81 | 80.24 |
| G3 | 11 | 393 | 3.73 | 3.66 | 9.18 |
| G4 | 277 | 946 | 76.19 | 72.31 | 87.36 |
| mask row memset (31 012 B) | — | — | 0.25 | 0.24 | 5.42 |
| mask row memcpy (31 012 B) | — | — | 0.45 | 0.43 | 0.75 |

`llama_grammar_init_impl` (parse + rules + initial stacks) is negligible once per request — even
the 277-rule JSON grammar compiles in 76 µs.

### 3.2 Full-vocabulary fill (`llama_grammar_apply_impl`), per grammar and state

decode/reject columns are the instrumented attribution of §4; residual = fill − decode − reject.

| Grammar | state | stacks | allow_eog | allowed tokens | fill med ms | fill min | fill max | decode med ms | reject med ms | residual ms |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | fresh | 2 | no | 6 | 12.37 | 12.22 | 14.51 | 8.99 | 1.29 | 2.09 |
| G1 | mid | 1 | no | 3 | 11.68 | 11.58 | 15.02 | 9.21 | 0.65 | 1.82 |
| G1 | near | 1 | no | 2 | 11.66 | 11.51 | 14.94 | 9.12 | 0.65 | 1.89 |
| G2 | fresh | 1 | no | 247 709 | 17.14 | 16.92 | 18.76 | 8.86 | 5.86 | 2.42 |
| G2 | mid | 2 | yes | 247 710 | 18.48 | 18.19 | 21.02 | 9.16 | 7.17 | 2.16 |
| G2 | near | 2 | yes | 49 559 | 14.27 | 13.97 | 16.95 | 8.86 | 2.96 | 2.46 |
| G3 | fresh | 1 | no | 3 | 11.69 | 11.61 | 13.83 | 9.05 | 0.68 | 1.96 |
| G3 | mid | 14 | no | 51 | 20.36 | 20.18 | 23.38 | 8.84 | 9.42 | 2.09 |
| G3 | near | 2 | yes | 247 520 | 18.98 | 18.77 | 21.01 | 8.89 | 7.64 | 2.44 |
| G4 | fresh | 1 | no | 2 | 11.56 | 11.49 | 11.85 | 8.85 | 0.64 | 2.07 |
| G4 | mid | 3 | no | 245 466 | 25.85 | 25.22 | 28.86 | 9.08 | 14.48 | 2.28 |
| G4 | near | 3 | no | 5 | 13.78 | 13.70 | 14.39 | 9.10 | 2.78 | 1.90 |

A full fill costs **11.6–25.9 ms** — **2.9×–6.5× the entire 4 ms/token budget**. Even the
cheapest state (G4 fresh: 2 allowed tokens out of 248 k) costs 11.56 ms; a near-empty mask is not
cheaper than a near-full one, because the vocabulary-wide candidate decode (8.9 ms, §4) is paid
regardless and the rest depends on grammar shape, not on how much the grammar restricts.

### 3.3 Sample-then-validate fast path (one-element apply)

| Grammar | state | accepted token | good med µs | good min/max | rejected token med µs | bad min/max |
|---|---|---|---|---|---|---|
| G1 | fresh | `нет` (id 151443) | 0.42 | 0.40 / 8.12 | 0.23 | 0.22 / 0.46 |
| G1 | mid | `ет` (id 7921) | 0.21 | 0.20 / 5.25 | 0.14 | 0.14 / 0.20 |
| G1 | near | `т` (id 1729) | 0.12 | 0.11 / 0.18 | 0.11 | 0.10 / 0.16 |
| G2 | fresh | `Hello` (id 9419) | 1.01 | 0.99 / 1.62 | 0.14 | 0.13 / 0.19 |
| G2 | mid | `Hello` (id 9419) | 1.26 | 1.24 / 6.25 | 0.22 | 0.22 / 3.60 |
| G2 | near | `xy` (id 3994) | 0.32 | 0.31 / 5.13 | 0.23 | 0.22 / 0.30 |
| G3 | fresh | `act` (id 519) | 1.19 | 1.13 / 7.43 | 0.14 | 0.14 / 4.43 |
| G3 | mid | `say` (id 35571) | 1.56 | 1.55 / 6.06 | 1.41 | 1.40 / 4.57 |
| G3 | near | ` for` (id 364) | 1.04 | 1.03 / 1.79 | 0.23 | 0.22 / 10.25 |
| G4 | fresh | `{"` (id 4754) | 1.20 | 1.17 / 6.13 | 0.42 | 0.40 / 3.93 |
| G4 | mid | ` was` (id 557) | 3.73 | 3.68 / 7.75 | 0.75 | 0.73 / 4.60 |
| G4 | near | `}` (id 92) | 0.58 | 0.57 / 0.80 | 0.57 | 0.56 / 4.87 |

The fast path is **0.11–3.73 µs** — free next to the 4 ms budget. The catch: every *rejection*
costs a full fill (§3.2) plus a resample. Amortized overhead = P(reject) × fill, so the 0.4 ms
(10%) allowance is exhausted when more than **~1.5%–3.5%** of steps reject (worst state G4-mid at
1.5%, most states ~2–3.5%). A grammar with 2–3 allowed tokens out of 248 k (G1/G3/G4 fresh) can
reject often: the sampled token comes from the model's *unmasked* top-k, and nothing bounds how
often it lands outside the grammar's set.

### 3.4 Token accept (`llama_grammar_accept_token`)

Continuations fed token-by-token from each state, each token's accept timed separately:

| Grammar | state | tokens (piece, bytes, median µs) | range µs |
|---|---|---|---|
| G1 | fresh | `нет`(6B)=0.29 | 0.29 |
| G1 | mid | `ет`(4B)=0.19 | 0.19 |
| G1 | near | `т`(2B)=0.11 | 0.11 |
| G2 | fresh | `Hello`(5B)=0.76 ` world`(6B)=0.96 `!`(1B)=0.26 ` The`(4B)=0.69 ` quick`(6B)=0.91 ` brown`(6B)=0.91 ` fox`(4B)=0.68 `.`(1B)=0.26 | 0.26–0.96 |
| G2 | mid | same sequence from 120 × `x` | 0.27–0.95 |
| G2 | near | `xy`(2B)=0.33 | 0.33 |
| G3 | fresh | `act`(3B)=0.31 `=b`(2B)=1.07 `ase`(3B)=0.66 `;s`(2B)=1.90 `ay`(2B)=0.26 `=R`(2B)=0.46 `ead`(3B)=0.79 `y`(1B)=0.34 `;t`(2B)=0.29 `one`(3B)=0.32 `=c`(2B)=0.49 `alm`(3B)=0.62 | 0.26–1.90 |
| G3 | mid | `say`(3B)=0.76 `=R`(2B)=0.44 `ead`(3B)=0.74 `y`(1B)=0.33 `;t`(2B)=0.28 `one`(3B)=0.31 `=c`(2B)=0.44 `alm`(3B)=0.57 | 0.28–0.76 |
| G3 | near | ` for`(4B)=0.68 ` now`(4B)=0.65 | 0.65–0.68 |
| G4 | fresh | 21 tokens, minimal doc (1–6 B pieces) | 0.19–1.83 |
| G4 | mid | ` was`(4B)=1.71 ` quiet`(6B)=2.11 `"`(1B)=0.24 | 0.24–2.11 |
| G4 | near | `}`(1B)=0.22 | 0.22 |

State advance is cheap (0.11–2.11 µs); the outliers (`;s` at 1.9 µs, ` quiet` at 2.11 µs) are
states where `llama_grammar_advance_stack` explores more rule branches.

## 4. Cost attribution of the fill

The fill was split by instrumenting its two stages (harness-side mirrors, same binary; see
`harness.cpp` "Auxiliary attribution" — a line-for-line copy of the upstream decode loop and the
exported per-stack reject function with upstream's 4-line composition):

- **decode stage** (per-candidate `decode_utf8` + candidate-array construction,
  `llama-grammar.cpp:1375-1395`): **8.85–9.21 ms, identical in every state and grammar** —
  it is vocabulary-bound, not grammar-bound. `decode_utf8` allocates a
  `std::vector<uint32_t>` per piece (`:42`, called at `:1391`): 248 075 heap allocations per fill,
  ~36 ns/candidate. Codepoints per token are static; a native port precomputes them once and this
  stage disappears.
- **reject stage** (`llama_grammar_reject_candidates`, `:1396`): **0.64–14.48 ms**, scaling with
  stack count and with how many candidates survive the first codepoint check — G1 fresh (2 stacks,
  everything dies on codepoint 1): 1.29 ms; G3 mid (14 stacks): 9.42 ms; G4 mid (3 stacks,
  permissive string class, every piece walked to its end): 14.48 ms.
- **residual 1.8–2.5 ms**: outer candidate-array allocation/destruction and the logit write-back
  inside `apply_impl`, which the two stages measured apart do not cover.

Consequence: even if a native port eliminates the decode stage entirely (codepoints per token are
static and can be precomputed once), the remaining measured per-step work — reject walk plus
write-back/residual — is **≈2.5–16.8 ms** (the fill column minus the decode column), i.e.
**62%–420% of the 4 ms budget**. A naive per-step full scan cannot be rescued by constant factors;
the numbers justify a design that walks the vocabulary **per grammar state, not per step**, plus
the mask row write (0.24–0.45 µs).

## 5. What xgrammar-style masks change (qualitative, per sources doc §3.2/§4.3)

xgrammar's LLM runtime does not walk the vocabulary per step. At compile time it classifies
vocabulary tokens per reachable grammar state into accepted / rejected / **uncertain** (walking the
sorted decoded-vocabulary trie against the grammar) and caches an int32 bitmask per state; the
per-step `FillNextTokenBitmask` unions the cached masks for the current states and visits only each
mask's *uncertain* tokens, with a common-prefix walk over vocabulary order; `BatchGrammarMatcher`
multi-threads fills across requests. Per-step work is therefore a **cached-row operation plus a
small uncertain-set walk** — the measured floor for a 31 012-byte row on this host is 0.45 µs
(memcpy) / 0.25 µs (memset) — while the full-vocabulary walk happens once per distinct state and is
amortized across steps, requests, and columns. For JSON-like grammars the reachable state count is
bounded by the schema, not by sequence length, so cache hits dominate (this spike did not measure
xgrammar itself — no xgrammar build was in scope; the per-state/cached structure is taken from the
sources doc).

## 6. Recommendation

**Do not port a naive per-step full-vocab fill; adopt the precomputed/adaptive (cached per-state
mask) design, with the sampled-token check as the fast path.** Measured grounds: a full fill is
11.6–25.9 ms against a 4 ms/token budget (2.9×–6.5×), and even after removing every per-candidate
decode cost the remaining per-step work measures 2.5–16.8 ms (62%–420% of the budget); the
sampled-token validate used by llama.cpp's sampler costs 0.11–3.73 µs, so it is worth keeping as a
shortcut (or better, as an O(1) bit test against the cached state's mask), but it only amortizes
the full fill while the rejection rate stays under **~1.5–3.5%** — exactly the operating point a
restrictive grammar can leave unpredictably (and MTP rounds need a mask per draft column,
multiplying the exposure) — and each rejection then costs 12–26 ms, i.e. several tokens of
latency. The cached-mask design inverts the cost: full walks happen once per newly seen grammar
state (amortized), cached rows are applied for 0.24–0.45 µs, and the per-step mask work does not
scale with vocabulary size — the cost of that robustness is the cache machinery (per-state masks,
uncertain-set partitioning, rollback for draft verification) that the xgrammar approach already
formalizes. Concretely for NInfer: keep the research doc §4.2 mask plumbing (a program-owned device
mask row per constrained row/column, stable address for graph capture), produce rows from a
per-state cache, and instrument the rejection rate (analogous to the tool-call demotion counter) so
the fast-path/fallback split can be validated in the field; a naive full-scan producer should not
be built even as a fallback, since a single rejection is 3–7 tokens' worth of time and the fallback
is on the hot path exactly when the grammar is doing its job.

## 7. Caveats

- **Tokenizer dump is a faithful mirror, not NInfer itself.** `dump_vocab.py` reproduces the
  loader's algorithm and data (byte-level decode, added-token literals, `added_tokens_decoder`
  merge) but was not executed through NInfer; the vocab size (248 077) matches the token domain
  documented in the sources doc §4.2 item 8. The `.ninfer` artifact's embedded tokenizer resource
  was not extracted.
- **EOG set approximated** as {248044, 248046} from `generation_config.json`. NInfer's runtime
  stop-id set can be larger; the EOG branch is a two-token special case, immaterial to the
  measured costs.
- **No device interplay measured.** Nothing here covers logits transfer, mask upload, or NInfer's
  sampling kernel; the numbers are host-side GBNF work only. The 4 ms/token reference is the
  measured decode rate of the served configuration, used as the budget denominator.
- **Single-threaded, unpinned.** The host was otherwise lightly loaded (loadavg 0.71 at start);
  max values include scheduler/allocator spikes (e.g. validate-good max 8.1 µs vs 0.42 µs median).
  Medians are the decision numbers.
- **Candidate array = full 248 077-id vocabulary**, matching the llama.cpp server path (sources
  doc §1.2). It is not necessarily the shape a NInfer mask producer would use.
- **Attribution split is harness-side instrumented** (decode mirror in the same binary; reject
  composition over the engine's exported per-stack function) — totals differ from the sum by the
  residual in §3.2.
- Speculative decoding is not measured; §1.4 of the sources doc covers llama.cpp's
  sample-then-validate with checkpoint rollback, and the MTP implication is reasoned, not measured.

## Appendix — fixture texts

`g1_literal_alternation.gbnf`:

```
root ::= "да" | "нет"
```

`g2_bounded_line.gbnf`:

```
root ::= [^\n]{3,240}
```

`g3_key_value_record.gbnf`:

```
root  ::= "act=" act rest ";say=" said ";tone=" text
act   ::= "none" | "gesture" | "arm_task" | "base" | "approach" | "power" | "view" | "delegate"
rest  ::= (";" pair)*
pair  ::= key "=" text
key   ::= "presay" | "power" | "emote" | "gesture" | "delegate" | "arm_task" | "approach" | "base" | "base_amount" | "needs_view" | "recall" | "look" | "halt"
text  ::= [^;\x0A]*
said  ::= [^;\x0A]+
```

`g4_diary_schema.json`:

```json
{"type":"object","additionalProperties":false,"required":["title","text","facts","people"],"properties":{"title":{"type":"string","maxLength":60},"text":{"type":"string","maxLength":2400},"facts":{"type":"array","maxItems":14,"items":{"type":"string","maxLength":120}},"people":{"type":"array","maxItems":4,"items":{"type":"object","additionalProperties":false,"required":["name","when"],"properties":{"name":{"type":"string","maxLength":30},"when":{"type":"string","maxLength":5}}}}}}
```

`g4_diary_schema.gbnf` (17 production lines; the compiled grammar has 277 rules after GBNF
`{m,n}` expansion — full file committed):

```
root ::= "{" space title-kv "," space text-kv "," space facts-kv "," space people-kv space "}"
space ::= | " " | "\n"{1,2} [ \t]{0,20}
title ::= "\"" char{0,60} "\""
facts ::= "[" space (facts-item ("," space facts-item){0,13})? space "]"
people-item ::= "{" space people-item-name-kv "," space people-item-when-kv space "}"
...
```

**Observed engine behavior:** GBNF repetition ranges silently drop the upper bound when
`max_times > 2000` (`llama-grammar.cpp:659-661`: `max_times = UINT64_MAX` before expansion), so
G4's `text` (`maxLength: 2400`) compiles as an *unbounded* char repetition. This is unrelated to
the measured states (all prefixes are short) but matters for the design: JSON-schema `maxLength` /
`maxItems` values above 2000 are not enforced by this engine.
