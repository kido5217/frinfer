# Logprobs evidence gate — OpenAI shape, Engine gather design, oracle

Wayfinder map #175, ticket #176 — evidence gate for the wave-1 port
[Implement: logprobs/top_logprobs (OpenAI spec shape)](https://github.com/kido5217/frinfer/issues/181).
Research branch `research/logprobs-evidence`. **No product code**: the deliverables are the spec
shape, the Engine gather design, and an independent oracle at representative shapes.

## Scope and method

Question (ticket #176), verbatim deliverables:

1. The exact OpenAI spec shape for `logprobs`/`top_logprobs` on the chat + Responses routes
   (request `bool` + `0..20` top-k; per-choice arrays, per-token entries, finish entry) — grounded
   in the official spec, **not** in llama.cpp's self-declared non-compatible shape.
2. A design for the bounded Engine top-k gather over sampler logits: where in the decode loop it
   sits, its cost model, interaction with the grammar mask + MTP/DFlash rounds.
3. The oracle: independent FP32/FP64 recomputation of per-token logprobs from final logits through
   the applied temperature/penalty pipeline at representative shapes.

Sources of record (all read this session, 2026-10-04):

- **OpenAI OpenAPI spec**, `openai/openai-openapi` @ `13fa6e7ab9301b00c03af2a5d2f584e7a9b84391`
  (default branch `main`, committed 2026-10-04T08:06:55Z; `openapi.yaml`, 114 942 lines, sha256
  `c5c8ee5977010a424b2bd06a42ee76913f3ac4cc5ee4defdfde429ebbd3860b9`). Cited as
  `<schema>:<line>` against that file. The `master` ref serves the identical blob.
- **FrInfer** master @ `014e985d` (the checkout at gate start): `src/runtime/engine/engine_core.h`,
  `src/models/qwen3_5/program/**`, `src/models/qwen3_5/execution/**` (cited as `execution/…`),
  `include/ninfer/ops/sampling.h`, `include/ninfer/ops/apply_mask.h`,
  `include/ninfer/ops/target_logprobs.h`, `include/ninfer/types.h`, `include/ninfer/engine.h`,
  `src/serve/**`, `bench/**`, `tests/**`. Cited as `<path>:<line>`.
- The survey `docs/research/llmcpp-port-serve.md` @ `research/llmcpp-serve` `dc60901a` (candidate 2),
  the gate's source of record for the llama.cpp state.

Machine: RTX 5090 (sm_120a), free at gate start (484 MiB used of 32 607 MiB).

---

## 1. OpenAI spec shape (grounded)

### 1.1 Chat Completions — `POST /v1/chat/completions`

Request (`CreateChatCompletionRequest`):

- `logprobs`: `boolean | null`, default `false` — "Whether to return log probabilities of the
  output tokens or not. If true, returns the log probabilities of each output token returned in the
  `content` of `message`." (`CreateChatCompletionRequest:43859`).
- `top_logprobs`: `integer | null`, `minimum: 0`, `maximum: 20` — "the maximum number of most
  likely tokens to return at each token position … `logprobs` must be set to `true` if this
  parameter is used." (`:43714`).

Response — each **choice** carries `logprobs` (sibling of `message`), `null` or
(`:44045`):

- `content`: `ChatCompletionTokenLogprob[] | null` — "A list of message content tokens with log
  probability information."
- `refusal`: `ChatCompletionTokenLogprob[] | null` — same for refusal tokens.
- both keys required when the object is present.

`ChatCompletionTokenLogprob` (`:42378`), all keys required:

| key | type | meaning |
|---|---|---|
| `token` | string | the token |
| `logprob` | number | its log probability; **`-9999.0` is the sentinel for "not within the top 20 most likely"** |
| `bytes` | `int[] \| null` | UTF-8 byte representation, null if none |
| `top_logprobs` | `{token, logprob, bytes}[]` | the most likely tokens at this position; "the number of entries may be fewer than the requested `top_logprobs`" |

Streaming (`CreateChatCompletionStreamResponse:44181`): every choice chunk carries the same
nullable `logprobs` object (`content`/`refusal` arrays of `ChatCompletionTokenLogprob`); the final
chunk has `"logprobs": null` with `finish_reason` (`:2924`).

### 1.2 Responses — `POST /v1/responses`

Request:

- `top_logprobs`: `integer`, `0..20` (`CreateModelResponseProperties:46333`) — same wording as chat.
- opt-in via `include: ["message.output_text.logprobs"]` (`IncludeEnum:80805`), "Include logprobs
  with assistant messages."

Aggregate output: `OutputMessage.content[]` → `OutputTextContent` (`OutputMessage:56861`,
`OutputMessageContent:56912`, `OutputTextContent:81031`) which **requires** `logprobs`:

- `LogProb[]` (`LogProb:81009`, required `token`, `logprob`, `bytes`, `top_logprobs`):
  `{token, logprob, bytes, top_logprobs: TopLogProb[]}`;
- `TopLogProb` (`:80992`, required `token`, `logprob`, `bytes`).

Streaming: `response.output_text.delta` (`ResponseTextDeltaEvent:72575`) and
`response.output_text.done` (`ResponseTextDoneEvent:72632`) both **require** `logprobs`:
`ResponseLogProb[]`, where `ResponseLogProb` (`:70747`) is
`{token, logprob, top_logprobs: [{token, logprob}]}` — **no `bytes`**.

### 1.3 What the spec fixes, and what it leaves open

Fixed:

- Chat: request `logprobs` bool + `top_logprobs` 0..20; per-choice `{content, refusal}`; per-token
  `{token, logprob, bytes, top_logprobs}`; the `-9999.0` sentinel is defined against the **top 20**.
- Responses: `top_logprobs` 0..20 + `include`; aggregate `LogProb[]` (with `bytes`), streaming
  `ResponseLogProb[]` (no `bytes`).

Open (decisions the gate records in §2.2):

- The spec never states whether the reported distribution is the model's **raw** distribution or
  the **post-sampling-transform** one, nor whether `temperature`/penalties apply. It only fixes the
  top-20 sentinel rule.
- **"Finish entry"**: there is no separate finish element. `content[]` holds one entry per generated
  content token; the finish chunk carries `logprobs: null`. The chosen token's own entry lives in
  `content[]`.
- The Responses route has two logprob schemas (`LogProb` with `bytes` on the aggregate,
  `ResponseLogProb` without on the stream) — an asymmetry to mirror, not to unify.

### 1.4 llama.cpp's shape — why the spec, not it

llama.cpp at `1537a0a8b2` (the survey's pinned commit), verified live this session from
`tools/server/server-common.cpp` and `tools/server/server-task.cpp`:

- The request mapping carries an **explicit non-compatibility TODO**
  (`server-common.cpp:1429-1430`): "The response format of this option is not yet OAI-compatible";
  it maps `logprobs=true` to `n_probs = top_logprobs` (default 20) and rejects `top_logprobs`
  without `logprobs`; it also rejects `logprobs` together with tools + stream (`:1433`).
- Per-token entries (`server-task.cpp:266-287`) carry an extra `"id"` key and switch key names with
  `post_sampling_probs` (`"prob"`/`"top_probs"` instead of `"logprob"`/`"top_logprobs"`),
  diverging from `ChatCompletionTokenLogprob` (no `id`, fixed key names).
- The chat choice emits `logprobs: {content: [...]}` with **no `refusal`**
  (`server-task.cpp:434-440`), while the spec requires both `content` and `refusal`.
- Its Responses output text hardcodes `"logprobs": []` (`server-task.cpp:558`) — llama.cpp does not
  populate Responses logprobs at all, so parity there would be strictly worse than the spec.

FrInfer therefore implements the OpenAI spec shape of §1.1/§1.2, not llama.cpp's.

---

## 2. Engine gather design

### 2.1 Where the logits live (from the code map)

- Round loop `EngineCore::worker_loop` (`engine_core.h:1977`) → `run_decode_round` (`:1845`) →
  `Program::decode` (`program.cpp:432`) → `ProgramImpl::decode` (`commit.cpp:154`) →
  `decode_raw` (`decode.cpp:786`).
- Ordinary body (`decode.cpp:26`): forward writes `ordinary.logits` BF16 `[vocab, batch_capacity]`
  (`round_buffers.cpp:67-70`), sliced `[vocab, batch]` at `decode.cpp:50`; then
  `ops::apply_mask` (`:57-64`) writes `-inf` **in place** for disallowed tokens; then
  `ops::sample` (`:65-68`); then one D2H of `OrdinaryDecodeEgress` (`:69-71`) and a host sync read
  of the token id (`:372`).
- Sampler pipeline (`include/ninfer/ops/sampling.h:50-66`):
  `adjusted_v = logits_v − presence·(c_v>0) − frequency·c_v`; greedy rows take `argmax` and skip
  filters/RNG; stochastic rows sort by adjusted descending, then `top_k` (≤20), `min_p`, `top_p`,
  softmax `exp(adjusted/T − max)`, draw. **The softmax temperature is applied inside the sampler**;
  penalties are applied at read time from `configs[b].token_counts` (a device I32 count array), and
  the selected token atomically increments it.
- The logits region is **post-mask, pre-softmax**; the softmax is internal to the sampler and not
  retained. There is no host-visible post-softmax logits tensor.
- Speculative rounds: MTP `mtp_decode_batch_body` (`speculative/mtp.cpp:70`), DFlash
  `dflash_decode_batch_body` (`execution/draft.cpp:565`). Both hold `target_logits` BF16
  `[vocab, k+1, B]` (`round_buffers.cpp:179-181`, `:238-240`), one column per verify position.
  Acceptance (`target_verify_accept`, `speculative/target_verification.cpp:8-45`) consumes a prefix
  of those columns and licenses `A+1` tokens; rejected columns are not selected and are overwritten
  next round. The grammar mask is applied **inside the shared text forward** over `width` columns
  (`execution/text.cpp:776`), so `target_logits` is post-mask for every verify column; the
  constraint machinery plans multi-column masks (`refresh_mask_rows`, `decode.cpp:863-894`;
  `constraint.plan_round(drafts)`).
- Prefill samples one row (`prefill.cpp:165` applies the mask, then sample), so the first generated
  token's logprob comes from the prefill row.

### 2.2 The reported distribution — recorded decision

For every committed token, report the log-probability under the **full-vocabulary, post-mask,
penalty-adjusted, temperature-scaled** distribution — equivalently the sampler's candidate weights
**before** `top_k`/`min_p`/`top_p` truncation and renormalization:

```
scaled_v  = ( l_v − presence_penalty·(c_v>0) − frequency_penalty·c_v ) / T
lse       = logsumexp_{v in [0,token_domain)} scaled_v
logprob_v = scaled_v − lse
```

with `T = temperature` for stochastic rows and `T = 1` for greedy rows (`temperature<=0`, where the
sampler applies no scaling). `c_v` is the token-count array **read before** `sample` increments it,
so it is the penalty state at that exact position.

Rationale: this is the "log probability the model assigns to producing this token" that the spec's
prose describes; it is deterministic and independent of the seed and the truncation knobs, so two
clients with the same request but different `top_p` see the same logprobs and a debugging client
sees the model's own confidence. It is a deliberate divergence from llama.cpp's
`post_sampling_probs` (which can report the post-truncation, renormalized distribution); llama.cpp's
own TODO marks its shape as non-spec-compatible, and the ticket pins the spec as the source.

Sentinel: with the sampler's runtime contract `top_k ∈ [1,20]`, the drawn token is always among the
top 20 by adjusted logit (temperature > 0 preserves that order), so `-9999.0` is effectively
unreachable for the chosen token; the Op/response builder still applies the rule faithfully.

### 2.3 The Op

A new read-only Op, `logprob_topk` (name tentative):

```
inputs : logits     BF16 contiguous [V, rows]   (post-mask rows)
         configs    SamplingConfig per row      (temperature, penalties, token_counts ptr)
         token_domain = V
         active     device I32 flag             (the mask_active pattern)
outputs: top_ids    I32   [K, rows]   K = 20
         top_values FP32  [K, rows]   logprobs, descending, lower id breaks ties
         lse        FP32  [rows]      (diagnostic / chosen-token lookup)
```

- **Parallelism (measured, §4)**: a **split-CTA** layout — P CTAs per row each producing a partial
  (local max, local sumexp, local top-K), then one combine CTA per row merging the partials into the
  final top-20 (`O(V·K)` worst case, K=20; no full sort). The obvious sketch — one CTA per row with
  a block reduction — is **~7× slower** (≈336 µs vs ≈46 µs at rows=1, V=151 936): with one
  256-thread CTA per row only 8 warps touch one SM, so the vocabulary-wide reduction is
  memory-latency bound. **This is a gate finding, not a benchmark artefact**: §2.3's original
  "one CTA per row" sketch under-specified the parallelism a 151 936-wide reduction needs.
- Purely a view of the sampler's distribution — it does not sample, mutate counts, or read RNG, so
  it is safe to run on any backend and in any order relative to `sample`, **provided** it is
  enqueued *before* `sample` when penalties are active (so `token_counts` is pre-increment).
- The chosen token's own entry is derived from the top-K list (guaranteed membership per §2.2);
  no post-sample lookup Op is needed.
- **Relation to the existing `target_logprobs` Op.** `include/ninfer/ops/target_logprobs.h:44`
  already computes `logits[target_ids] − logsumexp(valid rows)` — exactly the chosen token's
  log-probability — but only for caller-supplied target ids and with no top-K. It is the
  CausalScoring primitive and remains so; `logprob_topk` subsumes that value for this path (the
  chosen token is in the reported top-K), so reusing `target_logprobs` would add a second
  full-vocabulary pass.

### 2.4 Placement per backend

- **Ordinary**: enqueue after `ops::apply_mask` (`decode.cpp:57-64`) and before `ops::sample`
  (`:65`), over the one column, guarded by `active`.
- **MTP / DFlash**: enqueue after the shared forward, over all `k+1` columns of `target_logits`
  (`[V, k+1, B]`), and **before** the acceptance Op that consumes-and-increments the count array for
  those columns (`speculative/target_verification.cpp:16-40`) — not "immediately after", which
  would read post-increment penalties. The Engine publishes only the licensed prefix (`A+1`,
  `engine_core.h:1131-1181`); rejected columns are discarded. Anchor column 0 predicts the first
  token committed in this round and is published; the previous round's tokens are not re-reported.
  **Open (see §5):** if the sampler's penalty state evolves *per column within* a round, one
  pre-acceptance gather reproduces only the round-start counts and #181 must snapshot per column.
- **Prefill**: enqueue after the prefill mask (`prefill.cpp:165`) for the single row so the first
  generated token carries a logprob.

**Publication rule (pinned against `commit_pending`).** The Engine commits
`pending.tokens()[row·row_stride .. +count)` — `count` is the licensed extent: 1 for an ordinary or
prefill round, `A+1` for a speculative round (`engine_core.h:1131-1138`) — and then accepts only
`decision.accepted_tokens ≤ count` via `OutputPolicy::preview_model` (`:1150-1167`), inserting the
prefix at `:1170-1182`. A logprob record must therefore be built for the licensed tokens in the
same order and **truncated in lockstep with `decision.accepted_tokens`**; it cannot be
reconstructed at the response layer from emitted text. The chunked-round path
(`pending.row_stride() > count`) must clip to the licensed extent.

Every backend shares one Op; only the tensor view and the published range differ.

### 2.5 Cost model

Per row the work is `O(V)` for `logsumexp` plus `O(V·K)` for the bounded top-K (`K = 20`) — ≈3 M
comparisons per row at `V = 151 936`. The Op runs **only when requested**: the device `active` flag
gates the body, so an ordinary round with logprobs off pays one flag check.

**Measured (prototype, §4):** V = 151 936, standalone launch, 5-run medians (min–max): rows=1
≈46–48 µs (≈0.34 % of the 13.7 ms step), rows=8 ≈80–85 µs (≈0.6 %), masked rows ≈24 µs (r1) /
≈48–56 µs (r8), a speculative block (rows=8, `k+1=5` → 40 row-instances) ≈228 µs (1.66 %). The
1-row figure is launch-dominated — a real integration runs inside the captured decode graph, so
these are upper bounds. Cost scales with `k+1` on the speculative backends. rows=8 timings are noisy
(spread up to ≈46 %, §4.3), so treat them as order-of-magnitude.

### 2.6 Buffers, capture, and host transport

- Add a RoundState region per backend (ordinary + spec): `top_ids` I32 `[20, rows]`, `top_values`
  FP32 `[20, rows]`, `lse` FP32 `[rows]`, `active` I32 `[1]`.
- Allocate/bind **before** `prepare_graphs` (`graphs.cpp:106`) so the captured graph records stable
  addresses (the capture rule in §2.1 of the code map; `PinnedHostBuffer`, `arena.h:93-109`).
- D2H into a pinned host egress alongside the existing copy (the `ordinary_host` pattern,
  `program_impl.cpp:60-64`); the Engine reads it after its round sync.

### 2.7 Surfacing through Engine → serve

No per-committed-token host structure exists today: `GenerationResult` exposes only
`generated_token_ids` (`types.h:829-846`), `OutputDelta` carries text only (`types.h:597-604`),
and serve's `GenerationOutcome` exposes text/reasoning only (`generation_service.h:47-59`).

- Engine: add a per-committed-token logprob channel aligned with `generated_token_ids`
  (one record per committed token: id, logprob, top-K), surfaced on `GenerationResult` and, for
  streaming, a new field on `OutputDelta`/`OutputSink` or an accompanying per-token record.
  The channel carries the **raw token bytes** (for `bytes`), not the detokenized text.
- Serve chat: replace the hardcoded `{"logprobs", nullptr}` in `make_chat_completion_response`
  (`openai_chat_response.cpp:243`) and the streaming `stream_choice` (`:183-188`); both currently
  emit null. `content[]` gets one entry per **content** token, `refusal[]` per refusal token.
- Serve Responses: `build_response` output-text part (`openai_responses_response.cpp:124-126`,
  with the hardcoded `{"top_logprobs", 0}` at `:84`), and the streaming events
  (`content_delta:392`, `output_text.done:307-311`) which already emit `"logprobs": []`.
- Request gates to relax: chat `logprobs=true` / nonzero `top_logprobs`
  (`openai_chat_request.cpp:116-133`); Responses nonzero `top_logprobs`
  (`openai_responses_request.cpp:1221-1229`) and the non-empty `include` rejection (`:1190-1197`).
- `docs/serving.md` (the capability-rejection list, `:132-146`) and the schema tests
  (`tests/test_openai_schema.cpp:156-170,1034,1071`; `tests/test_openai_responses.cpp:197`) update
  together in #181 — the advertised protocol is an external contract.

### 2.8 Alignment risk (flagged)

The serve emits text through the output-session preview/commit machinery
(`output_session.cpp:458,687`), which may merge or split tokens across the content, reasoning, and
tool-call channels. Logprob entries are per committed token; the spec's `content[]` is per content
token. Recorded behaviour: logprobs attach to the **content** stream only — reasoning tokens and
tool-call tokens get no `content` entry (they are not content), and any content-token merging in
the preview path must keep the entry list aligned by token, not by emitted text delta. The exact
alignment is an implementation concern for #181, flagged here so the schema tests cover it.

---

## 3. Oracle and harness

### 3.1 Oracle definition

An **independent** FP64 recomputation from the represented BF16 inputs — every bit pattern decoded
as BF16, never reusing the kernel's FP32 arithmetic — of the §2.2 formula and the top-20 selection
(descending by scaled logit, lower token id breaking ties). Implemented in Python/NumPy `float64`
from the raw BF16 byte files the prototype consumed: a different language and implementation, so
agreement is evidence, not a tautology.

### 3.2 Harness (throwaway, `tools/spike/logprobs-evidence/`)

- `gather_prototype.cu` — standalone CUDA prototype of `logprob_topk` plus a driver that writes
  inputs (BF16 logits, per-row configs, pre-position count arrays) and outputs (ids, values, lse)
  to files, and a `--bench` mode reporting per-call time.
- `oracle.py` — the FP64 oracle; asserts the top-20 ordering and compares `logprob` values
  (max abs error, max relative error on the chosen token).
- `build.sh` / `run.sh` / `results/` — following the `research/grammar-fill-cost` spike convention
  (`tools/spike/<name>/build.sh`, linking the configured `build/` tree).
- Shapes: `V = 151 936`; rows ∈ {1..8}; speculative block width `k+1 ∈ {2..6}`; temperature
  ∈ {greedy, 1.0, 0.7}; a masked case (a contiguous 90 % of the vocab set to `-inf`); a
  penalty case (nonzero presence/frequency with a populated count array). These are the shape
  classes the scripted request matrix produces.

### 3.3 Scripted request matrix — scope note

The matrix classes are **plain, tool-call, constrained (grammar), thinking**. Because the gate adds
no product code, the matrix is exercised at the *shape* level in the harness (short/long rows,
masked rows, mixed greedy/stochastic rows, multi-column speculative blocks). End-to-end replay on
the live serve becomes possible only once #181 emits logprobs; that comparison is recorded as the
acceptance test for #181, not re-run here.

---

## 4. Verification results

Harness: `tools/spike/logprobs-evidence/`; `run.sh` builds with `nvcc -O3 -std=c++17 -arch=sm_120a`,
generates each case, runs the FP64 oracle, benchmarks, and writes `results/summary.md` (raw per-case
files under `results/data/<case>/`). The oracle is pure NumPy `float64` with hand-decoded BF16 — an
independent implementation. `oracle.py` exits non-zero on any mismatch.

### 4.1 Correctness — kernel FP32 vs independent FP64 oracle

All 13 case/row lines report `top-20 exact = yes` and `chosen ok = yes`; **overall PASS**.

| case | rows | max-abs err | max-rel err | lse max-abs |
|---|---:|---:|---:|---:|
| plain-t1 | 1 / 8 | 1.5e-7 / 6.4e-7 | 4.2e-8 / 1.6e-7 | 1.5e-7 / 6.4e-7 |
| plain-t07 | 1 / 8 | 1.1e-6 / 1.6e-6 | 5.5e-7 / 1.3e-6 | 2.5e-7 / 8.1e-7 |
| greedy | 1 / 8 | 1.5e-7 / 6.4e-7 | 4.2e-8 / 1.6e-7 | 1.5e-7 / 6.4e-7 |
| penalty | 1 / 8 | 6.1e-7 / 8.8e-7 | 1.2e-7 / 1.8e-7 | 2.5e-7 / 4.7e-7 |
| masked | 1 / 8 | 4.0e-7 / 3.7e-7 | 1.1e-7 / 1.5e-7 | 4.0e-7 / 3.7e-7 |
| mask-penalty | 1 / 8 | 2.6e-7 / 9.7e-7 | 5.2e-8 / 2.5e-7 | 2.0e-8 / 5.5e-7 |
| spec-k4 (rows 8, w 5) | 40 | 7.0e-7 | 1.7e-7 | 7.0e-7 |

### 4.2 Cost — CUDA events, 10 warm-up + 100 iterations, V = 151 936

Five repeated `--bench` runs per case (raw: `results/reruns.txt`; `results/summary.md` is the
harness's own single run, kept for the correctness tables); figures are the **median** with
(min–max) of the per-call mean:

| case | rows | median µs (min–max) | median % of 13.7 ms |
|---|---:|---|---:|
| plain-t1 | 1 | 47.9 (46.0–56.0) | 0.350 |
| plain-t1 | 8 | 83.3 (80.6–85.3) | 0.608 |
| plain-t07 | 1 | 45.9 (45.8–48.2) | 0.335 |
| plain-t07 | 8 | 83.3 (80.7–96.1) | 0.608 |
| greedy | 1 | 46.1 (46.0–48.2) | 0.336 |
| greedy | 8 | 85.3 (80.8–105.3) | 0.622 |
| penalty | 1 | 46.0 (46.0–50.3) | 0.336 |
| penalty | 8 | 80.6 (80.5–83.0) | 0.588 |
| masked | 1 | 23.6 (23.4–25.6) | 0.172 |
| masked | 8 | 48.1 (46.0–68.3) | 0.351 |
| mask-penalty | 1 | 23.6 (23.2–25.7) | 0.172 |
| mask-penalty | 8 | 55.7 (46.0–64.5) | 0.407 |
| spec-k4 (rows 8 × w 5) | 40 | 227.5 (227.2–266.9) | 1.660 |

Masked rows are ≈2× cheaper because the harness masks a contiguous suffix, so CTAs landing entirely
in the masked region skip the sum/top-K pass.

### 4.3 Caveats and scope of the measurement

- **Design correction**: the split-CTA layout of §2.3 supersedes the one-CTA-per-row sketch, which
  the spike's first implementation measured ~336 µs at rows=1 (that unsplit variant is **not** in the
  shipped harness, so the ~7× ratio is a development observation, not reproducible from this tree).
  This is the gate's most important output after the spec shape.
- **Timing reproducibility (review finding, fixed here)**: rows=8 timings vary run to run — spread up
  to ≈46 % (masked r8), with occasional outliers (greedy r8 to 105 µs) — so §4.2 reports medians and
  ranges, not single points. rows=1 is comparatively stable (≤≈21 %). No case's *worst* observation
  exceeds 1.95 % of the 13.7 ms step, so the cost conclusion is robust even at the noisy extreme.
- **Upper bound, not an in-situ figure**: the prototype times a **standalone launch**. In the real
  integration the Op is inside the captured decode graph (launch overhead removed), so the true
  added cost should be lower, especially at rows=1 where launch overhead dominates.
- **`spec-k4` is one 40-row-instance launch**, not 5 sequential launches; it models the round's whole
  verify block.
- **Chosen-token check is against the top-1.** The prototype Op does not sample (by design, §2.3), so
  the check asserts "the reported top-1 equals the oracle's argmax with the right value". #181's
  acceptance test must compare against the actually drawn token id.
- **Case ↔ request class** (shape classes, not live requests — §3.3): plain → `plain-t1`/`plain-t07`/
  `greedy`; constrained → `masked`/`mask-penalty`; speculative/tool-call → `spec-k4`; thinking → the
  long-row shapes. Live replay over the serve remains #181's acceptance test.
- `adjusted·(1/T)` in the kernel vs `adjusted/T` in the oracle: agreement ~1e-7, inside FP32 noise.

Harness environment note: on this NixOS host the CUDA runtime could not find `libcuda.so.1` inside
the devShell (misleading "driver version is insufficient" error); `run.sh` prepends
`/run/opengl-driver/lib` to `LD_LIBRARY_PATH`.

## 5. Open items handed to #181

- ~~The exact mapping from accepted speculative columns to published tokens~~ — pinned in §2.4
  (licensed extent + `preview_model` prefix truncation; `engine_core.h:1131-1182`). The per-backend
  column→token correspondence inside `target_verify_accept` (`speculative/target_verification.cpp:16-40`)
  still needs a per-column assertion during implementation.
- Whether `refusal[]` entries ever arise (the Engine's refusal path) and how they are populated.
- The content-stream alignment rule of §2.8, with schema tests.
- Whether the sampler's penalty state evolves per column within a speculative round (§2.4): if it
  does, the gather must snapshot `token_counts` per column rather than read round-start counts.
- `docs/serving.md` capability list + the schema tests, updated together.
