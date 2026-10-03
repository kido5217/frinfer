# llama.cpp sampling & decoding surface vs NInfer — port candidate survey

Ticket: #171 (wayfinder map #167 "llama.cpp port candidates").

## Scope and method

**Question (verbatim):** Which llama.cpp sampling/decoding features are port
candidates for NInfer's Engine + serve?

**Verdict criteria (user-fixed):** rank by (a) exposed contract/protocol gap on the
three routes (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages) >
(b) measurable 5090 performance > (c) QoL/ecosystem parity. Verdicts: port / defer /
no with one-line rationale. Items already verdicted by siblings — the sampler zoo as
request fields, `logit_bias`, `logprobs` (#170), speculative acceptance (#173) — are
cross-referenced, not re-surveyed; the ENGINE-LEVEL machinery those verdicts rest on
is surveyed where this ticket's scope requires it (penalty implementations,
stop-sequence semantics, chain architecture).

**Sources (all read this session):**

- llama.cpp master at commit `1537a0a8b2f8711d840878b0a0677ab2213c882c`
  (committer date 2026-10-03T15:19:13Z, verified live via
  `gh api repos/ggml-org/llama.cpp/commits/master` on 2026-10-03). Blobless local
  clone at `/tmp/opencode/llamacpp-serve-ref` checked out at exactly that commit;
  all llama.cpp citations are `1537a0a8b2:path`. Primary sources: `common/common.h`
  (`common_params_sampling`), `common/sampling.{h,cpp}` (`common_sampler`),
  `src/llama-sampler.cpp` (all core sampler initializers + the penalties sampler),
  `common/reasoning-budget.{h,cpp}` (thinking-budget sampler), `common/arg.cpp`
  (sampling/control-vector CLI), `include/llama.h` (seed constant, cvec API),
  `tools/server/server-schema.cpp` (request field registry),
  `tools/server/server-common.cpp` (OAI chat body parse),
  `tools/server/server-context.cpp` (stop-string check), `server-task.cpp`
  (finish_reason + logprobs response shape), `server-chat.cpp` (Anthropic
  passthrough), `tools/server/README.md` (determinism notes).
- NInfer master at `3e0ff840` (worktree of the primary repo, read-only):
  `include/ninfer/types.h` (Engine request/sampling types),
  `include/ninfer/ops/sampling.h` (the fused sampling Op contract),
  `src/serve/request.h` (serve `GenerationRequest`/`SamplingParams`),
  `src/serve/openai_chat_request.cpp` + `anthropic_messages_request.cpp`
  (per-route field validation), `src/serve/translate.cpp` (stop → `StopPolicy`),
  `src/serve/serve_options.cpp` (the executable `--help` text),
  `src/models/qwen3_5/frontend/frontend.cpp` (`merge_stop_policy`) and
  `output_session.cpp` (stop + thinking-budget enforcement), `docs/serving.md`
  (authoritative surface doc).
- OpenAI API contract: `developers.openai.com` API reference (Chat Completions
  create; legacy Completions) and the official `openai/openai-python` SDK
  (`completion_create_params.py`, `responses/response_create_params.py`,
  `resources/responses/responses.py`) fetched 2026-10-03.
- Anthropic API contract: `platform.claude.com/docs/en/api/messages/create`
  (Create a Message) fetched 2026-10-03.
- Sibling findings (cross-references only): #170
  `docs/research/llmcpp-port-serve.md` @ `research/llmcpp-serve` `dc60901a`, #173
  `docs/research/llmcpp-port-perf.md` @ `research/llmcpp-perf` `0cafbf67`, #169
  `docs/research/llmcpp-port-chat-drift.md` @ `research/llmcpp-chat-drift`
  `7e49db57`.

**Ticket-term notes (verified against the tree):**

- "Mirror sampling" — **no mirror sampler exists at `1537a0a8b2`** (grep of
  `src/llama-sampler.cpp`, `common/`, `examples/`, `tools/server/` finds no
  mirror sampler; the only hit is an unrelated comment,
  `1537a0a8b2:include/llama.h:1434`). The ticket hint is stale; recorded as a dead
  end, not a candidate.
- "GenerationRequest in `include/ninfer/types.h`" — the public Engine request types
  live in `include/ninfer/types.h` (`SamplingOverrides`, `StopPolicy`,
  `ExecutionOptions`, `RequestOptions`, `GrammarConstraint`); the serve-side
  `GenerationRequest` is `src/serve/request.h:177`. Both surveyed below.
- Sibling docs are not on master; they were read from their research branches.

## NInfer sampling surface today (verified at 3e0ff840)

**Engine public API (`include/ninfer/types.h:175-282`):**

| Type | Fields |
|---|---|
| `SamplingMode` | `Thinking`, `NonThinking` (request-resolved mode) |
| `SamplingPreset` (model-registered defaults) | `temperature`, `top_k`, `top_p`, `min_p`, `presence_penalty`, `frequency_penalty` |
| `SamplingOverrides` (request side; `nullopt` = model default, explicit 0 = real override) | the same six + `seed` (`uint64`) |
| `ResolvedSamplingParameters` | the same six + `seed` |
| `StopPolicy` | `token_ids`, `strings` (`StopString{text, channel: Content/Reasoning, include_in_output}`), `include_model_defaults`, `publish_stop_token` |
| `ThinkingControlOptions` | `budget` (`uint32`; model-origin thinking-token cap; injected control tokens consume the total output budget, not this one) |
| `ExecutionOptions` | `sampling`, `requested_output_tokens`, `allow_prefix_reuse`, `thinking` |
| `OutputOptions` | `raw`, `preserve_special_tokens`, `tool_name_max_length` (128) |
| `GrammarConstraint` | `gbnf` (root symbol `root`), `thinking_enabled` (thinking-wrapper rule) |
| `RequestOptions` | `execution`, `stop`, `output`, `constraint` |

**The sampling Op (`include/ninfer/ops/sampling.h:25-78`)** is one fused device
Op, not a chain: `SamplingConfig{temperature, top_k, top_p, min_p,
presence_penalty, frequency_penalty, seed, token_counts}`; per row, logits are
adjusted by `− presence·(c_v>0) − frequency·c_v` where `c_v` is the committed
whole-generation occurrence count (device i32 array; no window, no prompt
tokens); `temperature<=0` selects exact argmax (skips filters and RNG);
otherwise candidates are sorted by adjusted logit (lower token id breaks ties),
`top_k` keeps up to 20 (contract `[1,20]`, `<=0` or `>=20` keeps the 20-cap),
weights `exp(adjusted/temp−max)`, `min_p` removes the suffix below `min_p·max`,
`top_p` keeps the shortest prefix reaching `top_p` of the pre-truncation weight,
the best candidate is always retained, support is renormalized, and one id is
drawn from a counter-based RNG keyed `(seed, logical_position, purpose)` —
stateless, no dependence on compact row index. Speculative acceptance has its
own purpose codes (`kSamplePurposeSpeculativeAccept/Correction/Bonus`,
`kSamplePurposeDFlash2Proposal`).

**Serve surface (`docs/serving.md`, `src/serve/*`, `--help` from
`src/serve/serve_options.cpp:67-117`):**

- Chat Completions (`docs/serving.md:102-147`): `temperature` `[0,2]`,
  `top_p` `[0,1]`, `top_k` `0..20` and `min_p` `0..1` (documented vLLM/SGLang-style
  extensions, `src/serve/openai_chat_request.cpp:868-871`), `presence_penalty` /
  `frequency_penalty`, integer `seed` (fresh random per request when omitted, or
  `--seed` fixed; `docs/serving.md:914`), `stop` string or ≤4-string array applied
  to **both reasoning and answer** channels
  (`src/serve/openai_chat_request.cpp:840-858`), `n:1` only,
  `repetition_penalty` accepted **only at the neutral value 1** — non-neutral
  400s because "NInfer's Engine intentionally has no such sampler"
  (`src/serve/openai_chat_request.cpp:310-319`; `docs/serving.md:229-231`).
  Fail-closed rejections: nonzero `logit_bias` (`logit_bias_not_supported`),
  requested log probabilities (`logprobs_not_supported`), audio, `strict:true`,
  required/named tool choice, etc. (`docs/serving.md:132-141`); neutral forms
  (`logit_bias` all-zero, `logprobs:false`, `top_logprobs:0`) accepted; unknown
  top-level fields ignored (`docs/serving.md:143-147`).
- Responses (`docs/serving.md:512-542`): `temperature` `[0,2]`, `top_p`;
  `top_logprobs` only `0`; no penalties/seed/stop fields; unknown fields 400
  `unknown_parameter`.
- Anthropic Messages (`docs/serving.md:784-800`;
  `src/serve/anthropic_messages_request.cpp:897-944`): `temperature` `[0,1]`,
  `top_p` `[0,1]`, `top_k` `[0,20]`, `stop_sequences` (array of non-empty strings),
  Thinking `enabled` with `budget_tokens >= 1024` and `< max_tokens`,
  `output_config.effort` passes to the template; `output_config.format` 400s
  `output_config_format_not_supported` (`:895-904`).
- Process flags: `--temperature/--top-p/--top-k/--min-p/--presence-penalty/
  --frequency-penalty/--seed/--greedy` (`docs/serving.md:908-915`).
- Model presets (`docs/serving.md:924-929`): Qwen3.6/3.8-27B
  `1.0/0.95/20/0/0` (temp/top-p/top-k/min-p/presence) thinking,
  `0.7/0.80/20/0/1.5` non-thinking; Qwen3.6-35B-A3B differs only in thinking
  presence (1.5); frequency 0 for all. Precedence: registered value → process
  flag → request field → `--greedy` forces temperature 0.
- Constrained decoding (`docs/serving.md:149-165`; memory #392): chat-route
  `grammar` (GBNF) + `response_format` `json_object`/`json_schema` (serve converts
  via the vendored `json_schema_to_grammar` under a context-accurate allowlist,
  `src/serve/constraint_contract.cpp`); exactly one constraint kind; never with a
  `tools` field; fail-closed codes `grammar_invalid`, `json_schema_invalid`,
  `json_schema_unsupported`, `constrained_decoding_conflict`,
  `constraint_too_large`, `constrained_decoding_not_supported`; MTP + ordinary
  supported, DFlash rejected. Applied by an in-tree token-trie mask producer
  (map #45; `docs/maintainer/grammar-mask-production.md`).
- Speculative: MTP + DFlash/DFlash2; acceptance is the 16-candidate sparse
  rejection-sampling Op (`include/ninfer/ops/speculative_round.h`), verified
  equivalent to llama.cpp's (#173, perf doc C9).
- Stop enforcement (Engine side, model frontend, not serve):
  `merge_stop_policy` (`src/models/qwen3_5/frontend/frontend.cpp:360-390`)
  merges model-default token stops (`tokenizer.default_stop_token_ids()`) +
  caller token ids + caller strings (deduped per text/channel/`include_in_output`,
  UTF-8-validated); `output_session.cpp` enforces per round — token stops with
  `publish_stop_token` withholding (`:523-552`), string stops with UTF-8
  reassembly and a suffix-prefix hold so a stop word split across token
  boundaries is detected and withheld bytes are dropped on match
  (`stop_hold_size`, `:181-190`, `:194-225`).
- Thinking budget (Engine side, model frontend): model-origin thinking tokens are
  counted (`output_session.cpp:516-518`); at the cap the frontend closes the
  thinking phase and injects the target control sequence, which consumes the
  output budget, not the thinking budget (`types.h:241-246`). Verified in
  production 2026-10-03 (memory #421: runaway close at 32768 with control
  tokens + streamed early-close guidance).

## Candidates

### C1 — `logprobs` / `top_logprobs` (engine-level machinery per ticket scope)

- **llama.cpp state:** `logprobs` (boolean) and `top_logprobs` request fields map
  to `n_probs` with default 20 (`1537a0a8b2:tools/server/server-common.cpp:1429-1441`);
  `n_probs` is aliased in the schema (`server-schema.cpp:175-180`); the server keeps
  the post-sampling candidate array via `common_sampler_get_candidates`
  (`common/sampling.h:104`) and emits per-choice `logprobs.content[]`
  (`server-task.cpp:391-440`, completions shape at `:1086-1150`). A TODO at
  `server-common.cpp:1430` states the response shape "is not yet OAI-compatible" —
  llama.cpp's own logprobs output is an internal shape, not the OpenAI spec shape.
- **NInfer state:** rejected today with 400 `logprobs_not_supported`
  (`3e0ff840:docs/serving.md:132-137`); no Engine output channel for token log
  probabilities; the sampling Op consumes the logits row in-kernel and emits only
  the selected id.
- **Port shape + cost:** new Engine output channel (per committed step, top-k
  gather from the logits row the sampler already reads) + serve mapping to the
  **OpenAI spec** `logprobs` shape (`content[]` with `token`/`logprob`/`bytes`,
  `top_logprobs[]`), aggregate + streaming. Cross-reference: #170 sized this
  **medium**.
- **Contract impact:** chat + Responses routes — official OpenAI slots
  (verified in the current spec) that 400 today; Responses `top_logprobs` is the
  Responses-side slot.
- **Verdict:** **port** (a) — cross-reference #170 (unverdicted here); note for the
  implementer: port to the OpenAI spec shape, not to llama.cpp's
  self-declared non-OAI-compatible shape.

### C2 — `logit_bias` (engine-level machinery per ticket scope)

- **llama.cpp state:** `logit_bias` request field accepts an object or array of
  token-id → bias, plus the model's vocabulary suppress-tokens merged at −∞,
  applied by a dedicated sampler at chain head (`server-schema.cpp:434-471`,
  `common/sampling.cpp:329-342`, `llama_sampler_init_logit_bias` at
  `src/llama-sampler.cpp:4043`); `ignore_eos` additionally injects precomputed
  EOS bias (`logit_bias_eog`, `server-schema.cpp:475-486`).
- **NInfer state:** rejected with 400 `logit_bias_not_supported` for nonzero
  values; all-zero accepted as neutral (`3e0ff840:docs/serving.md:132-137,
  143-145`). No bias-table dimension in the sampling Op.
- **Port shape + cost:** per-request bias table as Program operand at a stable
  address + new Op dimension + entry cap; graph-capture interaction. Cross-
  reference: #170 sized this **medium–large**.
- **Contract impact:** chat + Responses; official OpenAI slot (token id → bias
  −100..100, verified in the current spec).
- **Verdict:** **defer** (a) — cross-reference #170: real gap, lowest-frequency
  field, highest Engine cost.

### C3 — Penalties: repeat / frequency / presence (engine-level comparison per ticket scope)

- **llama.cpp state:** one penalties sampler, `penalty_last_n` window (default 64,
  0 disables) over **accepted generated tokens only** (the sampler's own ring
  starts empty — prompt tokens never enter it), `src/llama-sampler.cpp:2919-2948`
  (accept) + `:2856-2896` (struct). Repeat is **multiplicative contrast** over the
  logits: active (in-window) tokens ÷ `penalty_repeat`, inactive × `penalty_repeat`
  (`:3090-3107`); freq subtracts `penalty_freq·count(in window)`; presence
  subtracts `penalty_present` for any in-window occurrence. GPU backend variant
  included (`:3031-3130`).
- **NInfer state:** one fused adjustment in the sampling Op:
  `adjusted = logits − presence·(c>0) − frequency·c`, `c` = committed occurrence
  count over the **entire generation**, no window, no prompt tokens, no
  multiplicative repeat term (`3e0ff840:include/ninfer/ops/sampling.h:52-59`).
  `repetition_penalty` is fail-closed at the serve (`openai_chat_request.cpp:310-319`).
- **Comparison:** presence is window-independent by definition (both are exact).
  Frequency diverges on long generations: NInfer's count grows unbounded while
  llama.cpp's saturates at the 64-token window; neither spec (OpenAI: "based on
  their existing frequency in the text so far") pins a window, and neither
  implementation counts prompt tokens. Shipped Qwen presets use frequency 0 /
  presence ≤1.5, so the divergence is unobservable on the deployed presets; it
  only matters if a client sets `frequency_penalty > 0` on a long turn.
  Repeat (multiplicative) is absent in NInfer by design and is not an OpenAI or
  Anthropic slot (verified: no `repetition_penalty`/`repeat_penalty` in either
  spec).
- **Port shape + cost:** a windowed frequency variant is a small Op change
  (bounded sliding count); a repeat term is a one-line multiplicative pass —
  both inside the existing fused Op.
- **Contract impact:** none (no spec slot; llama.cpp-native + vLLM-style
  extensions).
- **Verdict:** **no** (c) — no contract gap, shipped presets unaffected, and a
  semantics change would shift existing Qwen-preset behavior for no consumer;
  keep the fail-closed `repetition_penalty` rule. Cross-reference #170
  (repeat_penalty in the zoo → no).

### C4 — The sampler zoo + chain architecture (`--samplers` / `--sampling-seq`,
`backend_sampling`, xTC/dynatemp/mirostat/typical/dry/adaptive/top_n_sigma/min_keep)

- **llama.cpp state:** request fields for the whole zoo
  (`common/common.h:236-253`: `xtc_probability/threshold`, `typ_p`,
  `dynatemp_range/exponent`, `dry_*`, `adaptive_target/decay`, `mirostat{,_tau,_eta}`,
  `top_n_sigma`, `min_keep`; `server-schema.cpp` exposes them). The default chain
  is `PENALTIES, DRY, TOP_N_SIGMA, TOP_K, TYPICAL_P, TOP_P, MIN_P, XTC,
  TEMPERATURE` then a terminal `dist` (or the adaptive-p / mirostat branches)
  (`common/common.h:264-274`, `common/sampling.cpp:344-413`), **reorderable** via
  `--samplers` / `--sampler-seq` (`common/arg.cpp:1985-2007`) and the `samplers`
  request field; each sampler is a separate CPU pass over the candidate array,
  with an optional per-sampler GPU backend offload (`backend_sampling`,
  `server-schema.cpp`; disabled when grammar or reasoning budget is active,
  `common/sampling.cpp:419-429`). Terminal samplers: `dist` (mt19937 seeded,
  `src/llama-sampler.cpp:1399`), mirostat v1/v2 (`:2537`, `:2643`), adaptive-p
  (`:3860`).
- **NInfer state:** no zoo and no chain — the entire pipeline is the single fused
  device Op described in the surface section (fixed order: penalties → sort →
  top_k → min_p → top_p → draw; greedy short-circuit). There is no per-sampler
  pass, nothing to reorder, and no CPU round-trip; `backend_sampling` is a
  non-question (NInfer's sampling already runs fully on-device, which is the
  end-state llama.cpp's flag offloads *toward*). `min_keep`: NInfer always
  retains at least the best candidate (equivalent to `min_keep=1`); llama.cpp's
  default is 0 (no guarantee). `ignore_eos`: absent in NInfer (model-default
  token stops are unconditional, `frontend.cpp:373-375`).
- **Port shape + cost:** each zoo sampler would be a new Engine Op with its own
  qualification contract, or a variant switch inside the fused Op; a chain-
  reordering API would require the per-sampler intermediate representation
  NInfer deliberately does not have.
- **Contract impact:** none — no OpenAI/Anthropic slot is any of these
  (verified against the current specs); on the chat route they are ignored
  today, on Responses they 400 `unknown_parameter` — safe either way.
- **Verdict:** **no** (c) — cross-reference #170 (the zoo as request fields → no);
  the engine-level survey confirms it: NInfer's single fused device pass is
  structurally cheaper than llama.cpp's per-sampler chain, reordering adds no
  contract or measurable 5090 value, and Qwen's registered presets need none of
  the zoo. Sub-item verdicts: `--samplers`/`--sampling-seq` order override
  **no** (c); `backend_sampling` **no** (superseded by NInfer's on-device Op);
  `min_keep`/`ignore_eos` **defer** (c) — cross-reference #170 (small real QoL
  toggles if a consumer appears).

### C5 — `min_p` and `top_k`/`top_p` variants

- **llama.cpp state:** `min_p` (default 0.05, relative to the max candidate
  probability, `src/llama-sampler.cpp:1882`), `top_k` (default 40, `:1519`),
  `top_p` (default 0.95, `:1719`); typical-p variant at `:1994` (zoo, see C4).
- **NInfer state:** `min_p` and `top_p` are exposed on all three routes (chat
  `0..1` on the OpenAI routes; Anthropic `[0,1]` per spec) and `top_k 0..20` on
  chat + Anthropic; semantics match llama.cpp's (relative-to-max min_p; prefix
  nucleus top_p) per the Op contract. The only divergence is NInfer's
  documented 20-cap on `top_k` (Qwen's presets use 20; llama.cpp's default 40
  would 400 on NInfer's chat route — a client migrating from a llama.cpp
  deployment could hit it).
- **Verdict:** **no** — parity on the exposed fields; the `top_k` cap is a
  documented, intentional contract (not a gap to port around).

### C6 — Stop-sequence handling (semantics comparison per ticket scope)

- **llama.cpp state:** server-side, checked after each decoded token against the
  accumulated generated text (`find_stopping_strings`,
  `1537a0a8b2:tools/server/server-context.cpp:588-617`): full stop (string
  search from the tail window) plus **partial stop** via
  `string_find_partial_stop` (`common/common.h:865`) so a stop word split across
  token boundaries still stops and is not emitted; the matched string is
  excluded from the output; EOS is a separate stop type (`STOP_TYPE_EOS`,
  `tools/server/server-task.h:44-49`) and `stop` maps to `finish_reason:"stop"`,
  `end_turn`, or `tool_calls` per route (`server-task.cpp:383-384, 423-424,
  465-466, 732-734`); the chat template may add stops
  (`server-common.cpp:1403-1405`, `chat_params.additional_stops`).
- **NInfer state:** Engine-side (model frontend) per the surface section:
  token stops (model defaults + caller) and string stops with UTF-8 reassembly
  and suffix-prefix withholding — the same split-token-boundary semantics as
  llama.cpp's partial stop — on a per-channel basis (reasoning and/or content),
  with `include_in_output` per string and `publish_stop_token` per token stop
  (`output_session.cpp:181-225, 523-552`). Chat-route `stop` applies to both
  channels; Anthropic `stop_sequences` returns `stop_reason:"stop_sequence"`
  with the matched sequence (`docs/serving.md:786-789`).
- **Comparison:** equivalent for every spec-relevant behavior (both exclude the
  matched stop from output; both detect stops split across tokens; both map
  stop vs length vs tool-call finishes). NInfer's is strictly more capable
  (per-channel targeting, UTF-8 safety across partial code points, token-level
  stops, explicit include/withhold flags).
- **Verdict:** **no** — parity verified; nothing to port.

### C7 — "Mirror sampling"

- **State:** does not exist at `1537a0a8b2` (see ticket-term notes). Dead end;
  nothing to survey or port.

### C8 — Grammar / constraint extensions beyond GBNF + JSON schema

- **llama.cpp state (extensions):**
  - **Lazy grammar with triggers** — `grammar_lazy` + `grammar_triggers`
    (word / regex pattern / full pattern / token;
    `server-schema.cpp:287-379`; `common_grammar_trigger` handling in
    `common/sampling.cpp:224-276`; `llama_sampler_init_grammar_lazy_patterns`
    at `src/llama-sampler.cpp:2843`): the grammar applies only after a trigger
    is seen; the OAI chat path derives lazy tool-call grammars from the chat
    template (`server-common.cpp:1394-1404`,
    `llama_params["grammar_lazy"]`/`["grammar_triggers"]`).
  - **llguidance grammars** — `%llguidance`-prefixed grammar values select
    Lark-based parsing behind the `LLAMA_USE_LLGUIDANCE` build flag
    (`common/sampling.cpp:217-222`).
  - **Top-level `json_schema`** OAI chat body field (alternative to
    `response_format`; `server-common.cpp:1252-1277`) — a llama.cpp extension,
    not an OpenAI spec field.
  - **Grammar application strategy** — sample first, check the single token
    against the grammar, resample with grammar-first only on a miss
    ("grammar-based rejection sampling", `common/sampling.cpp:602-683`), or
    `grammar_first` to apply the mask before sampling; prefill of
    output-format/tool-call grammars from the generation prompt
    (`common.h:220-224`, `common/sampling.cpp:282-312`).
- **NInfer state:** GBNF + `response_format` json_object/json_schema on the chat
  route, applied by the in-tree token-trie **mask producer** (the constraint is
  applied to the logits every step — there is no sample-then-resample path and
  no lazy trigger concept; thinking-mode requests get the constraint wrapped
  over the reasoning stream, `types.h:263-275`). Top-level `json_schema` is an
  unknown chat field (ignored — safe; the official `response_format` slot is
  served). No Lark route.
- **Port shape + cost:** lazy triggers would need a trigger-matching state
  machine in the constraint path (serve or Engine); llguidance would need a
  second grammar compiler. Neither touches the OpenAI/Anthropic contracts.
- **Verdict:** **no** (c) for lazy grammar/triggers, llguidance, and the
  top-level `json_schema` extension — no contract slot, NInfer's mask approach
  is the stronger application strategy (always-constrained vs resample-on-miss),
  and the trigger use case (tool-call grammar arming) is already covered
  structurally by NInfer's chat-parser region grammar. **No** for the
  application-strategy difference itself (parity/superiority, not a gap).
  Cross-reference: map #45 owns NInfer's constrained-decoding mechanism; #169
  verified the vendored PEG engine is byte-identical at baseline vs master.

### C9 — Anthropic `output_config.format` (structured outputs on the Anthropic
route) — **new finding, not covered by siblings**

- **llama.cpp state:** n/a (llama.cpp has no Anthropic structured-output slot;
  its Anthropic route passes through `temperature/top_p/top_k/stream/
  chat_template_kwargs` only, `1537a0a8b2:tools/server/server-chat.cpp:593-601`).
- **NInfer state:** the current Anthropic spec (verified 2026-10-03) defines
  `output_config.format: {type: "json_schema", schema}`; NInfer's Anthropic
  route **rejects** it with 400 `output_config_format_not_supported`
  ("output_config.format requires constrained decoding, which NInfer does not
  provide", `3e0ff840:src/serve/anthropic_messages_request.cpp:895-904`).
- **Port shape + cost:** route wiring only — parse + validate the format object,
  run the **existing** serve-side schema→GBNF conversion
  (`src/serve/constraint_contract.cpp`, with its context-accurate allowlist and
  the `json_schema_invalid`/`json_schema_unsupported` fail-closed codes), set
  `GenerationRequest.grammar` + `constraint_source`, map the conflict/unsupported
  rules (tools present → `constrained_decoding_conflict`; DFlash backend →
  `constrained_decoding_not_supported` naming the backend — same as the chat
  route's `grammar`+`response_format` treatment). **Small–medium** (Engine
  machinery already exists and is qualified; new work is the request parsing,
  error mapping, and schema tests).
- **Contract impact:** Anthropic route — an official spec slot that 400s today;
  clients on this route (Claude Code-family) cannot use structured outputs at
  all.
- **Verdict:** **port** (a) — official contract slot, fail-closed today, and the
  whole Engine side already exists. Flag: it lands in the constrained-decoding
  family's territory (shares the converter allowlist and its tests); see
  "Flagged for the map".

### C10 — Seed / determinism

- **llama.cpp state:** `seed` request field (int; −1 = fresh random per request;
  `LLAMA_DEFAULT_SEED = 0xFFFFFFFF`, `1537a0a8b2:include/llama.h:37`,
  `server-schema.cpp:175-177`); `dist`/XTC/adaptive draw from an mt19937 seeded
  per request (`common/sampling.cpp:440`, `src/llama-sampler.cpp:1470`); the
  server README notes logits are **not** guaranteed bit-identical across
  different batch sizes when `cache_prompt` is on
  (`tools/server/README.md:587`) — so same-seed determinism is qualified by
  execution shape.
- **NInfer state:** integer `seed` (u64; fresh random per request when omitted,
  `--seed` fixed, `docs/serving.md:914`; parsing at
  `openai_chat_request.cpp:43-49`); the Op uses a **counter-based** RNG keyed
  `(seed, logical_position, purpose)` with no mutable state and no dependence on
  the compact row index (`ops/sampling.h:68-71`); greedy rows are exact argmax;
  the request log records the resolved sampler + seed
  (`docs/serving.md:969`).
- **Comparison:** on the RNG itself NInfer's guarantee is stronger than
  llama.cpp's (stateless counter keyed on the logical position vs stateful
  mt19937 seeded per request; the llama.cpp README's batch-shape caveat about
  logits is about its own context, not the RNG). OpenAI's spec now marks `seed`
  **Deprecated** (beta, "determinism is not guaranteed") in the current
  reference (verified 2026-10-03) while still accepting it; NInfer accepts it
  on chat (and documents the same best-effort framing via the request log).
  Anthropic's current spec has no `seed` field; NInfer's Anthropic route has
  none either.
- **Verdict:** **no** — parity (NInfer's guarantee is strictly stronger); the
  spec's deprecation lowers, not raises, the bar.

### C11 — Reasoning-budget sampler (llama.cpp's thinking-budget machinery)

- **llama.cpp state:** a first-class **sampler** in the chain:
  `common_reasoning_budget_init` (start/end tag sequences, forced sequence,
  token budget) with the state machine IDLE → COUNTING → WAITING_UTF8 → FORCING
  (all logits −∞ except the forced token) → DONE
  (`1537a0a8b2:common/reasoning-budget.h:9-57`); applied before the chain
  (`common/sampling.cpp:639`), lazy grammars only apply when it is
  IDLE/DONE (`common/sampling.cpp:458-471`); wired from the OAI chat body
  (`reasoning_budget_tokens` / `thinking_budget_tokens` aliases +
  `reasoning_control`, `server-common.cpp:1405-1422`) and forceable mid-stream
  via `POST /v1/chat/completions/control` `action:"reasoning_end"`
  (`server.cpp:267-268`, `server-context.cpp:5169-5204`).
- **NInfer state:** thinking budgets exist and are **already deployed**:
  request fields (`thinking_budget`, `enable_thinking`), process
  `--default-thinking-budget`, Engine `ThinkingControlOptions.budget`, enforced
  in the model frontend with control-token close (surface section; production
  verification 2026-10-03, memory #421). The Anthropic contract slot
  (`thinking.budget_tokens ≥ 1024 < max_tokens`) is served
  (`docs/serving.md:791-796`).
- **Comparison:** same capability, different vehicle — llama.cpp implements the
  budget as a logits-level sampler (forces the end sequence in-graph); NInfer
  enforces it in the frontend (counts model-origin tokens, injects the
  canonical Qwen close); both vehicles inject guidance before the forced end
  tag (llama.cpp's is a configurable `reasoning_budget_message`). The only
  genuinely new llama.cpp behavior is the **runtime** control endpoint (client
  can force reasoning-end mid-stream) — that is a serve/transaction item,
  verdicted by #170.
- **Verdict:** **no** for the sampler itself — the capability is covered and
  deployed; the sampler vehicle would be redundant machinery. Cross-reference
  #170 Candidate 4 (`/v1/chat/completions/control` real-time control):
  **port** (c) — the companion piece, tracked over there.

### C12 — Control vectors (`--control-vector`)

- **llama.cpp state:** CLI-only activation steering: `--control-vector
  FNAME[,…]`, `--control-vector-scaled FNAME:SCALE,…`,
  `--control-vector-layer-range START END`
  (`1537a0a8b2:common/arg.cpp:2980-3008`); vectors are loaded
  (`common_control_vector_load`) and applied at context init via
  `llama_set_adapter_cvec` — an `n_embd × n_layers` buffer added as a per-layer
  residual over the configured layer range during the forward pass
  (`include/llama.h:732-744`, `common/common.cpp:1415-1435`). **Not exposed as a
  request field** on any server route (grep of `tools/server/` finds no cvec
  field).
- **NInfer state:** none — no control-vector flag, field, or Op.
- **Port shape + cost:** a model-forward change (per-layer residual addition in
  the Engine's execution path, weight-format loading) — an order of magnitude
  beyond the sampling family.
- **Contract impact:** none — no OpenAI/Anthropic slot; llama.cpp itself keeps
  it process-level.
- **Verdict:** **no** (c) — no contract slot, no request-level consumer, no Qwen
  ecosystem usage; and the implementation is model-forward machinery, not
  sampling. Flagged below for routing.

### C13 — Boundary-crossing flag (no verdict): `backend_sampling` GPU offload

Covered inside C4's verdict (no — superseded by NInfer's fused device Op);
listed here so the summary table matches the candidate list.

## Summary table

| Candidate | Verdict | Axis | One-line rationale |
|---|---|---|---|
| `logprobs`/`top_logprobs` | **port** (cross-ref #170) | a | Official OpenAI slot on chat + Responses that 400s today; port to the spec shape, not llama.cpp's non-OAI-compatible one |
| `logit_bias` | **defer** (cross-ref #170) | a | Official slot, 400 today, but lowest-frequency field with the highest Engine cost (per-request bias tables + graph capture) |
| Penalties (repeat/freq/presence) engine-level | **no** | c | Presence is exact in both; frequency window divergence is unobservable on shipped presets (freq 0) and unregulated by any spec; repeat has no slot and is fail-closed by design |
| Sampler zoo (DRY/mirostat/XTC/typical/dynatemp/adaptive/top_n_sigma/min_keep) | **no** (cross-ref #170) | c | No contract slot; ignore-or-fail-closed is already safe; Qwen presets need none of it |
| Chain order override (`--samplers`/`--sampling-seq`) | **no** | c | NInfer's single fused device Op is structurally cheaper than llama.cpp's per-sampler chain; reordering adds nothing measurable |
| `backend_sampling` | **no** | b (negative) | NInfer's sampling already runs fully on-device — the end-state that flag offloads toward |
| `min_p`, `top_k`/`top_p` | **no** | — | Parity verified (semantics match; NInfer's documented `top_k ≤ 20` cap is intentional) |
| Stop sequences | **no** | — | Parity verified; NInfer is a superset (per-channel, UTF-8-safe split-token stops, token stops, include/withhold flags) |
| Mirror sampling | n/a | — | Does not exist at `1537a0a8b2` — stale ticket hint |
| Lazy grammar + triggers | **no** | c | No slot; NInfer's mask approach is always-constrained; the arming use case is covered by the chat-parser region grammar |
| llguidance (Lark) grammars | **no** | c | Opt-in non-default upstream feature; NInfer uses the vendored PEG (byte-identical per #169) |
| Top-level `json_schema` (OAI chat) | **no** | c | llama.cpp extension, not an OpenAI field; NInfer ignores it safely and serves the official `response_format` slot |
| Anthropic `output_config.format` | **port** | a | Official Anthropic structured-outputs slot that 400s today; the entire Engine constraint machinery already exists — route wiring only (new finding of this ticket) |
| Seed / determinism | **no** | — | Parity; NInfer's counter-based guarantee is stronger; OpenAI's spec deprecates `seed` |
| Reasoning-budget sampler | **no** | — | Capability covered by NInfer's deployed thinking budget (different vehicle); only the runtime control endpoint is new → #170 Candidate 4 (port, c) |
| Control vectors | **no** | c | No slot (CLI-only even upstream); implementation is model-forward machinery, not sampling — flagged for routing |

## Flagged for the map

1. **Anthropic `output_config.format` (C9)** — verdict **port (a)**, but the
   implementation shares the constrained-decoding family's converter
   allowlist/tests (`src/serve/constraint_contract.cpp`). Recommend the
   coordinator route the ticket to the constrained-decoding family (map #45's
   lineage) or to serve with a shared-test requirement; the serve-side parse +
   error mapping is the only genuinely new work.
2. **Control vectors (C12)** — if ever pursued this is model-forward machinery
   (per-layer residual addition + vector weight loading in the Engine), a
   product-level capability, not a sampling port. No evidence of consumer
   demand in the Qwen deployment.
3. **Runtime reasoning-end control** — llama.cpp's
   `POST /v1/chat/completions/control` `reasoning_end` action (C11) is the
   serve-family companion to #170 Candidate 4 (real-time completion control,
   verdict **port (c)** there); no Engine work needed — the decode-boundary
   injection path already exists for the thinking-budget close.
4. **Contract-drift observation (serve family, informational):** the current
   official specs deprecate sampling fields NInfer still accepts — OpenAI
   `seed` (beta, "determinism not guaranteed") and Anthropic
   `temperature`/`top_k`/`top_p` (models after Claude Opus 4.6: temperature
   only 1.0 accepted, any `top_k` 400s, `top_p` ≥0.99 only — verified
   2026-10-03). NInfer's acceptance is a **superset** of the current specs
   (harmless: spec-conforming clients keep working), so this needs no action —
   recorded so a future contract pass doesn't mistake the deprecations for a
   gap.
5. **llama.cpp's logprobs shape is self-declared non-OAI-compatible**
   (`server-common.cpp:1430` TODO) — relevant to the C1 port: target the OpenAI
   spec response shape (`content[]`/`top_logprobs[]` with `bytes`), not
   llama.cpp's internal `probs_vector_to_json` shape.

**Dead ends / could-not-verify:** "mirror sampling" (does not exist at the
verified commit — C7). `platform.openai.com/docs` and `developers.openai.com`
API-reference pages are 403/404 to direct scrapes; the OpenAI contract was
grounded instead via the `developers.openai.com` reference (through web search
results) and the official `openai-python`/`openai-ruby`/`openai-node` SDK
sources fetched the same day — three independent official sources in agreement.
The Anthropic reference was fetched directly
(`platform.claude.com/docs/en/api/messages/create`).
