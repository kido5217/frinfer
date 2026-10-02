# opencode ⇄ ninfer-yarn serve: compatibility matrix and verdict

Wayfinder map #139 — the destination artifact (ticket #144). Question: how
compatible is ninfer-yarn's serve with opencode, and can it replace
llama.cpp server or vLLM as the self-hosted model backend for opencode?
Verdict stated first against the deployed `ai.kido.ws` testbed
(deployment-first — user pin, 2026-10-02).

**Evidence base (four findings docs, all reviewed adversarially)**

| Side | Doc | Branch @ commit |
|---|---|---|
| Requirement (opencode v2.0.22) | `docs/research/opencode-requirements.md` | `research/opencode-requirements` @ 8cde0058 |
| Capability (ninfer-yarn serve) | `docs/research/serve-capabilities.md` | `research/serve-capabilities` @ d56d1987 |
| Reference (llama.cpp server, vLLM) | `docs/research/reference-baselines.md` | `research/reference-baselines` @ 7c976d17 |
| Live e2e (probes + real sessions) | `docs/research/live-e2e.md` | `research/live-e2e` @ a2266efb |

Evidence classes: **live-measured** (24 HTTP probes replaying opencode's
request shapes, 3 real opencode agent sessions = 13 serve requests, request-log
mining, all against the deployed `ninfer-yarn-qwen38-27b` unit, 2026-10-02/03);
**code-grounded** (ninfer-yarn repo at `master` 8a0d16bf; opencode source at
`v2.0.22` 527f0b931; llama.cpp master bed0a85660; vLLM master ced6857);
**documentation-only** (the llama.cpp / vLLM arms were not run live — user pin).

**Deployment under test.** opencode v2.0.22 → `https://ai.kido.ws/v1`
(caddy) → `127.0.0.1:8827`, unit `ninfer-yarn-qwen38-27b`: Qwen3.8-27B NVFP4
`.ninfer`, `--max-context 446902` (YaRN above the native 262144), MTP4 +
`--lm-head-draft`, `--kv-dtype nvfp4`, `--default-max-tokens 32768`,
`--vision` on, froggeric v22.5 chat template, C=4/P=64, unauthenticated.
The two incumbent arms serve the same model id on the same port when they
hold the GPU (one GPU tenant at a time): `llama-cpp-qwen38-27b` (GGUF
UD-Q4_K_XL, MTP×3, 1 slot, vision on, GGUF-native window) and
`vllm-qwen38-27b-nvfp4` (HF NVFP4, 262144 VRAM-bound window, vision off,
single-stream). All three run the same froggeric v22.5 template, so
tool-call *prompting* is identical across arms; only serving/parsing differs.

---

## Verdict (deployment-first)

**For this deployment: yes — ninfer-yarn can replace llama.cpp server and
vLLM as the self-hosted backend for opencode.** It meets every opencode
backend requirement this effort tested — 24/24 request-shape probes, 13
clean live agent requests, and the operator's own ~200K-context agent
sessions running on it (the session that produced this report) — with the
strongest observed form of opencode's tool-call contract (one terminal
delta carrying complete id + name + full arguments). The opencode provider
config itself is tuned to ninfer-yarn: its declared 446902 window matches
the unit exactly, while it *over-declares* both incumbent arms — the vLLM
arm serves exactly 262144 (an 184,758-token over-declaration) and the
llama.cpp arm the GGUF-declared native window (262144-class; the exact
GGUF figure unverified, no live arms) — so with `compaction.auto: false`,
any prompt beyond the served window is a fatal step failure while those
arms hold the GPU.
The deployment's opencode integration is already built around the
ninfer-yarn window.

**Two conditions** (both recorded as gaps G1/G2 below; neither blocks the
protocol, both block a clean operational picture):

1. **Set a thinking budget or accept zero-visible-content turns** (G2): on a
   thinking-heavy turn the client-set 32,768 output bound can be consumed
   entirely by reasoning, leaving a silent empty reply that opencode codes as
   succeeded. Config-mitigable today: `--default-thinking-budget` on the unit,
   a larger `limit.output` or lower effort in the opencode config.
2. **Accept or address the parser-demotion class** (G1): a malformed
   tool-call region is returned as assistant text with `finish_reason:
   "stop"` and no in-band signal; opencode then idles with nothing pending.
   Observed at low frequency (2 `malformed_structure` in the 71 h before the
   e2e window; 0 in the e2e window), and it is a failure mode opencode cannot
   see or retry — its silent cousin is F1's zero-visible `length` truncation
   (G2), which was live-observed once in the e2e window.

**For the general self-hosting opencode user:** replaceable **within the
Qwen3.5/3.6/3.8 `.ninfer` model family on a single NVIDIA GPU**, under the
same two conditions. Outside that family (any other model/quant), model
coverage is a product-level blocker (G3): llama.cpp serves any GGUF (with a
multi-model router) and vLLM serves any HF model vLLM supports.

What ninfer-yarn has that the incumbents do not (deployment-relevant):
the 446902 YaRN window (the only arm matching opencode's declared window),
vision on (with the llama.cpp arm; the vLLM arm runs
`--language-model-only`), a default 32,768 output bound (the other two have
no server-side default bound — unbounded on llama.cpp, the remaining window
on vLLM), 429 backpressure (the other two queue without backpressure), C=4
concurrency (the other two run 1 slot / 1 seq), and the only measured
full-turn decode number in this effort (≈146 tok/s over a 32,768-token turn
at the 446902 window, MTP4).

What the incumbents have that it does not: model breadth (G3), a native
multi-model router (llama.cpp), and — for the vLLM arm only — a tool-parse
failure shape that cannot leak markup into content (it drops malformed
preamble bytes instead; at the cost of accepting undeclared tool names and
truncating arguments silently).

---

## Compatibility matrix

Each row is an opencode requirement (R1–R15; "req §N" =
`opencode-requirements.md` section). Cell evidence tags: **[live]** =
measured in the e2e ticket (probe P-number or session W1–W3); **[code]** /
**[doc]** = code-grounded / documentation-only. "Cap §N" / "Base §N" =
sections of the serve-capabilities / reference-baselines docs.

| # | opencode requirement | ninfer-yarn serve | llama.cpp `llama-server` | vLLM OpenAI server |
|---|---|---|---|---|
| R1 | Exactly one route: `POST {base}/chat/completions`, always `stream: true` (req §1.1) | Registered; SSE + `[DONE]` **[live P03/P24]** **[code cap §1]** | Registered; SSE + `[DONE]` **[doc Base §1.1]** | Registered; SSE + `[DONE]` **[doc Base §2.1]** |
| R2 | SSE contract: one choice per event; known `finish_reason` before `[DONE]`; no content after finish (req §3.1–3.4) | Only `stop`/`length`/`tool_calls`; finish always terminal; usage chunk before `[DONE]` **[live P03]** **[code cap §3]** | `stop`/`length`/`tool_calls`; same ordering; usage chunk lacks `reasoning_tokens` **[doc Base §1.4]** | `stop`/`length`/`content_filter`/`tool_calls`; emits `[DONE]` even after an in-band error (unique) **[doc Base §2.4]** |
| R3 | Tool calls: `id`+`name` complete by stream end; parallel calls; partial args (req §3.3) | One terminal `tool_calls` delta with complete id+name+full arguments — strongest form **[live P06/P07; W1 batch-3, W3 batch-6]** | Incremental fragments; parse-anywhere prefix-skip; failing regions demote to content (markup leaks; silent-stall class) **[doc Base §1.2]** | Slot per index, id+name in one delta; malformed preamble **dropped** (no leak); unterminated regions closed at end; **undeclared names accepted** **[doc Base §2.2]** |
| R4 | `max_tokens` + `max_completion_tokens` on every agent step; honor the tighter (req §2.2) | Both accepted; `max_completion_tokens` wins **[code cap §2.1; live P04]**; default 32,768 **[live P05]** | `max_completion_tokens` wins (alias order); **no default output bound** (`n_predict=-1`) **[doc Base §1.5]** | `max_completion_tokens` wins; default = remaining window **[doc Base §2.5]** |
| R5 | Reasoning channel: `reasoning_content` deltas + replay (req §2.3/§3.6) | `reasoning_content` (first in opencode's fallback chain); replay of `content: null` + `reasoning_content` accepted **[live P03/P07]** | `reasoning_content` (first in chain); replay accepted **[doc Base §1.3]** | `reasoning` (second in chain); replay normalized `reasoning_content`→`reasoning` **[doc Base §2.3]** |
| R6 | `reasoning_effort` values incl. `xhigh` (req §2.6) | `xhigh` → template effort level (froggeric) **[code cap §2.2]** | `xhigh` known to template + server flag **[doc Base §1.3]** | any non-`none` **enables thinking; level not modeled** **[doc Base §2.3]** |
| R7 | Sampling fields absent (req §2.5) | Registered thinking preset applies (1.0/0.95/20) **[code cap §2.4; log]** | Block sampling flags (1.0/0.95/20/0) **[doc Base §3]** | Model `generation_config` defaults **[doc Base §3]** |
| R8 | Overflow must classify as `context-overflow` (fatal step here, `compaction.auto: false`) (req §4.1/§7.4) | 400 code `context_length_exceeded`, message names 446902, submit-time, no GPU work **[live P16]**; declared window == deployed 446902 | 400 `exceed_context_size_error`; phrase matches opencode's list **[doc Base §1.7]**; serves GGUF-native window only — **over-declared by the opencode config** | 400 "exceeds model's maximum context length"; phrase matches **[doc Base §2.5]**; 262144 ceiling — **over-declared by the opencode config** |
| R9 | Overload: 429 retryable; dead stream retryable with continuation (req §4.2–4.5) | 429 `server_overloaded` at C+P=68 full; 503 queue timeout; engine death = in-band SSE error, no `[DONE]` → opencode retryable + continues **[code cap §3c/§5]** | **No 429**; unbounded slot-queue wait (3600 s timeout) **[doc Base §1.7]** | **No 429**; scheduler wait; single-stream at `--max-num-seqs 1` **[doc Base §2.7]** |
| R10 | Image input: opencode sends `image_url` parts by default (req §2.7) | `--vision` on; data-URL image accepted end-to-end, `vision_tokens` logged **[live P17; code `docs/serving.md`]**; 32,768 merged-token envelope | On (`--image-min-tokens 1024`, mmproj on CPU) **[doc Base §3]** | **Off** (`--language-model-only`; VRAM constraint) **[doc Base §2.6]** |
| R11 | Model metadata from config only; declared window must match reality (req §5.1) | `/v1/models` reports 446902 = declared 446902 **[live P01; config]** | `/v1/models` exists; serves GGUF-declared context (no extension) — opencode config over-declares it **[doc Base §1.6]** | `/v1/models` exists; 262144 ceiling — opencode config over-declares it **[doc Base §2.6]** |
| R12 | No auth header in this deployment (req §1.3) | Unauthenticated (no `--api-key`) **[code cap §1; live all probes]** | Block has no auth config **[doc Base §3]** | Block has no auth config; note: vLLM `--api-key` would leave non-`/v1` routes open **[doc Base §2.1]** |
| R13 | Tolerate `store:false`, `strict:false` tools, `tool_choice:none`, unknown fields, `x-opencode-*` headers (req §1.4/§2.1/§7.5) | All accepted — `store:false` legal, non-strict tools fine, `none` fine, unknown fields ignored **[live P08/P14]** | Compat posture "suffices to support many apps" (no strong spec claim); unknown-field behavior not documented in the sources read **[doc Base §1.1]** | `strict: false` and `tool_choice: none` honored (opencode's only values); `store` / unknown-field behavior not documented in the sources read **[doc Base §2.2]** |
| R14 | Model coverage (deployment: Qwen3.8-27B only; general: any) | Qwen3.5/3.6/3.8 `.ninfer` family; one model id per unit (product: one resident model) **[doc cap §0]** | **Any GGUF, any quant, multi-model router** (up to 4 concurrent) **[doc Base §1.8]** | **Any HF model vLLM supports**; one model per instance (+router) **[doc Base §2.8]** |
| R15 | Performance envelope (deployment context sizes) | **Measured**: TTFT 0.09–1.44 s at 15–29K-token prompts; ≈146 tok/s over a 32,768-token turn (446902 window, MTP4); queue waits 12.4–18.9 ms with a concurrent session **[live W1–W3]** | MTP×3; no full-window benchmark recorded in the config; Xid-8 watchdog history **[doc Base §1.6]** | Full-window (depth 243,712): pp 1,466 t/s, tg 33–52 tok/s — documented in the config comment; **not live-run in this effort** **[doc Base §2.6]** |

R15 caveat: the three cells are **not comparable workloads** — the
ninfer-yarn number is measured here at a 15.6K-token prompt and 446902
window; the vLLM number is a 243K-depth llama-benchy run at the 262144
ceiling; the llama.cpp arm has no recorded equivalent. No like-for-like live
performance comparison was made (no live baseline arms — user pin). The
config's recorded history names the ninfer unit the full-window leader
(157.4 t/s, 2026-08-28 measurements, llm.nix comment) — documentation-only.

**Not on opencode's critical path** (all three backends neutral or
capability-adding): `/v1/models` (never fetched, req §1.6),
`/v1/embeddings` (never called, req §1.7), OpenAI Responses / Anthropic
Messages surfaces, constrained decoding (opencode sends no constraints,
req §2.1 — ninfer-yarn's grammar/`response_format` support is live and
enforced, **[live P19–P23]**, llama.cpp/vLLM have their own
structured-outputs stacks), prompt caching (opencode sends no
`prompt_cache_key`/breakpoints, req §2.8 — serve-side prefix reuse is
transparent), MTP (wire-transparent, **[live e2e §3]**).

---

## Gap list (severity-graded)

| # | Severity | Status | Gap | Failing scenario | Evidence | Fix class |
|---|---|---|---|---|---|---|
| G1 | **Major** | open | Chat-parser demotion → silent stall (markup returned as assistant text, `finish_reason: stop`, no in-band error) | Model emits a malformed / duplicate-parameter / undeclared tool region → opencode executes no tools and idles until re-engaged; the only detector is the JSONL `fallback_reason` | 5 demotions in 4 days (2026-09-29 journal); 2 `malformed_structure` in the 71 h before the e2e window (req 310, req 533); 0 in the e2e window (live-e2e §2); mechanism code-grounded (cap §3b) | Serve change (parser robustness / a client-visible demotion signal) — separate effort; the vLLM arm's drop-not-leak shape is the alternative design point |
| G2 | **Major** | open (config-mitigable) | Zero-visible-content `length` truncation under unbounded thinking + finite client output bound | Thinking-heavy turn consumes the whole client-set 32,768 bound in `reasoning_content` → silent empty reply; opencode codes the step `succeeded` | e2e W2 (live): one 98,642-char reasoning part, `finish: length`, `tokens {output: 0, reasoning: 32768}`, `outcome: succeeded` (live-e2e §2 F1) | Config: `--default-thinking-budget` on the unit, or larger opencode `limit.output`, or lower effort. Since opencode v2.0.22 sends `max_completion_tokens` on every agent step (req §2.2), the same exposure holds on both incumbent arms (doc-level: no thinking budget configured there); the llama.cpp arm's unbounded default only spares the boundless generate/title requests |
| G3 | **Blocker** | product-level | Model coverage: Qwen3.5/3.6/3.8 `.ninfer` family only | A self-hosting opencode user whose model is outside the family cannot adopt ninfer-yarn | cap §0 (product scope); Base §1.8/§2.8 (incumbent breadth) | Product-level (new architectures / conversion recipes) — ruled out of this effort's scope |
| G4 | **Minor** | product-level | Single resident model / single model id per unit | opencode multi-model workflows (different model per session) need a unit swap; the deployment already swaps units behind caddy | cap §0; Base §2.1 (vLLM same pattern); Base §1.8 (llama.cpp has a native router) | Product-level; operationally handled by the caddy-level unit swap the deployment already uses |
| G5 | Cosmetic | open | Unknown routes return 404 with an empty body (no error envelope) | Only affects non-opencode clients hitting wrong OpenAI paths; opencode's single route is registered | cap §1 **[code/probe]** | Serve change (optional hardening) |
| G6 | Cosmetic | recorded (client-side) | `opencode run --standalone --format json` stdout dropped the event stream on W2 despite full persistence | Anyone scripting against `opencode run` JSON output on long reasoning-heavy turns | live-e2e §2 O1 (opencode store proves full receipt) | opencode client (out of scope; recorded for the record) |

## Known-failure-modes review

Fixed and verified live in this effort:

- **`max_tokens` omission (2026-09-22 incident).** Superseded on the client
  side: opencode v2.0.22 sends both max fields on every agent step
  (req §2.2); serve honors `max_completion_tokens` over `max_tokens`
  **[live P04]**, with a 32,768 default when neither arrives **[live P05]**.
  The truncation mode is structurally closed.
- **Duplicate-parameter demotion (2026-09-29).** Byte-identical repeated
  parameters now merge (PR #42; `duplicate_arguments_merged` counter in the
  JSONL); e2e W3 exercised repeated-parameter calls with 0 merges needed and
  0 demotions. Differing-value repeats still fail closed — the class is
  reduced, not eliminated (feeds G1).
- **Constrained decoding (2026-10-02).** Shipped; opencode never sends
  constraints, but the capability is live and enforced **[live P19–P23]**.

Still open: G1 (demotion → silent stall) and G2 (zero-visible `length`)
above. Engine death mid-stream is handled per contract (retryable dead
stream + continuation, cap §3c) but was not inducible in this effort; 0
observed.

## What this assessment does not cover

- **No live baseline arms** (user pin, 2026-10-02): the llama.cpp / vLLM
  columns are documentation-level (official docs + source at the pinned
  commits + the user's llm.nix configs). Their cells can be wrong in ways
  only a live run would show (e.g. the vLLM `:latest` image resolves to
  whatever the registry held at pull time).
- **Single model, single deployment**: the verdict is about Qwen3.8-27B on
  the 5090. Other `.ninfer` family members, other GPUs, and the upstream
  (Neroued/ninfer) unit are out of scope.
- **No like-for-like performance comparison** (R15 caveat): no live
  baseline arms, so cross-arm performance is not measured here.
- **G1/G2 frequencies** are estimates from one production week + the e2e
  window, not a statistical rate.

## Evidence pointers

- Probes: `e2e/probe_suite.py` + `e2e/results/probes-20261003-002106.jsonl`
  (final, 24/24) @ `research/live-e2e`; reviewer's independent re-run
  (24/24, byte-identical checks) 2026-10-03.
- Sessions: `e2e/sessions/w{1,2,3}/` @ `research/live-e2e`; opencode store
  evidence for W2: `~/.local/share/opencode/opencode.db`, session
  `ses_f0180baaeffe2IOtQKxoOfHD5X`.
- Request log: `/home/kido/trash/temp/ninfer-yarn-log.jsonl` (schema v22);
  e2e window 2026-10-02 21:21:30–21:28:20 UTC (requests 627–630, 633, 641–648).
- Deployment config: `nixos-configs/users/kido/llm.nix` (yarn unit 484–580;
  llama.cpp 264–315; vLLM 342–407; identical-behavior contract 317–326;
  689-line file as of 2026-10-03).
