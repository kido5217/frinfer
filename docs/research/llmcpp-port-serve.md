# llama.cpp serve/API surface vs NInfer serve — parity gap survey

Ticket: #170 (wayfinder map #167 "llama.cpp port candidates").

## Scope and method

**Question (verbatim):** What does llama.cpp's HTTP server (`llama-server` and its
documented OpenAI-compatible API surface) offer that NInfer's serve does not, as port
candidates? For each candidate: (1) llama.cpp state with source + commit, (2) NInfer state
today (verified from the repo), (3) port shape + cost estimate + contract impact on the
three exposed routes, (4) verdict candidate — port / defer / no — ranked by
(a) exposed contract/protocol gap on the three routes > (b) measurable 5090 performance
headroom > (c) QoL/ecosystem parity. Boundary-crossing items (concurrency / preemption /
priority-QoS / multi-GPU / one-resident-model / new mathematical architecture) are flagged,
not verdicted.

**Sources (all read this session):**

- llama.cpp master at commit `1537a0a8b2f8711d840878b0a0677ab2213c882c`
  (committer date 2026-10-03T15:19:13Z, verified live via
  `gh api repos/ggml-org/llama.cpp/commits/master` on 2026-10-03). Blobless local clone at
  `/tmp/opencode/llamacpp-serve-ref`; all llama.cpp citations are
  `1537a0a8b2:path`. Primary sources: `tools/server/README.md` (the endpoint contract),
  `tools/server/server.cpp` (route table), `tools/server/server-schema.cpp` (request
  parameter schema), `tools/server/server-common.cpp`, `server-task.cpp`, `server-chat.cpp`,
  `server-context.cpp`, `server-http.cpp`, `server-stream.{h,cpp}`, `server-queue.{h,cpp}`.
  No web search was needed: every claim is grounded in the source at the verified commit,
  which is strictly stronger than secondary documentation.
- NInfer master at `3e0ff840` (read-only): `docs/serving.md` (authoritative surface doc,
  1131 lines), `src/serve/http_server.cpp` (route registration + CORS/auth),
  `src/serve/openai_common.cpp` (model objects, error body),
  `src/serve/openai_chat_request.cpp` (chat request field handling),
  `src/serve/serve_options.cpp` (full `--flag` inventory), `AGENTS.md` (fixed product
  boundaries). No prebuilt binary was required; `--help` was verified from
  `src/serve/serve_options.cpp`.

**Ticket-term note:** the ticket's hint list mentions "props/suffix and mid-suffix" and
"completion-details". At `1537a0a8b2` there is no `mid-suffix` feature and no
completion-details endpoint; `suffix` appears only as a *rejected* legacy parameter
(`tools/server/server-common.cpp:1056`, `unsupported_params = {"best_of", "suffix"}`).
These hints appear stale; the survey below covers what the tree actually contains.

**llama.cpp route table (complete, verified at 1537a0a8b2:tools/server/server.cpp:249-340):**

| Method + path | Handler family |
|---|---|
| GET `/health`, `/v1/health` | health (public, no API key) |
| GET `/metrics` | Prometheus (gated by `--metrics`) |
| GET/POST `/props` | global properties (POST gated by `--props`) |
| GET `/models`, `/v1/models` | model info |
| POST `/completion`, `/completions` | legacy llama.cpp completion |
| POST `/v1/completions` | OpenAI completions |
| POST `/chat/completions`, `/v1/chat/completions` | OpenAI chat completions |
| POST `/v1/chat/completions/control` | real-time in-flight control |
| POST `/responses`, `/v1/responses` | OpenAI Responses |
| POST `/v1/audio/transcriptions`, `/audio/transcriptions` | OpenAI audio transcription (MTMD audio models) |
| POST `/v1/messages` | Anthropic Messages |
| POST `/infill` | FIM |
| POST `/embedding`, `/embeddings` | embeddings (all poolings) |
| POST `/v1/embeddings` | OpenAI embeddings |
| POST `/rerank`, `/reranking`, `/v1/rerank`, `/v1/reranking` | reranking (gated by `--rerank`) |
| POST `/v1/systemone` | TypeSafe System One |
| POST `/tokenize`, `/detokenize`, `/apply-template` | tokenizer utilities |
| POST `/chat/completions/input_tokens`, `/v1/chat/completions/input_tokens` | chat token count |
| POST `/responses/input_tokens`, `/v1/responses/input_tokens` | Responses token count |
| POST `/v1/messages/count_tokens` | Anthropic token count |
| GET/POST `/lora-adapters` | LoRA scale control |
| GET `/slots`, POST `/slots/:id_slot` (save/restore/erase) | slot state + KV save/load |
| GET/DELETE `/v1/stream`, POST `/v1/streams/lookup` | resumable streaming |
| GET/POST `/tools` | server tools / MCP (gated by `--tools`/MCP config) |
| GET/POST `/cors-proxy` | MCP browser proxy (gated by `--ui-mcp-proxy`) |
| POST `/models/load`, `/models/unload`, GET `/models/sse`, POST/DELETE `/models` | multi-model router |
| `register_gcp_compat()` (server-http.cpp:814) | Vertex AI `/predict` + health route |
| static assets (`/index.html`, `/sw.js`, …) | embedded Web UI |

**NInfer route table (complete, verified at 3e0ff840:src/serve/http_server.cpp:428-490):**

| Method + path | Notes |
|---|---|
| GET `/health` | 200 `ok` / 503 `unavailable`, unauthenticated |
| GET `/v1/models`, `/v1/models/{id}` | alias + `max_model_len`; 404 `model_not_found` on other ids |
| POST `/v1/chat/completions` | OpenAI chat |
| POST `/v1/responses` | OpenAI Responses Core (typed Items, local store) |
| POST `/v1/responses/input_tokens` | prompt-token count, no generation |
| GET/DELETE `/v1/responses/{id}`, GET `/v1/responses/{id}/input_items` | local Response store |
| POST `/v1/responses/compact`, `/v1/responses/{id}/cancel` | explicit rejections |
| POST `/v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages + count |

NInfer unknown routes → 404; a 404 under the `/v1/messages*` prefix is rendered with the
Anthropic error envelope, other paths get the default 404 body (with the `x-request-id`
header) (`src/serve/http_server.cpp:336-360, 158-186`).

---

## Candidate 1 — `n` > 1 (multiple completions per request)

- **llama.cpp state:** `n_cmpl` (aliases `n`) request field, hard limits `1..n_parallel`,
  creates `n-1` child tasks sharing the prompt
  (`1537a0a8b2:tools/server/server-context.cpp:4551-4557`); available on both
  `/completion` and `/v1/chat/completions` (the chat path evaluates the same schema,
  `1537a0a8b2:tools/server/server-context.cpp:4534`).
- **NInfer state:** rejects `n != 1` with HTTP 400 code `n_not_supported`
  ("NInfer produces one completion per request",
  `3e0ff840:src/serve/openai_chat_request.cpp:874-878`).
- **Port shape + cost:** after the single shared prefill, fan out into `n` generation
  transactions and merge into an `n`-entry `choices` array (usage summed, per-choice
  finish reasons). The compact decode batch already carries up to 8 concurrent requests
  (`--max-concurrency`), so capacity exists. Serve-level request/response work plus a small
  Engine transaction fan-out API. **Medium** (the Engine fan-out is the only non-trivial
  part).
- **Contract impact:** chat + responses routes (an accepted OpenAI field currently
  fails-closed); no impact on Anthropic (its protocol has no `n`).
- **Verdict:** **defer** (a). Genuine OpenAI contract field NInfer rejects; cost/value is
  moderate (clients needing n>1 can issue n requests, and OpenAI itself de-emphasizes
  n>1), so it ranks below the logprobs gap.

## Candidate 2 — `logprobs` / `top_logprobs`

- **llama.cpp state:** `logprobs` (alias for `n_probs`) and `post_sampling_probs` request
  fields; per-choice `logprobs.content[]` with token/`logprob`/`bytes` and `top_logprobs`
  entries in aggregate chat and completions responses
  (`1537a0a8b2:tools/server/server-task.cpp:434-440, 264-300`).
- **NInfer state:** rejects requested log probabilities with 400 `logprobs_not_supported`
  (`3e0ff840:docs/serving.md:132-137`; `logprobs:false` and `top_logprobs:0` are the
  semantically-neutral accepted forms).
- **Port shape + cost:** Engine exposes, per committed output token, the top-`k` logprobs
  gathered from the logits the sampler already consumes (CPU-side top-k over the logits row
  at each committed step; a per-program buffer at stable addresses, consistent with the
  CUDA-graph capture rule). Serve maps to OpenAI's `logprobs` choice shape (aggregate +
  streaming). **Medium** (new Engine output channel; a CPU top-k gather per committed
  step is small against the ~13.7 ms/step decode baseline measured at `b84e4fb4`).
- **Contract impact:** chat + responses routes — an official OpenAI field currently
  fails-closed; used by eval/judge tooling. Anthropic protocol has no logprobs surface.
- **Verdict:** **port** (a). The highest-value OpenAI contract field gap: official field,
  bounded Engine change, clients fail today with a 400.

## Candidate 3 — `logit_bias`

- **llama.cpp state:** full `logit_bias` map token-id→bias, request field plus
  `--logit-bias` CLI (`1537a0a8b2:tools/server/server-schema.cpp`, README "Sampling
  params").
- **NInfer state:** rejects nonzero `logit_bias` with 400 `logit_bias_not_supported`
  (all-zero accepted as neutral; `3e0ff840:docs/serving.md:132-137, 230-231`).
- **Port shape + cost:** Engine sampler applies a per-request token-id bias table; the
  table is per-program data at a stable address (programs own their operands), sized/bounded
  by a cap on distinct bias entries. **Medium–large** (new sampler Op dimension + bound +
  graph-capture interaction), for a low-frequency client need.
- **Contract impact:** chat + responses routes; official OpenAI field.
- **Verdict:** **defer** (a). Real gap but low real-world frequency for the Qwen client
  population and the highest cost of the (a)-axis items.

## Candidate 4 — real-time completion control (`/v1/chat/completions/control`)

- **llama.cpp state:** request field `reasoning_control: true` arms in-flight control;
  `POST /v1/chat/completions/control` with `action: "reasoning_end"` forces the end of the
  current reasoning block mid-stream while the client still reads the SSE
  (`1537a0a8b2:tools/server/server.cpp:268`, README §1445-1461).
- **NInfer state:** no in-flight control. The closest machinery exists: the thinking-budget
  cap commits Qwen's canonical early-close guidance + close marker at the budget boundary
  mid-generation (`3e0ff840:docs/serving.md:310-327`), but only for a budget set at
  request/prepare time.
- **Port shape + cost:** register live generation transactions by request id; the control
  route sets a flag consumed at the next decode boundary; the boundary injection reuses the
  existing thinking-control-suffix path. Serve route + transaction registry + one Engine
  control signal. **Medium** (the hard part — the decode-boundary injection — already
  exists for the budget cap).
- **Contract impact:** chat route (new optional request field + new endpoint); no impact
  on the Anthropic/Responses contracts (Anthropic has no equivalent).
- **Verdict:** **port** (c). Top QoL item: it directly serves the thinking UX on the
  deployed client population (a "stop overthinking" mid-stream), at medium cost on
  existing machinery.

## Candidate 5 — resumable streaming (`/v1/stream`, `/v1/streams/lookup`)

- **llama.cpp state:** streams survive HTTP disconnect: the producer tees SSE bytes into a
  per-conversation ring buffer keyed by conversation id; `GET /v1/stream?conv_id=…` resumes
  from any offset, `POST /v1/streams/lookup` lists owned live ids, `DELETE` cancels
  (`1537a0a8b2:tools/server/server-stream.h`, `server.cpp:313-315`; the conv id travels in
  a query string because it may embed `/`).
- **NInfer state:** disconnect cancels the request: SSE keep-alive comment + 15 s
  `TCP_USER_TIMEOUT` aim to detect dead peers and cancel within ~20 s
  (`3e0ff840:docs/serving.md:78-85`); no resume surface.
- **Port shape + cost:** a stream-session manager (ring buffers, ownership, TTL) plus a
  change in request lifetime: the Engine transaction must keep generating after the client
  disconnects, with explicit cancel moved to the DELETE route. That collides with the
  current "transport terminal = request terminal" lifecycle documented for the three
  routes. **Large** (lifetime/ownership redesign across serve and Engine cancellation).
- **Contract impact:** all three routes — disconnect semantics change (currently
  disconnect = cancel).
- **Verdict:** **defer** (a). Real resilience gap for flaky clients, but the highest-cost
  item in the survey and it redefines a contract NInfer currently documents explicitly;
  needs a lifecycle decision before any port work.

## Candidate 6 — legacy Completions API (`/completion`, `/v1/completions`, `/infill`)

- **llama.cpp state:** `/completion` + `/completions` (llama.cpp-native, full 68-field
  schema), `/v1/completions` (OpenAI legacy; rejects `echo`, `best_of`, `suffix`;
  `1537a0a8b2:tools/server/server-common.cpp:1035-1060`), `/infill` (FIM with
  Suffix/Prefix/Middle and `--spm-infill` variants).
- **NInfer state:** no route (404). The product surface is chat-template-first; prompt-mode
  completion is not an exposed contract.
- **Port shape + cost:** a prompt-mode request path (raw prompt, no template render,
  `add_special`, `object: "text_completion"` responses) reusing the same Engine route;
  `/infill` would additionally need FIM template semantics the Qwen frontends don't carry.
  **Medium** for `/v1/completions` alone; **large** if infill is included.
- **Contract impact:** new fourth OpenAI route; none of the three routes affected.
- **Verdict:** **no**. OpenAI's own API guide marks the completions endpoint as legacy
  (final update July 2023, with a deprecation notice —
  https://developers.openai.com/api/docs/guides/completions, checked 2026-10-03); the
  client population on NInfer uses chat/responses/anthropic. A new prompt-mode contract
  is maintenance surface with no current consumer. (If a consumer appears,
  `/v1/completions` alone is a medium serve-side add.)

## Candidate 7 — tokenizer utilities (`/tokenize`, `/detokenize`, `/apply-template`) + chat `input_tokens`

- **llama.cpp state:** `/tokenize` (`content`, `add_special`, `parse_special`,
  `with_pieces`), `/detokenize` (tokens → text), `/apply-template` (render a conversation
  through the chat template), plus `/chat/completions/input_tokens` and
  `/v1/responses/input_tokens` counting endpoints (README §677-740, §1567-1593; handlers
  at `1537a0a8b2:tools/server/server-context.cpp:5303+`).
- **NInfer state:** no tokenizer routes. Token counting exists for two of the three
  routes: `POST /v1/responses/input_tokens` and `POST /v1/messages/count_tokens`
  (`3e0ff840:src/serve/http_server.cpp:447, 471`); chat completions has no counting
  endpoint.
- **Port shape + cost:** the artifact's tokenizer and the serve's template renderer are
  already in-process; `/tokenize`/`/detokenize` are thin queries, `/apply-template` reuses
  the request render path. Chat `input_tokens` reuses the exact same prompt-path counting
  the Responses endpoint already runs. **Small** (all serve-side, no Engine change).
- **Contract impact:** new endpoints; the chat counting endpoint fills a small parity hole
  on the chat route (llama.cpp and some providers expose it; OpenAI's own counting surface
  is Responses-scoped).
- **Verdict:** **defer** (c). Useful for debugging and client-side token budgeting, cheap
  to build, but no contract gap on the three routes and no current consumer in the
  deployment.

## Candidate 8 — Prometheus `/metrics`

- **llama.cpp state:** `GET /metrics`, Prometheus text, gated by `--metrics`; counters for
  prompt/predicted tokens and seconds, throughput gauges, `requests_processing`,
  `requests_deferred`, context high-watermark, decode counts, and speculative-decoding
  draft/accepted counters (README §1122-1148).
- **NInfer state:** no scrape endpoint. Machine-readable observability is the
  `--request-log-jsonl` schema-v24 stream plus 5 s stderr throughput records; the JSONL
  `throughput` event already snapshots the full Engine scheduler state per interval
  (`3e0ff840:docs/serving.md:953-1052`).
- **Port shape + cost:** render the same counter state the JSONL writer aggregates into
  Prometheus exposition format on a gated route (`--metrics` or on by default). Counters
  exist; the work is formatting + endpoint + contract tests. **Small–medium**.
- **Contract impact:** new endpoint; none of the three routes affected.
- **Verdict:** **port** (c). The one observability format the local production stack
  (caddy/opencode deployment) would consume directly; NInfer's JSONL is a log, not a
  scrape target.

## Candidate 9 — `GET /props` / `POST /props`

- **llama.cpp state:** `GET /props` returns default generation settings, slot count,
  model path, chat template + template capabilities, modalities, build info, sleeping
  state; `POST /props` mutates global properties when started with `--props`
  (README §830-934).
- **NInfer state:** no runtime introspection endpoint; configuration is startup-fixed
  (product: startup-fixed concurrency and sampling overrides) and is recorded in the JSONL
  `server_start` event (`3e0ff840:docs/serving.md:966-968, 1054-1057`).
- **Port shape + cost:** GET is a small JSON assembly from `serve_options` + Engine state
  (model id, context, KV dtype, spec backend, sampler defaults, build id). **Small.**
  POST would make process-level parameters mutable at runtime — incompatible with the
  startup-fixed design and with CUDA-graph capture (device state a captured kernel reads
  must be stable before capture; per-request sampling is the sanctioned mutation surface).
- **Contract impact:** new endpoint (GET only recommended).
- **Verdict:** **defer** for GET (c; overlaps what JSONL `server_start` gives offline, and
  there is no Web UI that consumes it). **no** for POST (c; conflicts with the
  startup-fixed product design).

## Candidate 10 — `/slots` + slot save/restore/erase

- **llama.cpp state:** `GET /slots` lists per-slot processing state (params, speeds,
  processing flags; `?fail_on_no_slot=1` for readiness); `POST /slots/{id}?action=
  save|restore|erase` (de)serializes the slot's KV/prompt cache to files under
  `--slot-save-path` (README §974-1199, §1148-1199).
- **NInfer state:** no slot model. Per-request state lives in the Engine's checkpoint
  system (Device/Host StateImage + pinned Host KV, pressure-managed, prompt-identity
  reuse); live scheduler state appears in JSONL `throughput` snapshots
  (`3e0ff840:docs/serving.md:970-973, 1089-1091`).
- **Port shape + cost:** a monitoring endpoint could be a thin live scheduler-snapshot
  accessor (**small–medium**). File save/restore would require a KV/checkpoint
  serialization format and a restore path in the Engine — NInfer re-derives prefix state
  from prompt identity instead, so the file workflow is a llama.cpp slot-model artifact
  rather than a general need. **Large** for save/restore.
- **Contract impact:** new endpoints; three routes unaffected.
- **Verdict:** **defer** for GET (c; JSONL covers the data, no consumer). **no** for
  save/restore/erase (c; Engine cost is large and NInfer's prefix reuse makes the file
  workflow redundant within its own design).

## Candidate 11 — embeddings (`/embedding`, `/v1/embeddings`) and reranking

- **llama.cpp state:** embeddings with pooling `none|mean|cls|last|rank`
  (`--pooling`), `/embedding` returns per-token embeddings for pooling-none,
  `/v1/embeddings` is OpenAI-shaped (normalized pooled vectors, multimodal content
  inputs); `/rerank`/`/v1/rerank` gated by `--rerank` (README §740-795, §1504-1566,
  server-specific params).
- **NInfer state:** none. The loaded artifact is a generative causal LM
  (Qwen3.5 dense/MoE, `Qwen3_5ForCausalLM`/`Qwen3_5MoeForCausalLM`); there is no pooling
  head and NInfer's docs reject non-generation uses
  (`3e0ff840:docs/serving.md`, `AGENTS.md` architecture section).
- **Port shape + cost:** a pooling head on the loaded architecture is a new mathematical
  representation/product change, not a serve add-on.
- **Contract impact:** none on the three routes.
- **Boundary-crossing flag: yes** (new mathematical architecture requires an explicit
  product change per `AGENTS.md`). No verdict — routes to the map's Out of scope if ever
  wanted.

## Candidate 12 — audio transcription (`/v1/audio/transcriptions`)

- **llama.cpp state:** OpenAI-shaped transcription for MTMD models with audio support
  (converts the transcription request to a chat completion over the audio input; rejects
  non-audio models with a not-supported error;
  `1537a0a8b2:tools/server/server-context.cpp:5228-5249`, `server-chat.cpp:651`). Not in
  the README — route-only documentation.
- **NInfer state:** none; audio input is an explicitly rejected capability
  (`3e0ff840:docs/serving.md:132-137` "audio/file input").
- **Port shape + cost:** requires audio-capable model architectures (encoders), i.e. new
  mathematical architecture.
- **Boundary-crossing flag: yes** (new architectural family; the Qwen3.5 artifacts have no
  audio path). No verdict.

## Candidate 13 — LoRA adapters (`/lora-adapters`, per-request `lora`)

- **llama.cpp state:** adapters loaded via `--lora`/`--lora-scaled`, `GET/POST
  /lora-adapters` sets global scales at runtime, per-request `lora` field overrides
  (README §1199-1242, server-specific params).
- **NInfer state:** none; the `.ninfer` artifact is a fixed converted representation and
  the loader binds stored weights; there is no adapter format or runtime weight
  modification.
- **Port shape + cost:** new artifact/converter support (adapter format) + Engine runtime
  application — a model-capability change, not a serve gap.
- **Contract impact:** none on the three routes.
- **Verdict:** **no**. No product requirement for LoRA; the boundary list for this ticket
  (concurrency/preemption/QoS/multi-GPU) does not cover it, but it is plainly outside the
  one-fixed-artifact design.

## Candidate 14 — multi-model router + model download (`/models*`)

- **llama.cpp state:** router mode runs multiple models as child processes:
  `GET /models` (list with status/architecture/modalities), `POST /models/load`,
  `POST /models/unload`, `GET /models/sse` (lifecycle events), `POST /models` (download
  from HF/Docker cache), `DELETE /models`; `--models-dir`, `--models-max`, `--models-autoload`,
  presets (README §1812-2215).
- **NInfer state:** one resident model, selected by artifact path at startup
  (`3e0ff840:docs/serving.md:1-4`); model selection is a public alias override only
  (`--model-id`).
- **Boundary-crossing flag: yes** (one GPU, one resident model is a fixed product
  boundary; model download is converter/tooling territory). No verdict.

## Candidate 15 — server-side tool execution (`/tools`, MCP servers, `--tools`)

- **llama.cpp state:** `GET /tools` lists, `POST /tools` invokes built-in tools
  (`read_file`, `file_glob_search`, `grep_search`, `exec_shell_command`, `write_file`,
  `edit_file`, `get_info`), optional MCP servers over stdio (Cursor-compatible config),
  tool execution in host or container/SSH runtimes, `--ui-mcp-proxy` + `/cors-proxy` for
  browser-side MCP (README §347-389, §1804-1811, server-specific params;
  `1537a0a8b2:tools/server/server-tools.cpp`, `server-mcp.cpp`).
- **NInfer state:** NInfer renders tool definitions into the prompt and parses model
  output, but "NInfer does not execute tools" is a documented product decision; hosted,
  remote MCP, and server tools are rejected with field-specific 400s
  (`3e0ff840:docs/serving.md:628-633, 803-806`).
- **Port shape + cost:** an executor (file/shell/remote MCP) inside serve, with a security
  surface llama.cpp itself flags as "do not expose to untrusted environments".
- **Contract impact:** none on the three routes (NInfer's contract already names this
  rejection).
- **Verdict:** **no**. Crosses the documented product decision; the AGENTS.md trust model
  (local, single-owner) does not create a requirement for server-side execution, and the
  security posture change is a product decision, not a parity fix.

## Candidate 16 — embedded Web UI

- **llama.cpp state:** shipped by default (`--ui`, default enabled), served as static
  assets with `--ui-config` defaults, Service Worker, and the resumable-stream + `/props`
  + `/slots` endpoints exist largely to serve it
  (`1537a0a8b2:tools/server/server-http.cpp:376-477`, server-specific params).
- **NInfer state:** none; clients are API consumers (opencode etc.).
- **Port shape + cost:** a standalone UI project, not a serve-API port.
- **Verdict:** **no** (c). Outside NInfer's product surface; the endpoints that exist
  "for the UI" are assessed individually above.

## Candidate 17 — `/v1/models` model metadata (`meta`, `--alias`, `--tags`)

- **llama.cpp state:** `GET /v1/models` returns `object/list/data[]` with
  `owned_by: "llamacpp"` and a `meta` object (`vocab_type`, `n_vocab`, `n_ctx_train`,
  `n_embd`, `n_params`, `size`, `ftype`); `--alias` sets the advertised id, `--tags`
  informational tags (`1537a0a8b2:tools/server/server-context.cpp:4762-4784`, README
  §1245-1276, server-specific params).
- **NInfer state:** `GET /v1/models` returns `object/list/data[]` with
  `owned_by: "ninfer"` and `max_model_len` (a vLLM-style extension), plus
  `GET /v1/models/{id}`; the id is the artifact name or `--model-id` override
  (`3e0ff840:src/serve/openai_common.cpp:175-197`).
- **Port shape + cost:** add a `meta` object sourced from artifact metadata (parameter
  count, size, vocab, native context) to the existing model objects. **Small**.
- **Contract impact:** `/v1/models` only.
- **Verdict:** **defer** (c). Trivial to add; no client is blocked today (both shapes are
  discovery metadata).

## Candidate 18 — llama.cpp-specific sampling parameters

- **llama.cpp state:** the request schema accepts the full sampler zoo beyond the OpenAI
  set: `dry_multiplier/dry_base/dry_allowed_length/dry_penalty_last_n/
  dry_sequence_breakers` (DRY), `mirostat/mirostat_tau/mirostat_eta`, `xtc_probability/
  xtc_threshold`, `typical_p`, `dynatemp_range/dynatemp_exponent`,
  `adaptive_target/adaptive_decay`, `top_n_sigma`, `min_keep`, `repeat_last_n`,
  `repeat_penalty`, `samplers` (order override), `backend_sampling`
  (`1537a0a8b2:tools/server/server-schema.cpp`; README "Sampling params").
- **NInfer state:** the Engine owns the sampler chain and selects tuned defaults per
  architecture; the serve exposes `temperature`, `top_p`, `top_k (0..20)`, `min_p`,
  presence/frequency penalties, `seed`, and process-level overrides
  (`3e0ff840:docs/serving.md:91-112, 867-929`). Unknown top-level fields (including these)
  are ignored, so such requests already execute safely.
- **Port shape + cost:** each sampler is a new Engine Op with its own qualification
  contract; a large batch of Engine work whose Qwen tuned presets do not need
  (DRY/mirostat/XTC are CPU-inference-era samplers with no 5090 performance argument).
- **Verdict:** **no** (c). The ignore-behavior is already safe; porting the zoo is Engine
  work with no contract gap (none is an official field) and no measurable performance
  headroom for the deployed models.

## Candidate 19 — context/generation-control request fields

- **llama.cpp state:** `n_keep`, `n_discard`, `n_indent`, `n_cache_reuse`, `ignore_eos`,
  `echo` (legacy completions only), `t_max_prompt_ms`/`t_max_predict_ms`, `cache_prompt`,
  `return_tokens`, `response_fields`, `verbose`, `message_delimiters` (per-message
  checkpoint spans), `sse_ping_interval` (per-request)
  (`1537a0a8b2:tools/server/server-schema.cpp`, `server-context.cpp:4524`).
- **NInfer state:** none of these are accepted fields (unknown fields ignored). Context
  policy is Engine-owned: `--max-context` + YaRN, checkpoint/pressure system,
  `--no-prefix-reuse` process toggle (`3e0ff840:docs/serving.md:1076-1123`).
- **Port shape + cost (sub-items):**
  - `sse_ping_interval` (server flag): trivial global keep-alive configurability — **small**.
  - `ignore_eos`, `min_keep`: small sampler-policy toggles — **small–medium** each, low value.
  - `return_tokens`: expose committed token ids in the response — **small–medium**
    (Engine output channel; useful for debug/bench).
  - `t_max_predict_ms`: per-request generation deadline — **medium** (deadline enforcement
    inside the generation transaction).
  - `n_keep`/`n_discard`/`n_indent`/`n_cache_reuse`/`echo`: per-request context-shaping
    that NInfer's Engine policy deliberately does not expose — would collide with the
    checkpoint/pressure design.
  - `message_delimiters`: per-message KV checkpointing — an Engine context-cache feature,
    not a serve one (see Flagged for the map).
  - `response_fields`/`verbose`: response projection/debug — negligible value.
- **Verdict:** **no** for `n_keep`/`n_discard`/`n_indent`/`n_cache_reuse`/`echo`/
  `response_fields`/`verbose` (c; Engine-owned context policy, no consumer). **defer** for
  `sse_ping_interval` flag, `ignore_eos`, `min_keep`, `return_tokens`, `t_max_predict_ms`
  (c; small real QoL items if a consumer appears).

## Candidate 20 — tool-calling and reasoning request fields

- **llama.cpp state:** `generation_prompt` (client-supplied prefill), `parse_tool_calls`
  (toggle tool-call parsing), `parallel_tool_calls` (template-capability gated),
  `reasoning_format` (`none|deepseek|deepseek-legacy`, also `--reasoning` CLI),
  `reasoning_budget_tokens` + budget start/end tags and message (request-level thinking
  budget), `chat_template_kwargs`, `preserve_thinking`-equivalent `--reasoning-preserve`
  (`1537a0a8b2:tools/server/server-schema.cpp`, `server-common.cpp:1318-1356`, README
  §1308-1445, server-specific params).
- **NInfer state:** `chat_template_kwargs`, `enable_thinking`, `preserve_thinking`,
  `reasoning_effort`, assistant `reasoning_content` history are supported; request-level
  thinking budget is not a chat field (server `--default-thinking-budget` + Anthropic
  `thinking.budget_tokens` are the budget surfaces); tool-call parsing always runs;
  assistant prefill exists on the Anthropic route only
  (`3e0ff840:docs/serving.md:102-147, 299-335, 773-775`).
- **Port shape + cost:** `reasoning_budget_tokens` as a chat-request field mapping onto
  the existing budget machinery — **small**. Chat assistant-prefill behavior is
  template-driven already (a final assistant message renders into the prompt), so
  `generation_prompt` adds little — **small** if wanted. `parse_tool_calls:false` and
  `reasoning_format` would fork NInfer's parser/format contract into per-request variants
  — **medium** with real contract complexity.
- **Verdict:** **defer** for `reasoning_budget_tokens` (c; small, reuses budget path).
  **defer** for chat assistant-prefill/`generation_prompt` (c; mostly already
  template-native). **no** for `parse_tool_calls`/`reasoning_format` (c; NInfer's
  fail-closed single-parser contract is deliberate).

## Candidate 21 — response extension fields

- **llama.cpp state:** `system_fingerprint` (build info) on chat/completions responses,
  `__verbose` debug object (gated by `verbose`), `post_sampling_probs`/`n_probs`
  per-token probability arrays, `timings` (gated by `--perf`-class stats)
  (`1537a0a8b2:tools/server/server-task.cpp:399-420, 264-300`).
- **NInfer state:** `timings` is a documented llama.cpp-compatible extension on every
  successful chat response, plus `timings_per_token` and `return_progress`; `usage`
  includes `cached_tokens` and `reasoning_tokens` details; `logprobs: null` and
  `refusal: null` are emitted; no `system_fingerprint`, `__verbose`, or prob arrays
  (`3e0ff840:docs/serving.md:344-403, 337-342`).
- **Verdict:** **no** for `__verbose`/`post_sampling_probs` (c; debug-only, logprobs
  candidate 2 covers the real need). **no** for `system_fingerprint` (c; the JSONL
  `server_start` event carries build identity). NInfer is at or ahead of llama.cpp on the
  timing/usage observation surface — noted so the coordinator does not re-derive it.

## Candidate 22 — deployment knobs (CORS, auth, prefix, SSL, timeouts, media path, idle sleep)

- **llama.cpp state:** `--cors-origins` (default reflects Origin with credentials),
  `--cors-methods`, `--cors-headers`, `--cors-credentials`; `--api-key` as a comma list
  plus `--api-key-file`; `--api-prefix`; `--ssl-key-file/--ssl-cert-file` (SSL build);
  `--timeout` (socket read/write, default 3600 s); `--sse-ping-interval` (default 30 s);
  `--media-path` (local file media via `file://`); `--sleep-idle-seconds` (unload model +
  KV after idle, reload on next task) (README §389-405, server-specific params, §2232).
- **NInfer state:** `--cors` adds permissive `*` headers + unauthenticated OPTIONS
  preflight (off by default); single `--api-key` accepted as bearer **or** `x-api-key`;
  no path prefix; plain HTTP; fixed 5 s SSE keep-alive + 15 s `TCP_USER_TIMEOUT`;
  media via HTTP(S)/data URIs only; model resident by design
  (`3e0ff840:src/serve/http_server.cpp:340-351`, `docs/serving.md:78-85, 851-861`).
- **Port shape + cost:** CORS configurability, multi-key auth, `--api-prefix`, socket
  timeout — **small** each (httplib header/config work). SSL/TLS — **small–medium**
  (httplib TLS build + flags; deployments sit behind caddy anyway). `--media-path` —
  **small** but a security posture change (server-local file fetch). `--sleep-idle-seconds`
  — **large** and contrary to the resident-model design (Engine unload/reload path).
- **Verdict:** **defer** for CORS configuration, multi-key auth + key file, `--api-prefix`,
  socket timeout, SSL (c; small deployment-parity items, no contract gap). **no** for
  `--media-path` (c; NInfer's HTTP/data-URI-only media acquisition is deliberate) and for
  sleep-on-idle (c; a single resident model is the product; unloading the model to save
  RAM has no local value on the trusted 5090 box).

## Candidate 23 — GCP/Vertex AI compatibility and TypeSafe `/v1/systemone`

- **llama.cpp state:** when `AIP_MODE=PREDICTION`, the server honors `AIP_HTTP_PORT` /
  `AIP_HEALTH_ROUTE` / `AIP_PREDICT_ROUTE` and serves the Vertex `/predict` contract
  (no streaming on the predict route; `1537a0a8b2:tools/server/server-http.cpp:81-118,
  814-838`). `POST /v1/systemone` is a TypeSafe "System One" compatible endpoint
  (README §1668-1803).
- **NInfer state:** none; NInfer is a local, single-owner deployment (trusted local models,
  local workflow per `AGENTS.md`), not a cloud-platform target.
- **Verdict:** **no** for both (c). Porting the Vertex container contract or a niche
  vendor protocol is ecosystem work for deployment targets NInfer explicitly is not.

## Protocol behavior notes (no action items)

Verified deltas where the comparison goes the other way or is equivalent — recorded so the
map does not treat them as gaps:

- **Error shape:** llama.cpp returns `{"error": {"code": <http number>, "message", "type"}}`
  (README "API errors"); NInfer returns `{"error": {"message", "type", "param", "code":
  <string>}}` (`3e0ff840:src/serve/openai_common.cpp:188-196`). Both are OpenAI-shaped;
  NInfer's string `code` is the vLLM convention and is deliberately more specific
  (field + stable per-class codes). No port.
- **Overload semantics:** llama.cpp's task queue is unbounded
  (`1537a0a8b2:tools/server/server-queue.cpp:39-41`, no capacity check; "deferred" is a
  temporary re-queue mechanism) — it never returns 429. NInfer is bounded FIFO with
  429 `server_overloaded` and 503 `request_queue_timeout`
  (`3e0ff840:docs/serving.md:1062-1068`). NInfer's behavior is the safer contract.
- **Rate limiting:** neither server has per-client rate limiting; nothing to port.
- **Constrained decoding:** both expose `grammar` + `response_format` json_object/
  json_schema with schema-to-grammar conversion; NInfer's fail-closed allowlist and error
  codes are stricter (docs/serving.md "Constrained output";
  `3e0ff840:src/serve/constraint_contract.cpp`). llama.cpp's `json_schema` + `grammar`
  conflict error and empty-schema-default-to-object behavior are matched in spirit.
- **Anthropic parity:** NInfer is ahead where it counts — opaque thinking `signature` +
  `signature_delta` (llama.cpp emits thinking blocks with no signature,
  `1537a0a8b2:tools/server/server-task.cpp:775-830`), block-level `cache_control`
  breakpoints, 529/504 overload codes, `request-id` header discipline. Both normalize the
  Claude Code `x-anthropic-billing-header` block (llama.cpp:
  `1537a0a8b2:tools/server/server-chat.cpp:326-349`).
- **OpenAI Responses:** NInfer implements the typed-Item core with a local store,
  `previous_response_id`, `input_items`, semantic SSE, and counting; llama.cpp converts
  Responses→chat and explicitly rejects `previous_response_id`
  (`1537a0a8b2:tools/server/server-chat.cpp:10-12`). NInfer ahead.
- **Prompt caching:** llama.cpp's `cache_prompt` per-request toggle + automatic KV reuse
  vs NInfer's OpenAI/Anthropic cache-hint surface + Engine checkpoint reuse — different
  designs, NInfer's is the protocol-native one. No port.

---

## Summary table

Axis: (a) exposed contract/protocol gap on the three routes > (b) measurable 5090
performance > (c) QoL/ecosystem parity. Cost: small = serve-side or trivial; medium =
small Engine change; large = Engine redesign or new architecture.

| Candidate | Verdict | Axis | One-line rationale |
|---|---|---|---|
| `logprobs`/`top_logprobs` | **port** | a | Official OpenAI field NInfer 400s; bounded Engine top-k-gather reusing existing logits. |
| Real-time control `reasoning_end` | **port** | c | Top QoL for thinking UX; reuses the existing thinking-budget control-suffix machinery. |
| Prometheus `/metrics` | **port** | c | The one observability format the deployment stack would consume; counters already exist. |
| `n` > 1 completions | defer | a | Real OpenAI field gap, but clients can issue n requests and the Engine fan-out costs medium for moderate value. |
| Resumable streaming `/v1/stream` | defer | a | Real resilience gap but requires redefining disconnect=cancel across all three routes; large lifecycle work. |
| `logit_bias` | defer | a | Official field, lowest frequency of the (a) set, highest Engine cost (per-request bias tables + graph capture). |
| Tokenizer endpoints + chat `input_tokens` | defer | c | Cheap serve-side additions (tokenizer/renderer/counting path in-process) but no current consumer. |
| `GET /props` | defer | c | Small introspection JSON; overlaps JSONL `server_start` and has no UI consumer. |
| `POST /props` | no | c | Runtime mutation of global properties contradicts startup-fixed design and CUDA-graph capture stability. |
| `GET /slots` | defer | c | JSONL `throughput` already snapshots scheduler state; no consumer. |
| Slot save/restore/erase | no | c | Large Engine serialization work for a file workflow NInfer's prompt-identity reuse makes redundant. |
| `/completion` + `/v1/completions` legacy | no | a/c | OpenAI marks the endpoint legacy (guide, 2026-10-03); no consumer in the deployment, new prompt-mode contract to maintain. |
| `/infill` | no | c | FIM template family with no product requirement. |
| `/lora-adapters` | no | c | No LoRA requirement; one fixed `.ninfer` artifact is the design. |
| Embeddings + reranking | FLAG | — | Pooling head = new mathematical architecture (explicit product change). |
| Audio transcription | FLAG | — | Audio model family = new mathematical architecture. |
| Multi-model router + download | FLAG | — | Violates one-GPU/one-resident-model boundary; download is converter territory. |
| `/tools` + MCP executors | no | c | NInfer's "does not execute tools" is a documented product decision; executor = security posture change. |
| Web UI | no | c | Standalone UI project, outside the serve-API product. |
| `/v1/models` `meta` + aliases | defer | c | Trivial artifact-metadata exposure; discovery metadata only. |
| Sampler zoo (DRY/mirostat/XTC/typical/dynatemp/adaptive/top_n_sigma/…) | no | c | Engine Op zoo with no contract gap (none official) and no 5090 performance argument; unknown fields already safely ignored. |
| Context fields (`n_keep`, `n_discard`, `n_indent`, `n_cache_reuse`, `echo`) | no | c | Per-request context shaping collides with Engine-owned checkpoint/pressure policy. |
| `sse_ping_interval` flag, `ignore_eos`, `min_keep`, `return_tokens`, `t_max_predict_ms` | defer | c | Small real QoL toggles; build if a consumer appears. |
| `message_delimiters` | FLAG | — | Per-message KV checkpointing is an Engine context-cache feature, not a serve surface. |
| `reasoning_budget_tokens` (chat request field) | defer | c | Small mapping onto the existing budget machinery. |
| Chat assistant prefill / `generation_prompt` | defer | c | Mostly already template-native; explicit prefill is Anthropic-route only today. |
| `parse_tool_calls` / `reasoning_format` | no | c | Per-request parser/format forks contradict the deliberate single-parser fail-closed contract. |
| `system_fingerprint`, `__verbose`, `post_sampling_probs` | no | c | Debug/identity fields; JSONL and the logprobs candidate cover the real needs. |
| CORS config, multi-key auth, `--api-prefix`, socket timeout, SSL | defer | c | Small deployment-parity knobs; no contract gap. |
| `--media-path` (local file media) | no | c | HTTP/data-URI-only media acquisition is NInfer's deliberate security posture. |
| `--sleep-idle-seconds` | no | c | Unloading the model contradicts the single-resident-model product; no local value. |
| GCP/Vertex compat, `/v1/systemone` | no | c | Cloud-platform/niche-vendor contracts for a local single-owner product. |
| `parallel_tool_calls:false` with tools | no (by design) | a | NInfer rejects with a named code; llama.cpp template-gates — documented fail-closed choice, not an oversight. |

## Flagged for the map (boundary-crossing + out-of-family observations)

**Boundary-crossing (route to Out of scope, no verdict):**

1. **Embeddings + reranking endpoints** — pooling head is a new mathematical
   representation; requires an explicit product change (`AGENTS.md`).
2. **Audio transcription** — audio-capable model architectures; new mathematical
   family.
3. **Multi-model router + model download/delete** — one GPU, one resident model;
   startup-fixed concurrency is also in tension with llama.cpp's per-slot model
   (`--parallel`, `--kv-unified`, continuous batching).
4. **`message_delimiters` per-message KV checkpointing** — an Engine context-cache
   mechanism (llama.cpp's `--ctx-checkpoints`/`--checkpoint-min-step` family); if ever
   wanted it is an Engine ticket, not a serve port.

**Out of this family (serve/API) — other llama.cpp surfaces seen at `1537a0a8b2` worth a
map glance, each owned by another area:**

- **Speculative-decoding zoo** (`--spec-type` draft-simple/eagle3/mtp/dflash/dspark,
  ngram-*, synthetic acceptance benchmarking, `--spec-draft-p-split/p-min`) — engine
  family; NInfer's MTP/DFlash/DFlash2 are the productized subset.
- **Sampler chain + `--sampling-seq`** — engine family (see candidate 18's rationale).
- **KV management flags** (`--kv-unified`, `--kv-unified-per-slot`, `--cache-ram`,
  `--cache-idle-slots`, `--context-shift`, `--swa-full`) — engine/context-cache family.
- **Model acquisition** (`--hf-repo`, `--docker-repo`, `--model-url`, `--offline`,
  `--models-preset`) — converter/tooling family.
- **Chat-template zoo + `--skip-chat-parsing` + `--prefill-assistant`** — frontend
  family; NInfer's vendored chat-parsing stack is the productized path.
- **`--slot-prompt-similarity`** (route a request to a slot with a similar prompt) —
  scheduler family; NInfer's admission reserves exact entitlements instead.
- **`--control-vector`** (activation steering) — engine/model-capability family.

**Dead ends / could-not-verify:** nothing material. `tools/server/README.md` at this
commit does not document the transcription or resumable-stream endpoints (route-table
verification was done in source instead). The ticket's "mid-suffix" and
"completion-details" hints match no feature at `1537a0a8b2` (see header note).
