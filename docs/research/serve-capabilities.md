# ninfer-yarn serve: capability inventory for opencode compatibility

Wayfinder map #139, ticket #141 — the capability side of the opencode ⇄ ninfer-yarn
compatibility matrix. The requirement side (what opencode v2.0.22 actually sends and
checks) is in `docs/research/opencode-requirements.md` on branch
`research/opencode-requirements` (ticket #140, commit 92c7be33); "companion doc
§N" below cites its sections.

**Sources of record (read this session, 2026-10-02)**

- Repository `kido5217/ninfer-yarn` at `master` commit `8a0d16bf`
  (primary checkout, clean tree). All `file:line` citations are against this commit.
  Provenance tags: **[code]** source read this session, **[test]** test suite pin read
  this session, **[docs]** repo documentation, **[probe]** live read-only HTTP probe of
  the running unit this session, **[config]** `nixos-configs` deployment file read this
  session, **[log]** live request-log / journal evidence read this session.
- Live unit `ninfer-yarn-qwen38-27b` on `127.0.0.1:8827` (behind caddy at
  `ai.kido.ws/v1`), probed with GETs only — no completions were generated.
- Deployment config: `nixos-configs/users/kido/llm.nix` (lines 408-579 cover both
  qwen38 units; the yarn unit at 483-579).
- Request log: `/home/kido/trash/temp/ninfer-yarn-log.jsonl` (schema v22; 31.6 MB,
  covering 2026-09-29 21:18:59 UTC → 2026-10-02 20:36:48 UTC, 71.3 h).
- Client config: `~/.config/opencode/opencode.jsonc` (provider `ai.kido.ws`).

---

## 0. Deployment reality

The serving unit is `ninfer-yarn-qwen38-27b` (user systemd, manual start,
`Restart=always`, `RestartSec=5`). **[config]** llm.nix:483-579; **[probe]** the running
process (`systemctl --user status`, 2026-10-02 23:3x MSK, active since 22:51 MSK, 42 min
at probe time) shows the exact argv, confirming the config is live:

| Flag | Value | Note |
|---|---|---|
| binary | `ninfer-yarn-serve` from `pkgs.ninfer-yarn` (store path `…/f0rpzl7lf4n9rxn3vfx34b465ihmldx6-ninfer-yarn-r3`) | release **r3** channel build |
| artifact | `/home/kido/trash/ai/models/hf/hub/models--neroued--Qwen3.8-27B-nvfp4-NInfer/snapshots/f0b43ad436b9fa8142c6ed6647c470a6fe409484/qwen3_8_27b_nvfp4.ninfer` | Qwen3.8-27B NVFP4, native window 262144 |
| `--host` / `--port` | `127.0.0.1` / `8827` | caddy routes `ai.kido.ws` → 8827 (128 MiB body cap, unbuffered SSE) |
| `--model-id` | `Qwen3.8-27B` | public OpenAI model alias; requests must use exactly this id |
| `--max-context` | **446902** | **above** the artifact's native 262144 → the fork's YaRN context extension is active for the whole window **[config]** llm.nix:510-514; **[probe]** `/v1/models` reports `max_model_len: 446902` |
| `--default-max-tokens` | 32768 | output cap when a request sends neither max field **[config]** llm.nix:515-516; incident comment llm.nix:418-419 |
| `--kv-dtype` | **nvfp4** | the unit's measured-config comment (llm.nix:409-411) names int8-group64 as the 2026-08-28 performance leader, but the flag actually passed is nvfp4 |
| `--max-concurrency` | 4 | active lanes (C) |
| `--max-pending-requests` | 64 | FIFO pending bound (P) |
| `--device-state-slots` | 0 | device checkpoint cache off; checkpoints stay on Host |
| `--host-state-slots` | 32 | pinned Host StateImage capacity (≈4.6 GiB) |
| `--host-kv-mib` | 24576 | pinned Host KV |
| `--request-log-jsonl` | `/home/kido/trash/temp/ninfer-yarn-log.jsonl` | append-mode JSONL, schema v22 |
| `--pending-timeout-ms` | 600000 | 10 min admission+preparation deadline |
| `--spec` / `--draft-tokens` | `mtp` / 4 | MTP4 speculative decoding |
| `--lm-head-draft` | set | optimized proposal head |
| `--preserve-thinking` | set | reasoning retention on by default |
| `--vision` | **set** | media input enabled (≈0.70 GiB fixed) |
| `--chat-template` | `/nix/store/safqpi3cpb5104xi7lvch0rm9mi26s0a-chat_template.jinja` | **froggeric v22.5 override** — not the artifact-baked template (the A/B partner unit `ninfer-qwen38-27b` uses the baked one) **[config]** llm.nix:479-482, 552-555 |
| `--max-long-anchors-per-continuation` | 4 | |
| `--max-shared-prefixes` | 0 | shared-prefix catalog off (2026-09-30 retune) |
| `--media-cache-mib` | 2048 | |
| `--api-key` | **not set** | serve is **unauthenticated** — consistent with opencode posting without an `Authorization` header (companion doc §1.3) |

Notable absences (all serve defaults apply): no `--kv-capacity` (follows
`--max-context` 446902), no `--no-thinking` (thinking on by default), no
`--default-thinking-budget` (no thinking cap), no `--temperature`/`--top-p`/`--seed`
process overrides (model-registered sampling defaults apply), no `--max-request-mib`
(384 MiB default), no `--cors`, no `--no-cuda-graph`/`--no-prefix-reuse` (both on).

**Chat template actually in use.** The froggeric v22.5 Jinja template
(`users/kido/chat-template/chat_template.jinja`, `template_version
"qwen3.8-froggeric-v22.5"`) is rendered by the fork's vendored llama.cpp Jinja engine.
Its behavior that shapes opencode traffic:

- Tool-call wire format defaults to **XML** (`tool_call_format` variable, default 'xml'; lines 1-2, 194-209): a generated call is a </think>-delimited block containing a <think>-delimited reasoning prefix and a function block with parameter blocks per argument; assistant history and tool results are rendered in the same format (lines 337-410, 412-449). The model is thus *prompted* in this XML dialect by the template, while the HTTP wire to the client stays OpenAI shape.
- The generation prompt opens a thinking block when thinking is on (lines 453-459);
  with thinking off it inserts an empty closed thinking block.
- Media placeholders are `vision_start`/`image_pad`/`vision_end` tokens (lines 99-116).

Tool-call parsing therefore happens **serve-side**: the model is prompted in the
froggeric XML format, and the vendored llama.cpp chat stack
(`src/models/qwen3_5/frontend/chat_parse_core.cpp`) parses the generated
`tool_call` regions into OpenAI `tool_calls` objects. Markup that fails to parse is
demoted to ordinary content (see §3b). The client-visible wire is always OpenAI
shape; the XML is not exposed on success.

---

## 1. Endpoints and protocol coverage

**Route table.** Registered in `src/serve/http_server.cpp:335-478` **[code]**, matching
the documented table at docs/serving.md:55-70 **[docs]**:

| Method + path | Behavior |
|---|---|
| `GET /health` | Engine readiness. 200 `{"status":"ok"}` while available; 503 `{"status":"unavailable"}` after an Engine-wide failure; never affected by queue saturation. Unauthenticated. **[code]** http_server.cpp:428-433; **[probe]** returned `{"status":"ok"}` [200] |
| `GET /v1/models` | One model object: id = `--model-id` alias, `owned_by: "ninfer"`, `max_model_len` = `--max-context`. **[probe]** `{"data":[{"created":1790973229,"id":"Qwen3.8-27B","max_model_len":446902,"object":"model","owned_by":"ninfer"}],"object":"list"}` |
| `GET /v1/models/{id}` | Same object for the configured alias; any other id → 404 `model_not_found`. **[probe]** `/v1/models/wrong-id` → 404 `{"error":{"code":"model_not_found","message":"model 'wrong-id' not found","param":null,"type":"invalid_request_error"}}` |
| `POST /v1/chat/completions` | OpenAI Chat Completions (streaming + non-streaming) |
| `POST /v1/responses` | OpenAI Responses Core (typed Items, local store, semantic SSE) |
| `POST /v1/responses/input_tokens` | prompt-token count, no generation |
| `GET /v1/responses/{id}` / `DELETE` / `GET …/input_items` | local response-store retrieval, deletion, normalized input Items |
| `POST /v1/responses/{id}/cancel` | exists to **explicitly fail** (no background execution) **[docs]** serving.md:706 |
| `POST /v1/responses/compact` | exists to **explicitly fail** with `compaction_not_supported` **[docs]** serving.md:707 |
| `POST /v1/messages` | Anthropic Messages (aggregate + SSE) |
| `POST /v1/messages/count_tokens` | tokenizer count, no generation |

**Not present** (opencode never calls these on the `@ai-sdk/openai-compatible`
package — companion doc §1.6/§1.7): `/v1/embeddings` (caddy routes that to a separate
CPU embeddings backend), `/v1/completions` (plain completions), `/v1/chat/…` beyond
completions, any `GET /v1/responses/{id}/cancel`-style extra. There is no
`/v1/chat/completions/{id}` retrieval (Chat Completions are not stored; `store: true`
is rejected, §2).

**Unknown routes.** A GET on any unregistered path returns **HTTP 404 with an empty
body** — no JSON error envelope. **[probe]** `GET /v1/unknown-route` → 404, empty
body; `GET /v1/chat/completions` (wrong method) → 404, empty body. The serve's error
handler only re-renders (a) 413s into a JSON `request_too_large` body and (b)
`/v1/messages`-prefixed 404s into the Anthropic envelope (http_server.cpp:158-184,
**[code]**; pinned by tests/test_http_error_handler.cpp:87-132, **[test]**). Any other
client hitting a wrong OpenAI path sees a bare 404 — relevant only to non-opencode
clients, since opencode's sole route (`POST /chat/completions`) is registered.

**Headers.** Every OpenAI-path response (including errors) carries a unique
`x-request-id` (`req_<32 hex>`); Anthropic paths carry `request-id` (`req_…`) with an
Anthropic-shaped body **[code]** http_server.cpp:35-39, 144-156, openai_common.cpp:227-237;
**[docs]** serving.md:75-76. All three generation SSE endpoints emit a `: keep-alive`
comment every 5 s of protocol silence, and accepted connections use TCP keepalive +
15 s `TCP_USER_TIMEOUT`, so a dead peer is cancelled within ~20 s
**[docs]** serving.md:78-85.

**Authentication.** With `--api-key` unset (the deployment) **every endpoint is
open** — the pre-routing auth check is skipped (http_server.cpp:353-357, **[code]**;
**[probe]** `/v1/models` answered with no `Authorization` header). If a key were set,
both `Authorization: Bearer <key>` and `x-api-key: <key>` are accepted
(case-insensitive scheme, whitespace-tolerant; matches_bearer_credential,
http_server.cpp:186-214, **[code]**) and failures render 401 `invalid_api_key`
"missing or invalid API key" in the target protocol's shape (http_server.cpp:364-377,
**[code]**). `GET /health` and OPTIONS preflight stay unauthenticated.

**CORS.** `--cors` off (not passed) → no CORS headers; only matters to browsers.

**Protocol coverage summary for opencode:** of the protocols ninfer-yarn serves,
opencode (custom provider, `aisdk:@ai-sdk/openai-compatible`) exercises **exactly one**:
`POST /v1/chat/completions` with `stream: true`. The OpenAI Responses Core and
Anthropic Messages surfaces exist and are complete for their respective protocols but
are not on opencode's critical path.

---

## 2. OpenAI Chat Completions — field-by-field

Parser: `src/serve/openai_chat_request.cpp` (**[code]**); rejection/neutral pins:
`tests/test_openai_schema.cpp:143-195` (**[test]**); docs: serving.md:87-147
(**[docs]**). A 400 body has the shape
`{"error":{"message","type":"invalid_request_error","param","code"}}`
(**[code]** openai_common.cpp:203-208; **[test]** test_http_error_handler.cpp:36-56).

### 2.1 Output limit — the `max_tokens` surprise

`parse_output_limit` (openai_chat_request.cpp:949-963, **[code]**):

1. `max_completion_tokens` is read **first**; if present and a nonnegative integer it
   wins and marks the request `output_tokens_explicit`.
2. `max_tokens` is read only as a fallback (legacy spelling).
3. Neither present → `--default-max-tokens` (32768 in this deployment).
4. Negative → 400 "`<field>` must be nonnegative"; non-integer → 400 "must be an
   integer"; out of int range → 400 "is out of range"
   (request_validation.cpp:21-38, **[code]**).

**When opencode sends both** (computed `max_completion_tokens` + the fixed
`body.max_tokens: 32768` overlay — companion doc §2.2), **`max_completion_tokens`
wins**. Pinned by tests/test_openai_schema.cpp:87-105 ("max_completion_tokens wins and
explicitness stays in envelope": `max_completion_tokens: 48` + `max_tokens: 9` →
effective 48). **[live]** every one of the 3,284 `request_start` records in the 71 h
log window carries `requested_output_tokens_source: "client"` — the overlay is always
on the wire, and the serve records the client-set value.

`max_tokens: 0` is legal: "zero performs prompt processing without generation"
(serving.md:109-110, **[docs]**; engine immediate-submission path engine.cpp:311-329,
**[code]**; pinned test_openai_schema.cpp:189-193, **[test]**).

### 2.2 Fields honored (opencode-relevant first)

| Field | Behavior | Citation |
|---|---|---|
| `model` | required non-empty string; must **exactly equal** the public alias `Qwen3.8-27B` — mismatch → **404** (not 400) `model_not_found`, `param: "model"` | openai_common.cpp:216-225 **[code]**; openai_chat_request.cpp:973-977 **[code]** |
| `messages` | required non-empty array; roles `system`, `developer`, `user`, `assistant`, `tool`, plus legacy `function` (mapped to tool) | openai_chat_request.cpp:51-58, 643-660 **[code]**; serving.md:104 **[docs]** |
| string / part-array content | string; parts `text`, `refusal` (assistant only), `image_url` (user **or** tool — Copilot screenshot compat), `video_url` (user only, NInfer extension). Adjacent parts preserved without separators; empty wire content stays an empty turn | openai_chat_request.cpp:388-446 **[code]**; serving.md:105-108 **[docs]** |
| `stream` | bool, default false; `true` → SSE | openai_chat_request.cpp:933 **[code]** |
| `stream_options.include_usage` | bool; final empty-`choices` usage chunk (serving.md:319-324) | openai_chat_request.cpp:934-941 **[code]** |
| `max_tokens` / `max_completion_tokens` | see §2.1 | §2.1 |
| `temperature`, `top_p`, `presence_penalty`, `frequency_penalty` | finite numbers → Engine sampling; absent → registered defaults (Qwen3.8-27B: thinking `1.0/0.95/20/0/0`, non-thinking `0.7/0.80/20/0/1.5`; request > process-flag > registered; `--greedy` forces 0 last) | openai_chat_request.cpp:860-866 **[code]**; serving.md:906-911 **[docs]** |
| `seed` | signed integer, maps modulo 2^64 | openai_chat_request.cpp:43-49, 866 **[code]**; test pin test_openai_schema.cpp:93, 106-107 **[test]** |
| `top_k` (0..20), `min_p` (0..1) | vLLM/SGLang sampler extensions, honored | openai_chat_request.cpp:868-871 **[code]** |
| `stop` | ≤4 non-empty strings; applied to **both** reasoning and answer output | openai_chat_request.cpp:840-858, 358-373 in translate.cpp **[code]**; serving.md:113 **[docs]** |
| `tools` | array of `{type:"function", function:{name, description, parameters, strict}}`; name must match `[A-Za-z0-9_-]{1,64}`; **`strict: true` → 400 `strict_tools_not_supported`**; missing/null `parameters` → `{type:object, properties:{}}`; tool schemas are **never** enforced via constrained decoding | openai_chat_request.cpp:674-723 **[code]**; serving.md:124-126, 253-255 **[docs]** |
| `tool_choice` | `"auto"` / `"none"` honored; `"required"` → 400 `tool_choice_not_supported`; `{type:"function"}` named choice → 400 `tool_choice_not_supported`; `{type:"allowed_tools", mode:"auto", tools:[…]}` filters the effective set (mode `"required"` rejected, unknown names rejected); `{type:"custom"}` → 400 `tool_type_not_supported` | openai_chat_request.cpp:725-823 **[code]** |
| `parallel_tool_calls` | `true`/absent fine; `false` **with tools enabled** → 400 `parallel_tool_calls_not_supported` (NInfer cannot guarantee single-call) | openai_chat_request.cpp:825-838 **[code]** |
| assistant `tool_calls` history | `{id, type:"function", function:{name, arguments:string}}`; id+string arguments required | openai_chat_request.cpp:448-478 **[code]** |
| assistant `reasoning_content` / `reasoning` | both accepted as aliases; conflicting non-equal values → 400 `conflicting_template_option`; replayed into the template's reasoning slot | openai_chat_request.cpp:499-520 **[code]**; serving.md:130 **[docs]** |
| tool message `name` | string accepted as an **ignored** compatibility hint (clients that mirror the function name); non-empty `name` on other roles → 400 `message_name_not_supported` | openai_chat_request.cpp:522-534 **[code]**; serving.md:223-227 **[docs]** |
| `reasoning_effort` | string; accepted values `none|minimal|low|medium|high|xhigh|max` → passed to the template (`none` requests disabled thinking; conflicting `enable_thinking` → 400 `conflicting_template_option`). froggeric v22.5 maps `high/xhigh/max/ultracode/extreme` → xhigh effort and `minimal/low` → low (template lines 19-30, 83-87) | openai_chat_request.cpp:917-930 **[code]**; serving.md:311-313 **[docs]**; chat_template.jinja:19-30, 83-87 **[config]** |
| `enable_thinking`, `preserve_thinking` | top-level or in `chat_template_kwargs`; conflicting aliases → 400 `conflicting_template_option`; request values override server flags | openai_chat_request.cpp:888-915 **[code]**; serving.md:315-317 **[docs]** |
| `chat_template_kwargs` | JSON object passed to the template; messages/tools/mode/tokenizer not overridable | openai_chat_request.cpp:896-903 **[code]**; serving.md:287-290 **[docs]** |
| `grammar` | GBNF text (≤64 KiB) → Engine constraint, compiled at submit (see §2.5) | openai_chat_request.cpp:243-257 **[code]** |
| `response_format` | `{"type":"text"}` (no constraint), `{"type":"json_object"}`, `{"type":"json_schema"}` (wrapper or bare schema; see §2.5); any other type → 400 `response_format_not_supported` | openai_chat_request.cpp:262-283 **[code]** |
| image `detail` | omitted or `"auto"` only; `low`/`high` → 400 `image_detail_not_supported` | openai_chat_request.cpp:358-370 **[code]**; serving.md:108 **[docs]** |
| prompt-caching fields | `prompt_cache_key` (≤64 chars), `safety_identifier` (≤64), `user` (any length), `prompt_cache_retention` (`in_memory`/`24h`), `prompt_cache_options` (`mode: implicit|explicit`, `ttl: 30m`), per-part `prompt_cache_breakpoint: {mode:"explicit"}` — all accepted as **optimization hints** translated into ≤4 shared-prefix write candidates; `prompt_cache_key` is not a session key | openai_common.cpp:41-178 **[code]**; serving.md:430-450 **[docs]** |
| `n` | only `1`; `n≥2` → 400 `n_not_supported` | openai_chat_request.cpp:873-879 **[code]** |
| `repetition_penalty` | accepted only at the neutral value `1` | openai_chat_request.cpp:312-323 **[code]**; serving.md:229-231 **[docs]** |
| `mm_processor_kwargs` | accepted only when empty or all-null | openai_chat_request.cpp:327-340 **[code]** |
| `timings_per_token`, `return_progress` | llama.cpp-compatible observations (cumulative timings per chunk; prefill progress chunks) | openai_chat_request.cpp:944-947 **[code]**; serving.md:358-385 **[docs]** |
| `store` | `false` accepted (and the only legal value); `true` → 400 `store_not_supported` (Chat Completions are not stored; Responses are the stored route) | openai_chat_request.cpp:181-189 **[code]** |
| `audio` (text-only config), `prediction`, `service_tier`, `metadata`, `user` | semantically neutral — accepted without effect | openai_chat_request.cpp:143-147 region + test pins test_openai_schema.cpp:165-187 **[code/test]** |

### 2.3 Fields rejected (exact codes)

All 400 `invalid_request_error` with the named `param`; pinned by
test_openai_schema.cpp:143-164 **[test]** and the parser **[code]**:

| Field / value | Code |
|---|---|
| non-zero `logit_bias` | `logit_bias_not_supported` |
| `logprobs: true`, non-zero `top_logprobs` | `logprobs_not_supported` |
| non-text `modalities` entry | `modality_not_supported` |
| `web_search_options` | `web_search_not_supported` |
| `moderation` | `moderation_not_supported` |
| `verbosity` ≠ `medium` | `verbosity_not_supported` |
| `store: true` | `store_not_supported` |
| non-empty legacy `functions`, object-form `function_call` | `legacy_tools_not_supported` |
| non-`auto`/`none` `tool_choice` (`required`, named function, custom) | `tool_choice_not_supported` / `tool_type_not_supported` |
| `parallel_tool_calls: false` with tools | `parallel_tool_calls_not_supported` |
| `strict: true` on a tool | `strict_tools_not_supported` |
| non-function tool type | `tool_type_not_supported` |
| `n` ≠ 1 | `n_not_supported` |
| `response_format` other type | `response_format_not_supported` |
| `structured_outputs`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` | `constrained_decoding_not_supported` (explicit rejection, not silent ignore) |
| constraint + `tools` field (even `[]`) | `constrained_decoding_not_supported` |
| `grammar` + constraining `response_format` | `constrained_decoding_conflict` |
| oversized grammar/schema (>64 KiB) or schema nesting >64 | `constraint_too_large` |
| bad GBNF / unsatisfiable initial mask / DFlash backend | `grammar_invalid` / `json_schema_invalid` / (DFlash: `constrained_decoding_not_supported` naming the backend) |
| unsupported JSON-Schema keyword (incl. `pattern`, `allOf`, `oneOf`) | `json_schema_unsupported` naming the keyword |
| malformed schema document | `json_schema_invalid` |
| non-string role / unsupported role | 400 (no code) `unsupported role: …` |
| non-empty `message.name` on non-tool role | `message_name_not_supported` |
| conflicting `reasoning`/`reasoning_content`, or `enable_thinking` vs effort | `conflicting_template_option` |
| image `detail` ≠ auto/omitted | `image_detail_not_supported` |
| media request without `--vision` (not the case here; `--vision` is on) | `vision_disabled` |
| `prompt_cache_breakpoint` not `{mode:"explicit"}` | `invalid_cache_breakpoint` |

**Unknown top-level fields are ignored** — the parser never iterates unrecognized
members (openai_chat_request.cpp:967-998, **[code]**) and the test pins it:
`future_unknown_field` accepted alongside all neutral controls
(test_openai_schema.cpp:185-187, **[test]**). So opencode's `store: false`,
`reasoning_effort`, `stream_options`, `x-opencode-*` headers (headers are opaque to
the HTTP layer) all pass cleanly.

### 2.4 Sampling defaults that opencode's requests actually get

opencode sends no `temperature`/`top_p`/`seed` (companion doc §2.5), so every
agent step runs on the registered Qwen3.8-27B **thinking-mode** preset
`1.0 / 0.95 / 20 / 0 / 0` (serving.md:907-909, **[docs]**) — the live log confirms
(`request_start.sampling` with `temperature 1.0`, `top_p 0.95`, `top_k 20`,
per-request random `seed`; companion doc §7.8 and the sample `request_start` record
read this session, **[log]**). Thinking resolves **on** by default (no
`--no-thinking`; froggeric template default `enable_thinking = true`,
chat_template.jinja:6) and opencode's `reasoning_effort: "xhigh"` (xhigh variant) is
rendered into the system block as the xhigh effort instruction
(chat_template.jinja:83-84) — the live serve stderr shows `thinking xhigh` on
opencode agent steps (**[log]**).
### 2.5 Constrained decoding (`grammar` / `response_format`)

opencode's custom-provider route never sends constraints (companion doc §2.1), but the
capability is live on this deployment and matters for the failure-mode inventory.

- **One constraint kind per request**; `grammar` + constraining `response_format` →
  `constrained_decoding_conflict`; **any constraint + a `tools` field (even `[]`) →
  `constrained_decoding_not_supported`** (openai_chat_request.cpp:285-306, **[code]**;
  serving.md:155-156, **[docs]**).
- `json_schema` goes through the vendored `json-schema-to-grammar` converter under a
  **context-accurate allowlist** (constraint_contract.cpp, **[code]**):
  - accepted keywords: `type`, `properties`, `required`, `additionalProperties`,
    `items`, `prefixItems`, `minItems`, `maxItems`, `minLength`, `maxLength`,
    `format`, `minimum`, `exclusiveMinimum`, `maximum`, `exclusiveMaximum`, `const`,
    `enum`, `$ref`, `$defs`, `definitions`, `anyOf`, plus annotations
    (`title`, `description`, `default`, `examples`, `$comment`, `deprecated`,
    `readOnly`, `writeOnly`, `$id`, `$schema`) (constraint_contract.cpp:23-36).
  - context gates: item bounds need `type:"array"` (and no tuple `items`/
    `prefixItems`); string keywords need `type:"string"`; numeric bounds need
    `type:"integer"` (a union containing `number` rejects them); object keywords need
    `type:"object"`; `items`+`prefixItems` cannot combine; `minimum`+
    `exclusiveMinimum` (and the max pair) cannot combine (constraint_contract.cpp:247-282).
  - enforced `format` values: `date`, `time`, `date-time`, `uuid` (versioned uuidN
    forms not enforced) (constraint_contract.cpp:46-51).
  - **`pattern` is rejected** `json_schema_unsupported` (the byte-identical vendored
    converter would silently degrade it to "any string") — likewise `allOf`/`oneOf`
    and every unknown keyword, each naming the keyword
    (constraint_contract.cpp:186-188, 247-288; serving.md:174-176).
  - dispatching keywords (`$ref`/`anyOf`/`const`/`enum`) admit no structural siblings;
    `const`/`enum` values must match a `type`; `required`/`enum` well-formedness is
    validated (constraint_contract.cpp:220-305).
  - payload ≤64 KiB, nesting ≤64 levels, else `constraint_too_large`
    (constraint_contract.h:16-17, constraint_contract.cpp:176-180, 341-345).
- **Grammar compilation is at submit time, before queue admission** (engine_core.h:185-214,
  **[code]**): unparseable GBNF or an initial mask that admits no token → 400
  `grammar_invalid` / `json_schema_invalid` (mapped per `constraint_source`,
  generation_service.cpp:63-72) with **no GPU work and no queue occupancy** — a
  pathological grammar cannot stall the single-tenant GPU.
- **Backend gate**: DFlash/DFlash2 reject constrained requests at serve translation
  with `constrained_decoding_not_supported`, message naming the backend
  ("constrained generation is not supported with the … speculative backend; use the
  ordinary or MTP backend", translate.cpp:334-350, **[code]**). This deployment runs
  MTP → constrained requests are supported.
- **Thinking wrapper**: a constraining request that resolves thinking enabled gets the
  constraint applied to the whole turn (free-form reasoning body that cannot contain
  the close marker, forced close + format whitespace, then the answer grammar).
  Constraining requests **default to non-thinking** unless the request explicitly
  enables thinking (serving.md:151-221, **[docs]**).
- **Completion semantics**: grammar completion ends generation; `max_tokens`
  truncation may cut a constrained answer mid-JSON with no per-prefix parseability
  promise (serving.md:187-188, **[docs]**). A mask fill that throws **after
  submission** is engine-fatal (worker fail-fast; see §3c).
- Performance: the in-tree token-trie producer landed (PR #81/#83/#84/#85); ordinary
  decode Δ within ±0.24 ms/step at ~13.7 ms/step, MTP ≤ ~0.3 ms/round, one-time
  ~0.7-0.8 s first-request grammar setup (docs/maintainer/grammar-mask-production.md:
  96-103, **[docs]**).

---

## 3. Streaming wire behavior (Chat Completions, `stream: true`)

Encoder: `src/serve/openai_chat_response.cpp` (**[code]**); HTTP plumbing:
`src/serve/openai_chat_http.cpp` (**[code]**). Exact event sequence the client sees:

1. **Start chunk** — `choices[0].delta = {"role":"assistant","content":""}`,
   `finish_reason: null` (openai_chat_response.cpp:256-260). Every chunk also carries
   `usage: null` when `include_usage` is set (L192-199) — a null opencode ignores.
2. Optional `prompt_progress` chunks (only with `return_progress: true`; opencode
   does not send it).
3. **Reasoning deltas** — `delta = {"reasoning_content": <text>}` per committed
   reasoning fragment (L328-335). This is the first field in opencode's reasoning
   fallback list, so it is consumed as the reasoning channel (companion doc §3.6).
4. **Content deltas** — `delta = {"content": <text>}` (L337-344).
5. **Terminal block** (L346-386):
   - any buffered suffix not yet streamed is flushed as a final reasoning/content
     delta (this is how withheld framing bytes land — see §3b);
   - if the turn produced parsed tool calls: **one chunk with the complete
     `delta.tool_calls` array** — `[{id: "call_<16 hex>", type: "function",
     function: {name, arguments: <full JSON string>}, index: i}]` — followed by a
     chunk with empty delta and **`finish_reason: "tool_calls"`**;
   - otherwise a chunk with empty delta and the mapped `finish_reason`;
   - if `include_usage`: a final chunk with **empty `choices: []`** carrying full
     `usage` and terminal `timings`;
   - **`data: [DONE]`** (L384).

**`finish_reason` values** (L118-130, **[code]**; serving.md:1091-1094, **[docs]**):
`OutputLimit`/`ContextCapacity` → `"length"`; `StopToken`/`StopString`/`None`/
`Cancelled` → `"stop"`; tool-call turns → `"tool_calls"`. **Only these three values
ever appear** — all are in opencode's accepted set (companion doc §3.4).
`"length"` is emitted when the output limit (max_tokens) or the context capacity is
hit; `"stop"` on a natural stop token/string. The aggregate (non-streaming) response
uses the same mapping (L227-249).

**Usage** (L157-174, **[code]**): `prompt_tokens`,
`prompt_tokens_details.cached_tokens` (exact checkpoint-proven prefix reuse),
`completion_tokens`, `completion_tokens_details.reasoning_tokens`, `total_tokens`.
Both fields opencode maps from are present (companion doc §3.5) — and because usage
arrives in the final chunk **before** `[DONE]` with one choice in every earlier chunk,
opencode's context fitting and compaction thresholds see real token counts.
**[log]** the live 71 h window shows accurate accounting in practice: 21
`output_limit` finishes, 3,259 `stop_token` finishes, with per-request
`prompt_tokens`/`completion_tokens` in `request_done`.

**Non-streaming shape** (L227-249, **[code]**): single `chat.completion` object,
`choices[0].message` with `role`, `content` (null when tool calls present and no
text), `refusal: null`, optional `reasoning_content`, optional `tool_calls` (no
`index` in aggregate), `finish_reason`, top-level `usage` + `timings`.

**Terminal timings** (llama.cpp-compatible, always present): top-level `timings`
object attached to the last JSON chunk before `[DONE]` (the usage chunk when
`include_usage`, else the finish chunk) (openai_chat_response.cpp:192-208, 354;
serving.md:326-356, **[docs]**). Extra top-level fields are ignored by opencode.

### 3a. Constraint violation mid-generation

Because the mask is a **hard per-token constraint**, the model cannot emit
constraint-violating bytes — there is no mid-stream "grammar parse failure" event the
client sees. The observable failure modes are: (i) synchronous 400 at submit
(`grammar_invalid`/`json_schema_invalid`, before any SSE is opened), or (ii) the
answer hitting `max_tokens` mid-JSON — the stream then ends normally with
`finish_reason: "length"`, and the partial JSON is what it is (no per-prefix
parseability promise, serving.md:187-188). For opencode this is a normal
`finish_reason: "length"` completion: the step ends with truncated output, no error
(companion doc §4.5).

### 3b. Chat-parser demotion — what the wire looks like

When the serve-side chat parser rejects a generated tool-call region (malformed
structure, undeclared tool name, conflicting duplicate parameters, …), the region's
markup is **returned to the client as ordinary assistant text**: the content deltas
carry the raw `tool_call`/`function`/`parameter` markup verbatim, `tool_calls` is
empty, `finish_reason` is `"stop"` (the turn ended on the model's stop token), and
**no in-band error is emitted** (serving.md:257-269 — "a region without a served
candidate stays ordinary content verbatim"; openai_chat_response.cpp:346-386 —
demoted text flows through the ordinary content channel and the `!outcome.tool_calls`
branch). Serve logs exactly one classification-level warning to stderr ("If a tool
marker is returned to text … Serve emits one warning with only the failure
classification, never the generated markup", serving.md:930-931, **[docs]**) and
records `request_done.result.tool_call_parse` with `marker_seen`,
`structured_call_count`, `duplicate_arguments_merged`,
`schema_mismatch_arguments`, `empty_arguments_omitted` and
`fallback_reason ∈ {none, malformed_structure, duplicate_parameter,
invalid_tool_name, undeclared_tool, trailing_content}`
(request_log.cpp:87-95, serving.md:958-963, **[code/docs]**).

**The client consequence is a silent stall**: opencode receives a clean
`finish_reason: "stop"` text-only turn, executes no tools, and ends the agent loop
with nothing pending — the session sits idle until the user types again (incident
record, §6.1). Partial tool-call recovery does exist *within* a region (the parser
serves the first fully-parsable `tool_call` region and consumes the response to its
end, serving.md:266-269), and a quoted close-marker stays in reasoning rather than
content (serving.md:240-243) — but a demoted region is demoted verbatim.

**[log]** the current 71 h log window contains exactly **one** demotion
(`fallback_reason: "malformed_structure"`, req#310, 2026-10-02 ~09:57 UTC,
prompt 90,823 / completion 10,073 tokens, `stream: true`, **`tools: null`** on that
request — an undeclared-markup turn), against 3,279 clean `none` finishes: a low but
non-zero rate on live opencode traffic.

### 3c. Engine death mid-stream

Failure path (engine_core.h:1953-1975 `fail_all_locked`, **[code]**): any fatal
worker error sets `failed_` and completes **every** active, materializing, and
pending request with that error; a clean shutdown completes them with
`RequestError(Unavailable, "inference engine is shutting down")` (engine_core.h:
1991-1998). In serve these map (generation_service.cpp:41-99, **[code]**) to:

- in-flight streaming request → `wait()` throws → the open SSE stream receives
  **one in-band `data: {"error":{…}}` event and then closes**
  (openai_chat_http.cpp:197-208 → `sse_error_event`, L19-21, **[code]**). No
  `finish_reason`, no `[DONE]`, no partial-usage chunk. The error body carries the
  mapped status/code (e.g. 503 `service_unavailable` for Unavailable; 500
  `internal_error` with the engine's message for a CUDA fault).
- subsequent requests → 503 `service_unavailable` "inference engine is unavailable"
  (engine_core.h:220-223, **[code]**);
- `GET /health` → 503 `{"status":"unavailable"}` (http_server.cpp:428-433,
  **[code]**).

opencode's handling (companion doc §3.7, §4.5): an in-band `error` event on a 2xx
stream fails the stream exactly like an HTTP failure — `Unavailable`/5xx-class
errors are **retryable**, the partial assistant output (text + settled tool calls) is
persisted, a synthetic "continue where you left off" user message is appended, and a
fresh step is issued. So an engine death mid-stream degrades to a *continued*
conversation turn rather than a lost session, as long as the unit's
`Restart=always` brings the GPU back within opencode's retry window (10 retries,
~84 s, §4.3 there).

**Client-initiated close** is the other mid-stream terminal: opencode's 300 s
chunk-timeout or a user stop disconnects; serve detects it via
`client_disconnected(req)` (openai_chat_http.cpp:185-188), the request is cancelled
(499 `client_disconnected` mapping, generation_service.cpp:85-90), the stream simply
stops — nothing more is written to the dead socket. **[log]** the 71 h window
contains 3 such `request_error: "client disconnected"` records (2026-09-30 08:41,
2026-10-02 08:47 and 10:52 UTC) and zero engine-death records.

---

## 4. Error semantics

OpenAI-shape error bodies: `{"error":{"message","type","param","code"}}`
(openai_common.cpp:203-208, **[code]**) — exactly the layout opencode's classifier
understands (companion doc §4.2). Status/code table for the Chat Completions route:

| Condition | HTTP | code / type | Message (shape) | Citation |
|---|---|---|---|---|
| malformed field (missing `model`/`messages`, wrong type, …) | 400 | `invalid_request_error` + field-specific `code` | field-specific | request_validation.cpp:11-19 **[code]**; §2.3 |
| model id ≠ `Qwen3.8-27B` | **404** | `model_not_found`, `param: model` | `model '<id>' not found` | openai_common.cpp:216-225 **[code]**; **[probe]** |
| prompt (after template + media expansion) > `--max-context` | 400 | `context_length_exceeded`, `param: messages` | `prepared prompt has <N> tokens, exceeding Engine max_context 446902` | engine.cpp:50-53, 304-309 **[code]**; test_http_error_handler.cpp:36-43 **[test]** |
| thinking-budget control suffix cannot fit | 400 | `thinking_budget_capacity_insufficient` | — | generation_service.cpp:50-54 **[code]** |
| media preprocessing over budget | 400 | `media_budget_exceeded` | — | generation_service.cpp:55-58 **[code]** |
| invalid media source/bytes | 400 | `invalid_media` | — | generation_service.cpp:59-62 **[code]** |
| remote media fetch failure / timeout | 502 / 504 | `media_fetch_failed` / `media_fetch_timeout` | — | generation_service.cpp:116-125 **[code]** |
| bad grammar / unsatisfiable / DFlash backend (see §2.5) | 400 | `grammar_invalid` / `json_schema_invalid` / `constrained_decoding_not_supported` | — | generation_service.cpp:63-72, translate.cpp:334-350 **[code]** |
| raw request body > `--max-request-mib` (default 384 MiB) | 413 | `request_too_large` | `request body exceeds the configured payload limit of <N> bytes` | http_server.cpp:164-170 **[code]**; test_http_error_handler.cpp:87-100 **[test]**. caddy's 128 MiB body cap fires first for `ai.kido.ws` traffic (companion doc §7.9) |
| lifetime capacity full (C+P = 68) | **429** | `server_overloaded`, type `rate_limit_error` | `inference request queue is full` | generation_service.cpp:73-78, 271-293 **[code]** |
| admission+preparation exceeds `--pending-timeout-ms` (600 s) | 503 | `request_queue_timeout` | `inference request expired before submission` / `…during preparation` | generation_service.cpp:79-84, engine_core.h:180-183 **[code]** |
| client disconnected | 499 | `client_disconnected`, type `request_cancelled` | — | generation_service.cpp:85-90 **[code]** |
| engine failed / shutting down (new request) | 503 | `service_unavailable` | `inference engine is unavailable` | generation_service.cpp:91-96, engine_core.h:220-223 **[code]** |
| uncaught internal | 500 | `internal_error` | exception message | http_server.cpp:398-425 **[code]** |
| Anthropic route overload | **529** | `overloaded_error` | — | anthropic_messages_response.cpp:135-141 **[code]** |
| Anthropic route queue/media timeout | **504** | `timeout_error` | — | anthropic_messages_response.cpp:142-147 **[code]** |

**The 446902-vs-262144 question.** The companion ticket frames the deployment as
"serving 262144"; the actual unit passes `--max-context 446902` (llm.nix:513-514,
**[config]**) and `/v1/models` reports `max_model_len: 446902` (**[probe]**). The
artifact's *native* position capacity is 262144; above it the fork's **YaRN context
extension** is active (README.md:260-265, **[docs]**; llm.nix:510-512 comment:
"prompts past 256K go through the fork's YaRN context extension", **[config]**).
So for a request between 262144 and 446902 tokens: it is **accepted and served**,
with YaRN-corrected RoPE for positions beyond the native range (the extension is
default β_fast=32/β_slow=1/ext_factor=1; DFlash would reject it, but this unit runs
MTP). Only a prompt beyond **446902** returns the 400
`context_length_exceeded` above. opencode's declared window (`limit.context: 446902`,
`~/.config/opencode/opencode.jsonc`, **[config]**) therefore **matches the deployment
exactly** — there is no over-declaration on this provider. Consequence for
opencode's overflow path: with `compaction.auto: false` (config), a genuine
overflow 400 is a *fatal* step failure (companion doc §7.4) — but such a request
would have to exceed 446902 prompt tokens, i.e. far past what opencode's own
`max_completion_tokens` fitting (window minus measured prompt) ever sends.

**Overflow classification for opencode.** The message text
("prepared prompt has N tokens, exceeding Engine max_context 446902") does not
contain the classic phrase patterns ("prompt is too long", "exceeds the context
window"), **but** the error `code` is `context_length_exceeded`, which opencode
reads from `error.code` and matches against its overflow code set (companion doc
§4.1: "a backend rejecting an over-window prompt should return a 4xx whose message
matches one of these patterns **(or the OpenAI `context_length_exceeded` code)**").
So overflow is classified `context-overflow` → non-retryable → (with auto-compaction
off) fatal step error with the provider message shown.

---

## 5. Concurrency, admission, queueing

- **Capacity model** (serving.md:1025-1046, **[docs]**; generation_service.cpp:17-39,
  271-293, **[code]**): C=4 active lanes + P=64 pending = **68-request lifetime
  capacity** (requests in preparation, queued, active, or awaiting response release).
  A full capacity rejects synchronously with **429 `server_overloaded`** *before*
  admission — the client sees an immediate 429, not a hang.
- **FIFO, no preemption**: the Engine keeps a FIFO pending queue
  (engine_core.h:216-233, **[code]**); a request waits until its full
  prompt+output page entitlement can be reserved; the absolute
  `--pending-timeout-ms` deadline (600 s here) covers preparation + queue wait and
  returns **503 `request_queue_timeout`** if admission never occurs (serving.md:
  1033-1039, **[docs]**; engine_core.h:180-183, generation_service.cpp:79-84,
  **[code]**). Active requests are never preempted.
- **Decoding**: at each decode boundary all decode-ready requests compact into one
  batch (one model traversal, one exact-batch CUDA Graph replay) (serving.md:
  1027-1031, **[docs]**). Live evidence of queueing under opencode traffic:
  `request_done.engine_timing.queue_wait_seconds` is small (e.g. 0.0295 s on the
  demoted req#310) and the operational log shows `queue 27-35 ms` on concurrent
  opencode requests (**[log]**).
- **opencode's view**: 429 is *retryable* (RateLimit class) with `Retry-After`
  honored; 503 `request_queue_timeout` is `ProviderInternal` → retryable. With C=4
  and the agent's sequential step pattern, sustained overload requires ≥5
  concurrent opencode sessions each in flight; the 64-deep pending queue absorbs
  bursts.
- HTTP worker pool: `C + P + 1 = 69` workers with a 68-slot task queue
  (http_server.cpp:221-226, **[code]**).

---

## 6. Structured request log and known failure modes

### 6.0 The log

`--request-log-jsonl /home/kido/trash/temp/ninfer-yarn-log.jsonl`
(**[config]** llm.nix:537-540). One JSON object per line, schema v22
(serving.md:933-1023, **[docs]**; request_log.cpp:554-591, **[code]**):
`server_start` (argv redacted, capacities, sampler defaults), `request_start`
(protocol, model, message/tool counts, `tool_choice`, `requested_output_tokens` +
`requested_output_tokens_source: "client"|"server_default"`,
`requested_reasoning_effort`, resolved `sampling`, `enable_thinking`,
`preserve_thinking`, `thinking_budget`, `stream`, preparation breakdown),
`request_rejected` (phase `prepare` + exact status/type/code/param/message),
`request_done` (finish reason, prompt/completion/computed-prefill tokens, prefix
reuse path, **`tool_call_parse` diagnostics**, thinking-budget counters, stage
seconds, `engine_timing` incl. `queue_wait_seconds`, speculative counters,
materialization), `request_error` (config + terminal message), `throughput`
(interval counter deltas, scheduler snapshot, host-work breakdown). No generated
text or API keys in the file (serving.md:984-985, **[docs]**).

**[log]** 71.3 h window (2026-09-29 21:18:59Z → 2026-10-02 20:36:48Z): 3,284
`request_start` / 3,280 `request_done` / 3 `request_error` / 22 `server_start`
(process restarts; `Restart=always`), **all** `openai_chat_completions` — i.e. this
window is pure opencode traffic. Finish reasons: 3,259 `stop_token`, 21
`output_limit`. Tool-parse fallbacks: 3,279 `none`, 1 `malformed_structure`. Output
tokens: 100% client-sourced. Three `request_error` records are all
"client disconnected".

The e2e ticket (#143) mines `request_done.result.tool_call_parse.fallback_reason`
for demotion rates; the field and vocabulary are pinned above.

### 6.1 Parser demotion → silent stall

- **Mechanism**: §3b — markup returned as assistant text, `finish_reason: "stop"`,
  no error; opencode ends the run with no tool calls; the session idles until the
  user re-engages, and the continuation does not necessarily re-issue the lost call.
- **Incident record (2026-09-29)**: 5 demotions in the 4 preceding days on the
  production journal — `malformed_structure` ×2, `undeclared_tool` ×2,
  `duplicate_parameter` ×1 (maintainer journal evidence recorded 2026-09-29; the
  pre-2026-09-29-21:18 UTC entries are no longer in the current log file, which
  starts at that boundary).
- **Current rate**: 1 demotion in the 71 h window above (req#310,
  `malformed_structure`) — the mode persists at low frequency.
- **Mitigations already landed**: the chat-parsing port (PR #36/#38, 2026-09-27)
  added candidate search + recovery so the first *parseable* region is served and
  the response consumed to its end (serving.md:266-269, **[docs]**); the
  duplicate-parameter merge (below) removes one demotion class for byte-identical
  repeats.

### 6.2 Duplicate-parameter merge policy (2026-09-29)

A repeated tool-call parameter **merges only when every later occurrence's value is
byte-identical** to the first occurrence's raw extracted value (first kept, position
preserved; non-adjacent and 3+ repeats collapse the same way). A repeat with
differing value bytes stays fail-closed `duplicate_parameter` demotion.
Landed as issue #41 / PR #42, squash `75dbfeec` "fix(chat-parse): merge
byte-identical duplicate parameters (#42)" (verified on master this session,
**[code/config]** `git log`). The `request_done.tool_call_parse.
duplicate_arguments_merged` counter records merges (request_log.cpp:92,
**[code]**); corpus now 42 vectors. **[docs]** serving.md:261-264.

### 6.3 `max_tokens` truncation incident (2026-09-22)

opencode 2.x sends no `max_tokens`, so the serve's 8192 default
(`--default-max-tokens`) truncated long replies and tool calls "died mid-stream"
(llm.nix:418-419 comment, **[config]**). Fixed by `--default-max-tokens 32768` on
both units (llm.nix:437-438, 515-516, **[config]**) plus the client-side
`body.max_tokens: 32768` overlay in opencode's config (**[config]**) — which is why
every request now carries both max fields and §2.1's precedence applies. **Live
footprint**: 21 `output_limit` (→ wire `finish_reason: "length"`) finishes in the 71
h window; each is a normal completed step for opencode (truncated text kept, no
error), not a failure (companion doc §4.5).

### 6.4 Cold constrained-decoding cost

First constrained request pays a one-time ~0.7-0.8 s grammar setup
(docs/maintainer/grammar-mask-production.md:96-103, **[docs]**); steady-state cost
is within the measured ±0.24 ms/step. Not an opencode issue (it sends no
constraints) — recorded for the capability inventory.

---

## 7. Compatibility notes for the verdict (cross-reference to companion doc)

One line per opencode requirement area; "match/gap/risk" is pre-decision evidence,
the verdict belongs to #144.

| opencode requirement (companion doc) | ninfer-yarn serve answer | Tag |
|---|---|---|
| exactly one route: `POST /v1/chat/completions`, always stream, SSE + `[DONE]` | route registered; stream + `[DONE]` always emitted on this route | code/probe |
| `finish_reason` mandatory before `[DONE]`, known values only | always present in the terminal chunk; only `stop`/`length`/`tool_calls` ever emitted | code |
| no content after `finish_reason` | encoder flushes all buffered content *before* the finish chunk; usage-only chunk follows (explicitly tolerated by opencode) | code |
| tool-call `id`+`function.name` complete by stream end | both (plus full arguments) arrive in the single terminal `tool_calls` delta — strongest possible form | code |
| both `max_tokens` + `max_completion_tokens` | both accepted; **`max_completion_tokens` wins**; default 32768 otherwise | code/test/log |
| `store: false`, `reasoning_effort`, `stream_options.include_usage`, `strict:false` tools, `tool_choice:"none"` (max-steps), `reasoning_content`/`content:null` replay, `<system-update>` user text, image data URLs, x-opencode-* headers | all accepted: `store:false` legal, effort values incl. `xhigh` honored by the template, usage chunk emitted, non-strict tools fine, `none` fine, both replay fields fine, arbitrary user text fine, data-URI images fine (vision on), headers opaque | code/config |
| final `usage` with `prompt_tokens_details.cached_tokens` + `completion_tokens_details.reasoning_tokens` | exactly those fields in the pre-`[DONE]` usage chunk | code |
| overflow 4xx with `context_length_exceeded` code or phrase | 400 with code `context_length_exceeded` + descriptive message; declared window 446902 == deployed 446902 (YaRN), so overflow requires exceeding the declared window | code/probe/config |
| dead stream → partial persisted + continue | in-band `data: {"error":{…}}` event then close (no `[DONE]`); maps to opencode's retryable dead-stream handling with continuation | code |
| 429 retryable | 429 `server_overloaded` only when 68-request lifetime capacity is full (immediate, synchronous) | code |
| model metadata from config (never `/v1/models`) | consistent: `/v1/models` reports 446902 anyway; opencode's declared 446902 matches | probe/config |
| **RISK — demotion silent stall** | parser demotion = clean text-only turn; opencode idles (5 in 4 days on 2026-09-29; 1 in 71 h now); no serve-side signal beyond the JSONL | code/log |
| **RISK — `length` truncation at 32768** | 21 in 71 h; benign for opencode (step ends, next input continues) but long tool-call-heavy turns can truncate mid-region → partial `tool_calls` are *tolerated* by opencode on `length` | log/code |
| **NOTE — template prompt shaping** | froggeric v22.5 injects a consecutive-tool-error "SYSTEM WARNING" into tool results and an xhigh effort instruction into the system block — opencode is unaware; behaviorally observable only through model output | config |
