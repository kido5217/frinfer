# Reference profiles: llama.cpp server and vLLM as opencode backends

Wayfinder map #139, ticket #142 — the reference side of the opencode ⇄ backend
compatibility matrix. The requirement side (what opencode v2.0.22 sends and
checks) is `docs/research/opencode-requirements.md` (ticket #140, branch
`research/opencode-requirements`); the capability side (what ninfer-yarn's
serve does) is `docs/research/serve-capabilities.md` (ticket #141, branch
`research/serve-capabilities`). "Companion doc §N" citations refer to those.

**No live runs** (user decision, 2026-10-02). Evidence is official
documentation, opencode's docs/source, and the user's own `nixos-configs`.

**Sources of record (all read this session, 2026-10-02)**

| Source | Pin |
|---|---|
| llama.cpp repo (`github.com/ggml-org/llama.cpp`) | master `bed0a85660` (2026-10-02): `tools/server/README.md`, `docs/function-calling.md`, `common/chat.h`, `common/chat.cpp`, `common/chat-peg-parser.cpp`, `tools/server/server-{task,chat,schema,context,common}.cpp/h`, `tools/server/server-queue.h`. Fetched as raw files. There is no llama.cpp docs site — `docs.llama.com` does not resolve (NXDOMAIN, checked 2026-10-02). |
| vLLM repo (`github.com/vllm-project/vllm`) clone at `/tmp/opencode/vllm-src` | master `ced6857` (`v0.30.0`-current): `docs/features/{tool_calling,reasoning_outputs,automatic_prefix_caching}.md`, `docs/serving/online_serving/openai_compatible_server.md`, `docs/usage/faq.md`, `docs/usage/v1_guide.md`, `vllm/parser/qwen3.py`, `vllm/parser/engine/{parser_engine,parser_engine_config,streaming_parser_engine}.py`, `vllm/entrypoints/openai/chat_completion/{serving,protocol}.py`, `vllm/entrypoints/serve/engine/protocol.py`, `vllm/entrypoints/serve/exception_handling/error_response.py`, `vllm/entrypoints/serve/utils/api_utils.py`, `vllm/config/{scheduler,model,cache,multimodal}.py`, `vllm/engine/arg_utils.py`. |
| vLLM published docs | `docs.vllm.ai` (stable; page existence verified 2026-10-02: `serving/online_serving/openai_compatible_server.html`, `features/tool_calling.html`, `features/automatic_prefix_caching.html`). Repo docs files cited are the source of these pages; implementation claims cite the `ced6857` clone. |
| opencode source (`github.com/sst/opencode`, now `anomalyco/opencode`), tag `v2.0.22` (`527f0b931`), clone at `/tmp/opencode/opencode-src` | `packages/web/src/content/docs/providers.mdx` (source of <https://opencode.ai/docs/providers/>), `packages/core/src/plugin/provider/ollama.ts`, `packages/ai/src/provider-error.ts`. |
| User deployment | `nixos-configs/users/kido/llm.nix` (688 lines, read in full): `llama-cpp-qwen38-27b` at llm.nix:263-314, `vllm-qwen38-27b-nvfp4` at llm.nix:341-406, identical-behavior contract comment at llm.nix:316-325. |
| opencode overflow classifier | `packages/ai/src/provider-error.ts:17-46` (read this session) — each backend's overflow error text below is cross-checked against this list. |

All three qwen38 backends (llama.cpp, vLLM, ninfer-yarn) run the **same
froggeric v22.5 chat template** (llm.nix:7, 283-284, 363, 396-397 — the
`chatTemplate` binding is mounted at `~/.local/share/ai/qwen3x/chat_template.jinja`
for the llama.cpp and vLLM blocks), so tool-call *prompting* is identical
across all three; only the serving/parsing side differs.

---

## 1. llama.cpp `llama-server`

### 1.1 Endpoints

OpenAI-compatible (served at the server root; the user's unit binds
`127.0.0.1:8827` and caddy proxies `ai.kido.ws/v1/…` onto it, so opencode hits
`https://ai.kido.ws/v1/chat/completions` → `:8827/v1/chat/completions`):

| Method + path | Behavior |
|---|---|
| `GET /v1/models` | Single model object; `id` = `--alias` value (else the model file path); extra non-OpenAI `meta` field (vocab, `n_ctx_train`, sizes) |
| `POST /v1/completions` | OpenAI-style plain completions (streaming supported) |
| `POST /v1/chat/completions` | OpenAI Chat Completions (sync + SSE streaming) |
| `POST /v1/chat/completions/control` | In-flight control (`reasoning_end`); llama.cpp extension |
| `POST /v1/responses` | OpenAI Responses API (internally converted to chat completions) |
| `POST /v1/embeddings` | OpenAI embeddings — requires a pooling-type model |
| `POST /v1/responses/input_tokens`, `POST /v1/chat/completions/input_tokens` | token counting (the chat one explicitly "not an official OAI endpoint") |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic-compatible Messages API |

Native (non-OpenAI) endpoints: `/completion`, `/tokenize`, `/detokenize`,
`/apply-template`, `/embedding`, `/reranking`, `/infill`, `/props` (GET/POST),
`/slots` (per-slot state; `?fail_on_no_slot=1` → error when no idle slot),
`/metrics` (Prometheus, `--metrics`), slot save/restore/erase,
`/lora-adapters`; plus a **multi-model router mode** (`GET /models`,
`/models/load`, `/models/unload`, `/models/sse`, `POST /models` download,
`--models-max` default 4; routes by the request `model` field with
auto-load) and an experimental built-in-tools/MCP surface (`--tools`,
`--mcp-servers-config`; `server-mcp.cpp`)
(`tools/server/README.md:468-1242, 1595-1820, 1932-2158` @ `bed0a85660`).

Compatibility posture: "While no strong claims of compatibility with OpenAI
API spec is being made, in our experience it suffices to support many apps"
(`tools/server/README.md:1310` @ `bed0a85660`).

opencode exercises exactly one of these: `POST /v1/chat/completions`
(companion doc §1.1); it never calls `/v1/models` or `/v1/embeddings`
(companion doc §1.6/§1.7).

### 1.2 Tool-call parsing

**Chat-template-driven, PEG-grammar-parsed; no schema enforcement at
generation.**

- OpenAI-style function calling works "with the `--jinja` flag (and may
  require a `--chat-template-file` override to get the right tool-use
  compatible Jinja template)" (`tools/server/README.md:1389-1393` @
  `bed0a85660`). The user's block sets both `--jinja` and the froggeric
  template (llm.nix:282-283).
- `chat.h` (PR #9639) implements the machinery. llama.cpp maintains **native
  per-architecture tool-call formats** (Llama 3.1/3.2/3.3, Functionary,
  Hermes 2/3, Qwen 2.5 / Qwen 2.5 Coder, Mistral Nemo, Firefunction v2,
  Command R7B, DeepSeek R1, GPT-OSS Harmony) and a **generic tool-call
  handler** used "when the template isn't recognized by native format
  handlers (you'll see `Chat format: Generic` in the logs)"
  (`docs/function-calling.md:6-22` @ `bed0a85660`). The current master parse
  layer is PEG-based: `common_chat_format` ∈ `CONTENT_ONLY | PEG_SIMPLE |
  PEG_NATIVE | PEG_GEMMA4 | PEG_MINIMAX_M3` (`common/chat.h:228-236`).
- The froggeric v22.5 template is **not** a native format → the llama.cpp
  unit runs the generic PEG path, the same family of parsing that
  ninfer-yarn vendored and extended (companion serve-capabilities doc §0/§3b).
- The request can opt out per call: `parse_tool_calls` (default on) and
  `parallel_tool_calls` (default off; "verification is based on jinja
  template") are OpenAI-route options (`tools/server/README.md:1328-1330`).
- Emitted wire format: OpenAI `tool_calls` with `index`, `id`, `type:
  "function"`, `function.name`, `function.arguments` (JSON string); streamed
  incrementally as chat-message diffs (`server_chat_msg_diff_to_json_oaicompat`,
  `tools/server/server-chat.cpp:621-652` @ `bed0a85660`).
- **Partial/failing regions**: the PEG parser's streaming recovery is
  `parse_anywhere_and_extract` — scan from the first position at which the
  grammar parses and extract from there (prefix skip;
  `common/chat-peg-parser.cpp:206-220` @ `bed0a85660`). Regions that fail the
  grammar are not extracted as tool calls, so their markup remains in
  `content` (the same silent-demotion class as ninfer-yarn's parser
  demotions). A prior research ticket live-verified stock llama.cpp's
  qwen3-coder parser against this template family and recorded that it
  "silently drops the turn on quoted `</tool_call>`, trailing prose, a missing
  `</tool_call>`, unknown names/params, and required-arg order/duplication"
  because the grammar bakes names/params as literals without an end-of-input
  requirement (map #25 finding, 2026-09-27 — prior-session empirical evidence,
  recorded in project memory; not re-run here). `--skip-chat-parsing`
  disables parsing entirely (pure content, `tools/server/README.md:240`).

### 1.3 Reasoning / thinking extraction

- `--reasoning-format`: `none` (thoughts stay unparsed in `message.content`),
  `deepseek` (thoughts moved to `message.reasoning_content`),
  `deepseek-legacy` (both); default `auto` (`tools/server/README.md:232` @
  `bed0a85660`). The user's block pins `deepseek` (llm.nix:284).
- Streamed as `delta.reasoning_content` chunks, interleaved with
  `delta.content` (`server-chat.cpp:621-626`); non-streaming messages carry
  `message.reasoning_content` (`tools/server/README.md:1439-1443`: "the server
  supports parsing and returning reasoning via the `reasoning_content` field,
  similar to Deepseek API"). `--reasoning-preserve` (default enabled) keeps
  reasoning in full history (`tools/server/README.md:237`); the block sets it
  explicitly (llm.nix:296).
- Request-side: `reasoning_effort` ("if `none`, reasoning/thinking is
  disabled. Otherwise, the value is made available to the jinja template",
  `tools/server/README.md:1320`), server-side `--reasoning-effort` accepts
  `minimal|low|medium|high|xhigh|max` (`tools/server/README.md:234`), and
  `chat_template_kwargs` passthrough (`tools/server/README.md:1318`).
  opencode's `reasoning_effort: "xhigh"` is thus a known value to the template
  and the server flag alike.
- On the wire, `reasoning_content` is the **first** field in opencode's
  reasoning fallback chain (companion doc §3.6) — direct match. Replay:
  opencode sends `reasoning_content` on assistant messages (companion doc
  §2.3); llama.cpp's message parser accepts it as a first-class field
  (`common/chat.cpp:453-454`, `common/chat.h:85` @ `bed0a85660`).

### 1.4 Streaming discipline (vs opencode's contract)

SSE shape (OpenAI-compatible chat route; `tools/server/server-task.cpp`,
`server-context.cpp` @ `bed0a85660`):

1. First chunk: `delta {role: "assistant", content: null}`, `finish_reason:
   null` (`server-task.cpp:1116-1123`).
2. Content deltas: one chunk per chat-message diff — `content`,
   `reasoning_content`, and/or `tool_calls` fragments (`server-task.cpp:
   470-481`, `server-chat.cpp:621-652`).
3. Terminal chunk: empty `delta {}` carrying `finish_reason`
   (`server-task.cpp:483-492`).
4. If `stream_options.include_usage`: a final chunk with **empty `choices:
   []`** carrying `usage` (`server-task.cpp:502-514` — comment cites the
   OpenAI spec for the empty-choices usage chunk).
5. `data: [DONE]` — emitted for OpenAI-compatible response types (and only
   those; Responses-API and Anthropic streams terminate differently)
   (`server-context.cpp:4640-4651`).

- **`finish_reason` values**: default `"length"`; only on stop by EOS token or
  stop string → `"stop"`, or `"tool_calls"` when the turn produced tool calls
  (`server-task.cpp:415-424, 464-467`). So the wire values are
  `stop` / `length` / `tool_calls` — all in opencode's accepted set (companion
  doc §3.4); `"content_filter"` is never emitted.
- **`[DONE]` after `finish_reason`**: yes, the sequence is finish chunk →
  optional usage chunk → `[DONE]`; no content after `finish_reason`
  (opencode-tolerant: companion doc §3.2).
- **`[DONE]` is mandatory and present** on every completed OpenAI-compatible
  stream (same code path, no branch).
- **Mid-stream error**: an in-band `data: {"error": …}` event then stream
  close — no `[DONE]` after an error (`server-context.cpp:4607-4614,
  4687-4690`). Maps to opencode's dead-stream handling (companion doc §3.7/
  §4.5).
- **`usage`**: `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens` (`server-task.cpp:365-372`). There is
  **no `completion_tokens_details.reasoning_tokens`** — opencode maps fields
  it finds and degrades gracefully on missing ones (companion doc §3.5), so
  this is a gap in accounting richness, not a contract break.
- **Keep-alive**: SSE comment pings every `--sse-ping-interval` seconds
  (default 30) during protocol silence (`server-context.cpp:4660-4669`,
  `tools/server/README.md:218`); opencode tolerates comment lines (companion
  doc §3.1).
- **`timings`** object (llama.cpp extension) rides the final chunk — ignored
  by opencode (companion doc §3.2: extra fields ignored).

### 1.5 `max_tokens` / `max_completion_tokens`

- The request field is `n_predict` with aliases **`max_completion_tokens`
  first, then `max_tokens`** (`tools/server/server-schema.cpp:44-48` @
  `bed0a85660`). Resolution walks the name vector in order and takes the first
  name present in the body (`server-schema.cpp:590-609`) → **when opencode
  sends both (computed `max_completion_tokens` + fixed `body.max_tokens:
  32768` overlay, companion doc §2.2), `max_completion_tokens` wins** — the
  same precedence as ninfer-yarn and vLLM.
- **Default when neither is sent: `n_predict = -1` = infinity**
  (`tools/server/README.md:523`: "Default: `-1`, where `-1` is infinity").
  There is **no server-side default output bound** — generation runs to EOS or
  context exhaustion. (Contrast ninfer-yarn's `--default-max-tokens 32768`,
  which exists precisely because of opencode 2.x's behavior, llm.nix:418-419.)
  In practice opencode always sends `max_completion_tokens` on
  primary/compaction steps (companion doc §2.2), so the unbounded default only
  bites for `generate`/`title` requests — the same exposure vLLM has (§2.5).
- Hitting the bound: stop type `limit` → `finish_reason: "length"`
  (`server-task.cpp:253-255, 415-424`).

### 1.6 Context options on a 32 GB RTX 5090 (user's block)

The `llama-cpp-qwen38-27b` block (llm.nix:263-314) serves
`unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` from the local HF cache:

| Flag | Value | Documented meaning |
|---|---|---|
| `-c` | `0` | context size; "default: 0, 0 = loaded from model" (`tools/server/README.md:52`) — i.e. the GGUF's declared context, not a fixed number |
| `--cache-type-k` / `--cache-type-v` | `q8_0` / `q8_0` | KV cache dtype (allowed: f32, f16, bf16, q8_0, q4_0, …; default f16) (`tools/server/README.md:71-72`) — halves KV size vs f16 |
| `-fa` | `on` | flash attention (default `auto`) (`tools/server/README.md:49`) |
| `-ngl` | `99` | all layers on GPU |
| `--parallel` | `1` | one server slot (default `-1` = auto) (`tools/server/README.md:176`) |
| `--kv-unified` | set | single unified KV buffer shared across slots (default enabled when slots are auto) (`tools/server/README.md:168`) |
| `-b` / `-ub` | `2048` / `1024` | batch / ubatch size (`tools/server/README.md:56-57`) |
| `--cont-batching` | set | continuous batching (default on) (`tools/server/README.md:177`) |
| `--image-min-tokens` / `--no-mmproj-offload` | `1024` / set | vision enabled, mmproj on CPU (`tools/server/README.md:183, 181`) |
| `--jinja` + `--chat-template-file` | froggeric | template override (§1.2) |
| `--reasoning-format` / `--reasoning-preserve` | `deepseek` / set | §1.3 |
| `--spec-type` / `--spec-draft-n-max` | `draft-mtp` / `3` | MTP speculative decoding (3 draft tokens; `tools/server/README.md:271, 261`), draft KV q8_0 (llm.nix:299-300) |

**Window served**: `-c 0` means the unit's ceiling is the GGUF file's declared
context (native window; the Qwen3.8-27B family's native 262144 is documented
by the companion docs and by the vLLM block's binding-constraint note,
llm.nix:327-339). The block sets **no** `--rope-scaling yarn` (the flag exists,
`tools/server/README.md:58`), so — unlike the ninfer-yarn unit — there is no
context extension above the native window. The exact declared context of the
unsloth GGUF was not verified this session (no live runs); it is whatever the
file's metadata says, with the 262144 figure as the model family's documented
native window.

Stability note recorded in the block (llm.nix:255-262): Xid 8 / "launch timed
out" under sustained load (Blackwell CUDA-graph hang, ggml#27102/#27330) was
fixed by `RmWatchDogTimeout=60` (kept); `GGML_CUDA_DISABLE_GRAPHS=1` (a
10-15% tok/s cost) is the recorded fallback, currently disabled.

### 1.7 Concurrency / scheduling semantics

- **Slot model**: `-np/--parallel N` slots (the block pins `1`); each slot has
  a task queue (`server_context::queue_tasks`, `tools/server/server-context.h:
  181`); `--cont-batching` (default on) batches decodes of queued/active tasks
  together.
- **Under contention**: requests arriving while all slots are busy are
  **queued, not rejected** — no 429 overload response exists in the server
  code read; the only slot-availability signal is `GET /slots?fail_on_no_slot=
  1` → error `unavailable_error` "no slot available" (503,
  `server-context.cpp:4953-4958`, `server-common.cpp:60-63`). The HTTP
  read/write timeout `--timeout` (default 3600 s, `tools/server/README.md:
  217`) bounds how long a client can sit. A client sees unbounded waiting
  (within that timeout), not backpressure.
- **Error semantics**: OpenAI-shaped body `{code, message, type}` where `code`
  is the numeric HTTP status (400/401/403/404/500/501/503) and `type` is a
  string (`invalid_request_error`, `exceed_context_size_error`,
  `unavailable_error`, …) (`server-common.cpp:36-76` @ `bed0a85660`).
  opencode's classifier reads top-level `message` and `code` layouts (companion
  doc §4.2), so these bodies parse.
- **Overflow**: prompt larger than the slot context → **400
  `exceed_context_size_error`** "request (N tokens) exceeds the available
  context size (M tokens), try increasing it" (+`n_prompt_tokens`, `n_ctx`
  fields) (`server-context.cpp:3325-3338`, `server-task.cpp:1514-1516`).
  Cross-check: opencode's phrase list includes `/exceeds the available context
  size/i` (`provider-error.ts:36`) → classified `context-overflow`,
  non-retryable (companion doc §4.1) — the correct behavior.

### 1.8 Model coverage / quantization

- **Any GGUF model** (single file, multi-shard, or multimodal with `mmproj`),
  loaded from local files, directories, or `-hf <org>/<repo>:<quant>` HF
  download (`tools/server/README.md:1810-1873`). This is the broadest model
  surface of the three backends: every GGUF ecosystem model/quant, not just
  the Qwen3.5/3.6/3.8 `.ninfer` family.
- **Quantization**: the full GGUF set (Q2_K…Q8_0, IQ-quants, F16; UD-*
  unsloth dynamic quants as in the user's `UD-Q4_K_XL`), plus KV cache
  quantization types (`tools/server/README.md:71-72`).
- **Multi-model**: router mode serves many GGUFs behind one port with
  per-model routing by the `model` field and auto-load
  (`tools/server/README.md:1932-1958`), up to `--models-max` (default 4)
  concurrently.
- **Structured outputs**: `response_format` `json_object` / `json_schema`
  (schema-constrained via the vendored grammar stack;
  `tools/server/README.md:1316`) and native `/completion` `grammar`/
  `json_schema` fields (`tools/server/README.md:569-571`).

### 1.9 How opencode users configure it

- **First-class in the docs**: opencode's providers page has a dedicated
  **llama.cpp section** — "You can configure opencode to use local models
  through llama.cpp's llama-server utility" — with this exact shape
  (`packages/web/src/content/docs/providers.mdx:1352-1386` @ `527f0b931`;
  published at <https://opencode.ai/docs/providers/>):

  ```json
  {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
      "llama.cpp": {
        "npm": "@ai-sdk/openai-compatible",
        "name": "llama-server (local)",
        "options": { "baseURL": "http://127.0.0.1:8080/v1" },
        "models": {
          "qwen3-coder:a3b": {
            "name": "Qwen3-Coder: a3b-30b (local)",
            "limit": { "context": 128000, "output": 65536 }
          }
        }
      }
    }
  }
  ```

- Mechanically it is the **generic OpenAI-compatible custom provider** route
  (`@ai-sdk/openai-compatible`, `POST {baseURL}/chat/completions`, always
  stream — companion doc §1.1); the docs section is guidance, not a dedicated
  protocol. The docs file uses the v1 config schema (`provider`/`npm`/
  `options`/`models`); the installed v2.0.22 binary uses
  `providers`/`package`/`settings` (companion doc, "Verification notes").
- **Not** first-class in the engine: opencode has discovery plugins for
  Ollama (`packages/core/src/plugin/provider/ollama.ts` — polls
  `127.0.0.1:11434` every 30 s) and LM Studio, but none for llama.cpp or
  vLLM; model metadata for these comes from config only (companion doc §5.1).
  So the declared `limit.context`/`limit.output` are the operator's
  responsibility — for the llama.cpp unit that means declaring the GGUF's
  actual context (262144-class), since the backend serves no window
  extension.

---

## 2. vLLM OpenAI server

### 2.1 Endpoints

`docs/serving/online_serving/openai_compatible_server.md` @ `ced6857`
(published: docs.vllm.ai → OpenAI-Compatible Server):

| Method + path | Behavior |
|---|---|
| `GET /v1/models` | lists the served model name(s) (the `--served-model-name` alias); used by the docs' own quickstart examples via `client.models.list()` |
| `POST /v1/completions` | OpenAI plain completions (`suffix` unsupported) |
| `POST /v1/chat/completions` | OpenAI Chat Completions (sync + streaming); `parallel_tool_calls: false` → at most one tool call per request |
| `POST /v1/chat/completions/batch` | batch chat completions (vLLM extension) |
| `POST /v1/responses`, `/v1/responses/{id}`, `/v1/responses/{id}/cancel` | OpenAI Responses API |
| `POST /v1/embeddings` | OpenAI embeddings — **only for pooling models** (not this chat model) |
| `POST /v1/audio/transcriptions`, `/v1/audio/translations` | ASR models only |

Auth caveat from the same page: `--api-key` only protects the `/v1`, `/v2`,
`/inference` prefixes — other endpoints (notably `/invocations`) are
unauthenticated.

Model name handling: `vllm serve <model> --served-model-name Qwen3.8-27B`
aliases the single loaded model (the user's block does exactly this,
llm.nix:372-373). `served_model_name` accepts `str | list[str]`
(`vllm/engine/arg_utils.py:439` @ `ced6857`) — multiple **aliases**, not
multiple models. **Multi-model serving**: "serving multiple models at once
[on the OpenAI-compatible server] is not currently supported; you can run
multiple instances … and have another layer to route" (`docs/usage/faq.md:1-
5` @ `ced6857`).

opencode exercises exactly one endpoint: `POST /v1/chat/completions`
(companion doc §1.1). caddy already routes `ai.kido.ws/v1/embeddings` to the
separate crispembed backend, so vLLM's embeddings-absent-for-chat-models
limitation is invisible to opencode (companion doc §7.9).

### 2.2 Tool-call parsing: `qwen3_xml`

**Per-architecture registered parser, state-machine-parsed, incremental
streaming** — a different architecture from llama.cpp's template-driven PEG.

- Parser registry: `--tool-call-parser qwen3_xml` under mandatory
  `--enable-auto-tool-choice` ("Qwen3-Coder Models (`qwen3_xml`) … Flags:
  `--tool-call-parser qwen3_xml`"; `docs/features/tool_calling.md:432-439` @
  `ced6857`; published docs.vllm.ai → Tool Calling). The user's block sets
  both (llm.nix:393-395).
- Implementation: `Qwen3Parser` — a declarative **parser engine** state
  machine (`vllm/parser/qwen3.py:99-248` @ `ced6857`) over the same XML wire
  format the froggeric template prompts the model to emit:

  ```
  <tool_call>
  <function=func_name>
  <parameter=key>value</parameter>
  </function>
  </tool_call>
  ```

  (`vllm/parser/qwen3.py:5-15, 41-49`). Arguments are extracted by regex into a
  JSON object (`_qwen3_arg_converter`, `vllm/parser/qwen3.py:51-86`);
  `stream_arg_deltas=True` streams argument fragments as deltas;
  `tool_args_json=False` means the wire `arguments` is the JSON string (OpenAI
  shape).
- **Wire format**: OpenAI `tool_calls` deltas with `index`, `id`
  (`chatcmpl-tool-<random_uuid>`, `vllm/entrypoints/chat_utils.py:2256-2261`),
  `type: "function"`, `function.name`, streamed `function.arguments` fragments;
  the id and name are present from the first tool delta of each index (slot
  created on `TOOL_CALL_START`, name on `TOOL_NAME`, coalesced into one delta —
  `vllm/parser/engine/parser_engine.py:759-800` @ `ced6857`), satisfying
  opencode's id+name-by-stream-end requirement (companion doc §3.3) in its
  strictest form.
- **Partial/failing regions**: the engine is per-state explicit
  (`vllm/parser/engine/parser_engine_config.py:66-73` @ `ced6857`):
  plain text in the `TOOL_PREAMBLE`/`TOOL_BETWEEN` states maps to **no content
  event — it is dropped**, not leaked into `content`; the `finish()` path
  closes any open tool state at stream end with a `TOOL_CALL_END` event, so an
  unterminated region still emits a (possibly argument-truncated) tool call
  (`streaming_parser_engine.py:305-316`); a `<function=…>` without a preceding
  `<tool_call>` is still recognized (fallback transition,
  `vllm/parser/qwen3.py:150-154`). Contrast with llama.cpp/ninfer-yarn
  demotion (markup returned verbatim as content): vLLM's failure mode is
  *silently dropped bytes + truncated JSON arguments*, not markup leakage.
  `validate_tool_names` defaults to `False` and the qwen3 config does not set
  it — **undeclared tool names are accepted and emitted** (contrast
  ninfer-yarn's `undeclared_tool` demotion class).
- **Constrained decoding**: with `tool_choice: "auto"` (opencode's default)
  and `strict: false` on every tool (opencode always sends `strict: false`,
  companion doc §2.4), vLLM extracts calls from raw text — no schema
  constraint during generation; named-function/`required` choice or
  `strict: true` opts into structured-outputs constraint
  (`docs/features/tool_calling.md:110-135` @ `ced6857`). opencode sends
  `tool_choice: "none"` only on the max-steps step (companion doc §2.4) — vLLM
  honors `none` ("the model will not generate any tool calls",
  `docs/features/tool_calling.md:103-105`).

### 2.3 Reasoning / thinking extraction

- `--reasoning-parser qwen3` (user block, llm.nix:391-392). The Qwen3 series
  is in the supported-models table ("Parser Name `qwen3`", tool calling ✅)
  and "the reasoning feature for the Qwen3 series is enabled by default. To
  disable it, you must pass `enable_thinking=False` in your
  `chat_template_kwargs`" (`docs/features/reasoning_outputs.md:11-37` @
  `ced6857`).
- **Wire field: `reasoning`, not `reasoning_content`** — "reasoning used to be
  called `reasoning_content`. To migrate, directly replace `reasoning_content`
  with `reasoning`" (reasoning_outputs.md:7-9). Streamed in `delta.reasoning`
  (reasoning_outputs.md:83-108, example chunk). opencode's delta fallback
  chain is "the configured field, `reasoning_content`, `reasoning`,
  `reasoning_text`" (companion doc §3.6) → **`reasoning` is consumed** (second
  in the chain). Replay direction: opencode sends `reasoning_content` on
  assistant messages (companion doc §2.3); vLLM's request normalizer renames
  `reasoning_content` → `reasoning` on every message before validation
  (`vllm/entrypoints/openai/chat_completion/protocol.py:534-557` @ `ced6857`).
- `reasoning_effort` is a vLLM request field: any non-`none` value injects
  `enable_thinking = true` into the chat template kwargs (absent value → no
  injection) (`protocol.py:577-599`; documented as low/medium/high → true,
  none → false, reasoning_outputs.md:321-334). opencode's `xhigh` therefore
  **enables thinking; the level is not modeled** (contrast llama.cpp §1.3 and
  ninfer-yarn, where `xhigh` is a template effort level).
- Thinking budget: `thinking_token_budget` sampling param +
  `--reasoning-config` start/end strings; if the budget is hit vLLM forces the
  end marker (reasoning_outputs.md:248-261).
- Structured-output interplay: the parser engine feeds `is_reasoning_end` so
  constrained decoding skips requests that are still reasoning
  (reasoning_outputs.md:489-521).

### 2.4 Streaming discipline (vs opencode's contract)

SSE shape (`vllm/entrypoints/openai/chat_completion/serving.py` @ `ced6857`):

1. First chunk: `delta {role, content: ""}`, `finish_reason: null`
   (`chat_completion/serving.py:544-573`).
2. Token deltas: `delta.content` / `delta.reasoning` / `delta.tool_calls`
   fragments (parser-engine deltas, §2.2/§2.3).
3. Terminal chunk: last token chunk carries `finish_reason` with an empty-ish
   delta (`serving.py:740-780`).
4. If `stream_options.include_usage` (or `--enable-force-include-usage`):
   final chunk with **empty `choices: []`** carrying `usage`
   (`serving.py:814-875`).
5. `data: [DONE]` (`serving.py:506, 910`).

- **`finish_reason` values**: engine stop/length/content_filter pass through;
  a tool-producing turn under `auto`/`required` is remapped `stop` →
  `tool_calls` ("finish_reason is: `tool_calls` for `auto` or `required` tool
  calls, and `stop` for named tool calls", `serving.py:757-763` and the
  aggregate path at `1069-1091`). Wire set: `stop` / `length` /
  `content_filter` / `tool_calls` — all in opencode's accepted set (companion
  doc §3.4).
- **`[DONE]` after an in-band error**: vLLM yields the error event and **then
  still emits `data: [DONE]`** (`serving.py:895-910` — the `yield [DONE]` is
  outside the try/except). This is unlike ninfer-yarn and llama.cpp (both
  close without `[DONE]` after an error). opencode tolerates it: a stream that
  carries an `error` event fails the stream regardless of the sentinel
  (companion doc §3.7).
- **`usage`**: `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens` (+`cache_creation_tokens`) and
  `completion_tokens_details.reasoning_tokens`
  (`serving.py:819-845`, `_make_prompt_tokens_details`). This is the
  **most complete usage chunk of the three backends** — it covers every field
  opencode maps (companion doc §3.5).
- **No content after `finish_reason`**: the terminal chunk is the last content
  chunk; only the usage chunk follows (opencode-tolerant, companion doc §3.2).
- `parallel_tool_calls: false` → "vLLM only returns zero or one tool call per
  request" (openai_compatible_server.md:23-26 @ `ced6857`); opencode sends
  neither value (companion doc §2.1), so the default (parallel allowed)
  applies.

### 2.5 `max_tokens` / `max_completion_tokens`

- Precedence: `max_completion_tokens` wins over `max_tokens`
  (`protocol.py:617-624`: `if self.max_completion_tokens is not None` → use it,
  else `max_tokens`; `serving.py:310-317` passes the same to `get_max_tokens`)
  — identical to ninfer-yarn and llama.cpp.
- **Default when neither is sent: the full remaining window.**
  `get_max_tokens` returns `min(model_max_tokens, fallback_max_tokens, …)` over
  non-None values, where `model_max_tokens = max_model_len − input_length` and
  `fallback_max_tokens = max_tokens or default_sampling_params.get("max_tokens")`
  (`vllm/entrypoints/serve/utils/api_utils.py:169-196` @ `ced6857`). With no
  client bound and no `--default-max-tokens`-style override (vLLM has no such
  flag), the effective output limit is `max_model_len − prompt_tokens`.
  Same exposure class as llama.cpp §1.5: bounded in practice by opencode's
  `max_completion_tokens` on primary/compaction steps (companion doc §2.2).
- **Overflow**: `max_model_len < input_length` raises
  `ValueError("Input length ({n}) exceeds model's maximum context length ({m}).")`
  (`api_utils.py:184-187`), mapped to **400 `BadRequestError`** in the OpenAI
  error envelope (`exception_handling/error_response.py:56, 87-90`). Cross-check:
  opencode's phrase list contains both
  `/exceeds (?:the )?(?:model'?s )?maximum context length(?: …|\s*\([\d,]+\))/i`
  (`provider-error.ts:22`) and `/input length.*exceeds.*context length/i`
  (`provider-error.ts:44`) → classified `context-overflow`, non-retryable
  (companion doc §4.1).
- Model name mismatch: unknown `model` → **404 `NotFoundError`** "The model
  `X` does not exist" (`vllm/entrypoints/serve/engine/serving.py:197`,
  `error_response.py:49-51`). opencode sends `model: "Qwen3.8-27B"`, which the
  unit aliases via `--served-model-name` (llm.nix:372-373) → matches.

### 2.6 Context options on a 32 GB RTX 5090 (user's block)

The `vllm-qwen38-27b-nvfp4` block (llm.nix:341-406) serves
`unsloth/Qwen3.8-27B-NVFP4` in a podman container
(`docker.io/vllm/vllm-openai:latest`, `-p 8827:8000`), offline from the local
HF cache (`HF_HUB_OFFLINE=1`):

| Flag | Value | Documented meaning |
|---|---|---|
| `--max-model-len` | `262144` | "Model context length (prompt and output). If unspecified, will be automatically derived from the model config" (`vllm/config/model.py:215-217` @ `ced6857`) |
| `--gpu-memory-utilization` | `0.92` | "The fraction of GPU memory to be used for the model executor … per-instance limit" (default 0.92; `vllm/config/cache.py:103-109`) |
| `--kv-cache-dtype` | `fp8_e4m3` | KV cache dtype; "auto" = model dtype, CUDA supports fp8 (=fp8_e4m3)/fp8_e5m2 (`vllm/config/cache.py:111-120`) |
| `--kv-cache-memory-bytes` | `9000000000` | KV pool pinned to 9.0e9 bytes = 263,650 tokens ≥ 262144 (llm.nix:336-337); "(when not-None) ignores gpu_memory_utilization" (`vllm/config/cache.py:228-235`) |
| `--max-num-seqs` | `1` | "Maximum number of sequences to be processed in a single iteration" (`vllm/config/scheduler.py:63-68`); unset default for the API server is 1024 (`vllm/engine/arg_utils.py:2736-2739`) |
| `--max-num-batched-tokens` | `512` | "Maximum number of tokens that can be processed in a single iteration" (`scheduler.py:49-54`); raising to 2048 OOMs at CUDA-graph capture (llm.nix:337-338) |
| `--language-model-only` | set | "disables all multimodal inputs by setting all modality limits to 0" (`vllm/config/multimodal.py:91-93`) — frees the vision tower; release vLLM has no CPU offload for it (llm.nix:334-335) |
| `--enable-chunked-prefill` | set | prefill chunked on the `max_num_batched_tokens` budget (default on; `scheduler.py:116-120`) |
| `--enable-prefix-caching` | set | automatic prefix caching (§2.7 note) |
| `--no-async-scheduling` | set | "If set to False, disable async scheduling. Async scheduling helps to avoid gaps in GPU utilization" (`scheduler.py:190-193`) |
| `--reasoning-parser` / `--tool-call-parser` | `qwen3` / `qwen3_xml` | §2.2/§2.3 |
| `--chat-template` | froggeric (mounted ro) | same template as the other two backends |
| `--trust-remote-code` | set | HF remote-code trust |

**Window served: exactly 262144.** The block's header comment records the
VRAM arithmetic: "262K context is mandatory and is the binding constraint on a
32 GB card (31.34 GiB usable, ~1 GiB desktop)" — MTP does not fit (the MTP
head adds ~0.9 GiB model VRAM and the hybrid GDN/attention KV footprint grows
9.13 vs 8.38 GiB → OOM), nvfp4 KV needs a patched nightly (release v0.27.1
rejects it), and vision must be off; "Verified serving max_model_len=262144;
full-window benchmark (llama-benchy depth 243712, tg 8192, 3 runs): pp 1466
t/s, TTFT ~168 s, tg 33-52 tok/s" (llm.nix:316-340). No YaRN/extension
equivalent: 262144 is the hard ceiling.

### 2.7 Concurrency / scheduling semantics

- **V1 engine scheduler**: a unified scheduler treats prompt and output tokens
  alike, allocating a token budget per request; "supports multiple scheduling
  policies, including First-Come, First-Served (FCFS) and priority-based
  scheduling … configurable via `--scheduling-policy`"
  (`docs/usage/v1_guide.md:73-81` @ `ced6857`). Preemptions are handled by
  recompute — "vLLM V1 no longer requires KV cache swapping to handle request
  preemptions" (v1_guide.md:185-186).
- **What a client sees under contention**: requests beyond
  `max_num_seqs` sit in the scheduler's waiting queue (FCFS). With the user's
  `--max-num-seqs 1`, the unit is **single-stream**: every additional request
  (including other opencode sessions) waits for the in-flight turn to finish;
  no 429 overload rejection exists in the entrypoint code read — saturation
  manifests as queue delay, not backpressure. (`--enable-prefix-caching`
  still pays for sequential turns of the same session: APC "caches the KV
  cache of existing queries, so that a new query … shares the same prefix" can
  skip recompute, `docs/features/automatic_prefix_caching.md:3-5` @
  `ced6857`; it reduces prefill only, not decode — "Limits" section.)
- **Error semantics**: OpenAI envelope `{"error":{"message","type","param"}}`
  (`exception_handling/error_response.py:87-95` @ `ced6857`): 400
  `BadRequestError` (validation, overflow), 404 `NotFoundError` (unknown
  model), 500 `InternalServerError`, 501 `NotImplementedError`; mid-stream
  generation errors ride the SSE as an error event plus `[DONE]` (§2.4). All
  layouts parse under opencode's classifier (companion doc §4.2).

### 2.8 Model coverage / quantization

- **Any HF model vLLM supports** (text-generation and pooling), loaded from
  HF or local cache — the Qwen3.8 NVFP4 checkpoint in the user's block is an
  `unsloth`-quantized safetensors artifact, not a `.ninfer`. Quantization
  follows the model's `quantization_config` or `--quantization`
  (`vllm/config/model.py:228-233` @ `ced6857`): NVFP4, FP8, AWQ, GPTQ, etc.
- **Structured outputs**: backend-constrained decoding (xgrammar/outlines/
  guidance) for `response_format`/guided fields by default
  (`docs/features/tool_calling.md:86-101` — "You are guaranteed a
  validly-parsable function call" for named/required choice).
- **Responses API** (`/v1/responses`), batch completions, LoRA adapters,
  per-request metrics, and the `include_reasoning` suppression knob
  (reasoning_outputs.md:356-362) — surfaces opencode does not use on this
  provider but other clients could.
- **Single model per instance** (§2.1) — multi-model coverage requires
  multiple instances plus a router (the FAQ's prescription), which is what the
  host already does at the caddy level for embeddings.

### 2.9 How opencode users configure it

- **No dedicated opencode docs section**: the providers page (source at
  `packages/web/src/content/docs/providers.mdx` @ `527f0b931`, published
  <https://opencode.ai/docs/providers/>) has sections for llama.cpp, LM
  Studio, and Ollama, and **zero references to vLLM** (case-insensitive search
  of the entire docs tree, this session). opencode users point a generic
  OpenAI-compatible custom provider at it — the same `@ai-sdk/openai-compatible`
  shape as the llama.cpp section (§1.9) with `baseURL` =
  `http://127.0.0.1:8000/v1` (or the caddy-fronted URL) and operator-declared
  `limit.context`/`limit.output` (companion doc §5.1: model metadata comes from
  config, never the backend).
- The vLLM docs themselves show the OpenAI-client shape (`base_url=
  http://localhost:8000/v1`, `docs/features/tool_calling.md:24` @ `ced6857`),
  which is exactly what the AI-SDK openai-compatible package emits.
- No discovery plugin exists for vLLM (only Ollama/LM Studio; §1.9), so
  `/v1/models` is never consulted even though the endpoint exists.

---

## 3. Operational reference (user configs)

Both blocks live in `nixos-configs/users/kido/llm.nix` and share the
identical-behavior contract (llm.nix:316-325, 408-419):

> "Same host port (8827) and same served model id (Qwen3.8-27B) as
> llama-cpp-qwen38-27b so tools hitting ai.kido.ws (caddy → 8827) work
> identically against either backend. Mutual Conflicts with the llama.cpp
> services (GPU VRAM: only one at a time)."

| Dimension | `llama-cpp-qwen38-27b` (llm.nix:263-314) | `vllm-qwen38-27b-nvfp4` (llm.nix:341-406) | (ninfer-yarn unit, for contrast: llm.nix:483-579) |
|---|---|---|---|
| Model artifact | `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL` (GGUF, HF cache) | `unsloth/Qwen3.8-27B-NVFP4` (safetensors NVFP4, HF cache) | `neroued/Qwen3.8-27B-nvfp4-NInfer` (`.ninfer` v3) |
| Port / id | `127.0.0.1:8827`, `--alias Qwen3.8-27B` | `8827:8000`, `--served-model-name Qwen3.8-27B` | `127.0.0.1:8827`, `--model-id Qwen3.8-27B` |
| Window | `-c 0` = GGUF-declared (native 262144-class; no extension flag) | `--max-model-len 262144` (VRAM-bound, documented binding constraint) | `--max-context 446902` (YaRN extension above 262144) |
| KV | q8_0 K/V, `--kv-unified` | fp8_e4m3, pool pinned 9.0e9 B | nvfp4, `--host-kv-mib 24576` host cache |
| Concurrency | `--parallel 1` slot, cont-batching | `--max-num-seqs 1` (single-stream) | `--max-concurrency 4` + 64 pending, 429 when full |
| Spec decoding | `draft-mtp` ×3, q8_0 draft KV | none — MTP OOMs (llm.nix:329-333) | MTP4 + lm-head-draft |
| Vision | on (`--image-min-tokens 1024`, mmproj on CPU) | off (`--language-model-only`) | on (`--vision`) |
| Sampling flags | temp 1.0 / top-p 0.95 / top-k 20 / min-p 0.0 (llm.nix:285-288) | model defaults (`generation_config.json` applied unless `--generation-config vllm`) | registered thinking preset 1.0/0.95/20 |
| Stability note | Xid 8 watchdog fix recorded (llm.nix:255-262) | graph-capture OOM boundary recorded (llm.nix:336-338) | YaRN + MTP tuning history |
| Restart | `always`, `RestartSec=0` | `always`, `RestartSec=10` | `always`, `RestartSec=5` |

What each config was chosen to satisfy: the llama.cpp block is the GGUF-route
A/B partner (same port/id, UD-Q4_K_XL quant, MTP on, one slot, q8_0 KV to fit
the window in 32 GB); the vLLM block is the HF-NVFP4-route partner tuned to
the *maximum context that physically fits* (262144) with every optional VRAM
consumer (MTP, vision) disabled and concurrency pinned to 1 to stay inside the
KV budget. Both were validated against the same "tools work identically"
bar — i.e., opencode traffic hitting `ai.kido.ws/v1/chat/completions` behaves
the same against whichever unit holds the GPU.

---

## 4. Coverage delta vs ninfer-yarn

Headline: **neither incumbent is a strict superset of ninfer-yarn for this
deployment; the delta cuts both ways** — the incumbents win on model coverage
and some contract niceties, ninfer-yarn wins on window size, output-bound
safety, and overload semantics.

| Area | llama.cpp | vLLM | ninfer-yarn (companion doc) | Delta direction |
|---|---|---|---|---|
| Models served | any GGUF, any quant, multi-model router | any HF model vLLM supports (NVFP4/FP8/AWQ), structured outputs, Responses API | Qwen3.5/3.6/3.8 `.ninfer` family only | incumbents win (breadth) |
| Context window (this deployment) | GGUF native (262144-class, no extension) | 262144 (VRAM hard ceiling) | **446902 via YaRN** | ninfer-yarn wins |
| Default output bound when client sends neither max field | none (`n_predict=-1`) | none (remaining window) | **32768 (`--default-max-tokens`)** | ninfer-yarn wins (incident-safe) |
| `max_tokens`+`max_completion_tokens` both sent | `max_completion_tokens` wins | `max_completion_tokens` wins | `max_completion_tokens` wins | all match opencode's need |
| Tool-parse failure shape | demote to content (markup leaks; silent-stall class) | drop malformed preamble bytes; emit truncated calls; no leakage; **undeclared names accepted** | demote to content (markup leaks) + fail-closed `undeclared_tool` class | vLLM avoids leakage; ninfer-yarn is strictest on names |
| Reasoning wire field | `reasoning_content` (first in opencode's chain) | `reasoning` (second; replay normalized) | `reasoning_content` | all work with opencode |
| `usage` chunk richness | prompt-side only (no `reasoning_tokens`) | **full** (`cached_tokens` + `reasoning_tokens`) | full (`cached_tokens` + `reasoning_tokens`) | vLLM == ninfer-yarn > llama.cpp |
| `finish_reason` values | stop/length/tool_calls | stop/length/content_filter/tool_calls | stop/length/tool_calls | all ⊆ opencode's accepted set |
| `[DONE]` after in-band stream error | no (closes) | **yes** | no (closes) | vLLM differs; opencode fails the stream either way |
| Overflow 4xx → opencode classification | `exceed_context_size_error`, phrase matches | 400 "exceeds model's maximum context length", phrase matches | `context_length_exceeded` code | all classify as `context-overflow` (non-retryable) |
| Under contention | slot queue, **no 429**, unbounded wait (timeout 3600 s) | scheduler wait, **no 429** | **429 `server_overloaded`** when C+P full; 503 queue timeout | ninfer-yarn is the only one with backpressure |
| Concurrency in the user's configs | 1 slot | 1 seq | 4 lanes + 64 pending | ninfer-yarn serves the most concurrent opencode sessions |
| Spec decoding in the user's configs | MTP ×3 | none (fits nothing) | MTP4 + lm-head-draft | llama.cpp/ninfer-yarn win |
| opencode docs support | **dedicated section** | none (generic custom provider) | none (generic custom provider) | llama.cpp is the only documented local backend |

Verification notes: all llama.cpp claims cite raw files from master
`bed0a85660` fetched this session (the repo moved its server docs from
`examples/server/README.md` to `tools/server/README.md`); all vLLM claims cite
the `ced6857` clone (v0.30.0-current) plus the stable docs.vllm.ai pages
verified reachable this session; the user's `:latest` vLLM image resolves to
whatever the registry had at pull time — the profile is versioned to
`ced6857` and the config-comment era (v0.27.1) is noted where behavior could
differ (the `qwen3_xml` parser name and the `qwen3` reasoning parser exist in
both). Where official docs were silent on an opencode-relevant detail (e.g.,
llama.cpp queue-depth limits, vLLM wait-queue bounds), this doc says so
rather than inferring.
</think>
