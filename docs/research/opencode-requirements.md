# What opencode requires from a model backend

Wayfinder map #139, ticket #140 — the requirement side of the opencode ⇄ ninfer-yarn
compatibility matrix.

**Sources of record**

- opencode source, tag `v2.0.22` (commit `527f0b931`, "release: v2.0.22"),
  `github.com/sst/opencode` (the repo now lives at `github.com/anomalyco/opencode`).
  All `file:line` citations below are against this tag. The locally installed binary is
  `opencode v2.0.22` (`/etc/profiles/per-user/kido/bin/opencode`), so the tag matches
  the deployment.
- opencode docs: <https://opencode.ai/docs/providers/> ("Custom provider" section,
  page last updated 2026-10-02).
- Local user config: `~/.config/opencode/opencode.jsonc` (provider `ai.kido.ws`), and the
  local opencode credential store (`~/.local/share/opencode/opencode.db`, table
  `credential`).

**Protocol answer (TL;DR)**

A custom self-hosted provider configured with `package: "aisdk:@ai-sdk/openai-compatible"`
speaks exactly one wire protocol: **OpenAI Chat Completions**,
`POST {baseURL}/chat/completions`, **always `stream: true`**, SSE-framed with a
`[DONE]` sentinel (`packages/ai/src/providers/openai-compatible.ts:24`,
`packages/ai/src/protocols/openai-compatible-chat.ts:16-22`). It never calls
`/v1/models` and never calls `/v1/embeddings` (see §1.6, §1.7). The `/v1/responses`
protocol is only reached by providers configured with the `@ai-sdk/openai` package,
and Anthropic `/v1/messages` only by `@ai-sdk/anthropic`-family packages
(§6). The package name `@ai-sdk/openai-compatible` is never loaded from npm: it is
remapped to the in-tree implementation `@opencode/ai/providers/openai-compatible`
(`packages/core/src/aisdk-native.ts:67`).

---

## 1. Protocol and provider wiring

### 1.1 Package → route mapping

| Config `package` | Route(s) | Wire endpoint |
|---|---|---|
| `aisdk:@ai-sdk/openai-compatible` | `OpenAICompatibleChat` only | `{baseURL}/chat/completions` |
| `aisdk:@ai-sdk/openai` | `OpenAIResponses` default, `OpenAIChat` per-model | `/responses` by default; individual catalog models may select the chat route |
| `aisdk:@ai-sdk/anthropic` | `AnthropicMessages` only | `{baseURL}/messages` |

- `packages/ai/src/providers/openai-compatible.ts:24` — `routes = [OpenAICompatibleChat.route]`.
- `packages/ai/src/providers/openai.ts:27` — `routes = [OpenAIResponses.route, OpenAIChat.route]`;
  the default `model()` factory is the responses route, with `chat` available per model
  (`openai.ts:97-131`); the catalog's per-model `provider.npm` selects the package
  (`packages/core/src/models-dev.ts:204`).
- `packages/ai/src/providers/anthropic.ts:15` — `routes = [AnthropicMessages.route]`.
- `packages/ai/src/protocols/openai-compatible-chat.ts:16-22` — endpoint
  `Endpoint.path("/chat/completions")`, protocol and framing reused from `openai-chat.ts`.
- Docs concur: "use `@ai-sdk/openai-compatible` (for `/v1/chat/completions`); if a
  model uses `/v1/responses`, use `@ai-sdk/openai`"
  (<https://opencode.ai/docs/providers/#custom-provider>).

The endpoint is `baseURL` + path; the user's provider sets `baseURL:
"https://ai.kido.ws/v1"`, so the hit is
`https://ai.kido.ws/v1/chat/completions` (`Endpoint.path` semantics in
`packages/ai/src/route/endpoint.ts`, composed in
`packages/ai/src/protocols/openai-compatible.ts`'s `configure`, lines 26-41).

### 1.2 Request method and body encoding

Single `POST` with a JSON body
(`packages/ai/src/route/transport/http.ts:81-97`, `ProviderShared.jsonPost`).
A config-level `body` overlay (model-level `body`/`settings.body` JSON records) is
deep-merged **over** the protocol-built body just before encoding
(`packages/ai/src/route/transport/http.ts:41-49`; overlay fields declared in
`packages/schema/src/config/provider.ts:31-41`). This is how the user's
`body: { "max_tokens": 32768 }` reaches the wire (§2.2).

### 1.3 Authentication

- Default for OpenAI-compatible providers: `Authorization: Bearer <apiKey>`
  (`packages/ai/src/route/auth.ts:50`, `packages/ai/src/route/auth-options.ts:51-62`).
  The key comes from config `settings.apiKey` or from opencode's credential store
  (`/connect`; `packages/core/src/model-resolver.ts:308-319` injects it as `apiKey`).
- **No credential and no `apiKey` in config ⇒ the request is sent with no
  `Authorization` header at all**: providers defined in config are force-enabled
  (`activation = "enabled"`,
  `packages/core/src/config/plugin/provider.ts:67`), so the resolver's swap to
  `Auth.none` applies whenever the provider is active, has no stored credential,
  and the config carries no `apiKey`/`authToken`/`accessToken`
  (`packages/core/src/model-resolver.ts:380-384`; `hasConfiguredAuth` at
  lines 422-425). Verified for the deployment: the local credential DB contains
  entries for `deepseek`, `headroom`, `Selectel` but **no `ai.kido.ws`** entry, and
  the user's config declares no `apiKey` — so opencode currently posts to
  `ai.kido.ws/v1/chat/completions` unauthenticated (no `Authorization` header).
- If a credential is expected but missing in the non-enabled path, the failure is an
  `AuthenticationError` (non-retryable, §4.2)
  (`packages/ai/src/route/auth.ts:157-162`).
- Auth is applied per request at send time, so a 401/403 from the backend is handled
  in §4, not at startup: opencode performs no startup handshake.

### 1.4 Headers opencode sends on every request

From `packages/core/src/session/model-request.ts:277-289` (plus auth):

| Header | Value |
|---|---|
| `User-Agent` | `opencode/<channel>/<version>/<name>`, e.g. `opencode/cli/2.0.22/opencode` (`packages/core/src/app.ts:24-26`) |
| `x-opencode-session-id` | session ID |
| `x-opencode-parent-session-id` | parent session (subagents only) |
| `x-session-affinity` / `X-Session-Id` / `x-opencode-session` | session affinity key |
| `x-parent-session-id` | parent session (subagents only) |
| `x-opencode-project` | project ID |
| `x-opencode-client` | client name |

A backend must tolerate all of these as opaque. Config-level `headers` overlays can
add more (`packages/schema/src/config/provider.ts:31-41`).

### 1.5 Timeouts

`packages/ai/src/route/transport/http.ts:100-143`,
`packages/ai/src/schema/options.ts:56`:

- `headerTimeout`: default **300 s** — time until response headers; failure is a
  transport `Timeout` (operation `request`).
- `chunkTimeout`: default **300 s** — max silence between streamed chunks; failure is
  a transport `Timeout` (operation `read`) and is treated as an *interrupted stream*
  (§4.5).
- overall `timeout`: **no default** (unbounded) unless the provider config sets
  `settings.timeout` (`packages/schema/src/config/provider.ts:9-19`).
- These are per-provider config: `timeout` / `headerTimeout` / `chunkTimeout`.

### 1.6 `/v1/models` — never fetched

No production code path fetches a model list from a custom OpenAI-compatible
provider. The only in-tree production `/models` call is the LM Studio discovery
plugin, which hits LM Studio's `{prefix}/api/v1/models`
(`packages/core/src/plugin/provider/lmstudio.ts:162`) and is irrelevant to a custom
provider. The remaining `/v1/models` references are dev tooling
(`packages/ai/script/setup-recording-env.ts`) and opencode's own web console
(`packages/console/...`). **Model metadata comes from config or the built-in
catalog, never from the backend** (§5).

### 1.7 Embeddings — never called

Case-insensitive search for `embedding` across `packages/ai/src` and
`packages/core/src` finds no production embeddings route in v2.0.22: only a test-seam
comment (`packages/core/src/session/runner/model.ts:45`), the LM Studio model *type*
vocabulary (`packages/core/src/plugin/provider/lmstudio.ts:13`, which still never
issues an embeddings request), and a commented V2-SDK stub
(`packages/core/src/github-copilot/copilot-provider.ts:44`). **The backend does not
need `/v1/embeddings`.**

---

## 2. Request shapes (OpenAI Chat Completions)

### 2.1 Body fields

Wire body schema: `packages/ai/src/protocols/openai-chat.ts:181-200`
(`bodyFields`); construction: lines 807-857 (`fromRequest`).

| Field | Sent | Notes / citation |
|---|---|---|
| `model` | always | the catalog `modelID` (config `modelID`, falling back to the config key); user's provider sends `"Qwen3.8-27B"` (`openai-chat.ts:834`, `packages/core/src/model.ts:291-297`, `packages/schema/src/config/provider.ts:70`) |
| `messages` | always | lowered array, §2.3 (`openai-chat.ts:835`) |
| `tools` | when active tools; `[]` when there is tool history but no active tools; omitted otherwise | §2.4 (`openai-chat.ts:836-841`) |
| `tool_choice` | only when opencode sets it: `"none"` on the final (max-steps) request | §2.4 (`openai-chat.ts:842`, `packages/core/src/session/runner/llm.ts:223,237-241`) |
| `stream` | **always `true`** (schema literal) | `openai-chat.ts:186,843` — opencode has no non-streaming path for this provider |
| `stream_options` | always `{ include_usage: true }` | `openai-chat.ts:187,761,844` (`supportsUsageInStreaming` defaults to true) |
| `store` | `false` for this provider | `detectSupportsStore` returns true for unknown providers, and stateless `store:false` is then sent: `openai-chat.ts:719-759,797-801`. Backend must tolerate the field. |
| `prompt_cache_key` | **not sent** by default | only when `compatibility.supportsPromptCacheKey` is set (`openai-chat.ts:794-795,802`); 64-char clamp (`packages/ai/src/protocols/shared.ts:40-49`). Not set in the user's config. |
| `reasoning_effort` | only when a variant with `settings.reasoningEffort` is selected | §2.6 (`openai-chat.ts:803`) |
| `tool_stream` | never for this provider (ZAI-only) | `openai-chat.ts:773-787,845` |
| `max_completion_tokens` | for `primary`/`compaction` requests | §2.2 (`openai-chat.ts:846-848`) |
| `max_tokens` | **only via the user's config `body` overlay** | §2.2 (`openai-compatible.ts:48-59` → `http.ts:41-49`) |
| `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `stop` | only if set (never by opencode defaults) | §2.5 (`openai-chat.ts:849-854`) |

Never sent on this route: `n`, `logprobs`, `parallel_tool_calls`, `response_format`,
`metadata`, `user`.

### 2.2 `max_tokens` semantics — opencode 2.0.22 DOES send it

This supersedes the earlier standing fact "opencode 2.x sends no `max_tokens`":

- For agent steps (`kind: "primary"`) and compaction summaries
  (`kind: "compaction"`), opencode sets
  `generation.maxTokens = outputLimit(model.limit, kind, inputTokens)`
  (`packages/core/src/session/model-request.ts:249-258`).
- `outputLimit` = `min(limit.output (fallback 32000), 256000)`, then fitted to the
  prompt's remaining room in the declared context window:
  `min(requested, max(1024, context − measured − ⌈1.15 × estimated⌉))`
  (`model-request.ts:43-57,88-98`). Compaction summaries are additionally capped at
  32000.
- For the OpenAI-compatible package the field name is
  **`max_completion_tokens`** (detection table: `openai-chat.ts:685-717` — unknown
  providers/baseURLs default to `max_completion_tokens`).
- `generate` and `title` requests carry **no max field at all**
  (`model-request.ts:256`: only `primary`/`compaction` get `maxTokens`).
- The user's config additionally overlays `body.max_tokens: 32768`
  (`~/.config/opencode/opencode.jsonc` → merged in `http.ts:41-49`), so the user's
  provider can receive **both** `max_completion_tokens` (computed, primary/compaction
  requests) and `max_tokens` (fixed 32768, every request from this provider). A
  backend must tolerate both fields simultaneously and must honor at least one —
  the 2026-09-22 truncation incident was exactly the case where neither bound
  reached the backend.

### 2.3 Message lowering

`openai-chat.ts:416-667` (`lowerUserMessage`, `lowerAssistantMessage`,
`lowerToolMessages`, `lowerMessages`):

- **System**: all system text joined with `\n` into a **first `{"role":"system"}`
  message** (single string content) (`openai-chat.ts:554-569, 569`
  `shared.ts:138` `joinText`). Chronological mid-conversation system updates are
  wrapped as `<system-update>…</system-update>` and demoted into user messages
  (`shared.ts:149-176`, `openai-chat.ts:608-649`) — the backend will see user
  messages containing that XML wrapper.
- **User**: plain string when all parts are text; otherwise an array of
  `{"type":"text"|"image_url"|"file"}` content parts (§2.7)
  (`openai-chat.ts:416-438,124-142`).
- **Assistant replay**: `content` is a joined string, or **`null` when the message
  carries only tool calls** (`openai-chat.ts:491-497`); `tool_calls` is an array of
  `{id, type:"function", function:{name, arguments}}` where `arguments` is the
  **complete JSON string** re-encoded from the parsed input
  (`openai-chat.ts:357-368`); a `reasoning_content` string is appended when the
  message contains reasoning (default field name;
  `openai-chat.ts:465-502`). The backend must accept all of: string/null content,
  `tool_calls`, and `reasoning_content` on assistant messages.
- **Tool results**: one `{"role":"tool","tool_call_id","content"}` message per tool
  call; content is the result text (multi-line results joined with `\n`)
  (`openai-chat.ts:505-540`). Tool-result file attachments are emitted as a
  following user message with media parts (`openai-chat.ts:529-533,600-605`).
- Consecutive identical-role user messages are merged
  (`openai-chat.ts:621-649`); empty assistant text-only messages are dropped
  (`openai-chat.ts:651-655`).

### 2.4 Tool declaration and `tool_choice`

- Each tool is `{"type":"function","function":{"name","description","parameters",
  "strict": false}}` — `strict: false` is added for every provider not in the
  exclusion list, which includes `ai.kido.ws`
  (`openai-chat.ts:338-347,763-771`). `parameters` is the tool's JSON Schema
  (`ToolDefinition.inputSchema`).
- Namespaces are flattened to `namespace_tool` leaf names
  (`shared.ts:295-302`).
- Builtin tool names in v2.0.22: `shell`, `read`, `write`, `edit`, `glob`, `grep`,
  `question`, `subagent`, `webfetch`, `websearch`, `skill`, `patch`
  (`packages/core/src/tool/plugin/<name>.ts` each `export const name = ...`), plus
  MCP tools.
- `tool_choice`: opencode sends it only as `"none"` on the final step when the agent
  has hit its configured `steps` limit, together with an appended
  `MAX_STEPS_PROMPT` assistant message asking for a text-only summary
  (`packages/core/src/session/runner/llm.ts:223,237-241`,
  `packages/core/src/session/runner/max-steps.ts:1-19`, `openai-chat.ts:842,349-355`,
  `packages/core/src/session/runner/step.ts:88-90`). `auto`/`required`/named-tool
  choices exist in the schema but are not emitted by the default agent flow.

### 2.5 Sampling parameters

`temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `seed`, `stop` are
copied from `GenerationOptions` only when a value exists
(`openai-chat.ts:849-854`, `packages/ai/src/schema/options.ts:94-103`). Nothing in
the default agent flow sets them — `temperature` is explicitly rejected as a
top-level model config key
(`packages/core/src/config/normalize.ts:55`) — so for the user's provider these
fields are **absent** unless a plugin or provider option injects them.

### 2.6 Reasoning / thinking parameters

- The user's model variants (`low`/`medium`/`xhigh`) each set
  `settings.reasoningEffort`, which flows: variant settings → provider route
  defaults → request `providerOptions` → body `reasoning_effort`
  (`packages/core/src/model-resolver.ts:137-161`,
  `packages/ai/src/route/client.ts:265-279`,
  `packages/ai/src/protocols/utils/open-responses-options.ts:60,78-79`,
  `openai-chat.ts:803`). Accepted effort values include `none|minimal|low|medium|high|xhigh|max`
  (`packages/ai/src/schema/options.ts:170`). With no variant selected the field is
  absent. **The current session runs on the `xhigh` variant** (its system prompt
  declares "Reasoning effort is set to xhigh"), i.e. live requests carry
  `reasoning_effort: "xhigh"`.
- Model-reference syntax for variants: `provider/model#variant`
  (`packages/schema/src/model.ts:18-30`).
- Replay: assistant messages that contained reasoning are replayed with a
  `reasoning_content` field by default (`openai-chat.ts:465-502`); a custom
  `compatibility.reasoningField` can rename it. The backend may use or ignore it.

### 2.7 Image / media content

- Gated by declared capabilities: media parts are replaced by an
  "ERROR: Cannot read …" text placeholder when the model's
  `capabilities.input` lacks the modality
  (`packages/core/src/session/model-request.ts:122-159`). The user's model declares
  `input: ["text","image"]`, so images are sent.
- Wire shape: `{"type":"image_url","image_url":{"url"}}` where `url` is the original
  URL (http/https) or an inline **data URL** (base64); PDFs go as
  `{"type":"file","file":{"filename","file_data"}}` data URLs; other documents are a
  client-side invalid-request error
  (`openai-chat.ts:370-385,124-142`, `shared.ts:199-222`).
- Inline image budget: total images over **25 MiB** trigger removal down to a
  **15 MiB** target, replaced with a text note
  (`model-request.ts:43-46,161-212`).

### 2.8 Prompt-caching fields

For the OpenAI Chat route: `cache_control` breakpoints are only attached when the
model's `compatibility` enables them (not in the user's config); `store: false` is
sent (§2.1); `prompt_cache_key` is off (§2.1). So the only caching-related field a
backend sees on this route is `store: false` plus whatever headers the client adds.
The session-affinity headers (§1.4) are the intended cache-keying signal for
prefix-stable backends.

### 2.9 Representative request (user's provider, agent step)

```json
POST https://ai.kido.ws/v1/chat/completions
Content-Type: application/json
User-Agent: opencode/cli/2.0.22/opencode
x-opencode-session-id: ses_…
x-session-affinity: …
(… other §1.4 headers …)

{
  "model": "Qwen3.8-27B",
  "messages": [
    { "role": "system", "content": "…agent system prompt…\n…initial context…" },
    { "role": "user", "content": "…" },
    { "role": "assistant", "content": null, "tool_calls": [
      { "id": "call_…", "type": "function",
        "function": { "name": "shell", "arguments": "{\"command\":\"…\"}" } }
    ], "reasoning_content": "…" },
    { "role": "tool", "tool_call_id": "call_…", "content": "…" }
  ],
  "tools": [
    { "type": "function",
      "function": { "name": "shell", "description": "…",
        "parameters": { "$schema": "…", "type": "object", "…": "…" },
        "strict": false } }
  ],
  "stream": true,
  "stream_options": { "include_usage": true },
  "store": false,
  "max_completion_tokens": 32768,
  "max_tokens": 32768,
  "reasoning_effort": "xhigh"
}
```

(`tool_choice: "none"` appears only on the max-steps final step, §2.4.)

---

## 3. Response requirements (streaming)

### 3.1 SSE framing

- Standard SSE events; each `data:` payload is one JSON event; the stream must
  terminate with `data: [DONE]`
  (`packages/ai/src/route/framing.ts:87-91` `sseWithDone`,
  `openai-chat.ts:297-298,1300-1301`).
- Tolerated and ignored: empty `data:` events, `data: null` (proxy flush),
  `data: : keepalive` comments, SSE retry control events, comment lines
  (`framing.ts:32-34,68-80`). A malformed SSE frame is a provider-output error
  (`framing.ts:55-65`).
- Only `choices[0]` is consumed (`openai-chat.ts:1059`).

### 3.2 Delta shape

Per-event schema: `openai-chat.ts:243-296`:

```
{ "choices": [ { "delta": { … }, "finish_reason": "…", "usage": { … } } ],
  "usage": { … }, "error": { … } }
```

`delta` fields consumed: `content` (text), `refusal` (streamed into the same text
channel, `openai-chat.ts:1125`), `reasoning_content` / `reasoning` /
`reasoning_text` / `reasoning_details` (§3.6), `tool_calls` (§3.3). Extra fields are
ignored (schemas are `StructWithRest`).

**Content after the finish reason is an error**: once a choice has carried
`finish_reason`, any subsequent event with non-empty delta content (text, refusal,
reasoning, or tool deltas) fails the stream with
`"OpenAI Chat received content after the finish reason"`
(`openai-chat.ts:1084-1097`). A usage-only event after finish is fine
(`openai-chat.ts:1090-1098`).

### 3.3 Tool-call streaming contract

- Tool calls stream as `delta.tool_calls[]` entries with `index`, `id`,
  `function.name`, `function.arguments` (a JSON **string fragment**); arguments
  accumulate across deltas per index
  (`openai-chat.ts:1129-1178`,
  `packages/ai/src/protocols/utils/tool-stream.ts:146-174`).
- **Identity**: `id` and `function.name` must both be present by the time the
  response finishes. Deltas lacking either are parked as *pending*; if
  `finish_reason` arrives (other than `length`/`content_filter`) while a pending
  tool still lacks `id` or `name`, the stream fails with
  `"OpenAI Chat tool call delta is missing id or name"`
  (`openai-chat.ts:1180-1191,1143-1149`). Standard OpenAI behavior (identity on the
  first delta of each index) satisfies this; opencode also tolerates identity
  arriving slightly later within the same stream.
- **Parallel calls**: multiple indices in one response are supported and executed
  concurrently by the agent (`openai-chat.ts:1131-1142` index matching;
  `packages/core/src/session/runner/step.ts:117-129` one fiber per tool call).
- **Partial arguments**: the accumulated JSON string is live-parsed with a
  partial-JSON parser (unterminated strings/objects/arrays/numbers accepted,
  control-char repair), so truncated tool calls degrade to a best-effort object
  (`packages/ai/src/protocols/utils/partial-json.ts:38-59,102-273`,
  `tool-stream.ts:72-97`). Empty argument string is treated as `{}`
  (`packages/ai/src/protocols/shared.ts:184-185`). A tool call truncated by
  `finish_reason: "length"` or `"content_filter"` is accepted (incomplete tools
  tolerated, `openai-chat.ts:1180,1193-1200`); a hard stream death mid-call is
  handled per §4.5 (partial output persisted, step retried/continued).
- Tool-call IDs are passed through unchanged for this provider
  (no sanitization; `openai-chat.ts:583-593`).
- OpenAI Chat has no per-tool stop event; all accumulated calls finalize together
  at the terminal `finish_reason` (`tool-stream.ts:232-247`).

### 3.4 `finish_reason`

`openai-chat.ts:865-899`:

| Wire value | opencode result |
|---|---|
| `stop`, `end` | `stop` |
| `length` | `length` (truncation; step completes, partial output kept) |
| `content_filter` | `content-filter` |
| `function_call`, `tool_calls` | `tool-calls` |
| `error` | provider error (`UnknownProvider`) |
| `network_error` | provider error (`ProviderInternal`) |
| anything else | **stream fails** — `UnknownProviderError` "Provider finish_reason: …" |

Two structural rules:

1. **A `finish_reason` is mandatory**: `requireFinishReason` defaults to `true` for
   this route; a stream that ends at `[DONE]` without ever carrying one fails with
   `InvalidProviderOutputError` classified `incomplete-stream`
   (`openai-chat.ts:1224-1232,1298`). A response ending in `finish_reason:
   "unknown"` (or an unknown value that maps to error) is likewise treated as an
   interrupted stream by the runner
   (`packages/core/src/session/runner/step.ts:163-172,281-285`).
2. `finish_reason: "stop"` + tool calls present is re-normalized to `tool-calls`
   (`openai-chat.ts:1239-1245`). `native_finish_reason`, when present, is recorded
   as the raw value (`openai-chat.ts:273,1067-1073`).

### 3.5 Usage accounting

- Read from top-level `usage` **or** `choices[0].usage`
  (`openai-chat.ts:1060-1066`), typically the final event requested via
  `stream_options.include_usage`.
- Mapped fields (`openai-chat.ts:210-241,910-933`): `prompt_tokens`,
  `completion_tokens`, `total_tokens`; cache breakdown from
  `prompt_tokens_details.cached_tokens` / `prompt_tokens_details.cache_write_tokens`
  / `prompt_cache_hit_tokens` / `cached_tokens` /
  `cache_read_input_tokens` / `cache_created_input_tokens`; reasoning subset from
  `completion_tokens_details.reasoning_tokens`. Non-cached input is derived as
  `prompt_tokens − cached − cache_write`, clamped at zero
  (`shared.ts:95-99`).
- **Missing usage is not fatal**, but it degrades context management: the
  runner's prompt-size estimate then uses a purely local character-based estimate
  (`measured: 0`) for fitting `max_completion_tokens` (§2.2) and for compaction
  thresholds (`packages/core/src/session/compaction.ts:862-884`), and step cost
  records carry no tokens (`packages/core/src/session/runner/step.ts:236-238`).
  A backend that reports accurate `usage` in the final stream event is required for
  correct long-session behavior.

### 3.6 Reasoning blocks

- One response-wide reasoning channel. Delta text is taken from the first non-empty
  of: the configured field, `reasoning_content`, `reasoning`, `reasoning_text`
  (`openai-chat.ts:945-957`); structured `reasoning_details` (OpenRouter/Kimi
  dialects) are merged and replayable (`openai-chat.ts:93-119,1102-1118`).
- Reasoning deltas may interleave with text and tool-call deltas; the block is
  closed once at stream finish (`openai-chat.ts:1120-1123,1248-1263`).
- On replay, reasoning is sent back as `reasoning_content` (§2.3/§2.6); the
  backend may consume or ignore it.

### 3.7 In-band errors

A 2xx stream may carry an `error: {message, code?}` event; it is classified exactly
like an HTTP failure (status from `error.code` when numeric) and fails the stream
(`openai-chat.ts:280-286,1048-1057`).

---

## 4. Failure semantics

### 4.1 HTTP status classification

`packages/ai/src/provider-error.ts:229-299` (`classifyProviderFailure`), applied to
every non-2xx response (body fully read first,
`packages/ai/src/route/executor.ts:100-112`):

| Signal | Class | Retryable? (§4.2) |
|---|---|---|
| 4xx + context-overflow code/phrase (or `400/413 (no body)`) | `InvalidRequest` / `context-overflow` | **No** — fatal to the step; triggers overflow recovery if auto-compaction is on (§4.5) |
| 413 or payload-too-large text | `InvalidRequest` / `payload-too-large` | No |
| content-policy codes/text (`content_filter`, `refusal`, …) | `ContentPolicy` | No |
| 402, quota codes, or 429 + quota text | `QuotaExceeded` | No |
| 401 / 403, auth codes, or 400 + "incorrect api key" | `Authentication` | No |
| 429, rate-limit codes/text | `RateLimit` | **Yes** (`Retry-After` honored, capped at 15 min, `packages/core/src/session/runner/retry.ts:31-40`) |
| 408 / 409 / any 5xx; or server-error codes/text without a contradicting 4xx status | `ProviderInternal` | **Yes** |
| invalid-request codes (`model_not_found`, `invalid_request_error`, …) | `InvalidRequest` | No |
| any other 4xx | `InvalidRequest` | No |
| unrecognized | `UnknownProvider` | **Yes** |

Context-overflow detection is phrase-based on the error message **and** the raw
body (`provider-error.ts:16-54`, e.g. "prompt is too long", "exceeds the context
window", "context_length_exceeded") — a backend rejecting an over-window prompt
should return a 4xx whose message matches one of these patterns (or the OpenAI
`context_length_exceeded` code) so opencode classifies it as overflow rather than a
generic (still non-retryable) 4xx.

### 4.2 Error body layouts and retry control

- Recognized message layouts, in priority order: `error.message`, `error` (string),
  `message`, AWS `Message`, RFC 9457 `detail`, `errors[0].message`
  (`provider-error.ts:189-207`). Unrecognized bodies are shown raw (truncated at
  2000 chars, `executor.ts:87-98`).
- Error codes are read from `code`, `error.code`, `error.type`, `error.status`,
  `error.error_type`, `response.error.code`, Google `details[].reason`, …
  (`provider-error.ts:301-329`) — OpenAI-style
  `{"error":{"message","code","type"}}` bodies are fully understood.
- Response header `x-should-retry: true|false` overrides retryability
  (`provider-error.ts:75-78`); `Retry-After` and `rate limit` headers are parsed for
  backoff (`executor.ts:40-78`, `retry.ts:34-40`).

### 4.3 Retry policy (session step level)

`packages/core/src/session/runner/retry.ts:42-62`:

- Exponential backoff from **2 s**, per-gap cap **10 s**, at most **10 retries**
  (~84 s total), jittered.
- Provider-requested `Retry-After` (RateLimit/ProviderInternal) is a floor on the
  delay, capped at **15 min**.
- Transport timeouts: at most **3** timeout retries.
- Plugins may override any decision (session `retry` hook, `retry.ts:78-91`).
- There is no HTTP-layer retry: each attempt is one full request.

### 4.4 What is retryable

`provider-error.ts:75-110` (`isRetryable`): `RateLimit`, `ProviderInternal`,
`UnknownProvider` → yes; transport failures → yes unless the write was rejected by
the backend; `InvalidProviderOutput` → yes **only** when classified
`incomplete-stream` (dead stream); `Authentication`, `QuotaExceeded`,
`ContentPolicy`, `InvalidRequest`, `UnsupportedOperation`, `NoRoute`, `Timeout` →
no.

### 4.5 Agent-loop behavior on a dead stream

`packages/core/src/session/runner/step.ts` + `…/llm.ts`:

- **Failure before any output** → the whole step is **retried** with the same
  request (up to §4.3 limits); if it exhausts, the step fails and the error is
  surfaced in the session (`step.ts:192-197`).
- **Failure mid-stream after output started** (including
  `incomplete-stream` and transport *read* failures, `step.ts:281-285`) → the
  partial assistant message (text + any settled tool calls) is persisted, a
  synthetic user message *"The previous response was interrupted. Continue from
  where you left off without repeating completed content."* is appended
  (`llm.ts:35-37,278-286`), and the loop **continues with a fresh step**
  (`step.ts:253-261`). Tool calls that started before the death keep running;
  unsettled provider-hosted tool calls get a "Provider did not return a tool
  result" failure (`step.ts:220-222`).
- **Context overflow before output** → automatic compaction and re-issue, if
  `compaction.auto` is enabled (`step.ts:147-159`,
  `packages/core/src/session/compaction.ts:199-224`); when auto-compaction is off,
  the rejection is a fatal step failure (see the user-config note §7.4).
- **`finish_reason: "length"`** is a normal completion (not a failure): the step
  ends with the truncated text; the agent loop simply waits for the next input.
  This is the observed shape of the 2026-09-22 truncation incident: no error was
  raised — replies were just cut short until the backend's effective max output
  bound was raised.
- A stream that ends with `finish_reason: "unknown"` is escalated to an
  `incomplete-stream` error and handled as a dead stream
  (`step.ts:163-172`).

---

## 5. Model metadata

### 5.1 Source: config, then catalog — never the backend

- Per-model config: `limit {context, input, output}`, `capabilities
  {tools, input[], output[]}`, `modelID` (wire id), `name`, `compatibility`,
  `body`/`headers` overlays, `variants`, `cost`
  (`packages/schema/src/config/provider.ts:63-91`).
- Defaults when undeclared: capabilities `{tools: true, input: ["text","image"],
  output: ["text"]}` and limits `{context: 200_000, output: 32_000}`
  (`packages/schema/src/model.ts:85-160`).
- The backend is never queried (`/v1/models` absent, §1.6). A wrongly declared
  window is opencode's problem until a request is rejected.

### 5.2 What the declared limits drive

- `max_completion_tokens` fitting per request (§2.2,
  `model-request.ts:88-98`).
- Auto-compaction ceiling: compaction runs when the estimated context reaches
  `calculateCeiling(model.limit, buffer)` (window minus a reserve of at least
  16k tokens / 10%), with shrink steps 0.7/0.5/0.35 after an overflow rejection
  (`packages/core/src/session/compaction.ts:91-99,199-224,227-236`).
- Cost accounting per step (`step.ts:236-238`).

### 5.3 Capability gating (client-side)

- `capabilities.tools: false` would strip tool definitions (not used here).
- `capabilities.input` without `image` converts images to text placeholders
  before lowering (§2.7, `model-request.ts:122-159`).
- `capabilities.output` without `text` filters which models qualify as "small"
  models (`packages/core/src/model.ts:266-278`).

### 5.4 Variants

Declared per model (`Config.Model.variants`), selected in model references as
`provider/model#variant` (`packages/schema/src/model.ts:18-30`). Each variant can
overlay `settings`/`headers`/`body`; the user's variants map to
`reasoning_effort` on the wire (§2.6). When a provider's package is
`openai-compatible` and no variants are declared, opencode *generates*
`low`/`medium`/`high` effort variants automatically
(`packages/core/src/variant.ts:77-80`); the user's config overrides with its own
`low`/`medium`/`xhigh`.

---

## 6. Other provider types (context for the matrix)

### 6.1 Anthropic Messages (`@ai-sdk/anthropic` family)

- `POST {baseURL}/messages`, header `anthropic-version: 2023-06-01`; native
  Anthropic adds `?beta=true` (`packages/ai/src/protocols/anthropic-messages.ts:38,
  1648-1661`).
- Body requires `stream: true` and **`max_tokens` (mandatory number)**; optional
  `thinking {type, budget_tokens}` capped at half of `max_tokens`
  (`anthropic-messages.ts:379-380,1014-1021,1062-1068`,
  `packages/ai/src/protocols/shared.ts:114-119`).
- Stream events: `message_start`, `content_block_start`, `content_block_delta`,
  `content_block_stop`, `message_delta`, `message_stop`, `ping`
  (`anthropic-messages.ts:45-51,1517-1553`).
- `stop_reason` mapping: `end_turn`/`stop_sequence`/`pause_turn` → `stop`;
  `max_tokens`/`model_context_window_exceeded` → `length`; `tool_use` →
  `tool-calls`; `refusal` → `content-filter` (`anthropic-messages.ts:1090-1097`).
- Usage: `message_start.message.usage` + `message_delta.usage`
  (`input_tokens` is *non-cached*; cache read/write separate; reasoning inside
  `output_tokens`, `anthropic-messages.ts:1106-1130`).

### 6.2 OpenAI Responses (`@ai-sdk/openai`)

`POST {baseURL}/responses` (`packages/ai/src/protocols/openai-responses.ts:23,
314`; `packages/ai/src/protocols/openai-compatible-responses.ts:18`). Used by the
native OpenAI route and by some provider packages (GitHub Copilot, ZAI
responses-skin, xAI); **not** reachable via `@ai-sdk/openai-compatible`. Not part
of the user's provider surface.

---

## 7. Notes on the user's `ai.kido.ws` provider config

Observed in `~/.config/opencode/opencode.jsonc` (v2.0.22 format: `providers` /
`package` / `settings`):

1. **Wiring**: `package: "aisdk:@ai-sdk/openai-compatible"` +
   `settings.baseURL: "https://ai.kido.ws/v1"` ⇒ every completion is
   `POST https://ai.kido.ws/v1/chat/completions`, streaming, per §1-§3. Model key
   and `modelID` are both `Qwen3.8-27B`.
2. **Auth: none.** Config providers are force-enabled
   (`config/plugin/provider.ts:67`), and with no `apiKey` in config and no
   `ai.kido.ws` row in the credential DB, the resolver swaps the route to
   `Auth.none` (`model-resolver.ts:380-384`) ⇒ requests go out **without an
   `Authorization` header**. If the deployment later requires auth, a
   `settings.apiKey` (or `/connect` credential) must be added and the backend must
   accept `Authorization: Bearer <key>`.
3. **Both max fields on the wire**: computed `max_completion_tokens` (§2.2) plus
   the fixed `body.max_tokens: 32768` overlay on every request. The backend should
   accept both and honor the tighter bound; the `body` overlay exists precisely
   because of the 2026-09-22 truncation incident (serve needed a known output
   bound).
4. **`compaction: { auto: false, prune: false }`** ⇒ on a context-overflow 4xx the
   runner does **not** auto-compact (`compaction.ts:203-204`): the step fails with
   the provider error and only a manual `/compact` recovers. Combined with the
   declared window (`limit.context: 446902`) — which matches the deployed unit's
   actual window (verified by serve-capabilities, ticket #141; the 262144 figure
   in the map notes at the time of writing was superseded) — over-window prompts
   will surface as fatal errors rather than recovering silently (if a different
   arm serving only 262144 holds the GPU, the declaration over-declares it by
   184,758 tokens). Declared limits are opencode's only
   window knowledge (§5.1).
5. **Fields/headers the backend must tolerate**: `store: false`, `strict: false`
   per tool, `stream_options.include_usage`, `reasoning_effort` (variant
   selected; absent for the base model), `tool_choice: "none"` (max-steps step),
   all `x-opencode-*` / affinity headers, `reasoning_content` and
   `content: null` on replayed assistant messages, `<system-update>`-wrapped user
   text, image `data:` URLs, and `max_tokens` alongside
   `max_completion_tokens`.
6. **What the backend must produce**: SSE with `[DONE]`; exactly one choice; a
   `finish_reason` from the accepted set (§3.4) before any post-finish silence;
   no content after `finish_reason`; tool-call deltas carrying `id`+`name` per
   index; a final `usage` event (top-level or `choices[0].usage`) with
   `prompt_tokens`/`completion_tokens` and OpenAI-style
   `prompt_tokens_details`/`completion_tokens_details` breakdown; 4xx error bodies
   in `{"error":{"message","code"}}` shape with overflow phrasing when the prompt
   is too long (§4.1).
7. **Not needed**: `/v1/models`, `/v1/embeddings`, non-streaming completions,
   `/v1/responses`, `/v1/messages` (for this provider package).
8. **Live evidence** (serve request log
   `/home/kido/trash/temp/ninfer-yarn-log.jsonl`, `request_done` entries,
   2026-10-02): agent-step requests from opencode arrive as
   `protocol: "openai_chat_completions"`, `model: "Qwen3.8-27B"`, `stream: true`,
   `requested_output_tokens: 32768` with `requested_output_tokens_source:
   "client"`, and — when a variant is selected — `requested_reasoning_effort:
   "xhigh"`, confirming the §2.2/§2.6 wiring end-to-end. Sampling fields the log
   shows (`temperature 1.0`, `top_p 0.95`, `top_k 20`, `seed …`) are the serve's
   effective defaults, not client-supplied values: opencode does not send
   `temperature`/`seed` (§2.5), and `tool_choice` in the log is the serve default
   `"auto"` for the absent client field.
9. **Proxy in between**: `ai.kido.ws` is served by local caddy
   (`hosts/nerevar/caddy.nix`): `handle /v1/embeddings` → `127.0.0.1:8806`
   (crispembed, a separate embeddings backend for other clients — opencode never
   calls it, §1.7), everything else → `127.0.0.1:8827` (the ninfer serve), with
   `request_body 128MB` and unbuffered SSE (`flush_interval -1`, zero read/write
   timeouts). Request bodies above 128 MiB would be rejected by caddy before
   reaching the serve (relevant to the §2.7 image budget: opencode caps inline
   images at 15 MiB total, so opencode traffic stays far below the proxy limit).

---

## Verification notes

- All source citations read directly from the blobless clone of
  `github.com/sst/opencode` at tag `v2.0.22` (commit `527f0b931`), checked out
  under `/tmp/opencode/opencode-src` on 2026-10-02.
- Local facts verified this session: installed `opencode v2.0.22`
  (`opencode --version`); user config content; credential DB rows (3 entries, no
  `ai.kido.ws`); current session running with reasoning effort `xhigh`
  (system-prompt declaration matching the config's `xhigh` variant).
- No behavior is asserted from documentation alone where source was available;
  the docs are cited for the package→endpoint mapping and `limit` semantics
  only. Where the published docs (v1-style `provider`/`npm`/`options` schema)
  disagree with the installed binary's v2 config schema (`providers`/`package`/
  `settings`), the source of record wins; the v1→v2 migration path exists in-tree
  (`packages/core/src/v1/config/migrate.ts:268`).
