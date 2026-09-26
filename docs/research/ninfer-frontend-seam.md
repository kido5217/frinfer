# NInfer frontend seam — parsing contracts, template inputs, rendering-parity oracle

Research for ticket [#31](https://github.com/kido5217/ninfer-yarn/issues/31), part of the
wayfinder map [#25](https://github.com/kido5217/ninfer-yarn/issues/25) (port llama.cpp chat
parsing into NInfer). This document audits the receiving end of that port in the tree at
`master` = `c6a3b1e1` (fork tip, upstream tip `e31bc99b`), with `file:line` references. It is
descriptive; the design decisions belong to tickets #28 (integration) and #29 (qualification
contract).

Method: read-only source audit in the worktree `/tmp/opencode/wt-ninfer-frontend-seam`; one
decisive Python-only check against the prebuilt `ninfer_jinja_test` binary
(`/home/kido/network/projects/ninfer-yarn/build/tests/ninfer_jinja_test`, sha256
`064fda50a16afa72b196d4d7df80655c4201e8f740bd722ef0ffd48d60387d85`, built 2026-09-26 20:35 from
the same engine sources as `c6a3b1e1` — the last change to `src/text/`, `third_party/llama-jinja`,
`tests/text/`, `tools/chat_templates/` is upstream `8eaed538`, 2026-09-16). No GPU jobs, no full
build.

---

## 1. Output parsing: where it lives, what feeds it, what consumes it

All model-output parsing lives in the **model frontend**, not in `src/serve/`:
`src/models/qwen3_5/frontend/output_session.cpp` (presentation state machine) and
`tool_call_parser.cpp` (tool-call markup). Serve is a pure consumer of the parsed results.

### 1.1 The seam object: `OutputSession`

`OutputSession` (`src/models/qwen3_5/frontend/output_session.h:55-91`) is constructed once per
request by `Frontend::make_output_session` (`frontend.cpp:903-913`) with:

| input | source | contract |
|---|---|---|
| tokenizer | `Frontend::Impl` | `decoded_token(id)` → bytes + `special` flag (`tokenizer.h:141`) |
| stop policy | `merge_stop_policy(tokenizer, caller_stop)` | tokenizer defaults from `generation_config.json` (`tokenizer.cpp:834`) ∪ caller token ids; strings per channel (`frontend.cpp:355-384`) |
| output options | `OutputOptions{raw, preserve_special_tokens, tool_name_max_length}` (`include/ninfer/types.h:255-261`) | `preserve_special = raw \|\| preserve_special_tokens`; `raw` also disables tool-call decoding (`output_session.cpp:337-340`) |
| `starts_in_reasoning` | rendered prompt layout (`prompt_layout.cpp:109-110`) | initial channel |
| thinking control | `ThinkingControlOptions.budget` + frontend-owned control token span | the injected canonical suffix (frontend.cpp:41-44: `"\n\n Considering … now.\n</think>\n\n"`), tokenized once at load (`frontend.cpp:604-624`) |
| tool-call contract | `build_tool_call_output_contract(tool_jsons, enabled)` | declared names/parameter types, `enforce_declared_names` (`tool_call_parser.h:19-53`) |

The engine calls it through a **preview → commit** protocol (documented in
`docs/maintainer/engine-architecture.md:111-122` and `:428-462`): the frontend may preview the
semantic effect of a licensed token prefix, but only the Engine's commit publishes it.

Call sites in `src/runtime/engine/engine_core.h`:

| call site | line | meaning |
|---|---|---|
| `validate_generation_capacity(effective_output_tokens)` | 207 | admission check that the thinking-budget control suffix fits (`output_session.cpp:587-602`) |
| `preview_model(row_tokens, remaining, limit_reason)` | 1113 | once per decode/prefill **round**, over the raw generated token ids of that row |
| `preview_terminal(reason)` for cancel | 1104, 894, 984 | terminal flush between rounds |
| `commit_preview()` | 1226, 895, 989, 1906 | swap preview into committed state, produce ≤2 `OutputDelta`s |
| `preview_control(tokens, remaining)` | 1863 | accepts exactly the pending control span (`pending_control_tokens()`, `output_session.cpp:531-536`) |
| `take_tool_calls()`, `tool_call_parse_diagnostics()`, `reasoning_tokens()`, `thinking_stats()`, `matched_stop_string()` | 852-865 | terminal `GenerationResult` fields (`include/ninfer/types.h:793-800`) |

`preview_model` is fed **raw accepted token ids**, not text: it iterates
`tokens` and calls `tokenizer->decoded_token(id)` per token
(`output_session.cpp:451-459`). Parsing therefore runs on the detokenized byte stream, while the
committed truth stays the token ids in `request->generated`. `preview_control` is the same over
engine-injected control tokens.

### 1.2 What the presentation state machine does per token

Per decoded token (`output_session.cpp:409-515`):

1. `PrefixExecutionTracker` (lines 118-139, fed at 456-459) watches for the canonical
   `"\n</think>\n\n"` (`chat_template.h:19`) and reports `prefix_execution_split_after` so the
   engine can split execution inside a speculative round.
2. semantic thinking tracker (only when a budget is set, 461-469) counts model-origin thinking
   tokens and drives `ContinuationAction::ApplyTargetControl` at the cap (509-513).
3. stop-token check against merged policy (471-501): suppressed from output unless
   `publish_stop_token`; `FinishReason::StopToken`.
4. stop-string matching per channel with hold-back: `feed_channel` (203-237) keeps the longest
   suffix that is a prefix of any declared stop string and picks the earliest match by
   `(committed_tokens, byte_cut, declaration_order)` (184-192); `FinishReason::StopString`.
5. presentation decode `feed_token_bytes` (290-296): UTF-8 reassembly (`consume_generated_utf8`),
   `</think>` transition Reasoning→Content with marker hold (259-288), special-token suppression
   unless `preserve_special` (482-484), leading-whitespace strip after the transition (245-257).
6. limit: if the round consumes the remaining budget, terminalize with
   `limit_reason` (OutputLimit | ContextCapacity) (504-508).

Terminal flush (`preview_terminal`, 604-621; `terminalize`, 298-317) publishes pending windows
and replaces a dangling partial UTF-8 code point with U+FFFD.

### 1.3 Tool-call decoding

`ToolCallOutputDecoder` (`tool_call_parser.h:69-94`) is incremental: `feed(text)` returns only
bytes **provably outside a possible terminal Qwen tool-call suffix**; `finish()` is called when
the session is terminal (`output_session.cpp:638-654`) and returns the retained content, the
parsed `GeneratedToolCall`s, and `ToolCallParseDiagnostics`. A malformed region falls back to
ordinary content verbatim (the documented semantics in `docs/serving.md:163-177`; the reasons are
`ToolCallParseFallbackReason` in `include/ninfer/types.h:302-328`).

Both `parse_qwen_tool_call_output` (whole-text) and the incremental decoder are exercised by
`tests/test_tool_call_parser.cpp` (761 lines): legacy vs declared-type normalization, embedded
markup, CRLF framing, unions, empty/non-string parameters, malformed fallback, and byte-by-byte
feed equivalence.

### 1.4 Interfaces a ported parser must satisfy

The port must keep the following contract surface; everything below is observable today.

1. **Construction/ownership.** One session per request, built from tokenizer + merged
   `StopPolicy` + `OutputOptions` + `starts_in_reasoning` + thinking control + tool contract
   (`frontend.h:76-78`, `output_session.h:84-87`). No shared mutable state across requests; the
   engine runs up to `kMaximumConcurrency` lanes on one worker thread.
2. **Preview purity and commit.** `preview_model` must not mutate committed state; all committed
   state advances only through `commit_preview` (a copy/swap today: `preview_state`/`state`,
   `output_session.cpp:429-433, 623-631`). A preview may be discarded (abort path
   `engine_core.h:1146-1157`).
3. **Decision type.** `runtime::OutputDecision{accepted_tokens, finish_reason, continuation,
   prefix_execution_split_after}` with the engine's validation: `0 < accepted ≤ licensed count`,
   unfinished rounds must accept the whole license, and the split must fall inside the prefix
   (`engine_core.h:1113-1121`).
4. **Streaming deltas.** A commit produces at most **two** deltas, at most one per channel
   (`PublishedOutput::push_back` throws above 2, `output_session.cpp:388-393`); deltas are
   append-only per channel; all Reasoning precedes all Content
   (`openai_chat_response.cpp:328-344`, `anthropic_messages_response.cpp:246-305`); consumers
   verify terminal output starts with the streamed text (`require_prefix`,
   `openai_chat_response.cpp:210-215`).
5. **Tool calls are terminal.** Wire streams never carry partial tool-call syntax; the adapters
   materialize calls from the terminal outcome (`openai_chat_response.cpp:371-376`,
   `anthropic_messages_response.cpp:332-349`). The streaming channel carries the *content* around
   the markup (with the suffix held back until terminal).
6. **Stop/budget decisions.** Token-id stops (tokenizer defaults + caller, deduplicated),
   per-channel stop strings with `include_in_output`, `publish_stop_token`, `FinishReason`
   mapping, `matched_stop_string()` for the Anthropic `stop_sequence` presentation
   (`anthropic_messages_response.cpp:60-80`).
7. **Thinking.** Starts in Reasoning IFF the rendered prompt ends in an open `<think>` block
   (`prompt_layout.cpp:87-110`); budget accounting counts only model-origin thinking tokens;
   injected control tokens do not consume the thinking budget but do consume output tokens
   (`docs/serving.md:200-216`); at the cap the session asks for the exact control span
   (`pending_control_tokens` / `model_token_budget_remaining`, `engine_core.h:1854-1906`).
8. **Tokenizer facts are the decode authority.** `decoded_token` byte views + `special` flag;
   `preserve_special` gating; UTF-8 streaming; no re-encode of generated text for parsing.
9. **Observation fields.** `reasoning_tokens`, `thinking_stats`, `tool_call_parse` diagnostics
   feed usage/details and logs (`generation_service.cpp:417-454`, `types.h:330-339`).
10. **Performance envelope.** `preview_model` sits on the decode path and sees every accepted
    token; today it copies a small fixed decoder state per round. A ported parser that re-scans
    the whole prefix per round changes decode host cost (design point for #28; measure per
    `docs/maintainer/op-development.md` when it becomes a performance claim).

### 1.5 Streaming granularity (observed)

| event | granularity | source |
|---|---|---|
| reasoning delta | per commit round, Reasoning channel | `engine_core.h:703-718` |
| content delta | per commit round, Content channel | same |
| channel transition | inside one round possible (≤1 delta each) | `output_session.cpp:270-278` |
| tool-call markup | withheld until terminal; parsed at `finish()` | `tool_call_parser.h:69-84`, `output_session.cpp:633-654` |
| stop-string match | first match by (round, byte cut, declaration order) | `output_session.cpp:184-192` |
| control suffix | one exact span, injected between rounds | `output_session.cpp:538-585` |
| prompt progress / timings | engine-level, not parser-level | `generation_service.h:61-68` |

---

## 2. How templates reach rendering

### 2.1 Artifact resource and `--chat-template`

- The `.ninfer` artifact stores the template as text resource `chat_template.jinja`
  (`src/models/qwen3_5/load/resources.cpp:12-15`, component `"text"`); conversion writes it via
  `--resource chat_template.jinja=…` (`docs/weight-conversion.md:340-345`).
- Startup override: CLI `--chat-template FILE` → `EngineOptions.chat_template_path`
  (`apps/cli/options.cpp:133-134`, `include/ninfer/types.h:153`, serve parity
  `src/serve/serve_options.cpp:290-291`).
- `compile_chat_template` (`frontend.cpp:197-231`): validates `tokenizer_config.json`
  (`add_bos_token=false`, `add_prefix_space=false`, pad `<|endoftext|>`, lines 182-195), collects
  `bos/eos/pad/unk/sep/cls/mask_token` + `additional_special_tokens` into the template context
  (200-212), then loads the file override: nonempty, ≤16 MiB, read as bytes (215-229). Compile
  errors are attributed to `artifact:chat_template.jinja` or the file path.
- The template is compiled **once per model load** and shared by all requests
  (`frontend.cpp:570`); rendering is per request.

### 2.2 The context the template sees

`CompiledChatTemplate::render` (`chat_template.cpp:153-461`) builds, in order:

- typed options merged with `chat_template_kwargs` (`template_parameters`, 50-99): reserved keys
  (`messages`, `tools`, `add_generation_prompt`, `continue_final_message`, tokenizer special
  tokens) cannot be overridden; `enable_thinking`/`preserve_thinking`/`reasoning_effort`/
  `add_vision_id` merge with conflict rejection; `reasoning_effort` values are the seven protocol
  names; `"none"` forces `enable_thinking=false` (83-96); the leftover kwargs pass through as
  template variables.
- `messages`: `role`, `content` (string, or array of `{"type":"text","text":…}` /
  `{"type":"image"}` / `{"type":"video"}`), `reasoning_content`, `tool_calls` (OpenAI shape with
  `function.arguments` as a JSON object), `tool_call_id` (189-236).
- `tools`: OpenAI function definitions (`{"type":"function","function":{name,parameters,strict:false}}`,
  rendered in `translate.cpp:98-107`).
- `continue_final_message`, `add_generation_prompt`, and the special-token variables.
- `add_generation_prompt=false` is used for prefix probes: message boundaries, cache boundaries
  and the rewrite checkpoint are derived by re-rendering prefixes and comparing text + literal
  spans (`same_prefix`, `chat_template.cpp:110-123`; probes 284-304, 386-412; boundaries
  413-457).

After rendering, `prompt_layout.cpp` parses the ChatML structure (`<|im_start|>`/`<|im_end|>`,
lines 9-10, 58-110) to find message extents and `starts_in_reasoning`; the tokenizer encodes with
literal spans so user text cannot be reinterpreted as control tokens
(`encode_rendered_chat`, `frontend.cpp:796-801`; `docs/serving.md:192-193`).

**Seam relevance:** the ChatML layout parse and `<think>` start detection are a *rendering-side*
contract that the ported parsing stack must keep producing — the froggeric template renders the
same markup (see §3).

### 2.3 Protocol → kwargs mapping (serve)

`resolve_prompt_semantics` (`translate.cpp:111-195`) merges top-level typed fields with
`chat_template_kwargs` (conflicting values → `conflicting_template_option`), maps
`reasoning_effort` to `ninfer::ReasoningEffort`, rejects assistant prefill + enabled thinking,
and strips the four standard keys from the kwargs before they reach the template.
`to_request_options` (311-343) builds `StopPolicy` (stop strings duplicated per channel when
`stop_strings_apply_to_reasoning`), the thinking budget (skipped when thinking is disabled), and
`preserve_special_tokens = uses_tools() || has_tool_history()`. `to_prompt_input` (197-309) maps
turns, media, and cache markers. Wire forms are documented in `docs/serving.md:122-125,
195-224`.

---

## 3. Rendering-parity oracle feasibility

### 3.1 The oracle as it exists

- `tests/text/test_chat_templates.py` renders the in-tree templates
  (`tools/chat_templates/qwen3_6.jinja`, `qwen3_8.jinja`) with Python
  `jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
  extensions=["jinja2.ext.loopcontrols"])`, a `tojson` shim (`json.dumps(..., ensure_ascii=False)`)
  and `raise_exception` → `ValueError` (lines 25-37). This mirrors the HF/transformers template
  environment, which builds
  `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[…, jinja2.ext.loopcontrols])`
  with `tojson`/`raise_exception` globals (source:
  [transformers `chat_template_utils.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/utils/chat_template_utils.py),
  read 2026-09-27; NInfer's oracle additionally omits `strftime_now`, which the vendored engine
  supports natively, `tests/text/test_jinja.cpp:131-134`).
- The C++ side is `build/tests/ninfer_jinja_test --render`, which reads JSONL
  `{"source": …, "context": …}` and answers `{"ok": true, "text": …}` or `{"ok": false,
  "error": …}` (`tests/text/test_jinja.cpp:223-240`). `test_cpp_matches_independent_renderer`
  feeds both renderers the same contexts and compares **text** (lines 192-279). The test target
  and its Python driver are registered at `tests/cmake/CoreTests.cmake:63-69`.
- Devshell Python 3.13.15 with jinja2 3.1.6.

**The oracle accepts arbitrary template sources through the `--render` protocol** — adding a new
template to the parity comparison requires only a fixture plus a test entry, no C++ change.

### 3.2 Froggeric validation (executed 2026-09-27)

Template: `https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja`
(version string `qwen3.8-froggeric-v22.5`), 28 234 bytes, sha256
`e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`, saved to
`/tmp/opencode/froggeric_chat_template.jinja`.

I rendered 22 contexts with the Python oracle and with the vendored C++ engine (same binary as
the repo's own parity test) and compared byte-for-byte:

- **22/22 agree**, including both `raise_exception` paths (image in system message,
  empty `messages`). Contexts covered: defaults, `enable_thinking=false`, all
  `reasoning_effort` spellings incl. `" none "` and `ultracode`, `preserve_thinking` /
  `preserve_reasoning`, in-content `<|think_*|>` markers, tools (xml and `tool_call_format:"json"`),
  string and mapping arguments, tool-only history, multi-step tool detection, consecutive tool
  errors, `max_tool_arg_chars` / `max_tool_response_chars` truncation, vision items and
  `add_vision_id`, assistant continuation, `<think>` content repair, `auto_disable_thinking_with_tools`.

Reproduction (probe script at `/tmp/opencode/froggeric_probe.py`, outside the repo):

```bash
curl -fsSL -o /tmp/opencode/froggeric_chat_template.jinja \
  https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja
nix develop -c python3.13 -B /tmp/opencode/froggeric_probe.py \
  build/tests/ninfer_jinja_test        # from the main checkout, binary path is the oracle arg
```

The probe uses the Python oracle environment from §3.1 and the `--render` JSONL protocol, so it
is directly promotable into `tests/text/test_chat_templates.py`.

### 3.3 Coverage limits

- The oracle compares **text only**. Literal-span (`literal_content`), region/origin mapping,
  cancellation checkpoints and concurrency are covered separately in
  `tests/text/test_jinja.cpp:86-220`; a froggeric fixture in the text oracle would not extend
  that coverage by itself.
- The vendored engine explicitly refuses a small set of Jinja features
  (`third_party/llama-jinja/jinja/value.cpp:433, 438, 443, 514, 775, 862, 965, 984, 1002`:
  `sameas`/`escaped`/`filter` tests, `join` on strings and objects, `unique`, filter-mapping
  `map`, `min`/`max` with attributes). The froggeric template uses none of them (checked by
  grep), and `raise_exception` is supported (`value.cpp:252`).
- The vendored source baseline is llama.cpp `7609846557c50f9d984719a9e1e8c5f3d02f807b`
  (`third_party/llama-jinja/README.ninfer.md:1-7`).

**Verdict:** the rendering side already meets the map's "vendored jinja stays" decision and can
carry the froggeric template as a parity fixture today; the remaining rendering risk is the
ChatML layout/`<think>` coupling in `prompt_layout.cpp`, which the froggeric output satisfies.

---

## 4. Upstream conflict surface since `98dada0e`

Baseline `98dada0e` = upstream `feat(frontend): execute custom jinja chat templates`
(2026-09-15), the commit that introduced the artifact/`--chat-template` Jinja rendering seam
(the vendored jinja source base arrived the same day in `b9219f3f`, and
`tools/chat_templates/qwen3_8.jinja` came with `98dada0e` itself).

Upstream `master` is still `e31bc99b` (2026-09-26), already merged into the fork
(`230ccd19`); the fork is 10 commits ahead, 0 behind (`git rev-list --left-right --count
master...upstream/master`). In `98dada0e..upstream/master` (36 commits), **exactly one** commit
touches the frontend/template surface:

- `8eaed538` — 2026-09-16, `fix(frontend): preserve literal content in chat templates`
  (upstream #258), touching `chat_template.{cpp,h}`, `processor.cpp`, `prompt_layout.{cpp,h}`,
  `tokenizer.{cpp,h}`, `src/text/{byte_span.h,jinja.cpp,jinja.h}`,
  `third_party/llama-jinja/{string,unicode,value}`, `tests/text/test_jinja.cpp`,
  `tests/models/qwen3_5/test_frontend.cpp`, `docs/serving.md`. Already in the fork's master.

Everything else upstream since then is ops/perf/docs. The files the port will most likely
replace or touch (`output_session.{cpp,h}`, `tool_call_parser.{cpp,h}`, `frontend.{cpp,h}`,
`chat_template.{cpp,h}`, `prompt_layout.{cpp,h}`, `src/text/jinja.*`,
`third_party/llama-jinja/**`, `tests/text/*`, `tests/test_tool_call_parser.cpp`,
`tests/models/qwen3_5/test_frontend.cpp`, `tools/chat_templates/*`) are therefore exposed to
already-merged upstream work only; the next sync conflict can only come from new upstream
frontend commits. The sync runbook requires recomputing the overlap set each sync
(`docs/maintainer/upstream-sync.md:45-53`) and treats upstream as append-only
(ground rules, lines 7-19); a drift note surfaces when upstream moves past `e31bc99b` (72-76).

---

## 5. Open points for the design tickets

1. **Parser shape vs preview purity (#28).** The seam requires side-effect-free previews and a
   state that can be discarded (a preview is dropped on abort). How the llama.cpp parser's
   incremental state can be snapshotted, cloned or replayed is #26's ground to establish; the
   integration decides how to fit it into the copy-preview model.
2. **Delta budget (#28).** A round publishes ≤1 delta per channel. If the ported parse expresses
   transitions that do not fit that budget (e.g. a tool-call update inside one round), the design
   must map them onto — or revise — `PublishedOutput`; the ported parser's diff granularity is
   #26's finding.
3. **Tool-call fallback semantics (#28/#29).** The current decoder restores a malformed tool
   region verbatim as content. The PEG parser will have its own failure semantics; the regression
   corpus (PR [#309](https://github.com/Neroued/ninfer/pull/309) fixtures per the map, plus
   `tests/test_tool_call_parser.cpp`) fixes what must stay observable.
4. **Oracle ownership (#27/#29).** This document establishes the rendering oracle and lists the
   existing parsing corpora; the independent parsing oracle (vLLM `Qwen3Parser` semantics) is
   ticket #27's deliverable and its results must be wired into the #29 contract.
5. **Perf measurement point (#28).** Any parser that rescans the prefix per round changes decode
   host cost; the acceptance measurement belongs to the #29 qualification contract, not to this
   audit.
