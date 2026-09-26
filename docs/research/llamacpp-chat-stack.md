# llama.cpp chat-parsing stack — anatomy and port surface

Research for wayfinder ticket [#26](https://github.com/kido5217/ninfer-yarn/issues/26) (map [#25](https://github.com/kido5217/ninfer-yarn/issues/25)).

| | |
|---|---|
| **Baseline recorded** | `ggml-org/llama.cpp` commit `95887577ab5fead779581a7030a83c7752ff3234` (`refs/heads/master`, committed 2026-09-26 22:05:52 +0200, subject `cuda: support Nemotron 3 Puzzle state size 96 for ssm scan (#28717)`) |
| **Retrieved** | 2026-09-27, `git ls-remote` + `git clone --depth 1` at that SHA (shallow clone, verified `git rev-parse HEAD`) |
| **License** | MIT, `LICENSE` in-tree: "Copyright (c) 2023-2026 The ggml authors" (llama.cpp). `common/jinja` is llama.cpp's own engine, introduced by upstream PR [#18462](https://github.com/ggml-org/llama.cpp/pull/18462), "originally inspired by huggingface.js's jinja package" (`common/jinja/README.md:1-5`) |
| **Method** | primary source only: the tree at the pinned SHA (headers + implementations read in full or by function), upstream `docs/autoparser.md`, upstream tests, plus one live experiment (section 9) built from the same SHA on this host |
| **Location note** | this is the first file under `docs/research/` in this fork; the directory was created for this ticket |

All `common/...:line` citations refer to the pinned commit. Where a statement is an inference (not directly read), it is marked **[inference]**; where it was verified by running the built tools, it is marked **[verified]**.

---

## 0. Answers in brief

1. **What the stack is.** NInfer-relevant parsing in llama.cpp is: a template engine (`common/jinja`, a maintained in-tree fork), a render+analyze *apply* path (`common/chat.{h,cpp}`, `common/jinja/caps.*`), a differential auto-parser that derives a parser from any template (`common/chat-auto-parser*`, `common/chat-diff-analyzer.cpp`), a string-based PEG engine with an AST→message mapper (`common/peg-parser.*`, `common/chat-peg-parser.*`), 15 hand-written specialized parsers for formats the auto-parser deliberately does not cover (`common/parsers/*`), and JSON-schema/GBNF plumbing for constrained decoding (`common/json-schema*`, `common/json-schema-to-grammar.*`). `common/reasoning-budget.*` is a *sampler* (token budget for thinking), not part of the parse path.

2. **The parser is data, not code.** `common_chat_templates_apply()` returns `common_chat_params`, including the generated parser serialized as a JSON string (`common_chat_params::parser`, `common/chat.h:281`); `common_chat_parse()` consumes the deserialized `common_peg_arena` plus a generation prompt and produces `common_chat_msg` (`common/chat.cpp:1441-1452`). Apply and parse are two separate phases; the serialized arena is designed to be passed around (llama-server sends it as an opaque `chat_parser` field, `tools/server/server-common.cpp:1382-1384`).

3. **Froggeric is NOT handled by the auto-parser.** The froggeric template (Qwen3.8-froggeric-v22.5, fetched 2026-09-27, sha256 `e57684ba…c4b2`) matches the *Qwen3-Coder specialized parser* detector: its source contains `<tool_call>`, `<function=` and `<parameter=` and not the excluded Qwen-template concatenation form (`common/chat.cpp:1212-1220`); the comment there says this handler is shared by "Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash". **[verified]** — `test-chat-auto-parser <froggeric>` prints `Using specialized template: Qwen3-Coder`. The in-tree `models/templates/Qwen3.5-4B.jinja` routes identically. **[verified]**

4. **But the auto-parser can cover it.** When the detection markers are obfuscated in the source (rendered output byte-identical), the differential analyzer derives a correct parser: `reasoning_mode=TAG_BASED <think>…</think>`, `tool_mode=TAG_WITH_TAGGED`, per-call `<tool_call>\n…</tool_call>`, `<function=NAME>\n…</function>\n`, `<parameter=NAME>\nVALUE\n</parameter>\n`, lazy grammar trigger `<tool_call>`. **[verified]** Full output in Appendix B.

5. **Minimal port for the product (Qwen3.5/froggeric-shaped).** The parse side (`peg-parser`, `chat-peg-parser`, the mapper, `chat.h` message/params types) plus the Qwen3-Coder specialized parser (`common/parsers/qwen3-coder.cpp`, 194 lines) and the apply path (render/normalization) is enough to parse what the shipped templates produce. The differential analyzer + `jinja/caps` probing are only needed if NInfer must derive parsers for arbitrary user templates at runtime. `json-schema*` + GBNF is needed only if NInfer wants grammar-constrained tool output. `reasoning-budget` is out of parse scope.

6. **Cost of coupling.** The parse engine itself is tokenizer-free and string-based; llama.cpp-specific dependencies are `common_json` (thin wrapper over vendored nlohmann), `trie.*`/`unicode.*` (Aho-Corasick, UTF-8), `log.h`, and for the *apply* side `llama_vocab`/`llama_tokens` (bos/eos pieces, message-delimiter tokenization) and `common.h` (reasoning-format enum, grammar triggers). Total surface to vendor: roughly 9k lines of core (engine + chat layer + schema/GBNF + small shared deps) + ~2.5k lines of specialized parsers + the jinja engine (already vendored in this fork, but *without* the `caps` layer).

7. **Sync cost is non-trivial.** 12-month churn at baseline: `chat.cpp` ≥100 commits (API cap), `chat.h` 33, `chat-auto-parser-generator.cpp` 32, `chat-peg-parser.cpp` 25, `jinja/runtime.cpp` 25, `chat-diff-analyzer.cpp` 21, `peg-parser.cpp` 19. Repo-wide velocity: 147 commits in the 7 days before 2026-09-27 (~21/day). The map's decision (recorded baseline + drift alert + re-vendor runbook) is justified by these numbers.

---

## 1. Architecture: how a template becomes a parser, then a message

```
GGUF metadata tokenizer.chat_template[.tool_use]
  │  common_chat_templates_init()                 common/chat.cpp:757-855
  ▼
common_chat_template { jinja::program, src, bos/eos, caps }   common/chat.h:51-78
  │  jinja::caps_get(prog) — behavioral probing                common/jinja/caps.cpp:111
  ▼
common_chat_templates_apply(inputs)               common/chat.cpp:1434-1439
  ├─ use_jinja=false → legacy llama_chat_apply_template      common/chat.cpp:1369-1432
  └─ use_jinja=true  → common_chat_templates_apply_jinja     common/chat.cpp:1225-1366
        │ messages → OAI-compat JSON (caps-normalized)       common/chat.cpp:538-548
        │ render + post-process workarounds                  common/chat.cpp:1273-1301
        ├─ force_pure_content? → one-shot content parser     common/chat.cpp:1315-1329
        ├─ specialized detector match? → hand-written parser common/chat.cpp:1090-1223
        │      (15 handlers, common/parsers/*)
        └─ else → differential autoparser                    common/chat.cpp:1335-1365
               analyze_template()                            chat-diff-analyzer.cpp:257
               peg_generator::generate_parser()              chat-auto-parser-generator.cpp:23
  ▼
common_chat_params { prompt, generation_prompt, format=PEG_*,
                     grammar(+lazy,triggers), preserved_tokens,
                     additional_stops, parser(serialized PEG JSON),
                     message_delimiters, thinking tags }     common/chat.h:269-283
  ▼
common_chat_parse(input, is_partial, parser_params)          common/chat.cpp:1441-1445
  = common_peg_parse: effective_input = generation_prompt+input,
    LENIENT parse, AST → common_chat_msg via mapper            common/chat.cpp:1447-1523
  ▼
streaming: re-parse the full text each chunk, diff messages   tools/server/server-task.cpp:162-209
```

Two deliberate invariants:

- **Specialization first, heuristics second.** A template is never auto-analyzed if a hand-written handler matches (`common/chat.cpp:1331-1333`); the auto-parser is a *fallback*, and its failure is a hard error (`common/chat.cpp:1363-1365`: `"Unable to generate parser for this template..."`). The in-tree design doc `docs/autoparser.md` says the same ("it will never be used if a specialized parser matched").
- **Preserved tokens + triggers are part of the contract.** Apply returns the marker strings the generation loop must not split (`preserved_tokens`), literal stop strings (`additional_stops`), and lazy-grammar triggers (`grammar_triggers`); the server forwards them into the sampling/grammar layer (`tools/server/server-common.cpp:1364-1384`).

---

## 2. File-by-file inventory

LOC measured with `wc -l` at the baseline.

### 2.1 PEG engine (tokenizer-free, string-based)

| File | LOC | What it does |
|---|---|---|
| `common/peg-parser.h` | 557 | Public engine types: 21 parser variants (`common_peg_*_parser`), `common_peg_ast_node`/`arena`, 3-state parse result (`FAIL`/`SUCCESS`/`NEED_MORE_INPUT`), `common_peg_parse_context` + flags (`LENIENT`, `DEBUG`), `common_peg_arena` (parse/`resolve_refs`/`build_grammar`/`to_json`/`from_json`/`save`/`load`/`dump`), `common_peg_parser_builder` (eps/start/end/literal/sequence/choice/repeat/peek/negate/any/space/chars/string/until/ref/atomic/tag/schema/rule/trigger_rule/gbnf/ac, JSON & Python value grammars, `build()`), `build_peg_parser()` helper |
| `common/peg-parser.cpp` | 2124 | Engine: char-class parsing, UTF-8 escape handling, AST helpers (`find_by_tag`/`find_by_rule`/`sanitized_text`), `resolve_refs`, recursive-descent `parse` with depth tracking and `NEED_MORE_INPUT` propagation, `dump` (debug printer), GBNF generation (`build_grammar`, incl. Aho-Corasick grammars, schema/GBNF nodes, lazy trigger rules), JSON (de)serialization (`to_json`/`from_json`/`save`/`load`) |
| `common/trie.h` / `.cpp` | 73 / 123 | Aho-Corasick automaton used by `ac()` parsers, GBNF generation and `reasoning-budget` |
| `common/unicode.h` / `.cpp` | 30 / 129 | UTF-8 validation/decoding used for invalid-UTF-8 recording and `sanitized_text` |

The engine supports streaming by design: every alternative can return `NEED_MORE_INPUT`, and the AST keeps partial nodes (`is_partial`) whose text can be mapped as it arrives.

### 2.2 Chat layer

| File | LOC | What it does |
|---|---|---|
| `common/chat.h` | 391 | All public chat types: `common_chat_tool_call`, `common_chat_msg` (+ `content_parts`), `common_chat_msg_diff` (streaming diffs), `common_chat_template` (jinja program + src + bos/eos + caps), `common_chat_msg_delimiter(s)` (role → delimiter + tokenized form, `split()` over token streams), `common_chat_tool`, tool-choice enum, format enum (`CONTENT_ONLY`, `PEG_SIMPLE`, `PEG_NATIVE`, `PEG_GEMMA4`, `PEG_MINIMAX_M3`), continuation modes, `common_chat_templates_inputs`, `common_chat_params`, `common_chat_parser_params`, and every entry point (§7) |
| `common/chat.cpp` | 1533 | Template init from GGUF/override, OAI-compat message/tool parsing and serialization, `render_message_to_json` + caps-driven normalizer, `common_chat_templates_apply` + jinja/legacy routes, `common_chat_get_asr_prompt`, `common_chat_parse`/`common_chat_peg_parse`, `common_chat_templates_support_enable_thinking`, `common_chat_tool_choice_parse_oaicompat`, template delimiter parse/serialize |
| `common/chat-peg-parser.h` | 220 | AST→message mapper (`common_chat_peg_mapper` + gemma4/minimax-m3 subclasses), `common_chat_peg_builder` (tag constants + typed builders: reasoning/content/tool/tool-open/close/id/name/args/arg...), `permute()` (all-orders matching), `standard_json_tools()`, `standard_constructed_tools()`, `python_style_tool_calls()`, `build_chat_peg_parser()`; the legacy `tagged_peg_parser`/`tag_based_peg_mapper` pair |
| `common/chat-peg-parser.cpp` | 1232 | Mapper implementation (JSON-string argument assembly, pythonic→JSON normalization, partial tool-call flushing, whitespace-only reasoning discard), tool-call builders for the three JSON layouts (name-as-key / nested keys / flat keys), python-style calls, gemma4 + minimax-m3 mappers, `tag_with_safe_content`, Aho-Corasick arg-string helper |
| `common/json-schema.h` / `.cpp` | 198 / 514 | Typed JSON-Schema model for the subset GBNF supports (`common_chat_schema` hierarchy: any/ref/anyOf/allOf/const/enum/null/bool/number/integer/string/array/tuple/object; `common_chat_schema_property{name,schema,required}`; `common_chat_schema_document` with `$ref` table; `common_chat_schema_from_json`) |
| `common/json-schema-to-grammar.h` / `.cpp` | 23 / 1028 | `json_schema_to_grammar()` (legacy string API), `common_grammar_builder` (the callback interface `{add_rule, add_schema}`), `build_grammar()`, `gbnf_format_literal()` |
| `common/json.h` / `.cpp` | 356 / 433 | `common_json`, a thin ordered-JSON wrapper over vendored nlohmann (`vendor/nlohmann`); the whole chat stack uses this type, never nlohmann directly |
| `common/reasoning-budget.h` / `.cpp` | 52 / 310 | A `llama_sampler` state machine (IDLE→COUNTING→WAITING_UTF8→FORCING→DONE) that caps tokens inside a reasoning block and forces an end sequence; token-level, built on `llama_vocab` + `trie`/AC. **Not part of parsing**; used by server/CLI thinking-budget features |

### 2.3 Template→parser analysis (the auto-parser)

| File | LOC | What it does |
|---|---|---|
| `common/chat-auto-parser.h` | 453 | Analysis data model and generator API: `template_params`, `diff_split`, `compare_variants_result`, `generation_params` (messages/tools/tool_choice/json_schema/parallel/reasoning_format/stream/grammar/continuation/enable_thinking/extra_context/add_bos/eos/mark_input...), enums (`reasoning_mode` NONE/TAG_BASED/TOOLS_ONLY; `content_mode` PLAIN/ALWAYS_WRAPPED/WRAPPED_WITH_REASONING; `tool_format` NONE/JSON_NATIVE/TAG_WITH_JSON/TAG_WITH_TAGGED; `call_id_position` NONE/PRE_FUNC_NAME/BETWEEN_FUNC_AND_ARGS/POST_ARGS), marker structs (`tool_format_analysis`, `tool_function_analysis`, `tool_arguments_analysis`, `tool_id_analysis`), `autoparser` (caps, user/assistant start, reasoning/content/tools analyzers, `preserved_tokens`, `additional_stops`, `analyze_template`, `build_parser`) and `peg_generator` |
| `common/chat-auto-parser-generator.cpp` | 478 | `peg_generator::generate_parser`: render prompt/generation-prompt, continuation prompt surgery, `build_parser` (reasoning parser, response-format grammar, tools per `tool_format`, else content), `parser.save()`, grammar + lazy triggers when tools/json_schema present |
| `common/chat-auto-parser-helpers.h` / `.cpp` | 74 / 363 | Marker-diff primitives: `calculate_diff_split` (longest common prefix/suffix *snapped to marker boundaries*), `until_common_prefix`, `after_common_suffix`, `segmentize_markers`, `prune_whitespace_segments`, `apply_template`, `compare_variants` (render A and B, diff) |
| `common/chat-diff-analyzer.cpp` | 1635 | The actual analysis: `autoparser::analyze_template`, reasoning analysis (presence / thinking-enabled / scope), content analysis, tool analysis (format classification, per-call + function + argument + call-id markers, parallel/array wrapping), preserved-token collection, start-marker detection, and an ordered list of **12 template-specific workaround patches** (`common/chat-diff-analyzer.cpp:36-228`, each logging `[Patch: ...]`) |

### 2.4 Specialized parsers (`common/parsers/*`, 2492 lines total)

`parsers.h` declares `foreach_function`/`foreach_parameter` helpers (implemented in `parsers.cpp`, 25 lines) and one `common_chat_params_init_*` per format:

| File | LOC | Format |
|---|---|---|
| `qwen3-coder.cpp` | 194 | `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` (Qwen3-Coder, **Qwen3.5**, Nemotron Nano 3, StepFun-3.5-Flash) |
| `deepseek.cpp` | 273 | DeepSeek V3.2/V4 DSML markers |
| `gemma4.cpp` | 307 | Gemma 4 `call:` format (own mapper) |
| `minimax-m3.cpp` | 229 | MiniMax-M3 namespace markers (own mapper) |
| `ling3.cpp` | 194 | Ling 3.0/Bailing V3 `<arg_key>/<arg_value>` |
| `kimi-k2.cpp` / `kimi-k3.cpp` | 128 / 167 | Kimi K2 thinking / K3 markers |
| `gpt-oss.cpp` | 158 | `<|channel|>` analysis/final channels |
| `muse-glimmer.cpp` | 138 | `to=<recipient>` + `<|eom|>/<|eot|>` |
| `cohere2moe.cpp` | 141 | `<|START_TEXT|>` / `<|START_ACTION|>` |
| `minicpm5.cpp` | 130 | `<function name="..."><param name="...">` |
| `ministral3.cpp` | 126 | `[TOOL_CALLS]`/`[ARGS]` (no `[CALL_ID]`) |
| `lfm2.cpp` | 110 | LFM2/LFM2.5 python-style calls |
| `functionary-v3-2.cpp` | 96 | `>>>all` / `>>>{recipient}` |
| `gigachat-v3.cpp` | 76 | `<|role_sep|>`/`<|message_sep|>` |

Detection is ordered source-substring matching in `common_chat_try_specialized_template` (`common/chat.cpp:1090-1223`). Every handler calls the same helpers (`common_chat_template_direct_apply_impl`, `common_chat_template_generation_prompt_impl`) and builds a PEG parser with `common_chat_peg_builder`; some reuse `standard_constructed_tools`/`standard_json_tools`.

### 2.5 Jinja + caps

| File | LOC | What it does |
|---|---|---|
| `common/jinja/*` | 6337 total | Full Jinja engine: lexer, parser, runtime, value model, `jinja::string` with input-marking (security against special-token injection from user text), Unicode case data, caps. NInfer already vendors a maintained fork of this engine (see §11.3) |
| `common/jinja/caps.h` / `.cpp` | 40 / 573 | `jinja::caps` (9 booleans) computed by *executing* the template with synthesized contexts and observing failures/`value::stats` operations (`caps_try_execute`, `caps.cpp:35-72`); `caps_get` (§5) |

---

## 3. The apply path (`common_chat_templates_apply`)

Ordered stages in `common_chat_templates_apply_jinja` (`common/chat.cpp:1225-1366`):

1. **Template selection**: tools present and GGUF provides `tokenizer.chat_template.tool_use` → use that variant, else the default (`:1229-1230`; init at `:757-855`).
2. **Message rendering**: `common_chat_msg` → OAI-compat JSON via `render_message_to_json` (`:538-548`), which runs `messages_inp_normalizer(caps)` to reshape content for the template's content capabilities.
3. **Caps-driven workarounds** (`:1273-1291`): developer→system role mapping (all but GPT-OSS); system-role removal; `null` content → `""` for tool-calling templates; object arguments instead of JSON strings when the template wants them.
4. **Extra context**: `datetime`/`date_string` plus user `chat_template_kwargs` JSON-parsed and merged (`:1293-1296`); `json_schema` parsed (`:1298-1300`); grammar+tools conflict rejected (`:1305-1307`).
5. **Continue-final-message** handling (`:1252-1271`): the last message becomes `continue_msg` and is *not* rendered; `AUTO` resolves to content or reasoning.
6. **`force_pure_content`** (`:1315-1329`): builds a content-only parser (`literal(generation_prompt) << content(rest())`) — used by e.g. perplexity-style flows.
7. **Specialized dispatch** (`:1331-1333`).
8. **Auto-parser fallback** (`:1335-1365`): analyze + generate; on exception throw `std::invalid_argument("Unable to generate parser for this template...")`. Delimiters and thinking tags are lifted from the analysis into `common_chat_params`.

Note that all four capabilities of the output (`prompt`, `grammar`, `parser`, `message_delimiters`) are produced *here*; the parse call later is stateless w.r.t. the template (`common_chat_parser_params` carries only `format`, `generation_prompt`, the arena, and parsing options; the converting constructor at `common/chat.h:299-302` copies just the first two).

---

## 4. The differential auto-parser in detail

### 4.1 What it renders and diffs

`analyze_template` (`common/chat-diff-analyzer.cpp:257-311`) runs three analyzers and two marker detectors, then applies workaround patches. Each analyzer uses `compare_variants` (`common/chat-auto-parser-helpers.cpp:332-360`): render the template twice with one parameter changed, then `calculate_diff_split` which computes the longest common prefix/suffix **aligned to marker boundaries** (segmentize into `MARKER`/`TEXT` runs, prune whitespace-only text, `helpers.h:22-59`) and returns `{prefix, suffix, left, right}`.

Variants exercised (from the constructors and the in-tree design doc `docs/autoparser.md`):

- **Reasoning** (`analyze_reasoning`, ctor `:461`): assistant message with vs without `reasoning_content`; generation prompt with `enable_thinking=true` vs `false`; reasoning present with content vs only around tool calls. Yields `reasoning_mode` + `start`/`end` markers.
- **Content** (`analyze_content`, ctor `:677`): content with/without surrounding markers; yields `content_mode` (PLAIN / ALWAYS_WRAPPED / WRAPPED_WITH_REASONING) + markers.
- **Tools** (`analyze_tools`, ctor `:768`): with/without tools to isolate the tool "haystack", then classify by *where the function name and argument name appear*: JSON-native (`{...}` object), tag-with-JSON (`<function=X>{...}</function>`), tag-with-tagged (`<parameter=K>V</parameter>`); then extract per-call wrappers, function opener/closer, argument name/value/separator/args wrappers, call-id markers/position; probe parallel calls and JSON-array wrapping.
- **Start markers**: `detect_assistant_start_marker` / `detect_user_start_marker` render the template around one message to find role delimiters (`:347-460`), later stored as `message_delimiters` for the server.
- **Preserved tokens**: union of all non-empty markers (`:313-345`).

### 4.2 Deriving the parser

`peg_generator::generate_parser` (`chat-auto-parser-generator.cpp:23-97`):

1. `data.prompt = direct_apply(...)`, `data.generation_prompt = generation_prompt(...)`, `format = PEG_NATIVE`, preserved tokens + additional stops copied.
2. Continuation: if continuing a message, the generation prompt is rebuilt from `reasoning.start` + `reasoning_content` (+ content when continuing content).
3. `autoparser::build_parser` (`:99-136`) assembles, in priority order: response-format (JSON schema, optionally fenced in ```` ```json ````), tools (if any and `jinja_caps.supports_tool_calls`), else pure content. Reasoning is an optional prefix. `parser.save()` serializes the arena to the `parser` string.
4. Grammar: built when a JSON response format exists or tools are present; `grammar_lazy` only for `tool_choice=auto` with tools; triggers are the tool section start (or per-call start), plus an extra trigger when the OpenAI wrapper `{"type": "function",` is expected (`:67-94`).

The three tool builders (`:180-476`) mirror the classification: `build_tool_parser_json_native`, `build_tool_parser_tag_json`, `build_tool_parser_tag_tagged`. Arguments typed by JSON schema use `p.schema(...)`; tagged string args use an Aho-Corasick rule; required args are ordered (JSON) or permuted (tagged, via `p.permute`).

### 4.3 Workarounds, failures, fallbacks

- 12 ordered patch lambdas for concrete real-world templates (old Qwen/DeepSeek thinking templates, Granite 3.3, Cohere Command R+, Functionary 3.1, DeepSeek-R1-Distill-Qwen, Nemotron Nano v2, …) (`:36-222`, array closes at `:223`). They mutate the analysis (set markers, change formats) after the generic analysis ran.
- If a comparison cannot render (template raises), that analyzer degrades; `analyze_template` itself does not throw for that. The *generation* step can throw; the apply wrapper turns it into a user-facing `invalid_argument` (`common/chat.cpp:1363-1365`).
- The in-tree design doc (`docs/autoparser.md`) lists the failure taxonomy: reasoning not detected → reasoning stays in content; tool call ID not detected → synthesized IDs; parallel calls not detected → single call; markers not detected → whole output is content. It also notes the one explicit hard failure: a grammar that matches only EOS.
- **Doc drift**: `docs/autoparser.md` cites `common/chat-auto-parser.h:367-388` for the `autoparser` class (actual: 381-409) and `common/chat.cpp:1280-1310` for the fallback decision (actual: 1331-1339; the specialized dispatcher is 1090-1223). Treat the doc's structure as accurate but its line refs as stale.

---

## 5. `jinja/caps` — what it analyzes and how

`caps_get(jinja::program &)` (`common/jinja/caps.cpp:111+`) does **not** statically inspect the AST. It *executes* the compiled template against synthesized message/tool contexts with `context.is_get_stats = true`, then reads:

- **success/failure** of each probe render, and
- **operation statistics** per value (`value->stats.ops`, e.g. `test_is_string`, `selectattr`, `array_access`, `used`) to see how the template consumed the inputs.

Probes include: content as string vs array (typed content); empty content array; a system message (is it used?); a single tool call with object arguments (are arguments consumed? are they expected as strings? are `tool_calls`/`tools` iterated?); parallel calls; reasoning preservation; reasoning effort (`reasoning_effort` variable).

Result fields (`common/jinja/caps.h:10-33`): `supports_tools`, `supports_tool_calls`, `supports_system_role`, `supports_parallel_tool_calls`, `supports_preserve_reasoning`, `supports_reasoning_effort`, `supports_string_content`, `supports_typed_content`, `supports_object_arguments`; reported to `/props` via `to_map()` and used by `common_chat_templates_apply` to normalize messages *before* rendering. `caps_apply_preserve_reasoning` / `caps_apply_reasoning_effort` inject the corresponding context variables.

Cost note: caps run once at template construction and execute the template several times; the auto-parser then renders each variant twice per comparison, plus the workaround patches. The specialized routes render a bounded number of times (prompt, generation prompt, and the render-identical checks their builders make).

---

## 6. Specialized parsers, and the Qwen3-Coder handler

### 6.1 Detection

`common_chat_try_specialized_template` (`common/chat.cpp:1090-1223`) checks formats in order; each check is a conjunction of `src.find(...)` substrings (markers unique enough in practice). The Qwen3-Coder check (`:1212-1220`):

```cpp
if (src.find("<tool_call>")  != std::string::npos &&
    src.find("<function=")   != std::string::npos &&
    src.find("<parameter=")  != std::string::npos &&
    // Exclude models that don't use \n between tags
    src.find("'<tool_call><function=' ~ tool_call.name ~ '>'") == std::string::npos) {
    return common_chat_params_init_qwen3_coder(tmpl, params);
}
```

The comment names the family: **"Qwen3-Coder XML tool calls, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash"**. The exclusion only rejects the official Qwen templates that concatenate `<tool_call><function=` without a newline (those go to the auto-parser).

### 6.2 What the Qwen3-Coder handler does (`common/parsers/qwen3-coder.cpp`, 194 lines)

- `GEN_PREFIX = "<|im_start|>assistant\n"`; prompt/generation-prompt via the standard helpers; format `PEG_NATIVE`.
- `supports_reasoning = src.find("<think>")`; `is_qwen3_coder = !supports_reasoning` (Qwen3-Coder itself is not a thinking model).
- Thinking: `thinking_start_tag="<think>"`, `thinking_end_tags={"\\n</think>", "</think>", "<tool_call>"}` (force-close support); preserved tokens `<tool_call>`,`</tool_call>`, plus `<think>`,`</think>` when reasoning.
- `message_delimiters` include **`<|im_start|>user\n<tool_response>` labelled "Qwen3-Coder, Qwen3.5, Nemotron Nano 3"** and `<|im_start|>tool_response` for StepFun (`:32-38`) — direct evidence of upstream's Qwen3.5 intent.
- Parser: optional reasoning `<think> space until_one_of("</think>","<tool_call>") (">/think<" | peek("<tool_call>"))`; content until one of the tool-call starts; XML args as `Ac` (Aho-Corasick) rule `"xml-arg-string"` over `\n</parameter>\n`; JSON-schema-driven typed args; **required args permuted** ("Qwen does not always adhere to the order provided"); optional args `zero_or_more`; root trigger rule `tool-call-root`; for plain Qwen3-Coder, a leading `<tool_call>` is optional and `<function=NAME>` openers are added to the accepted starts (models sometimes drop `<tool_call>`).
- Lazy grammar: `grammar_lazy = tools && tool_choice == auto`; triggers per tool-call start (`<tool_call>` plus, for Qwen3-Coder, `<function=NAME>`).

Three upstream quirks worth carrying into a port: the arg string rule is Aho-Corasick-based (not plain `until`), the required-arg order is permuted with a `cycle` rule (bounded by `COMMON_CHAT_MAX_PERMUTE = 6`, `chat-peg-parser.h:58`), and the tool-call grammar uses `punctuation`-free exact newlines (the template's env-var `tool_call_format=json` mode is not modeled — see §9.4).

---

## 7. Public entry points

From `common/chat.h` (all C++ free functions; there is no C API in this stack):

| Entry point | Signature → result | Notes |
|---|---|---|
| `common_chat_templates_init` | `(llama_model*, chat_template_override, bos_override, eos_override)` → `common_chat_templates_ptr` | Reads `tokenizer.chat_template` (+`.tool_use`), patches two known template bugs, uses vocab bos/eos pieces; throws on parse failure (`chat.cpp:757-855`) |
| `common_chat_templates_apply` | `(tmpls, inputs)` → `common_chat_params` | The render+analyze step; `inputs.use_jinja` selects the jinja or legacy route (`:1434-1439`) |
| `common_chat_parse` | `(input, is_partial, parser_params)` → `common_chat_msg` | Requires `parser_params.parser` (arena) + `generation_prompt`; stateless w.r.t. template (`:1441-1445`) |
| `common_chat_peg_parse` | `(arena, input, is_partial, params)` → `common_chat_msg` | Direct-arena variant used by server tests and tools (`:1447-1523`) |
| `common_chat_msg_diff::compute_diffs` | `(prev, new)` → `vector<common_chat_msg_diff>` | Streaming deltas (content, reasoning, per-tool-call name/id/arguments) (`chat.h:130-143`) |
| `common_chat_format_single` | `(tmpls, past_msgs, new_msg, add_ass, use_jinja)` → `string` | Render one message in history position |
| `common_chat_format_example` | `(tmpls, use_jinja, kwargs)` → `string` | Example render for `/props` |
| `common_chat_verify_template` | `(src, use_jinja)` → `bool` | Apply to a test message; used by CLI validation |
| `common_chat_templates_support_enable_thinking` | `(tmpls)` → `bool` | Executes apply with a user message and returns `supports_thinking` (`:358-371`) |
| `common_chat_templates_get_caps` | `(tmpls)` → `map<string,bool>` | `/props` reporting |
| `common_chat_msgs_parse_oaicompat` / `common_chat_tools_parse_oaicompat` | JSON → `vector<common_chat_msg>` / `vector<common_chat_tool>` | OAI input conversion; tools keep `parameters` as a JSON string |
| `common_chat_tool_choice_parse_oaicompat` | `string` → `common_chat_tool_choice` | `auto`/`none`/`required` |
| `common_chat_tool_parameters` | `(function_json)` → `json` | Parameters schema of a function tool, `{}` if absent |
| `common_chat_role_from_string` / `_to_string` | role enum helpers | |
| `common_reasoning_format_from_name` / `_name` | string ↔ enum | |

`common_chat_msg` (`chat.h:80-128`): `role`, `content`, `content_parts` (typed parts; media markers), `tool_calls[{name, arguments(JSON string), id}]`, `reasoning_content`, `tool_name`, `tool_call_id`; `render_content(delimiter)` flattens typed parts; `set_tool_call_ids(cache, generator)` assigns stable streaming ids.

`common_chat_params` (`chat.h:269-283`): `format`, `prompt`, `grammar`, `grammar_lazy`, `generation_prompt`, `supports_thinking`, `thinking_start_tag`, `thinking_end_tags`, `grammar_triggers`, `preserved_tokens`, `additional_stops`, `parser` (serialized PEG arena), `message_delimiters`.

**Reasoning formats** (`common.h:420-426`, default `COMMON_REASONING_FORMAT_DEEPSEEK` at `common.h:650`):

- `NONE` — no extraction; reasoning text stays in content.
- `AUTO` — "Same as deepseek, using `message.reasoning_content`".
- `DEEPSEEK_LEGACY` — extract thinking-tag contents to `reasoning_content`, or leave inline in `<think>` tags **in stream mode**.
- `DEEPSEEK` — extract including in streaming deltas.

Extraction is realized at parser-construction time (`reasoning_format != NONE` gates the reasoning parser in both the generator and qwen3-coder); the server additionally sets `reasoning_in_content = stream && format == DEEPSEEK_LEGACY` (`tools/server/server-schema.cpp:307`) for its protocol writers.

---

## 8. Parse side: PEG engine, mapper, streaming

**Engine** (`common/peg-parser.cpp`):

- `common_peg_arena::parse(ctx)` → `std::visit` executor; results are `FAIL` / `SUCCESS` / `NEED_MORE_INPUT`, with `start`/`end`, AST node ids and invalid-UTF-8 runs (`peg-parser.h:136-160`).
- `NEED_MORE_INPUT` propagates through sequences/repetitions so that a prefix of a valid parse yields a *partial* AST; `atomic()` suppresses partial nodes for places where partial output is undesirable (`peg-parser.h:531-534`).
- `resolve_refs()` flattens `ref` nodes to rule ids; `build_grammar(builder, lazy)` generates GBNF for constrained decoding, emitting only trigger rules and descendants when `lazy` (`peg-parser.cpp:1606+`).
- Serialization is JSON (`{"parsers":[...],"rules":{...},"root":id}`), `save()=to_json().dump()`, `load()=from_json(parse())` (`peg-parser.cpp:1899-2118`). Schema parser nodes serialize only `{child,name,raw}` — the schema *document* isn't serialized, and the deserialized node has `node=nullptr`; that is harmless for parsing because the schema executor delegates to the child, and grammar is generated at apply time before serialization. **[inference from code read]**

**Mapper** (`common/chat-peg-parser.cpp:284-456`): walks the AST; tags drive semantics — `reasoning`/`content` texts are concatenated; `tool-open` starts a pending call; `tool-name` finalizes it (arguments buffered until the name is known, so streams don't leak unnamed calls); `tool-id` is trimmed of quotes; `tool-args` takes JSON text verbatim when it starts with `{`; arg names/values are assembled into a JSON object with proper escaping and brace balancing; pythonic scalars/containers are normalized to JSON; a whitespace-only `reasoning_content` is discarded (this is what makes the froggeric "`<think>\n\n</think>\n\n` prefill" parse cleanly). Variants: `common_chat_peg_gemma4_mapper`, `common_chat_peg_minimax_m3_mapper`, plus the legacy `tag_based_peg_mapper` (`chat-peg-parser.h:189-220`).

**Streaming** (`tools/server/server-task.cpp:151-209`): `task_result_state` re-parses the whole generated text on every chunk with `is_partial=true`; on parse failure during partial streaming it maps whatever AST was captured (`common/chat.cpp:1473-1495`); otherwise it diffs the previous and new `common_chat_msg`. The server then optionally delays tool-call deltas until a tool name is known (`filter_tool_calls`). Continuation requests call `common_chat_parse("", true, params)` once to prime the state (`server-task.cpp:156-159`).

---

## 9. The froggeric question — experiment and evidence

### 9.1 Template digest

Fetched `https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja` on 2026-09-27; sha256 `e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2`; 459 lines; `template_version = "qwen3.8-froggeric-v22.5"`.

Relevant shape: ChatML (`<|im_start|>role ... <|im_end|>`); thinking via `<think>…</think>` with prefill when thinking is disabled (`<|im_start|>assistant\n<think>\n\n</think>\n\n`); tool calls in XML by default (`tool_call_format='xml'`, a template variable, not in the default set): `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`; tool responses wrapped in `<tool_response>` inside a user turn; effort/thinking control markers `<|think_on|>`, `<|think_off|>`, `<|think_low|>` etc.; optional JSON tool mode via `tool_call_format='json'` rendering `<tool_call>\n{"name": ..., "arguments": ...}\n</tool_call>`.

### 9.2 Route: specialized Qwen3-Coder (not the auto-parser) **[verified]**

Built `test-chat-auto-parser` from the pinned SHA (CPU-only, `-DGGML_CUDA=OFF -DLLAMA_CURL=OFF`), then:

```
$ ./build/bin/test-chat-auto-parser /tmp/opencode/froggeric-chat_template.jinja --output=analysis
Options: with_tools=true, generation_prompt=true, enable_reasoning=true
Using specialized template: Qwen3-Coder
This template uses a specialized parser, analysis results will not be available.
```

The same holds for the in-tree `models/templates/Qwen3.5-4B.jinja`:

```
$ ./build/bin/test-chat-auto-parser models/templates/Qwen3.5-4B.jinja --output=analysis
Using specialized template: Qwen3-Coder
```

The generated parser and lazy grammar (both runs) are in Appendix B.1.

*Why:* the template source contains `<tool_call>`, `<function=` (in the prompt's example block and in the render loop) and `<parameter=`, and does **not** contain the excluded Qwen concatenation form `'<tool_call><function=' ~ tool_call.name ~ '>'` (0 occurrences).

### 9.3 Counterfactual: the auto-parser also covers it **[verified]**

To see what the differential analyzer would derive from froggeric's *actual renders*, a copy was made where only the detection trigger is obfuscated: every `<function=` in the source was rewritten as `<function' + '=' + '` (5 occurrences), a Jinja string-concatenation that renders byte-identically. sha256 of the modified copy: `8f7013a851859ec99565a570bde66b5e71d7ba824c2fcd314acc4638ab482832`.

```
$ ./build/bin/test-chat-auto-parser /tmp/opencode/froggeric-nospec.jinja --output=analysis
...
reasoning_mode: TAG_BASED
reasoning_start: '<think>\n'
reasoning_end: '\n</think>\n\n'
content_mode: PLAIN
tool_mode: TAG_WITH_TAGGED
supports_tools: true
supports_parallel_calls: true
per_call_start: '<tool_call>\n'
per_call_end: '</tool_call>'
func_name_prefix: '<function='
func_name_suffix: '>\n'
func_close: '</function>\n'
arg_name_prefix: '<parameter='
arg_value_suffix: '\n</parameter>\n'
call_id_pos: 'NONE'
...
=== Generated Grammar Triggers ===
Token: -1 | Type: 1 | Value: <tool_call>
=== Preserved Tokens ===
  '<think>' '</think>' '<tool_call>' '</tool_call>' '<function=' '>' '</function>' '<parameter=' '</parameter>'
```

Full parser/grammar in Appendix B.2. Conclusion: both routes can serve a Qwen3.8-froggeric-shaped template; upstream chose the hand-written handler, whose grammar and argument typing are tighter than the heuristic one.

### 9.4 Caveats

- **Source-based dispatch ignores runtime kwargs.** `tool_call_format='json'` for froggeric changes only the *rendered* prompt (the JSON branch), but detection reads the static source, so the XML handler still runs and would expect XML tool calls while the prompt teaches JSON. **[inference from `common/chat.cpp:1212-1220` + the template's `_tool_format` default; not exercised end-to-end]**. NInfer should either not expose such a mode or dispatch on the rendered shape.
- **Instruction text pollutes marker detection.** Froggeric's `<IMPORTANT>` block contains literal `<tool_call>…<function=…</parameter>` examples; the auto-parser handles that here (markers are derived from *renders*, where the instructions come from the tools-present branch), but any template whose prompt text contains tool syntax is a false-positive risk for the substring detectors.
- The counterfactual is a source-perturbed copy; only the detection string was changed, and rendered output was verified identical by construction (string concatenation), not by a byte diff of renders. The derived markers match froggeric's render loop exactly.

---

## 10. Tool schemas and grammars

- Tools arrive as OAI `[{type:"function", function:{name, description, parameters}}]`; `common_chat_tools_parse_oaicompat` keeps `parameters` as a JSON *string*; handlers parse it via `common_chat_tool_parameters` → always an object schema (`common/chat.cpp:577-585`).
- `common_chat_schema_from_json` builds the typed document (`json-schema.h:189-198`); a `$ref` table supports local refs; unsupported constructs throw.
- `foreach_parameter` (`parsers/parsers.cpp`) iterates `properties` in schema order with their document (owning shared_ptr), giving each arg a schema to constrain.
- Grammar emission: `json_schema_to_grammar(json)` (string API) or `build_grammar(callback)` with `common_grammar_builder::add_rule`/`add_schema` (`json-schema-to-grammar.h:12-23`); the PEG side additionally emits grammars for literals, `<`/`>`-style char classes, Aho-Corasick strings, and schema nodes (`peg-parser.cpp:1606-1816`).
- Triggers: `common_grammar_trigger{type: TOKEN|WORD|PATTERN|PATTERN_FULL, value}` (`common.h:145-155`); lazy grammars only allow trigger rules and descendants.
- Tool choice: `NONE` disables tool parsing; `REQUIRED` makes at least one call mandatory (min-call counts in the builders); `AUTO` makes calls optional and enables lazy grammar when a trigger marker exists.

---

## 11. Porting surface

### 11.1 What is llama.cpp-specific

| Dependency | Where used | Port implication |
|---|---|---|
| `common_json` (`common/json.{h,cpp}`) + vendored nlohmann | every file in the stack | vendor as-is (MIT; nlohmann is the only external library) |
| `llama_vocab` / `llama_tokens` / `common_token_to_piece` | `common_chat_templates_init` (bos/eos pieces), `common_chat_msg_delimiters::tokenize/split` | needed only for GGUF-init and message-span features; the PEG engine and mapper are tokenizer-free (string-based) |
| jinja runtime (`common/jinja`) | caps + all rendering | already vendored in this fork (§11.3) |
| `log.h` macros | all files | trivial shim |
| `trie.*` (Aho-Corasick), `unicode.*` (UTF-8) | `ac()` parsers, GBNF, AST sanitization | small, self-contained |
| `common.h` (reasoning-format enum, grammar triggers, `common_token_to_piece`) | `chat.h`, generator, handlers | split: keep the two enums; the rest is unrelated config surface |
| Grammar/decoding integration (`llama_grammar` triggers, sampling) | server only | NInfer must decide where GBNF/triggers land in its sampler |
| Server/CLI plumbing (`tools/server/*`, `common/arg.cpp`) | request parsing, `/props`, streaming protocol | not part of the port; NInfer owns its own transport (per product rules) |

### 11.2 Minimal vs full port

- **Minimal (product-fixed templates: Qwen3.5/3.6/3.8 family).** `peg-parser` (2681) + `chat-peg-parser` (1452) + `chat.{h,cpp}` (1924) + `json-schema.{h,cpp}` (712) + `common/parsers/qwen3-coder.cpp` (194) ≈ 7.0k lines, plus the GBNF path (`json-schema-to-grammar` + builder, 1051) ≈ 8.0k if constrained decoding is wanted. No `jinja/caps`, no diff analyzer, no other specialized parsers.
- **Full (arbitrary user templates, portable frontend).** Adds the analysis stack `chat-auto-parser*` + `chat-diff-analyzer` (≈3.0k) + `jinja/caps` (613) + the 14 other specialized parsers (≈2.3k). This is the configuration that reproduces llama.cpp's robustness across the 71 in-tree fixtures.
- **Deliberately not needed for parsing:** `reasoning-budget` (sampler), the legacy `llama_chat_apply_template` route, the server's protocol writers, and NInfer's own media/vision handling (out of scope per map: vision RoPE etc.).

### 11.3 Current NInfer state (fork context, for design tickets)

- `third_party/llama-jinja/` is a **maintained fork of llama.cpp `common/jinja` at commit `7609846557c50f9d984719a9e1e8c5f3d02f807b`** (`third_party/llama-jinja/README.ninfer.md`), MIT, with vendored Unicode 17 case data; wrapper at `src/text/jinja.{h,cpp}`. It does **not** contain `caps.h`/`caps.cpp` (the caps layer is newer than the vendored commit and is required only for the auto-parser/apply normalization path). Updating that fork or adding caps is a design decision for #28.
- The current bespoke frontend lives in `src/models/qwen3_5/frontend/` (e.g. `output_session.cpp`, `tool_call_parser.h`, `prompt_layout.h`), which is exactly what map #25 replaces with the ported machinery.
- Product templates are shipped at `tools/chat_templates/qwen3_6.jinja` and `tools/chat_templates/qwen3_8.jinja`.

---

## 12. Provenance, license, churn

- **License**: MIT (`LICENSE`), "Copyright (c) 2023-2026 The ggml authors". `common/jinja` is upstream's own implementation — "a Jinja template engine implementation in C++, originally inspired by huggingface.js's jinja package", introduced in PR #18462 (`common/jinja/README.md:1-5`) — under the same MIT grant.
- **Baseline for the port**: `95887577ab5fead779581a7030a83c7752ff3234` (2026-09-26). Record this in the same way the fork records the jinja base (`76098465…` in `third_party/llama-jinja/README.ninfer.md`).

Churn in the 12 months before 2026-09-27 (GitHub API, `commits?path=…&since=2025-09-27`, capped at 100 by pagination) and last-touch date:

| File | commits (12 mo) | last touched |
|---|---|---|
| `common/chat.cpp` | ≥100 | 2026-09-22 |
| `common/chat.h` | 33 | 2026-09-12 |
| `common/chat-auto-parser-generator.cpp` | 32 | 2026-09-12 |
| `common/chat-peg-parser.cpp` | 25 | 2026-09-20 |
| `common/jinja/runtime.cpp` | 25 | 2026-09-23 |
| `common/chat-diff-analyzer.cpp` | 21 | 2026-08-22 |
| `common/peg-parser.cpp` | 19 | 2026-09-20 |
| `common/json-schema-to-grammar.cpp` | 15 | 2026-09-19 |
| `common/chat-auto-parser.h` | 14 | 2026-08-22 |
| `common/jinja/caps.cpp` | 14 | 2026-09-07 |
| `common/reasoning-budget.cpp` | 12 | 2026-08-10 |
| `common/chat-auto-parser-helpers.cpp` | 7 | 2026-08-22 |
| `common/parsers/qwen3-coder.cpp` | 4 | 2026-09-16 |
| `common/json-schema.cpp` | 1 | 2026-09-12 |

Repo-wide: 147 commits in the 7 days before 2026-09-27 (~21/day). Implication: the *parse engine* moves much slower than the *format zoo*; vendoring `peg-parser`/`chat-peg-parser`/`qwen3-coder.cpp` is a stable base, while `chat.cpp`/`chat-auto-parser*` need the drift alert the map already decided on.

---

## 13. Test and verification surface

- `tests/test-chat.cpp` (7647 lines) — the PEG/format acceptance suite, one `peg_tester` per `models/templates/*.jinja` (71 fixtures), with chat/tool/reasoning expectations; supports `--template <substr>` and a file argument that runs a template against expected outputs.
- `tests/test-chat-peg-parser.cpp` (1106) — builder/mapper unit tests incl. hand-built qwen3-coder-like parsers.
- `tests/test-chat-auto-parser.cpp` (2707) — diff-split/compare-variants unit tests, per-template analysis tests (Seed-OSS, Cohere, Nemotron, Laguna…), plus the debug CLI used in §9.
- `tests/test-chat-analysis.cpp` (615) — template diff/analysis reporter (`--template <name>`, `--template-file <path>`).
- `tests/test-chat-template.cpp` (738), `tests/test-peg-parser.cpp` (26), `tests/test-jinja.cpp`, `tests/test-grammar-parser.cpp`.
- Baseline run of the auto-parser suite at the pinned commit **[verified]**: `tests: 111, assertions: 524, failures: 0, exceptions: 0, skipped: 0`.
- Reproduce (CPU-only): `cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON && cmake --build build --target test-chat-auto-parser test-chat-analysis -j`.

---

## 14. Inputs for the design tickets (#28 integration, #29 qualification)

Open decision points this research surfaces:

1. **Route choice.** The product's Qwen3.5/3.8 templates are covered by the specialized Qwen3-Coder handler; whether NInfer ports only that path plus the PEG engine, or also the auto-parser + caps (needed for arbitrary user templates), determines whether `jinja/caps` must be ported and whether the vendored jinja fork needs a rebase. The map says "user recipes use the same architecture", not necessarily arbitrary chat templates — confirm.
2. **Rendering ownership.** The apply path in llama.cpp does message normalization *before* rendering (caps-driven workarounds). If NInfer keeps its current frontend contract (streaming channels, thinking budget, stop policy, prefix-cache rendering — map #25 decisions), the port must decide which normalizations move into the renderer and which into the parser.
3. **Parser lifecycle.** llama.cpp's serialized arena (`save`/`load`, opaque string, schema docs stripped) works for a server. NInfer can keep arenas in memory per template; decide whether to copy the serialization (for config parity/tests) or build the arena at startup.
4. **Grammar integration.** Lazy grammar + triggers + `preserved_tokens` are part of the apply output. NInfer must map `common_grammar_trigger` onto its own sampling/constraint layer (does NInfer currently support GBNF-style constraints? — outside this ticket's scope, needed for parity of constrained tool calls).
5. **Verification corpus (#29).** Available oracles: upstream's 71-template suite + the in-tree per-format tests (`test-chat.cpp`), the froggeric template itself (as a Qwen3.8-shaped fixture), and the two live experiments in §9. The qualification contract should define independent-oracle math/semantics for the marker analysis (renders and diffs) and for the PEG engine (string-level, not kernel-level).
6. **Failure semantics.** Upstream throws when the auto-parser cannot build (`invalid_argument`), and streams partial ASTs when a close is missing. NInfer's serving contract needs an explicit decision on both (error vs pure-content fallback).

Uncertainties that remain after this research: the exact set of templates NInfer must support beyond Qwen3.5/3.6/3.8 (decides minimal vs full port); whether the product wants llama.cpp-compatible `reasoning_format` semantics or its own channels; and whether `message_delimiters` (tokenizer-level prompt spans) is needed in NInfer at all.

---

## Appendix A — commands used

```bash
# fetch the baseline
git ls-remote https://github.com/ggml-org/llama.cpp refs/heads/master   # 95887577ab5fead779581a7030a83c7752ff3234
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/opencode/llamacpp-ref

# fetch the froggeric template
curl -fsSL https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja \
     -o /tmp/opencode/froggeric-chat_template.jinja

# build the analysis tools (CPU only)
cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test-chat-auto-parser test-chat-analysis -j "$(nproc)"

# experiments
./build/bin/test-chat-auto-parser /tmp/opencode/froggeric-chat_template.jinja --output=analysis      # → specialized Qwen3-Coder
./build/bin/test-chat-auto-parser /tmp/opencode/froggeric-nospec.jinja     --output=analysis       # → auto-parser derivation
./build/bin/test-chat-auto-parser                                                                   # 111 tests / 524 assertions, 0 failures
```

The nospec copy was produced with:

```python
src  = open('/tmp/opencode/froggeric-chat_template.jinja').read()
out  = src.replace('<function=', "<function' + '=' + '")   # 5 occurrences; render-identical
open('/tmp/opencode/froggeric-nospec.jinja','w').write(out)
```

## Appendix B — raw experiment excerpts

### B.1 Froggeric → specialized Qwen3-Coder (excerpt)

```
Analyzing template: /tmp/opencode/froggeric-chat_template.jinja
Options: with_tools=true, generation_prompt=true, enable_reasoning=true
Using specialized template: Qwen3-Coder

=== Generated Parser ===
Sequence(Literal(<|im_start|>assistant
), Repetition(Sequence(Literal(<think>), Space, Tag(reasoning, Until(</think> | <tool_call>)),
Choice(Literal(</think>), And(Literal(<tool_call>)))), 0, 1), Space, Tag(content, Until(<tool_call>)),
Space, Rule(tool-call-root, Repetition(Sequence(Rule(tool-call, Sequence(Literal(<tool_call>
), Choice(Rule(tool-test-function-name, Tag(tool, Sequence(Atomic(Tag(tool-open, Sequence(Literal(<function=),
Atomic(Tag(tool-name, Literal(test_function_name))), Literal(>
)))), Tag(tool-args, Rule(tool-test-function-name-args-3, Choice(... Rule(tool-test-function-name-arg-param1,
Tag(tool-arg, Sequence(Atomic(Tag(tool-arg-open, Sequence(Literal(<parameter=), Atomic(Tag(tool-arg-name,
Literal(param1))), Literal(>
)))), Rule(xml-arg-string, Ac(
</parameter>
, Sequence(Tag(tool-arg-string-value, Until(
</parameter>
)), Atomic(Tag(tool-arg-close, Literal(
</parameter>
))))))))), ...)))), Atomic(Tag(tool-close, Literal(</function>
))))))), Literal(</tool_call>), Space)), Repetition([cycle], 0, unbounded)), 0, 1)))

=== Generated Grammar Triggers ===
Token: -1 | Type: 1 | Value: <tool_call>

=== Preserved Tokens ===
  '<tool_call>'  '</tool_call>'  '<think>'  '</think>'
```

### B.2 Auto-parser derivation of the render-identical froggeric copy (excerpt)

```
user_msg_start: <|im_start|>user
assistant_msg_start: <|im_start|>assistant
reasoning_mode: TAG_BASED
reasoning_start: '<think>\n'
reasoning_end: '\n</think>\n\n'
content_mode: PLAIN
tool_mode: TAG_WITH_TAGGED
supports_tools: true
supports_parallel_calls: true
tool_section_start: ''
tool_section_end: ''
per_call_start: '<tool_call>\n'
per_call_end: '</tool_call>'
func_name_prefix: '<function='
func_name_suffix: '>\n'
func_args_separator: ''
func_close: '</function>\n'
call_id_pos: 'NONE'
arg_name_prefix: '<parameter='
arg_name_suffix: '>\n'
arg_value_suffix: '\n</parameter>\n'
...
=== Generated Parser ===
Sequence(Literal(<|im_start|>assistant
), Space, Repetition(Sequence(Epsilon, Literal(<think>), ... Tag(reasoning, Until(</think>)) ...,
Literal(</think>), ...), 0, 1), Repetition(Tag(content, Until(<tool_call>
)), 0, 1), Repetition(Rule(tool-call, Sequence(Literal(<tool_call>
), Space, Choice(Rule(tool-test-function-name, Sequence(Atomic(Tag(tool-open, Sequence(Literal(<function=), ...))),
..., Tag(tool-args, ... Ac(
</parameter>
, ...)))))), Space, Literal(</tool_call>), ...), 0, 1), End)

=== Generated Grammar ===
root ::= tool-call
tool-call ::= "<tool_call>\n" space (tool-test-function-name) space "</tool_call>" (space "<tool_call>\n" ... )* space
tool-test-function-name ::= ("<function=" "test_function_name" ">\n") space ... "</function>\n"
...

=== Preserved Tokens ===
  '<think>'  '</think>'  '<tool_call>'  '</tool_call>'  '<function='  '>'  '</function>'  '<parameter='  '</parameter>'
```
