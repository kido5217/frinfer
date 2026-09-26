# Turn semantics and oracle corpus: Qwen3.5 on the froggeric wire format

Ticket: [#27](https://github.com/kido5217/ninfer-yarn/issues/27) (map [#25](https://github.com/kido5217/ninfer-yarn/issues/25)).
Branch: `research/chat-parsing-oracle`. Scope: the parsing semantics of one assistant turn on the
froggeric wire format, the points where independent implementations agree, the points where they
disagree, the semantics NInfer should follow, and a machine-usable input→expected-output corpus that
encodes them.

This document is the oracle for design tickets #28 (port seam) and #29 (qualification contract) and
for the port ticket #34. Section 6 is written to be lifted into a test corpus as-is.

## 1. Pinned revisions

| Source | Revision | Read at |
|---|---|---|
| llama.cpp (port source) | `95887577ab5fead779581a7030a83c7752ff3234` (master, 2026-09-26) | local sparse clone `/tmp/opencode/src/llama.cpp` |
| vLLM (semantic oracle) | `77871126f9b69cff9feffff390d1171a60dbd7e2` (main, 2026-09-26) | local sparse clone `/tmp/opencode/src/vllm` |
| froggeric `Qwen-Fixed-Chat-Templates` | `main` = `855bffc49448e299789730ff92c9b8d834d6cc14`; version v22.5; `chat_template.jinja` sha256 `e57684ba…49c4b2` (28 234 B); `README.md` sha256 `b746a44e…64b63` | `git ls-remote` + byte-exact download |
| NInfer fork (current frontend) | `c6a3b1e1` (master) | worktree `/tmp/opencode/wt-chat-parsing-oracle` |
| PR #309 (Neroued/ninfer, closed unmerged) | head `713045812a5bb861b089d4c430c4a03bdbd526df`, base `594930e7b609efa4bcea3ae4f24cd9d66b5f224f`, +301/−23, 5 files, closed 2026-09-25 | `gh pr diff 309 --repo Neroued/ninfer`, `gh pr view 309 --repo Neroued/ninfer` |

Source tiers used below: **P1** = code read at the pinned revision; **P1-run** = the pinned code
executed for this document (method in §8); **P2** = the project's own tests/docs at that revision;
**P3** = README-level guidance.

## 2. The froggeric wire format (P1: `chat_template.jinja` at `855bffc4`)

The template is a pure Jinja template; the wire format is what it renders for the prompt and what
the model reproduces in the output. Three strings matter.

### 2.1 Generation prompt (template lines 453-460)

| `enable_thinking` | rendered tail after the last turn |
|---|---|
| `true` (default) | `<|im_start|>assistant\n<think>\n` |
| `false` | `<|im_start|>assistant\n<think>\n\n</think>\n\n` |

With thinking enabled the model opens directly inside the reasoning block: the `</think>` close is
generated, the `<think>` open is not. With thinking disabled the block is pre-closed by the prompt
and the model's output starts in the content channel.

### 2.2 Assistant history (template lines 331-343, 407-411)

```
<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}
...
<|im_end|>\n
```
Tool calls are appended inside the same assistant turn as canonical XML (template lines 366-375,
377-390, 407):

```
<tool_call>\n<function={name}>\n<parameter={key}>\n{value}\n</parameter>\n…</function>\n</tool_call>
```

Multiple calls are separate complete `<tool_call>…</tool_call>` blocks; the template's instructions
(lines 193, 209) say they must be "a separate, completely closed `<tool_call></tool_call>` block for
EACH function" and "Do NOT nest `<tool_call>` blocks". Values are the argument text verbatim
(strings) or `tojson` (non-strings); each value is wrapped by a newline on both sides.

### 2.3 Reasoning/instruction text the template puts in the system prompt

The template instructs the model to emit `</think>` (line 181/197) and to place the tool call
"immediately after thinking, with NO conversational text before it" (line 189). The model is *taught*
not to emit prose before a call, but no engine can assume it complies.

### 2.4 Froggeric's engine guidance (P3: `README.md` at `855bffc4`)

- llama.cpp: `--jinja --chat-template-file chat_template.jinja --reasoning-format deepseek`
  ("this flag extracts `<think>` blocks into the dedicated `reasoning_content` API response field.
  This prevents raw thinking tokens from leaking into the text stream and stopping tool calls
  midway", README line 55).
- vLLM: `--reasoning-parser qwen3 --tool-call-parser qwen3_xml` (README line 68); current releases
  use `qwen3_xml`, older builds `qwen3_coder`; `tool_call_format="json"` switches to `hermes`.
- The compatibility table (README line 127) claims "Qwen-native parsers (like vLLM) crash on JSON
  formatting" and that canonical XML is the default precisely because of it.

The two vLLM flags name real registrations at the pinned revision: `qwen3_xml` and `qwen3_coder`
both resolve to `Qwen3EngineToolParser` (`tool_parsers/__init__.py`), which is a thin subclass of the
engine adapter `Qwen3ParserToolAdapter` (`tool_parsers/qwen3_engine_tool_parser.py`); `qwen3`
reasoning resolves to `Qwen3ParserReasoningAdapter` (`reasoning/__init__.py`), i.e. the same
`Qwen3Parser` (§3.1). All of froggeric's vLLM advice therefore describes one parser.

## 3. Reference semantics

### 3.1 vLLM at `77871126` (P1)

`vllm/parser/qwen3.py` defines the format declaratively and `vllm/parser/engine/*` executes it.

**Terminals** (`qwen3.py` 41-57, 105-123): `<think>`, `</think>`, `<tool_call>`, `</tool_call>`,
`<function=`, `</function>`, `<parameter=`, `</parameter>`, `>`; the first four are also registered
as *token-id* terminals (`token_id_terminals`), so a special token id can pre-lex them.

**States and transitions** (`qwen3.py` 124-197): `REASONING → CONTENT` on `</think>`;
`REASONING → TOOL_PREAMBLE` on `<tool_call>` (implicit reasoning end, emitting
`REASONING_END` + `TOOL_CALL_START`); `CONTENT → TOOL_PREAMBLE` on `<tool_call>`
(*also* emitting `REASONING_END`, a no-op when reasoning already ended);
`CONTENT → TOOL_NAME` on `<function=` (the no-wrapper fallback);
`TOOL_NAME → TOOL_ARGS` on `>`; `TOOL_ARGS → TOOL_ARGS` on `<parameter=`/`</parameter>`;
`TOOL_ARGS/TOOL_NAME → TOOL_BETWEEN` on `</function>` (each emitting `TOOL_CALL_END`);
`TOOL_BETWEEN → CONTENT` on `</tool_call>`;
`TOOL_BETWEEN → TOOL_PREAMBLE/TOOL_NAME` on `<tool_call>`/`<function=` (consecutive calls *without*
a closing `</tool_call>` are accepted);
`CONTENT → CONTENT` on `</think>` (duplicate close is **absorbed silently**).

**Configuration**: `initial_state = REASONING` iff thinking enabled; `wait_for_reasoning` likewise;
`strip_trailing_reasoning_whitespace = False`; `stream_arg_deltas = True`; `tool_args_json = False`.
`Qwen3Parser.extract_reasoning` short-circuits to `(None, whole_output)` when thinking is disabled.

**Argument conversion** (`qwen3.py` 51-86): non-streaming args are recovered with two regexes
(`_PARAM_RE` with a `(?=<parameter=)` lookahead, `_PARTIAL_PARAM_RE` for the trailing partial);
`_trim_wrapping_newlines` strips **exactly one** leading and one trailing newline from a value.
Without a tool schema every value stays a JSON **string** (P2 `test_various_data_types`: `42`,
`3.14`, `true`, `null`, `["a","b"]`, `{"nested":"value"}` all stay strings). With a schema,
`_fix_arg_types` coerces per declared type, recursing into objects/arrays; a type union admitting
`string` is never coerced (`test_string_param_not_coerced_to_int`, `test_anyof_string_param_not_coerced`).

**Streaming text matching** (`incremental_lexer.py`): terminals are matched textually with
prefix-match buffering — a suffix that could still grow into a terminal is held until the next
delta; terminals take priority literals-first, longest-pattern-first. Token-id terminals pre-lex
only when the id is in `delta_token_ids` *and* the text arrived.

**Client-visible streaming** (P2 `tests/parser/engine/test_qwen3.py`,
`test_qwen3_reasoning.py`): reasoning and content deltas per round; tool-call deltas carry a partial
JSON argument fragment as the value streams
(`args_after_partial_tag == '{"query": "hello '` while `<param…` is still arriving, i.e. partial
markup must never leak into the previous value and the value's trailing space is preserved);
duplicate `</think>` in the content state is dropped from the emitted content
(`test_streaming_duplicate_think_end_absorbed` → `content == "contentmore"`); an unclosed reasoning
block at end of turn is closed implicitly and everything stays reasoning
(`finish()` emits `REASONING_END`; `test_no_end_tag_all_reasoning` → everything reasoning).

**P1-run** (real engine driven standalone on text-only input, method §8.1) confirms both failure
shapes that matter here:

- a quoted `</think>` inside reasoning ends the reasoning channel at that marker, the tail goes to
  content, and a second `</think>` is dropped silently;
- a quoted `<tool_call>` in reasoning emits `REASONING_END` + `TOOL_CALL_START`, then **emits nothing
  for the rest of the turn** (no content, no name, no error; `finish()` emits an empty
  `TOOL_CALL_END`);
- a *malformed* `<tool_call>` region that ends with `</tool_call>` returns to `CONTENT`, and the tail
  is emitted as content (so vLLM recovers the tail only when the malformed region is closed).

### 3.2 llama.cpp at `95887577` (P1 + P1-run)

The froggeric template selects the **specialized Qwen3-Coder parser**
(`common/parsers/qwen3-coder.cpp`), not the differential auto-parser (`docs/autoparser.md`; verified
live in ticket #26 and reproduced here: the generated parser dump starts with the qwen3-coder rule
set). The parser is a PEG grammar built from the *template source and the declared tools*; the
generation prompt literal and the tool names/parameter names are baked in as literals.

Grammar shape (P1-run AST dump, §8.2):

```
Seq( Literal("<|im_start|>assistant\n"),
     Opt( Seq( Literal("<think>"), Space, Tag(reasoning, Until(</think> | <tool_call>)),
               (Literal("</think>") | And(Literal("<tool_call>"))) ) ),
     Space, Tag(content, Until(<tool_call>)), Space,
     Rule(tool-call-root, Repeat( Rule(tool-call, Seq(Literal("<tool_call>\n"), <one tool rule>, Literal("</tool_call>"), Space)), 0, 1)) )
```

Per tool: `<function=NAME>\n` + args + `</function>\n`; a string arg is
`<parameter=NAME>\n` + `Until("\n</parameter>\n")` + `"\n</parameter>\n"`; required args are
`permute`d (any order) and optional args are a `zero_or_more` *after* them; non-string args are
schema-constrained JSON plus the same close (`qwen3-coder.cpp` 91-151). `parallel_tool_calls`
controls whether one or many `<tool_call>` blocks are accepted (line 166).

Semantics that differ from what a "wire format" alone implies (all P1-run, §8.2; the template's
thinking prompt was used, thinking on/off as stated):

| Input | Result |
|---|---|
| `I need to think.\n</think>\n\nThe answer is 42.` | reasoning `I need to think.\n`, content `The answer is 42.` |
| quoted `</think>` mid-reasoning then a real `</think>` | reasoning ends at the **first** marker; the tail (including the second marker) is content, and the byte after the marker is eaten by `space()` |
| quoted `<tool_call>` mid-reasoning | reasoning ends at the marker; **the rest of the turn is silently dropped** (no call, no content, no error) |
| valid call + trailing prose | call parsed, prose silently dropped |
| quoted malformed region + prose + real call | **everything after the quoted marker dropped**, real call lost |
| malformed `<tool_call>` region + `</tool_call>` + prose | whole message empty (nothing is recovered) |
| two complete calls, `parallel_tool_calls=false` | second call silently dropped |
| two complete calls, `parallel_tool_calls=true` | both calls parsed |
| consecutive calls without `</tool_call>` | everything dropped |
| `<parameter=x>value</parameter>` (vLLM style, no newlines) | dropped (newline framing is required) |
| unknown tool name / undeclared parameter / required parameter missing / optional parameter before a required one / a repeated required parameter | dropped |
| a repeated optional parameter | accepted, and the JSON argument text carries the duplicate key |
| `</think>` inside a parameter value | value preserved |
| split marker across rounds | buffered, no leak |
| unclosed reasoning at end of turn | stays reasoning |
| `--reasoning-format none` | the reasoning grammar node is not built: everything (including `<think>` tags) is content and reasoning is empty |

Two further structural facts matter for the port:

- there is **no end-of-input requirement**: the root grammar may stop early and succeed, which is why
  malformed tails are dropped without an error. `common_chat_peg_parse` throws only when the root
  *fails* (`common/chat.cpp` 1473-1499); a root that succeeds before EOF returns a message that
  silently omits the rest;
- tool calls are published all-or-nothing: the PEG mapper produces a tool call only from a complete
  node subtree, so streaming deltas show name and arguments only once the call is complete
  (P1-run server-diff emulation, §8.2).

Server-level facts (P1): `task_result_state::update_chat_msg` only replaces its message when the new
parse is non-empty (`tools/server/server-task.cpp` 170-177); there is no content fallback for an
empty parse. `--reasoning-format` = `none` | `auto` (default, = deepseek) | `deepseek` |
`deepseek-legacy` (`common/arg.cpp` 3675-3684; `common/common.h` 420-426).

### 3.3 NInfer today and PR #309

Current master has two bespoke components: `OutputSession` closes the reasoning channel at the first
`</think>` (no boundary rule, and no implicit close at a `<tool_call>`: while reasoning is open the
marker is ordinary reasoning text, `src/models/qwen3_5/frontend/output_session.cpp` 259-288);
`parse_qwen_tool_call_output` commits to the **first** `<tool_call>`
marker and accepts the whole region only if it parses completely (all-or-nothing per region, with
`TrailingContent`, `MalformedStructure`, `DuplicateParameter`, `InvalidToolName`, `UndeclaredTool`
fallback reasons; `src/models/qwen3_5/frontend/tool_call_parser.cpp` 410-625). Values are trimmed of
format whitespace and, when the tool is declared with a non-string type, normalized per the declared
schema (`normalize_declared_parameter`, `normalize_parameter`); a schema that admits `string` always
produces a JSON string.

`docs/serving.md` (pre-PR309) states the nested-parameter rule: an unmatched nested `<parameter=…>`
opener or a standalone `</parameter>` "cannot be represented unambiguously; either causes the
complete tool-call region to fall back to ordinary content".

PR #309 (`fix(frontend): keep quoted reasoning closes and later tool-call markers`, closed unmerged;
the map's decision is that its semantics and fixtures are absorbed by the port) adds exactly the two
recovery rules this corpus needs:

1. **Boundary rule for the reasoning close** (`output_session.cpp`): a `</think>` counts as the close
   only when followed by a format-whitespace byte (` \t\r\n`) or by the implicit end of turn; a
   marker at the end of the available bytes is *held* until the following byte arrives; at terminal
   time a still-pending marker is the implicit close (consumed, not published) and the bytes before
   it are published as reasoning.
2. **Consume-to-the-end candidate search for the tool region** (`tool_call_parser.cpp`): try each
   `<tool_call>` occurrence in order and accept the first region that parses completely *and*
   consumes the response to its end; earlier markers and the prose between them stay ordinary
   content; if no candidate qualifies, the whole response is content (all-or-nothing, diagnostics
   carry the first failure). The streaming decoder holds bytes from the first marker and flushes the
   held prefix so that `visible + terminal.content` stays byte-exact.

PR #309 also records an inherent limit: "a quoted marker followed by real whitespace is locally
indistinguishable from a close, so an adversarial quote can still end the reasoning channel.
Tool-call recovery keeps the turn usable in that case." Its live measurements: built-in template
5/5 demoted → 4/4 calls; froggeric v22.5 4/4 demoted → 4/4 calls.

### 3.4 What is *not* in this document

Rendering (already byte-identical, ticket #31), template kwargs, media, the OpenAI/Anthropic schema
mapping of the parsed result, and the port's C++ seam. The corpus below is parser/channel-level only,
using the vocabulary of the frontend seam: per-round append-only channel deltas and terminal tool
calls.

## 4. Agreement matrix

Codes: **T** thinking on, **F** thinking off. "drop tail" = the rest of the turn is silently lost;
"content" = demoted to the content channel; "hold" = bytes withheld until disambiguated.

| # | Case | vLLM `77871126` | llama.cpp `95887577` (qwen3-coder) | NInfer `c6a3b1e1` | NInfer target |
|---|---|---|---|---|---|
| 1 | `</think>` then content | close, content | close, content | close, content | same |
| 2 | `<tool_call>` without `</think>` | implicit close | implicit close (`peek`) | **no implicit close**: stays reasoning | implicit close (row 2) |
| 3 | `</think>` at turn end | implicit close at `finish()` | implicit close (lenient `until`) | terminal flush closes reasoning | same |
| 4 | no close at turn end | reasoning | reasoning | reasoning | same |
| 5 | quoted `</think>` + quote after | close at first marker | close at first marker | close at first marker | **stays reasoning** (boundary rule) |
| 6 | `</think>` at end of round | hold (prefix buffering) | hold (PEG prefix leniency) | hold (`longest_suffix_prefix`) | **hold** (same) |
| 7 | `</think>` followed by space | close | close | close | close (inherent limit) |
| 8 | model re-emits `<think>` | start consumed | start consumed | n/a | consumed |
| 9 | T-off: plain text | content | content | content | same |
| 10 | T-off: `</think>` in text | **dropped** from content | kept in content | kept (content) | kept (byte-exact) |
| 11 | `</think>` inside an arg value | value | value | value | value |
| 12 | canonical call | call | call | call | call |
| 13 | value framing newlines | 1+1 newline trimmed | `\n` after `>` and before `</parameter>` | framing newline, then whitespace trim | 1+1 newline, then policy trim |
| 14 | two complete calls | both | both only with `parallel_tool_calls` | both | both |
| 15 | calls without `</tool_call>` | both (tolerant) | **drop tail** | fallback | tolerant (vLLM) |
| 16 | empty arguments | `{}` | `{}` | `{}` | `{}` |
| 16b | required argument missing | accepted (schema-less parse) | **drop tail** (required args are grammar) | accepted | accepted + diagnostic |
| 17 | optional arg before required | any order | **drop tail** | any order | any order |
| 18 | JSON-looking text in string arg | string | string | string (declared) / raw JSON (undeclared) | declared-string rule (see B6) |
| 19 | typed value (`42`) | typed only with schema | typed | typed (declared) | typed (declared) |
| 20 | duplicate parameter | last wins | required duplicate: **drop tail**; optional duplicate: **emits duplicate JSON keys** | fallback | fallback (reject region) |
| 21 | nested `<parameter=…>` in a value | **re-split as args** | value is `until("\n</parameter>\n")`: a non-newline-framed nested close stays in the value; a newline-framed nested close ends the value and the turn fails | balanced nesting is the value | balanced nesting is the value |
| 22 | unmatched nested opener / standalone close | text or drop | drop tail | fallback (region → content) | region → content, **later candidates tried** |
| 23 | quoted malformed `<tool_call>` before a real call | **drop tail** (swallow) | **drop tail** | first marker commits → whole turn content | **recover the call, quoted bytes as content** |
| 24 | real call + trailing prose | swallow tail | drop tail | fallback (TrailingContent) | fallback (all-or-nothing, PR309) |
| 25 | `<tool_call>` split across rounds | hold, no leak | hold, no leak | hold (`marker_prefix_bytes_`) | hold, no leak |
| 26 | quoted `<tool_call>`, no valid call | drop tail | drop tail | thinking on: stays reasoning (no implicit close); thinking off: first marker commits → content | content (all-or-nothing) |
| 27 | undeclared parameter | accepted | **drop tail** | accepted + `SchemaMismatch` diagnostic | accepted + diagnostic |
| 28 | undeclared tool name | accepted | **drop tail** | fallback (`UndeclaredTool`) | fallback to content (serving policy) |
| 29 | thinking budget with quoted close | n/a | n/a | budget fires at first marker | budget uses the boundary rule |

Rows 5, 10, 15, 16b, 17, 20, 21, 23, 24, 26, 27, 28 are the disagreements a port decision has to
settle; every other row is agreement across at least three implementations and is a plain regression
vector.

## 5. Semantics NInfer should follow

The rules below are the target for the port. `R` rules are channel semantics; `B` rules are
structural (tool region). Each states the reason and the references it follows or rejects.

**Reasoning channel.**

- **R1 — open.** With thinking enabled the turn starts in the reasoning channel; a `<think>` in the
  output is consumed and never published (the template already opened the block). Without thinking
  the turn starts in content. *Follows all references; P2 vLLM `test_no_start_token_in_output`,
  `test_thinking_disabled_initial_state_is_content`; P1-run llama.cpp.*
- **R2 — close.** The reasoning channel closes at the first `</think>` that is followed by a
  format-whitespace byte (space, tab, CR, LF) or by the end of the turn. Whitespace directly after
  the close is not published as content. *Boundary rule from PR #309; the two other references close
  at any marker (row 5). Reason: the model reasons about its own protocol and quotes the marker;
  closing there cuts the channel and republishes thinking as answer text — the observed live defect
  (PR #309 problem statement).*
- **R3 — quoted marker.** A `</think>` followed by any other byte stays in reasoning, and is
  published together with the bytes that follow. *PR #309; reason as R2.*
- **R4 — hold.** A proper suffix of `</think>` and a complete `</think>` that ends the available
  bytes are held, not published and not acted on, until the next byte arrives. *PR #309, P1-run
  llama.cpp and vLLM lexer buffering agree that markers must never be split across rounds (row 6).*
- **R5 — implicit close at terminal.** At end of turn a still-held close marker is the model's
  implicit close: the preceding bytes are published as reasoning, the marker is consumed and not
  published, nothing is published as content. *PR #309, P2 vLLM (`finish()` emits `REASONING_END`),
  P1-run llama.cpp (row 3).*
- **R6 — byte exactness in content.** A `</think>` or `<think>` that reaches the content channel is
  ordinary content, published once, never dropped and never re-routed. *Rejects vLLM's silent
  absorption of duplicate closes (row 10); follows NInfer master and llama.cpp. Reason: the content
  channel is answer text; silently deleting model bytes corrupts code blocks and quoted protocol
  text.*
- **R7 — thinking off.** The channel starts in content; a `<think>…</think>` block in the output
  stays in content. *All references (P1-run llama.cpp; P2 vLLM).*
- **R8 — control follows the close.** The thinking-budget tracker, stop policy and the
  `finish_reason` decision observe the same close rule (R2-R4); a quoted close must not end a
  thinking budget round. *PR #309 (`test_thinking_budget_ignores_quoted_close`).*

**Tool-call channel.**

- **B1 — marker and hold.** The first `<tool_call>` in the content channel starts the structured
  region; everything before it is content (trailing format whitespace trimmed). A `<tool_call>` seen
  while the reasoning channel is still open closes it first (the implicit close of row 2 — the ported
  grammar has it, current master does not). From the marker on, bytes are held until the region
  resolves. *vLLM and llama.cpp for the implicit close; NInfer master's `ToolCallOutputDecoder` for
  the hold; PR #309 keeps the held prefix flushable.*
- **B2 — candidate search (the PR #309 rule).** If the bytes from a `<tool_call>` candidate do not
  parse as a complete region that consumes the response to its end (only format whitespace after the
  last call), the next `<tool_call>` candidate is tried. Earlier candidates and the prose between
  them are content. If no candidate qualifies, the whole response is content (all-or-nothing) with
  diagnostics. *Rejects llama.cpp's "drop tail" and vLLM's silent swallow (rows 23, 26); reason:
  the observed live defect is a real call demoted after a quoted malformed marker, and the model's
  prose must survive either way.*
- **B3 — region grammar.** `<tool_call>` + format whitespace* + `<function=NAME>` + parameters +
  `</function>` + format whitespace* + `</tool_call>` (format whitespace* between them). NAME must
  be non-empty, ≤ the name limit, and match `[A-Za-z0-9_-]+`. Several calls may follow one another
  in one response; a missing `</tool_call>` before the next `<tool_call>` is tolerated (vLLM, row
  15). *Master's grammar plus vLLM's tolerance; llama.cpp demands the close and a literal name.*
- **B4 — parameters.** `<parameter=NAME>` + value + `</parameter>`; one leading and one trailing
  newline are removed from the value (the template wraps every value), then the value is normalized
  (B6). A balanced nested `<parameter=…>…</parameter>` sequence inside a value is part of that
  value (row 21). An unmatched nested opener or a standalone `</parameter>` makes the region
  unrepresentable: the region becomes content (B2 tries the next candidate). Duplicate parameter
  names make the region invalid. *Master's documented rule in `docs/serving.md` and PR #309's
  scoping of it; rejects vLLM's re-splitting and llama.cpp's exact-`\n</parameter>\n`-only rule
  (row 21).*
- **B5 — names.** A tool name that is not declared in the request is a serving-policy fallback: the
  region falls back to content with an `UndeclaredTool` diagnostic (master's `enforce_declared_names`).
  *Chosen over vLLM's accept-anything (row 28): the request's tool list is the contract the client
  dispatches on; the failure must be visible as text plus a diagnostic, not as an unservable call.*
- **B6 — value normalization.** With a declared schema: a schema admitting `string` produces a JSON
  string (whatever the text looks like); otherwise the framed text is emitted as raw JSON when it
  parses as JSON, text `true`/`false` becomes a boolean when the schema admits boolean, and anything
  else becomes a JSON string with a `SchemaMismatch` diagnostic. Without a schema: master's
  `normalize_parameter` (raw JSON when it parses, else a JSON string). *Master's rule; vLLM without a
  schema keeps every value a string (row 18/19), llama.cpp types by JSON syntax — master's rule is
  the only one that is exact for a declared `string` argument and still types declared
  `integer`/`boolean` arguments.*
- **B7 — publication.** Tool calls are published only at terminal, verbatim (name + argument JSON
  text, in order), all-or-nothing per accepted region. Partial regions never publish a call.
  *Master's contract; P1-run llama.cpp publishes calls only when complete; vLLM streams argument
  fragments (rejected: the frontend seam publishes ≤2 append-only channel deltas per round and
  terminal-only tool calls, ticket #31).*

## 6. Oracle corpus

Format. One JSON object per vector. `tools` refers to the tool list in §6.1; `thinking` is
`enable_thinking`; `rounds[].feed` is the exact decoded output text the model generated in that
round (the generation prompt is not part of `feed`). `reasoning` and `content` in a round are the
**cumulative** published bytes of that channel after the round (append-only; `null`/absent = nothing
published). `held` is the byte suffix of the round's feed that is not published yet, given only where
the decoder must withhold bytes. The **last round of every vector is the end of the turn** (terminal):
held bytes resolve there, tool calls are published there, and `final` repeats the end state.
`final.tool_calls` are the published calls. `basis` lists the implementations whose pinned behavior
matches the vector; `diverges` names the ones that do not, with the row number in §4.

### 6.1 Tools used by the vectors

```json
{
  "tools": [
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_count", "parameters": {"type": "object", "properties": {"count": {"type": "integer"}, "label": {"type": "string"}}, "required": ["count"]}}},
    {"type": "function", "function": {"name": "toggle", "parameters": {"type": "object", "properties": {"enabled": {"type": "boolean"}}, "required": ["enabled"]}}}
  ]
}
```

### 6.2 Reasoning-channel vectors

```json
[
  {
    "id": "reasoning-plain-close-then-content",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "I need to think.\n", "reasoning": "I need to think.\n", "content": ""},
      {"feed": "</think>", "reasoning": "I need to think.\n", "content": "", "held": "</think>"},
      {"feed": "\n\nThe answer is 42.", "reasoning": "I need to think.\n", "content": "The answer is 42."}
    ],
    "final": {"reasoning": "I need to think.\n", "content": "The answer is 42.", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master", "pr309"],
    "note": "Whitespace immediately after the close is stripped, not published. llama.cpp also eats the byte after the marker via its PEG space(); NInfer publishes content after stripping leading format whitespace."
  },
  {
    "id": "reasoning-close-implicit-at-tool-call",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "I need to check.\n\n", "reasoning": "I need to check.\n\n", "content": ""},
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}]}
    ],
    "final": {"reasoning": "I need to check.\n\n", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}]},
    "basis": ["vllm", "llamacpp"],
    "diverges": ["ninfer_master(row 2): no implicit close at the tool marker while reasoning is open — the marker and the call stay in reasoning and no call is published; the port takes llama.cpp's `peek(<tool_call>)` rule"],
    "note": "Thinking on and the model jumps straight to the call. The port must close reasoning at the marker and route the marker into the tool region (B1)."
  },
  {
    "id": "reasoning-close-implicit-at-turn-end",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "done thinking</think>", "reasoning": "done thinking", "content": "", "held": "</think>", "terminal": true}
    ],
    "final": {"reasoning": "done thinking", "content": "", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master", "pr309"],
    "note": "The marker itself is consumed as the implicit close and is not published."
  },
  {
    "id": "reasoning-unclosed-at-turn-end",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "thinking without any close", "reasoning": "thinking without any close", "content": "", "terminal": true}
    ],
    "final": {"reasoning": "thinking without any close", "content": "", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master"]
  },
  {
    "id": "reasoning-quoted-close-followed-by-quote",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "I compared strings like ", "reasoning": "I compared strings like ", "content": ""},
      {"feed": "</think>", "reasoning": "I compared strings like ", "content": "", "held": "</think>"},
      {"feed": "'; then hidden\n", "reasoning": "I compared strings like </think>'; then hidden\n", "content": ""},
      {"feed": "</think>\n\nreal answer", "reasoning": "I compared strings like </think>'; then hidden\n", "content": "real answer"}
    ],
    "final": {"reasoning": "I compared strings like </think>'; then hidden\n", "content": "real answer", "tool_calls": []},
    "basis": ["pr309"],
    "diverges": ["vllm(row 5): closes at the first marker, tail becomes content, second marker dropped", "llamacpp(row 5): closes at the first marker, tail becomes content with the second marker kept"],
    "note": "PR #309 test_reasoning_close_requires_boundary, compressed to four rounds."
  },
  {
    "id": "reasoning-close-marker-split-across-rounds",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "done </thi", "reasoning": "done ", "content": "", "held": "</thi"},
      {"feed": "nk>\n\nAnswer.", "reasoning": "done ", "content": "Answer."}
    ],
    "final": {"reasoning": "done ", "content": "Answer.", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master", "pr309"],
    "note": "The canonical close split across rounds: nothing is published until the marker is unambiguous, then the marker closes and the newline after it is stripped."
  },
  {
    "id": "reasoning-quoted-close-split-across-rounds",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "check </thi", "reasoning": "check ", "content": "", "held": "</thi"},
      {"feed": "nk>'", "reasoning": "check </think>'", "content": ""},
      {"feed": " then the close\n</think>\n\nanswer", "reasoning": "check </think>' then the close\n", "content": "answer"}
    ],
    "final": {"reasoning": "check </think>' then the close\n", "content": "answer", "tool_calls": []},
    "basis": ["pr309"],
    "diverges": ["vllm(row 5): closes at the first marker", "llamacpp(row 5): closes at the first marker"],
    "note": "After the held marker is completed, the following byte (a quote) decides: it is not a close, so the marker and the byte are published into reasoning. The third round carries the real close."
  },
  {
    "id": "reasoning-quoted-close-followed-by-space",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "see the tag </think> in the docs\n", "reasoning": "see the tag ", "content": "in the docs\n"},
      {"feed": "still reasoning?\n", "content": "in the docs\nstill reasoning?\n"}
    ],
    "final": {"reasoning": "see the tag ", "content": "in the docs\nstill reasoning?\n", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master", "pr309"],
    "note": "Inherent limit documented by PR #309: a quoted marker followed by real whitespace is locally indistinguishable from a close."
  },
  {
    "id": "reasoning-repeated-think-open-in-output",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<think>\nold template repeats start</think>", "reasoning": "\nold template repeats start", "content": ""}
    ],
    "final": {"reasoning": "\nold template repeats start", "content": "", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master"],
    "note": "The start marker is consumed; nothing is published for it."
  },
  {
    "id": "reasoning-thinking-disabled-plain",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "The answer is 42.", "reasoning": "", "content": "The answer is 42."}
    ],
    "final": {"reasoning": "", "content": "The answer is 42.", "tool_calls": []},
    "basis": ["vllm", "llamacpp", "ninfer_master"]
  },
  {
    "id": "reasoning-thinking-disabled-close-in-content",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "Use </think> to close the block.", "reasoning": "", "content": "Use </think> to close the block."}
    ],
    "final": {"reasoning": "", "content": "Use </think> to close the block.", "tool_calls": []},
    "basis": ["llamacpp", "ninfer_master"],
    "diverges": ["vllm(row 10): drops the duplicate close token from content"],
    "note": "Byte exactness wins: the content channel must not delete model bytes."
  },
  {
    "id": "reasoning-close-inside-tool-arg-value",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\necho </think> ok\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "echo </think> ok"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "echo </think> ok"}}]},
    "basis": ["vllm", "llamacpp", "ninfer_master"]
  }
]
```

### 6.3 Tool-call channel vectors

```json
[
  {
    "id": "tool-call-canonical-single",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}]},
    "basis": ["vllm", "llamacpp", "ninfer_master"]
  },
  {
    "id": "tool-call-prose-before-call",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "Let me list the files.\n\n", "content": "Let me list the files.\n\n"},
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>", "content": "Let me list the files.\n\n", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}], "held": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"}
    ],
    "final": {"reasoning": "", "content": "Let me list the files.\n\n", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
    "basis": ["vllm", "llamacpp", "ninfer_master"],
    "note": "Bytes from the marker on are held; at terminal the accepted region is consumed and the earlier content is published once."
  },
  {
    "id": "tool-call-multiline-value-and-framing-newlines",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nprintf 'a\nb'\n</parameter>\n<parameter=description>\nlist files\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "printf 'a\nb'", "description": "list files"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "printf 'a\nb'", "description": "list files"}}]},
    "basis": ["vllm", "llamacpp", "ninfer_master"],
    "note": "One leading and one trailing newline of each value are the template's framing and are removed; internal newlines survive."
  },
  {
    "id": "tool-call-two-complete-calls",
    "thinking": false,
    "tools": ["bash", "get_time"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}, {"name": "get_time", "arguments": {}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}, {"name": "get_time", "arguments": {}}]},
    "basis": ["vllm", "ninfer_master", "llamacpp(row 14) only with parallel_tool_calls=true"],
    "note": "Declared by the froggeric instructions (separate closed blocks); NInfer always accepts multiple calls in one turn."
  },
  {
    "id": "tool-call-consecutive-without-tool-close",
    "thinking": false,
    "tools": ["bash", "get_time"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n<tool_call>\n<function=get_time>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}, {"name": "get_time", "arguments": {}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}, {"name": "get_time", "arguments": {}}]},
    "basis": ["vllm(row 15)"],
    "diverges": ["llamacpp(row 15): requires </tool_call>, drops the tail", "ninfer_master(row 15): TrailingContent fallback"],
    "note": "Tolerance chosen with vLLM: a missing close between two calls loses the whole turn otherwise. Implementations may keep master's stricter rule if the port's grammar makes tolerance expensive; then this vector moves to the rejected set."
  },
  {
    "id": "tool-call-empty-arguments",
    "thinking": false,
    "tools": ["get_time"],
    "rounds": [
      {"feed": "<tool_call>\n<function=get_time>\n</function>\n</tool_call>", "tool_calls": [{"name": "get_time", "arguments": {}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "get_time", "arguments": {}}]},
    "basis": ["vllm", "llamacpp", "ninfer_master"]
  },
  {
    "id": "tool-call-arg-order-any",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=description>\nlist files\n</parameter>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"description": "list files", "command": "ls"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"description": "list files", "command": "ls"}}]},
    "basis": ["vllm", "ninfer_master"],
    "diverges": ["llamacpp(row 17): optional args must follow all required args"],
    "note": "Argument order in the emitted JSON follows the model's order."
  },
  {
    "id": "tool-call-value-json-looking-string",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\n{\"nested\": 1}\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "{\"nested\": 1}"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "{\"nested\": 1}"}}]},
    "basis": ["vllm(row 18, no schema)", "llamacpp(row 18, string-typed)", "ninfer_master(row 18, declared string)"],
    "note": "A declared string parameter keeps the text as a JSON string even when it looks like JSON."
  },
  {
    "id": "tool-call-typed-values",
    "thinking": false,
    "tools": ["set_count", "toggle"],
    "rounds": [
      {"feed": "<tool_call>\n<function=set_count>\n<parameter=count>\n42\n</parameter>\n</function>\n</tool_call>\n", "content": "", "held": "<tool_call>\n<function=set_count>\n<parameter=count>\n42\n</parameter>\n</function>\n</tool_call>\n"},
      {"feed": "<tool_call>\n<function=toggle>\n<parameter=enabled>\ntrue\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "set_count", "arguments": {"count": 42}}, {"name": "toggle", "arguments": {"enabled": true}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "set_count", "arguments": {"count": 42}}, {"name": "toggle", "arguments": {"enabled": true}}]},
    "basis": ["llamacpp", "ninfer_master", "vllm(with tool schemas)"],
    "diverges": ["vllm(row 19, without schemas): every value stays a string"],
    "note": "Both calls in one response tests the region sequence as well. Calls are published at terminal only, so the first round publishes nothing."
  },
  {
    "id": "tool-call-required-arg-missing",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=description>\nlist files\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"description": "list files"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"description": "list files"}}]},
    "basis": ["vllm", "ninfer_master"],
    "diverges": ["llamacpp(row 16b): required arguments are part of the grammar, the whole turn is dropped"],
    "note": "Fail-open: the call is dispatchable and the client validates its arguments; dropping the turn loses the model's intent. Required-argument validation is a serving-layer concern, not a representability one."
  },
  {
    "id": "tool-call-duplicate-parameter",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=command>\nrm -rf /\n</parameter>\n</function>\n</tool_call>", "content": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=command>\nrm -rf /\n</parameter>\n</function>\n</tool_call>"}
    ],
    "final": {"reasoning": "", "content": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=command>\nrm -rf /\n</parameter>\n</function>\n</tool_call>", "tool_calls": [], "diagnostics": {"fallback_reason": "DuplicateParameter"}},
    "basis": ["ninfer_master"],
    "diverges": ["vllm(row 20): last value wins", "llamacpp(row 20): a repeated required argument drops the turn; a repeated optional argument is accepted and emits duplicate JSON keys"],
    "note": "Rejecting the region is the fail-closed choice for a conflicting argument."
  },
  {
    "id": "tool-call-nested-parameter-balanced",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nprintf '<parameter=x>y</parameter>'\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "printf '<parameter=x>y</parameter>'"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "printf '<parameter=x>y</parameter>'"}}]},
    "basis": ["ninfer_master(docs/serving.md)", "llamacpp(when the nested close is not newline-framed)"],
    "diverges": ["vllm(row 21): the nested opener is treated as an implicit end of the previous value"],
    "note": "Balanced nested markup stays inside the value. The region is only unrepresentable when the nesting does not balance."
  },
  {
    "id": "tool-call-nested-parameter-newline-framed",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nprintf '<parameter=x>\ny\n</parameter>\n'\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "printf '<parameter=x>\ny\n</parameter>'"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "printf '<parameter=x>\ny\n</parameter>'"}}]},
    "basis": ["ninfer_master(docs/serving.md)"],
    "diverges": ["llamacpp(row 21): the newline-framed nested close ends the value and the turn fails", "vllm(row 21): re-split"],
    "note": "The nested close is written with the wire framing (newline on both sides); balance tracking is what keeps it inside the value."
  },
  {
    "id": "tool-call-unmatched-nested-parameter",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nprintf 'x<parameter=y'\n</parameter>\n</function>\n</tool_call>", "content": "<tool_call>\n<function=bash>\n<parameter=command>\nprintf 'x<parameter=y'\n</parameter>\n</function>\n</tool_call>"}
    ],
    "final": {"reasoning": "", "content": "<tool_call>\n<function=bash>\n<parameter=command>\nprintf 'x<parameter=y'\n</parameter>\n</function>\n</tool_call>", "tool_calls": [], "diagnostics": {"fallback_reason": "MalformedStructure"}},
    "basis": ["ninfer_master(docs/serving.md)", "pr309(scoping)"],
    "diverges": ["vllm(row 22): value bytes may absorb the text", "llamacpp(row 22): drops the tail"]
  },
  {
    "id": "tool-call-standalone-parameter-close",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\necho </parameter> extra\n</parameter>\n</function>\n</tool_call>", "content": "<tool_call>\n<function=bash>\n<parameter=command>\necho </parameter> extra\n</parameter>\n</function>\n</tool_call>"}
    ],
    "final": {"reasoning": "", "content": "<tool_call>\n<function=bash>\n<parameter=command>\necho </parameter> extra\n</parameter>\n</function>\n</tool_call>", "tool_calls": [], "diagnostics": {"fallback_reason": "MalformedStructure"}},
    "basis": ["ninfer_master(docs/serving.md)", "pr309(scoping)"],
    "note": "A standalone </parameter> unbalances the value; per PR #309 this only demotes the region, later candidates are still examined."
  },
  {
    "id": "tool-call-quoted-marker-before-real-call",
    "thinking": false,
    "tools": ["bash"],
    "note": "PR #309 test_quoted_marker_before_real_call. The quoted region uses the observed live shape: the model quotes the markup with escaped newlines (literal backslash-n), and it also mistypes <function=command> where <parameter=command> belongs. Vector tool-call-quoted-marker-real-newlines is the same case with real newlines.",
    "rounds": [
      {"feed": "The failure looked like <tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n</function>\\n</tool_call>\nThen the real call:\n<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>", "content": "The failure looked like <tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n</function>\\n</tool_call>\nThen the real call:", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}]}
    ],
    "final": {"reasoning": "", "content": "The failure looked like <tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n</function>\\n</tool_call>\nThen the real call:", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}], "diagnostics": {"structured_call_count": 1, "marker_seen": true}},
    "basis": ["pr309"],
    "diverges": ["vllm(row 23): swallows the rest of the turn after the first marker", "llamacpp(row 23): drops the rest of the turn"],
    "note": "The quoted candidate is rejected on structure; the search then finds the real call. The first candidate's failure reason is not pinned (MalformedStructure when the structural check runs first, UndeclaredTool when the declared-name policy runs first — the order is an implementation decision for #34)."
  },
  {
    "id": "tool-call-quoted-marker-real-newlines",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "The failure looked like <tool_call>\n<function=shell>\n<function=command>\nbroken\n</parameter>\n</function>\n</tool_call>\nThen the real call:\n<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>", "content": "The failure looked like <tool_call>\n<function=shell>\n<function=command>\nbroken\n</parameter>\n</function>\n</tool_call>\nThen the real call:", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}]}
    ],
    "final": {"reasoning": "", "content": "The failure looked like <tool_call>\n<function=shell>\n<function=command>\nbroken\n</parameter>\n</function>\n</tool_call>\nThen the real call:", "tool_calls": [{"name": "bash", "arguments": {"command": "echo ok"}}], "diagnostics": {"structured_call_count": 1, "marker_seen": true}},
    "basis": ["pr309"],
    "diverges": ["vllm(row 23): swallows the rest of the turn after the first marker", "llamacpp(row 23): drops the rest of the turn"]
  },
  {
    "id": "tool-call-later-candidate-must-consume-the-end",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n</function>\\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>\nstill explaining", "content": "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n</function>\\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>\nstill explaining"}
    ],
    "final": {"reasoning": "", "content": "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n</function>\\n</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>\necho ok\n</parameter>\n</function>\n</tool_call>\nstill explaining", "tool_calls": [], "diagnostics": {"fallback_reason": "MalformedStructure", "marker_seen": true}},
    "basis": ["pr309"],
    "diverges": ["vllm(row 24): swallow", "llamacpp(row 24): drop tail"],
    "note": "PR #309 test_later_candidate_must_consume_the_end. A well-formed call followed by prose is not a structured turn: the response stays content (all-or-nothing). PR #309 asserts MalformedStructure here; see §7 item 6 on the recorded first-failure reason."
  },
  {
    "id": "tool-call-marker-split-across-rounds",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "calling now ", "content": "calling now "},
      {"feed": "<tool_c", "content": "calling now ", "held": "<tool_c"},
      {"feed": "all>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>", "content": "calling now ", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]}
    ],
    "final": {"reasoning": "", "content": "calling now ", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
    "basis": ["vllm", "llamacpp", "ninfer_master"],
    "note": "The partial marker must never be published as content; trailing whitespace before the marker is withheld with it and re-emitted only if the region is demoted."
  },
  {
    "id": "tool-call-marker-quoted-without-call",
    "thinking": true,
    "tools": ["bash"],
    "rounds": [
      {"feed": "The user said ", "reasoning": "The user said ", "content": ""},
      {"feed": "<tool_call>", "reasoning": "The user said ", "content": "", "held": "<tool_call>"},
      {"feed": " is the marker.\nI will not call it.\n</think>\n\nAnswer.", "reasoning": "The user said ", "content": "<tool_call> is the marker.\nI will not call it.\n</think>\n\nAnswer."}
    ],
    "final": {"reasoning": "The user said ", "content": "<tool_call> is the marker.\nI will not call it.\n</think>\n\nAnswer.", "tool_calls": [], "diagnostics": {"marker_seen": true, "fallback_reason": "MalformedStructure"}},
    "basis": ["pr309(rule extrapolated to a reasoning-open input)"],
    "diverges": ["vllm(row 26): drops the rest of the turn", "llamacpp(row 26): drops the rest of the turn", "ninfer_master(row 2 and 26): no implicit close while reasoning is open, so the marker and tail stay reasoning and the content channel never starts"],
    "note": "The <tool_call> inside reasoning is the implicit reasoning end (row 2), then the candidate region fails to parse, so at terminal the whole region (marker included) is flushed as content: no byte is lost. The </think> inside the flushed region is ordinary content (R6) — the reasoning channel already closed at the tool marker."
  },
  {
    "id": "tool-call-undeclared-parameter",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n<parameter=bogus>\nx\n</parameter>\n</function>\n</tool_call>", "tool_calls": [{"name": "bash", "arguments": {"command": "ls", "bogus": "x"}}]}
    ],
    "final": {"reasoning": "", "content": "", "tool_calls": [{"name": "bash", "arguments": {"command": "ls", "bogus": "x"}}], "diagnostics": {"schema_mismatch_arguments": 1}},
    "basis": ["vllm", "ninfer_master"],
    "diverges": ["llamacpp(row 27): drops the tail"],
    "note": "The call is served; the undeclared argument is recorded as a schema-mismatch diagnostic."
  },
  {
    "id": "tool-call-undeclared-tool-name",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=nonexistent_tool>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", "content": "<tool_call>\n<function=nonexistent_tool>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"}
    ],
    "final": {"reasoning": "", "content": "<tool_call>\n<function=nonexistent_tool>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", "tool_calls": [], "diagnostics": {"fallback_reason": "UndeclaredTool"}},
    "basis": ["ninfer_master"],
    "diverges": ["vllm(row 28): accepts any name", "llamacpp(row 28): drops the tail"],
    "note": "Serving policy: names must come from the request's tool list; the failure is visible as content plus a diagnostic."
  },
  {
    "id": "tool-call-name-invalid-characters",
    "thinking": false,
    "tools": ["bash"],
    "rounds": [
      {"feed": "<tool_call>\n<function=ba sh>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", "content": "<tool_call>\n<function=ba sh>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"}
    ],
    "final": {"reasoning": "", "content": "<tool_call>\n<function=ba sh>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>", "tool_calls": [], "diagnostics": {"fallback_reason": "InvalidToolName"}},
    "basis": ["ninfer_master"],
    "diverges": ["llamacpp: no match for the literal name", "vllm: accepts any text before '>'"]
  },
  {
    "id": "tool-call-thinking-budget-ignores-quoted-close",
    "thinking": true,
    "tools": [],
    "note": "Frontend control vector, not a parser vector: PR #309 test_thinking_budget_ignores_quoted_close.",
    "rounds": [
      {"feed": "x</think>x", "reasoning": "x</think>x", "content": "", "control": "thinking budget still applies (continuation=ApplyTargetControl)"}
    ],
    "final": {"reasoning": "x</think>x", "content": "", "tool_calls": []},
    "basis": ["pr309"],
    "diverges": ["ninfer_master(pre-pr309): the marker closes reasoning and suppresses the budget"]
  }
]
```

### 6.4 Cross-reference of vectors to evidence

- Matrix rows 1-4, 7-9, 11 (agreement) — P1-run llama.cpp `scratch-oracle` (§8.2) and P1-run vLLM
  driver (§8.1) agree; P2 vLLM tests `test_reasoning_then_content`, `test_no_start_token_in_output`,
  `test_tool_call_implicit_end`, `test_no_end_tag_all_reasoning`.
- Vectors `reasoning-quoted-close-*` and `tool-call-quoted-marker-before-real-call`,
  `tool-call-later-candidate-must-consume-the-end`, `tool-call-undeclared-*`,
  `tool-call-standalone-parameter-close`, `tool-call-unmatched-nested-parameter` — PR #309 diff
  (`tests/models/qwen3_5/test_frontend.cpp`, `tests/test_tool_call_parser.cpp`).
- Tool-call rows 12-16, 18-19, 25 — P1-run llama.cpp and P1-run vLLM; P2 vLLM
  `test_streaming_multiline_param_values`, `test_streaming_split_next_parameter_tag_is_buffered`,
  `test_consecutive_tool_calls_without_tool_end`, `test_empty_arguments`.
- `tool-call-nested-parameter-balanced`, `tool-call-nested-parameter-newline-framed`,
  `tool-call-required-arg-missing`, `tool-call-duplicate-parameter`,
  `tool-call-undeclared-tool-name`, `tool-call-name-invalid-characters` — NInfer `docs/serving.md`
  and `src/models/qwen3_5/frontend/tool_call_parser.cpp` at `c6a3b1e1`.
- `reasoning-close-marker-split-across-rounds`, `tool-call-marker-split-across-rounds` — P1-run
  vLLM (prefix-match buffering) and P1-run llama.cpp (round-split cases).

## 7. Open items for the design tickets

1. **Where the recovery lives.** llama.cpp's generated PEG has no end-of-input requirement and the
   root can succeed before EOF (`common/chat.cpp` 1447-1499), which is why every malformed case in
   §3.2 silently drops the tail. The port must add either a candidate-search layer over the PEG
   (PR #309's loop, which needs the region grammar to expose "parsed to the end") or a post-parse
   recovery over the raw text. Both reasoning rules (R2-R5) live *outside* the PEG: the boundary and
   hold rules need bytes the PEG does not track. Design ticket #28 should name the seam; the corpus
   vectors are the acceptance test either way.
2. **Strictness tokens of the region grammar.** B3/B4 choose vLLM-style tolerance (missing
   `</tool_call>`, argument order, undeclared parameters) and master-style rejection (duplicates,
   unbalanced markup, undeclared tool names). Each is a token in the ported grammar or a check in
   the recovery layer; the corpus pins the outcome, not the mechanism.
3. **Streaming publication.** llama.cpp publishes a tool call only when it is complete; vLLM streams
   argument fragments. NInfer's seam contract (≤2 append-only channel deltas per round, terminal-only
   tool calls, ticket #31) matches llama.cpp; PR #309's decoder holds from the first marker and
   flushes `visible + terminal.content` byte-exactly. Keep that.
4. **Diagnostics vocabulary.** The `diagnostics` fields in the corpus (`fallback_reason`,
   `marker_seen`, `structured_call_count`, `schema_mismatch_arguments`, `empty_arguments_omitted`)
   mirror master's `ToolCallParseDiagnostics`. Map them onto whatever the port produces; do not
   freeze the names in the port's public API.
5. **R6 is a behavior change against vLLM, not against master.** vLLM drops a duplicate close token
   from content; master and llama.cpp keep it. The corpus chooses byte exactness. If the served
   protocol must match vLLM's visible content for some client, this is the one row to revisit.
6. **Name-check order.** PR #309's `test_later_candidate_must_consume_the_end` asserts
   `MalformedStructure` while the code records the first candidate's failure, which is
   `UndeclaredTool` when the candidate's tool name is not declared and the declared-name policy runs
   before the parameter loop (verified at the PR's base commit). Pin the order in the unit tests.
7. **Wrapper-less calls.** vLLM accepts `<function=…>` without `<tool_call>`; llama.cpp accepts it
   only for Qwen3-Coder (no-reasoning) templates, not for froggeric (verified: it becomes content).
   The corpus does not include that vector; if the port wants vLLM's tolerance it must be added
   deliberately.
8. **Schema-less type policy.** With no tool contract, vLLM keeps all values as strings; master
   emits raw JSON when the text parses. The corpus pins master's rule (B6).
9. **Cost.** The candidate search re-parses on failure; the hold rules delay one byte at a round
   boundary. Neither is measured yet; #29 should ask for a decode-path profile only if the port is
   on the hot path of every token.

## 8. How the evidence was produced

### 8.1 vLLM (real engine, standalone)

The parser engine's core modules are pure Python apart from package imports
(`incremental_lexer.py`, `parser_engine_config.py`, `token_id_scanner.py`,
`streaming_parser_engine.py`, `events.py`). They were copied verbatim from the pinned checkout into
a scratch package together with the unmodified `vllm/parser/qwen3.py`, with a stub
`vllm/parser/engine/parser_engine.py` providing the `ParserEngine` base class that `qwen3.py` imports
(the engine itself is not needed to run `qwen3_config`). Driving it:

```python
cfg = qwen3_config(thinking=True, turn_boundary_tokens=frozenset(("<|im_start|>", "<|im_end|>")))
engine = StreamingParserEngine(cfg, mock_tokenizer)   # mock get_vocab()/decode() over the token ids
for chunk in chunks:
    events = engine.feed(chunk, token_ids_for_chunk)  # text-only chunks: []
events = engine.finish()
```

`EventType` values per round are the channel routing used in §3.1. The unmodified test suite
(`tests/parser/engine/test_qwen3.py`, `test_qwen3_reasoning.py`) supplies the exact client-visible
fragment assertions.

### 8.2 llama.cpp (real parser, scratch harness)

Built in the flake devShell (`nix develop -c cmake --build build --target scratch-oracle`), with a
scratch `tests/scratch-oracle.cpp` that links `chat.h`/`chat-auto-parser.h` and mirrors the server's
parser construction:

```cpp
auto tmpl = common_chat_templates_init(nullptr, template_source, "", "");
// inputs: messages={user}, tools=..., add_generation_prompt=true, use_jinja=true,
//         enable_thinking=..., reasoning_format=AUTO, parallel_tool_calls=...
auto params = common_chat_templates_apply(tmpl.get(), inputs);
common_peg_arena arena; arena.load(params.parser);
common_chat_parser_params pp(params);
// per round: msg = common_chat_peg_parse(arena, accumulated_output, is_partial, pp);
// streaming deltas: common_chat_msg_diff::compute_diffs(previous_msg, msg)
```

The parser selection is the specialized path (`format=peg-native`, generation prompt
`<|im_start|>assistant\n<think>\n` for the froggeric template with thinking enabled). The harness was
scratch scaffolding outside the repository and is not part of the deliverable; the vectors in §6 are
the durable artifact.

### 8.3 froggeric

Byte-exact download from `https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/`
(`chat_template.jinja`, `README.md`); revision from
`git ls-remote https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates refs/heads/main`
(`855bffc49448e299789730ff92c9b8d834d6cc14`); digests with `sha256sum`.

## 9. Sources

- llama.cpp `95887577ab5fead779581a7030a83c7752ff3234`: `common/parsers/qwen3-coder.cpp`,
  `common/chat-peg-parser.{h,cpp}`, `common/peg-parser.{h,cpp}`, `common/chat.{h,cpp}`,
  `common/chat-auto-parser*.{h,cpp}`, `common/json-schema*`, `tools/server/server-task.cpp`,
  `tools/server/server-chat.cpp`, `common/arg.cpp`, `docs/autoparser.md`, `tests/test-chat*.cpp`.
- vLLM `77871126f9b69cff9feffff390d1171a60dbd7e2`: `vllm/parser/qwen3.py`,
  `vllm/parser/engine/{parser_engine,streaming_parser_engine,incremental_lexer,token_id_scanner,parser_engine_config,events,registered_adapters}.py`,
  `vllm/tool_parsers/{__init__.py,qwen3_engine_tool_parser.py}`,
  `vllm/reasoning/{__init__.py,qwen3_engine_reasoning_parser.py}`,
  `tests/parser/engine/test_qwen3.py`, `tests/parser/engine/test_qwen3_reasoning.py`,
  `docs/features/tool_calling.md`.
- froggeric `Qwen-Fixed-Chat-Templates` at `main` `855bffc4…`: `chat_template.jinja` (sha256
  `e57684ba…`), `README.md` (sha256 `b746a44e…`).
- NInfer fork `c6a3b1e1`: `src/models/qwen3_5/frontend/tool_call_parser.{h,cpp}`,
  `src/models/qwen3_5/frontend/output_session.cpp`, `docs/serving.md`,
  `tests/test_tool_call_parser.cpp`, `tests/models/qwen3_5/test_frontend.cpp`.
- PR Neroued/ninfer#309 (closed unmerged, head `71304581…`, base `594930e7…`): full diff and
  description (read-only).
- Map #25 and ticket #26/#31 resolutions in `kido5217/ninfer-yarn` for the standing decisions this
  document builds on.
