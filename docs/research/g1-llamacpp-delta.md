# G1: the llama.cpp port surface and the baseline delta

Wayfinder map [#145](https://github.com/kido5217/ninfer-yarn/issues/145) (G1 — tool-call demotion
→ silent stall), ticket #147. Question: what parse/recovery machinery does llama.cpp have that
NInfer's pinned vendor baseline lacks **and** NInfer's bespoke recovery does not already provide —
the robustness-half port surface, ranked, with a vehicle per mechanism. The ranking constraint is
the user steer (2026-10-03): llama.cpp-derived machinery is preferred over bespoke where feasible.

Sources read this session (2026-10-03). Upstream is a blobless clone of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) at
`bed0a856606ee4a24a164066f73d2379447033f5` (master, the ticket's comparison point; `/tmp/opencode/llamacpp-chat-ref`,
sparse `common/` + `tools/server/`). NInfer is origin/master `8a0d16bf` plus the worktree of this
branch. Prior artifacts referenced: `docs/research/g1-demotion-pipeline.md` @
`research/g1-demotion-pipeline` `606b91aa` (#146), `docs/research/reference-baselines.md` @
`research/reference-baselines` `7c976d17` (§2.2, design only), `docs/research/chat-parsing-oracle.md`
@ `research/chat-parsing-oracle` `0e115f4f` (the port's agreement matrix, rows cited below), and the
committed corpus `tests/fixtures/chat_parsing/corpus.json` (42 vectors, the semantic authority).
No builds or runs were performed (research ticket); every behavioral claim below is a source read,
and §6 lists what that leaves unverified.

## 1. Summary

Measured first, because it kills one premise: the pinned baseline `05af0d2b` (2026-09-30) to master
`bed0a85660` (2026-10-02) is **68 commits, and exactly one touches the vendored set** —
`b8f96c3e8` adds the LLM-jp-4.1 Harmony dialect registration (`common/chat.cpp` +6 lines,
`common/parsers/parsers.h` +2, a new non-vendored parser file). `parse_anywhere_and_extract` and the
"#9639 machinery" are **already in the pinned baseline** and are not a delta (§2). A baseline
advance buys nothing for G1.

Ranked port surface (details in §3):

1. **M1 — lenient truncation typing + partial-AST salvage** (upstream's streaming parse contract):
   distinguish "bytes ran out" (`need_more_input`) from "definitely wrong" (`fail`), and map the AST
   captured before a failing tail so a completed call is not thrown away with it. Covers the
   prose-after-call shape, the truncated-region shape, and gives the signaling half its
   bound-cut-vs-malformed discrimination. **Vehicle: port into `chat_parse_core`, reusing the
   already-vendored engine semantics — zero `third_party/` byte changes, zero compat impact.**
2. **M2 — delimiter-terminated argument values** (`until("\n</parameter>\n")`): llama.cpp's string
   value syntax admits marker fragments inside a value as text where NInfer's B4 balanced-nesting
   rule demotes. This is the lever for the live family (b) call losses, but it contradicts
   corpus-pinned rows 21/22 → a design decision with a vector rewrite, not a silent port.
   **Vehicle: semantics port into the region grammar, only if #148 adopts it.**
3. **M3 — `tag_with_safe_content`**: the content-marker-safe chunking pattern. Shipped upstream as
   an **unused helper** (no callers in `common/` or `tools/server/`); a pattern with a llama.cpp
   address, not a drop-in. Serves need 1 (quoted-marker discrimination) only after NInfer-side
   design completes it. **Vehicle: design in `chat_parse_core`.**
4. **M4 — first-marker commit + no-end trailing drop** (qwen3-coder's actual grammar): recorded so
   it is not mistaken for a recovery mechanism; **not recommended** — it silently drops the call
   and tail where NInfer's B2 already recovers, and violates the map's no-silent-loss goal.
5. **Out of scope by prior decision — lazy-grammar generation-time constraint** (upstream's primary
   *prevention* of malformed regions): the map's 2026-10-03 user decision excludes serve-side
   generation-time constraints from this fix; recorded with source so #148 need not re-litigate.
6. **No surface — degenerate repetition**: no parse-side repetition guard exists in the vendored
   engine or upstream; the parse-side contribution is only M1's bound-cut handling.

## 2. The delta in facts: `05af0d2b` → `bed0a85660`

Commands (in the clone): `git log --oneline 05af0d2b...bed0a85660` → 68 commits;
`git diff --stat 05af0d2b bed0a85660 -- <the 21 vendored paths>` → `common/chat.cpp | 6 ++++++`,
`common/parsers/parsers.h | 2 ++`, everything else byte-identical. The single commit is
`b8f96c3e8` ("common : add LLM-jp-4.1 Harmony dialect handler (#29681)"); its server-visible parse
changes are a template sniff (`src.find("chat_format=llm-jp-harmony-v1")`) and a declaration —
nothing to do with qwen3-xml. `tools/server/server-task.{cpp,h}`, `server.cpp`,
`docs/function-calling.md`, `docs/autoparser.md` in the range carry no other parse-behavior change
(the two commits touching `server-task.{cpp,h}`/`server.cpp` are an unrelated `/v1/systemone` API
and an embedding 400; six commits touch `tools/server/` overall).

Two corrections to the map notes, both load-bearing:

- **`parse_anywhere_and_extract` is already vendored at the pinned baseline.**
  `third_party/llama-chat/common/chat-peg-parser.h:216`, implementation `:206-220` (= master,
  byte-identical). It is a **template-analysis** routine: at master its only runtime callers are in
  `common/chat-diff-analyzer.cpp` (the differential auto-parser's analyzer, excluded by design #28);
  outside that, only `tests/test-chat-peg-parser.cpp` uses it. It is not part of any request-path
  parse, so there is no "prefix skip" G1 mechanism behind it.
- **"chat.h #9639 machinery" is the origin, not a delta.** PR #9639 (merged 2025-01-30, "Tool call
  support (generic + native …) w/ lazy grammars") introduced `chat.h` function calling; the doc
  `docs/function-calling.md:3` cites it. The machinery is in the pinned baseline; what NInfer
  *uses* from it is the PEG builder/arena (`chat_parse_core.cpp:86-108`), not the runtime dispatch
  (`common_chat_parse`, lazy grammars).

Vehicle verdict: a re-vendor for G1 = 8 lines of unrelated code plus README bookkeeping. The
current drift signal the re-vendor runbook would pick up is exactly this one commit; an
opportunistic advance can ride any future maintenance PR, but it is not part of #148's fix.

## 3. Mechanisms, ranked

Each mechanism: what it does → source @ `bed0a85660` → failure shapes it covers (mapped to the
`ToolCallParseFallbackReason` classes of `include/ninfer/types.h` and the four robustness needs of
#146 §7) → NInfer's current coverage → vehicle → risks.

### M1 — Lenient truncation typing + partial-AST salvage (rank 1)

**What.** Upstream parses every chat turn with `COMMON_PEG_PARSE_FLAG_LENIENT`
(`common/chat.cpp:1471`) and the engine distinguishes three outcomes: success; `need_more_input`
(a construct ran out of input — result constant `common/peg-parser.h:70`, method `:158`; sites `common/peg-parser.cpp:280-284`
literal, `:456-461` repeat, `:505-509` any, `:727-730` until); and `fail` (a definite mismatch). On
`fail` after some progress (`result.end > 0`) during streaming, upstream maps the AST captured so
far instead of discarding it — "return partial results if any AST nodes were captured"
(`common/chat.cpp:1479-1501`) — and the mapper flushes a started call once its name is known,
including partial argument bytes (`common/chat-peg-parser.cpp:287-298`; name-triggered push at
`:358-373`; a cut schema-string argument gets its closing quote appended at `:293-295`). The salvage condition is `is_partial && result.end > 0` (`common/chat.cpp:1479-1505`) — a
**partial** parse that made progress; every other failure, including **any** failure of the final
parse, throws (`common/chat.cpp:1502-1504`), which the server turns into a 500
(`tools/server/server.cpp:54-90`); the per-chunk streaming entry point is
`tools/server/server-task.cpp:988-993` (`is_partial=true`) while the final one is
`server-task.h:392-394` (`is_partial=false`). The upstream test contract for this lives in
`tests/test-chat.cpp:1122-1218`: **every prefix of every test input must parse** (partial), and the
incremental diffs must reconstruct the current message — a reusable acceptance methodology.

**Covers** (needs 3, 4-partial, and one half of the `malformed_structure` surface):

- *A completed call followed by a failing tail* — prose after the last call, the truncated-region
  shape (payload 907), and a degenerate loop cut at the bound when a complete call exists before
  the cut. Today all of these demote the whole held region, call included.
- *Bound-cut vs malformed classification* — the input to the signaling half: `need_more_input`
  is a bound cut (today unreachable; `parse_candidate_region` never asks), `fail` is malformed.
- Does **not** cover family (b): an unbalanced parameter fragment *inside* the call's own value
  fails the value grammar itself, so there is no completed-call AST to salvage — that needs M2.

**NInfer's current coverage.** `parse_candidate_region` parses with
`COMMON_PEG_PARSE_FLAG_NONE` and requires `result.success() && result.end == input.size()`
(`src/models/qwen3_5/frontend/chat_parse_core.cpp:164-172`); `resolve_tool_region` accepts only a
candidate that parses to end-of-input and passes the adapter checks, otherwise it publishes the
whole region as content (`:473-539`). NInfer is deliberately **stronger** than upstream where it
works (B2 tries every marker occurrence; corpus `tool-call-quoted-marker-before-real-call` recovers
what upstream drops) and weaker exactly at the end-of-input boundary (all-or-nothing). Note the
architecture point: NInfer resolves once at terminal, so the port is a **terminal taxonomy**, not a
change to the hold/stream architecture.

**Vehicle: port into `chat_parse_core`.** The vendored engine already provides the flag, the
`need_more_input` type, and the arena (`common_peg_parse_context::ast` is fully populated for
partial parses); no `third_party/` file changes and no compat work. Concretely: parse candidates
LENIENT, branch on `need_more_input` vs `fail`, and extend the accept rule so a candidate whose AST
contains a structurally complete call (name node present) is salvaged when the region is truncated.
**Risks:** (a) partial arguments are wrong-by-construction; NInfer must pick an accept threshold
(upstream's is "name known"; a stricter threshold — all required parameters closed — is defensible
and less surprising); (b) B2's all-or-nothing rule is corpus-pinned
(`tool-call-later-candidate-must-consume-the-end` expects `MalformedStructure`), so the design
decision must come with rewritten/new vectors; (c) lenient changes candidate acceptance *inside*
the region grammar (the `negate`/`any` loops now report `need_more_input` at EOF), which must be
revalidated against all 42 vectors.

### M2 — Delimiter-terminated argument values (rank 2)

**What.** llama.cpp's qwen3-coder handler frames a string-typed argument as
`<parameter=NAME>\n` + raw bytes up to the exact delimiter `\n</parameter>\n`
(`third_party/llama-chat/common/parsers/qwen3-coder.cpp:91-93`, `:106-107`; typed/union fallback
`:108-135`; the engine primitive `until` at `common/peg-parser.cpp:685-732` and
`chat-peg-parser.h:106-107`). Marker fragments inside a value are ordinary text; a nested
`<parameter=…>` is only structural when it is newline-framed like a real close — then it ends the
value early and the turn fails (`chat-parsing-oracle.md` row 21).

**Covers** the live family (b) shape (needs 2): the dispatch that quoted `<parameter=…>` fragments
inside its own argument text. Under NInfer's B4 the value grammar admits `<parameter=`/`</parameter>`
only as **balanced nested markup** (`chat_parse_core.cpp:90-97`, comment `:74-76`), so an inline
fragment is unrepresentable and the whole region demotes (`malformed_structure`); under upstream's
rule the same call parses with the fragment as value bytes. Framing still divides the two rules: a
**balanced** newline-framed nested pair inside a value is accepted by NInfer (corpus
`tool-call-nested-parameter-newline-framed`, `tool_calls: 1`, markup preserved) and fails under
llama.cpp (oracle row 21); only the **unbalanced** newline-framed case fails under both.

**NInfer's current coverage / conflict.** B4 is deliberate and corpus-pinned:
`tool-call-unmatched-nested-parameter` and `tool-call-standalone-parameter-close` both expect
`MalformedStructure`, and the unmatched-nested vector records `diverges: ["llamacpp(row 22): drops
the tail"]`. Adopting
upstream's semantics therefore **changes pinned behavior** and rewrites vectors.

**Vehicle: semantics port into the region grammar** — only as an explicit #148 decision, and
possibly hybrid (keep nested-parameter representation; relax only the "bare opener/standalone close
inside a value" rule toward raw text). Do not copy the `\n`-framed delimiters literally: they are
template-shaped, and NInfer's corpus canonical form does not carry the framing newlines. **Risks:**
medium-high; behavior change pinned by the corpus; normalization (B6) interactions; it buys the
inline-fragment case only.

### M3 — `tag_with_safe_content` (rank 3)

**What.** A builder helper that returns `zero_or_more(choice({p, rule(tag_name,
content(negate(literal(marker)) + any() + until(marker))))})` — a structural element tried first, a
content chunk that runs from a non-marker byte to the next marker otherwise
(`common/chat-peg-parser.cpp:229-237`). It is the closest thing in the tree to a "content owns
marker occurrences" rule for need 1.

**Covers:** need 1 as a *pattern* only — as written, a chunk cannot start at a marker byte, so a
marker occurrence that is not a valid structural element still stalls the repeat; the
quoted-marker-stays-text case is not implemented by it. Upstream never used it: neither `common/`
nor `tools/server/` has a caller for `tag_with_safe_content`.

**Vehicle: NInfer-side design** borrowing the shape (in the region arena or the channel scanner);
it cannot be described as porting a working upstream mechanism. **Risks:** the semantics that matter
(marker at a chunk boundary) must be designed; corpus vector `tool-call-marker-quoted-without-call`
is the target shape.

### M4 — First-marker commit + trailing drop (documented, not recommended)

qwen3-coder's grammar has **no `p.end()`** (`qwen3-coder.cpp:170`, `:174`): content stops at the
first `<tool_call>` (`until_one_of(tool_call_starts)`), the call section is an optional repeat
(`p.repeat(calls, min_calls, 1)`, min 0 — `:156-167`), so a region that fails to parse is simply
left unparsed and dropped: no demotion, no error, no bytes. The corpus's `diverges` fields and the
oracle matrix record exactly this for rows 22/23/24/26 ("llamacpp: drops the tail / drops the rest
of the turn"). For NInfer this is a regression on every axis it cares about: B2 recovers a later
real call precisely where upstream drops it, and R6's byte-exactness exists because silent drops
corrupt answers. Recorded only so the design does not port it by accident while borrowing the
grammar layout.

### Design reference (not a port source): vLLM `qwen3_xml`

The second reference (digested in `docs/research/reference-baselines.md` §2.2 @ `7c976d17`; vLLM
`ced6857`) is a per-architecture state machine, not a port source, but three of its design points
bear on the ranking above:

- **close-at-end**: at stream end an unterminated tool state still emits a (possibly
  argument-truncated) call rather than dropping it (`streaming_parser_engine.py:305-316`). That is
  the same outcome M1 reaches through llama.cpp's `need_more_input` path, and it independently
  confirms "emit the completed prefix, close the call at the cut" as an accepted policy for need 3.
- **drop-not-leak preamble**: malformed preamble maps to no content event
  (`parser_engine_config.py:66-73`). That is M4's shape — silent byte loss — and conflicts with
  NInfer's R6 byte-exactness / no-silent-loss goal; informative, not adoptable.
- **undeclared names accepted** (`validate_tool_names` defaults to false): conflicts with the
  corpus-pinned fail-closed `undeclared_tool` policy (oracle row 28, chosen deliberately); not
  adoptable.

So vLLM's contribution is to validate M1's truncation policy; its drop and name policies stay
rejected.

### Explicitly not a port surface

- **Undeclared tool name, duplicate parameter, invalid name.** Upstream bakes declared names and
  required arguments into the grammar and drops the tail on a violation; for optional duplicate
  parameters it emits duplicate JSON keys (oracle row 20). NInfer's adapter policies
  (`chat_parse_core.cpp:186-192`, `:486-520`) are deliberate and corpus-pinned (`undeclared_tool`
  and the #42 duplicate merge) — nothing to port, and upstream's behavior is worse.
- **Lazy-grammar generation-time constraint** (`qwen3-coder.cpp:179-191`: `grammar_lazy` +
  `<tool_call>`/`<function=name>` triggers). This is upstream's primary *prevention* mechanism:
  under `tool_choice: auto` the grammar activates at the first marker and forces well-formed
  structure, so malformed tails rarely occur. The map's user decision (2026-10-03) excludes
  serve-side generation-time auto-constraint from this fix; cite, don't re-open.
- **Degenerate repetition.** No parse-side guard exists: `parse_depth` is debug indentation only
  (`common/peg-parser.cpp:233`, `:300`), the only repetition bounds are grammar-construction bounds
  (`qwen3-coder.cpp:156-167`), and the upstream mitigations are sampling-side (repeat
  penalty/DRY) or the lazy grammar. The parse-side contribution to need 4 is M1's bound-cut
  handling; earlier detection is a sampling/serve-policy question outside parse/recovery scope.

## 4. Coverage grid

| | need 1: quoted markers | need 2: value balance | need 3: truncation | need 4: repetition | classes affected |
|---|---|---|---|---|---|
| M1 lenient + salvage | partially (complete call + failing tail) | no | classification + salvage | bound-cut only | `malformed_structure` ↓ |
| M2 delimiter values | no | inline fragments accepted | no | no | `malformed_structure` ↓ (family b) |
| M3 safe content | pattern only (needs design) | no | no | no | `malformed_structure` ↓ (quoted markers) |
| M4 drop tail | no (drops bytes) | no | no | no | — |
| lazy grammar | prevention (out of scope) | no | no | prevention | — |

The three classes not in the table (`invalid_tool_name`, `undeclared_tool`, `duplicate_parameter`)
have no upstream mechanism worth porting (§3, "not a port surface"); `trailing_content` is
unassigned in NInfer and needs no port.

## 5. Vehicle analysis and recommendation for #148

- **Port into `chat_parse_core` (recommended).** M1 and M3 are NInfer-owned changes that reuse
  byte-identical vendored machinery: the LENIENT flag and `need_more_input` propagation are part of
  the pinned baseline's PEG engine, and the failed/partial parse's AST is already accessible. Zero
  `third_party/` diffs, zero compat-shim impact, no new vendored files — the "prefer llama.cpp"
  constraint is satisfied by *using the vendored engine per its own upstream semantics* rather than
  reimplementing a scanner.
- **Baseline advance (not for G1).** The runbook (`docs/maintainer/llama-chat-vendor.md`) makes it
  mechanical, but the measured yield is the LLM-jp registration only; spending a re-vendor PR here
  adds risk without G1 value. The drift note can be quieted on the next maintenance pass.
- **New vendored file / patch (not needed).** No G1-relevant mechanism lives in an unvendored file,
  and nothing requires changing vendored bytes — the llama-grammar patch precedent exists because
  the engine itself had to change (MAX_REPETITION_BOUND); here it does not. A patch would only add
  drift debt.
- **Recommended mix for #148:** (1) M1 as the primary robustness vehicle, with new corpus vectors
  for prose-after-call, the truncated-region shape, quoted-marker-before-real-call-with-failing-tail,
  and a bound-cut turn; (2) treat M2 as a separate, explicit semantics decision — if adopted, write
  it as a narrow value-rule change with the rows-21/22 vectors rewritten and the nested-parameter
  representation preserved; (3) use M3 only if need 1 is not adequately covered by M1, and write it
  as an NInfer grammar rule tested against `tool-call-marker-quoted-without-call`; (4) use upstream's
  prefix-stability + diff-accumulation test methodology (`tests/test-chat.cpp:1122-1218`) as the
  model for the post-fix robustness tests.

## 6. Verification gaps

- No builds/runs (research ticket): the lenient/`need_more_input` tracing through
  choice/sequence/repeat/until (`common/peg-parser.cpp:295-475`, `:685-732`) is a code read, not an
  execution; the exact AST content of a partial parse at each truncation point is not measured.
- The live payloads of #146 (req 876/878 argument fragments) were not re-examined byte-for-byte;
  the family (b) mapping assumes the fragments are inline rather than newline-framed. If they were
  newline-framed, M2 buys less than stated.
- The oracle matrix and corpus `diverges` rows are prior research artifacts (branches cited above);
  the upstream half of this document was re-derived directly from `bed0a85660` this session, and
  the NInfer half from `src/models/qwen3_5/frontend/chat_parse_core.{h,cpp}` at `8a0d16bf`.
