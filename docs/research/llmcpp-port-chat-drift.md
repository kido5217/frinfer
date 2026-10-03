# Upstream drift survey: llama-chat / llama-grammar (wayfinder map #167, ticket #169)

**Scope.** What has landed upstream in llama.cpp's chat-parsing + grammar stack since NInfer's pinned
baselines, and which of it is a port candidate. Chat/grammar-stack drift only; boundary-crossing
items are flagged for the map.

**Method.** Blobless llama.cpp clone at `/tmp/opencode/llamacpp-serve-ref` (checked out at live
master); commit-level range diffs and per-commit `ls-tree` at both ends; `gh api` for the live
master SHA; NInfer side verified read-only in this worktree plus `gh issue view`/`gh pr view` of
map #145's tickets (#145–#151) and PRs #153–#155. No builds, no GPU work.

**Refs (all verified this session).**

| Ref | Value |
|---|---|
| Pinned chat baseline | `05af0d2b1398394cfa67e1918fee7feabccaa9bc` (2026-09-30, "glm5-next: give dead indexer slots unique scatter rows (#29745)"); per-file byte-identity recorded in `third_party/llama-chat/README.ninfer.md` |
| Pinned grammar baseline | same commit; `src/llama-grammar.{h,cpp}` + 3 local patches (`0001-repetition-bound`, `0002-parser-last-error`, `0003-quiet-parse-errors`); verified by `third_party/llama-grammar/verify-vendor.sh` (env `LLAMA_CPP_REF` overrides the ref clone) |
| Jinja fork base | `7609846557c50f9d984719a9e1e8c5f3d02f807b` (`third_party/llama-jinja/README.ninfer.md`); verified ancestor of both the pinned baseline and live master |
| Live llama.cpp master | `1537a0a8b2f8711d840878b0a0677ab2213c882c` (2026-10-03, "server : fix laya abort by limiting n_batch to n_ubatch (#29903)"), via `gh api repos/ggml-org/llama.cpp/commits/master` |
| NInfer master | `3e0ff840b9f3f3efc6ee82305b540198f6e6bd81` (2026-10-03) |

## Delta overview

`05af0d2b` → `1537a0a8b2` is **77 commits / 267 files repo-wide** (2026-09-30 → 2026-10-03). The
chat + grammar surface — `common/` (recursive), `src/llama-grammar.{h,cpp}`, `include/` —
exactly **15 files** changed, all under `common/` (`arg.cpp`, `chat.cpp`, `common.cpp/h`,
`download.cpp`, `jinja/runtime.cpp`, `log.cpp`, `parsers/{ling3,llm-jp-harmony}.cpp`,
`parsers/{parsers.h,sources.cmake}`, `sampling.cpp/h`, `speculative.cpp/h` — the per-file table
below). Everything else on the surface is **byte-identical** at both commits: the PEG grammar
engine (`src/llama-grammar.{h,cpp}`), every vendored parse mechanism (`peg-parser.*`,
`chat-peg-parser.*`, `chat.h`, `json*`, `json-schema*`, `json-schema-to-grammar*`, `trie.*`,
`unicode.*`, `chat-auto-parser*`, `parsers/qwen3-coder.cpp`, `parsers/parsers.cpp`), the jinja
caps layer, `common/llguidance.cpp`, `include/`, and the `grammars/` corpus. A full-range sweep
over `tools/`, `ggml/`, `tests/`, `models/templates/`, `examples/` found no further
chat/grammar-surface changes beyond the excluded-subsystem rows in the table (which account for
the rest of the repo-wide 267 files). What changed: (1) one new specialized parser for the
**LLM-jp-4.1** Japanese family (GPT-OSS-style channel dialect, `b8f96c3e8` / #29681) — the only
commit touching the vendored set (`common/chat.cpp` +6 registration, `common/parsers/parsers.h` +2
declaration; the 164-line parser file itself is outside the vendored set); (2) **Ling-3.0**
`json_schema` response-format honoring in the excluded Ling parser (`9bf55f4a3` / #29813); (3) a
3-line **jinja `for`-loop scope perf fix** (`def4d406a` / #29776), postdating the jinja fork base;
(4) **speculative rejection sampling** for MTP draft acceptance in the excluded sampler/speculative
stack (`1fb7ef3e3` / #27694); (5) non-vendored plumbing (`common/common.{h,cpp}`, `log.cpp`,
`download.cpp`, `arg.cpp` — u8path/`common_is_tty` #29860, cache-dir symlink fix #29816, mmproj arg
fix #28977, spec CLI options). Upstream's own `tests/test-chat.cpp` gained only family-specific
vectors (LLM-jp dialect; Ling response-format) — no new generic mechanism.

| Surface file | Change | Commit (PR) | In vendored set? |
|---|---|---|---|
| `common/chat.cpp` | +6 (LLM-jp registration) | `b8f96c3e8` (#29681) | **yes** |
| `common/parsers/parsers.h` | +2 (LLM-jp declaration) | `b8f96c3e8` (#29681) | **yes** |
| `common/parsers/llm-jp-harmony.cpp` | new, 164 | `b8f96c3e8` (#29681) | no (excluded parser) |
| `common/parsers/sources.cmake` | +1 | `b8f96c3e8` (#29681) | no |
| `common/parsers/ling3.cpp` | +16 (json_schema) | `9bf55f4a3` (#29813) | no (excluded parser) |
| `common/jinja/runtime.cpp` | 3 (for-loop scope) | `def4d406a` (#29776) | no — jinja fork family |
| `common/sampling.{h,cpp}` | +3 / +126 (rejection sampling) | `1fb7ef3e3` (#27694) | no (excluded) |
| `common/speculative.{h,cpp}` | +10± / +119± | `1fb7ef3e3` (#27694), `c061df198` (#29761), `60e9cf7a7` (#29601) | no (excluded) |
| `common/arg.cpp` | +18 | `1fb7ef3e3` (#27694), `feb9a3d6d` (#28977) | no |
| `common/common.{h,cpp}`, `log.cpp`, `download.cpp` | plumbing | `edd6e2bbd` (#29860), `f1cee9941` (#29816) | no |
| `models/templates/llm-jp-llm-jp-4.1-8b-thinking.jinja` | new | `b8f96c3e8` (#29681) | no |
| `tests/test-chat.cpp` | +86 (family vectors) | `b8f96c3e8` (#29681, +70), `9bf55f4a3` (#29813, +16) | no |
| `examples/{simple-chat,retrieval,llama.android}` | sampler API | `1fb7ef3e3` (#27694) | no |

Cross-check on map #147's measurement: `05af0d2b` → `bed0a85660` (2026-10-02) is 68 commits with
exactly one commit touching the vendored set — `b8f96c3e8` — re-verified byte-for-byte against
`#147`'s "8 lines of unrelated LLM-jp dialect registration". Live master adds 9 commits; none of
them touch the vendored set.

## M1/M2/M3 status (re-verified, map #145 — CLOSED, 6/6 sub-issues)

- **M1 — lenient truncation typing + partial-AST salvage: ported.** #149 closed, merged `856d26fb`
  (PR #154, "feat(chat): second-chance raw values and partial-AST salvage for tool calls"). The
  vehicle was refined in ADR-0002's implementation note: the strict parse's retained completed
  nodes (a failed rule creates no node) supply the structurally-complete-call criterion directly,
  so the LENIENT `need_more_input`/`fail` typing was dropped — no consumer branches on
  bound-cut-vs-malformed, and a LENIENT pass would have contradicted B3 optional-close tolerance.
- **M2 — delimiter-terminated values: decided and implemented.** Full upstream
  `until("\n</parameter>\n")` value semantics rejected (it fails a value carrying a balanced
  newline-framed nested pair, a shape that works today); the second-chance raw-value parse
  (strict-first, raw bytes to a newline-framed close, LF/CRLF) landed in the same PR #154.
  Corpus now 48 vectors (42 pre-#154: 4 rewritten in place + 6 new, net +6) with the upstream
  prefix-stability gate (`tool-call-unmatched-nested-parameter`, `tool-call-standalone-parameter-close`,
  both `later-candidate-must-consume-the-end` variants rewritten; `complete-call-before-cut`,
  `inline-fragment-live-shape` (live-876), `later-call-not-swallowed`, `raw-crlf-framing`,
  `salvage-adapter-failure-stays-fail-closed`, `truncated-mid-parameter` (live-907) added).
- **M3 — `tag_with_safe_content`: parked.** Unused upstream (no callers at the pinned baseline)
  and incomplete for the marker-at-chunk-boundary case; needed only if M1+M2 leave a residual
  gap. No open ticket; map #145 closed with #151 verification done.
- Signaling half (out of M scope): #150/#155 merged `91b9c045` — call-loss demotions signal
  in-band (SSE error / `response.failed` / `event: error`) with per-class codes, non-streaming
  409 + `x-should-retry: true`; benign prose demotions stay silent byte-exact round-trips.

## Candidates

### C1 — LLM-jp-4.1 Harmony dialect handler

**Upstream.** `b8f96c3e8` (2026-10-01), PR #29681, "common : add LLM-jp-4.1 Harmony dialect
handler". Sources: `common/parsers/llm-jp-harmony.cpp` (new, 164 lines), `common/chat.cpp` (+6,
`common_chat_try_specialized_template` gains a `chat_format=llm-jp-harmony-v1` branch),
`common/parsers/parsers.h` (+2 declaration), `models/templates/llm-jp-llm-jp-4.1-8b-thinking.jinja`,
`tests/test-chat.cpp` (dialect vectors: channels, one-space rule, reasoning, parallel
`<|end|>`-separated calls, `to=functions.<name>` recipients).

**What it does.** A specialized PEG parser for the LLM-jp-4.1 Japanese model family: a GPT-OSS
channel-based dialect (`<|channel|> final<|message|>`), spaces after special tokens,
`<|end|>`-delimited parallel tool calls.

**Vehicle.** Not a port. The parser is family-specific and NInfer supports only the Qwen3.5/3.6/3.8
`.ninfer` family (map #139's G3 is a product-level coverage decision, not this ticket's). It can
only reach NInfer as a byproduct of advancing the vendored baseline (C5): two shared files
(`chat.cpp`, `parsers.h`) carry the registration; the 164-line parser stays excluded and gains a
throwing stub in `compat/excluded_stubs.cpp` (14 excluded parsers → 15, per the file-map
adaptation table).

**Cost + contract impact.** 8 vendored lines + ~10 stub lines + README count bump; build +
`ctest -R 'llama_chat|chat_pars|frontend'`. Zero behavior change on the Qwen3.5 route: the
registration is gated on the LLM-jp template marker NInfer never loads, `parsers/qwen3-coder.cpp`
and the PEG engine are byte-identical, and the 48-vector corpus + prefix-stability + demotion
classes are untouched.

**Verdict: no** — new model family outside product scope; no contract gap (a), no performance (b)
axis, and the drift it causes is absorbable at near-zero cost by C5.

**Evidence needed (if C5 lands).** Per-file `cmp` of `chat.cpp`/`parsers.h` against
`1537a0a8b2`; the new stub throws (runbook: excluded symbols never no-op); vendor smoke + corpus
green.

### C2 — Ling 3.0 `json_schema` response-format honoring

**Upstream.** `9bf55f4a3` (2026-10-03), PR #29813, "chat : honor json_schema in Ling 3.0 parser".
Sources: `common/parsers/ling3.cpp` (+16), `tests/test-chat.cpp` (+16: response-format vectors
with thinking on/off on the PEG parser).

**What it does.** The Ling-3.0 specialized parser now enforces a `response_format` JSON schema on
the content channel.

**Vehicle.** None — `ling3.cpp` is an excluded specialized parser; NInfer does not support the
Ling family. The capability it implements (schema-enforced response format) already exists in
NInfer as its own product: `response_format` `json_object`/`json_schema` → GBNF → constrained
decoding on the public Engine route (map #45, `src/serve/constraint_contract.cpp`).

**Verdict: no** — excluded family-specific parser with no NInfer-side surface; the mechanism NInfer
needs is already a delivered product feature.

### C3 — jinja `for`-loop scope perf fix

**Upstream.** `def4d406a` (2026-10-01), PR #29776, "jinja : skip copying loop scope unless a loop
filter needs it". Source: `common/jinja/runtime.cpp` (3 lines, `for_statement::execute_impl`).

**What it does.** The per-iteration `context loop_scope(scope)` copy is now constructed only when
the loop has a `select`/`test` filter. Verified semantics-preserving at `1537a0a8b2`: `loop_scope`
is referenced only inside the `if (select_expr && test_expr)` branch; the loop body executes on
`scope` directly, so unfiltered loops no longer pay a context copy per iteration. Pure
performance.

**Vehicle.** NInfer-side selective adoption into the maintained jinja fork
(`third_party/llama-jinja`) — the fork README's stated policy ("adopts upstream fixes
selectively"). **Not a verbatim apply:** the fork's `runtime.cpp` is NInfer-reformatted relative
to fork base `76098465` (~850-line diff — checkpoint calls, includes, style; its blob matches no
upstream commit in the range), so `def4d406a`'s patch lands only with fuzz 1 against the fork
file — a small manual 2-hunk port placed by hand, then verified by the fork's jinja test suite.
`76098465` is a verified ancestor of `def4d406a`, so the change is semantics-compatible with the
fork's base. Not a re-vendor: the runbook explicitly keeps `third_party/llama-jinja` on its own
baseline, outside the chat procedure.

**Cost + contract impact.** Small manual port (2 hunks in the reformatted fork file) + README
adoption note + the existing template parity gate (`NINFER_PARITY_TEMPLATE=<froggeric> ctest -R
chat_templates`) plus the fork's jinja test suite to confirm semantics. No parse-corpus impact
(the 48 vectors exercise the parse side, not rendering); no demotion-class impact.

**Axis.** (b) performance — template rendering runs once per request on CPU, not per token; the
gain scales with loop iterations (message/tool count in the prompt), so it is small at current
deployment shapes but free to take.

**Verdict: port (low priority)** — verified-semantics-preserving adoption via the fork's own
documented policy; the port is manual (fuzz-1 patch) rather than verbatim, but still small. **Flagged for the map:** it belongs to the `llama-jinja` fork
family, not the chat/grammar vendored set — routing is the map's call.

**Evidence needed.** Post-port fork file: the `context loop_scope(scope);` construction moved
inside the `select && test` branch (the diff against base `76098465` then shows only the NInfer
reformatting plus that move); fork jinja test suite green; froggeric parity test unchanged;
optional: template-render microbenchmark on a long multi-tool conversation to make the (b) claim
measurable.

### C4 — Speculative rejection sampling for MTP draft acceptance

**Upstream.** `1fb7ef3e3` (2026-10-02), PR #27694, "spec : add probabilistic sampling for simple
draft and MTP". Sources: `common/sampling.cpp` (+126:
`common_sampler_sample_and_accept_n_rejection` — accept a drafted token with probability
`min(1, p/q)`, else draw from the residual `norm(max(0, p - q))`, with grammar masking on the
candidate array), `common/sampling.h` (+3, sampler `rng` state), `common/speculative.{h,cpp}`,
`common/arg.cpp` (+18 options), `examples/` API updates.

**What it does.** A distribution-exact rejection-sampling acceptance path for speculative/MTP
draft tokens in upstream's sampler chain.

**Vehicle.** None — the sampler/speculative stack is excluded from the vendored set; NInfer's MTP
is its own product execution path (constrained e2e already covers ordinary+MTP, map #45), and
NInfer does not use `common_sampler` at all.

**Verdict: no** — not the chat/grammar surface and not the vendored set; NInfer's MTP acceptance
is a product-design concern, not a parse-stack port. **Flagged for the map** as an MTP
product/runtime item should acceptance-path quality ever be revisited.

### C5 — Advance the vendored baseline to `1537a0a8b2` (the vehicle action)

**Drift state.** The runbook's drift alert is live by its own definition: `b8f96c3e8`
(2026-10-01) postdates the pinned baseline and touches the vendored `chat.cpp` + `parsers.h`.
Re-arming the note requires the baseline advance.

**Constraint analysis.**
- Grammar engine: `src/llama-grammar.{h,cpp}` is byte-identical between baseline and live master
  (absent from the range diff) → the 3 local patches re-apply cleanly with no adaptation;
  `verify-vendor.sh` re-verifies against the same file set.
- Chat set: exactly 2 vendored files change (`chat.cpp` +6, `parsers.h` +2); the new
  `llm-jp-harmony.cpp` stays excluded → one new throwing stub
  (`common_chat_params_init_llm_jp_harmony`); `common/parsers/sources.cmake` is not vendored, so
  no CMake change; `README.ninfer.md` gains the new baseline SHA/date and the excluded-parser
  count 14 → 15.
- Precedent: #44 (advance with zero vendored-file changes); the minimal-set decision is #28.

**Cost estimate.** A small pipeline PR: 2 file copies, ~10 stub lines, README edits, then the
standard gate — build + `ctest -R 'llama_chat|chat_pars|frontend'` (vendor smoke, 48-vector
corpus, prefix-stability, frontend fixtures) and per-file `cmp` against `1537a0a8b2`.

**Contract impact.** None on the Qwen3.5 route (registration gated on an unloaded template marker;
`qwen3-coder.cpp` and the PEG engine byte-identical); corpus and demotion classes untouched.

**Verdict: defer** — the advance buys nothing functional for NInfer (C1 is out of scope; the
grammar engine is unchanged); do it as a standalone hygiene PR when the map wants the drift alert
re-armed, or fold it into the next content-bearing re-vendor.

## Verified unchanged (negative findings)

- `src/llama-grammar.{h,cpp}` — the PEG grammar engine: parsing, compilation, repetition handling
  (including the `MAX_REPETITION_THRESHOLD` context the 0001 patch modifies) — byte-identical.
- The vendored parse machinery: `peg-parser.{h,cpp}`, `chat-peg-parser.{h,cpp}`, `chat.h`,
  `json.{h,cpp}`, `json-schema.{h,cpp}`, `json-schema-to-grammar.{h,cpp}`, `trie.{h,cpp}`,
  `unicode.{h,cpp}`, `chat-auto-parser.h`, `chat-auto-parser-helpers.h`,
  `parsers/qwen3-coder.cpp`, `parsers/parsers.cpp`.
- `common/llguidance.cpp` (guidance sampling) and the `grammars/` GBNF corpus.
- Jinja caps layer (`common/jinja/caps.{h,cpp}`), `parser.{h,cpp}`, `lexer.{h,cpp}`,
  `string.{h,cpp}`, `value.{h,cpp}`, `utils.h` — the compat shim's "fork predates the caps layer"
  gap is unchanged.
- No new lenient/recovery parse machinery in the range: map #147's standing finding holds —
  `parse_anywhere_and_extract` and the PEG parse layer are already in the pinned baseline, and
  nothing in the 77 commits adds a new one. No new jinja features.

## Summary table

| Candidate | Verdict | Vehicle | Axis | One-line rationale |
|---|---|---|---|---|
| C1 LLM-jp-4.1 Harmony (#29681) | no | — (re-vendor byproduct only) | — | new Japanese model family; NInfer serves the Qwen3.5 `.ninfer` family only |
| C2 Ling-3.0 `json_schema` (#29813) | no | — | — | excluded family-specific parser; schema-enforced responses are NInfer's own constrained-decoding product |
| C3 jinja `for`-loop scope (#29776) | port (low priority) | NInfer-side selective adoption into `third_party/llama-jinja` | (b) | 3-line upstream, verified semantics-preserving perf fix; manual 2-hunk port into the NInfer-reformatted fork file (patch lands with fuzz 1) |
| C4 spec rejection sampling (#27694) | no | — | — | excluded sampler stack; NInfer's MTP acceptance is its own product path |
| C5 baseline advance to `1537a0a8b2` | defer | re-vendor (runbook `docs/maintainer/llama-chat-vendor.md`) | (c) | buys nothing functional (grammar engine unchanged, 3 patches re-apply); near-zero cost; do when the drift alert needs re-arming |

## Flagged for the map

1. **C3 belongs to the `llama-jinja` fork family** — a third vendored llama.cpp subset with its own
   baseline (`76098465`) and selective-adoption policy, explicitly outside the chat re-vendor
   procedure. Recommend a small standalone adoption PR; not a chat/grammar-vendor decision.
2. **C4 is an MTP product/runtime item**, not chat/grammar-stack drift — only relevant if NInfer's
   MTP acceptance path is ever revisited (upstream now has a distribution-exact rejection path +
   grammar-masked candidate arrays to compare against).
3. **Drift alert is live** by the runbook's definition (`b8f96c3e8` post-baseline touches vendored
   files); it re-arms only with the C5 baseline advance. The alert's cost of staying deferred is
   one recurring note; the advance's cost is one small pipeline PR.
4. **No generic upstream recovery/lenient machinery arrived** in the range — the M1/M2 machinery
   map #145 consumed was already in the pinned baseline (confirmed again by the range diff), so
   there is no new G1-side content to chase upstream.