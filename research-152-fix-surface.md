# Research: #152 fix surface on current master

Wayfinder map #157, ticket #159. Resolved against `master` at `91b9c045`
(`feat(serve): signal call-loss tool-call demotions to clients (#155)`).
Static analysis only; no build or engine run. All references are `file:line`
in that tree.

## Q1 — Realized-count flow

Confirmed. The realized reasoning count flows through one unbroken chain and
is in scope at the `request_done` write site, where it is currently not
serialized:

1. **Source counter.** `OutputSession::DecoderState.reasoning_tokens`
   (`src/models/qwen3_5/frontend/output_session.cpp:151`) increments per
   accepted token while the parsing core is in the reasoning phase and
   reasoning splitting is on: `if (in_reasoning && impl_->split_reasoning) {
   ++impl_->preview_state.reasoning_tokens; }` (`output_session.cpp:515`),
   with the same condition on the thinking-control preview path
   (`output_session.cpp:626-627`). Preview commits to live state by swap in
   `OutputSession::commit_preview` (`output_session.cpp:687-698`).
   Accessor: `OutputSession::reasoning_tokens()`
   (`src/models/qwen3_5/frontend/output_session.h:78`,
   `output_session.cpp:708-709`).
   Note the independence from budgets: it increments regardless of whether a
   thinking budget is configured (contrast `output_session.cpp:516-517`,
   which gates the budget tracker on `budget` being set).
2. **Engine result.** The single write site of
   `GenerationResult::reasoning_tokens` (`include/ninfer/types.h:836`) is
   `complete_success`: `result.reasoning_tokens =
   request->output.reasoning_tokens();` (`src/runtime/engine/engine_core.h:891`).
   `Request::output` is an `OutputSession`
   (`src/runtime/engine/request_record.h:160`). `complete_success`
   (`engine_core.h:863`) is the only `GenerationResult` construction for
   generation requests (called from every terminal path: `engine_core.h:933,
   1001, 1028, 1308, 1315`).
3. **Serve outcome.** `GenerationService::run`
   (`src/serve/generation_service.cpp:419-474`) is the only
   `GenerationResult` → `GenerationOutcome` conversion: `outcome.reasoning_tokens
   = static_cast<int>(result.reasoning_tokens);`
   (`src/serve/generation_service.cpp:441`). `GenerationOutcome` carries
   `int reasoning_tokens = 0` (`src/serve/generation_service.h:54`) alongside
   `ninfer::ThinkingBudgetStats thinking` (`generation_service.h:55`).
4. **Write site.** `format_request_done_json`
   (`src/serve/request_log.cpp:554-584`) takes `const GenerationOutcome&`
   (`request_log.cpp:556`). `outcome.reasoning_tokens` is in scope but absent
   from the `result` object (`request_log.cpp:559-575`), which currently
   serializes only the budget-tracker fields: `thinking_budget` (568-570),
   `model_thinking_tokens` (571), `thinking_control_tokens` (572),
   `thinking_control_applied` (573).

**Routes that write `request_done`.** There is exactly one funnel:
`HttpServer::RequestLifecycle::done` (`src/serve/http_server.cpp:242-244`) →
`record_request_done` (`http_server.cpp:268-272`) →
`JsonlRequestLog::write_request_done` (`src/serve/request_log.cpp:851-854`)
plus the human-facing `OperationalLog::request_done`
(`src/serve/operational_log.cpp:409-411`). Three HTTP generation routes
reach it, each in non-streaming and streaming variants, and each obtains its
`GenerationOutcome` from the same `service_->run(...)` call:

| Route | Registered | Handler: `run` / `done` sites |
|---|---|---|
| `POST /v1/chat/completions` | `http_server.cpp:440-443` | `src/serve/openai_chat_http.cpp:77` / `:205` → `:93` / `:231` |
| `POST /v1/responses` | `http_server.cpp:444-446` | `src/serve/openai_responses_http.cpp:299` / `:445` → `:311` / `:466` |
| `POST /v1/messages` (Anthropic) | `http_server.cpp:475-477` | `src/serve/anthropic_messages_http.cpp:101` / `:201` → `:116` / `:226` |

**CausalScoring is not a serve route** — the ticket's parenthetical needs
correction. `EnginePurpose::CausalScoring` (`include/ninfer/types.h:40`) is
consumed directly by the perplexity CLI (`apps/perplexity/main.cpp:214, 384`)
and the real-model score test (`tests/models/qwen3_5/test_engine_score_real.cpp:19`)
through the Engine's scoring path (`src/runtime/engine/engine.cpp:248-249`),
which never produces a `GenerationResult`/`GenerationOutcome`. The perplexity
app references no request-log facility (grep `request_log|jsonl` in
`apps/perplexity/`: no matches). Offline scoring therefore writes no
`request_done` records; no change is needed on that path for #152.

**Realized count already emitted by API usage chunks** (same field, three
protocol renderings): OpenAI chat
`completion_tokens_details.reasoning_tokens`
(`src/serve/openai_chat_response.cpp:163`), OpenAI Responses
`output_tokens_details.reasoning_tokens`
(`src/serve/openai_responses_response.cpp:184`), Anthropic `thinking_tokens =
max(0, outcome.reasoning_tokens)` (`src/serve/anthropic_messages_response.cpp:92`).

## Q2 — Schema-bump convention (v22 → v23)

**The ticket background is stale here:** v22→v23 was not done in `75dbfeec`.
`git log -L23,23:src/serve/request_log.h` shows the actual sequence:

- `75dbfeec` (fix(chat-parse): merge byte-identical duplicate parameters (#42))
  — **v21 → v22**; its commit message says "request-log schema v21 -> v22".
- `856d26fb` (feat(chat): second-chance raw values and partial-AST salvage for
  tool calls (#154), 2026-10-03) — **v22 → v23**, adding the
  `call_attempted`/`salvaged_calls` diagnostics.

The convention (visible in both bumps, and in the earlier `04350ba9`
v20→v21 and `719d56ef` v19→v20): one atomic commit that

1. bumps the constant `kRequestLogSchemaVersion`
   (`src/serve/request_log.h:23`, currently `= 23`);
2. adjusts the `format_*` JSONL writers in `src/serve/request_log.cpp`
   (#154: +2 lines to the `request_done` `tool_call_parse` object);
3. follows the bench consumer constant
   `SERVER_LOG_SCHEMA_VERSION = 23` (`tools/bench/run_serve_corpus.py:89`).
   That consumer validates the identity `(artifact_type, schema_version,
   event)` of **every** log line via `require_server_log_identity`
   (`tools/bench/run_serve_corpus.py:465-473`) and aborts the campaign on any
   mismatch — so a v23→v24 writer without the consumer bump fails the bench;
4. follows the fixture test, which hardcodes the literal:
   `"schema_version": 23` in the `request_done` fixture
   (`tests/test_serve_corpus.py:44`); the expected server-log identity is
   derived from `corpus.SERVER_LOG_SCHEMA_VERSION`
   (`tests/test_serve_corpus.py:164`);
5. follows the prose mentions: `tools/bench/README.md:161` ("schema-v23
   serving records") and `docs/serving.md:960` ("Every line is one
   `ninfer_serve_request_log` schema-v23 JSON object").

Tests that pin the version:

- `tests/test_request_log.cpp:155` asserts the `server_start` identity
  **through the constant** (`server.at("schema_version") ==
  kRequestLogSchemaVersion`) — it does not hardcode the number, so it needs no
  literal change on a bump; its `request_done` field assertions
  (`tests/test_request_log.cpp:433-455`, thinking block at 440-444) are what a
  field rename/new-field must update.
- `tests/test_serve_corpus.py:44` — the only in-repo literal `23` that must
  follow the bump (per `856d26fb`'s one-line change there).
- The corpus-consumer agreement is the documented verification policy:
  "The corpus consumer test protects its exact schema-version agreement with
  Serve" (`docs/maintainer/logging.md:169-173`).

So v23 → v24 follows: bump `request_log.h:23` + writer change +
`run_serve_corpus.py:89` + `test_serve_corpus.py:44` + the two prose lines,
all in one commit.

## Q3 — JSONL consumer surface

### In-repo

| Consumer | Role | Reads `model_thinking_tokens`? | Rename/bump impact |
|---|---|---|---|
| `tools/bench/run_serve_corpus.py` | primary measurement consumer | no — reads only `result.prompt_tokens`/`result.completion_tokens` (`:562-563`) from `result` (`:537`) | **must follow** `SERVER_LOG_SCHEMA_VERSION` (`:89`); identity gate at `:465-473` |
| `tests/test_serve_corpus.py` | fixture test of the consumer | no | **must follow** literal `23` (`:44`) |
| `tools/bench/README.md:161` | prose "schema-v23 serving records" | no | **must follow** prose |
| `docs/serving.md:960` | prose "schema-v23" | no | **must follow** prose |
| `tests/test_request_log.cpp` | unit tests of the formatters | yes — asserts the JSONL key (`:441`) and sets it in the fixture (`:422`) | rename the key; new field needs an assertion |
| `tools/bench/run_serve_concurrency.py` | writes the log (`:308`), parses `request_done` results: `prompt_tokens`/`completion_tokens`/`computed_prefill_tokens` (`:554-560`), `finish_reason` (`:714`) | no | none (it gates no server-log schema version; its own `SCHEMA_VERSION = 4` at `:48` is the concurrency *report* schema) |
| `tools/bench/run_serve_ttft_campaign.py` | writes the log (`:285`), records the path as an artifact (`:441, 486-487, 530`); parses no events | no | none |
| `tools/bench/ttft/report.py` | validates/persists the `request_log_jsonl` artifact path (`:584-600, 644-645`); parses no lines | no | none |
| `tools/bench/ttft/render.py` | renders the `request_log_jsonl` field (`:285, 315`); parses no lines | no | none |
| `tools/e2e/chat_parsing_e2e.sh` | writes the log as evidence (`:212`); its assertion script reads the opencode session JSONL, not the serve log | no | none |
| `tools/smoke/serve_thinking_preservation.py` | writes the log (`:292`), parses `result.prefix_reuse_path`/`prefix_cache_hit_tokens`/`prompt_tokens` (`:188-195, 231-241`) | no | none |
| `tests/README.md:20` | describes the corpus-consumer test | no | none |

### Out-of-repo

- `nixos-configs/users/kido/llm.nix:540-541` — the deployed
  `ninfer-yarn-qwen38-27b` unit **writes** the log to
  `/home/kido/trash/temp/ninfer-yarn-log.jsonl` (live file, ~38 MB, actively
  appended by the deployed v23 server). It is the only nixos-configs
  reference to the log: a writer, not a parser. Nothing in nixos-configs
  parses the JSONL (repo-wide grep for `ninfer-yarn-log|model_thinking|
  reasoning_tokens` matches only the config itself, `CHANGELOG.md:36`, and a
  doc note).
- `nixos-configs/CHANGELOG.md:36` — documents intended use: "the request log
  is for cache-miss attribution (`prefix_cache_hit_tokens`,
  `prefix_reuse_path`, materialization search counters)". None of the named
  fields is a thinking counter.
- `nixos-configs/docs/qwen38_27b_config/ninfer_neroued_ninfer_nvfp4.md:169` —
  mentions the `--request-log-jsonl` flag.
- `nixos-configs/.scratch/graphiti/research/structured-output-probe-ninfer-yarn.md:82-128` —
  reads the API `completion_tokens_details.reasoning_tokens` from probe
  responses (the realized count via the API usage chunk, not the request log).
- `/home/kido/trash/temp/` — the log file itself plus
  `ninfer_parser_error.md` (a 2026-09-23 tool-call demotion diagnosis that
  cites serve request IDs but does not parse the JSONL). **No analysis
  scripts found**: a bounded search (depth ≤ 2, `*.py/*.sh/*.md/*.ipynb`) for
  `ninfer-yarn-log` or `model_thinking` returned no matches, and there are no
  top-level `*.py` files. (An unbounded recursive scan of the whole directory
  timed out at 120 s; the bounded searches above cover the named spots.)

**Conclusion:** no in-repo or out-of-repo consumer reads
`model_thinking_tokens` from the JSONL, so the rename breaks no reader; the
five surfaces that must follow the schema bump are exactly the #154 set
listed under Q2.

## Q4 — Rename surface (`model_thinking_tokens`)

All 15 occurrences in the worktree (repo-wide grep; no misses):

| Location | Role |
|---|---|
| `include/ninfer/types.h:725` | public field `ThinkingBudgetStats::model_thinking_tokens` (struct at `:722-729`; comment: "Model-origin tokens accepted while capped thinking remained open") |
| `apps/cli/main.cpp:167` | CLI display `print_metric("model thinking tokens", …)` in `print_generation_summary` — gated on `configured_budget` (`:164-170`); displays the budget tracker, not the realized `result.reasoning_tokens` |
| `src/serve/request_log.cpp:571` | JSONL writer key `result.model_thinking_tokens` |
| `src/serve/operational_log.cpp:285` | operational stderr `thinking N/M` clause in `render_request_done` — gated on `configured_budget` (`:283-290`) |
| `src/models/qwen3_5/frontend/output_session.cpp:157` | `ThinkingSessionState::model_thinking_tokens` (internal state struct `:155-161`) |
| `src/models/qwen3_5/frontend/output_session.cpp:517-518` | preview-path increment **gated on a configured budget** (`:516`: `if (impl_->preview_thinking.budget && in_reasoning)`) plus budget-overflow throw |
| `src/models/qwen3_5/frontend/output_session.cpp:570` | budget-exact check that arms `control_pending` |
| `src/models/qwen3_5/frontend/output_session.cpp:584, 588` | `model_token_budget_remaining` — caps remaining model tokens at `budget - model_thinking_tokens` |
| `src/models/qwen3_5/frontend/output_session.cpp:716` | `thinking_stats()` → `ThinkingBudgetStats` publication |
| `tests/test_request_log.cpp:422` | fixture value in the `request_done` test |
| `tests/test_request_log.cpp:441` | assertion on the JSONL key |
| `tests/models/qwen3_5/test_frontend.cpp:1888` | assertion on `ThinkingBudgetStats` from a session (budget=2 case) |
| `eval/corpora/perplexity-1m/data/ninfer/01.txt:2145` | model-generated corpus text that quotes the CLI source verbatim (see `:2140-2150`); confirmed the **only** hit under `eval/corpora/` (the 15-match repo-wide grep above). Not a consumer; leave untouched |

**Docs naming it: none found.** Grep for `model_thinking_tokens` across
`docs/` and `README.md`: zero matches. `docs/serving.md:971` describes the
`request_done` event generically as containing "thinking-budget application
counters"; `docs/cli.md:54-78` documents the `--thinking-budget` option
semantics without naming any counter.

**Semantic note for the rename:** the two counters measure different things.
`model_thinking_tokens` increments only while a thinking budget is configured
(`output_session.cpp:516`); the realized `reasoning_tokens` increments on
every in-reasoning accepted token regardless of budget
(`output_session.cpp:514-515`). In the request log today,
`model_thinking_tokens` is therefore 0 for every request without a budget —
including requests where the model produced substantial reasoning — which is
the #152 gap. (The budget tracker's name is also a misnomer for its value: it
counts model-origin tokens *toward* the budget, which is exactly what the
proposed `budget_thinking_tokens` name would say.)

## Q5 — Docs coverage

- **README.md**: does not describe the request-log `result` fields or the
  thinking counters. No "request log"/"jsonl"/measurement-log text at all;
  "thinking" appears only as the `--preserve-thinking` CLI flag
  (`README.md:85, 226`) and capability notes (`:176, 233`).
- **docs/cli.md**: documents `--thinking-budget` semantics (`:54-78`) and the
  option row (`:222`) — but not the CLI summary's metric labels (so "model
  thinking tokens" at `apps/cli/main.cpp:166` is undocumented) and not the
  request log.
- **docs/serving.md**:
  - "Structured request log" section (`:953-1046`): the `request_done` row
    (`:971`) lists "thinking-budget application counters" without naming the
    fields; the `tool_call_parse` field list (`:978-986`) is explicit. The
    schema version prose is at `:960`. A new `reasoning_tokens` result field
    and the rename both belong in this section (row `:971` and a field-level
    note).
  - The realized count **is** documented for the Responses API usage chunk:
    `"output_tokens_details": {"reasoning_tokens": 5}` (`:657`) with the
    semantics "`reasoning_tokens` is counted in the Qwen output decoder while
    accepted tokens are still in the reasoning channel; it is not estimated by
    re-tokenizing decoded text" (`:664-666`). No equivalent usage-field
    documentation was found for the chat route's
    `completion_tokens_details.reasoning_tokens` or the Anthropic
    `thinking_tokens` (grep for `completion_tokens_details` in `docs/`: no
    matches) — a pre-existing gap, not a #152 one.
- **docs/maintainer/logging.md**: architecture and verification policy only —
  "The corpus consumer test protects its exact schema-version agreement with
  Serve" (`:169-173`); no field names.
- **docs/perplexity.md**: nothing on thinking counters or the request log
  (grep: no matches).
- **tools/bench/README.md:161**: prose "schema-v23 serving records" (bump
  follower, see Q2).

## Search log (for "none found" claims)

- `grep -rn model_thinking_tokens` (whole worktree): 15 matches, all listed in Q4.
- `grep -rn reasoning_tokens` (whole worktree): 25 matches; serve-facing ones listed in Q1.
- `grep -rn "request_done|format_request_done_json|log_request_done"` (`src/**/*.cpp`): single write funnel per Q1.
- `grep -rln "request_log|request-log|ninfer_serve_request_log"` over `tools/ tests/ docs/ bench/ README.md`: consumer table in Q3.
- `grep -n "scor"` in `src/serve/`: no matches; `grep -rn CausalScor|causal_scoring` repo-wide: Q1's causal-scoring finding.
- `git log -L23,23:src/serve/request_log.h` and `git log -S kRequestLogSchemaVersion -- src/serve/request_log.h`: Q2's bump history.
- nixos-configs: `grep -rn "request_log|request-log|requests.jsonl|jsonl" --include=*.nix` → only `users/kido/llm.nix`; `grep -rn "ninfer-yarn-log|model_thinking|reasoning_tokens"` → `llm.nix:541`, `CHANGELOG.md:36`, `docs/qwen38_27b_config/ninfer_neroued_ninfer_nvfp4.md:169`, `.scratch/graphiti/research/structured-output-probe-ninfer-yarn.md`.
- `/home/kido/trash/temp`: depth-2 search of `*.py/*.sh/*.md/*.ipynb` for `ninfer-yarn-log|model_thinking` → no matches; top-level `*.py` → none; `ninfer_parser_error.md` read (diagnosis report, not a log parser). Unbounded recursive scan timed out (directory contains large binary/media subtrees) and is not claimed covered.
