# llama.cpp QoL & tooling candidates vs NInfer — survey

Ticket: #172 (map #167, `kido5217/ninfer-yarn`).

## Scope and method

**Question.** Which llama.cpp QoL and tooling features are port candidates for NInfer: CLI
ergonomics (interactive mode, per-token/per-request stats shape, prompt flags), prompt-template
management, perplexity/eval tooling, benchmarking (llama-bench vs `ninfer_bench`), tokenizer
tools, log/diagnostic ergonomics, and model/config introspection tooling.

**llama.cpp reference.** Live master verified 2026-10-03 via
`gh api repos/ggml-org/llama.cpp/commits/master`: **`eec18f5d32099fb15d4ba15003a231bcc72757d5`
(2026-10-03T17:08:47Z, "vendor : update cpp-httplib to 0.59.0 (#29886)")**. A blobless clone was
fetched to this tip; the delta vs the sibling tickets' reference `1537a0a8b2` is
`scripts/sync_vendor.py` + `vendor/cpp-httplib/*` only (diff-verified: 3 files, no surface
impact), so all sibling findings at `1537a0a8b2` carry over unchanged. Every llama.cpp claim
below is cited as (`eec18f5d32:<path>`), read from the clone at that commit.

**NInfer reference.** Primary repo at **`3e0ff840`** (master, "feat(cli): report realized
reasoning tokens in the generation summary (#162)"), read-only. Verified surfaces:
`README.md`, `docs/cli.md`, `docs/perplexity.md`, `docs/serving.md`, `tools/README.md`,
`bench/README.md`, `apps/{cli,perplexity,serve}` (flag definitions in source),
`src/serve/serve_options.cpp`, `src/serve/http_server.cpp`, `src/serve/operational_log.h`,
`include/ninfer/{engine.h,types.h}`, `tools/artifact/{inspect.py,reader.py,writer.py,schema.py}`,
`tools/bench/`, `tools/fetch_froggeric_template.py`.

**Verdict criteria (user-fixed).** Rank by (a) exposed contract/protocol gap >
(b) measurable 5090 performance > (c) QoL/ecosystem parity; this family lives on axis (c), but a
contract-slot gap still outranks pure convenience. Verdicts: **port** (names vehicle + evidence
needed) / **defer** / **no**, each with a one-line rationale. Boundary-crossing items are flagged,
not verdicted.

**Sibling cross-references** (their verdicts are not re-derived here): #170 serve/API
(`docs/research/llmcpp-port-serve.md` @ `research/llmcpp-serve` dc60901a), #168 model/artifact
(`llmcpp-port-model.md` @ 46478763), #171 sampling (@ 4ff1f210), #173 performance (@ 0cafbf67),
#169 chat-drift (@ 7e49db57).

---

## C1 — `--version` on the product binaries

- **llama.cpp state:** `--version` ("show version and build info") is registered for every
  common tool and prints version + git commit + build flags via
  `llama_print_build_info` (`eec18f5d32:common/arg.cpp:1451-1456`,
  `common/build-info.cpp.in`).
- **NInfer state (verified):** none of `ninfer-yarn`, `ninfer-yarn-serve`,
  `ninfer-yarn-perplexity`, or `ninfer_bench` accepts `--version`
  (`3e0ff840:apps/cli/options.cpp:114-120`, `src/serve/serve_options.cpp:119-136`,
  `apps/perplexity/main.cpp:78-82`, `bench/README.md` CLI section). `project(ninfer)` declares
  no VERSION and no commit-string mechanism exists in `CMakeLists.txt`.
- **Port shape + cost:** add a configure-time version/commit string (CMake configure_file) and a
  `--version` flag to the four binaries. **Trivial** (< 0.5 d). Vehicle: new flag on all product
  binaries; one-line doc note.
- **Verdict:** **port (c).** Every tool is probed with `--version` by scripts and humans;
  absence is the most basic ecosystem-parity gap in the survey. Evidence: `--version` output
  (name, version, git sha, build config) on all four binaries.

## C2 — Interactive mode and interactive commands

- **llama.cpp state:** `llama-cli` runs an interactive chat loop: banner + commands
  `/exit`, `/regen`, `/clear`, `/read <file>`, `/glob <pattern>`, `/image|/audio|/video <file>`
  (`eec18f5d32:tools/cli/cli-context.cpp:415-600`); flags `-i/--interactive`,
  `-if/--interactive-first`, `-mli/--multiline-input` (`common/arg.cpp:1922-1965`);
  `-o/--output` writes the User/Assistant transcript to a file
  (`common/arg.cpp:3146-3151`, `tools/cli/cli-context.cpp:175, 343-346`); `--simple-io`
  disables console decoration for subprocesses (`common/arg.cpp:3820-3826`).
- **NInfer state (verified):** the CLI is one-shot by design — "runs one request against one v3
  `.ninfer` artifact" (`3e0ff840:docs/cli.md:1-4`); one invocation owns exactly one request and
  disables the context cache (`apps/cli/main.cpp:283-287`). Multi-turn conversations are the
  serve's job (three protocol routes; the deployment's agent harness is the conversation client).
- **Port shape + cost:** an in-process loop of `prepare/submit/wait` over stdin, transcript
  file, media commands — **large** (new CLI state machine, media staging, stdin protocol).
- **Verdict:** **no.** One-request-per-process is the product shape of the CLI; interactive
  multi-turn duplicates the serve contract that the deployment already consumes, and
  `--simple-io`-style subprocess concerns are already met by the CLI's plain-text stdout.

## C3 — Constrained decoding (grammar) exposure on the CLI

- **llama.cpp state:** `-j/--json-schema`, `-jf/--json-schema-file`, `--grammar`,
  `--grammar-file` are common to `llama-cli` and `llama-server`
  (`eec18f5d32:common/arg.cpp:2268-2296`).
- **NInfer state (verified):** the public Engine route carries
  `RequestOptions` → `GrammarConstraint` (GBNF + `thinking_enabled`)
  (`3e0ff840:include/ninfer/types.h:269-281`); the serve exposes `grammar` +
  `response_format` (memory #392; `docs/serving.md` "Constrained output"); **the CLI has no
  grammar flag** (`apps/cli/options.cpp:114-239` — full flag set, no grammar entry).
- **Port shape + cost:** `--grammar FILE` on `ninfer-yarn`: read GBNF text, set
  `RequestOptions.grammar` with the request's resolved thinking flag; reuse the Engine's
  compile/validate and the serve's error mapping for the new CLI error surface. **Small**
  (0.5–1 d incl. tests + docs).
- **Verdict:** **port (a).** Constrained decoding is a delivered capability on two of the three
  public Engine routes; the CLI is the third public route and has no slot for it — a contract
  slot gap, which outranks pure QoL per the ticket's ranking. Evidence: a CLI generation under
  a GBNF constraint produces schema-valid output; the same grammar through CLI and serve yields
  consistent constraint behavior; invalid grammar is rejected with the same error class the
  serve maps.

## C4 — Prompt-from-file / stdin input flags

- **llama.cpp state:** `-f/--file` ("a file containing the prompt"), `-sysf`, `--in-file`,
  `-bf/--binary-file`, `-e/--escape` (escape-sequence processing), `--stdin` (tokenize tool;
  takes precedence over `-f`/`-p`) (`eec18f5d32:common/arg.cpp:1801-1865, 3220-3228`).
- **NInfer state (verified):** `--prompt` accepts inline text only; structured input goes
  through `--messages FILE` (`3e0ff840:apps/cli/options.cpp:131-136`). There is no way to feed a
  plain-text prompt from a file or stdin without shell-quoting it (`docs/cli.md:8-17`).
- **Port shape + cost:** `--prompt-file FILE` (and optionally `--prompt-stdin`) on the CLI;
  read verbatim (document the escape semantics or keep them absent — NInfer prompts have none
  today). **Trivial** (< 0.5 d).
- **Verdict:** **port (c).** Long/awkward prompts (code, CJK, newlines) hit shell-quoting pain
  in exactly the one-shot CLI; the messages route already proves file input is the expected
  pattern. Evidence: byte-identical token counts for `--prompt X` vs `--prompt-file` of the same
  content; one docs example.

## C5 — Rendered-prompt dump (template debugging)

- **llama.cpp state:** `--verbose-prompt` ("print a verbose prompt before generation", CLI +
  completion + embedding + retrieval; prints prompt tokens,
  `eec18f5d32:common/arg.cpp:1485-1491`, `tools/completion/completion.cpp:379, 418-443`);
  `--display-prompt` ("whether to print prompt at generation", CLI,
  `common/arg.cpp:1492-1499`); server `--log-prompts-dir PATH` writes the rendered prompt of
  every request to a timestamped `.txt` "only used for debugging"
  (`common/arg.cpp:3895-3905`, `tools/server/server-context.cpp:4500-4502`).
- **NInfer state (verified):** no rendered-prompt dump exists anywhere — not at `--log-level
  debug` (grep for prompt-text logging in `src/product`, `src/models/qwen3_5/frontend`:
  none). The public `PreparedPrompt` exposes only `summary()` (token counts,
  `3e0ff840:include/ninfer/types.h:548-552`) and `preparation_stats()`
  (`include/ninfer/engine.h:12-33`), so the text is not currently reachable from the CLI.
- **Port shape + cost:** (i) small public API addition (e.g. `PreparedPrompt::rendered_text()`)
  + CLI flag `--show-prompt` printing the rendered prompt to stderr before streaming (stdout
  stays the answer channel); (ii) serve flag `--log-prompts-dir PATH` writing per-request
  rendered prompts. (i) **small** (0.5 d); (ii) **small** (0.5 d).
- **Verdict:** **port (c)** for the CLI flag; **defer (c)** for the serve directory. Chat
  template debugging has been this project's recurring pain (G1 demotions, chat-drift,
  froggeric template pinning); a one-flag dump of exactly what the model sees is the direct
  tool for it. Evidence: `--show-prompt` output matches an independent Jinja render of the same
  messages; stdout byte stream unchanged with the flag set.

## C6 — Per-token / per-request stats shape (`--show-timings`, `--perf`, `-ptc`)

- **llama.cpp state:** `--show-timings` prints `[ Prompt: X t/s | Generation: Y t/s ]` after
  each CLI turn (`eec18f5d32:common/arg.cpp:1793-1799`,
  `tools/cli/cli-context.cpp:655-661`); `--perf` enables internal libllama performance counters
  reported at exit (`common/arg.cpp:1784-1791`); `-ptc/--print-token-count N` prints the token
  count every N tokens (completion example only, `common/arg.cpp:1860-1866`);
  `--show-statistics` is imatrix-only (`common/arg.cpp:3199-3205`).
- **NInfer state (verified):** the one-shot CLI report already prints stage timings
  (render/preprocess, vision, prefill, decode, total), prompt/generated/reused token counts,
  prefill/decode/overall rates, memory arenas, sampling, finish reason, realized thinking, and
  speculative stats (`3e0ff840:apps/cli/main.cpp:146-230`). llama.cpp's CLI has no per-token
  timing either; its per-turn rate line is the last line of what NInfer prints at the end.
- **Verdict:** **no.** Parity verified — NInfer's end-of-run report is a superset of the
  observable llama.cpp CLI stats shape for a one-request tool.

## C7 — CLI configuration presets

- **llama.cpp state:** INI-based preset system: named option sets loadable from a file, the HF
  model cache, or a models dir; `--models-preset PATH` runs the server in router mode from a
  preset file (`eec18f5d32:common/preset.h:1-80`, `common/arg.cpp:751-760, 3649-3670`).
- **NInfer state (verified):** no preset mechanism; repeated flag sets are codified in the
  Python bench orchestrator (`tools/bench/run_ninfer_bench_matrix.py --preset core`) and the
  deployment (nixos-configs) rather than in the binaries.
- **Port shape + cost:** an INI/JSON file of CLI/serve flags applied at startup, per binary.
  **Small–medium**; low reuse value in a one-owner, two-call-site project.
- **Verdict:** **defer (c).** The two real consumers (bench matrix, deployment) already own
  their flag sets; revisit only if manual flag repetition becomes observed pain.

## C8 — Shell completion

- **llama.cpp state:** `--completion-bash` prints a source-able bash completion script per
  binary (`eec18f5d32:common/arg.cpp:1471-1477`); usage documented in
  `docs/completions.md` (bash only; `tools/completion/` is the legacy completion example whose
  README is a stale copy of the old completion-app README).
- **NInfer state (verified):** no completion support.
- **Port shape + cost:** a generated or hand-maintained bash script for the four binaries
  (flag names are long-form only — a small static table). **Trivial**.
- **Verdict:** **defer (c).** Ecosystem parity at near-zero cost, but near-zero value for a
  one-owner box; a five-minute job if ever wanted.

## C9 — CLI as a client of a running server (`--server-base`)

- **llama.cpp state:** `llama-cli --server-base URL` runs the interactive CLI against a live
  `llama-server` instead of loading a model (health/completion/SSE via
  `eec18f5d32:common/arg.cpp:1478-1483`, `tools/cli/cli-client.cpp:17-125`).
- **NInfer state (verified):** the CLI always loads the Engine in-process
  (`3e0ff840:apps/cli/main.cpp:272-307`); serve clients are the deployment (opencode) and
  direct HTTP (smoke tools, `tools/smoke/`).
- **Verdict:** **no.** The serve's real conversation client is the agent harness; a second
  thin client duplicating the OpenAI routes has no consumer in a one-model/one-GPU product.

## C10 — Chat-template zoo (inline/named templates, kwargs, parsing toggles)

Out-of-family flag from #170 ("chat-template zoo — server-side `--chat-template`
loading/management beyond the vendored chat params"); #170 routed the frontend aspects to the
chat-parsing family. The tooling/QoL aspects are surveyed here; frontend-semantics items are
cross-referenced, not re-adjudicated.

- **llama.cpp state:** `--chat-template JINJA_TEMPLATE` (inline text) and
  `--chat-template-file` (file), both accepting named built-in templates from
  `LLM_CHAT_TEMPLATES` (54 non-Jinja templates) unless `--jinja` is set first
  (`eec18f5d32:common/arg.cpp:3750-3773, 1322-1336`, `src/llama-chat.cpp:28-83, 950-957`);
  `--chat-template-kwargs` (JSON extra params; `enable_thinking`/`preserve_reasoning` are
  deprecated in favor of dedicated flags, `common/arg.cpp:3534-3545`); `--reasoning-format
  none|deepseek|deepseek-legacy` (`common/arg.cpp:3679-3689`); `--skip-chat-parsing` (force a
  pure-content parser even with a Jinja template, `common/arg.cpp:3774-3784`);
  `--prefill-assistant` (`common/arg.cpp:3785-3791`).
- **NInfer state (verified):** `--chat-template FILE` (file only) on CLI and serve
  (`3e0ff840:apps/cli/options.cpp:133-134`, `src/serve/serve_options.cpp:84`); templates are
  artifact resources (`tools/chat_templates/qwen3_{6,8}.jinja`); thinking control is explicit
  flags (`--no-thinking`, `--reasoning-effort`, `--thinking-budget`, `--preserve-thinking`);
  `--raw-output` exposes the raw output channel.
- **Verdicts (per item):**
  - **inline `--chat-template` text (vs file): no.** The file route covers the workflow —
    templates are committed resources (`tools/chat_templates/`, artifact `--resource`), and
    experiments edit a file; inline text on a startup flag adds ambiguity without a consumer.
  - **named built-in template zoo: no.** Qwen-family product; templates belong to the artifact,
    and a zoo of 54 other families' templates is out of purpose (cross-ref #169/#170: the
    vendored chat stack is the productized path).
  - **`--chat-template-kwargs`: no.** NInfer's explicit thinking flags are a stronger contract
    than raw template kwargs (llama.cpp itself deprecates the two relevant kwargs in favor of
    dedicated flags).
  - **`--reasoning-format`: no.** The Qwen wire format (reasoning channel + control machinery)
    is fixed by the product; llama.cpp's deepseek formats serve other families.
  - **`--skip-chat-parsing`: no** (frontend family — cross-ref #170 flag; the product is a
    chat-first parser, and `--raw-output` already exposes the unparsed output channel).
  - **`--prefill-assistant`: boundary** — already in #170's defer band ("chat prefill").

## C11 — Model-acquisition convenience flags (download / cache / offline)

Out-of-family flag from #170 ("model download/delete/`--no-warmup`-class convenience flags",
routed to the converter/tooling family).

- **llama.cpp state:** `-hf/-hfr/--hf-repo <user>/<model>[:quant]`, `-hff/--hf-file`,
  `-hft/--hf-token`, `-mu/--model-url`, `-dr/--docker-repo`, `--offline` (force cache, no
  network), `-cl/--cache-list` (list cached models), `--mtp/--dflash/--eagle3` (fetch the
  matching draft model), mmproj URL variants; downloads land in the HF hub cache with progress
  callbacks (`eec18f5d32:common/arg.cpp:3035-3098, 3932-3938, 1459-1468, 2586-2594`,
  `common/download.cpp`, `common/hf-cache.cpp`). Server-side multi-model management
  (`--models-dir/--models-preset/--models-max/--models-autoload`, `GET/POST/DELETE /models*`) is
  the router — boundary item, see #170 candidate 14.
- **NInfer state (verified):** no in-product download — `README.md` quick start instructs
  `hf download neroued/Qwen3.8-27B-nvfp4-NInfer …` with the external HF CLI; artifact paths are
  local-only on all binaries. Precedent for a narrow fetch-with-verification helper:
  `tools/fetch_froggeric_template.py` (sha256-gated download, `3e0ff840`). `libcurl >= 7.85`
  is already a product dependency (`README.md` quick start). llama.cpp exposes **no CLI model
  deletion** (only router-side `DELETE /models` for loaded models) — nothing to port there.
- **Port shape + cost:** accept an HF repo reference in the artifact argument (or a
  `--hf-repo`-class flag) on `ninfer-yarn`/`ninfer-yarn-serve`: resolve
  `<repo>:<file>` → download via libcurl to a local cache dir with a progress line, reuse
  existing files, fail closed when the file is absent and offline. **Small–medium** (2–4 d:
  download + progress + cache + failure modes + tests + docs). Vehicle: new CLI/serve
  argument form (a Python `tools/` fetcher is the cheaper alternative if C++ product surface is
  preferred to be kept minimal).
- **Verdict:** **port (c).** The README's own quick start is a two-tool dance (hf CLI +
  ninfer), artifacts live on Hugging Face, and the trust model (local, trusted artifacts)
  matches llama.cpp's posture. Evidence: download-then-start e2e on a clean cache dir;
  resume/failure behavior (404, interrupted, offline with cached file); docs update.

## C12 — `--warmup` / `--no-warmup`

- **llama.cpp state:** `--warmup/--no-warmup` ("whether to perform warmup with an empty run",
  default enabled) on CLI/server/perplexity/bench (`eec18f5d32:common/arg.cpp:1967-1974`).
- **NInfer state (verified):** the serve warms up unconditionally at startup and logs it
  (`3e0ff840:apps/serve/main.cpp:68-79`, `src/serve/operational_log.h`); `ninfer_bench` takes an
  explicit `--warmup <n>` repetition count (`bench/README.md` CLI section); the one-shot CLI
  needs none.
- **Verdict:** **no.** Serve warmup is deliberate startup behavior (CUDA-graph capture +
  first-run stabilization); the benchmark already controls warmups explicitly; no consumer
  needs to skip the serve's warmup.

## C13 — Perplexity/eval dataset checks (HellaSwag, Winogrande, multiple choice)

- **llama.cpp state:** `--hellaswag`, `--winogrande`, `--multiple-choice` (+ `-tasks N`)
  compute accuracy (acc_norm) scores; the preprocessed task data is supplied as the prompt
  input (six lines per task, loaded via `-f`/`-p`), with raw datasets fetched by
  `scripts/get-{hellaswag,wikitext-2,winogrande}.sh`
  (`eec18f5d32:common/arg.cpp:2463-2505`,
  `tools/perplexity/perplexity.cpp:726-775`, `tools/perplexity/README.md`). Purpose per the
  README: judging quantized-vs-FP16 quality loss.
- **NInfer state (verified):** capability evaluation runs through the serve with EvalScope
  (AIME/GPQA/ERQA/RealWorldQA, `3e0ff840:README.md` Evaluation section); `ninfer-yarn-perplexity`
  is a fixed-window causal-PPL evaluator over a fixed 16-stream corpus or one file
  (`docs/perplexity.md`), used for KV-format and weight-format comparisons.
- **Verdict:** **no.** Wrong purpose: these suites QA open-weight GGUF quantization, while
  NInfer's artifact QA is owned by the converter, the oracle test contracts, and the EvalScope
  campaign.

## C14 — KL divergence + logit recording

- **llama.cpp state:** `--kl-divergence` with `--save-all-logits/--kl-divergence-base FNAME`
  records full-vocab logits from a reference run (binary, 11–37 GiB for LLaMA-class models on
  Wikitext-2) and computes KLD + probability-delta statistics against the current model
  (`eec18f5d32:common/arg.cpp:2505-2518`, `tools/perplexity/README.md`).
- **NInfer state (verified):** the CausalScoring route (perplexity app) is the same public
  Engine route; the public headers expose NLL-based results only — no logits/logprobs surface
  (`3e0ff840:include/ninfer/engine.h`, `include/ninfer/types.h`: no `logprob`/`logit` member).
  Numerical qualification is owned by the oracle test contracts (AGENTS.md).
- **Port shape + cost:** a logit sink (public API change) + binary recording + the KLD stats —
  **medium** (2–3 d API + tooling), and KV-format PPL comparison (the decision these stats
  serve here) is already possible with the fixed corpus.
- **Verdict:** **defer (c).** Cross-KV-format logit divergence would add evidence to KV-dtype
  decisions, but PPL comparison already drives them and the port costs an Engine API surface;
  revisit only if a KV/weight-format decision needs finer discrimination than PPL provides.

## C15 — Perplexity chunking / output-type / stride knobs

- **llama.cpp state:** `--chunk/--from-chunk` split corpus scoring across invocations
  (`eec18f5d32:common/arg.cpp:3192-3198`); `--ppl-stride` and `--ppl-output-type 0|1` (per-window
  print format, `common/arg.cpp:2519-2531`, `tools/perplexity/perplexity.cpp:431-435`);
  `--context-file/--chunk-size/--chunk-separator/--junk/--pos` are the needle-in-haystack
  passkey test, not perplexity chunking (`common/arg.cpp:3105-3144`,
  `examples/passkey/README.md`).
- **NInfer state (verified):** `--context`, `--stride`, `--quick` (one stream per domain),
  `--corpus` manifest or `--text` file; per-window unrounded values in `report.json`
  (`3e0ff840:docs/perplexity.md`, `apps/perplexity/main.cpp:92-133`).
- **Verdict:** **no.** Parity on the knobs that matter (context/stride per-window output); the
  chunking sugar and passkey test have no NInfer consumer (scale control is the corpus/quick
  design).

## C16 — Benchmarking: llama-bench vs `ninfer_bench`

- **llama.cpp state:** `llama-bench` measures pp/tg(+pg) matrices with `-p/-n/-pg/-d`
  (n_depth: extra context headroom, `n_ctx = n_prompt + n_gen + n_depth`,
  `eec18f5d32:tools/llama-bench/llama-bench.cpp:460-467, 1308`), `-b` (batch/concurrent
  requests), `-ub`, KV cache types, range syntax (`first-last`, `first-last+step`,
  `first-last*mult`), `-r`, `--delay`, `--prio`, `--no-warmup`, and output in
  csv/json/jsonl/markdown/sql (`llama-bench.cpp:437-441, 495-497, 1023-1046`).
- **NInfer state (verified):** `ninfer_bench` measures the complete public Engine route:
  `pp{P}`, `tg{G}`, `pp{P}+tg{G}` matrices, `-p/-n/-pg` lists, `-r`, `--warmup <n>`,
  `--grammar` constrained mode, `--kv-dtype`, `--spec/--draft-tokens/--lm-head-draft`,
  `--max-ctx`, `--prefill-chunk`, `--no-cuda-graph`, `--profile-measured`, output
  table/json/csv, schema v15 (`3e0ff840:bench/README.md`). The concurrency dimension is measured
  by the separate serve bench tools (`tools/bench/run_serve_concurrency.py`; README saturation
  tables) by design — `ninfer_bench` is the single-request Engine route.
- **Gaps:** batch (`-b`), ubatch, threads/offload axes (out of family — single-request CUDA
  product); range syntax, jsonl/markdown/sql output, `--delay` (sugar); `-hf` download (C11).
- **Verdict:** **no** for the core (coverage of the single-request scope is complete and the
  Engine-level report is richer — spec/KV/grammar modes llama-bench lacks); **defer (c)** for
  the sugar (range syntax in `-p/-n`, jsonl output, `--delay`) — trivial but no consumer.

## C17 — Tokenizer CLI (`llama-tokenize`)

- **llama.cpp state:** encode-only tool: `-p`/`-f`/`--stdin` input, prints `id -> 'piece'`
  lines or a Python-parseable `--ids` list, `--show-count`, `-pps/--parse-special`,
  `--no-bos`, escape handling; loads the model with `vocab_only` (no weights)
  (`eec18f5d32:tools/tokenize/tokenize.cpp:24-28, 95-230`).
- **NInfer state (verified):** no tokenizer tool. Token visibility: `--print-token-ids` on
  generated tokens only (`3e0ff840:apps/cli/main.cpp:310-317`); serve token counting exists for
  Responses + Anthropic routes (`POST /v1/responses/input_tokens`,
  `POST /v1/messages/count_tokens`, `src/serve/http_server.cpp:447, 471`) but chat
  completions has no counting endpoint and no route prints token IDs. Serve-side
  `/tokenize`/`/detokenize`/`/apply-template` endpoints were surveyed by #170 (candidate 7,
  verdict **defer (c)**) and are not re-adjudicated here.
- **Port shape + cost:** a small product binary `ninfer-yarn-tokenize` (text → token IDs with
  the artifact's tokenizer; `--ids` list + piece view + `--count`; input via `-p`/`-f`/stdin).
  The perplexity route already loads Text weights+resources from an artifact
  (`docs/perplexity.md`); a tokenizer-only load path would be the fast variant — either reuse
  the Engine text-only load or add tokenizer-only access. **Small** (1–2 d incl. docs).
- **Verdict:** **port (c).** Token-boundary visibility is the daily debugging primitive for
  prompt framing, template drift, and token-budget questions (all active areas in this project),
  and no existing surface answers "how does this text tokenize?". Evidence: round-trip
  encode/decode spot checks on the corpus + CJK; token counts agree with the serve
  `count_tokens` endpoints for identical messages; a `docs/` section.

## C18 — Log/diagnostic ergonomics (JSONL ops log, file log, log toggles)

- **llama.cpp state:** `--log-jsonl` (JSONL operational log to stdout), `--log-file`,
  `--log-disable` (pause at runtime), `--log-colors/--log-prefix/--log-timestamps`,
  `-lv/--verbosity N` (`eec18f5d32:common/arg.cpp:3873-3970`).
- **NInfer state (verified):** `--log-level trace|debug|info|warning|error|critical|off` on the
  CLI, serve, and perplexity binaries (spdlog, stderr-only human logs; `ninfer_bench` has no
  log-level flag — its full option set is
  `bench/inference/ninfer_bench_support.cpp:282-313`); the serve writes machine-readable full-precision
  request records via `--request-log-jsonl` (schema v24) and human operational lifecycle events
  (warmup/ready/request/throughput) via the operational log
  (`3e0ff840:src/serve/operational_log.h`, `tools/README.md`).
- **Verdict:** **defer (c).** The machine-readable need is already served by the request JSONL;
  a JSONL *operational* stream and file logging have no current consumer (shell redirection
  covers files; the deployment reads the request log). `--log-disable` is **no** (one-shot CLI,
  nothing to pause).

## C19 — `--check-tensors` (invalid-value scan)

- **llama.cpp state:** scan model tensors for invalid values at load
  (`eec18f5d32:common/arg.cpp:2928-2933`). Purpose: catching corrupted GGUF downloads.
- **NInfer state (verified):** the artifact is validated at load (structure, framing, inventory,
  `3e0ff840:docs/cli.md:98`); download integrity has no check today (see C22).
- **Verdict:** **no.** Corrupt-artifact detection belongs on the download/integrity path (C20's
  hash manifest catches bit-rot before load); a full-tensor NaN scan on every startup of a
  27B-class artifact is not a product behavior.

## C20 — Artifact integrity hashing (llama-gguf-hash equivalent)

- **llama.cpp state:** `llama-gguf-hash` hashes GGUF files per tensor layer + whole file
  (xxh64/sha1/sha256), emits checkable manifests, verifies with `--check`, and derives UUIDv5
  content IDs (`eec18f5d32:examples/gguf-hash/README.md`, `gguf-hash.cpp`).
- **NInfer state (verified):** `artifact_id` is a **random `uuid4()`** minted at conversion
  (`3e0ff840:tools/artifact/writer.py:141`) — no content integrity in the artifact;
  `tools/artifact/inspect.py` has no hash mode. Artifacts are downloaded from Hugging Face
  (`README.md` quick start) and upgraded v2→v3 to a new output path
  (`docs/weight-conversion.md`).
- **Port shape + cost:** extend `tools/artifact/inspect.py` with `--hash` (whole-entry +
  per-object sha256, manifest lines) and `--check MANIFEST` (verify). Stdlib-only Python,
  streaming over the reader's object ranges. **Small** (0.5–1 d).
- **Verdict:** **port (c).** Downloaded artifacts need an integrity story that a random UUID
  does not provide, and v2→v3 upgrades should prove the payload survived; a per-object manifest
  also localizes corruption (mirroring gguf-hash's design rationale). Evidence: manifest +
  verify round-trip on an official artifact; a 1-byte perturbation is detected and localized to
  the object.

## C21 — Artifact introspection parity (llama-gguf print)

- **llama.cpp state:** the `gguf` example prints version/alignment/data offset, the full KV
  metadata, and a per-tensor table (name/dims/type), with key lookup
  (`eec18f5d32:examples/gguf/gguf.cpp:86-140` tensor name/size/offset in `gguf_ex_read_0`;
  type/dims in `gguf_ex_read_1`, `:151-230`).
- **NInfer state (verified):** `tools/artifact/inspect.py` prints a richer summary (artifact_id,
  components — including each component's **config**, a required directory member that lands in
  the default output via `artifact_summary`'s `"components": artifact.directory.components`
  (`tools/artifact/inspect.py:21`, `schema.py:313-320`) — file/payload bytes, object/tensor
  counts, formats, layouts) plus `--objects` (offset/bytes/kind/storage per object) and
  `--bindings`/`--json` (`3e0ff840:tools/artifact/inspect.py`). Verified by running the decode
  path: the default summary emits each component's `config` (e.g. `max_position_embeddings`).
- **Verdict:** **no (c), parity.** "What does this artifact say its config is?" is already
  answered by the default summary; the llama-gguf print parity items (KV metadata, per-tensor
  table) are covered at artifact level by `--objects`/`--bindings`. No port.

## C22 — `llama-results` (logits NMSE regression check)

- **llama.cpp state:** writes logits for a prompt to a GGUF file and `--check`s a later run
  against it with normalized MSE tolerance 1e-6 — a cross-commit regression detector
  (`eec18f5d32:tools/results/results.cpp` (NMSE at :12-24), `tools/results/README.md`).
- **NInfer state (verified):** regression protection is the test-suite/oracle contracts
  (AGENTS.md Verification); there is no standalone logits-dump/diff tool, and no public logits
  surface (see C14).
- **Port shape + cost:** a logit dump route (C14's API change) + a Python diff tool under
  `research/`; the interesting use is cross-engine parity (NInfer vs llama.cpp GGUF logits on
  the same Qwen weights). **Medium**.
- **Verdict:** **defer (c).** The engine's numerics are oracle-qualified; a cross-engine logits
  parity check is a one-off research artifact, not standing tooling — build ad hoc if a specific
  numerical question demands it.

## C23 — Tensor/activation dump tooling

- **llama.cpp state:** `-o/--output FNAME` (gguf/dat) + `--output-format`,
  `--save-frequency`, `--process-output` dump intermediate tensors/activations during a run
  (`eec18f5d32:common/arg.cpp:3146-3177`, imatrix/debug examples).
- **NInfer state (verified):** kernel-level visibility is `ncu`/`nsys` via
  `--profile-measured` and the `bench/ops/` microbenchmarks (`bench/README.md`); no in-product
  tensor dump.
- **Verdict:** **boundary — no QoL verdict.** This is Op/maintainer debugging machinery, not
  user QoL; if ever needed it is a maintainer tool under `bench/`, routed to the Op family.
  **Flagged for the map.**

## C24 — Micro-flags with no NInfer consumer

Surveyed and closed individually (all `no`):

- `--list-devices` (single-CUDA product; `--device` index suffices —
  `eec18f5d32:common/arg.cpp:2746-2752`).
- `-co/--color`, `-lv` verbosity scale, `--log-prefix/--log-timestamps` (spdlog `--log-level`
  + the stderr/stdout channel split already cover the ergonomics; NInfer CLI/serve).
- `-e/--escape` (no escape semantics exist in NInfer prompts; C4's file input keeps bytes
  verbatim by design).
- `-sys/-sysf` (parity: `--messages` system/developer roles), `-r/--reverse-prompt` (parity:
  `--stop`/`--reasoning-stop`), `-sp/--special` (parity: `--raw-output` /
  `preserve_special_tokens`), `-cnv/-no-cnv/-st/--in-prefix*/--in-suffix/--no-bos` (the
  artifact's template owns framing; NInfer is one-request).
- `--prompt-cache/--prompt-cache-all/--prompt-cache-ro` (completion-example-only cross-run KV
  files, `eec18f5d32:common/arg.cpp:1867-1887`; NInfer's CLI is one-shot by design — reuse is
  the serve's prefix cache; a cross-process KV file would be an Engine context-cache item,
  not QoL).
- `-bf/--binary-file` (all examples except server, `eec18f5d32:common/arg.cpp:1836-1840`),
  `--in-file` (imatrix-only, `common/arg.cpp:1823-1835`): no consumer — NInfer's media input is
  the `--messages` route.

---

## Summary table

| Candidate | Verdict | Axis | One-line rationale |
|---|---|---|---|
| C1 `--version` on product binaries | **port** | c | Most basic ecosystem probe missing on all four binaries; trivial CMake + flag work |
| C2 Interactive mode + commands | **no** | c | One-request CLI is the product shape; multi-turn is the serve's contract |
| C3 CLI grammar exposure | **port** | a | Engine + serve ship constrained decoding; the CLI has no slot for a delivered Engine route |
| C4 Prompt-from-file/stdin | **port** | c | Shell-quoting pain for long prompts; file input is already the structured route's pattern |
| C5 Rendered-prompt dump (CLI `--show-prompt`) | **port** | c | Direct tool for the project's recurring template-debugging pain; small public accessor |
| C5b Serve `--log-prompts-dir` | **defer** | c | Same value, no current consumer; CLI flag covers the debug need |
| C6 Per-token/per-request stats shape | **no** | — | Parity verified: NInfer's end-of-run report is a superset of llama.cpp's CLI stats |
| C7 Config presets | **defer** | c | Bench orchestrator + deployment already own flag sets; no observed pain |
| C8 Shell completion | **defer** | c | Near-zero cost, near-zero value for a one-owner box |
| C9 CLI as server client (`--server-base`) | **no** | c | The serve's conversation client is the agent harness; no second-client consumer |
| C10 Chat-template zoo (inline/named/kwargs/reasoning-format) | **no** (items) | c | Qwen-family artifact templates + explicit thinking flags are the stronger contract; `--prefill-assistant` → #170 defer band |
| C11 Model-acquisition flags (HF download/offline/cache) | **port** | c | README quick start is a two-tool dance; libcurl already a dependency; no delete to port |
| C12 `--warmup/--no-warmup` | **no** | c | Serve warmup is deliberate; bench controls warmups explicitly |
| C13 Dataset eval checks (hellaswag/winogrande/mc) | **no** | c | Quantization-QA purpose; NInfer QA is converter + oracles + EvalScope |
| C14 KLD + logit recording | **defer** | c | Needs a public logits API; PPL comparison already drives the KV-format decisions |
| C15 PPL chunking/output-type knobs | **no** | — | Parity on the knobs that matter; no consumer for the sugar |
| C16 llama-bench vs ninfer_bench | **no** (core) / **defer** (sugar) | c | Single-request coverage complete and richer; range/jsonl/delay sugar has no consumer |
| C17 Tokenizer CLI | **port** | c | Daily token-boundary primitive with no existing NInfer surface; small app on the artifact's tokenizer |
| C18 JSONL ops log / file log | **defer** | c | Request JSONL (v24) already covers machine-readable; no consumer for the rest |
| C19 `--check-tensors` | **no** | c | Integrity belongs on the download path (C20); startup NaN scan is not product behavior |
| C20 Artifact integrity hashing | **port** | c | `artifact_id` is a random UUID; downloaded + upgraded artifacts need verifiable integrity |
| C21 Config dump (gguf print parity) | **no** (parity) | c | The default `inspect.py` summary already prints per-component `config` (verified by running the decode path); object/tensor-level parity is covered by `--objects`/`--bindings` |
| C22 `llama-results` logits NMSE | **defer** | c | Oracle tests own regression; cross-engine parity is one-off research |
| C23 Tensor/activation dumps | **flag** | — | Op/maintainer machinery, not QoL — route to the Op family |
| C24 Micro-flags (color, escape, prompt-cache, …) | **no** | — | No consumer; parity items covered elsewhere |

**Ports (7):** C1, C3 (axis a), C4, C5, C11 (small–medium, 2–4 d), C17, C20 — the rest small.
No candidate on axis (b): this family is QoL by construction, and nothing here changes 5090
performance.

## Flagged for the map (boundary-crossing + out-of-family)

Boundary-crossing (no verdict):

1. **Multi-model router + `/models*` download/delete endpoints** — one GPU, one resident model
   is a fixed product boundary; already flagged at #170 candidate 14 (boundary). C11 ports the
   *CLI-side* convenience only.
2. **Draft/mmproj model downloads** (`--mtp/--dflash/--eagle3 -hfd`, `--mmproj-url`) — model
   acquisition for speculative/multimodal components; NInfer carries those weights in the
   artifact, so this is model-family, not QoL (overlaps C11's vehicle only if a draft-artifact
   flow is ever wanted).
3. **Lookup/retrieval cache** (`-lcs/--lcd`, `examples/retrieval`) — a new feature family
   (retrieval over embeddings); requires an explicit product change.
4. **Group attention** (`-gan/--gaw`) — model/attention family; no Qwen consumer.
5. **Tensor/activation dump** (C23) — Op/maintainer debugging family.
6. **CPU threading/offload/multi-device/RPC/NUMA/tensor-split/load-mode families**
   (`-t/-tb/-C/-Cr/--prio/--poll/-ngl/-sm/-ts/-mg/-lm/-lzm/-kvo/-ot/-cmoe/-ncffn/--numa/--dev/--rpc/
   --op-offload/--repack/--no-host/--defrag-thold`, `tools/rpc`) — single-GPU sm_120a CUDA
   product boundary (product change if ever wanted).
7. **Context/KV-management flags** (`--kv-unified`, `--cache-ram`, `--cache-idle-slots`,
   `--ctx-checkpoints/--checkpoint-min-step`, `--context-shift`, `--swa-full`, `--chunks`,
   `--prompt-cache*`) — engine/context-cache family; #170's flag section routes the same set to
   Engine.
8. **Concurrency/batching** (`-np/--parallel`, `-ns/--sequences`, `-cb`, `tools/batched-bench`)
   — #173 flag 4a concluded llama.cpp's slot model is the closest analogue to NInfer's
   bounded-FIFO rounds with nothing to port; #170 candidate 10 covers the serve-side slots.
9. **TTS + `tools/ui` (Web UI) + server tool execution/MCP** — new mathematical family /
   #170 candidates 15–16 (verdict **no** there); not QoL.
10. **`mtmd` media toolkit** (`tools/mtmd/`) — standalone media-processing CLI; NInfer's media
    path is Engine-internal (serve `--media-*` knobs, #170 candidate 22); route to the
    multimodal area if a standalone tool is ever wanted.

Out of family (tooling seen, owned elsewhere — cross-referenced, not re-adjudicated):

- **Quantization/workflow tools** — `tools/quantize`, `tools/imatrix`, `tools/export-lora`,
  `tools/fit-params`, `tools/cvector-generator`, `tools/gguf-split` → #168 (GGUF source,
  quant families, LoRA/adapters verdicted there; imatrix/cvector/fit-params are the
  quantization workflow's QoL layer).
- **Embeddings/rerank/pooling/attention** (`--embedding`, `--rerank`, `--pooling`,
  `--attention`, `--embd-*`, `POST /embedding`, `POST /reranking`) → #168 C-E boundary flag.
- **Sampling flags** (`--samplers`, `--sampler-seq`, mirostat/dynatemp/XTC/typical/DRY/
  adaptive, `--logit-bias`, `-bs/--backend-sampling`, `--control-vector*`, `--keep`,
  `--ignore-eos`) → #171 (sampler zoo **no**; `min_keep`/`ignore_eos` **defer (c)**).
- **Serve-side knobs** (`--metrics`, `--props`, `--slots*`, `--sse-ping-interval`,
  `--timeout`, `--cors-*`, `--api-key-file`, `--api-prefix`, `--ssl-*`, `--media-path`,
  `--sleep-idle-seconds`, `--reuse-port`, `--threads-http`, `--alias`, `--tags`) → #170
  candidates 8–10, 17, 22 (defer/no as adjudicated there).
- **Serve tokenizer endpoints** (`/tokenize`, `/detokenize`, `/apply-template`) → #170
  candidate 7 (**defer (c)**); C17 is the CLI-side complement.
- **`--simple-io`** — not covered by any sibling doc; serve family. NInfer's CLI stdout is
  already plain text and the serve is JSON-by-design; no consumer.
- **`--spm-infill` / `POST /infill`** — #170 candidate 6 (legacy routes, no band) / a new
  product mode if ever wanted.
- **`--override-kv`, `-ot/--override-tensor`** — artifact/config override territory → #168
  family.
- **Speculative draft zoo** (`--dspark`, EAGLE3, ngram drafts, `--spec-synth-len`,
  `--spec-draft-*`) → #173 flags (draft-family expansion) + C11's model-acquisition note.

## Dead ends / could not verify

- `tools/completion/README.md` is a **stale copy** of the legacy completion example's README
  (describes a chat app, not completion generation); the actual feature is `--completion-bash`
  in `common/arg.cpp` + `docs/completions.md`.
- The ticket's "print-info equivalents" live in `examples/gguf/` (print mode), not
  `tools/` — the current tree has no `tools/gguf` entry.
- `--show-statistics` initially read as "per-token stats"; it is **imatrix-only**
  (`common/arg.cpp:3199-3205`). llama.cpp's CLI has no per-token timing output either.
- llama-bench `-d/--n-depth` is extra context headroom (`n_ctx = n_prompt + n_gen + n_depth`),
  not a decode-depth parameter.
- llama.cpp exposes **no CLI model deletion** (only router-side `DELETE /models` for loaded
  models); the ticket's "download/delete" delete half has no llama.cpp CLI counterpart to port.
- `--prompt-cache` is registered for the **completion example only**, not CLI/server —
  cross-run KV persistence is not a llama.cpp CLI/serve feature at this commit.
- Sibling docs cite `1537a0a8b2`; re-verification showed `eec18f5d32` differs by the
  cpp-httplib vendor bump only, so those line/path citations remain valid.
