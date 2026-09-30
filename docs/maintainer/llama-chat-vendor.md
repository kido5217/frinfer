# Vendored llama.cpp chat stack

How `third_party/llama-chat/` — the parser-only port of llama.cpp's chat stack — is kept current
with upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). The recorded baseline,
the file map and every compat adaptation live in
[`third_party/llama-chat/README.ninfer.md`](../../third_party/llama-chat/README.ninfer.md); that
README is the authority for *what* is vendored, this document for *how* it is refreshed. The port
itself is recorded on wayfinder map [#25](https://github.com/kido5217/ninfer-yarn/issues/25)
(design [#28](https://github.com/kido5217/ninfer-yarn/issues/28)).

## Ground rules

- **Upstream is read-only** — never post to `ggml-org/llama.cpp`; fetch only.
- **Never hand-edit `third_party/llama-chat/common/`.** Upstream changes reconcile in `compat/`,
  and the README's file map and adaptation tables stay current.
- **The vendored set stays minimal** (the #28 decision): `peg-parser`, `chat-peg-parser`,
  `chat.{h,cpp}`, `json.{h,cpp}`, `json-schema*`, `trie`, `unicode`, `parsers/qwen3-coder.cpp`.
  The analysis/auto-parser stack, `jinja/caps` probing, the other specialized parsers, GBNF
  triggers and `reasoning-budget` are excluded; widening the set is a new decision, not a
  re-vendor.
- **A re-vendor lands through the pipeline** (branch → commit → push → PR → squash merge), one
  logical change per PR.

## The drift alert

A session-side smart note (`llama.cpp chat-stack drift`) surfaces when upstream `master` has
commits **after the recorded baseline that touch the vendored file set** — not when master merely
moves, because the excluded subsystems are where upstream churns. Measured 2026-09-27 for the
twelve months before: `peg-parser.cpp` 19 commits and `chat-peg-parser.cpp` 25, against
`chat.cpp` ≥100 and `chat-auto-parser-generator.cpp` 32. Checked again 2026-09-30 with master at
`05af0d2b`: 97 commits and 259 files past the baseline, **zero of them vendored** — every vendored
path is byte-identical to `05af0d2b` (verified per file), and the recorded baseline was advanced
to `05af0d2b`. Check at any time:

```bash
BASELINE=05af0d2b1398394cfa67e1918fee7feabccaa9bc   # keep in sync with the README
gh api "repos/ggml-org/llama.cpp/compare/$BASELINE...master?per_page=100" --jq '.files[].filename' \
  | grep -E '^common/(peg-parser|chat-peg-parser|chat|chat-auto-parser|chat-auto-parser-helpers|json|json-schema|json-schema-to-grammar|trie|unicode)\.(h|cpp)$|^common/parsers/(parsers|qwen3-coder)\.(h|cpp)$|^LICENSE$'
```

The filter must name the exact vendored paths: `common/parsers/` holds non-vendored sibling
parsers, and on 2026-09-30 a broad `^common/parsers/` filter matched `muse-glimmer.cpp` — a false
alarm; only the paths above count. Non-empty output means drift (the compare API lists up to 300
files; for a very large gap, compare against the newest baseline first). When the note fires: run
the procedure below, then re-arm the note with the new baseline.

## The re-vendor procedure

1. **Record the new baseline**:
   `git ls-remote https://github.com/ggml-org/llama.cpp refs/heads/master`.
2. **Fetch and diff the vendored set** (paths from the README's file map):

   ```bash
   NEW=<new-baseline-sha>
   git clone --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp /tmp/opencode/llamacpp-chat-ref
   git -C /tmp/opencode/llamacpp-chat-ref fetch --depth 1 origin "$NEW"
   git -C /tmp/opencode/llamacpp-chat-ref checkout --detach FETCH_HEAD
   # per vendored path: cmp the checkout against third_party/llama-chat/<path>
   ```

3. **Copy the changed files verbatim** into `third_party/llama-chat/<path>`, then re-verify every
   `compat/` shim against the new sources: new includes, renamed symbols, and the excluded-symbol
   list in `compat/excluded_stubs.cpp` (which must keep throwing, never no-op).
4. **Update the README**: baseline commit and date, the file map, the adaptation table, the
   unused-path notes.
5. **Verify** (below), then land through the pipeline.
6. **Record** the re-vendor on the tracker and re-arm the drift note with the new baseline.

## Verification standard

- Build and the host-only parse set, at least
  `ctest --test-dir build -R 'llama_chat|chat_pars|frontend'`: the vendor smoke, the 42-vector
  corpus (`tests/fixtures/chat_parsing/corpus.json`, the semantic authority) and the frontend
  fixtures.
- The port's deliberate overrides must survive: the [#35](https://github.com/kido5217/ninfer-yarn/issues/35)
  recovery layer (no silent *drop tail*) and the two-layer B1 framing are behavior the upstream
  baseline does **not** provide — the corpus catches a regression; never "simplify" toward
  upstream.
- `third_party/llama-jinja` has its own baseline and is not part of this procedure. Rendering
  parity (`NINFER_PARITY_TEMPLATE=<froggeric file> ctest --test-dir build -R chat_templates`) is
  unaffected unless the jinja fork moves.
