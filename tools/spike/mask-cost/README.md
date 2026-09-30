# mask-cost spike

Throwaway measurement harness for the constrained-decoding design question:
what does llama.cpp's GBNF runtime cost per step over this model's full
vocabulary (248 077 tokens)? Answer and analysis:
`docs/research/mask-cost-spike.md`.

Nothing here ships; the engine is not modified and llama.cpp is used read-only.

## Layout

| Path | What |
|---|---|
| `upstream/llama-grammar.{h,cpp}` | byte-identical copies of llama.cpp `src/llama-grammar.*` at baseline `05af0d2b1398394cfa67e1918fee7feabccaa9bc` (sha256 in the doc) |
| `shim/` | minimal adapter: `llama.h` (typedefs + candidate array), `llama-impl.h` (LLAMA_LOG_* / GGML_ASSERT → stderr/abort), `llama-vocab.h` (piece/eog/tokenize), `llama-sampler.h` (unused upstream include) |
| `harness.cpp` | the measurement driver (init / full fill / validate fast path / accept, plus an instrumented decode-vs-reject split) |
| `dump_vocab.py` | token-piece dump `id<TAB>hex-bytes`, mirroring NInfer's byte-level-BPE token decode |
| `gbnf_from_schema.cpp` | tiny driver around the in-tree vendored `json_schema_to_grammar` (builds `grammars/g4_diary_schema.gbnf`) |
| `grammars/` | G1–G3 fixtures (as specified) + the diary JSON schema and its converted GBNF |
| `results/` | captured run: stdout transcript + JSON report (200 iterations) |

## Reproduce

```bash
# 0. from the repo root, inside the devShell (g++ + CUDA toolchain)
nix develop -c bash tools/spike/mask-cost/build.sh /tmp/opencode/mask-cost-build

# 1. token-piece dump for the served revision (see docs for why this snapshot)
SNAP=$HOME/trash/ai/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
python3 tools/spike/mask-cost/dump_vocab.py "$SNAP/tokenizer.json" \
    --tokenizer-config "$SNAP/tokenizer_config.json" \
    --out /tmp/opencode/mask-cost-build/vocab.tsv

# 2. (only if the schema fixture changes) regenerate the G4 grammar
/tmp/opencode/mask-cost-build/gbnf_from_schema \
    tools/spike/mask-cost/grammars/g4_diary_schema.json \
    > tools/spike/mask-cost/grammars/g4_diary_schema.gbnf

# 3. measure (single-threaded; ~2 min with --iters 200)
/tmp/opencode/mask-cost-build/mask-cost-harness \
    --vocab /tmp/opencode/mask-cost-build/vocab.tsv \
    --grammars tools/spike/mask-cost/grammars \
    --eog 248044,248046 --iters 200 \
    --json /tmp/opencode/mask-cost-build/run.json
```

Notes:

- The vocab dump is regenerated per machine; it is a derived artifact and is not
  committed. `results/run-200.json` is the committed capture of one run.
- EOG is approximated by the model's `generation_config.json`
  `eos_token_id` list; NInfer's runtime stop set can be larger.
- `--iters N` sets the iteration count (fill/init/accept `N`, validate and the
  mask-row floor `5N`); warmups are discarded.
