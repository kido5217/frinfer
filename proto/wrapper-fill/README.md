# Wrapper-grammar prototype (ticket #96) — throwaway

Prototype code for "wrapper-grammar routes at the thinking boundary". Not a product
artifact; see `../proto-wrapper-findings.md` for the question, measurements and verdict.

## CPU walk

```bash
# from the repository root, inside `nix develop`, with a configured build/ tree
SRC=$PWD B=$PWD/build nix develop -c bash proto/wrapper-fill/build.sh
# then, with the Qwen3.8 tokenizer snapshot:
SNAP=$HOME/trash/ai/models/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
/tmp/opencode/proto-wrapper-cpu/wrapper_probe --snapshot "$SNAP" \
  --schema <combined-extraction.schema.json> --reasoning <reasoning.txt> \
  --answer <answer.json> --variant a --emit-grammar wrapper-a.gbnf
```

Variants: `a` (forced close, full encoding), `c` (permissive prefix), `dot`
(`root ::= .*` reference), `plain` (bare converted schema). `--max-reasoning-steps`,
`--dump-steps FILE` and `--emit-grammar FILE` control the walk and outputs.

## Live serve

`e2e/launch-serve.sh` starts the worktree-built `ninfer-yarn-serve` on 127.0.0.1:8827 with
the deployed artifact/flags, `NINFER_PROTO_WRAPPER=1`, and a scratch request log;
`e2e/probe_wrapper.py` sends the graphiti payload with each emitted wrapper as the request
`grammar` field and `enable_thinking: true`. Results: `e2e/results-wrapper.jsonl`,
`e2e/raw/`.
