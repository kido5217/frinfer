#!/usr/bin/env python3
"""Dump the model vocabulary as `id<TAB>hex-bytes` lines for the mask-cost spike.

Mirrors NInfer's tokenizer load path (src/models/qwen3_5/frontend/tokenizer.cpp):
ids from `model.vocab` (byte-level BPE) are decoded to raw bytes through the
GPT-2 byte<->unicode alphabet; tokens from `added_tokens` (tokenizer.json, plus
`added_tokens_decoder` in tokenizer_config.json when supplied) keep their literal
UTF-8 content. The resulting byte strings are what a llama.cpp-style grammar
engine sees through `token_to_piece`.

Pure stdlib; no model or CUDA needed.

Usage:
  dump_vocab.py <tokenizer.json> [--tokenizer-config <tokenizer_config.json>]
                [--out <vocab.tsv>]

Writes the TSV to --out (default: stdout) and a summary to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys


def byte_level_decoder() -> dict[int, int]:
    """Unicode codepoint -> original byte, per NInfer's build_byte_level_decoder."""
    decoder: dict[int, int] = {}
    nxt = 256
    for byte in range(256):
        visible = (33 <= byte <= 126) or (161 <= byte <= 172) or (174 <= byte <= 255)
        codepoint = byte if visible else nxt
        if not visible:
            nxt += 1
        decoder[codepoint] = byte
    return decoder


def decode_token(token: str, decoder: dict[int, int], token_id: int) -> bytes:
    out = bytearray()
    for ch in token:
        codepoint = ord(ch)
        try:
            out.append(decoder[codepoint])
        except KeyError:
            raise SystemExit(
                f"token {token_id} contains codepoint U+{codepoint:04X} outside the "
                "byte-level alphabet"
            )
    return bytes(out)


def load_added_tokens(
    tokenizer_json: dict, tokenizer_config: dict | None
) -> dict[int, str]:
    """id -> literal content; mirrors load_added_tokens + merge_added_tokens_decoder."""
    added: dict[int, str] = {}
    for item in tokenizer_json.get("added_tokens", []):
        added[int(item["id"])] = str(item["content"])
    if tokenizer_config is not None:
        for key, value in tokenizer_config.get("added_tokens_decoder", {}).items():
            token_id = int(key)
            content = str(value["content"])
            if token_id in added:
                if added[token_id] != content:
                    raise SystemExit(
                        f"conflicting added-token definition for id {token_id} between "
                        "tokenizer.json and tokenizer_config.json"
                    )
                continue
            added[token_id] = content
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_json")
    parser.add_argument("--tokenizer-config", default=None)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()

    with open(args.tokenizer_json, encoding="utf-8") as handle:
        tokenizer = json.load(handle)
    tokenizer_config = None
    if args.tokenizer_config:
        with open(args.tokenizer_config, encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)

    model = tokenizer["model"]
    if model.get("type") != "BPE":
        raise SystemExit(f"unexpected model.type: {model.get('type')!r}")
    vocab: dict[str, int] = model["vocab"]

    id_to_token: dict[int, str] = {}
    for token, token_id in vocab.items():
        if token_id in id_to_token:
            raise SystemExit(f"duplicate id in model.vocab: {token_id}")
        id_to_token[token_id] = token
    added = load_added_tokens(tokenizer, tokenizer_config)
    for token_id in added:
        if token_id in id_to_token:
            raise SystemExit(f"added token id {token_id} overlaps model.vocab")

    n = max(max(id_to_token), max(added)) + 1
    decoder = byte_level_decoder()

    lines = []
    added_count = 0
    for token_id in range(n):
        if token_id in added:
            piece = added[token_id].encode("utf-8")
            added_count += 1
        elif token_id in id_to_token:
            piece = decode_token(id_to_token[token_id], decoder, token_id)
        else:
            piece = b""
        lines.append(f"{token_id}\t{piece.hex()}")

    payload = "\n".join(lines) + "\n"
    if args.out == "-":
        sys.stdout.write(payload)
    else:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload)

    mask_bytes = ((n + 31) // 32) * 4
    print(f"vocab ids:        {n}", file=sys.stderr)
    print(f"model.vocab:      {len(id_to_token)}", file=sys.stderr)
    print(f"added (literal):  {added_count}", file=sys.stderr)
    print(f"mask row:         ceil({n}/32)*4 = {mask_bytes} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
