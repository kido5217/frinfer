#!/usr/bin/env python3
"""Fetch the froggeric fixed chat template and report its digest.

The froggeric Qwen-Fixed-Chat-Templates `main` branch is the reference rendering stack for the
Qwen3.8 frontend (tracked at `main`, digest recorded per run). This helper downloads
`chat_template.jinja` to a destination path, prints the sha256 and byte size, and fails loudly
when `--expect-sha256` does not match the fetched bytes (the destination is not written in that
case, so a mismatching fetch never lands at the expected path).

Usage:
  tools/fetch_froggeric_template.py [PATH] [--url URL] [--expect-sha256 HEX]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

DEFAULT_URL = "https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/raw/main/chat_template.jinja"
DEFAULT_PATH = Path("/tmp/opencode/froggeric_chat_template.jinja")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=f"destination file (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL, help=f"template URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--expect-sha256",
        default=None,
        help="fail (and keep the destination untouched) when the fetched digest differs",
    )
    args = parser.parse_args()

    with urllib.request.urlopen(args.url, timeout=120) as response:
        body = response.read()
    digest = hashlib.sha256(body).hexdigest()

    if args.expect_sha256 is not None and digest != args.expect_sha256.strip().lower():
        print(
            f"error: {args.url} digest mismatch: expected {args.expect_sha256.strip().lower()}, "
            f"got {digest} ({len(body)} bytes); destination {args.path} not written",
            file=sys.stderr,
        )
        return 1

    args.path.parent.mkdir(parents=True, exist_ok=True)
    args.path.write_bytes(body)
    print(f"url={args.url}")
    print(f"path={args.path}")
    print(f"sha256={digest}")
    print(f"bytes={len(body)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
