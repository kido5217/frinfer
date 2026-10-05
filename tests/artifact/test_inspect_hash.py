"""Verify inspect.py --hash / --check round-trip and object-localized tamper detection."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.artifact import inspect as artifact_inspect
from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter

RESOURCE = b'{"name": "toy", "version": 1}'
TENSOR_BYTES = 130 * 130 * 2


def run(argv: list[str]) -> tuple[int, str]:
    stdout = StringIO()
    with redirect_stdout(stdout):
        code = artifact_inspect.main(argv)
    return code, stdout.getvalue()


def write_artifact(path: Path, *, max_file_bytes: int) -> None:
    tensor = bytes((index * 31 + 7) % 251 for index in range(TENSOR_BYTES))
    with ArtifactWriter(
        path,
        [
            TensorSpec("text.embed.weight", (130, 130), "bf16", "contiguous_le_v1"),
            ResourceSpec("chat_template.jinja", len(RESOURCE)),
        ],
        components={"text": {"config": {}}},
        bindings={"embed": {"object": "text.embed.weight"}},
        metadata={"name": "toy"},
        max_file_bytes=max_file_bytes,
    ) as writer:
        writer.write_object("text.embed.weight", tensor)
        writer.write_object("chat_template.jinja", RESOURCE)


def check_json_matches_manifest(label: str, manifest_text: str, document: dict) -> int:
    for line in manifest_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, _, tail = line.partition("  ")
        if tail.startswith("object "):
            object_id = tail[len("object ") :]
            match = next(
                (entry for entry in document["object_sha256"] if entry["id"] == object_id),
                None,
            )
            if match is None or match["sha256"] != digest:
                print(f"{label}: --json digest disagrees for {object_id}")
                return 1
        elif tail == "payload" and document["payload_sha256"] != digest:
            print(f"{label}: --json payload digest disagrees")
            return 1
    return 0


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory(prefix="ninfer-inspect-hash-") as temporary:
        for label, limit in (("single", 4_000_000), ("sharded", 12288)):
            path = Path(temporary) / f"{label}.ninfer"
            manifest_path = Path(temporary) / f"{label}.manifest"
            write_artifact(path, max_file_bytes=limit)

            code, manifest_text = run([str(path), "--hash"])
            if code != 0 or "  payload" not in manifest_text:
                print(f"{label}: --hash failed ({code})")
                failures += 1
                continue
            manifest_path.write_text(manifest_text, encoding="utf-8")

            code, out = run([str(path), "--check", str(manifest_path)])
            if code != 0 or ": OK" not in out:
                print(f"{label}: --check did not accept a fresh manifest:\n{out}")
                failures += 1

            code, json_text = run([str(path), "--hash", "--json"])
            try:
                document = json.loads(json_text)
            except json.JSONDecodeError:
                document = None
            if code != 0 or document is None or document.get("objects") != 2:
                print(f"{label}: --hash --json did not emit a two-object document")
                failures += 1
            else:
                failures += check_json_matches_manifest(label, manifest_text, document)

            lines = manifest_path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if line.startswith("#") or not line.strip():
                    continue
                digest, _, tail = line.partition("  ")
                if tail.startswith("object "):
                    flipped = ("f" if digest[0] != "f" else "0") + digest[1:]
                    lines[index] = f"{flipped}  {tail}"
                    break
            tampered = Path(temporary) / f"{label}.tampered.manifest"
            tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
            code, out = run([str(path), "--check", str(tampered)])
            if code == 0 or "sha256 mismatch" not in out:
                print(f"{label}: --check accepted a tampered manifest digest:\n{out}")
                failures += 1

        # A one-byte payload perturbation is detected and localized to the object.
        path = Path(temporary) / "perturbed.ninfer"
        manifest_path = Path(temporary) / "perturbed.manifest"
        write_artifact(path, max_file_bytes=4_000_000)
        _, manifest_text = run([str(path), "--hash"])
        manifest_path.write_text(manifest_text, encoding="utf-8")
        with Artifact(path) as artifact:
            target = artifact.object("chat_template.jinja")
            fd = os.open(path, os.O_RDWR)
            try:
                offset = artifact.payload_offset + target.offset
                original = os.pread(fd, 1, offset)
                os.pwrite(fd, bytes([original[0] ^ 0x01]), offset)
            finally:
                os.close(fd)
        code, out = run([str(path), "--check", str(manifest_path)])
        if code == 0 or "chat_template.jinja: sha256 mismatch" not in out:
            print(f"perturbed: --check did not localize the change:\n{out}")
            failures += 1
        if "payload: sha256 mismatch" not in out:
            print("perturbed: whole-payload digest did not change")
            failures += 1

    if failures:
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
