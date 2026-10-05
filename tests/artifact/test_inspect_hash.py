"""Verify inspect.py --hash / --check round-trip and object-localized tamper detection."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
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
TENSOR_SHAPE = (130, 130)  # bf16, 33 800 bytes
LARGE_SHAPE = (2048, 2100)  # bf16, 8 601 600 bytes — crosses the 8 MiB read chunk


def run(argv: list[str]) -> tuple[int, str]:
    stdout = StringIO()
    with redirect_stdout(stdout):
        code = artifact_inspect.main(argv)
    return code, stdout.getvalue()


def tensor_bytes(shape: tuple[int, int]) -> bytes:
    elements = shape[0] * shape[1] * 2
    return bytes((index * 31 + 7) % 251 for index in range(elements))


def build(path: Path, specs: list, chunks: dict[str, bytes], max_file_bytes: int) -> None:
    with ArtifactWriter(
        path,
        specs,
        components={"text": {"config": {}}},
        bindings={"embed": {"object": "text.embed.weight"}},
        metadata={"name": "toy"},
        max_file_bytes=max_file_bytes,
    ) as writer:
        for object_id, blob in chunks.items():
            writer.write_object(object_id, blob)


def independent_digests(path: Path) -> tuple[dict[str, str], str]:
    """Recompute digests straight from the reader, independent of inspect._payload_digests."""
    with Artifact(path) as artifact:
        per_object = {
            obj.id: hashlib.sha256(artifact.read_object(obj.id)).hexdigest()
            for obj in artifact.objects
        }
        payload = hashlib.sha256(artifact.read_range(0, artifact.payload_bytes)).hexdigest()
    return per_object, payload


def parse_manifest(text: str) -> tuple[dict[str, str], str | None]:
    objects: dict[str, str] = {}
    payload: str | None = None
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, _, tail = line.partition("  ")
        if tail == "payload":
            payload = digest
        elif tail.startswith("object "):
            objects[tail[len("object ") :]] = digest
    return objects, payload


def verify_case(label: str, path: Path, temporary: Path, single_file: bool) -> int:
    failures = 0
    manifest_path = temporary / f"{label}.manifest"
    code, manifest_text = run([str(path), "--hash"])
    if code != 0 or "  payload" not in manifest_text:
        print(f"{label}: --hash failed ({code})")
        return 1

    manifest_objects, manifest_payload = parse_manifest(manifest_text)
    expected_objects, expected_payload = independent_digests(path)
    if manifest_objects != expected_objects:
        print(f"{label}: manifest object digests disagree with the reader oracle")
        failures += 1
    if manifest_payload != expected_payload:
        print(
            f"{label}: manifest payload digest {manifest_payload} disagrees with the reader "
            f"oracle {expected_payload}"
        )
        failures += 1
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
    if code != 0 or document is None or document.get("objects") != len(expected_objects):
        print(f"{label}: --hash --json did not emit the expected document")
        failures += 1
    else:
        for object_id, digest in expected_objects.items():
            match = next(
                (entry for entry in document["object_sha256"] if entry["id"] == object_id),
                None,
            )
            if match is None or match["sha256"] != digest:
                print(f"{label}: --json digest disagrees for {object_id}")
                failures += 1
        if document["payload_sha256"] != expected_payload:
            print(f"{label}: --json payload digest disagrees")
            failures += 1

    # A tampered manifest digest is rejected.
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        digest, _, tail = line.partition("  ")
        flipped = ("f" if digest[0] != "f" else "0") + digest[1:]
        lines[index] = f"{flipped}  {tail}"
        break
    tampered = temporary / f"{label}.tampered.manifest"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code, out = run([str(path), "--check", str(tampered)])
    if code == 0 or "sha256 mismatch" not in out:
        print(f"{label}: --check accepted a tampered manifest digest:\n{out}")
        failures += 1

    # A manifest for a different artifact is rejected before hashing.
    wrong_id = temporary / f"{label}.wrong-id.manifest"
    wrong_id.write_text(
        f"# ninfer artifact integrity manifest v1\n# artifact_id {'0' * 32}\n{manifest_text.splitlines()[-1]}\n",
        encoding="utf-8",
    )
    code, out = run([str(path), "--check", str(wrong_id)])
    if code == 0 or "artifact_id mismatch" not in out:
        print(f"{label}: --check accepted a manifest for a different artifact:\n{out}")
        failures += 1

    if single_file:
        # A one-byte payload perturbation is detected and localized to the object.
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
            print(f"{label}: --check did not localize the perturbation:\n{out}")
            failures += 1
        if "payload: sha256 mismatch" not in out:
            print(f"{label}: whole-payload digest did not change")
            failures += 1
    return failures


def main() -> int:
    failures = 0
    cases = {
        # single file; resource written after the tensor
        "single": (
            [
                TensorSpec("text.embed.weight", TENSOR_SHAPE, "bf16", "contiguous_le_v1"),
                ResourceSpec("chat_template.jinja", len(RESOURCE)),
            ],
            {"text.embed.weight": tensor_bytes(TENSOR_SHAPE), "chat_template.jinja": RESOURCE},
            4_000_000,
            True,
        ),
        # sharded into continuation files
        "sharded": (
            [
                TensorSpec("text.embed.weight", TENSOR_SHAPE, "bf16", "contiguous_le_v1"),
                ResourceSpec("chat_template.jinja", len(RESOURCE)),
            ],
            {"text.embed.weight": tensor_bytes(TENSOR_SHAPE), "chat_template.jinja": RESOURCE},
            12288,
            False,
        ),
        # resource first, so the alignment-padded tensor leaves a payload gap
        "padded": (
            [
                ResourceSpec("chat_template.jinja", len(RESOURCE)),
                TensorSpec("text.embed.weight", TENSOR_SHAPE, "bf16", "contiguous_le_v1"),
            ],
            {"text.embed.weight": tensor_bytes(TENSOR_SHAPE), "chat_template.jinja": RESOURCE},
            4_000_000,
            True,
        ),
        # payload larger than the 8 MiB read chunk; resource digest comes from a later chunk
        "large": (
            [
                TensorSpec("text.embed.weight", LARGE_SHAPE, "bf16", "contiguous_le_v1"),
                ResourceSpec("chat_template.jinja", len(RESOURCE)),
            ],
            {"text.embed.weight": tensor_bytes(LARGE_SHAPE), "chat_template.jinja": RESOURCE},
            64_000_000,
            True,
        ),
    }
    with tempfile.TemporaryDirectory(prefix="ninfer-inspect-hash-") as temporary:
        root = Path(temporary)
        for label, (specs, chunks, limit, single_file) in cases.items():
            path = root / f"{label}.ninfer"
            build(path, specs, chunks, limit)
            failures += verify_case(label, path, root, single_file)

    if failures:
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
