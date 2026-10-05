"""Inspect FrInfer v3 configurations, objects, bindings and files without numerical libraries."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from .reader import Artifact
from .schema import ArtifactError, TensorObject

MANIFEST_VERSION = 1


def artifact_summary(artifact: Artifact) -> dict:
    tensors = [obj for obj in artifact.objects if isinstance(obj, TensorObject)]
    return {
        "path": str(artifact.path),
        "version": 3,
        "artifact_id": artifact.artifact_id.hex(),
        "name": artifact.directory.metadata.get("name"),
        "components": artifact.directory.components,
        "file_bytes": artifact.file_bytes,
        "payload_bytes": artifact.payload_bytes,
        "files": len(artifact.directory.files),
        "objects": len(artifact.objects),
        "tensors": len(tensors),
        "resources": len(artifact.objects) - len(tensors),
        "bindings": len(artifact.directory.bindings),
        "uses": len(artifact.directory.uses),
        "formats": dict(sorted(Counter(obj.format for obj in tensors).items())),
        "layouts": dict(sorted(Counter(obj.layout for obj in tensors).items())),
    }


def _payload_digests(artifact: Artifact) -> tuple[dict[str, Any], Any]:
    """Stream the logical payload once, returning per-object and whole-payload sha256.

    Object ranges may leave alignment padding between them, so the whole-artifact digest covers
    the complete logical payload `[0, payload_bytes)` while each object digest covers only that
    object's bytes.
    """
    ordered = sorted(artifact.objects, key=lambda obj: obj.offset)
    per_object = {obj.id: hashlib.sha256() for obj in artifact.objects}
    whole = hashlib.sha256()
    position = 0
    first = 0
    for chunk in artifact.iter_range(0, artifact.payload_bytes):
        whole.update(chunk)
        end = position + len(chunk)
        while (
            first < len(ordered)
            and ordered[first].offset + ordered[first].bytes <= position
        ):
            first += 1
        index = first
        while index < len(ordered) and ordered[index].offset < end:
            obj = ordered[index]
            begin = max(obj.offset, position)
            stop = min(obj.offset + obj.bytes, end)
            per_object[obj.id].update(chunk[begin - position : stop - position])
            index += 1
        position = end
    return per_object, whole


def manifest_lines(artifact: Artifact) -> list[str]:
    """Render the --hash manifest: metadata comments, one line per object, then the payload."""
    per_object, whole = _payload_digests(artifact)
    lines = [
        f"# ninfer artifact integrity manifest v{MANIFEST_VERSION}",
        f"# artifact_id {artifact.artifact_id.hex()}",
        f"# payload_bytes {artifact.payload_bytes}",
        f"# objects {len(artifact.objects)}",
    ]
    for obj in artifact.objects:
        lines.append(f"{per_object[obj.id].hexdigest()}  object {obj.id}")
    lines.append(f"{whole.hexdigest()}  payload")
    return lines


def manifest_json(artifact: Artifact) -> dict:
    per_object, whole = _payload_digests(artifact)
    return {
        "path": str(artifact.path),
        "version": 3,
        "artifact_id": artifact.artifact_id.hex(),
        "payload_bytes": artifact.payload_bytes,
        "objects": len(artifact.objects),
        "object_sha256": [
            {
                "id": obj.id,
                "kind": obj.kind,
                "offset": obj.offset,
                "bytes": obj.bytes,
                "sha256": per_object[obj.id].hexdigest(),
            }
            for obj in artifact.objects
        ],
        "payload_sha256": whole.hexdigest(),
    }


def _parse_manifest(text: str) -> tuple[dict[str, str], dict[str, str], str | None, list[str]]:
    metadata: dict[str, str] = {}
    objects: dict[str, str] = {}
    payload: str | None = None
    issues: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            for key in ("artifact_id", "payload_bytes"):
                if body.startswith(key + " "):
                    metadata[key] = body[len(key) + 1 :].strip()
            continue
        digest, separator, tail = line.partition("  ")
        if not separator or len(digest) != 64:
            issues.append(f"line {number}: malformed manifest entry: {line!r}")
            continue
        if tail == "payload":
            payload = digest
        elif tail.startswith("object "):
            object_id = tail[len("object ") :]
            if object_id in objects:
                issues.append(f"line {number}: duplicate object {object_id!r}")
            objects[object_id] = digest
        else:
            issues.append(f"line {number}: unknown manifest entry: {line!r}")
    return metadata, objects, payload, issues


def check_manifest(artifact: Artifact, manifest_path: Path) -> tuple[bool, dict]:
    """Verify a --hash manifest against the artifact, localizing mismatches to the object."""
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ArtifactError(f"cannot read manifest {manifest_path}: {error}") from error

    metadata, expected_objects, expected_payload, issues = _parse_manifest(text)
    artifact_id = artifact.artifact_id.hex()
    declared_id = metadata.get("artifact_id")
    if declared_id is not None and declared_id != artifact_id:
        issues.append(
            f"artifact_id mismatch: manifest {declared_id} != artifact {artifact_id}"
        )
        report = {
            "path": str(artifact.path),
            "manifest": str(manifest_path),
            "artifact_id": artifact_id,
            "ok": False,
            "issues": issues,
        }
        return False, report

    if expected_payload is None:
        issues.append("manifest has no payload entry")
    if "payload_bytes" in metadata and metadata["payload_bytes"] != str(
        artifact.payload_bytes
    ):
        issues.append(
            f"payload_bytes mismatch: manifest {metadata['payload_bytes']} != artifact "
            f"{artifact.payload_bytes}"
        )

    per_object, whole = _payload_digests(artifact)
    actual_ids = {obj.id for obj in artifact.objects}
    for obj in artifact.objects:
        expected = expected_objects.get(obj.id)
        actual = per_object[obj.id].hexdigest()
        if expected is None:
            issues.append(
                f"object {obj.id}: missing from manifest (offset {obj.offset}, bytes {obj.bytes})"
            )
        elif expected != actual:
            issues.append(
                f"object {obj.id}: sha256 mismatch (manifest {expected}, artifact {actual}; "
                f"offset {obj.offset}, bytes {obj.bytes})"
            )
    for object_id in expected_objects:
        if object_id not in actual_ids:
            issues.append(f"object {object_id}: in manifest but missing from artifact")
    if expected_payload is not None and expected_payload != whole.hexdigest():
        issues.append(
            f"payload: sha256 mismatch (manifest {expected_payload}, artifact {whole.hexdigest()})"
        )

    report = {
        "path": str(artifact.path),
        "manifest": str(manifest_path),
        "artifact_id": artifact_id,
        "objects": len(artifact.objects),
        "payload_sha256": whole.hexdigest(),
        "ok": not issues,
        "issues": issues,
    }
    return not issues, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--objects", action="store_true", help="include physical object records"
    )
    parser.add_argument(
        "--bindings", action="store_true", help="include logical bindings and Uses"
    )
    parser.add_argument("--json", action="store_true", help="emit one JSON object")
    parser.add_argument(
        "--hash",
        action="store_true",
        help="print per-object and whole-artifact sha256 manifest lines",
    )
    parser.add_argument(
        "--check",
        type=Path,
        metavar="MANIFEST",
        help="verify the artifact against a manifest written by --hash",
    )
    args = parser.parse_args(argv)
    if args.hash and args.check is not None:
        parser.error("--hash and --check are mutually exclusive")
    if (args.hash or args.check is not None) and (args.objects or args.bindings):
        parser.error("--objects and --bindings apply only to the default summary")

    with Artifact(args.artifact) as artifact:
        if args.check is not None:
            ok, report = check_manifest(artifact, args.check)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            elif ok:
                print(
                    f"{artifact.path}: OK ({report['objects']} objects, artifact_id "
                    f"{report['artifact_id']}, payload sha256 {report['payload_sha256']})"
                )
            else:
                for issue in report["issues"]:
                    print(f"{artifact.path}: {issue}")
            return 0 if ok else 1

        if args.hash:
            if args.json:
                print(json.dumps(manifest_json(artifact), ensure_ascii=False, indent=2))
            else:
                for line in manifest_lines(artifact):
                    print(line)
            return 0

        summary = artifact_summary(artifact)
        if args.json:
            if args.objects:
                summary["object_records"] = [obj.to_json() for obj in artifact.objects]
            if args.bindings:
                summary["binding_records"] = artifact.directory.bindings
                summary["use_records"] = list(artifact.directory.uses)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        for key, value in summary.items():
            print(f"{key}: {value}")
        if args.objects:
            for obj in artifact.objects:
                storage = (
                    f"{obj.format}/{obj.layout} {list(obj.shape)}"
                    if isinstance(obj, TensorObject)
                    else obj.encoding
                )
                print(
                    f"{obj.offset:>14} {obj.bytes:>14} {obj.kind:<8} {storage:<42} {obj.id}"
                )
        if args.bindings:
            print(
                json.dumps(
                    {
                        "bindings": artifact.directory.bindings,
                        "uses": list(artifact.directory.uses),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
