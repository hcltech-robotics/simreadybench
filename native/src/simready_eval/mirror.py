"""Describe one optional mutable Storage Bucket mirror of a pinned NV100 set.

This helper is offline and writes only a local JSON plan. The bucket name is
chosen later. A future authorized copy can use supported server-side Xet copy;
no provisioning, copy API, network access or account operation occurs here.
Canonical task manifests continue to refer to the immutable HF dataset revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .export import _safe_path, _validate_release

PREFIX = "benchmark/nv100/v1/"
EXTRA_NAMES = frozenset({"README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICES.md",
                         "THIRD_PARTY_NOTICES.md", "checksums.json", "checksums.sha256", "SHA256SUMS"})


def _entry(path, size, sha256):
    path = _safe_path(path)
    if not path.startswith(PREFIX):
        raise ValueError("mirror files must belong to benchmark/nv100/v1")
    parts = [PurePosixPath(part).stem.lower() for part in path.split("/")]
    if any(part in {"results", "runs", "viewer", "media", "private"}
           or part.startswith(("private_", "_private")) or "orichalcum" in part for part in parts):
        raise ValueError("historical results, private and Orichalcum files cannot enter the set mirror")
    if type(size) is not int or size < 0:
        raise ValueError("size_bytes must be a nonnegative integer")
    if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("sha256 must be a lowercase SHA-256 digest")
    return {"source_path": path, "destination_key": path.removeprefix(PREFIX),
            "size_bytes": size, "sha256": sha256}


def build_mirror_manifest(release, extra_files=(), *, fixture=False):
    """Build one set-wide mirror plan, preserving all canonical file hashes.

    ``extra_files`` explicitly supplies per-set README, license/notice or checksum
    records as ``{path, size_bytes, sha256}``. Only the root metadata names in
    ``EXTRA_NAMES`` are supported. No repository traversal or implicit discovery
    occurs. Duplicate paths must describe identical bytes. ``fixture=True``
    permits the exporter's smaller local fixture roster; it never establishes
    release qualification or changes the canonical dataset authority.
    """
    if type(fixture) is not bool:
        raise ValueError("fixture must be a boolean")
    release = _validate_release(release, fixture)
    if not isinstance(extra_files, (list, tuple)):
        raise ValueError("extra_files must be an explicit list of metadata records")
    by_path = {}

    def include(entry):
        previous = by_path.get(entry["source_path"])
        if previous is not None and previous != entry:
            raise ValueError("duplicate mirror path has conflicting size or checksum")
        by_path[entry["source_path"]] = entry

    for task in release["tasks"]:
        for role in ("actor", "verifier"):
            for file in task[role]["files"]:
                include(_entry(file["path"], file["size_bytes"], file["sha256"]))
    for file in extra_files:
        if not isinstance(file, dict) or set(file) != {"path", "size_bytes", "sha256"}:
            raise ValueError("extra files require exactly path, size_bytes and sha256")
        entry = _entry(file["path"], file["size_bytes"], file["sha256"])
        if entry["destination_key"] not in EXTRA_NAMES:
            raise ValueError("unsupported per-set extra; only explicit README, license and checksum files are allowed")
        include(entry)
    manifest = {
        "schema": "simready-set-mirror-v1", "set_id": "nv100",
        "canonical": {"repo_id": release["repo_id"], "revision": release["revision"]},
        "files": [by_path[path] for path in sorted(by_path)],
        "warnings": ["mutable_not_authority", "bucket_mirror_mutable_nonversioned",
                     "bucket_name_chosen_later", "server_side_xet_copy_not_performed"],
    }
    if fixture:
        manifest["warnings"].append("fixture_not_publish_ready")
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {**manifest, "manifest_sha256": hashlib.sha256(serialized).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--extra-files", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    try:
        release = json.loads(args.release.read_text())
        extras = json.loads(args.extra_files.read_text()) if args.extra_files else []
        manifest = build_mirror_manifest(release, extras, fixture=args.fixture)
        # Exclusive creation keeps a local review plan from overwriting a
        # previous release plan or following a user-supplied output symlink.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    except (ValueError, OSError) as error:
        parser.exit(1, f"set mirror planning failed: {error}\n")
    print(json.dumps({"output": str(args.output), "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
