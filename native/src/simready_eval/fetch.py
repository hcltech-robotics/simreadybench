"""Pinned, role-scoped Hub/Xet acquisition for trusted environment setup.

Scoring never calls this module. The injected ``downloader`` has the public
``huggingface_hub.hf_hub_download`` keyword interface and returns a local path.
Only that external transport is replaceable; validation and copies stay local.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading


SCHEMA = "simready-role-assets-v1"
REPO_ID = "hcltech-robotics/simreadybench"


def _relative(value, label):
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or value.startswith("/") or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"{label} must be a canonical safe relative path")
    return value


def validate_manifest(manifest):
    """Validate an exact task/role allowlist and return an independent dict.

    Reference dependency completeness is established by the release/exporter;
    this boundary verifies every declared file and forbids broader acquisition.
    Actor manifests deliberately have no extensible public-file wildcard.
    """
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "repo_id", "revision", "task_id", "role", "files"
    }:
        raise ValueError("manifest must contain exactly the role-asset schema fields")
    if manifest["schema"] != SCHEMA or manifest["repo_id"] != REPO_ID:
        raise ValueError("unsupported manifest schema or repo_id")
    revision = manifest["revision"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full immutable 40-character lowercase commit hash")
    task_id = manifest["task_id"]
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id):
        raise ValueError("task_id must be a safe single path component")
    role = manifest["role"]
    if role not in ("actor", "verifier"):
        raise ValueError("role must be actor or verifier")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("files must be a nonempty exact allowlist")
    prefix = f"benchmark/nv100/v1/tasks/{task_id}/"
    actor_paths = {prefix + "public/observation.png", prefix + "public/instruction.md"}
    paths, destinations, validated = set(), set(), []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "destination", "size_bytes", "sha256"}:
            raise ValueError("file entries require exactly path, destination, size_bytes and sha256")
        path = _relative(entry["path"], "file path")
        destination = _relative(entry["destination"], "destination")
        if role == "actor":
            allowed = path in actor_paths
        else:
            allowed = path.startswith(prefix + "reference/") or path.startswith(prefix + "comparison/")
        if not allowed:
            raise ValueError("file path is outside this task's role allowlist")
        if path in paths or destination in destinations:
            raise ValueError("duplicate file path or destination")
        if any(destination.startswith(other + "/") or other.startswith(destination + "/") for other in destinations):
            raise ValueError("destinations cannot be both a file and a parent directory")
        size = entry["size_bytes"]
        if type(size) is not int or size < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        digest = entry["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest")
        paths.add(path)
        destinations.add(destination)
        validated.append(dict(entry))
    if role == "actor" and paths != actor_paths:
        raise ValueError("actor requires exactly its own observation.png and instruction.md")
    return {**manifest, "files": validated}


def _checked_path(path):
    """Reject existing symlink components before writing any setup path."""
    path = Path(os.path.abspath(os.fspath(path)))
    for parent in (*reversed(path.parents), path):
        try:
            mode = parent.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError("setup paths must not contain symlinks")
        if parent != path and not stat.S_ISDIR(mode):
            raise ValueError("setup parent must be a directory")
    return path


def _private_directory(path):
    _checked_path(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ValueError("cache path must be a directory")


def _digest(path, expected_size):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            count += len(chunk)
            if count > expected_size:
                raise ValueError("asset exceeds expected size_bytes")
            digest.update(chunk)
    if count != expected_size:
        raise ValueError("asset size_bytes mismatch")
    return digest.hexdigest()


def _verify_tree(root, files):
    if not root.is_dir() or root.is_symlink():
        raise ValueError("asset tree must be a regular directory")
    expected = {entry["destination"] for entry in files}
    expected_dirs = {str(parent) for name in expected for parent in PurePosixPath(name).parents if str(parent) != "."}
    found = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink() or path.relative_to(root).as_posix() not in expected_dirs:
                raise ValueError("unexpected directory or symlink in asset tree")
        for name in names:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode) or path.stat().st_nlink != 1:
                raise ValueError("asset tree must contain independent regular files")
            found.add(path.relative_to(root).as_posix())
    if found != expected:
        raise ValueError("asset tree has missing or unexpected files")
    for entry in files:
        if _digest(root / entry["destination"], entry["size_bytes"]) != entry["sha256"]:
            raise ValueError("asset sha256 mismatch")


def _copy_verified(source, target, entry):
    # HF cache symlinks are transport details. Copy bytes, never symlinks or
    # hardlinks. Verify the actual copy so source mutations cannot bypass hashes.
    if not stat.S_ISREG(source.stat().st_mode):
        raise ValueError("download must resolve to a regular file")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    count, digest = 0, hashlib.sha256()
    try:
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            os.chmod(target, 0o600)
            while chunk := incoming.read(1024 * 1024):
                count += len(chunk)
                if count > entry["size_bytes"]:
                    raise ValueError("asset exceeds expected size_bytes")
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if count != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("asset size_bytes or sha256 mismatch")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


@contextmanager
def _entry_lock(path):
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("cache lock must be a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _hub_download(**kwargs):
    # hf_xet owns a global process session, so cache_dir alone cannot isolate
    # its chunk cache. Import the official client in a fresh process with a
    # task/role-private HF home and Xet cache, without ambient login state.
    root = Path(kwargs["cache_dir"])
    environment = os.environ.copy()
    environment.update({
        "HF_HOME": str(root / "hf-home"),
        "HF_HUB_CACHE": str(root / "hub"),
        "HF_XET_CACHE": str(root / "xet"),
        "HF_TOKEN_PATH": str(root / "hf-home" / "token"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    })
    environment.pop("HF_TOKEN", None)
    environment.pop("HUGGING_FACE_HUB_TOKEN", None)
    script = (
        "import json, sys\n"
        "from huggingface_hub import hf_hub_download\n"
        "path = hf_hub_download(**json.load(sys.stdin))\n"
        "print(json.dumps({'path': str(path)}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], input=json.dumps(kwargs),
        text=True, capture_output=True, check=True, env=environment, cwd=root,
    )
    return json.loads(completed.stdout)["path"]


class _DeadlineExpired(BaseException):
    """Bypass ordinary transport retries when the entire setup budget expires."""


@contextmanager
def _setup_deadline(seconds):
    if seconds is None:
        yield
        return
    if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("timeout_seconds must be finite and greater than zero")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("timeout_seconds requires setup in the main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise ValueError("timeout_seconds cannot replace an existing process timer")

    def expired(signum, frame):
        raise _DeadlineExpired()

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    except _DeadlineExpired:
        raise TimeoutError("asset setup exceeded timeout_seconds") from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fetch_assets(manifest, destination, *, cache_dir=None, downloader=None, attempts=3, timeout_seconds=None):
    """Acquire an exact manifest during trusted setup and return a JSON receipt.

    Cache identity includes the complete manifest (including revision, role,
    paths and hashes). The public downloader seam returns a local Hub cache
    path; a separate private transport cache prevents ambient HF-cache reuse.
    Existing destination trees must already match exactly or be empty. Optional
    ``timeout_seconds`` bounds the entire setup, including locking and transport,
    using a Linux process timer; it requires the main thread and no active timer.
    """
    with _setup_deadline(timeout_seconds):
        return _fetch_assets(manifest, destination, cache_dir=cache_dir, downloader=downloader, attempts=attempts)


def _fetch_assets(manifest, destination, *, cache_dir, downloader, attempts):
    manifest = validate_manifest(manifest)
    if type(attempts) is not int or not 1 <= attempts <= 10:
        raise ValueError("attempts must be an integer from 1 to 10")
    destination = _checked_path(destination)
    if destination.exists() and not destination.is_dir():
        raise ValueError("destination must be a directory")
    # Fail before transport if a nonempty output contains unexpected data.
    existing = destination.exists() and any(destination.iterdir())
    if existing:
        _verify_tree(destination, manifest["files"])
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    cache_root = _checked_path(cache_dir if cache_dir is not None else Path.home() / ".cache" / "simready-role-assets")
    namespace = cache_root / manifest["role"] / manifest["task_id"]
    _private_directory(namespace)
    # The caller must mount only this task/role namespace into its environment.
    os.chmod(namespace, 0o700)
    entry = namespace / manifest_hash
    downloader = _hub_download if downloader is None else downloader
    with _entry_lock(namespace / f"{manifest_hash}.lock"):
        cache_hit = False
        try:
            _verify_tree(entry, manifest["files"])
            cache_hit = True
        except ValueError:
            if entry.is_symlink():
                raise ValueError("cache entry must not be a symlink")
            if entry.exists():
                if not entry.is_dir():
                    raise ValueError("cache entry must be a directory")
                shutil.rmtree(entry)
        if not cache_hit:
            with tempfile.TemporaryDirectory(prefix=f".{manifest_hash}-", dir=namespace) as workspace:
                workspace = Path(workspace)
                staged = workspace / "assets"
                staged.mkdir(mode=0o700)
                for file in manifest["files"]:
                    last_error = None
                    for attempt in range(attempts):
                        try:
                            # A fresh isolated Hub cache also prevents a bad Hub
                            # response from poisoning later checksum retries.
                            with tempfile.TemporaryDirectory(prefix="transport-", dir=workspace) as transport:
                                source = downloader(repo_id=manifest["repo_id"], repo_type="dataset",
                                                    filename=file["path"], revision=manifest["revision"],
                                                    token=False, cache_dir=transport)
                                _copy_verified(Path(source), staged / file["destination"], file)
                            break
                        except Exception as error:
                            last_error = error
                    else:
                        raise RuntimeError(f"asset acquisition failed after {attempts} attempts: {file['path']}") from last_error
                _verify_tree(staged, manifest["files"])
                os.replace(staged, entry)
        # Warm cache checks precede every materialization; copy validation checks
        # the actual bytes again instead of trusting a path/revision match.
        if not existing:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".simready-assets-", dir=destination.parent) as workspace:
                staged = Path(workspace) / "assets"
                staged.mkdir(mode=0o700)
                for file in manifest["files"]:
                    _copy_verified(entry / file["destination"], staged / file["destination"], file)
                _verify_tree(staged, manifest["files"])
                _checked_path(destination)
                os.replace(staged, destination)
        _verify_tree(destination, manifest["files"])
    return {
        "schema": "simready-asset-receipt-v1", "repo_id": manifest["repo_id"],
        "revision": manifest["revision"], "task_id": manifest["task_id"],
        "role": manifest["role"], "manifest_sha256": manifest_hash,
        "cache_hit": cache_hit, "files": [dict(file) for file in manifest["files"]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        receipt = fetch_assets(manifest, args.destination, cache_dir=args.cache, attempts=args.attempts,
                               timeout_seconds=args.timeout_seconds)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"asset setup failed: {error}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
