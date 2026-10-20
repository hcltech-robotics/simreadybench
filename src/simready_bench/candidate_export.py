"""Durable copies of the runner's frozen authored submission, never references."""

from __future__ import annotations
import hashlib, json, os, re, shutil, stat, tempfile
from pathlib import Path, PurePosixPath

MAX_BYTES = 256 * 1024 * 1024  # Controller's default frozen candidate package ceiling.
MAX_ENTRIES = 10000
_HASH = re.compile(r"[a-f0-9]{64}")


def _relative(value):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(c in value for c in ("\\", ":", "?", "#", "%"))
        or any(p in ("", ".", "..") for p in value.split("/"))
    ):
        raise ValueError("candidate_unsafe_relative_path")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) > 64:
        raise ValueError("candidate_unsafe_relative_path")
    return path


def _signature(st):
    return (
        st.st_dev,
        st.st_ino,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
        st.st_nlink,
    )


def _read_file(root, relative, *, maximum, target=None, retain=True):
    """Open every path component without following links, then stream exact bytes."""
    relative = _relative(relative)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        stream_fd = os.open(
            relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
    finally:
        os.close(fd)
    try:
        before = os.fstat(stream_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
        ):
            raise ValueError("candidate_regular_bounded_file_required")
        digest = hashlib.sha256()
        size = 0
        chunks = []
        with os.fdopen(os.dup(stream_fd), "rb") as incoming:
            while block := incoming.read(1024 * 1024):
                size += len(block)
                if size > maximum:
                    raise ValueError("candidate_file_size_changed")
                digest.update(block)
                if target is None:
                    if retain:
                        chunks.append(block)
                else:
                    target.write(block)
        if (
            _signature(before) != _signature(os.fstat(stream_fd))
            or size != before.st_size
        ):
            raise ValueError("candidate_source_changed")
        return b"".join(chunks), size, digest.hexdigest()
    finally:
        os.close(stream_fd)


def _roster(root):
    actual = set()
    entries = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            entries += 1
            if entries > MAX_ENTRIES:
                raise ValueError("candidate_entry_limit_exceeded")
            path = Path(directory) / name
            st = path.lstat()
            if stat.S_ISDIR(st.st_mode):
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ValueError("candidate_link_or_special_file_denied")
            actual.add(path.relative_to(root).as_posix())
    return actual


def export_candidate(
    submission, *, work_root, report_root, condition_id, expected_snapshot_sha256=None
):
    """Verify simready-bench-submission-v1 and copy only its closed frozen roster.

    Returns report-relative paths. Missing/invalid data is reported by the caller;
    no partial candidate tree survives an export failure.
    """
    if (
        type(submission) is not dict
        or submission.get("copied_independently") is not True
    ):
        raise ValueError("candidate_frozen_submission_receipt_required")
    work = Path(work_root)
    out = Path(report_root)
    source = Path(submission["directory"]).absolute()
    manifest = Path(submission["manifest"]).absolute()
    if (
        source.resolve() != source
        or not source.is_relative_to(work)
        or manifest != source.parent / "snapshot-manifest.json"
        or source.parts[-3:] != ("artifacts", "submission", "files")
    ):
        raise ValueError("candidate_authored_submission_location_required")
    raw, _, manifest_sha = _read_file(
        work, manifest.relative_to(work).as_posix(), maximum=4 * 1024 * 1024
    )

    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("candidate_duplicate_manifest_key")
            result[k] = v
        return result

    doc = json.loads(raw, object_pairs_hook=unique)
    if (
        type(doc) is not dict
        or set(doc)
        != {
            "schema",
            "entrypoint",
            "snapshot_sha256",
            "files",
            "size_bytes",
            "copied_independently",
        }
        or doc.get("schema") != "simready-bench-submission-v1"
        or doc.get("copied_independently") is not True
        or type(doc.get("files")) is not list
        or not 1 <= len(doc["files"]) <= MAX_ENTRIES
        or type(doc.get("size_bytes")) is not int
        or not 0 <= doc["size_bytes"] <= MAX_BYTES
    ):
        raise ValueError("candidate_frozen_manifest_invalid_or_oversize")
    entry = str(_relative(doc["entrypoint"]))
    rows = doc["files"]
    names = []
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != {"relative_path", "size_bytes", "sha256"}
            or type(row.get("size_bytes")) is not int
            or not 0 <= row["size_bytes"] <= MAX_BYTES
            or not isinstance(row.get("sha256"), str)
            or not _HASH.fullmatch(row["sha256"])
        ):
            raise ValueError("candidate_frozen_file_record_invalid")
        names.append(str(_relative(row["relative_path"])))
    if (
        names != sorted(set(names))
        or entry not in names
        or sum(r["size_bytes"] for r in rows) != doc["size_bytes"]
    ):
        raise ValueError("candidate_frozen_roster_invalid")
    identity = {"entrypoint": entry, "files": rows}
    snapshot_sha = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        snapshot_sha != doc.get("snapshot_sha256")
        or snapshot_sha != submission.get("snapshot_sha256")
        or expected_snapshot_sha256 is not None
        and snapshot_sha != expected_snapshot_sha256
    ):
        raise ValueError("candidate_snapshot_digest_mismatch")
    if _roster(source) != set(names):
        raise ValueError("candidate_source_roster_changed")
    safe = (
        condition_id
        if isinstance(condition_id, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", condition_id)
        else "condition-" + hashlib.sha256(str(condition_id).encode()).hexdigest()
    )
    parent = out / "candidates"
    parent.mkdir(mode=0o700, exist_ok=True)
    destination = parent / safe
    if destination.exists() or destination.is_symlink():
        raise ValueError("candidate_destination_collision")
    stage = Path(tempfile.mkdtemp(prefix=".candidate-", dir=parent))
    exported = []
    try:
        files = stage / "files"
        files.mkdir(mode=0o700)
        for row in rows:
            target = files.joinpath(*PurePosixPath(row["relative_path"]).parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with target.open("xb") as stream:
                _, size, sha = _read_file(
                    source,
                    row["relative_path"],
                    maximum=row["size_bytes"],
                    target=stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            if size != row["size_bytes"] or sha != row["sha256"]:
                raise ValueError("candidate_file_digest_mismatch")
            target.chmod(0o600)
            _, copied_size, copied_sha = _read_file(
                files, row["relative_path"], maximum=row["size_bytes"], retain=False
            )
            if copied_size != row["size_bytes"] or copied_sha != row["sha256"]:
                raise ValueError("candidate_durable_copy_digest_mismatch")
            exported.append(
                {
                    **row,
                    "path": (destination / "files" / row["relative_path"])
                    .relative_to(out)
                    .as_posix(),
                }
            )
        if _roster(source) != set(names):
            raise ValueError("candidate_source_roster_changed")
        after, _, _ = _read_file(
            work, manifest.relative_to(work).as_posix(), maximum=4 * 1024 * 1024
        )
        if after != raw:
            raise ValueError("candidate_source_manifest_changed")
        copied = stage / "snapshot-manifest.json"
        with copied.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        copied.chmod(0o600)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return {
        "status": "available",
        "schema": "simready-bench-portable-candidate-v1",
        "snapshot_sha256": snapshot_sha,
        "manifest_sha256": manifest_sha,
        "manifest": (destination / "snapshot-manifest.json")
        .relative_to(out)
        .as_posix(),
        "directory": (destination / "files").relative_to(out).as_posix(),
        "entrypoint": (destination / "files" / entry).relative_to(out).as_posix(),
        "source_entrypoint": entry,
        "size_bytes": doc["size_bytes"],
        "files": exported,
        "copied_independently": True,
        "maximum_export_bytes": MAX_BYTES,
    }
