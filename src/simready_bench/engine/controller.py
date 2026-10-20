"""Trusted SQLite quotas and frozen candidate copies.

Only the host controller registers actor roots. Actor calls select an episode and
request ID, never a filesystem path. The only candidate directory is <actor_root>/output.
All symbolic links are rejected, including internal links: this deliberately stronger
package rule must be published in the common authoring contract. Packages have at
most 10,000 expanded entries (files and directories, excluding the output root).
Snapshot work has a five-second monotonic limit, further capped by the original
request/episode deadline. Neither USD dependency validation nor evaluator scheduling
happens in this module.

The state directory must remain outside every actor mount and accessible only to the
trusted service. Read-only snapshot permissions protect against accidental writes;
mount isolation, rather than chmod alone, enforces the actor/trust boundary.
"""

from contextlib import closing, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import time
from uuid import uuid4
from simready_bench.engine.contracts import (
    Episode,
    Snapshot,
    SnapshotFile,
    ValidationJob,
)


class AdmissionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _InvalidCandidate(Exception):
    pass


class _Connection(sqlite3.Connection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup_paths = []


def _cleanup_snapshot(path):
    def retry_readonly(function, filename, error):
        os.chmod(Path(filename).parent, 448)
        os.chmod(filename, 448)
        function(filename)

    shutil.rmtree(path, onexc=retry_readonly)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identifier(value):
    if not isinstance(value, str) or not value or len(value) > 256 or ("\x00" in value):
        raise ValueError("Identifiers must contain 1 to 256 non-NUL characters")
    return value


def _open_directory(path):
    """Walk from / using directory descriptors, never following path symlinks."""
    path = Path(os.path.abspath(path))
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


class Controller:
    def __init__(
        self,
        state_dir,
        *,
        max_package_bytes=256 * 1024 * 1024,
        max_package_entries=10000,
        snapshot_timeout_seconds=5.0,
    ):
        if not isinstance(max_package_bytes, int) or max_package_bytes <= 0:
            raise ValueError("max_package_bytes must be positive")
        if type(max_package_entries) is not int or max_package_entries <= 0:
            raise ValueError("max_package_entries must be positive")
        if (
            type(snapshot_timeout_seconds) not in (int, float)
            or not math.isfinite(snapshot_timeout_seconds)
            or snapshot_timeout_seconds <= 0
        ):
            raise ValueError("snapshot_timeout_seconds must be finite and positive")
        self.state_dir = Path(os.path.abspath(state_dir))
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=448)
        fd = _open_directory(self.state_dir)
        os.close(fd)
        self.snapshots_dir = self.state_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True, mode=448)
        fd = _open_directory(self.snapshots_dir)
        os.close(fd)
        self.db_path = self.state_dir / "controller.sqlite3"
        self.max_package_bytes = max_package_bytes
        self.max_package_entries = max_package_entries
        self.snapshot_timeout_seconds = snapshot_timeout_seconds
        with closing(self._connect()) as db:
            db.executescript(
                "\n            CREATE TABLE IF NOT EXISTS episodes (\n                episode_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, trial_id TEXT NOT NULL,\n                runner TEXT NOT NULL, actor_root TEXT NOT NULL, root_dev INTEGER NOT NULL,\n                root_ino INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'registered',\n                deadline_seconds REAL NOT NULL, started_at REAL, deadline_at REAL,\n                termination_reason TEXT, validation_limit INTEGER NOT NULL,\n                preview_limit INTEGER NOT NULL, inference_limit INTEGER NOT NULL,\n                final_snapshot_id TEXT\n            );\n            CREATE TABLE IF NOT EXISTS snapshots (\n                snapshot_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes,\n                record TEXT NOT NULL\n            );\n            CREATE TABLE IF NOT EXISTS jobs (\n                job_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL REFERENCES episodes,\n                kind TEXT NOT NULL, request_id TEXT NOT NULL, sequence INTEGER NOT NULL,\n                snapshot_id TEXT REFERENCES snapshots, status TEXT NOT NULL,\n                admitted_at REAL NOT NULL, deadline_at REAL NOT NULL, result_json TEXT,\n                UNIQUE(episode_id, kind, request_id),\n                UNIQUE(episode_id, kind, sequence)\n            );\n            "
            )
        os.chmod(self.db_path, 384)

    def _connect(self):
        db = sqlite3.connect(
            self.db_path, timeout=30, isolation_level=None, factory=_Connection
        )
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 30000")
        return db

    @contextmanager
    def _transaction(self):
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
            for incomplete_copy in db.cleanup_paths:
                _cleanup_snapshot(incomplete_copy)

    def register_episode(
        self,
        episode_id,
        actor_root,
        *,
        task_id,
        trial_id,
        runner="harbor",
        validation_limit=5,
        preview_limit=12,
        inference_limit=128,
        deadline_seconds=1800,
    ):
        for value in (episode_id, task_id, trial_id, runner):
            _identifier(value)
        if (
            not math.isfinite(deadline_seconds)
            or deadline_seconds <= 0
            or any(
                (
                    not isinstance(n, int) or isinstance(n, bool) or n < 0
                    for n in (validation_limit, preview_limit, inference_limit)
                )
            )
        ):
            raise ValueError("Invalid episode limits")
        root = Path(os.path.abspath(actor_root))
        if root.is_relative_to(self.state_dir) or self.state_dir.is_relative_to(root):
            raise ValueError("Trusted state must be outside the actor root")
        fd = _open_directory(root)
        try:
            root_stat = os.fstat(fd)
        finally:
            os.close(fd)
        with self._transaction() as db:
            if db.execute(
                "SELECT 1 FROM episodes WHERE episode_id=?", (episode_id,)
            ).fetchone():
                raise AdmissionError("episode_exists")
            db.execute(
                "INSERT INTO episodes\n                (episode_id,task_id,trial_id,runner,actor_root,root_dev,root_ino,\n                 deadline_seconds,validation_limit,preview_limit,inference_limit)\n                VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    episode_id,
                    task_id,
                    trial_id,
                    runner,
                    str(root),
                    root_stat.st_dev,
                    root_stat.st_ino,
                    deadline_seconds,
                    validation_limit,
                    preview_limit,
                    inference_limit,
                ),
            )
        return self.get_episode(episode_id)

    def _episode_row(self, db, episode_id):
        row = db.execute(
            "SELECT * FROM episodes WHERE episode_id=?", (episode_id,)
        ).fetchone()
        if row is None:
            raise AdmissionError("unknown_episode")
        return row

    def get_episode(self, episode_id):
        with closing(self._connect()) as db:
            row = self._episode_row(db, episode_id)
            counts = dict(
                db.execute(
                    "SELECT kind, count(*) FROM jobs WHERE episode_id=? GROUP BY kind",
                    (episode_id,),
                ).fetchall()
            )
            return Episode(
                **{
                    field: row[field]
                    for field in Episode.__dataclass_fields__
                    if not field.endswith("_calls_used")
                },
                validation_calls_used=counts.get("validation", 0),
                preview_calls_used=counts.get("preview", 0),
                inference_calls_used=counts.get("inference", 0),
            )

    def start_episode(self, episode_id):
        with self._transaction() as db:
            row = self._episode_row(db, episode_id)
            if row["status"] != "registered":
                raise AdmissionError("episode_already_started")
            now = time.time()
            db.execute(
                "UPDATE episodes SET status='active',started_at=?,deadline_at=? WHERE episode_id=?",
                (now, now + row["deadline_seconds"], episode_id),
            )
        return self.get_episode(episode_id)

    def _active(self, row):
        if row["status"] == "closed":
            raise AdmissionError("episode_closed")
        if row["status"] != "active":
            raise AdmissionError("episode_not_started")
        if time.time() >= row["deadline_at"]:
            raise AdmissionError("deadline_exceeded")

    def _load_snapshot(self, db, snapshot_id):
        row = db.execute(
            "SELECT record FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise AdmissionError("invalid_snapshot")
        data = json.loads(row["record"])
        data["files"] = tuple((SnapshotFile(**item) for item in data["files"]))
        return Snapshot(**data)

    def freeze_candidate(self, episode_id):
        """Trusted capture; available after deadline and independent of feedback K."""
        with self._transaction() as db:
            row = self._episode_row(db, episode_id)
            if row["final_snapshot_id"]:
                return self._load_snapshot(db, row["final_snapshot_id"])
            return self._snapshot(db, row)

    def _snapshot(self, db, row, *, deadline_at=None):
        stop_at = time.monotonic() + self.snapshot_timeout_seconds
        if deadline_at is not None:
            stop_at = min(stop_at, time.monotonic() + max(0, deadline_at - time.time()))

        def check_time():
            if time.monotonic() >= stop_at or (
                deadline_at is not None and time.time() >= deadline_at
            ):
                raise _InvalidCandidate("snapshot_deadline_exceeded")

        entry_counts = {"copy": 0, "verify": 0}

        def names(source_fd, phase):
            result = []
            with os.scandir(source_fd) as entries:
                for entry in entries:
                    check_time()
                    entry_counts[phase] += 1
                    if entry_counts[phase] > self.max_package_entries:
                        raise _InvalidCandidate("package_entry_limit_exceeded")
                    result.append(entry.name)
            check_time()
            return sorted(result)

        snapshot_id = uuid4().hex
        destination = self.snapshots_dir / snapshot_id
        destination.mkdir(mode=448)
        files = []
        source_signatures = {}
        total = 0

        def signature(info):
            return (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
                info.st_nlink,
            )

        def copy_dir(source_fd, target, prefix="", depth=0):
            nonlocal total
            check_time()
            if depth > 64:
                raise _InvalidCandidate("package_too_deep")
            before = os.fstat(source_fd)
            source_signatures[prefix] = signature(before)
            for name in names(source_fd, "copy"):
                check_time()
                relative = f"{prefix}/{name}" if prefix else name
                if (
                    name in (".", "..")
                    or "/" in name
                    or "\\" in name
                    or PurePosixPath(relative).is_absolute()
                ):
                    raise _InvalidCandidate("unsafe_path")
                info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise _InvalidCandidate("symlink_not_allowed")
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=source_fd,
                    )
                    try:
                        if (os.fstat(child).st_dev, os.fstat(child).st_ino) != (
                            info.st_dev,
                            info.st_ino,
                        ):
                            raise _InvalidCandidate("candidate_changed_during_snapshot")
                        child_target = target / name
                        child_target.mkdir(mode=448)
                        copy_dir(child, child_target, relative, depth + 1)
                    finally:
                        os.close(child)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise _InvalidCandidate("special_file_not_allowed")
                if info.st_nlink != 1:
                    raise _InvalidCandidate("hardlink_not_allowed")
                if total + info.st_size > self.max_package_bytes:
                    raise _InvalidCandidate("package_too_large")
                fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd
                )
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode) or signature(
                        opened
                    ) != signature(info):
                        raise _InvalidCandidate("candidate_changed_during_snapshot")
                    source_signatures[relative] = signature(opened)
                    digest = hashlib.sha256()
                    size = 0
                    with (target / name).open("xb") as out:
                        while True:
                            check_time()
                            chunk = os.read(fd, 1024 * 1024)
                            check_time()
                            if not chunk:
                                break
                            size += len(chunk)
                            total += len(chunk)
                            if total > self.max_package_bytes:
                                raise _InvalidCandidate("package_too_large")
                            digest.update(chunk)
                            check_time()
                            out.write(chunk)
                            check_time()
                    if (
                        signature(os.fstat(fd)) != signature(opened)
                        or size != opened.st_size
                    ):
                        raise _InvalidCandidate("candidate_changed_during_snapshot")
                    if signature(
                        os.stat(name, dir_fd=source_fd, follow_symlinks=False)
                    ) != signature(opened):
                        raise _InvalidCandidate("candidate_changed_during_snapshot")
                    files.append(SnapshotFile(relative, size, digest.hexdigest()))
                finally:
                    os.close(fd)
            after = os.fstat(source_fd)
            if (before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise _InvalidCandidate("candidate_changed_during_snapshot")

        def verify_tree(source_fd, prefix=""):
            check_time()
            if signature(os.fstat(source_fd)) != source_signatures[prefix]:
                raise _InvalidCandidate("candidate_changed_during_snapshot")
            for name in names(source_fd, "verify"):
                check_time()
                relative = f"{prefix}/{name}" if prefix else name
                info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
                if source_signatures.get(relative) != signature(info):
                    raise _InvalidCandidate("candidate_changed_during_snapshot")
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=source_fd,
                    )
                    try:
                        verify_tree(child, relative)
                    finally:
                        os.close(child)

        root_fd = None
        output_fd = None
        try:
            try:
                check_time()
                root_fd = _open_directory(row["actor_root"])
                root_stat = os.fstat(root_fd)
                if (root_stat.st_dev, root_stat.st_ino) != (
                    row["root_dev"],
                    row["root_ino"],
                ):
                    raise _InvalidCandidate("actor_root_changed")
                output_fd = os.open(
                    "output",
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
                copy_dir(output_fd, destination)
                verify_tree(output_fd)
                current_output = os.stat(
                    "output", dir_fd=root_fd, follow_symlinks=False
                )
                opened_output = os.fstat(output_fd)
                if (current_output.st_dev, current_output.st_ino) != (
                    opened_output.st_dev,
                    opened_output.st_ino,
                ):
                    raise _InvalidCandidate("candidate_changed_during_snapshot")
                missing_entrypoint = "asset.usd" not in {
                    item.relative_path for item in files
                }
                if missing_entrypoint:
                    raise _InvalidCandidate("missing_entrypoint")
            except OSError as exc:
                if exc.errno in (2, 13, 20, 21, 40):
                    raise _InvalidCandidate("candidate_unreadable_or_unsafe") from exc
                raise
            check_time()
            files.sort(key=lambda item: item.relative_path)
            digest = hashlib.sha256(b'{"entrypoint":"asset.usd","files":[')
            for index, item in enumerate(files):
                check_time()
                if index:
                    digest.update(b",")
                digest.update(_json(asdict(item)).encode())
            digest.update(b"]}")
            check_time()
            for directory, _, copied_names in os.walk(destination, topdown=False):
                check_time()
                for name in copied_names:
                    check_time()
                    os.chmod(Path(directory) / name, 292)
                os.chmod(directory, 365)
            check_time()
            snapshot = Snapshot(
                snapshot_id,
                row["episode_id"],
                digest.hexdigest(),
                str(destination),
                tuple(files),
                total,
                error_code="missing_entrypoint" if missing_entrypoint else None,
            )
        except _InvalidCandidate as exc:
            db.cleanup_paths.append(destination)
            snapshot = Snapshot(
                snapshot_id,
                row["episode_id"],
                None,
                None,
                (),
                total,
                error_code=str(exc),
            )
        except BaseException:
            db.cleanup_paths.append(destination)
            raise
        finally:
            if output_fd is not None:
                os.close(output_fd)
            if root_fd is not None:
                os.close(root_fd)
        db.execute(
            "INSERT INTO snapshots VALUES (?,?,?)",
            (snapshot_id, row["episode_id"], _json(asdict(snapshot))),
        )
        return snapshot

    def _job(self, db, row):
        return ValidationJob(
            **{
                name: row[name]
                for name in ValidationJob.__dataclass_fields__
                if name != "snapshot"
            },
            snapshot=self._load_snapshot(db, row["snapshot_id"])
            if row["snapshot_id"]
            else None,
        )

    def _reserve(self, episode_id, request_id, kind, snapshot=None):
        _identifier(request_id)
        with self._transaction() as db:
            episode = self._episode_row(db, episode_id)
            existing = db.execute(
                "SELECT * FROM jobs WHERE episode_id=? AND kind=? AND request_id=?",
                (episode_id, kind, request_id),
            ).fetchone()
            if existing:
                return self._job(db, existing)
            self._active(episode)
            count = db.execute(
                "SELECT count(*) FROM jobs WHERE episode_id=? AND kind=?",
                (episode_id, kind),
            ).fetchone()[0]
            if count >= episode[f"{kind}_limit"]:
                raise AdmissionError(f"{kind}_budget_exhausted")
            admitted_at = time.time()
            deadline_at = min(
                episode["deadline_at"],
                admitted_at
                + {"validation": 120, "preview": 90, "inference": 1800}[kind],
            )
            if kind != "inference":
                if snapshot is None:
                    snapshot = self._snapshot(db, episode, deadline_at=deadline_at)
                elif (
                    snapshot.episode_id != episode_id
                    or self._load_snapshot(db, snapshot.snapshot_id) != snapshot
                ):
                    raise AdmissionError("invalid_snapshot")
            job_id = uuid4().hex
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    episode_id,
                    kind,
                    request_id,
                    count + 1,
                    snapshot.snapshot_id if snapshot else None,
                    "reserved",
                    admitted_at,
                    deadline_at,
                    None,
                ),
            )
            return self._job(
                db,
                db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone(),
            )

    def reserve_validation(self, episode_id, request_id, snapshot=None):
        return self._reserve(episode_id, request_id, "validation", snapshot)

    def reserve_preview(self, episode_id, request_id, snapshot=None):
        return self._reserve(episode_id, request_id, "preview", snapshot)

    def reserve_inference(self, episode_id, request_id):
        """Use a fresh ID for EVERY upstream attempt, including model retries/compaction."""
        return self._reserve(episode_id, request_id, "inference")

    def complete_job(self, job_id, result):
        encoded = _json(result)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise AdmissionError("unknown_job")
            if row["status"] == "completed":
                if row["result_json"] != encoded:
                    raise AdmissionError("result_conflict")
            elif row["status"] in ("cancelled", "cancel_requested"):
                raise AdmissionError("job_cancelled")
            else:
                db.execute(
                    "UPDATE jobs SET status='completed',result_json=? WHERE job_id=?",
                    (encoded, job_id),
                )
            return self._job(
                db,
                db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone(),
            )

    def mark_job_running(self, job_id):
        with self._transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise AdmissionError("unknown_job")
            self._active(self._episode_row(db, row["episode_id"]))
            if row["status"] != "reserved" or time.time() >= row["deadline_at"]:
                raise AdmissionError("job_not_runnable")
            db.execute("UPDATE jobs SET status='running' WHERE job_id=?", (job_id,))
            return self._job(
                db,
                db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone(),
            )

    def _close(self, db, episode_id, reason):
        row = self._episode_row(db, episode_id)
        if row["status"] != "closed":
            db.execute(
                "UPDATE episodes SET status='closed',termination_reason=? WHERE episode_id=?",
                (_identifier(reason), episode_id),
            )
        db.execute(
            "UPDATE jobs SET status=CASE WHEN status='running'\n                   THEN 'cancel_requested' ELSE 'cancelled' END\n                   WHERE episode_id=? AND status IN ('reserved','running')",
            (episode_id,),
        )

    def close_episode(self, episode_id, reason):
        """Worker dispatch must honour cancelled/cancel_requested jobs and deadlines."""
        with self._transaction() as db:
            self._close(db, episode_id, reason)
        return self.get_episode(episode_id)

    def finalize_submission(
        self, episode_id, *, reason, request_id=None, candidate_error=None
    ):
        """Call only after Harbor has stopped actor writes. Never selects a past best."""
        if request_id is not None:
            _identifier(request_id)
        if candidate_error not in (None, "unsafe_remote_candidate_archive"):
            raise ValueError("Known trusted remote candidate admission error required")
        self.close_episode(episode_id, reason)
        with self._transaction() as db:
            row = self._episode_row(db, episode_id)
            if row["final_snapshot_id"]:
                return self._load_snapshot(db, row["final_snapshot_id"])
            if candidate_error is None:
                snapshot = self._snapshot(db, row)
            else:
                snapshot = Snapshot(
                    str(uuid4()), episode_id, None, None, error_code=candidate_error
                )
                db.execute(
                    "INSERT INTO snapshots VALUES (?,?,?)",
                    (snapshot.snapshot_id, episode_id, _json(asdict(snapshot))),
                )
            db.execute(
                "UPDATE episodes SET final_snapshot_id=? WHERE episode_id=?",
                (snapshot.snapshot_id, episode_id),
            )
            return snapshot
