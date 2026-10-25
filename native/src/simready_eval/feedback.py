"""Candidate-only formal feedback, with a durable five-call controller ledger.

The production controller runs as root. Candidate USD is composed only in a
separate UID/GID 65534 subprocess, with no inherited credentials or ledger access.
A callable evaluator is an explicit in-process test seam; the serve CLI never uses it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import sqlite3
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid

MAX_ARCHIVE = 256 * 1024 * 1024
MAX_FILES = 10000
MAX_RESPONSE = 8 * 1024 * 1024
MAX_CALLS = 5
WORKER_UID = WORKER_GID = 65534
REQUEST_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")


def _json(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def unknown(reason):
    return {"outcome": "infrastructure_error", "score_percent": None,
            "reward": None, "infrastructure_errors": [{"code": reason}]}


class FeedbackLedger:
    """Atomic persistent quota. Opening a ledger recovers previously reserved jobs.

    A FeedbackService holds an exclusive lifetime controller lock. This class's
    standalone API is intended for ledger tests and offline inspection.
    """
    def __init__(self, state):
        self.state = Path(state).absolute()
        if self.state.is_symlink():
            raise ValueError("State must be a private real directory")
        self.state.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.state.stat().st_uid != os.geteuid():
            raise PermissionError("State owner must be the controller")
        self.state.chmod(0o700)
        self.path = self.state / "feedback.sqlite3"
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_uid != os.geteuid():
                raise ValueError("Ledger must be a private regular file")
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        with self._transaction() as db:
            db.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS calls (request_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, status TEXT NOT NULL, result TEXT)")
            config = dict(db.execute("SELECT key,value FROM config"))
            expected = {"max_calls": str(MAX_CALLS), "schema": "simready-feedback-v1"}
            if not config:
                config = {**expected, "trial": uuid.uuid4().hex}
                db.executemany("INSERT INTO config VALUES (?,?)", config.items())
            elif any(config.get(key) != value for key, value in expected.items()) or not config.get("trial"):
                raise ValueError("Incompatible server-owned feedback configuration")
            interrupted = _json({"status": "interrupted_unknown", "result": unknown("interrupted_unknown"), "score_percent": None}).decode()
            db.execute("UPDATE calls SET status='finished', result=? WHERE status='pending'", (interrupted,))

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def health(self):
        with self._transaction() as db:
            used = db.execute("SELECT count(*) FROM calls").fetchone()[0]
        return {"status": "ok", "limit": MAX_CALLS, "used": used, "remaining": max(0, MAX_CALLS - used)}

    def reserve(self, request_id, content_hash):
        if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id) or not isinstance(content_hash, str) or not SHA256.fullmatch(content_hash):
            return 400, {"error": "invalid_request_identity"}, False
        with self._transaction() as db:
            row = db.execute("SELECT content_hash,status,result FROM calls WHERE request_id=?", (request_id,)).fetchone()
            if row:
                if row[0] != content_hash:
                    return 409, {"error": "request_id_content_conflict"}, False
                return (202, {"status": "pending", "request_id": request_id}, False) if row[1] == "pending" else (200, json.loads(row[2]), False)
            count = db.execute("SELECT count(*) FROM calls").fetchone()[0]
            if count >= MAX_CALLS:
                return 429, {"error": "feedback_budget_exhausted", "limit": MAX_CALLS, "used": count, "remaining": 0}, False
            db.execute("INSERT INTO calls VALUES (?,?,'pending',NULL)", (request_id, content_hash))
        return 202, {"status": "pending", "request_id": request_id}, True

    def finish(self, request_id, result):
        encoded = _json(result)
        if len(encoded) > MAX_RESPONSE:
            raise ValueError("Feedback response exceeds bound")
        with self._transaction() as db:
            changed = db.execute("UPDATE calls SET status='finished',result=? WHERE request_id=? AND status='pending'", (encoded.decode(), request_id)).rowcount
            if changed != 1:
                raise ValueError("Only a reserved call may be completed")


def safe_extract(data, destination):
    """Extract only bounded plain tar regular files/directories, before USD imports."""
    if len(data) > MAX_ARCHIVE:
        raise ValueError("archive_limit_exceeded")
    root = Path(destination)
    root.mkdir(mode=0o700)
    total, count, seen, entries = 0, 0, set(), set()
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
            for item in archive:
                count += 1
                path = PurePosixPath(item.name)
                normalized = path.as_posix()
                if (count > MAX_FILES or len(item.name) > 4096 or not path.parts
                    or path.is_absolute() or ".." in path.parts
                    or any(c in item.name for c in ("\\", ":", "\x00"))
                    or normalized in seen or not (item.isfile() or item.isdir())
                    or item.size < 0):
                    raise ValueError("unsafe_archive_member")
                seen.add(normalized)
                # Implicit parent directories also consume the extraction budget.
                entries.update(path.parents)
                entries.add(path)
                entries.discard(PurePosixPath("."))
                if len(entries) > MAX_FILES:
                    raise ValueError("expanded_package_file_limit_exceeded")
                total += item.size
                if total > MAX_ARCHIVE:
                    raise ValueError("expanded_package_limit_exceeded")
                target = root.joinpath(*path.parts)
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o755)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = archive.extractfile(item)
                if source is None:
                    raise ValueError("invalid_archive")
                with source, target.open("xb") as output:
                    remaining = item.size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("truncated_archive")
                        output.write(chunk)
                        remaining -= len(chunk)
                target.chmod(0o444)
        for folder in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
            folder.chmod(0o555)
        root.chmod(0o555)
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ValueError("invalid_archive") from exc
    return root


def sanitize(value):
    """Keep useful public scoring and findings; omit logs, inventory and raw paths."""
    def clean(item, depth=0):
        if depth > 25:
            return "[truncated]"
        if item is None or type(item) in (bool, int):
            return item
        if isinstance(item, float):
            return item if math.isfinite(item) else None
        if isinstance(item, str):
            item = item[:4096]
            item = re.sub(r"(?i)\b(?:bearer\s+\S+|(?:token|secret|password|api[_-]?key)\s*[:=]\s*\S+)", "[redacted]", item)
            item = re.sub(r"\b(?:sk-[A-Za-z0-9_-]+|hf_[A-Za-z0-9]+|AKIA[A-Z0-9]{16})\b", "[redacted]", item)
            return re.sub(r"(?:[A-Za-z]:[\\/]|/)[A-Za-z0-9_~.][^\s\"'<>]*", "[path]", item)
        if isinstance(item, list):
            return [clean(child, depth + 1) for child in item[:1000]]
        if isinstance(item, dict):
            return {str(key)[:128]: clean(child, depth + 1) for key, child in list(item.items())[:1000]
                    if isinstance(key, str) and key not in {"logs", "log", "stdout", "stderr", "traceback", "asset_path", "asset", "path", "prim_path", "inventory"}}
        return None
    allowed = {key: value[key] for key in ("schema_version", "outcome", "score_percent", "reward", "simready", "optional_score", "coverage", "infrastructure_errors", "findings", "package_findings") if key in value}
    raw = value.get("nvidia_validator_report")
    if isinstance(raw, dict):
        allowed["findings"] = [
            {key: row[key] for key in ("rule_id", "rule", "severity", "message") if key in row}
            for name in ("issues", "soft_issues", "warnings", "infos")
            for row in (raw.get(name) or [])[:1000] if isinstance(row, dict)
        ]
    preflight = value.get("preflight")
    if isinstance(preflight, dict):
        allowed["package_findings"] = preflight.get("findings", [])
    result = clean(allowed)
    if len(_json(result)) > MAX_RESPONSE - 1024:
        return unknown("feedback_response_limit_exceeded")
    return result


def _worker_limits():
    for kind, ceiling in ((resource.RLIMIT_CPU, 120), (resource.RLIMIT_AS, 4 * 1024**3),
                          (resource.RLIMIT_FSIZE, MAX_RESPONSE), (resource.RLIMIT_NOFILE, 128),
                          (resource.RLIMIT_NPROC, 128), (resource.RLIMIT_CORE, 0)):
        hard = resource.getrlimit(kind)[1]
        resource.setrlimit(kind, (min(ceiling, hard) if hard != resource.RLIM_INFINITY else ceiling,) * 2)
    # setuid already happened in Popen, before Python startup. Prevent privilege gain.
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0) != 0:
        raise OSError("Cannot establish no_new_privs")


def _read_worker_result(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_RESPONSE:
            raise ValueError("Invalid worker result")
        data = source.read(MAX_RESPONSE + 1)
    if len(data) > MAX_RESPONSE:
        raise ValueError("Worker response limit exceeded")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("Worker result must be an object")
    return result


class FeedbackService:
    def __init__(self, state, *, evaluator=None, deadline=120):
        if evaluator is None and os.geteuid() != 0:
            raise PermissionError("Production feedback requires root controller with isolated worker UID 65534")
        if not 0 < deadline <= 120:
            raise ValueError("Worker deadline must be at most 120 seconds")
        self.evaluator, self.deadline = evaluator, deadline
        state = Path(state).absolute()
        if state.is_symlink():
            raise ValueError("State must be a real directory")
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        if state.stat().st_uid != os.geteuid():
            raise PermissionError("State must belong to controller")
        state.chmod(0o700)
        self._lock = os.open(state / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.ledger = FeedbackLedger(state)
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "_lock", None) is not None:
            os.close(self._lock)
            self._lock = None

    def _evaluate(self, package, output):
        if self.evaluator is not None:
            return self.evaluator(package)
        output.chmod(0o700)
        os.chown(output, WORKER_UID, WORKER_GID)
        result_path = output / "result.json"
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(output),
               "TMPDIR": str(output), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "PXR_WORK_THREAD_LIMIT": "2"}
        try:
            # Popen changes cwd before dropping UID: root without DAC_OVERRIDE
            # can enter this root-owned parent, but not the worker's 0700 output.
            with subprocess.Popen([sys.executable, "-I", "-m", "simready_eval.feedback", "_worker",
                                   "--package", str(package), "--result", str(result_path)],
                                  cwd=output.parent, env=env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  user=WORKER_UID, group=WORKER_GID, extra_groups=[],
                                  close_fds=True, start_new_session=True) as process:
                expires = time.monotonic() + self.deadline
                try:
                    process.wait(timeout=max(0.001, expires - time.monotonic()))
                except subprocess.TimeoutExpired:
                    return unknown("validator_timeout")
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                if process.returncode != 0:
                    return unknown("validator_worker_failed")
        finally:
            # CAP_CHOWN suffices; the controller intentionally lacks DAC_OVERRIDE.
            # Reclaim even failed/timed-out jobs so private output can be removed.
            if not stat.S_ISDIR(output.lstat().st_mode):
                raise ValueError("Worker output must remain a real directory")
            os.chown(output, 0, 0, follow_symlinks=False)
            output.chmod(0o700)
        info = result_path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_RESPONSE:
            raise ValueError("Invalid worker result")
        os.chown(result_path, 0, 0, follow_symlinks=False)
        return _read_worker_result(result_path)

    def validate(self, data, request_id, content_hash):
        if len(data) > MAX_ARCHIVE:
            return 413, {"error": "archive_limit_exceeded"}
        if not isinstance(content_hash, str) or not SHA256.fullmatch(content_hash) or hashlib.sha256(data).hexdigest() != content_hash:
            return 400, {"error": "content_hash_mismatch"}
        status, response, accepted = self.ledger.reserve(request_id, content_hash)
        if not accepted:
            return status, response
        try:
            with tempfile.TemporaryDirectory(prefix="simready-feedback-", dir="/tmp") as scratch:
                base = Path(scratch)
                base.chmod(0o711)
                try:
                    package = safe_extract(data, base / "candidate")
                except ValueError:
                    from simready_eval.formal import score_validation
                    score = score_validation(None, candidate_invalid=True)
                    result = {"outcome": "invalid_submission", "score_percent": 0, "reward": 0,
                              "simready": score, "optional_score": score["optional_score"],
                              "package_findings": [{"code": "unsafe_or_invalid_archive"}]}
                else:
                    output = base / "worker"
                    output.mkdir(mode=0o700)
                    result = sanitize(self._evaluate(package, output))
        except Exception:
            result = unknown("feedback_worker_error")
        response = {"status": "complete", "request_id": request_id, "result": result}
        self.ledger.finish(request_id, response)
        return 200, response

    def make_server(self, host="0.0.0.0", port=8080):
        service = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def setup(self):
                super().setup()
                self.connection.settimeout(15)

            def reply(self, status, body):
                data = _json(body)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self.reply(200, service.ledger.health()) if self.path == "/health" else self.reply(404, {"error": "not_found"})

            def do_POST(self):
                if self.path != "/validate":
                    return self.reply(404, {"error": "not_found"})
                length = self.headers.get_all("Content-Length", [])
                if self.headers.get("Transfer-Encoding") is not None or len(length) != 1 or not re.fullmatch(r"[0-9]{1,12}", length[0]):
                    return self.reply(400, {"error": "bounded_content_length_required"})
                size = int(length[0])
                if size > MAX_ARCHIVE:
                    return self.reply(413, {"error": "archive_limit_exceeded"})
                request_ids = self.headers.get_all("X-Request-ID", [])
                hashes = self.headers.get_all("X-Content-SHA256", [])
                if len(request_ids) != 1 or len(hashes) != 1 or not REQUEST_ID.fullmatch(request_ids[0]) or not SHA256.fullmatch(hashes[0]):
                    return self.reply(400, {"error": "invalid_request_identity"})
                try:
                    data = self.rfile.read(size)
                except (OSError, TimeoutError):
                    return self.reply(408, {"error": "upload_timeout"})
                if len(data) != size:
                    return self.reply(400, {"error": "truncated_upload"})
                status, result = service.validate(data, request_ids[0], hashes[0])
                self.reply(status, result)
        return ThreadingHTTPServer((host, port), Handler)


def candidate_archive(package):
    root = Path(package)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Package must be a regular directory")
    data = io.BytesIO()
    count, total = 0, 0
    with tarfile.open(fileobj=data, mode="w") as archive:
        for path in sorted(root.rglob("*")):
            info = path.lstat()
            count += 1
            if count > MAX_FILES or (not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode)) or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1):
                raise ValueError("Unsafe package file")
            if path.is_dir():
                continue
            total += info.st_size
            if total > MAX_ARCHIVE:
                raise ValueError("Package exceeds upload limit")
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
            if data.tell() > MAX_ARCHIVE:
                raise ValueError("Archive exceeds upload limit")
    result = data.getvalue()
    if len(result) > MAX_ARCHIVE:
        raise ValueError("Archive exceeds upload limit")
    return result


def validate_candidate(package, request_id, url):
    if not REQUEST_ID.fullmatch(request_id):
        raise ValueError("Invalid request ID")
    if not url.startswith(("http://", "https://")):
        raise ValueError("Feedback URL must use HTTP")
    data = candidate_archive(package)
    request = urllib.request.Request(url.rstrip("/") + "/validate", data=data, headers={
        "Content-Type": "application/x-tar", "X-Request-ID": request_id,
        "X-Content-SHA256": hashlib.sha256(data).hexdigest()})
    try:
        response = urllib.request.urlopen(request, timeout=140)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        body = response.read(MAX_RESPONSE + 1)
        if len(body) > MAX_RESPONSE:
            raise ValueError("Feedback response exceeds limit")
        return response.status, json.loads(body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--state", type=Path, required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    client = sub.add_parser("validate_candidate")
    client.add_argument("--package", type=Path, required=True)
    client.add_argument("--request-id", required=True)
    client.add_argument("--url", default="http://feedback:8080")
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--package", type=Path, required=True)
    worker.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        service = FeedbackService(args.state)
        try:
            with service.make_server(args.host, args.port) as server:
                server.serve_forever()
        finally:
            service.close()
        return 0
    if args.command == "validate_candidate":
        status, result = validate_candidate(args.package, args.request_id, args.url)
        print(_json(result).decode())
        return 0 if status < 400 else 1
    if os.geteuid() != WORKER_UID or os.getegid() != WORKER_GID or os.getgroups():
        raise PermissionError("Formal worker must run as isolated UID/GID 65534")
    _worker_limits()
    os.chdir(args.result.parent)
    from simready_eval.formal import evaluate
    result = evaluate(args.package)
    data = _json(sanitize(result))
    with args.result.open("xb") as output:
        output.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
