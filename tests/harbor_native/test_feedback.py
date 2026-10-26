"""Feedback quotas are durable; hostile archives never reach USD composition."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import threading
import urllib.error
import urllib.request

import pytest

from simready_eval.feedback import FeedbackLedger, FeedbackService, safe_extract


def archive(files=None):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, contents in (files or {"asset.usd": b"#usda 1.0\n"}).items():
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            tar.addfile(member, io.BytesIO(contents))
    return stream.getvalue()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_feedback_worker_file_bound_permits_declared_archive_expansion(tmp_path):
    import subprocess
    import sys
    from simready_eval.feedback import MAX_RESPONSE

    path = tmp_path / 'expanded-member'
    size = MAX_RESPONSE + 1
    result = subprocess.run([sys.executable, '-c',
        'from simready_eval.feedback import _worker_limits; import sys; '
        '_worker_limits(); f=open(sys.argv[1], "wb"); f.truncate(int(sys.argv[2])); f.close()',
        str(path), str(size)], capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr.decode()
    assert path.stat().st_size == size


def test_atomic_five_call_quota_idempotency_and_restart(tmp_path):
    ledger = FeedbackLedger(tmp_path / "state")
    status, body, accepted = ledger.reserve("a", "a" * 64)
    assert (status, accepted, body["status"]) == (202, True, "pending")
    assert ledger.reserve("a", "a" * 64)[2] is False
    assert ledger.reserve("a", "b" * 64)[0] == 409
    ledger.finish("a", {"status": "complete", "score_percent": 12})
    assert ledger.reserve("a", "a" * 64)[1]["score_percent"] == 12
    ledger.reserve("reserved-before-crash", "c" * 64)
    recovered = FeedbackLedger(tmp_path / "state")
    result = recovered.reserve("reserved-before-crash", "c" * 64)[1]
    assert result["status"] == "interrupted_unknown" and result["score_percent"] is None
    for i in range(3):
        assert recovered.reserve(f"n{i}", str(i) * 64)[2]
    assert recovered.reserve("sixth", "f" * 64)[0] == 429
    assert recovered.health()["used"] == 5
    assert (tmp_path / "state").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "state" / "feedback.sqlite3").stat().st_mode & 0o777 == 0o600


def test_concurrent_reservation_never_exceeds_five(tmp_path):
    ledger = FeedbackLedger(tmp_path / "state")
    results = []
    threads = [threading.Thread(target=lambda i=i: results.append(ledger.reserve(str(i), str(i % 10) * 64))) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(result[2] for result in results) == 5
    assert ledger.health()["remaining"] == 0


@pytest.mark.parametrize("kind", ["traversal", "absolute", "symlink", "hardlink", "fifo", "duplicate", "oversize", "backslash"])
def test_safe_tar_extraction_rejects_unsafe_members(tmp_path, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        member = tarfile.TarInfo({"traversal": "../escape", "absolute": "/escape", "backslash": "a\\b"}.get(kind, "asset.usd"))
        if kind in {"symlink", "hardlink"}:
            member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../escape"
        elif kind == "fifo":
            member.type = tarfile.FIFOTYPE
        elif kind == "oversize":
            member.size = 257 * 1024 * 1024
        tar.addfile(member)
        if kind == "duplicate":
            tar.addfile(member)
    with pytest.raises(ValueError):
        safe_extract(stream.getvalue(), tmp_path / "unpacked")
    assert not (tmp_path / "escape").exists()


def post(url, data, request_id, *, hash_value=None):
    request = urllib.request.Request(url + "/validate", data=data, headers={
        "X-Request-ID": request_id, "X-Content-SHA256": hash_value or digest(data),
        "Content-Type": "application/x-tar"})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        response = error
    return response.status, json.loads(response.read())


def test_real_http_budget_and_sanitized_reports(tmp_path):
    called = []
    def evaluate(root):
        called.append(root)
        return {"outcome": "incomplete", "score_percent": 42, "reward": 42,
                "nvidia_validator_report": {"logs": ["secret"], "asset_path": "/private/target.usd",
                    "issues": [{"rule_id": "RULE", "message": "failed /private/target.usd token=secretvalue"}]}}
    service = FeedbackService(tmp_path / "state", evaluator=evaluate)
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        data = archive()
        first = post(url, data, "first")
        assert first[0] == 200 and first[1]["result"]["score_percent"] == 42
        assert post(url, data, "first") == first
        assert post(url, archive({"different": b"x"}), "first")[0] == 409
        assert len(called) == 1
        public = json.dumps(first[1])
        assert "/private" not in public and "secretvalue" not in public and '"logs"' not in public
        for i in range(4):
            assert post(url, b"invalid tar", f"bad{i}")[1]["result"]["outcome"] == "invalid_submission"
        assert post(url, data, "sixth")[0] == 429
        with urllib.request.urlopen(url + "/health") as response:
            assert json.load(response)["used"] == 5
        assert post(url, data, "wronghash", hash_value="0" * 64)[0] == 400
    finally:
        server.shutdown()
        server.server_close()
        service.close()
    restarted = FeedbackService(tmp_path / "state", evaluator=evaluate)
    assert restarted.ledger.health()["used"] == 5
    assert restarted.validate(data, "sixth", digest(data))[0] == 429
    restarted.close()


def test_pending_duplicate_does_not_dispatch_twice(tmp_path):
    started, release = threading.Event(), threading.Event()
    def evaluate(root):
        started.set()
        assert release.wait(5)
        return {"outcome": "incomplete", "score_percent": 0}
    service = FeedbackService(tmp_path / "state", evaluator=evaluate)
    data = archive()
    worker = threading.Thread(target=lambda: service.validate(data, "same", digest(data)))
    worker.start()
    assert started.wait(5)
    assert service.validate(data, "same", digest(data))[0] == 503
    release.set()
    worker.join()
    assert service.ledger.health()["used"] == 1
    service.close()


def test_production_service_requires_privilege_separation(tmp_path, monkeypatch):
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    with pytest.raises(PermissionError, match="root"):
        FeedbackService(tmp_path / "state")


def test_real_sdk_through_http_candidate_only(tmp_path):
    pytest.importorskip("simready.validate")
    from simready_eval.formal import evaluate
    service = FeedbackService(tmp_path / "state", evaluator=evaluate)
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    data = archive({"asset.usd": b'#usda 1.0\n(defaultPrim = "Asset"\n upAxis = "Z"\n metersPerUnit = 1)\ndef Xform "Asset" {\n def Cube "Body" {}\n}\n'})
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        result = post(url, data, "genuine")[1]["result"]
        assert result["simready"]["score_percent"] is not None
        assert len(result["simready"]["checks"]) == 46
        assert len(result["optional_score"]["checks"]) == 51
        for index in range(1, 5):
            status, reply = post(url, data, f"genuine-{index}")
            assert status == 200 and reply["result"]["simready"]["score_percent"] is not None
        assert post(url, data, "sixth")[0] == 429
    finally:
        server.shutdown()
        server.server_close()
        service.close()
    restarted = FeedbackService(tmp_path / "state", evaluator=evaluate)
    try:
        assert restarted.ledger.health()["used"] == 5
        assert restarted.validate(data, "genuine", digest(data))[1]["result"] == result
        assert restarted.validate(data, "sixth", digest(data))[0] == 429
    finally:
        restarted.close()


def test_production_worker_identity_clean_environment_deadline_and_kill(tmp_path, monkeypatch):
    import os
    import subprocess
    from simready_eval import feedback
    # Explicit test seam permits testing launch arguments without changing UID.
    service = FeedbackService(tmp_path / "state", evaluator=lambda root: {})
    service.evaluator = None
    monkeypatch.setenv("SECRET_TOKEN", "must-not-inherit")
    monkeypatch.setattr(os, "chown", lambda *args, **kwargs: None)
    captured, waits, killed = {}, [], []
    class Process:
        pid = 123456789
        returncode = -9
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self, timeout=None):
            waits.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired("validator", timeout)
            return self.returncode
    def launch(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    package, output = tmp_path / "candidate", tmp_path / "worker"
    package.mkdir()
    output.mkdir()
    result = service._evaluate(package, output)
    assert result["score_percent"] is None
    assert result["infrastructure_errors"][0]["code"] == "validator_timeout"
    assert captured["user"] == captured["group"] == 65534 and captured["extra_groups"] == []
    assert captured["start_new_session"] is True and captured["close_fds"] is True
    assert "-I" in captured["command"] and "_worker" in captured["command"]
    assert "SECRET_TOKEN" not in captured["env"]
    assert 0 < waits[0] <= 120 and waits[-1] is None
    assert killed == [(Process.pid, feedback.signal.SIGKILL)]
    service.close()


def test_worker_result_symlink_fifo_and_oversize_rejected(tmp_path):
    from simready_eval.feedback import _read_worker_result, MAX_RESPONSE
    secret = tmp_path / "secret"
    secret.write_text('{"secret":"private"}')
    link = tmp_path / "result"
    link.symlink_to(secret)
    with pytest.raises(OSError):
        _read_worker_result(link)
    link.unlink()
    import os
    os.mkfifo(link)
    with pytest.raises(ValueError):
        _read_worker_result(link)
    link.unlink()
    with link.open("wb") as output:
        output.truncate(MAX_RESPONSE + 1)
    with pytest.raises(ValueError):
        _read_worker_result(link)


def test_client_real_http_round_trip_and_idempotency(tmp_path):
    from simready_eval.feedback import validate_candidate
    package = tmp_path / "package"
    package.mkdir()
    (package / "asset.usd").write_text("#usda 1.0\n")
    calls = []
    service = FeedbackService(tmp_path / "state", evaluator=lambda root: calls.append(root) or {"outcome": "incomplete", "score_percent": 23})
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = validate_candidate(package, "client", url)
        assert first[0] == 200 and first[1]["result"]["score_percent"] == 23
        assert validate_candidate(package, "client", url) == first
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def test_service_lock_blocks_parallel_controller_and_missing_sdk_stays_unknown(tmp_path):
    from simready_eval.formal import score_validation
    state = tmp_path / "state"
    service = FeedbackService(state, evaluator=lambda root: {"outcome": "infrastructure_error", "score_percent": None, "simready": score_validation(None)})
    with pytest.raises(BlockingIOError):
        FeedbackService(state, evaluator=lambda root: pytest.fail("second controller"))
    data = archive()
    result = service.validate(data, "sdk-unavailable", digest(data))[1]["result"]
    assert result["score_percent"] is None
    assert result["simready"]["unknown_count"] == 46
    assert service.ledger.health()["used"] == 1
    service.close()


def test_timeout_result_consumes_call_and_does_not_retry(tmp_path):
    calls = []
    from simready_eval.feedback import unknown
    service = FeedbackService(tmp_path / "state", evaluator=lambda root: calls.append(root) or unknown("validator_timeout"))
    data = archive()
    first = service.validate(data, "timeout", digest(data))
    assert first[1]["result"]["score_percent"] is None
    assert service.validate(data, "timeout", digest(data)) == first
    assert len(calls) == 1 and service.ledger.health()["used"] == 1
    service.close()


def test_implicit_archive_directories_count_toward_extract_limit(tmp_path, monkeypatch):
    from simready_eval import feedback
    monkeypatch.setattr(feedback, "MAX_FILES", 3)
    data = archive({"one/two/three/four/asset.usd": b"#usda 1.0\n"})
    with pytest.raises(ValueError, match="limit"):
        safe_extract(data, tmp_path / "candidate")


def test_controller_reclaims_worker_output_after_exit_before_read(tmp_path, monkeypatch):
    import os
    import subprocess
    from simready_eval import feedback
    service = FeedbackService(tmp_path / "state", evaluator=lambda root: {})
    service.evaluator = None
    package, output = tmp_path / "candidate", tmp_path / "worker"
    package.mkdir()
    output.mkdir()
    result_path = output / "result.json"
    events = []
    monkeypatch.setattr(os, "chown", lambda path, uid, gid, **kwargs: events.append(("chown", Path(path), uid, kwargs)))
    class Process:
        pid = 123456789
        returncode = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def wait(self, timeout=None):
            events.append(("wait", timeout))
            result_path.write_text('{"outcome":"incomplete","score_percent":12}')
            result_path.chmod(0o600)
            return 0
    captured = {}
    def launch(command, **kwargs):
        captured.update(kwargs)
        return Process()
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(os, "killpg", lambda *args: events.append(("kill",)))
    real_read = feedback._read_worker_result
    def read(path):
        assert ("chown", output, 0, {"follow_symlinks": False}) in events
        assert ("chown", result_path, 0, {"follow_symlinks": False}) in events
        reclaimed = events.index(("chown", output, 0, {"follow_symlinks": False}))
        assert events.index(("kill",)) < reclaimed
        assert events.index(("wait", None)) < reclaimed
        return real_read(path)
    monkeypatch.setattr(feedback, "_read_worker_result", read)
    assert service._evaluate(package, output)["score_percent"] == 12
    # chdir happens before setuid inside Popen, so its cwd must be root accessible.
    assert captured["cwd"] == output.parent
    service.close()


def test_candidate_limit_is_separate_from_tar_overhead(tmp_path, monkeypatch):
    from simready_eval import feedback
    monkeypatch.setattr(feedback, "CANDIDATE_BYTES", 8, raising=False)
    monkeypatch.setattr(feedback, "MAX_ARCHIVE", 20 * 1024)
    package = tmp_path / "package"
    package.mkdir()
    (package / "asset.usd").write_bytes(b"12345678")
    data = feedback.candidate_archive(package)
    assert 8 < len(data) <= feedback.MAX_ARCHIVE
    assert (safe_extract(data, tmp_path / "exact") / "asset.usd").read_bytes() == b"12345678"
    (package / "extra").write_bytes(b"9")
    with pytest.raises(ValueError, match="limit"):
        feedback.candidate_archive(package)
    with pytest.raises(ValueError, match="limit"):
        safe_extract(archive({"asset.usd": b"12345678", "extra": b"9"}), tmp_path / "above")


def test_busy_http_rejected_before_body_without_quota_and_health_works(tmp_path):
    import http.client
    started, release = threading.Event(), threading.Event()
    calls = []
    def evaluate(root):
        calls.append(root)
        started.set()
        assert release.wait(5)
        return {"outcome": "incomplete", "score_percent": 12}
    service = FeedbackService(tmp_path / "state", evaluator=evaluate)
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    data = archive()
    active = threading.Thread(target=lambda: service.validate(data, "active", digest(data)))
    active.start()
    assert started.wait(5)
    connection = http.client.HTTPConnection(*server.server_address, timeout=1)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        connection.putrequest("POST", "/validate")
        connection.putheader("Content-Length", str(len(data)))
        connection.putheader("X-Request-ID", "busy")
        connection.putheader("X-Content-SHA256", digest(data))
        connection.endheaders()
        # No body is sent. A busy server must respond without reading it.
        response = connection.getresponse()
        assert response.status == 503
        assert int(response.getheader("Retry-After")) >= 1
        assert response.getheader("Connection") == "close"
        assert json.loads(response.read())["error"] == "feedback_busy"
        with urllib.request.urlopen(url + "/health", timeout=1) as health:
            assert json.load(health)["used"] == 1
        assert service.validate(data, "direct-busy", digest(data))[0] == 503
        assert len(calls) == 1 and service.ledger.health()["used"] == 1
    finally:
        connection.close()
        release.set()
        active.join(5)
    try:
        assert post(url, data, "after-release")[0] == 200
        assert len(calls) == 2 and service.ledger.health()["used"] == 2
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def test_single_inflight_slot_released_after_upload_and_worker_errors(tmp_path):
    calls = []
    def evaluate(root):
        calls.append(root)
        if len(calls) == 1:
            raise RuntimeError("test worker infrastructure failure")
        return {"outcome": "incomplete", "score_percent": 12}
    service = FeedbackService(tmp_path / "state", evaluator=evaluate)
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    data = archive()
    try:
        assert post(url, data, "bad-hash", hash_value="0" * 64)[0] == 400
        assert service.ledger.health()["used"] == 0
        first = post(url, data, "worker-error")
        assert first[0] == 200 and first[1]["result"]["score_percent"] is None
        second = post(url, data, "after-error")
        assert second[0] == 200 and second[1]["result"]["score_percent"] == 12
        assert service.ledger.health()["used"] == 2
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def test_worker_file_limit_is_candidate_allowance_not_response_limit(monkeypatch):
    import ctypes
    import resource
    from simready_eval import feedback
    observed = {}
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (resource.RLIM_INFINITY,) * 2)
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: observed.__setitem__(kind, limits))
    class Lib:
        def prctl(self, *args):
            return 0
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: Lib())
    feedback._worker_limits()
    assert observed[resource.RLIMIT_FSIZE] == (1024 ** 3,) * 2
    assert feedback.MAX_RESPONSE == 8 * 1024 ** 2
