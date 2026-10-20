import errno
import hashlib
import http.client
import io
import json
import logging
from pathlib import Path
import socket
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import urllib.error

import pytest

from simready_bench.engine import acquisition, downloads
from simready_bench.sources import import_source


class Response(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(payload)
        self.headers = {} if headers is None else headers


class InterruptedResponse(Response):
    def read(self, size=-1):
        if self.tell():
            raise ConnectionResetError("midstream reset")
        return super().read(3)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(downloads.time, "sleep", waits.append)
    return waits


def http_error(status):
    return urllib.error.HTTPError(
        "https://fixture.invalid/file?private=secret", status, "upstream", {}, None
    )


def queued_opener(monkeypatch, actions):
    calls = []
    selected = iter(actions)

    def open_url(url, *, timeout):
        calls.append((url, timeout))
        result = next(selected)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(downloads.urllib.request, "urlopen", open_url)
    return calls


@pytest.mark.parametrize("status", [429, 500, 501, 502, 503, 504, 599])
def test_transient_http_get_recovers(monkeypatch, no_wait, status):
    calls = queued_opener(
        monkeypatch, [http_error(status), Response(b'[1]', {"Link": "next-page"})]
    )
    assert downloads.read_json_page("https://fixture.invalid/tree") == ([1], "next-page")
    assert len(calls) == 2 and len(no_wait) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 425])
def test_other_http_failures_are_not_retried(monkeypatch, no_wait, status):
    error = http_error(status)
    calls = queued_opener(monkeypatch, [error])
    with pytest.raises(urllib.error.HTTPError) as caught:
        downloads.read_json_page("https://fixture.invalid/tree")
    assert caught.value is error
    assert len(calls) == 1 and not no_wait
    assert downloads.retry_receipt(error) is None


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(ConnectionResetError("TLS handshake reset")),
        ssl.SSLEOFError(8, "EOF while reading"),
        TimeoutError("socket timed out"),
        socket.gaierror(socket.EAI_AGAIN, "temporary DNS failure"),
        http.client.IncompleteRead(b"partial", 7),
    ],
)
def test_transient_network_failure_recovers(monkeypatch, error):
    calls = queued_opener(monkeypatch, [error, Response(b"[]")])
    assert downloads.read_json_page("https://fixture.invalid/tree") == ([], "")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        ssl.SSLCertVerificationError(1, "certificate verify failed"),
        urllib.error.URLError(ssl.SSLCertVerificationError(1, "certificate failed")),
        urllib.error.URLError("permanent URI error"),
        socket.gaierror(socket.EAI_NONAME, "no such host"),
        PermissionError(errno.EACCES, "access denied"),
        OSError(errno.ENOSPC, "disk full"),
    ],
)
def test_authority_certificate_and_filesystem_failures_fail_fast(
    monkeypatch, no_wait, error
):
    calls = queued_opener(monkeypatch, [error])
    with pytest.raises(type(error)) as caught:
        downloads.read_json_page("https://fixture.invalid/tree")
    assert caught.value is error
    assert len(calls) == 1 and not no_wait
    assert downloads.retry_receipt(error) is None


def test_exhaustion_preserves_exact_exception_and_get_receipt(monkeypatch, no_wait):
    errors = [ConnectionResetError(str(index)) for index in range(5)]
    calls = queued_opener(monkeypatch, errors)
    url = "https://fixture.invalid/tree?private=secret"
    with pytest.raises(ConnectionResetError) as caught:
        downloads.read_json_page(url)
    assert caught.value is errors[-1]
    assert len(calls) == 5 and len(no_wait) == 4
    receipt = downloads.retry_receipt(caught.value)
    assert receipt == {
        "schema": "simready-bench-download-exhaustion-v1",
        "method": "GET",
        "attempts": 5,
        "maximum_attempts": 5,
        "transient": True,
        "host": "fixture.invalid",
        "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
        "exception_class": "builtins.ConnectionResetError",
    }
    assert all(downloads.retry_receipt(error) is None for error in errors[:-1])
    receipt["attempts"] = 1
    assert downloads.retry_receipt(caught.value)["attempts"] == 5
    assert downloads.retry_receipt(ConnectionResetError("unrelated")) is None


def test_parallel_exhaustions_keep_separate_selected_exception_receipts(monkeypatch):
    errors = {
        "https://fixture.invalid/a": TimeoutError("first"),
        "https://fixture.invalid/b": ConnectionResetError("second"),
    }
    counts = {url: 0 for url in errors}
    lock = threading.Lock()

    def open_url(url, *, timeout):
        with lock:
            counts[url] += 1
        raise errors[url]

    def run(url):
        with pytest.raises(type(errors[url])) as caught:
            downloads.read_json_page(url)
        return caught.value

    monkeypatch.setattr(downloads.urllib.request, "urlopen", open_url)
    with ThreadPoolExecutor(max_workers=2) as pool:
        selected = list(pool.map(run, errors))
    for url, error in zip(errors, selected):
        assert error is errors[url] and counts[url] == 5
        assert downloads.retry_receipt(error)["url_sha256"] == hashlib.sha256(
            url.encode()
        ).hexdigest()


def test_retry_delay_is_exponential_jittered_and_capped(monkeypatch):
    monkeypatch.setattr(downloads.random, "uniform", lambda low, high: high)
    assert [downloads.retry_delay(index) for index in range(1, 5)] == [
        1.25,
        2.5,
        5.0,
        8.0,
    ]
    monkeypatch.setattr(downloads.random, "uniform", lambda low, high: low)
    assert [downloads.retry_delay(index) for index in range(1, 5)] == [
        0.75,
        1.5,
        3.0,
        6.0,
    ]


def test_retry_logs_omit_url_query_and_exception_payload(monkeypatch, caplog):
    queued_opener(
        monkeypatch,
        [ConnectionResetError("private-token-in-error"), Response(b"[]")],
    )
    with caplog.at_level(logging.WARNING, logger=downloads.__name__):
        downloads.read_json_page("https://fixture.invalid/tree?private=secret")
    assert "fixture.invalid" in caplog.text
    assert "private-token-in-error" not in caplog.text
    assert "private=secret" not in caplog.text


def test_catalogue_short_body_restarts_complete_get(monkeypatch):
    calls = queued_opener(
        monkeypatch,
        [Response(b"[", {"Content-Length": "2"}), Response(b"[]")],
    )
    assert downloads.read_json_page("https://fixture.invalid/tree") == ([], "")
    assert len(calls) == 2


@pytest.mark.parametrize("payload,headers", [(b"invalid", {}), (b"[]", {"Content-Length": "1"})])
def test_catalogue_content_failures_are_not_retried(monkeypatch, payload, headers):
    calls = queued_opener(monkeypatch, [Response(payload, headers)])
    with pytest.raises(ValueError):
        downloads.read_json_page("https://fixture.invalid/tree")
    assert len(calls) == 1


def test_file_midstream_reset_removes_partial_before_restarting(monkeypatch, tmp_path):
    temporary = tmp_path / "asset.downloading"
    responses = [InterruptedResponse(b"complete"), Response(b"complete")]
    calls = []

    def open_url(url, *, timeout):
        assert not temporary.exists()
        calls.append((url, timeout))
        return responses[len(calls) - 1]

    monkeypatch.setattr(downloads.urllib.request, "urlopen", open_url)
    downloads.download_file("https://fixture.invalid/asset", temporary, 8)
    assert temporary.read_bytes() == b"complete"
    assert len(calls) == 2 and all(response.closed for response in responses)


def test_short_file_exhausts_five_gets_and_removes_temporary(monkeypatch, tmp_path):
    temporary = tmp_path / "asset.downloading"
    calls = queued_opener(monkeypatch, [Response(b"short") for _ in range(5)])
    with pytest.raises(downloads.IncompleteTransferError) as caught:
        downloads.download_file("https://fixture.invalid/asset", temporary, 8)
    assert len(calls) == 5 and not temporary.exists()
    assert downloads.retry_receipt(caught.value)["method"] == "GET"


def test_oversized_file_is_not_retried_or_retained(monkeypatch, tmp_path):
    temporary = tmp_path / "asset.downloading"
    calls = queued_opener(monkeypatch, [Response(b"oversized")])
    with pytest.raises(ValueError, match="size mismatch") as caught:
        downloads.download_file("https://fixture.invalid/asset", temporary, 3)
    assert len(calls) == 1 and not temporary.exists()
    assert downloads.retry_receipt(caught.value) is None


def package_http(monkeypatch, *, broken_asset=False, broken_bom=False, reset_asset=False):
    asset = b"#usda 1.0\ndef Xform \"Asset\" {}\n"
    bom = json.dumps(
        {"items": [{"relative_path": "asset.usda", "size": len(asset), "hash": {"sha256": hashlib.sha256(asset).hexdigest()}}]},
        sort_keys=True,
    ).encode()
    bom_name = ".metadata/com.nvidia.simready.packaging.bom.json"
    packaging = json.dumps(
        {"metadata": [{"name": Path(bom_name).name, "hash": {"sha256": hashlib.sha256(bom).hexdigest()}}], "license": "fixture-licence"},
        sort_keys=True,
    ).encode()
    files = {"asset.usda": asset, bom_name: bom, "com.nvidia.simready.packaging.json": packaging}
    entries = [{"type": "file", "path": "fixture/" + name, "size": len(raw)} for name, raw in files.items()]
    served = dict(files)
    if broken_asset:
        served["asset.usda"] = asset.replace(b"Asset", b"Wrong")
    if broken_bom:
        served[bom_name] = bom.replace(b"asset.usda", b"other.usda")
    calls, lock = {}, threading.Lock()

    def open_url(url, *, timeout):
        with lock:
            calls[url] = calls.get(url, 0) + 1
            number = calls[url]
        if "/api/" in url:
            return Response(json.dumps(entries).encode())
        name = url.split("/fixture/", 1)[1]
        if reset_asset and name == "asset.usda" and number == 1:
            return InterruptedResponse(served[name])
        return Response(served[name])

    monkeypatch.setattr(downloads.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(acquisition.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))
    return files, calls


def test_actual_source_import_recovers_and_preserves_canonical_bom(monkeypatch, tmp_path):
    files, calls = package_http(monkeypatch, reset_asset=True)
    assets = tmp_path / "assets"
    assets.mkdir()
    source = {"kind": "nvidia-hf-packages", "revision": acquisition.REVISION, "packages": ["fixture"], "destination": "private-source"}
    receipt = import_source(source, base_dir=tmp_path, assets_root=assets)
    package = assets / "private-source/fixture"
    assert all((package / name).read_bytes() == raw for name, raw in files.items())
    verified = receipt["packages"][0]
    assert verified["revision"] == acquisition.REVISION
    assert verified["bom_sha256"] == hashlib.sha256(files[".metadata/com.nvidia.simready.packaging.bom.json"]).hexdigest()
    assert verified["verified_bom_files"] == [{"path": "asset.usda", "size": len(files["asset.usda"]), "sha256": hashlib.sha256(files["asset.usda"]).hexdigest()}]
    assert receipt["provider_calls"] == 0
    assert next(count for url, count in calls.items() if url.endswith("/asset.usda")) == 2
    assert not list(package.rglob("*.downloading"))


@pytest.mark.parametrize("failure,message", [("asset", "BOM mismatch"), ("bom", "BOM manifest hash mismatch")])
def test_actual_bom_integrity_failure_is_never_retried(
    monkeypatch, tmp_path, failure, message
):
    _, calls = package_http(monkeypatch, broken_asset=failure == "asset", broken_bom=failure == "bom")
    references = tmp_path / "references"
    references.mkdir()
    with pytest.raises(ValueError, match=message) as caught:
        acquisition.acquire("fixture", references)
    assert all(count == 1 for count in calls.values())
    assert downloads.retry_receipt(caught.value) is None
    assert not list(references.rglob("*.downloading"))


def test_existing_same_size_corrupt_source_still_fails_bom(monkeypatch, tmp_path):
    files, calls = package_http(monkeypatch)
    references = tmp_path / "references"
    references.mkdir()
    acquisition.acquire("fixture", references)
    original_calls = dict(calls)
    (references / "fixture/asset.usda").write_bytes(b"x" * len(files["asset.usda"]))
    with pytest.raises(ValueError, match="BOM mismatch"):
        acquisition.acquire("fixture", references)
    assert next(count for url, count in calls.items() if url.endswith("/asset.usda")) == 1
    assert all(calls[url] == count for url, count in original_calls.items() if "/api/" not in url)
