"""Offline tests of role boundaries and immutable selective setup acquisition."""
from __future__ import annotations

import copy
import hashlib
import http.server
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest


REPO = "hcltech-robotics/simreadybench"
REVISION = "1" * 40
PREFIX = "benchmark/nv100/v1/tasks/widget-001/"
PUBLIC = {"public/observation.png": b"observation png", "public/instruction.md": b"Build the widget."}


def manifest_for(payloads=None, *, role="actor", revision=REVISION):
    payloads = PUBLIC if payloads is None else payloads
    return {
        "schema": "simready-role-assets-v1", "repo_id": REPO,
        "revision": revision, "task_id": "widget-001", "role": role,
        "files": [{"path": PREFIX + path, "destination": path.split("/", 1)[1],
                   "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for path, data in payloads.items()],
    }


def fetch_module():
    # A missing implementation fails an assertion rather than collection.
    assert importlib.util.find_spec("simready_eval.fetch") is not None, "selective fetch helper is missing"
    return importlib.import_module("simready_eval.fetch")


@pytest.fixture
def hub(tmp_path):
    """Serve real HTTP bytes; replace only the external Hub transport seam."""
    payloads = {PREFIX + path: data for path, data in PUBLIC.items()}
    requests = []
    failures = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path[1:])
            if failures:
                status = failures.pop(0)
                self.send_error(status)
                return
            data = payloads.get(self.path[1:])
            if data is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    downloads = tmp_path / "http-downloads"
    downloads.mkdir()
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == REPO
        assert kwargs["repo_type"] == "dataset"
        assert len(kwargs["revision"]) == 40
        assert kwargs["token"] is False
        target = downloads / str(len(calls))
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/{kwargs['filename']}", timeout=2) as response:
            target.write_bytes(response.read())
        return str(target)

    yield {"downloader": downloader, "requests": requests, "payloads": payloads, "failures": failures, "calls": calls, "url": f"http://127.0.0.1:{server.server_port}/"}
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_cold_fetch_is_exact_and_warm_cache_works_without_transport(tmp_path, hub):
    fetch = fetch_module()
    manifest = manifest_for()
    receipt = fetch.fetch_assets(manifest, tmp_path / "first", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert set(hub["requests"]) == {PREFIX + "public/observation.png", PREFIX + "public/instruction.md"}
    assert (tmp_path / "first/observation.png").read_bytes() == b"observation png"
    assert (tmp_path / "first/instruction.md").read_bytes() == b"Build the widget."
    assert receipt["revision"] == REVISION
    assert receipt["role"] == "actor"
    assert receipt["cache_hit"] is False
    assert len(receipt["manifest_sha256"]) == 64

    def offline(**kwargs):
        pytest.fail("exact warm cache attempted network")

    warm = fetch.fetch_assets(manifest, tmp_path / "second", cache_dir=tmp_path / "cache", downloader=offline)
    assert warm["cache_hit"] is True
    assert (tmp_path / "second/observation.png").read_bytes() == b"observation png"
    assert (tmp_path / "first/observation.png").stat().st_ino != (tmp_path / "second/observation.png").stat().st_ino


@pytest.mark.parametrize("field,value", [
    ("revision", "main"), ("revision", "latest"), ("revision", "1" * 39),
    ("repo_id", "another/repository"), ("task_id", "../other"), ("role", "feedback"),
    ("schema", "unknown"),
])
def test_invalid_manifest_identity_never_downloads(tmp_path, field, value):
    manifest = manifest_for()
    manifest[field] = value
    with pytest.raises(ValueError):
        fetch_module().fetch_assets(manifest, tmp_path / "out", cache_dir=tmp_path / "cache", downloader=lambda **_: pytest.fail("unsafe download"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("path", [
    "results/observation.png", "runs/old/observation.png", "viewer/observation.png", "media/observation.png",
    "benchmark/nv100/v1/tasks/other/public/observation.png",
    PREFIX + "reference/observation.png", PREFIX + "comparison/observation.png",
    PREFIX + "public/../reference/observation.png", PREFIX + "public//observation.png",
    PREFIX + "public/extra.png", "/" + PREFIX + "public/observation.png",
    PREFIX + "public\\observation.png",
])
def test_actor_rejects_reference_result_other_task_and_noncanonical_paths(path):
    manifest = manifest_for()
    manifest["files"][0]["path"] = path
    with pytest.raises(ValueError):
        fetch_module().validate_manifest(manifest)


@pytest.mark.parametrize("destination", ["../outside", "/absolute", "a/../../outside", "a//b", "a/./b", "a\\b", ".", ""])
def test_materialization_rejects_unsafe_destinations(destination):
    manifest = manifest_for()
    manifest["files"][0]["destination"] = destination
    with pytest.raises(ValueError):
        fetch_module().validate_manifest(manifest)


@pytest.mark.parametrize("mutation", ["missing", "duplicate_path", "duplicate_destination", "parent_destination", "negative_size", "bool_size", "bad_hash", "unknown_field"])
def test_manifest_rejects_ambiguous_or_unverifiable_files(mutation):
    manifest = manifest_for()
    if mutation == "missing":
        manifest["files"].pop()
    elif mutation == "duplicate_path":
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    elif mutation == "duplicate_destination":
        manifest["files"][1]["destination"] = manifest["files"][0]["destination"]
    elif mutation == "parent_destination":
        manifest["files"][1]["destination"] = manifest["files"][0]["destination"] + "/child"
    elif mutation == "negative_size":
        manifest["files"][0]["size_bytes"] = -1
    elif mutation == "bool_size":
        manifest["files"][0]["size_bytes"] = True
    elif mutation == "bad_hash":
        manifest["files"][0]["sha256"] = "no hash"
    else:
        manifest["files"][0]["url"] = "https://untrusted.example"
    with pytest.raises(ValueError):
        fetch_module().validate_manifest(manifest)


def test_verifier_fetches_only_its_declared_reference_closure(tmp_path, hub):
    payloads = {"reference/source.usda": b"#usda 1.0", "reference/deps/material.usda": b"material",
                "reference/eligibility.json": b"{}", "comparison/cameras.json": b"{}",
                **{f"comparison/view_{i}.png": f"view {i}".encode() for i in range(4)}}
    hub["payloads"].update({PREFIX + path: data for path, data in payloads.items()})
    manifest = manifest_for(payloads, role="verifier")
    receipt = fetch_module().fetch_assets(manifest, tmp_path / "fixtures", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert set(hub["requests"]) == {PREFIX + path for path in payloads}
    assert receipt["role"] == "verifier"
    assert (tmp_path / "fixtures/deps/material.usda").read_bytes() == b"material"
    manifest["files"][0]["path"] = PREFIX + "public/observation.png"
    with pytest.raises(ValueError):
        fetch_module().validate_manifest(manifest)


@pytest.mark.parametrize("corruption", ["size", "hash"])
def test_corrupt_download_retries_boundedly_and_publishes_nothing(tmp_path, hub, corruption):
    hub["payloads"][PREFIX + "public/observation.png"] = b"x" if corruption == "size" else b"X" * len(b"observation png")
    with pytest.raises((ValueError, RuntimeError)):
        fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache", downloader=hub["downloader"], attempts=2)
    assert hub["requests"].count(PREFIX + "public/observation.png") == 2
    assert not (tmp_path / "out").exists()


def test_transient_http_failure_retries_then_succeeds(tmp_path, hub):
    hub["failures"].append(503)
    fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache", downloader=hub["downloader"], attempts=2)
    assert (tmp_path / "out/instruction.md").read_bytes() == b"Build the widget."
    assert len(hub["requests"]) == 3


def test_warm_cache_is_rehashed_and_corruption_refetched(tmp_path, hub):
    fetch = fetch_module()
    fetch.fetch_assets(manifest_for(), tmp_path / "first", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    cached = list((tmp_path / "cache").rglob("observation.png"))
    assert len(cached) == 1
    cached[0].write_bytes(b"X" * len(b"observation png"))
    hub["requests"].clear()
    receipt = fetch.fetch_assets(manifest_for(), tmp_path / "second", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert receipt["cache_hit"] is False
    assert PREFIX + "public/observation.png" in hub["requests"]
    assert (tmp_path / "second/observation.png").read_bytes() == b"observation png"


def test_cache_revision_identity_cannot_reuse_other_release(tmp_path, hub):
    fetch = fetch_module()
    fetch.fetch_assets(manifest_for(), tmp_path / "first", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    hub["requests"].clear()
    fetch.fetch_assets(manifest_for(revision="2" * 40), tmp_path / "second", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert len(hub["requests"]) == 2


def test_hub_cache_symlink_is_copied_as_private_regular_file(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    for path, data in PUBLIC.items():
        (sources / Path(path).name).write_bytes(data)
    link = tmp_path / "hub-link"

    def downloader(**kwargs):
        link.unlink(missing_ok=True)
        link.symlink_to(sources / Path(kwargs["filename"]).name)
        return link

    fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache", downloader=downloader)
    output = tmp_path / "out/observation.png"
    assert not output.is_symlink()
    assert output.stat().st_nlink == 1
    (sources / "observation.png").write_bytes(b"changed")
    assert output.read_bytes() == b"observation png"


def test_symlink_destination_cannot_write_outside_fixture_tree(tmp_path, hub):
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "out"
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises((ValueError, RuntimeError)):
        fetch_module().fetch_assets(manifest_for(), destination, cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert list(outside.iterdir()) == []
    assert hub["requests"] == []


def test_unexpected_cached_file_is_not_materialized(tmp_path, hub):
    fetch = fetch_module()
    fetch.fetch_assets(manifest_for(), tmp_path / "first", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    cached = next((tmp_path / "cache").rglob("observation.png"))
    (cached.parent / "reference.usda").write_bytes(b"should not leak")
    fetch.fetch_assets(manifest_for(), tmp_path / "second", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert {p.name for p in (tmp_path / "second").iterdir()} == {"observation.png", "instruction.md"}


def test_cli_invalid_manifest_has_nonzero_safe_error(tmp_path):
    manifest = manifest_for()
    manifest["revision"] = "main"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    process = subprocess.run([sys.executable, "-m", "simready_eval.fetch", "--manifest", str(path), "--destination", str(tmp_path / "out"), "--cache", str(tmp_path / "cache")], text=True, capture_output=True)
    assert process.returncode != 0
    assert "revision" in process.stderr.lower()
    assert "Traceback" not in process.stderr
    assert not (tmp_path / "out").exists()


def test_default_hub_transport_uses_isolated_public_pinned_dataset_calls(tmp_path, hub, monkeypatch):
    import types
    fake_module = tmp_path / "transport-module"
    fake_module.mkdir()
    audit = tmp_path / "transport-audit.jsonl"
    (fake_module / "huggingface_hub.py").write_text("""
import json, os
from pathlib import Path
from urllib.request import urlopen

def hf_hub_download(**kwargs):
    audit = {'kwargs': kwargs, 'environment': {name: os.environ.get(name) for name in
             ('HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'HF_TOKEN_PATH', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN')}}
    with open(os.environ['TEST_HUB_AUDIT'], 'a') as stream:
        stream.write(json.dumps(audit) + chr(10))
    target = Path(kwargs['cache_dir']) / 'response'
    with urlopen(os.environ['TEST_HUB_URL'] + kwargs['filename'], timeout=2) as response:
        target.write_bytes(response.read())
    return str(target)
""")
    # The subprocess exercises the true default transport with an installed
    # local stand-in solely for the external huggingface_hub boundary.
    monkeypatch.setenv("PYTHONPATH", str(fake_module) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("TEST_HUB_AUDIT", str(audit))
    monkeypatch.setenv("TEST_HUB_URL", hub["url"])
    monkeypatch.setenv("HF_TOKEN", "credential-must-not-enter-child")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "legacy-credential-must-not-enter-child")

    def in_process_transport(**kwargs):
        pytest.fail("default transport must isolate the global HF/Xet session in a child process")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=in_process_transport))
    fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache")
    assert (tmp_path / "out/observation.png").read_bytes() == b"observation png"
    calls = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(calls) == 2
    for call in calls:
        kwargs = call['kwargs']
        assert kwargs['repo_id'] == REPO
        assert kwargs['repo_type'] == 'dataset'
        assert kwargs['revision'] == REVISION
        assert kwargs['token'] is False
        assert Path(kwargs['cache_dir']).is_relative_to(tmp_path / "cache/actor/widget-001")
        assert not Path(kwargs['cache_dir']).exists()
        for key in ('HF_HOME', 'HF_HUB_CACHE', 'HF_XET_CACHE', 'HF_TOKEN_PATH'):
            assert Path(call['environment'][key]).is_relative_to(Path(kwargs['cache_dir']))
        assert call['environment']['HF_TOKEN'] is None
        assert call['environment']['HUGGING_FACE_HUB_TOKEN'] is None


def test_cli_warm_cache_returns_json_offline(tmp_path, hub):
    fetch_module().fetch_assets(manifest_for(), tmp_path / "first", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_for()))
    process = subprocess.run([sys.executable, "-m", "simready_eval.fetch", "--manifest", str(path), "--destination", str(tmp_path / "second"), "--cache", str(tmp_path / "cache")], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    receipt = json.loads(process.stdout)
    assert receipt["cache_hit"] is True
    assert receipt["role"] == "actor"
    assert (tmp_path / "second/instruction.md").read_bytes() == b"Build the widget."


def test_concurrent_same_manifest_has_one_verified_cache_publication(tmp_path, hub):
    import concurrent.futures
    fetch = fetch_module()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch.fetch_assets, manifest_for(), tmp_path / str(index), cache_dir=tmp_path / "cache", downloader=hub["downloader"]) for index in range(2)]
        receipts = [future.result(timeout=5) for future in futures]
    assert sorted(receipt["cache_hit"] for receipt in receipts) == [False, True]
    assert len(hub["requests"]) == 2
    for index in range(2):
        assert (tmp_path / str(index) / "observation.png").read_bytes() == b"observation png"


@pytest.mark.parametrize("attempts", [0, -1, 11, True, 1.5])
def test_invalid_attempt_budget_fails_before_transport(tmp_path, attempts):
    with pytest.raises(ValueError):
        fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache", attempts=attempts, downloader=lambda **_: pytest.fail("invalid budget used transport"))


def test_offline_cache_corruption_cannot_publish_fixture(tmp_path, hub):
    fetch = fetch_module()
    fetch.fetch_assets(manifest_for(), tmp_path / "first", cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    cached = next((tmp_path / "cache").rglob("observation.png"))
    cached.write_bytes(b"X" * len(b"observation png"))

    def offline(**kwargs):
        raise ConnectionError("offline")

    with pytest.raises(RuntimeError):
        fetch.fetch_assets(manifest_for(), tmp_path / "second", cache_dir=tmp_path / "cache", downloader=offline, attempts=1)
    assert not (tmp_path / "second").exists()


def test_existing_unexpected_output_is_never_overwritten(tmp_path, hub):
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "keep.txt").write_text("existing user file")
    with pytest.raises(ValueError):
        fetch_module().fetch_assets(manifest_for(), destination, cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    assert (destination / "keep.txt").read_text() == "existing user file"
    assert hub["requests"] == []


def test_setup_deadline_interrupts_stalled_transport_without_publishing(tmp_path):
    import time

    def stalled(**kwargs):
        time.sleep(2)
        pytest.fail("setup deadline did not interrupt the external transport")

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache", downloader=stalled, timeout_seconds=0.05)
    assert time.monotonic() - started < 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_invalid_setup_deadline_fails_before_transport(tmp_path, timeout):
    with pytest.raises(ValueError):
        fetch_module().fetch_assets(manifest_for(), tmp_path / "out", cache_dir=tmp_path / "cache", timeout_seconds=timeout, downloader=lambda **_: pytest.fail("invalid deadline used transport"))


def test_exact_tree_hash_failure_identifies_public_file_and_both_digests(tmp_path, hub):
    fetch = fetch_module()
    manifest = manifest_for()
    root = tmp_path / "public"
    fetch.fetch_assets(manifest, root, cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    image = root / "observation.png"
    original = image.read_bytes()
    changed = bytes([original[0] ^ 1]) + original[1:]
    image.write_bytes(changed)
    with pytest.raises(ValueError) as caught:
        fetch._verify_tree(root, manifest["files"])
    message = str(caught.value)
    assert "observation.png" in message
    assert hashlib.sha256(original).hexdigest() in message
    assert hashlib.sha256(changed).hexdigest() in message


def test_exact_tree_size_failure_reports_actual_and_expected_without_relaxation(tmp_path, hub):
    fetch = fetch_module(); manifest = manifest_for()
    root = tmp_path / "public"
    fetch.fetch_assets(manifest, root, cache_dir=tmp_path / "cache", downloader=hub["downloader"])
    image = root / "observation.png"; original = image.read_bytes(); image.write_bytes(original + b"x")
    with pytest.raises(ValueError) as caught:
        fetch._verify_tree(root, manifest["files"])
    message = str(caught.value)
    assert "observation.png" in message
    assert f"expected_size={len(original)}" in message
    assert f"observed_size={len(original) + 1}" in message
