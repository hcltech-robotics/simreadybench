"""Unavailable local runtime images fail before paid condition admission."""

import json
import os
from types import SimpleNamespace

import pytest

from simready_bench import local
from test_local_resume_budget import interrupted_local_run


def runtime():
    return {"shared_profile": {"actor_image": "sha256:" + "a" * 64, "verifier_image": "sha256:" + "b" * 64}, "fidelity_image": "sha256:" + "c" * 64, "illustration_image": "sha256:" + "c" * 64}


def test_preflight_inspects_each_declared_image_once_without_running_or_pulling(tmp_path, monkeypatch):
    calls = []

    def inspect(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=command[-1] + "\n", stderr="")

    monkeypatch.setattr(local.subprocess, "run", inspect)
    local.preflight_runtime_images(runtime(), tmp_path)
    assert len(calls) == 3
    assert all(command[:3] == ["docker", "image", "inspect"] and options["timeout"] == 15 for command, options in calls)
    proof = json.loads((tmp_path / "private/runtime-preflight.json").read_text())
    assert all(check["available"] for check in proof["checks"])
    assert proof["checks"][-1]["roles"] == ["fidelity", "illustration"]


def test_failed_image_preflight_retains_commands_status_and_redacted_output(tmp_path, monkeypatch):
    secret = "private-fixture-provider-secret"
    monkeypatch.setenv("TEST_PROVIDER_TOKEN", secret)
    monkeypatch.setattr(local.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="Cannot inspect image: " + secret + " Bearer token-value https://user:password@registry.invalid/"))
    with pytest.raises(local.LocalRuntimePreflightError, match="actor, verifier, fidelity, illustration"):
        local.preflight_runtime_images(runtime(), tmp_path)
    raw = (tmp_path / "private/runtime-preflight.json").read_text()
    proof = json.loads(raw)
    assert secret not in raw and "token-value" not in raw and "user:password" not in raw
    assert all(check["returncode"] == 1 and check["command"][-1] == check["image"] for check in proof["checks"])
    assert "Cannot inspect image" in raw and "[redacted]" in raw


@pytest.mark.skipif(os.name == "nt", reason="Local ownership requires Linux")
def test_unavailable_runtime_stops_once_before_matrix_or_parent_admission(interrupted_local_run, monkeypatch):
    import sqlite3

    _, _, policy, _, resolved, _, work = interrupted_local_run
    monkeypatch.setattr(local, "runtime_policy", lambda *args, **kwargs: runtime())
    calls = []

    def inspect(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="No such image: " + command[-1])

    monkeypatch.setattr(local.subprocess, "run", inspect)
    monkeypatch.setattr(local, "prepare", lambda *args, **kwargs: pytest.fail("Unavailable images reached condition preparation"))
    with pytest.raises(local.RunExecutionError) as stopped:
        local.run(resolved, work_dir=work, policy=policy, workers=24)
    assert len(calls) == 3
    assert not (work / "matrix.sqlite3").exists()
    assert not (work / "controller").exists()
    with sqlite3.connect(policy["runtime"]["authority_state_dir"] + "/parent-allowances.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM allocations").fetchone()[0] == 0
    assert all(row["status"] == "blocked" and row["error_code"] == "local_runtime_image_unavailable" for row in stopped.value.result["results"])
    assert "runtime-preflight.json" in stopped.value.result["results"][0]["error_message"]
