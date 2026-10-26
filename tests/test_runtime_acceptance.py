"""Installed-product acceptance through CLI, coordinator and real local services."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest


pytestmark = [pytest.mark.acceptance, pytest.mark.skipif(os.name == "nt", reason="Linux authority ownership")]


@pytest.fixture(scope="module")
def installed_product(tmp_path_factory):
    from simready_bench.daytona import build_wheel

    root = tmp_path_factory.mktemp("installed-product")
    wheel = build_wheel(root / "wheel")
    installed = root / "installed"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(wheel)],
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return installed, hashlib.sha256(wheel.read_bytes()).hexdigest()


def invoke(installed, root, fault):
    env = {key: value for key, value in os.environ.items()
           if key not in {"PYTHONPATH", "OPENAI_API_KEY", "HF_TOKEN", "DAYTONA_API_KEY"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, str(Path(__file__).with_name("acceptance_helpers.py")), str(installed), str(root), fault],
        cwd=root, env=env, text=True, capture_output=True, timeout=60,
    )


def wait_publication_retry(root):
    publication_state = root / "run/publication-state.json"
    if publication_state.is_file():
        pending = json.loads(publication_state.read_text())
        if pending.get("status") == "pending":
            remaining = max(0, pending.get("retry_after", 0) - time.time())
            assert remaining <= 10, "Unexpected publication recovery deadline"
            time.sleep(remaining + 0.02)


@pytest.mark.parametrize("fault", [
    "none", "admission", "candidate_sealing", "evaluation", "collection",
    "remote_commit", "deletion_acknowledgement", "collection_transient",
    "publication_transient", "delete_transient",
])
def test_installed_cli_converges_without_reauthoring(installed_product, tmp_path, fault):
    installed, wheel_hash = installed_product
    first = invoke(installed, tmp_path, fault)
    assert first.returncode in {0, 2, 75, 77}, first.stdout + first.stderr
    if fault in {"admission", "candidate_sealing", "evaluation", "collection", "remote_commit", "deletion_acknowledgement"}:
        assert first.returncode == 77, first.stdout + first.stderr
    # A fresh process reconciles retained external state and the same run identity.
    wait_publication_retry(tmp_path)
    final = invoke(installed, tmp_path, fault)
    assert final.returncode == 0, first.stdout + first.stderr + final.stdout + final.stderr
    outcome = json.loads((tmp_path / "acceptance-result.json").read_text())
    state = outcome["transport"]
    assert state["creates"] == state["generation"] == state["count"] == state["deletes"] == 1
    assert state["deleted"] is True
    assert state["downloads"] <= 3 and state["commit_attempts"] <= 4
    rows = outcome["results"]["results"]
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed" and row["simready_reward"] == 0
    published = json.loads((tmp_path / "hub/results/runs/fixture/results.json").read_text())
    assert len(published["results"]) == 1
    assert published["results"][0]["condition_id"] == row["condition_id"]
    assert published["results"][0]["simready_reward"] == 0
    assert published["results"][0]["candidate"]["manifest_sha256"] == row["candidate"]["manifest_sha256"]
    coordinator = json.loads((tmp_path / "run/coordinator-state.json").read_text())
    assert coordinator["status"] == "completed"
    assert len(coordinator["dispatches"]) == 1
    deliveries = list((tmp_path / "run").glob("executions/*/daytona-delivery.json"))
    assert len(deliveries) == 1
    delivery = json.loads(deliveries[0].read_text())
    assert delivery["cleanup_confirmed"] is True
    assert delivery["collection"] == "completed"
    assert delivery["wheel_sha256"] == wheel_hash
    with sqlite3.connect(tmp_path / "run/attempt-state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1
        assert db.execute("SELECT cleanup,publication FROM attempts").fetchall() == [("deleted", "published")]
        assert db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    gateway = json.loads((tmp_path / "remote/gateway-records.json").read_text())
    assert len(gateway) == 1 and gateway[0]["upstream_status"] == 200
    acceptance = {"wheel_sha256": wheel_hash, "runtime_images": delivery["runtime_images"],
                  "fault": fault, "outcome": "passed", "transport": state,
                  "condition_id": row["condition_id"], "snapshot_sha256": row["snapshot_sha256"]}
    (tmp_path / "acceptance-receipt.json").write_text(json.dumps(acceptance, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("fault", ["collection_permanent", "receipt_corrupt", "receipt_corrupt_legacy", "metadata_hold"])
def test_installed_cli_bounds_failed_collection_and_retires_exact_owner(installed_product, tmp_path, fault):
    installed, wheel_hash = installed_product
    first = invoke(installed, tmp_path, fault)
    assert first.returncode == 2, first.stdout + first.stderr
    state = json.loads((tmp_path / "transport-state.json").read_text())
    assert state["creates"] == state["generation"] == state["deletes"] == 1
    assert state["deleted"] is True and state["downloads"] <= 3
    assert state["commits"] == 0
    delivery_path, = (tmp_path / "run/executions").glob("*/daytona-delivery.json")
    delivery = json.loads(delivery_path.read_text())
    assert delivery["cleanup_confirmed"] is True
    assert delivery["collection"] == "lost"
    assert (delivery_path.parent / "collection-hold.json").is_file()
    if fault == "metadata_hold":
        retained = json.loads((delivery_path.parent / "results.json").read_text())["results"][0]
        assert retained["infrastructure_hold"] == "hf_pricing_metadata_changed"
        assert retained["infrastructure_hold_evidence"]["source"] == "trusted_gateway_record"
        assert retained["automatic_retry"] is False
    with sqlite3.connect(tmp_path / "run/attempt-state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 0
        assert db.execute("SELECT cleanup,infrastructure FROM attempts").fetchall() == [("deleted", "held")]
    second = invoke(installed, tmp_path, fault)
    assert second.returncode == 2, second.stdout + second.stderr
    restored = json.loads((tmp_path / "transport-state.json").read_text())
    assert restored["creates"] == restored["generation"] == restored["deletes"] == 1
    assert restored["downloads"] == state["downloads"]
    (tmp_path / "acceptance-receipt.json").write_text(json.dumps({
        "wheel_sha256": wheel_hash, "runtime_images": delivery["runtime_images"],
        "fault": fault, "outcome": "held_and_retired", "transport": restored,
    }, sort_keys=True, indent=2) + "\n")


def test_installed_cli_bounds_publication_across_restarts_without_retaining_compute(installed_product, tmp_path):
    installed, wheel_hash = installed_product
    for _ in range(3):
        wait_publication_retry(tmp_path)
        attempt = invoke(installed, tmp_path, "publication_permanent")
        assert attempt.returncode in {2, 75}, attempt.stdout + attempt.stderr
    state = json.loads((tmp_path / "transport-state.json").read_text())
    publication = json.loads((tmp_path / "run/publication-state.json").read_text())
    assert publication["status"] == "held" and publication["attempts"] == 3
    assert state["commit_attempts"] == 3 and state["commits"] == 0
    assert state["creates"] == state["generation"] == state["deletes"] == 1
    assert state["deleted"] is True
    with sqlite3.connect(tmp_path / "run/attempt-state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1
        assert db.execute("SELECT cleanup,publication FROM attempts").fetchall() == [("deleted", "held")]
    resumed = invoke(installed, tmp_path, "publication_permanent")
    assert resumed.returncode in {2, 75}, resumed.stdout + resumed.stderr
    restored = json.loads((tmp_path / "transport-state.json").read_text())
    assert restored == state
    delivery_path, = (tmp_path / "run/executions").glob("*/daytona-delivery.json")
    delivery = json.loads(delivery_path.read_text())
    (tmp_path / "acceptance-receipt.json").write_text(json.dumps({
        "wheel_sha256": wheel_hash, "runtime_images": delivery["runtime_images"],
        "fault": "publication_permanent", "outcome": "measurement_retained_delivery_held",
        "transport": restored,
    }, sort_keys=True, indent=2) + "\n")


def test_installed_cli_holds_measured_condition_when_delivery_receipt_is_lost(installed_product, tmp_path):
    installed, wheel_hash = installed_product
    first = invoke(installed, tmp_path, "missing_delivery")
    assert first.returncode == 77, first.stdout + first.stderr
    before = json.loads((tmp_path / "transport-state.json").read_text())
    published_path = tmp_path / "hub/results/runs/fixture/results.json"
    published = json.loads(published_path.read_text())
    assert len(published["results"]) == 1 and published["results"][0]["simready_reward"] == 0
    delivery_path, = (tmp_path / "run/executions").glob("*/daytona-delivery.json")
    delivery = json.loads(delivery_path.read_text())
    delivery_path.unlink()
    resumed = invoke(installed, tmp_path, "missing_delivery")
    assert resumed.returncode == 2, resumed.stdout + resumed.stderr
    after = json.loads((tmp_path / "transport-state.json").read_text())
    assert after["creates"] == after["generation"] == before["creates"] == before["generation"] == 1
    assert after["deletes"] == before["deletes"] == 1
    retained = json.loads(published_path.read_text())
    assert retained["results"][0]["condition_id"] == published["results"][0]["condition_id"]
    assert retained["results"][0]["simready_reward"] == 0
    with sqlite3.connect(tmp_path / "run/attempt-state.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM measurements").fetchone()[0] == 1
        assert db.execute("SELECT infrastructure,cleanup FROM attempts").fetchall() == [("held", "pending")]
    (tmp_path / "acceptance-receipt.json").write_text(json.dumps({
        "wheel_sha256": wheel_hash, "runtime_images": delivery["runtime_images"],
        "fault": "missing_delivery", "outcome": "measurement_retained_delivery_missing_held",
        "transport": after, "condition_id": retained["results"][0]["condition_id"],
    }, sort_keys=True, indent=2) + "\n")
