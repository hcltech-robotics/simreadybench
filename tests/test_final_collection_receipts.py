import copy
import json

import pytest

from simready_bench.daytona import (
    _collected_result,
    _verify_final_reconstruction_receipts,
)
from test_daytona_reports import remote_reports


def test_final_collection_verifies_item_completed_after_last_poll(tmp_path):
    output, progress = remote_reports(tmp_path)
    record = {"collected_reconstruction_ids": [], "collected_reconstruction_manifests": {}}
    _verify_final_reconstruction_receipts(output, record)
    identity = progress["results"][0]["condition_id"]
    assert record["collected_reconstruction_ids"] == [identity]
    assert len(record["collected_reconstruction_manifests"][identity]) == 64
    before = copy.deepcopy(record)
    _verify_final_reconstruction_receipts(output, record)
    assert record == before


def test_final_collection_rejects_changed_recorded_manifest(tmp_path):
    output, progress = remote_reports(tmp_path)
    identity = progress["results"][0]["condition_id"]
    record = {"collected_reconstruction_manifests": {identity: "f" * 64}}
    with pytest.raises(ValueError, match="manifest has changed"):
        _verify_final_reconstruction_receipts(output, record)
    assert record["collected_reconstruction_manifests"][identity] == "f" * 64


def test_final_collection_rejects_changed_payload_without_receipt(tmp_path):
    output, progress = remote_reports(tmp_path)
    identity = progress["results"][0]["condition_id"]
    path = output / "reconstructions" / identity / "report.html"
    path.write_bytes(path.read_bytes() + b"changed")
    record = {}
    with pytest.raises(ValueError, match="artefact size differs"):
        _verify_final_reconstruction_receipts(output, record)
    assert not record


def test_terminal_failure_keeps_retained_counts_without_run_result(tmp_path):
    output, _ = remote_reports(tmp_path)
    (output / "results.json").write_bytes((output / "reconstructions/results.json").read_bytes())
    remote = output / "daytona-executor"
    remote.mkdir()
    (remote / "controller-status.json").write_text(json.dumps({
        "status": "failed", "error": "asset boundary reached"}))
    record = {"controller_exit": 2}
    result = _collected_result({}, output, record, clean=str)
    assert result["completed_conditions"] == record["completed_conditions"] == 1
    assert record["attempted_conditions"] == 1
    assert record["status"] == "failed"
    assert len(result["results"]) == 1
