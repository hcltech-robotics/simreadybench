"""Collected missing submissions retain their canonical measured zero."""

from copy import deepcopy
import hashlib
import json

import pytest

from simready_bench import orchestration, resume
from simready_bench.controller_runner import seal_reconstruction, verify_reconstruction
from simready_bench.engine.scoring import score_validation
from simready_bench.engine.verifier import scoring_policy
from simready_bench.reporting import compact_report, read_generated_report
from test_parallel import manifest
from test_resume import portable, write


def invalid_report(tmp_path, *, available=False):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    output = tmp_path / "run"
    batch = output / "executions/001"
    write(batch / "experiment.json", selected)
    identity, condition = next(iter(resume.condition_plan(selected).items()))
    path = portable(
        batch / "reconstructions" / identity, [(identity, condition, "completed")]
    )
    value = json.loads(path.read_text())
    value["results"][0].update(
        nvidia_validator_report_file=None,
        simready_reward=0.0,
        simready=score_validation(None, scoring_policy(), candidate_invalid=True),
        images=[],
        traces=[],
    )
    if available:
        candidate = value["results"][0]["candidate"]
        (path.parent / "candidate").rename(path.parent / "candidates")
        for key in ("entrypoint", "manifest", "directory"):
            candidate[key] = candidate[key].replace("candidate", "candidates", 1)
        for member in candidate["files"]:
            member["path"] = member["path"].replace("candidate/", "candidates/", 1)
    else:
        value["results"][0].update(
            candidate={"status": "unavailable", "reason": "no_frozen_authored_submission"},
            snapshot_sha256=None,
        )
    value = compact_report(value)
    write(path, value)
    if not available:
        (path.parent / "candidate/asset.usd").unlink()
        (path.parent / "candidate/manifest.json").unlink()
        (path.parent / "candidate").rmdir()
    (path.parent / "evidence/nvidia.json").unlink()
    (path.parent / "report.html").write_text("<html></html>\n")
    (path.parent / "scores.csv").write_text(
        "condition_id,simready_reward\n" + identity + ",0\n"
    )
    return output, batch, selected, identity, path, value["results"][0]


def test_sealed_invalid_submission_zero_is_retained_across_resume_and_readout(tmp_path):
    output, batch, selected, identity, path, row = invalid_report(tmp_path)
    seal_reconstruction(path.parent, identity)
    verify_reconstruction(path.parent, identity, row)
    original = path.read_bytes()
    assert resume._completed(path.parent, row)
    index = orchestration.ReportIndex(output, selected)
    index.backfill([batch])
    readout = read_generated_report(index.aggregate()["results"])
    assert readout["completed_conditions"] == 1
    assert readout["results"][0]["simready_reward"] == 0.0
    assert index.closed == {identity}
    assert path.read_bytes() == original
    restored = orchestration.ReportIndex(output, selected)
    restored.backfill([batch])
    assert restored.closed == {identity}
    assert (
        read_generated_report(restored.aggregate()["results"])["completed_conditions"]
        == 1
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "simready_reward", 1),
        (None, "simready_reward", None),
        (None, "trial_id", None),
        (None, "trial_id", ""),
        (None, "inference_calls_used", 0),
        (None, "inference_calls_used", True),
        (None, "snapshot_sha256", "a" * 64),
        (None, "nvidia_validator_report_file", {"path": "missing.json"}),
        ("simready", "score_percent", 1),
        ("simready", "score_percent", False),
        ("simready", "status", "unknown"),
        ("simready", "status", "scored"),
        ("simready", "unknown_count", 1),
        ("simready", "passed_count", 1),
        ("simready", "failed_count", 45),
        ("simready", "applicable_count", 45),
        ("simready", "profile_mandatory_count", 45),
        ("simready", "profile_coverage_fraction", 0.5),
        ("simready", "profile_coverage_fraction", True),
        ("simready", "policy_sha256", "a" * 64),
        ("simready", "scope", "filtered"),
        ("simready", "score_upper_bound", 5),
        ("simready", "checks", []),
        ("candidate", "entrypoint", "missing.usd"),
        ("candidate", "files", []),
        (
            "candidate",
            "reason",
            "frozen_submission_missing_unsafe_changed_or_over_export_limit",
        ),
    ],
)
def test_unmeasured_or_changed_missing_submission_cannot_be_completed(
    tmp_path, section, key, value
):
    _, _, _, _, path, row = invalid_report(tmp_path)
    changed = deepcopy(row)
    (changed if section is None else changed[section])[key] = value
    assert not resume._completed(path.parent, changed)


@pytest.mark.parametrize("kind", ["images", "traces"])
def test_invalid_submission_still_requires_every_exported_evidence_receipt(tmp_path, kind):
    _, _, _, _, path, row = invalid_report(tmp_path)
    evidence = path.parent / "evidence/actor.log"
    evidence.write_bytes(b"attempted authoring")
    row[kind] = [
        {
            "path": "actor.log",
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    ]
    assert resume._completed(path.parent, row)
    evidence.write_bytes(b"changed authoring")
    assert not resume._completed(path.parent, row)
    evidence.unlink()
    assert not resume._completed(path.parent, row)


def test_invalid_submission_cannot_change_a_pinned_requirement_identity(tmp_path):
    _, _, _, _, path, row = invalid_report(tmp_path)
    row["simready"]["checks"][0]["requirement_id"] = "unrecognised"
    assert not resume._completed(path.parent, row)


def test_available_candidate_still_requires_its_original_frozen_files(tmp_path):
    _, batch, _, identity, _, _ = invalid_report(tmp_path)
    selected = json.loads((batch / "experiment.json").read_text())
    path = portable(
        batch / "attempts/final",
        [(identity, resume.condition_plan(selected)[identity], "completed")],
    )
    row = read_generated_report(path)["results"][0]
    assert resume._completed(path.parent, row)
    (path.parent / row["candidate"]["entrypoint"]).unlink()
    assert not resume._completed(path.parent, row)


def test_sealed_invalid_available_candidate_keeps_zero_without_native_findings(tmp_path):
    output, batch, selected, identity, path, row = invalid_report(tmp_path, available=True)
    seal_reconstruction(path.parent, identity)
    verify_reconstruction(path.parent, identity, row)
    assert resume._completed(path.parent, row)
    index = orchestration.ReportIndex(output, selected)
    index.backfill([batch])
    readout = read_generated_report(index.aggregate()["results"])
    assert readout["completed_conditions"] == 1
    assert readout["results"][0]["simready_reward"] == 0.0
    assert readout["results"][0]["candidate"]["status"] == "available"
    assert identity in index.closed


@pytest.mark.parametrize(
    "fault",
    [
        "manifest_tampered", "file_missing", "file_changed", "manifest_hash_missing",
        "unknown", "positive", "unattempted", "native_receipt_missing",
    ],
)
def test_invalid_available_candidate_without_native_findings_remains_fully_bound(
    tmp_path, fault
):
    _, _, _, _, path, row = invalid_report(tmp_path, available=True)
    candidate = row["candidate"]
    if fault == "manifest_tampered":
        (path.parent / candidate["manifest"]).write_text("{}")
    elif fault == "file_missing":
        (path.parent / candidate["entrypoint"]).unlink()
    elif fault == "file_changed":
        (path.parent / candidate["entrypoint"]).write_text("changed")
    elif fault == "manifest_hash_missing":
        candidate.pop("manifest_sha256")
    elif fault == "unknown":
        row["simready"]["unknown_count"] = 1
    elif fault == "positive":
        row["simready_reward"] = row["simready"]["score_percent"] = 1
    elif fault == "unattempted":
        row["inference_calls_used"] = 0
    else:
        row["nvidia_validator_report_file"] = {"path": "evidence/missing.json"}
    assert not resume._completed(path.parent, row)
