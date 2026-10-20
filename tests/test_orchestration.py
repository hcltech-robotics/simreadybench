import json
import os

import pytest

from simready_bench import orchestration, resume
from simready_bench.reporting import read_generated_report
from test_parallel import manifest
from test_resume import portable, write


def fixture_report(tmp_path):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    output = tmp_path / "run"
    output.mkdir()
    batch = output / "executions/001"
    write(batch / "experiment.json", selected)
    identity, condition = next(iter(resume.condition_plan(selected).items()))
    path = portable(
        batch / "reconstructions" / identity, [(identity, condition, "completed")]
    )
    report = {"directory": str(path.parent), "results": str(path)}
    return output, batch, selected, identity, path, report


@pytest.mark.parametrize("restart", [False, True])
def test_completed_measurements_are_immutable_across_callbacks_and_restart(
    tmp_path, restart
):
    output, batch, selected, identity, path, report = fixture_report(tmp_path)
    index = orchestration.ReportIndex(output, selected)
    index.backfill([batch])
    first = index.aggregate()
    assert (
        read_generated_report(first["results"])["results"][0]["simready_reward"] == 100
    )
    if restart:
        index = orchestration.ReportIndex(output, selected)
    changed = json.loads(path.read_text())
    changed["results"][0]["simready_reward"] = 7
    changed["results"][0]["simready"]["score_percent"] = 7
    write(path, changed)
    with pytest.raises(ValueError, match="completed reconstruction changed"):
        index.add(batch, report)
    assert identity in index.closed
    assert (
        read_generated_report(first["results"])["results"][0]["simready_reward"] == 100
    )


def test_completed_identity_cannot_change_through_another_report_path(tmp_path):
    output, batch, selected, identity, path, report = fixture_report(tmp_path)
    index = orchestration.ReportIndex(output, selected)
    index.backfill([batch])
    index.aggregate()
    alternate = portable(
        batch / "attempts/final",
        [(identity, resume.condition_plan(selected)[identity], "completed")],
    )
    changed = json.loads(alternate.read_text())
    changed["results"][0]["simready_reward"] = 8
    write(alternate, changed)
    with pytest.raises(ValueError, match="completed reconstruction changed"):
        index.add(
            batch, {"directory": str(alternate.parent), "results": str(alternate)}
        )


def test_leaf_to_final_handoff_permits_only_artefact_relocation(tmp_path):
    output, batch, selected, identity, path, report = fixture_report(tmp_path)
    index = orchestration.ReportIndex(output, selected)
    index.backfill([batch])
    index.aggregate()
    alternate = portable(
        batch / "attempts/final",
        [(identity, resume.condition_plan(selected)[identity], "completed")],
    )
    index.add(batch, {"directory": str(alternate.parent), "results": str(alternate)})
    restored = orchestration.ReportIndex(output, selected)
    restored.backfill([batch])
    assert (
        read_generated_report(restored.aggregate()["results"])["completed_conditions"]
        == 1
    )


def test_backfill_keeps_all_live_rows_before_first_publication(tmp_path):
    output, batch, selected, identity, path, report = fixture_report(tmp_path)
    identities = list(resume.condition_plan(selected).items())
    other = output / "executions/002"
    write(other / "experiment.json", selected)
    portable(
        other / "reconstructions" / identities[1][0], [(*identities[1], "completed")]
    )
    published = []

    def publish(value, **_):
        published.append(
            {
                row["condition_id"]
                for row in read_generated_report(value["results"])["results"]
            }
        )

    first = orchestration.ReportIndex(output, selected)
    first.backfill([batch])
    first.aggregate()
    restored = orchestration.ReportIndex(output, selected, publish)
    restored.backfill([batch, other])
    assert published == []
    restored.aggregate()
    assert published == [{identity, identities[1][0]}]
    assert restored.closed == {identity, identities[1][0]}


def test_publication_failure_is_retained_without_corrupting_valid_index(tmp_path):
    output, batch, selected, identity, path, report = fixture_report(tmp_path)
    calls = []

    def publish(value, **_):
        calls.append(value)
        raise ConnectionResetError("interrupted upload")

    index = orchestration.ReportIndex(output, selected, publish)
    index.backfill([batch])
    result = index.aggregate()
    assert index.publication_pending
    assert read_generated_report(result["results"])["completed_conditions"] == 1
    assert (
        json.loads((output / "publication-state.json").read_text())["status"]
        == "pending"
    )
    index.publisher = lambda value, **_: None
    index.retry_after = 0
    index.aggregate(final=True)
    assert not index.publication_pending
    assert len(calls) == 1


def test_retry_policy_is_bound_to_the_original_coordinator_journal(
    tmp_path, monkeypatch
):
    from test_resume import stopped_run

    output, selected, policy, _ = stopped_run(tmp_path, monkeypatch)
    options = dict(
        policy=policy,
        output_dir=output,
        workers=1,
        wheel=None,
        cpu_snapshot=None,
        runtime_images={},
        dispatch_pending=False,
        max_retries=2,
    )
    orchestration.coordinate_daytona(selected, **options)
    with pytest.raises(ValueError, match="retry policy"):
        orchestration.coordinate_daytona(selected, **{**options, "max_retries": 3})


@pytest.mark.skipif(os.name == "nt", reason="Native allowance ownership requires Linux")
@pytest.mark.parametrize("turns", [32, 64])
def test_explicit_resume_turn_limit_must_match_the_saved_legacy_protocol(
    tmp_path, monkeypatch, turns
):
    from test_automatic_recovery import arguments
    from test_resume import stopped_run

    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    options = arguments(output, dispatch_pending=False, authoring_turns=turns)
    if turns == 32:
        assert resume.resume_benchmark(**options)["completed_conditions"] == 3
    else:
        with pytest.raises(ValueError, match="authoring turn"):
            resume.resume_benchmark(**options)


@pytest.mark.parametrize("kind", ["images", "traces"])
def test_completed_capture_and_trace_bytes_remain_bound_to_receipts(tmp_path, kind):
    import hashlib

    output, batch, selected, identity, path, report = fixture_report(tmp_path)
    artefact = path.parent / "evidence/capture.bin"
    artefact.write_bytes(b"original evidence")
    value = json.loads(path.read_text())
    value["results"][0][kind] = [
        {
            "path": "capture.bin",
            "sha256": hashlib.sha256(artefact.read_bytes()).hexdigest(),
        }
    ]
    write(path, value)
    index = orchestration.ReportIndex(output, selected)
    index.backfill([batch])
    index.aggregate()
    artefact.write_bytes(b"changed evidence")
    with pytest.raises(ValueError, match="frozen artefacts"):
        orchestration.ReportIndex(output, selected)
