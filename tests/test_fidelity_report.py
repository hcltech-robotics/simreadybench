import json, csv
from pathlib import Path
import pytest
from simready_bench.reporting import export_report


def test_missing_candidate_surface_distances_remain_unmeasured(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    report = export_report(
        work,
        tmp_path / "report",
        results={
            "schema": "simready-bench-matrix-result-v1",
            "results": [
                {
                    "condition_id": "missing-candidate",
                    "fidelity": {
                        "components": {
                            "shape": {
                                "precision": 0.0,
                                "recall": 0.0,
                                "score_percent": 0.0,
                                "eligible": 1,
                                "band_m": 0.003,
                                "rms_m": None,
                                "p95_m": None,
                            }
                        }
                    },
                }
            ],
        },
    )
    page = Path(report["html"]).read_text()
    assert "RMS distance (mm)" in page
    assert "<td>3</td><td>Not measured</td><td>Not measured</td>" in page
    shape = json.loads(Path(report["results"]).read_text())["results"][0]["fidelity"][
        "components"
    ]["shape"]
    assert shape["rms_m"] is None and shape["p95_m"] is None


def test_report_exposes_separate_fidelity_components_and_fixed_coverage(tmp_path):
    fidelity = {
        "schema": "source-usd-fidelity-v1",
        "status": "measured",
        "aggregate_score_percent": None,
        "components": {
            "shape": {
                "score_percent": 12,
                "earned": 0.12,
                "eligible": 1,
                "coverage_fraction": 0.5,
                "status": "measured",
            },
            "joints": {
                "score_percent": None,
                "eligible": 0,
                "coverage_fraction": 0,
                "status": "not_applicable",
            },
        },
    }
    work = tmp_path / "work"
    work.mkdir()
    report = export_report(
        work,
        tmp_path / "report",
        results={
            "schema": "simready-bench-matrix-result-v1",
            "results": [{"condition_id": "one", "fidelity": fidelity}],
        },
    )
    data = json.loads(open(report["results"]).read())["results"][0]
    assert data["fidelity"] == fidelity
    row = list(csv.DictReader(open(report["scores"])))[0]
    assert (
        row["fidelity_shape_percent"] == "12"
        and row["fidelity_shape_coverage"] == "0.5"
    )
    assert row["fidelity_joints_percent"] == ""
    assert row["fidelity_reference_geometry_status"] == "partial"
    assert data["fidelity_complete_geometry_comparison"] is None
    assert data["fidelity_complete_reconstruction_success"] is None
    assert "fidelity_complete_geometry_comparison" not in row
    page = open(report["html"]).read()
    assert (
        "Direct USD reconstruction fidelity" in page
        and "Reference coverage (%)" in page
    )
    assert "12% (0.12/1)" in page
    assert (
        "<td>Shape</td><td>12</td><td>0.12/1</td>"
        "<td>Not measured</td><td>1</td><td>50</td>"
    ) in page


def test_report_ignores_scores_and_judges_without_a_supported_producer(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    orphaned = {
        "normalized_score_percent": 99,
        "earned_credit": 99,
        "scorable_weight": 100,
        "coverage_fraction": 1,
    }
    report = export_report(
        work,
        tmp_path / "report",
        results={
            "schema": "simready-bench-matrix-result-v1",
            "results": [
                {
                    "condition_id": "one",
                    "normalized_quality": orphaned,
                    "evaluation": {"normalized_quality": orphaned},
                    "supplemental_visual_judge": {
                        "status": "fixture-orphan-judge",
                        "score": 99,
                    },
                }
            ],
        },
    )
    data = json.loads(open(report["results"]).read())["results"][0]
    assert data["simready"] is None
    assert "normalized_score_percent" not in data
    assert "earned_credit" not in data and "scorable_weight" not in data
    assert "coverage_fraction" not in data
    assert "supplemental_visual_judge" not in data
    assert "fixture-orphan-judge" not in open(report["html"]).read()


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="Frozen candidate export uses Linux filesystem APIs",
)
def test_report_leads_with_scores_captures_and_generated_assets(tmp_path):
    import hashlib
    from PIL import Image
    from helpers import candidate_fixture
    from simready_bench.local import export_submission

    work = tmp_path / "work"
    work.mkdir()
    _, controller, _, _, _ = candidate_fixture(work)
    snapshot = controller.freeze_candidate("ep")
    (work / "artifacts").mkdir()
    submission = export_submission(snapshot, work / "artifacts/submission")
    images = []
    for role, colour in [("reference", "brown"), ("candidate", "grey")]:
        image = work / (role + ".png")
        Image.new("RGB", (8, 8), colour).save(image)
        images.append(
            {
                "file": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "view": role + " / front",
            }
        )
    trace = work / "trajectory.json"
    trace.write_text('{"steps": []}')
    primary = {
        "schema": "simready-validator-score-v1",
        "score_percent": 0,
        "base_credit_percent": 0,
        "completion_bonus": 0,
        "passed_count": 0,
        "applicable_count": 46,
        "official_conformance": {"passed": False},
    }
    report = export_report(
        work,
        tmp_path / "report",
        results={
            "schema": "simready-bench-matrix-result-v1",
            "planned_conditions": 2,
            "results": [
                {
                    "condition_id": "one",
                    "asset_id": "reference-box",
                    "requested_model": "example/model",
                    "harness_id": "pi",
                    "skill_set_id": "none",
                    "repetition": 0,
                    "status": "completed",
                    "snapshot_sha256": snapshot.sha256,
                    "submission": submission,
                    "evaluation": {
                        "simready": primary,
                        "nvidia_validator_report": {"issues": []},
                    },
                    "fidelity": {
                        "status": "measured",
                        "components": {
                            "shape": {
                                "score_percent": 0,
                                "coverage_fraction": 1,
                                "eligible": 1,
                            },
                            "joints": {"score_percent": None, "eligible": 0},
                        },
                    },
                    "illustrations": {"images": images},
                    "harbor_trace_paths": [str(trace)],
                    "inference_calls_used": 9,
                    "validation_calls_used": 5,
                    "inference_cost_usd": 3,
                },
                {
                    "condition_id": "two",
                    "asset_id": "reference-box",
                    "requested_model": "example/model",
                    "harness_id": "pi",
                    "skill_set_id": "none",
                    "repetition": 3,
                    "status": "blocked",
                    "blocker": "provider_unavailable",
                    "error_message": "Provider did not accept the request.",
                },
            ],
        },
    )
    data = json.loads(Path(report["results"]).read_text())["results"]
    assert data[0]["repetition"] == 0 and data[1]["repetition"] == 3
    assert data[0]["simready_reward"] == 0 and data[1]["simready_reward"] is None
    assert data[0]["inference_calls_used"] == 9
    exported = Path(report["directory"]) / data[0]["candidate"]["entrypoint"]
    assert (
        exported.is_file()
        and exported.read_bytes()
        == (Path(submission["directory"]) / "asset.usd").read_bytes()
    )
    assert (
        Path(report["directory"]) / data[0]["nvidia_validator_report_file"]["path"]
    ).is_file()
    with Path(report["scores"]).open() as stream:
        scores = list(csv.DictReader(stream))
    assert scores[0]["repetition"] == "1" and scores[1]["repetition"] == "4"
    assert scores[0]["simready_reward"] == scores[0]["fidelity_shape_percent"] == "0"
    assert scores[0]["fidelity_joints_percent"] == scores[1]["simready_reward"] == ""
    assert not {
        "diagnostic_reward",
        "official_reward",
        "components",
    } & set(scores[0])
    assert scores[0]["inference_calls_used"] == "9"
    assert scores[0]["validation_calls_used"] == "5"
    assert scores[0]["input_tokens"] == scores[1]["inference_calls_used"] == ""
    assert scores[0]["inference_cost_usd"] == "3"
    page = Path(report["html"]).read_text()
    for text in (
        "Formal fidelity (%)",
        "Visual fidelity (%)",
        "Kinematic fidelity (%)",
        "SimReady Robotics-Prop optional",
        "Shape (%)",
        "Rigid parts (%)",
        "Joints (%)",
        "Materials (%)",
        "example/model",
        "pi",
        "Repetition",
        "Reference: front",
        "Reconstruction: front",
        "Generated USD",
        "NVIDIA validation findings",
        "Harbor trajectory",
        "Not applicable",
    ):
        assert text in page
    assert (
        page.count("Failure details") == 1
        and "Provider did not accept the request." in page
    )
    assert "Reference: front | pi | none | repetition 1" in page
    for text in (
        "Normalized score",
        "Earned credit",
        "Scorable weight",
        "Diagnostic sum",
        "Official reward",
        "Active maximum",
        "Execution details",
    ):
        assert text not in page


def test_large_retained_report_compacts_without_raising_the_manifest_limit(tmp_path):
    from simready_bench.benchmark import _aggregate_reports, read_json
    from simready_bench.reporting import read_generated_report

    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "results"
    attempt = output / "attempts/000-reference-box"
    attempt.parent.mkdir(parents=True)
    trace = work / "trajectory.json"
    trace.write_text('{"steps": []}')
    final_findings = {
        "issues": [{"prim_path": "/Asset/Part", "message": "Original NVIDIA finding"}]
    }
    giant_evaluation = {
        "nvidia_validator_report": {
            "issues": [{"message": "old-turn-payload-" + "x" * (17 * 1024**2)}]
        }
    }
    report = export_report(
        work,
        attempt,
        results={
            "schema": "simready-bench-matrix-result-v1",
            "run_details": {
                "original_manifest": {"profile_inventory": giant_evaluation}
            },
            "results": [
                {
                    "condition_id": "one",
                    "asset_id": "reference-box",
                    "requested_model": "example/model",
                    "harness_id": "pi",
                    "skill_set_id": "none",
                    "repetition": 0,
                    "status": "completed",
                    "inference_calls_used": 1,
                    "harbor_trace_paths": [str(trace)],
                    "trajectory_events": [{"turn": 1, "evaluation": giant_evaluation}],
                    "evaluation": {
                        "simready": {
                            "schema": "simready-validator-score-v1",
                            "score_percent": 0,
                            "checks": [
                                {
                                    "requirement_id": "rule",
                                    "status": "fail",
                                    "findings": final_findings["issues"],
                                }
                            ],
                        },
                        "nvidia_validator_report": final_findings,
                    },
                    "gateway_records": [
                        {
                            "status": "completed",
                            "usage": {"input_tokens": 120, "output_tokens": 25},
                            "request_body": giant_evaluation,
                        }
                    ],
                }
            ],
        },
    )
    fresh = json.loads(Path(report["results"]).read_text())
    assert Path(report["results"]).stat().st_size < 100 * 1024
    assert "run_details" not in fresh and "trajectory_events" not in fresh["results"][0]
    assert "findings" not in fresh["results"][0]["simready"]["checks"][0]
    assert fresh["results"][0]["model_requests"][0]["usage"] == {
        "input_tokens": 120,
        "output_tokens": 25,
    }
    legacy = fresh
    legacy["run_details"] = {
        "original_manifest": {"profile_inventory": ["old inventory"]}
    }
    legacy["results"][0].update(
        trajectory_events=[{"evaluation": giant_evaluation}],
        nvidia_validator_report=final_findings,
        profile_inventory={"duplicate": True},
    )
    Path(report["results"]).write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="16 MiB"):
        read_json(report["results"])
    loaded = read_generated_report(report["results"])
    assert loaded["results"][0]["simready_reward"] == 0
    assert (
        "trajectory_events" not in loaded["results"][0] and "run_details" not in loaded
    )
    Path(report["scores"]).write_text(
        "asset_id,legacy_engineering_metric\nreference-box,123\n"
    )
    combined = _aggregate_reports(
        [(attempt, report)], output_dir=output, planned_conditions=1
    )
    assert Path(combined["results"]).stat().st_size < 100 * 1024
    compact = json.loads(Path(combined["results"]).read_text())["results"][0]
    assert (
        compact["condition_id"] == "one"
        and compact["simready_reward"] == 0
        and compact["repetition"] == 0
    )
    assert (
        not {"trajectory_events", "profile_inventory", "nvidia_validator_report"}
        & compact.keys()
    )
    findings = json.loads(
        (output / compact["nvidia_validator_report_file"]["path"]).read_text()
    )
    assert findings == final_findings
    assert (output / "evidence" / compact["traces"][0]["path"]).is_file()
    with Path(combined["scores"]).open() as stream:
        scores = list(csv.DictReader(stream))
    assert scores[0]["simready_reward"] == "0" and scores[0]["repetition"] == "1"
    assert "legacy_engineering_metric" not in scores[0]
    oversized = tmp_path / "oversized-report.json"
    with oversized.open("wb") as stream:
        stream.truncate(128 * 1024**2 + 1)
    with pytest.raises(ValueError, match="128 MiB"):
        read_generated_report(oversized)


def test_aggregate_preserves_same_named_evidence_across_execution_directories(tmp_path):
    import hashlib
    from simready_bench.benchmark import _aggregate_reports

    output = tmp_path / "results"
    reports = []
    expected = []
    for index, execution in enumerate(("first-execution", "resumed-execution")):
        work = tmp_path / ("work-" + str(index))
        work.mkdir()
        image = work / "capture.png"
        image.write_bytes(b"first capture" if index == 0 else b"resumed capture")
        trace = work / "trajectory.json"
        trace.write_text(json.dumps({"steps": [], "execution": execution}))
        attempt = output / execution / "attempts/000-reference-box"
        attempt.parent.mkdir(parents=True)
        report = export_report(
            work,
            attempt,
            results={
                "schema": "simready-bench-matrix-result-v1",
                "results": [
                    {
                        "condition_id": str(index),
                        "asset_id": "reference-box",
                        "repetition": index,
                        "status": "completed" if index == 0 else "blocked",
                        "harbor_trace_paths": [str(trace)],
                        "illustrations": {
                            "images": [
                                {
                                    "file": str(image),
                                    "view": "candidate / front",
                                    "sha256": hashlib.sha256(
                                        image.read_bytes()
                                    ).hexdigest(),
                                }
                            ]
                        },
                    }
                ],
            },
        )
        portable = json.loads(Path(report["results"]).read_text())
        for kind, name in (("images", "capture.png"), ("traces", "trajectory.json")):
            item = portable["results"][0][kind][0]
            (attempt / "evidence" / item["path"]).rename(attempt / "evidence" / name)
            item["path"] = name
        Path(report["results"]).write_text(json.dumps(portable))
        reports.append((attempt, report))
        expected.append((image.read_bytes(), trace.read_bytes()))
    combined = _aggregate_reports(reports, output_dir=output, planned_conditions=2)
    original = Path(combined["results"]).read_bytes()
    data = json.loads(original)
    assert data["completed_conditions"] == 1 and data["partial"] is True
    names = []
    for index, row in enumerate(data["results"]):
        image_name = row["images"][0]["path"]
        trace_name = row["traces"][0]["path"]
        assert (output / "evidence" / image_name).read_bytes() == expected[index][0]
        assert (output / "evidence" / trace_name).read_bytes() == expected[index][1]
        assert (
            row["report_path"]
            == reports[index][0].relative_to(output).as_posix() + "/report.html"
        )
        names.append((image_name, trace_name))
    assert names[0][0] != names[1][0] and names[0][1] != names[1][1]
    _aggregate_reports(reports, output_dir=output, planned_conditions=2)
    assert Path(combined["results"]).read_bytes() == original
