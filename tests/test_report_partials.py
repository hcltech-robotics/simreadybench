import csv
import json

from simready_bench.reporting import compact_report, render_html, write_scores


def condition():
    optional = {
        "status": "scored",
        "score_percent": 100 * 33 / 51,
        "passed_count": 33,
        "failed_count": 18,
        "unknown_count": 0,
        "applicable_count": 51,
        "measured_count": 51,
        "coverage_fraction": 1,
        "lower_bound_percent": 100 * 33 / 51,
        "upper_bound_percent": 100 * 33 / 51,
    }
    return {
        "condition_id": "condition-fixture",
        "asset_id": "articulated_prop",
        "status": "completed",
        "simready_reward": 100,
        "simready": {
            "passed_count": 46,
            "applicable_count": 46,
            "optional_score": optional,
        },
        "optional_score": optional,
        "fidelity": {
            "components": {
                "shape": {
                    "score_percent": 100 / 3,
                    "earned": 1 / 3,
                    "eligible": 1,
                    "measured": 1,
                    "coverage_fraction": 0.75,
                    "precision": 0.5,
                    "recall": 0.25,
                },
                "rigid_parts": {
                    "score_percent": 25,
                    "earned": 1,
                    "eligible": 4,
                    "measured": 4,
                    "coverage_fraction": 1,
                },
                "joints": {
                    "score_percent": 40,
                    "earned": 4,
                    "eligible": 10,
                    "measured": 10,
                    "coverage_fraction": 1,
                    "reference_joint_count": 2,
                    "candidate_joint_count": 1,
                    "checks": [
                        {"feature": "anchors", "passed": True},
                        {"feature": "anchors", "passed": False},
                    ],
                },
                "materials": {
                    "score_percent": 12.5,
                    "earned": 1,
                    "eligible": 8,
                    "measured": 8,
                    "coverage_fraction": 0.5,
                    "features": {"visual:roughness": {"earned": 1, "eligible": 8}},
                },
            }
        },
    }


def test_table_shows_partial_credit_and_eligible_counts_for_every_component():
    text = render_html({"results": [condition()]})
    for fraction in ("33/51", "0.333333/1", "1/4", "4/10", "1/8"):
        assert fraction in text
    assert "40% (4/10)" in text
    assert "Rigid-part correspondence" not in text
    assert "Joint correspondence" in text
    assert "anchors" in text
    assert "Material correspondence" in text


def test_no_eligible_joints_is_na_while_zero_of_eligible_checks_is_zero():
    row = condition()
    joints = row["fidelity"]["components"]["joints"]
    joints.update(
        earned=0, eligible=0, measured=0, score_percent=None, status="not_applicable"
    )
    assert ">N/A<" in render_html({"results": [row]})
    joints.update(eligible=10, measured=10, score_percent=0, status="measured")
    assert "0% (0/10)" in render_html({"results": [row]})


def test_csv_retains_full_precision_and_independent_optional_and_fidelity_partials(
    tmp_path,
):
    path = tmp_path / "scores.csv"
    row = condition()
    write_scores(path, [row])
    with path.open(newline="") as stream:
        actual = next(csv.DictReader(stream))
    assert float(actual["fidelity_shape_earned"]) == 1 / 3
    assert float(actual["fidelity_shape_percent"]) == 100 / 3
    assert actual["fidelity_joints_earned"] == "4"
    assert actual["fidelity_joints_eligible"] == "10"
    assert actual["optional_passed"] == "33"
    assert actual["optional_total"] == "51"
    assert float(actual["optional_score_percent"]) == 100 * 33 / 51


def test_compaction_preserves_optional_measurements_and_unmeasured_bounds():
    row = condition()
    row["optional_score"].update(
        status="unknown",
        score_percent=None,
        unknown_count=2,
        failed_count=16,
        measured_count=49,
        coverage_fraction=49 / 51,
        upper_bound_percent=100 * 35 / 51,
    )
    result = compact_report(
        {"schema": "simready-bench-portable-report-v1", "results": [row]}
    )
    actual = result["results"][0]
    assert actual["simready"]["optional_score"] == row["optional_score"]
    assert actual["optional_score"] == row["optional_score"]
    assert actual["optional_unknown"] == 2
    assert actual["fidelity_materials_earned"] == 1
    assert "2 unmeasured" in render_html(result)
    assert "bounds 64.71%-68.63%" in render_html(result)
    json.dumps(result, allow_nan=False)
