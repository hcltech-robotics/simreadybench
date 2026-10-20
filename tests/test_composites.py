from copy import deepcopy
import csv
import json

import pytest

from simready_bench.composites import scores
from simready_bench.reporting import export_report


def condition():
    return {
        "condition_id": "synthetic-fixture",
        "evaluation": {
            "simready": {
                "score_percent": 100,
                "passed_count": 46,
                "failed_count": 0,
                "applicable_count": 46,
                "official_conformance": {"passed": True},
            }
        },
        "fidelity": {
            "components": {
                "shape": {"score_percent": 64, "status": "measured", "eligible": 1},
                "materials": {
                    "features": {
                        "visual:diffuseColor": {"eligible": 4, "earned": 1},
                        "visual:assignment": {"eligible": 4, "earned": 4},
                        "physics:density": {"eligible": 4, "earned": 4},
                    }
                },
                "kinematics": {
                    "score_percent": 0,
                    "status": "measured",
                    "earned": 0,
                    "eligible": 2,
                    "measured": 2,
                    "coverage_fraction": 1,
                },
            }
        },
    }


def test_three_composites_are_independent_and_visual_excludes_physics():
    value = condition()
    original = deepcopy(value)
    result = scores(value)
    assert result["formal"]["score_percent"] == 100
    assert result["formal"]["passed"] is True
    assert result["formal"]["measured"] == 46
    assert result["visual"]["score_percent"] == pytest.approx(40)
    assert result["kinematic"]["score_percent"] == 0
    assert result["visual"]["factors"]["appearance"]["earned"] == 1
    assert value == original


def test_visual_geometric_mean_preserves_a_failed_factor():
    value = condition()
    value["fidelity"]["components"]["shape"]["score_percent"] = 0
    assert scores(value)["visual"]["score_percent"] == 0


def test_unknown_appearance_is_not_replaced_with_shape():
    value = condition()
    features = value["fidelity"]["components"]["materials"]["features"]
    del features["visual:diffuseColor"]
    result = scores(value)["visual"]
    assert result["status"] == "unknown" and result["score_percent"] is None
    assert result["coverage_fraction"] == 0.5


def test_reference_without_visual_materials_uses_its_eligible_shape():
    value = condition()
    value["fidelity"]["components"]["materials"] = {"features": {}}
    result = scores(value)["visual"]
    assert result["score_percent"] == 64
    assert result["factors"]["appearance"]["status"] == "not_applicable"


def test_old_reports_do_not_infer_motion_from_joint_attribute_checks():
    value = condition()
    value["fidelity"]["components"].pop("kinematics")
    value["fidelity"]["components"]["joints"] = {"score_percent": 100}
    assert scores(value)["kinematic"]["score_percent"] is None


def test_portable_readout_contains_three_composites(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    paths = export_report(
        work,
        tmp_path / "report",
        results={
            "schema": "simready-bench-matrix-result-v1",
            "results": [condition()],
        },
    )
    result = json.loads(open(paths["results"]).read())["results"][0]
    assert result["composite_scores"]["kinematic"]["score_percent"] == 0
    row = next(csv.DictReader(open(paths["scores"])))
    assert float(row["formal_fidelity_percent"]) == 100
    assert float(row["visual_fidelity_percent"]) == pytest.approx(40)
    assert float(row["kinematic_fidelity_percent"]) == 0
    assert row["formal_fidelity_measured"] == "46"
    page = open(paths["html"]).read()
    for name in ("Formal", "Visual", "Kinematic"):
        assert name + " fidelity (%)" in page


def test_unsupported_reference_regions_retain_partial_appearance_and_bounds():
    from simready_bench.fidelity import _appearance_component

    rows = [
        {"sample": i, "feature": "visual:assignment", "passed": True} for i in range(8)
    ]
    rows.extend(
        {"sample": i, "feature": "visual:diffuseColor", "passed": True}
        for i in range(4)
    )
    appearance = _appearance_component(rows)
    assert appearance["score_percent"] is None
    assert appearance["earned"] == 4 and appearance["eligible"] == 8
    assert appearance["measured"] == 4 and appearance["coverage_fraction"] == 0.5
    assert (
        appearance["lower_bound_percent"] == 50
        and appearance["upper_bound_percent"] == 100
    )
    value = condition()
    value["fidelity"]["components"]["materials"]["appearance"] = appearance
    visual = scores(value)["visual"]
    assert visual["score_percent"] is None
    assert visual["lower_bound_percent"] == pytest.approx((64 * 50) ** 0.5)
    assert visual["upper_bound_percent"] == pytest.approx(80)


def test_appearance_samples_have_equal_area_weight_with_different_property_counts():
    from simready_bench.fidelity import _appearance_component

    rows = [
        {"sample": 0, "feature": "visual:diffuseColor", "passed": True},
        {"sample": 0, "feature": "visual:roughness", "passed": True},
        {"sample": 1, "feature": "visual:diffuseColor", "passed": False},
    ]
    appearance = _appearance_component(rows)
    assert appearance["score_percent"] == 50
    assert appearance["earned"] == 1 and appearance["eligible"] == 2


def test_comparison_keeps_mixed_shader_coverage_in_visual_composite():
    from simready_bench.fidelity import compare, settings
    from test_kinematics import fixture

    reference = fixture()
    reference["materials"] = [
        {"visual:assignment": True, "visual:diffuseColor": [1, 0, 0]},
        {"visual:assignment": True},
    ]
    reference["eligibility"] = {
        "visible_prims": 2,
        "supported_prims": 2,
        "unsupported": [],
    }
    result = compare(reference, deepcopy(reference), settings({"samples": 128}))
    composite = scores({"fidelity": result})["visual"]
    appearance = composite["factors"]["appearance"]
    assert 0 < appearance["measured"] < appearance["eligible"] == 128
    assert 0 < appearance["coverage_fraction"] < 1
    assert composite["score_percent"] is None
    assert composite["lower_bound_percent"] < composite["upper_bound_percent"]
