import copy
import numpy as np
from simready_bench.fidelity import compare, settings


def box(extent=(1, 1, 1), offset=(0, 0, 0)):
    points = (
        np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        * extent
        + offset
    )
    faces = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]
    return np.array(
        [[points[f[0]], points[f[i]], points[f[i + 1]]] for f in faces for i in (1, 2)]
    )


def observation(triangles, materials=None, joints=None, body="body"):
    return {
        "triangles": triangles,
        "triangle_bodies": [body] * len(triangles),
        "materials": materials or [{} for _ in triangles],
        "bodies": {body: {}},
        "joints": joints or [],
        "eligibility": {"visible_prims": 1, "supported_prims": 1, "unsupported": []},
    }


def test_equivalent_retessellation_and_names_pass_but_cube_cannot_match_non_cube():
    reference = observation(box((1, 2, 3)))
    # Every triangle is split around its centroid: identical surface, different topology/name.
    triangles = np.array(
        [[a, b, (a + b + c) / 3] for a, b, c in reference["triangles"]]
        + [[b, c, (a + b + c) / 3] for a, b, c in reference["triangles"]]
        + [[c, a, (a + b + c) / 3] for a, b, c in reference["triangles"]]
    )
    equivalent = compare(
        reference, observation(triangles, body="renamed"), settings({"samples": 256})
    )
    assert equivalent["components"]["shape"]["score_percent"] == 100
    cube = compare(reference, observation(box()), settings({"samples": 256}))
    assert cube["components"]["shape"]["score_percent"] < 50
    assert cube["aggregate_score_percent"] is None


def test_reference_missingness_is_fixed_candidate_material_omission_loses_credit():
    triangles = box()
    materials = [
        {"physics:staticFriction": 0.7, "visual:diffuseColor": [0.4, 0.3, 0.2]}
        for _ in triangles
    ]
    reference = observation(triangles, materials)
    good = compare(reference, copy.deepcopy(reference), settings({"samples": 128}))
    missing = compare(reference, observation(triangles), settings({"samples": 128}))
    assert good["components"]["materials"]["score_percent"] == 100
    assert missing["components"]["materials"]["score_percent"] == 0
    assert (
        good["reference_eligibility_sha256"] == missing["reference_eligibility_sha256"]
    )
    assert (
        good["components"]["materials"]["eligible"]
        == missing["components"]["materials"]["eligible"]
    )
    absent = compare(observation(triangles), reference, settings({"samples": 128}))
    assert absent["components"]["materials"]["status"] == "not_applicable"


def test_missing_joint_is_failure_and_joint_free_reference_has_no_free_points():
    triangles = box()
    joint = {
        "type": "PhysicsRevoluteJoint",
        "bodies": ["body", None],
        "anchors": [[0, 0, 0], [0, 0, 0]],
        "axes": [[1, 0, 0], [1, 0, 0]],
        "limits": {
            "state": "bounded",
            "units": "radians",
            "bounds": {
                "physics:lowerLimit": {"state": "finite", "value": 0},
                "physics:upperLimit": {"state": "finite", "value": 1},
            },
        },
    }
    reference = observation(triangles, joints=[joint])
    result = compare(reference, observation(triangles), settings({"samples": 128}))
    assert result["components"]["joints"]["score_percent"] == 0
    assert result["components"]["joints"]["eligible"] > 0
    empty = compare(
        observation(triangles), observation(triangles), settings({"samples": 128})
    )
    assert empty["components"]["joints"]["score_percent"] is None
    assert empty["components"]["joints"]["status"] == "not_applicable"


def test_invalid_candidate_does_not_remove_reference_criteria():
    reference = observation(box(), [{"physics:restitution": 0.2}] * 12)
    result = compare(
        reference, None, settings({"samples": 128}), candidate_invalid=True
    )
    assert result["components"]["shape"]["score_percent"] == 0
    assert result["components"]["materials"]["score_percent"] == 0
    assert result["components"]["materials"]["eligible"] > 0


def test_point_to_surface_uses_triangle_interior_and_extra_bodies_lose_structure_credit():
    from simready_bench.fidelity import nearest

    distance, _, _ = nearest(
        [[0.25, 0.25, 0], [1, 1, 0]], [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]
    )
    assert distance[0] == 0 and abs(distance[1] - 2**-0.5) < 1e-12
    reference = observation(box())
    candidate = observation(np.concatenate([box(), box()]))
    candidate["triangle_bodies"] = ["body"] * 12 + ["extra"] * 12
    candidate["bodies"]["extra"] = {}
    result = compare(reference, candidate, settings({"samples": 128}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["rigid_parts"]["score_percent"] < 100


def test_identical_partial_reference_has_fixed_partial_coverage():
    reference = observation(box())
    reference["eligibility"] = {
        "visible_prims": 2,
        "supported_prims": 1,
        "unsupported": [
            {"prim": "/Curve", "reason": "unsupported curve", "geometry": True}
        ],
    }
    result = compare(reference, copy.deepcopy(reference), settings({"samples": 128}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["shape"]["coverage_fraction"] == 0.5
    assert result["components"]["shape"]["full_surface_correspondence"] == "partial"


def test_swapped_joint_endpoints_require_reversed_limit_coordinate():
    reference = observation(box())
    reference["bodies"]["other"] = {}
    joint = {
        "type": "PhysicsRevoluteJoint",
        "bodies": ["body", "other"],
        "anchors": [[0, 0, 0], [0, 0, 0]],
        "axes": [[1, 0, 0], [1, 0, 0]],
        "frames": [np.eye(3).tolist(), np.eye(3).tolist()],
        "enabled": True,
        "limits": {
            "state": "bounded",
            "units": "radians",
            "bounds": {
                "physics:lowerLimit": {"state": "finite", "value": 0.0},
                "physics:upperLimit": {"state": "finite", "value": 1.0},
            },
        },
    }
    reference["joints"] = [joint]
    candidate = copy.deepcopy(reference)
    candidate["joints"][0]["bodies"].reverse()
    bad = compare(reference, candidate, settings({"samples": 128}))
    assert bad["components"]["joints"]["score_percent"] < 100
    candidate["joints"][0]["limits"]["bounds"]["physics:lowerLimit"]["value"] = -1.0
    candidate["joints"][0]["limits"]["bounds"]["physics:upperLimit"]["value"] = 0.0
    good = compare(reference, candidate, settings({"samples": 128}))
    assert good["components"]["joints"]["score_percent"] == 100


def test_fidelity_configuration_is_bounded_and_does_not_blend_rewards():
    import pytest

    for value in (
        [],
        {"samples": 0},
        {"distance_m": float("nan")},
        {"enabled": "yes"},
        {"weight": 0.5},
    ):
        with pytest.raises(ValueError):
            settings(value)
    result = compare(observation(box()), observation(box()), settings({"samples": 128}))
    assert (
        result["compliance_weight"] is None
        and result["aggregate_score_percent"] is None
    )
    assert result["illustrations_scored"] is False


def test_extra_joint_on_joint_free_reference_fails_fixed_structure_without_free_articulation_credit():
    reference = observation(box())
    candidate = copy.deepcopy(reference)
    candidate["joints"] = [
        {
            "type": "PhysicsFixedJoint",
            "bodies": ["body", None],
            "anchors": [[0, 0, 0], [0, 0, 0]],
            "axes": [],
        }
    ]
    result = compare(reference, candidate, settings({"samples": 128}))
    assert result["components"]["joints"]["status"] == "not_applicable"
    assert result["components"]["joints"]["score_percent"] is None
    assert result["components"]["joints"]["extra_joint_count"] == 1
    assert result["components"]["rigid_parts"]["score_percent"] < 100
    assert any(
        r["feature"] == "joint_inventory_count" and not r["passed"]
        for r in result["components"]["rigid_parts"]["structure"]["checks"]
    )
    toggled = compare(
        reference, candidate, settings({"samples": 128, "joint_existence_type": False})
    )
    assert toggled["components"]["rigid_parts"]["score_percent"] == 100
    assert all(
        r["feature"] != "joint_inventory_count"
        for r in toggled["components"]["rigid_parts"]["structure"]["checks"]
    )


def test_partial_reference_cannot_claim_complete_reconstruction_fidelity():
    reference = observation(box())
    reference["eligibility"].update(
        visible_prims=2,
        supported_prims=1,
        unsupported=[
            {
                "prim": "missing",
                "geometry": True,
                "reason": "unsupported source surface",
            }
        ],
    )
    result = compare(reference, copy.deepcopy(reference), settings({"samples": 128}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["reference_geometry_status"] == "partial"
    assert result["complete_reconstruction_success"] is None
    assert result["complete_geometry_comparison"] is False


def test_thin_triangle_interior_has_stable_distance_and_barycentric_coordinates():
    from simready_bench.fidelity import nearest

    triangle = np.array([[[0, 0, 0], [1, 0, 0], [1, 1e-10, 0]]], float)
    distance, index, bary = nearest(triangle.mean(axis=1), triangle)
    assert distance[0] < 1e-24
    assert np.allclose(bary[0], [1 / 3] * 3, atol=1e-12)


def test_spatial_nearest_matches_exhaustive_distances_and_material_triangle_identity():
    from simready_bench.fidelity import nearest, _nearest_bruteforce

    rng = np.random.default_rng(341)
    triangles = rng.normal(size=(4200, 3, 3))
    points = rng.normal(size=(19, 3))
    expected = _nearest_bruteforce(points, triangles)
    actual = nearest(points, triangles)
    assert np.allclose(actual[0], expected[0], atol=1e-12)
    assert np.array_equal(actual[1], expected[1])
    assert np.allclose(actual[2], expected[2], atol=1e-12)


def test_spatial_nearest_prunes_distant_geometry_and_keeps_lowest_index_ties():
    from simready_bench.fidelity import _TriangleIndex

    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    triangles = triangle[None] + np.arange(10000)[:, None, None] * np.array([10, 0, 0])
    triangles[9999] = triangles[0]
    index = _TriangleIndex(triangles)
    distance, chosen, _ = index.query([[0.2, 0.2, 1], [10000.2, 0.2, 0]])
    assert np.allclose(distance, [1, 0]) and chosen[0] == 0 and chosen[1] == 1000
    assert index.triangle_tests < 2000


def test_single_surface_fast_path_respects_mapped_body_with_no_geometry():
    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    reference = observation(triangle)
    candidate = observation(
        np.concatenate([triangle, triangle * 1e-5 + np.array([10, 0, 0])])
    )
    reference["bodies"]["empty"] = {"position": [5, 0.5, 0]}
    candidate["bodies"]["empty"] = {"position": [0.5, 0.5, 0]}
    result = compare(reference, candidate, settings({"samples": 128}))
    assert result["body_correspondence"]["body"] == "empty"
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["rigid_parts"]["score_percent"] == 0


def test_candidate_resource_failure_propagates_instead_of_becoming_measured_zero(
    tmp_path,
):
    import hashlib
    from pathlib import Path
    from unittest import TestCase
    from unittest.mock import patch
    from simready_bench import fidelity_worker
    from simready_bench.fidelity import digest, reference_profile

    source = tmp_path / "source.usda"
    candidate = tmp_path / "candidate.usda"
    source.write_text("source fixture")
    candidate.write_text("candidate fixture")
    reference = observation(box())
    cfg = settings({"samples": 128})
    request = {
        "operation": "compare",
        "reference": str(source),
        "candidate": str(candidate),
        "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": cfg,
        "profile_sha256": digest(reference_profile(reference, cfg)),
    }
    # Role confinement and dependency rejection have separate mounted controls.
    # This isolated fault injection exercises only observation error treatment.
    with (
        patch.object(Path, "is_relative_to", return_value=True),
        patch.object(Path, "relative_to", return_value=Path("candidate.usda")),
        patch(
            "simready_bench.engine.package_checks.inspect_package",
            return_value={"package_ok": True},
        ),
        patch.object(
            fidelity_worker,
            "observe",
            side_effect=[reference, MemoryError("CPU surface byte budget exceeded")],
        ),
    ):
        with TestCase().assertRaises(MemoryError):
            fidelity_worker.run(request)
