"""Whole-object pose invariance retains scale, articulation and material criteria."""

from copy import deepcopy
import math

import numpy as np
import pytest

from simready_bench.fidelity import compare, settings
from simready_bench.geometry_alignment import align_observation, transform_observation
from test_fidelity import box, observation


def rotation(degrees):
    angle = math.radians(degrees)
    return np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def asymmetric():
    triangles = np.concatenate([box((1, 2, 3)), box((0.4, 0.6, 0.7), (1.4, 0.3, 0.2))])
    return observation(triangles)


@pytest.mark.parametrize("yaw", [90, -90, 180, 19])
def test_shape_ignores_one_global_yaw_and_translation(yaw):
    reference = asymmetric()
    candidate = transform_observation(reference, rotation(yaw), np.array([4, -3, 2]))
    result = compare(reference, candidate, settings({"samples": 256}))
    assert result["schema"] == "source-usd-fidelity-v4"
    assert result["components"]["shape"]["score_percent"] > 99
    assert result["alignment"]["scale"] == 1
    assert np.allclose(result["alignment"]["rotation"][2], [0, 0, 1])


def test_alignment_preserves_si_scale_and_cannot_resize_the_candidate():
    reference = asymmetric()
    candidate = observation(reference["triangles"] * 3)
    aligned, details = align_observation(reference, candidate)
    assert details["scale"] == 1
    assert np.allclose(
        np.linalg.norm(aligned["triangles"][:, 1] - aligned["triangles"][:, 0], axis=1),
        3
        * np.linalg.norm(
            reference["triangles"][:, 1] - reference["triangles"][:, 0], axis=1
        ),
    )
    result = compare(reference, candidate, settings({"samples": 256}))
    assert result["components"]["shape"]["score_percent"] < 50


def test_one_transform_applies_to_geometry_body_positions_and_joint_frames():
    reference = asymmetric()
    reference["bodies"]["body"] = {
        "position": [0, 0, 0],
        "enabled": True,
        "kinematic": False,
    }
    reference["joints"] = [
        {
            "type": "PhysicsRevoluteJoint",
            "bodies": ["body", None],
            "anchors": [[0.5, 1, 1.5], [0.5, 1, 1.5]],
            "axes": [[1, 0, 0], [1, 0, 0]],
            "frames": [np.eye(3).tolist(), np.eye(3).tolist()],
            "enabled": True,
            "limits": {"state": "unbounded", "units": "radians", "bounds": {}},
        }
    ]
    candidate = transform_observation(reference, rotation(90), np.array([4, -3, 2]))
    before = deepcopy(candidate)
    result = compare(reference, candidate, settings({"samples": 256}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["joints"]["score_percent"] == 100
    assert result["components"]["rigid_parts"]["score_percent"] == 100
    assert np.array_equal(before["triangles"], candidate["triangles"])
    assert before["bodies"] == candidate["bodies"]
    assert before["joints"] == candidate["joints"]


def test_parts_cannot_be_independently_reposed_to_hide_their_relative_displacement():
    reference = observation(np.concatenate([box(), box(offset=(3, 0, 0))]))
    reference["triangle_bodies"] = ["first"] * 12 + ["second"] * 12
    reference["bodies"] = {
        "first": {"position": [0, 0, 0]},
        "second": {"position": [3, 0, 0]},
    }
    candidate = deepcopy(reference)
    candidate["triangles"][12:] += np.array([0, 1, 0])
    candidate["bodies"]["second"]["position"] = [3, 1, 0]
    result = compare(reference, candidate, settings({"samples": 256}))
    assert result["components"]["rigid_parts"]["score_percent"] < 90


def test_alignment_is_deterministic_and_does_not_credit_missing_materials():
    reference = asymmetric()
    reference["materials"] = [
        {"physics:restitution": 0.2} for _ in reference["triangles"]
    ]
    candidate = transform_observation(reference, rotation(90), np.array([4, -3, 2]))
    candidate["materials"] = [{} for _ in candidate["triangles"]]
    first, details = align_observation(reference, candidate)
    second, repeated = align_observation(reference, candidate)
    assert details == repeated
    assert np.array_equal(first["triangles"], second["triangles"])
    result = compare(reference, candidate, settings({"samples": 128}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["materials"]["score_percent"] == 0
