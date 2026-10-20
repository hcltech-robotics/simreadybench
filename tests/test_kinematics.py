import copy
import math

import numpy as np
import pytest

from simready_bench.fidelity import settings
from simready_bench.kinematics import compare_kinematics


def fixture(kind="PhysicsRevoluteJoint"):
    triangle = np.array([[[0.2, 0, 0], [0.5, 0.2, 0], [0.5, 0, 0.3]]])
    joint = {
        "path": "/Joint",
        "type": kind,
        "bodies": ["base", "leaf"],
        "anchors": [[0, 0, 0], [0, 0, 0]],
        "axes": [[0, 0, 1], [0, 0, 1]],
        "frames": [np.eye(3).tolist(), np.eye(3).tolist()],
        "enabled": True,
        "limits": {
            "state": "bounded",
            "units": "radians" if kind == "PhysicsRevoluteJoint" else "m",
            "bounds": {
                "physics:lowerLimit": {"state": "finite", "value": 0.0},
                "physics:upperLimit": {"state": "finite", "value": 1.0},
            },
        },
    }
    observation = {
        "triangles": np.concatenate([triangle, triangle + 1]),
        "triangle_bodies": ["base", "leaf"],
        "bodies": {
            "base": {"enabled": True, "kinematic": True},
            "leaf": {"enabled": True, "kinematic": False},
        },
        "joints": [joint],
    }
    return observation


def measure(reference, candidate, **kwargs):
    return compare_kinematics(
        reference,
        candidate,
        {"base": "base", "leaf": "leaf"},
        settings(),
        2.0,
        **kwargs,
    )


@pytest.mark.parametrize(
    "kind", ["PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]
)
def test_identical_joint_has_full_credit_with_kinematic_base(kind):
    reference = fixture(kind)
    result = measure(reference, copy.deepcopy(reference))
    assert result["score_percent"] == 100
    assert result["matched_joint_count"] == 1
    assert result["joints"][0]["motion_rms_m"] == 0
    assert result["coverage_fraction"] == 1


def test_joint_free_source_is_not_applicable_and_spurious_joints_fail():
    candidate = fixture()
    reference = copy.deepcopy(candidate)
    reference["joints"] = []
    empty = copy.deepcopy(reference)
    result = measure(reference, empty)
    assert result["score_percent"] is None and result["eligible"] == 0
    assert result["status"] == "not_applicable"
    spurious = measure(reference, candidate)
    assert spurious["score_percent"] == 0
    assert spurious["extra_joint_count"] == 1


def test_missing_hinge_cannot_be_hidden_by_many_correct_fixed_constraints():
    reference = fixture()
    fixed = fixture("PhysicsFixedJoint")["joints"][0]
    reference["joints"] += [copy.deepcopy(fixed) for _ in range(12)]
    candidate = copy.deepcopy(reference)
    candidate["joints"] = candidate["joints"][1:]
    result = measure(reference, candidate)
    assert result["groups"]["fixed_constraints"]["score_percent"] == 100
    assert result["groups"]["moving_joints"]["score_percent"] == 0
    assert result["score_percent"] == 0


def test_extra_joint_loses_recovery_precision():
    reference = fixture()
    candidate = copy.deepcopy(reference)
    candidate["joints"].append(copy.deepcopy(candidate["joints"][0]))
    result = measure(reference, candidate)
    assert result["score_percent"] == pytest.approx(200 / 3)
    assert result["groups"]["moving_joints"]["earned"] == 2
    assert result["groups"]["moving_joints"]["eligible"] == 3
    assert result["extra_joint_count"] == 1


@pytest.mark.parametrize(
    "change", ["type", "connectivity", "joint_disabled", "body_disabled", "immobilised"]
)
def test_wrong_or_immobilised_joint_gets_zero_motion_credit(change):
    reference = fixture()
    candidate = copy.deepcopy(reference)
    joint = candidate["joints"][0]
    if change == "type":
        joint["type"] = "PhysicsFixedJoint"
    elif change == "connectivity":
        joint["bodies"] = ["base", None]
    elif change == "joint_disabled":
        joint["enabled"] = False
    elif change == "body_disabled":
        candidate["bodies"]["leaf"]["enabled"] = False
    else:
        candidate["bodies"]["leaf"]["kinematic"] = True
    assert measure(reference, candidate)["score_percent"] == 0


@pytest.mark.parametrize("state", ["joint_disabled", "body_disabled", "kinematic"])
def test_reference_disabled_or_kinematic_state_is_preserved(state):
    reference = fixture()
    if state == "joint_disabled":
        reference["joints"][0]["enabled"] = False
    elif state == "body_disabled":
        reference["bodies"]["leaf"]["enabled"] = False
    else:
        reference["bodies"]["leaf"]["kinematic"] = True
    assert measure(reference, copy.deepcopy(reference))["score_percent"] == 100


def test_hinge_origin_can_shift_along_its_own_axis():
    reference = fixture()
    candidate = copy.deepcopy(reference)
    candidate["joints"][0]["anchors"] = [[0, 0, 4], [0, 0, 4]]
    assert measure(reference, candidate)["score_percent"] == 100


@pytest.mark.parametrize("kind", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
def test_axis_reversal_with_negated_limits_is_equivalent(kind):
    reference = fixture(kind)
    candidate = copy.deepcopy(reference)
    joint = candidate["joints"][0]
    joint["axes"] = [[0, 0, -1], [0, 0, -1]]
    joint["limits"]["bounds"]["physics:lowerLimit"]["value"] = -1.0
    joint["limits"]["bounds"]["physics:upperLimit"]["value"] = 0.0
    assert measure(reference, candidate)["score_percent"] == 100


@pytest.mark.parametrize(
    "kind", ["PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]
)
def test_swapping_endpoints_and_limit_coordinate_preserves_the_constraint(kind):
    reference = fixture(kind)
    candidate = copy.deepcopy(reference)
    joint = candidate["joints"][0]
    for key in ("bodies", "anchors", "frames", "axes"):
        joint[key].reverse()
    joint["limits"]["bounds"]["physics:lowerLimit"]["value"] = -1.0
    joint["limits"]["bounds"]["physics:upperLimit"]["value"] = 0.0
    assert measure(reference, candidate)["score_percent"] == 100


@pytest.mark.parametrize("change", ["pivot", "axis", "range", "attachment"])
def test_continuous_error_retains_partials_and_declines_as_motion_error_grows(change):
    reference = fixture()
    scores = []
    for amount in (0.001, 0.01, 0.1):
        candidate = copy.deepcopy(reference)
        joint = candidate["joints"][0]
        if change == "pivot":
            joint["anchors"] = [[amount, 0, 0], [amount, 0, 0]]
        elif change == "axis":
            joint["axes"] = [[math.sin(amount), 0, math.cos(amount)]] * 2
        elif change == "range":
            joint["limits"]["bounds"]["physics:upperLimit"]["value"] += amount
        else:
            joint["anchors"][1][0] += amount
        result = measure(reference, candidate)
        assert 0 < result["score_percent"] < 100
        scores.append(result["score_percent"])
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize(
    "change", ["spherical", "unbounded", "missing_frame", "ambiguous"]
)
def test_unsupported_reference_semantics_and_ambiguous_body_matching_remain_unknown(
    change,
):
    reference = fixture()
    kwargs = {}
    if change == "spherical":
        reference["joints"][0]["type"] = "PhysicsSphericalJoint"
    elif change == "unbounded":
        reference["joints"][0]["limits"]["state"] = "unbounded"
    elif change == "missing_frame":
        reference["joints"][0].pop("frames")
    else:
        kwargs["ambiguous_bodies"] = ["leaf"]
    result = measure(reference, copy.deepcopy(reference), **kwargs)
    assert result["score_percent"] is None
    assert result["status"] == "unknown"
    assert result["joints"][0]["earned"] is None
    assert result["coverage_fraction"] < 1


def test_reference_unknown_upper_bounds_do_not_exceed_100_with_missing_candidates():
    reference = fixture()
    reference["joints"][0]["type"] = "PhysicsSphericalJoint"
    result = measure(reference, {"joints": []})
    assert result["groups"]["moving_joints"]["upper_bound_percent"] == 0


def test_reference_eligibility_does_not_depend_on_candidate_geometry():
    reference = fixture()
    reference["triangle_bodies"] = ["unrelated", "unrelated"]
    complete = measure(reference, copy.deepcopy(reference))
    missing = measure(reference, {})
    assert complete["status"] == missing["status"] == "unknown"
    assert complete["joints"][0]["measured"] == missing["joints"][0]["measured"] == 0


def test_common_fixed_joint_frame_gauge_does_not_change_constraint():
    from simready_bench.kinematics import _rotation

    reference = fixture("PhysicsFixedJoint")
    joint = reference["joints"][0]
    joint["anchors"] = [[0.2, 0, 0], [0.1, 0.3, 0]]
    joint["frames"][1] = _rotation([0, 0, 1], 0.2).tolist()
    candidate = copy.deepcopy(reference)
    other = candidate["joints"][0]
    gauge = _rotation([1, 0, 0], 0.4)
    offset = np.array([0.1, 0.4, 0.3])
    for endpoint in (0, 1):
        frame = np.asarray(joint["frames"][endpoint])
        other["anchors"][endpoint] = (
            np.asarray(joint["anchors"][endpoint]) + frame.T @ offset
        ).tolist()
        other["frames"][endpoint] = (gauge.T @ frame).tolist()
    assert measure(reference, candidate)["score_percent"] == pytest.approx(100)


def test_joint_coordinate_zero_shift_with_compensating_limits_is_equivalent():
    from simready_bench.kinematics import _rotation

    reference = fixture()
    candidate = copy.deepcopy(reference)
    joint = candidate["joints"][0]
    shift = 0.3
    joint["frames"][1] = _rotation([0, 0, 1], -shift).tolist()
    joint["limits"]["bounds"]["physics:lowerLimit"]["value"] += shift
    joint["limits"]["bounds"]["physics:upperLimit"]["value"] += shift
    assert measure(reference, candidate)["score_percent"] == pytest.approx(100)


def test_actual_usd_joint_units_and_up_axis_normalisation_share_kinematic_score(
    tmp_path,
):
    pytest.importorskip("pxr")
    from simready_bench.fidelity import compare
    from simready_bench.usd_geometry import observe
    from test_fidelity_usd import write

    source = tmp_path / "source.usda"
    candidate = tmp_path / "candidate.usda"
    write(source)
    write(candidate, name="Renamed", units=0.01, up="Y", retessellated=True)
    result = compare(observe(source), observe(candidate), settings({"samples": 128}))
    assert result["schema"] == "source-usd-fidelity-v4"
    assert result["components"]["kinematics"]["score_percent"] == pytest.approx(100)


def test_one_global_alignment_applies_to_joint_motion_and_common_landmarks():
    from simready_bench.fidelity import compare
    from simready_bench.geometry_alignment import transform_observation
    from test_geometry_alignment import rotation

    reference = fixture()
    reference["materials"] = [{}, {}]
    reference["eligibility"] = {
        "visible_prims": 2,
        "supported_prims": 2,
        "unsupported": [],
    }
    candidate = transform_observation(reference, rotation(90), np.array([4, -3, 2]))
    result = compare(reference, candidate, settings({"samples": 128}))
    assert result["components"]["kinematics"]["score_percent"] == pytest.approx(100)
    assert result["alignment"]["scale"] == 1


def set_limits(joint, lower, upper):
    bounds = {}
    for field, value in (("physics:lowerLimit", lower), ("physics:upperLimit", upper)):
        bounds[field] = (
            {
                "state": "unbounded",
                "value": None,
                "direction": "negative" if value < 0 else "positive",
            }
            if math.isinf(value)
            else {"state": "finite", "value": value}
        )
    joint["limits"].update(
        state="unbounded" if math.isinf(lower) and math.isinf(upper) else "bounded",
        bounds=bounds,
    )


@pytest.mark.parametrize("kind", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize(
    "interval", [(-math.inf, math.inf), (-math.inf, 0.4), (-0.2, math.inf)]
)
def test_known_infinite_bounds_have_full_identity_credit_without_rewriting_authored_limits(
    kind, interval
):
    reference = fixture(kind)
    set_limits(reference["joints"][0], *interval)
    before = copy.deepcopy(reference["joints"][0]["limits"])
    result = measure(reference, copy.deepcopy(reference))
    assert result["score_percent"] == 100
    assert reference["joints"][0]["limits"] == before
    probes = result["joints"][0]["reference_probe_interval"]
    assert np.isfinite(probes).all()
    assert probes[1] - probes[0] == pytest.approx(
        2 * math.pi if kind == "PhysicsRevoluteJoint" else 2.0
    )
    assert result["joints"][0]["limit_pattern_mismatches"] == 0


@pytest.mark.parametrize("kind", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize(
    "interval", [(-math.inf, math.inf), (-math.inf, 0.4), (-0.2, math.inf)]
)
def test_infinite_bound_axis_and_endpoint_reversals_preserve_the_limit_contract(
    kind, interval
):
    reference = fixture(kind)
    set_limits(reference["joints"][0], *interval)
    for reverse in ("axis", "endpoints"):
        candidate = copy.deepcopy(reference)
        other = candidate["joints"][0]
        if reverse == "axis":
            other["axes"] = [[0, 0, -1], [0, 0, -1]]
        else:
            for key in ("bodies", "anchors", "frames", "axes"):
                other[key].reverse()
        set_limits(other, -interval[1], -interval[0])
        assert measure(reference, candidate)["score_percent"] == pytest.approx(100)


@pytest.mark.parametrize("kind", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
@pytest.mark.parametrize(
    "interval", [(-math.inf, math.inf), (-math.inf, 0.4), (-0.2, math.inf)]
)
def test_infinite_bound_coordinate_origin_is_a_gauge(kind, interval):
    from simready_bench.kinematics import _rotation

    reference = fixture(kind)
    set_limits(reference["joints"][0], *interval)
    candidate = copy.deepcopy(reference)
    other = candidate["joints"][0]
    shift = 0.7
    if kind == "PhysicsRevoluteJoint":
        other["frames"][1] = _rotation([0, 0, 1], -shift).tolist()
    else:
        other["anchors"][1][2] += shift
    set_limits(other, interval[0] + shift, interval[1] + shift)
    result = measure(reference, candidate)
    assert result["score_percent"] == pytest.approx(100)


@pytest.mark.parametrize("kind", ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"])
def test_finite_substitute_cannot_get_full_credit_for_unbounded_joint(kind):
    reference = fixture(kind)
    set_limits(reference["joints"][0], -math.inf, math.inf)
    candidate = copy.deepcopy(reference)
    half_span = math.pi if kind == "PhysicsRevoluteJoint" else 1.0
    set_limits(candidate["joints"][0], -half_span, half_span)
    result = measure(reference, candidate)
    assert result["score_percent"] == pytest.approx(100 / 3)
    assert result["joints"][0]["limit_pattern_mismatches"] == 2


def test_multiturn_interval_cannot_alias_uniform_sweep_samples():
    reference = fixture()
    set_limits(reference["joints"][0], 0.0, 2 * math.pi)
    candidate = copy.deepcopy(reference)
    set_limits(candidate["joints"][0], 0.0, 34 * math.pi)
    result = measure(reference, candidate)
    assert result["score_percent"] < 1
    joint = result["joints"][0]
    assert joint["motion_rms_m"] > 0.1
    assert joint["unwrapped_extent_error"] == pytest.approx(32 * math.pi)
    assert joint["unwrapped_extent_credit"] < 0.001
    assert not np.allclose(result["settings"]["sweep_fractions"], np.linspace(0, 1, 17))


@pytest.mark.parametrize(
    "change", ["missing", "blocked", "invalid", "direction", "contradictory_state"]
)
def test_missing_or_invalid_bound_evidence_stays_unknown(change):
    reference = fixture()
    set_limits(reference["joints"][0], -math.inf, math.inf)
    limits = reference["joints"][0]["limits"]
    if change == "missing":
        limits["bounds"].pop("physics:lowerLimit")
    elif change == "blocked":
        limits["bounds"]["physics:lowerLimit"] = {"state": "unknown", "value": None}
    elif change == "invalid":
        limits["bounds"]["physics:lowerLimit"]["state"] = "invalid"
    elif change == "direction":
        limits["bounds"]["physics:lowerLimit"]["direction"] = "positive"
    else:
        limits["state"] = "bounded"
    assert measure(reference, copy.deepcopy(reference))["status"] == "unknown"


def test_actual_usd_unbounded_schema_defaults_are_measured(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics
    from simready_bench.fidelity import compare
    from simready_bench.usd_geometry import observe
    from test_fidelity_usd import write

    source = tmp_path / "source.usda"
    write(source)
    stage = Usd.Stage.Open(str(source))
    hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/AnyJointName"))
    hinge.GetLowerLimitAttr().Clear()
    hinge.GetUpperLimitAttr().Clear()
    stage.GetRootLayer().Save()
    reference = observe(source)
    assert reference["joints"][0]["limits"]["state"] == "unbounded"
    result = compare(reference, observe(source), settings({"samples": 128}))
    assert result["components"]["kinematics"]["score_percent"] == 100


def test_one_missing_infinite_bound_retains_partial_contract_credit():
    reference = fixture()
    set_limits(reference["joints"][0], -math.inf, 1.0)
    candidate = copy.deepcopy(reference)
    set_limits(candidate["joints"][0], 1.0 - 2 * math.pi, 1.0)
    result = measure(reference, candidate)
    assert result["score_percent"] == pytest.approx(50)
    assert result["joints"][0]["limit_pattern_mismatches"] == 1


def test_extra_joint_without_candidate_geometry_is_not_inapplicable():
    from simready_bench.fidelity import compare

    reference = fixture()
    reference["joints"] = []
    reference["materials"] = [{}, {}]
    reference["eligibility"] = {
        "visible_prims": 2,
        "supported_prims": 2,
        "unsupported": [],
    }
    candidate = fixture()
    candidate["triangles"] = []
    candidate["triangle_bodies"] = []
    candidate["materials"] = []
    candidate["eligibility"] = {
        "visible_prims": 0,
        "supported_prims": 0,
        "unsupported": [],
    }
    result = compare(reference, candidate, settings({"samples": 128}))["components"][
        "kinematics"
    ]
    assert result["status"] == "measured" and result["score_percent"] == 0
    assert result["candidate_joint_count"] == result["extra_joint_count"] == 1
