"""Reference-conditioned joint recovery and sampled kinematic correspondence."""

from __future__ import annotations

import itertools
import math

import numpy as np


VERSION = "source-usd-kinematics-v1"
SWEEP_STATES = 17
SWEEP_FRACTIONS = np.sort(
    np.r_[0.0, 1.0, (np.arange(1, SWEEP_STATES - 1) * ((math.sqrt(5) - 1) / 2)) % 1]
)
SUPPORTED_TYPES = (
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
)


def _array(value, shape):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.shape == shape and np.isfinite(result).all() else None


def _interval(joint):
    """Resolve observed finite/infinite limits without inventing missing bounds."""
    if joint.get("type") == "PhysicsFixedJoint":
        return np.array([0.0, 0.0])
    limits = joint.get("limits", {})
    expected_units = "radians" if joint.get("type") == "PhysicsRevoluteJoint" else "m"
    state = limits.get("state")
    if (
        state not in ("bounded", "unbounded", "locked")
        or limits.get("units") != expected_units
    ):
        return None
    bounds = limits.get("bounds", {})
    values = []
    for key, direction in (
        ("physics:lowerLimit", "negative"),
        ("physics:upperLimit", "positive"),
    ):
        bound = bounds.get(key, {})
        value = bound.get("value")
        if bound.get("state") == "unbounded" and bound.get("direction") == direction:
            values.append(-math.inf if direction == "negative" else math.inf)
        elif (
            bound.get("state") == "finite"
            and type(value) in (int, float)
            and math.isfinite(value)
        ):
            values.append(value)
        else:
            return None
    interval = np.asarray(values, dtype=float)
    if state == "locked":
        return (
            np.zeros(2)
            if np.isfinite(interval).all() and values[0] > values[1]
            else None
        )
    if values[0] > values[1] or (state == "unbounded") != bool(
        np.isinf(interval).all()
    ):
        return None
    return interval


def _joint_data(joint):
    anchors = _array(joint.get("anchors"), (2, 3))
    frames = _array(joint.get("frames"), (2, 3, 3))
    interval = _interval(joint)
    if anchors is None or frames is None or interval is None:
        return None
    if any(
        not np.allclose(frame @ frame.T, np.eye(3), atol=1e-6)
        or np.linalg.det(frame) < 0
        for frame in frames
    ):
        return None
    axes = None
    if joint.get("type") != "PhysicsFixedJoint":
        axes = _array(joint.get("axes"), (2, 3))
        if axes is None or np.any(np.linalg.norm(axes, axis=1) < 1e-12):
            return None
        axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    return anchors, frames, axes, interval


def _landmarks(reference, body):
    if body is None:
        return None
    triangles = np.asarray(reference.get("triangles", []), float)
    owners = np.asarray(reference.get("triangle_bodies", []), dtype=object)
    if len(triangles) != len(owners):
        return None
    points = triangles[owners == body].reshape(-1, 3)
    if not len(points) or not np.isfinite(points).all():
        return None
    low, high = points.min(axis=0), points.max(axis=0)
    corners = np.asarray(list(itertools.product(*zip(low, high))))
    return np.vstack((corners, (low + high) / 2))


def _rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def _transform(kind, data, coordinate):
    anchors, frames, axes, _ = data
    rotation = frames[0].T @ frames[1]
    offset = np.zeros(3)
    if kind == "PhysicsRevoluteJoint":
        rotation = _rotation(axes[0], coordinate) @ rotation
    elif kind == "PhysicsPrismaticJoint":
        offset = coordinate * axes[0]
    return rotation, anchors[0] + offset - rotation @ anchors[1]


def _bound_pattern(interval):
    return [
        "negative_unbounded"
        if value == -math.inf
        else "positive_unbounded"
        if value == math.inf
        else "finite"
        for value in interval
    ]


def _canonical_coordinate(kind, data, centre):
    """Choose a coordinate gauge for a fully unbounded motion family."""
    rotation, translation = _transform(kind, data, 0.0)
    axis = data[2][0]
    if kind == "PhysicsPrismaticJoint":
        return -float(np.dot(axis, rotation @ centre + translation - centre))
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    cosine = float(np.trace(rotation) - axis @ rotation @ axis)
    sine = float(np.trace(skew @ rotation))
    return math.atan2(sine, cosine)


def _probe_interval(kind, data, scale, centre):
    bounds = data[3]
    if np.isfinite(bounds).all():
        return bounds
    width = 2 * math.pi if kind == "PhysicsRevoluteJoint" else scale
    if np.isinf(bounds).all():
        coordinate = _canonical_coordinate(kind, data, centre)
        return np.array([coordinate - width / 2, coordinate + width / 2])
    if bounds[0] == -math.inf:
        return np.array([bounds[1] - width, bounds[1]])
    return np.array([bounds[0], bounds[0] + width])


def _motion_error(reference, joint, other, cfg, scale, *, swapped):
    """Compare J0 M(q) J1^-1 on identical reference-body landmarks."""
    target = _joint_data(joint)
    actual = _joint_data(other)
    if target is None or actual is None:
        return None
    ca, cf, cx, bounds = actual
    if swapped:
        ca, cf = ca[::-1], cf[::-1]
        cx = cx[::-1] if cx is not None else None
        bounds = -bounds[::-1]
    if cx is not None and np.dot(target[2][0], cx[0]) < 0:
        cx = -cx
        bounds = -bounds[::-1]
    actual = ca, cf, cx, bounds
    probes = [_landmarks(reference, body) for body in joint["bodies"]]
    if all(points is None for points in probes):
        return None
    centre = np.vstack([points for points in probes if points is not None]).mean(axis=0)
    target_probe = _probe_interval(joint["type"], target, scale, centre)
    actual_probe = _probe_interval(joint["type"], actual, scale, centre)
    errors = []
    for fraction in SWEEP_FRACTIONS:
        rq = float(target_probe[0] + fraction * np.ptp(target_probe))
        cq = float(actual_probe[0] + fraction * np.ptp(actual_probe))
        rr, rt = _transform(joint["type"], target, rq)
        cr, ct = _transform(joint["type"], actual, cq)
        for endpoint, points in enumerate(probes):
            if points is None:
                continue
            if endpoint == 1:
                first = points @ rr.T + rt
                second = points @ cr.T + ct
            else:
                first = (points - rt) @ rr
                second = (points - ct) @ cr
            errors.extend(np.sum((first - second) ** 2, axis=1))
    target_pattern, actual_pattern = _bound_pattern(target[3]), _bound_pattern(bounds)
    pattern_errors = sum(a != b for a, b in zip(target_pattern, actual_pattern))
    extent_error = None
    extent_credit = 1.0
    if np.isfinite(target[3]).all() and np.isfinite(bounds).all():
        extent_error = abs(float(np.ptp(target[3]) - np.ptp(bounds)))
        absolute = (
            math.radians(cfg["range_degrees"])
            if joint["type"] == "PhysicsRevoluteJoint"
            else cfg["range_m"]
        )
        tolerance = absolute + cfg["range_relative"] * float(np.ptp(target[3]))
        extent_credit = 1 / (1 + (extent_error / tolerance) ** 2)
    return {
        "motion_rms_m": float(math.sqrt(np.mean(errors))),
        "reference_probe_interval": target_probe.tolist(),
        "candidate_probe_interval": actual_probe.tolist(),
        "probe_units": "radians" if joint["type"] == "PhysicsRevoluteJoint" else "m",
        "reference_limit_pattern": target_pattern,
        "candidate_limit_pattern": actual_pattern,
        "limit_pattern_mismatches": pattern_errors,
        "limit_pattern_credit": 1 / (1 + pattern_errors),
        "unwrapped_extent_error": extent_error,
        "unwrapped_extent_credit": extent_credit,
    }


def _flags_match(reference, candidate, joint, other, mapping):
    if joint.get("enabled") is not None and other.get("enabled") != joint["enabled"]:
        return False
    for body in joint["bodies"]:
        if body is None:
            continue
        target = reference.get("bodies", {}).get(body, {})
        actual = candidate.get("bodies", {}).get(mapping.get(body), {})
        for key in ("enabled", "kinematic"):
            if target.get(key) is not None and actual.get(key) != target[key]:
                return False
    return True


def _connections(joint, other, mapping):
    if any(body is not None and body not in mapping for body in joint["bodies"]):
        return None
    expected = [
        mapping.get(body) if body is not None else None for body in joint["bodies"]
    ]
    if expected == other.get("bodies"):
        return False
    if expected == list(reversed(other.get("bodies", []))):
        return True
    return None


def _group(joints, candidates, rows, fixed):
    selected = [
        row
        for joint, row in zip(joints, rows)
        if (joint.get("type") == "PhysicsFixedJoint") == fixed
    ]
    count = sum(
        (joint.get("type") == "PhysicsFixedJoint") == fixed for joint in candidates
    )
    denominator = len(selected) + count
    known = all(row["earned"] is not None for row in selected)
    lower = (
        2 * sum(row["earned"] or 0 for row in selected) / denominator
        if denominator
        else 0
    )
    possible_credit = sum(
        int(row["candidate_joint"] is not None)
        if row["earned"] is None
        else row["earned"]
        for row in selected
    )
    upper = 2 * possible_credit / denominator if denominator else 0
    unmeasured = sum(
        1 + int(row["candidate_joint"] is not None)
        for row in selected
        if row["earned"] is None
    )
    return {
        "status": "not_applicable"
        if not denominator
        else "measured"
        if known
        else "unknown",
        "score_percent": 100 * lower if denominator and known else None,
        "earned": 2 * sum(row["earned"] for row in selected) if known else None,
        "eligible": denominator,
        "measured": denominator - unmeasured,
        "coverage_fraction": (denominator - unmeasured) / denominator
        if denominator
        else 0,
        "lower_bound_percent": 100 * lower if denominator else None,
        "upper_bound_percent": 100 * upper if denominator else None,
        "reference_joint_count": len(selected),
        "candidate_joint_count": count,
    }


def compare_kinematics(
    reference, candidate, mapping, cfg, scale, *, ambiguous_bodies=()
):
    """Measure joint recovery with continuous, geometry-conditioned motion credit."""
    from simready_eval.correspondence import assignment

    candidate = candidate or {}
    joints = reference.get("joints", [])
    candidates = candidate.get("joints", [])
    band = cfg["joint_position_m"] + cfg["joint_position_relative"] * scale
    rows, options = [], []
    for index, joint in enumerate(joints):
        unknown = (
            joint.get("type") not in SUPPORTED_TYPES
            or _joint_data(joint) is None
            or any(
                body in ambiguous_bodies
                for body in joint.get("bodies", [])
                if body is not None
            )
            or not any(
                _landmarks(reference, body) is not None
                for body in joint.get("bodies", [])
            )
        )
        row = {
            "reference_joint": index,
            "reference_path": joint.get("path"),
            "type": joint.get("type"),
            "candidate_joint": None,
            "status": "unknown" if unknown else "measured",
            "earned": None if unknown else 0.0,
            "eligible": 1,
            "measured": 0 if unknown else 1,
            "coverage_fraction": 0 if unknown else 1,
            "motion_rms_m": None,
            "reason": "unsupported_reference_or_ambiguous_correspondence"
            if unknown
            else "missing_joint",
        }
        rows.append(row)
        choices = []
        for other in candidates:
            swapped = _connections(joint, other, mapping)
            connected = swapped is not None and joint.get("type") == other.get("type")
            quality = 0.0
            error = None
            measurement = {}
            reason = "wrong_type_or_connectivity"
            if connected:
                if unknown:
                    reason = row["reason"]
                elif not _flags_match(reference, candidate, joint, other, mapping):
                    reason = "joint_or_body_state_mismatch"
                elif _joint_data(other) is None:
                    reason = "invalid_candidate_joint_parameters"
                else:
                    measurement = _motion_error(
                        reference, joint, other, cfg, scale, swapped=swapped
                    )
                    error = measurement["motion_rms_m"]
                    motion_credit = 1 / (1 + (error / band) ** 2)
                    measurement["motion_credit"] = motion_credit
                    quality = (
                        motion_credit
                        * measurement["unwrapped_extent_credit"]
                        * measurement["limit_pattern_credit"]
                    )
                    reason = "measured_motion"
            choices.append(
                {
                    "connected": connected,
                    "quality": quality,
                    "error": error,
                    "reason": reason,
                    "swapped": swapped,
                    "measurement": measurement,
                }
            )
        options.append(choices)
    matches = (
        assignment(
            [
                [1 - item["quality"] if item["connected"] else 10 for item in choices]
                for choices in options
            ]
        )
        if joints and candidates
        else {}
    )
    recovered = set()
    for index, other_index in matches.items():
        choice = options[index][other_index]
        if not choice["connected"]:
            continue
        recovered.add(other_index)
        row = rows[index]
        row.update(
            candidate_joint=other_index,
            candidate_path=candidates[other_index].get("path"),
            swapped_endpoints=choice["swapped"],
        )
        if row["status"] != "unknown":
            row.update(
                earned=choice["quality"],
                motion_rms_m=choice["error"],
                reason=choice["reason"],
            )
            row.update(choice["measurement"])
    for row in rows:
        row["score_percent"] = (
            100 * row["earned"] if row["earned"] is not None else None
        )
    groups = {
        "moving_joints": _group(joints, candidates, rows, False),
        "fixed_constraints": _group(joints, candidates, rows, True),
    }
    active = [group for group in groups.values() if group["eligible"]]
    known = all(group["score_percent"] is not None for group in active)
    score = (
        100
        * math.prod(group["score_percent"] / 100 for group in active)
        ** (1 / len(active))
        if active and known
        else None
    )
    bounds = {
        field: 100
        * math.prod(group[field] / 100 for group in active) ** (1 / len(active))
        if active
        else None
        for field in ("lower_bound_percent", "upper_bound_percent")
    }
    return {
        "schema": VERSION,
        "status": "not_applicable"
        if not active
        else "measured"
        if known
        else "unknown",
        "score_percent": score,
        **bounds,
        "earned": score / 100 * len(active)
        if score is not None
        else 0
        if not active
        else None,
        "eligible": len(active),
        "measured": sum(group["score_percent"] is not None for group in active),
        "coverage_fraction": sum(group["measured"] for group in active)
        / sum(group["eligible"] for group in active)
        if active
        else 0,
        "reference_joint_count": len(joints),
        "candidate_joint_count": len(candidates),
        "matched_joint_count": len(recovered),
        "missing_joint_count": len(joints) - len(recovered),
        "extra_joint_count": len(candidates) - len(recovered),
        "groups": groups,
        "joints": rows,
        "aggregation": "geometric mean of moving-joint and fixed-constraint recovery, each 2*sum(quality)/(reference_count+candidate_count)",
        "settings": {
            "sweep_states": SWEEP_STATES,
            "sweep_fractions": SWEEP_FRACTIONS.tolist(),
            "range_absolute_degrees": cfg["range_degrees"],
            "range_absolute_m": cfg["range_m"],
            "range_relative": cfg["range_relative"],
            "distance_band_m": band,
            "reference_diagonal_m": scale,
        },
        "supported_joint_types": list(SUPPORTED_TYPES),
        "range_scope": "observed finite, half-bounded, unbounded and locked intervals",
        "infinite_range_probes": "one full revolution or one reference diagonal, centred at a canonical coordinate for fully unbounded motion or anchored at the finite endpoint for half-bounded motion; authored limits remain unchanged",
        "limit_pattern_treatment": "1/(1+number of differing bound endpoint states after coordinate normalisation)",
        "finite_extent_treatment": "1/(1+(unwrapped interval extent error/(absolute range tolerance+relative tolerance*reference extent))^2)",
        "joint_quality": "motion_credit*unwrapped_extent_credit*limit_pattern_credit, with motion_credit=1/(1+(motion_rms_m/distance_band_m)^2)",
        "measurement": "sampled joint constraint transforms on common reference-body bounding-box landmarks",
    }
