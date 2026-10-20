"""Independent formal, visual and kinematic reconstruction scores."""

from __future__ import annotations

import math

VERSION = "simreadybench-composites-v1"
NAMES = ("formal", "visual", "kinematic")
SUFFIXES = (
    "percent",
    "status",
    "earned",
    "eligible",
    "measured",
    "coverage",
    "lower_bound_percent",
    "upper_bound_percent",
)
APPEARANCE_FIELDS = tuple("visual_appearance_" + suffix for suffix in SUFFIXES)
KINEMATIC_FIELDS = tuple(
    "kinematic_" + field + "_joint_count"
    for field in ("reference", "candidate", "matched", "missing", "extra")
) + tuple(
    "kinematic_" + name + "_" + suffix
    for name in ("moving_joints", "fixed_constraints")
    for suffix in SUFFIXES
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100


def _appearance(materials):
    features = materials.get("features", {})
    selected = {
        key: value
        for key, value in features.items()
        if key.startswith("visual:") and key != "visual:assignment"
    }
    if isinstance(materials.get("appearance"), dict):
        return {**materials["appearance"], "features": selected}
    eligible = sum(value["eligible"] for value in selected.values())
    earned = sum(value["earned"] for value in selected.values())
    # An authored shader with no supported inputs is unmeasured appearance.
    assigned = features.get("visual:assignment", {}).get("eligible", 0)
    supported_samples = max(
        (value["eligible"] for value in selected.values()), default=0
    )
    unsupported = bool(assigned and supported_samples < assigned)
    return {
        "status": "unknown"
        if unsupported
        else "measured"
        if eligible
        else "not_applicable",
        "score_percent": 100 * earned / eligible
        if eligible and not unsupported
        else None,
        "earned": earned,
        "eligible": eligible,
        "measured": eligible,
        "coverage_fraction": supported_samples / assigned
        if assigned
        else 1
        if eligible
        else 0,
        "coverage_basis": "supported reference appearance fields",
        "features": selected,
    }


def scores(condition):
    """Derive independent composites without altering source measurements."""
    primary = (
        condition.get("simready")
        or (condition.get("evaluation") or {}).get("simready")
        or {}
    )
    formal_value = primary.get("score_percent")
    measured = primary.get("measured_count")
    if measured is None and all(
        type(primary.get(key)) is int for key in ("passed_count", "failed_count")
    ):
        measured = primary["passed_count"] + primary["failed_count"]
    formal = {
        "score_percent": formal_value if _number(formal_value) else None,
        "status": "measured" if _number(formal_value) else "unknown",
        "passed": (primary.get("official_conformance") or {}).get("passed"),
        "earned": primary.get("passed_count"),
        "eligible": primary.get("applicable_count"),
        "measured": measured,
        "coverage_fraction": primary.get(
            "profile_coverage_fraction",
            measured / primary["applicable_count"]
            if measured is not None and primary.get("applicable_count")
            else None,
        ),
        "lower_bound_percent": primary.get("score_lower_bound"),
        "upper_bound_percent": primary.get("score_upper_bound"),
    }
    components = (condition.get("fidelity") or {}).get("components", {})
    shape = components.get("shape", {})
    appearance = _appearance(components.get("materials", {}))
    factors = {"shape": shape, "appearance": appearance}
    active = [
        part for part in factors.values() if part.get("status") != "not_applicable"
    ]
    values = [part.get("score_percent") for part in active]
    known = bool(active) and all(_number(value) for value in values)
    visual_value = (
        100 * math.prod(value / 100 for value in values) ** (1 / len(values))
        if known
        else None
    )
    visual = {
        "score_percent": visual_value,
        "status": "measured" if known else "unknown" if active else "not_applicable",
        "earned": visual_value / 100 * len(active)
        if visual_value is not None
        else None,
        "eligible": len(active),
        "measured": sum(_number(value) for value in values),
        "coverage_fraction": sum(
            part.get("coverage_fraction", int(_number(part.get("score_percent")))) or 0
            for part in active
        )
        / len(active)
        if active
        else 0,
        "aggregation": "geometric_mean",
        "factors": factors,
    }
    for bound, missing in (("lower_bound_percent", 0), ("upper_bound_percent", 100)):
        visual[bound] = (
            100
            * math.prod(
                (
                    part["score_percent"]
                    if _number(part.get("score_percent"))
                    else part[bound]
                    if _number(part.get(bound))
                    else missing
                )
                / 100
                for part in active
            )
            ** (1 / len(active))
            if active
            else None
        )
    kinematic = dict(
        components.get("kinematics")
        or {
            "status": "unknown",
            "score_percent": None,
            "earned": None,
            "eligible": None,
            "measured": 0,
            "coverage_fraction": 0,
        }
    )
    return {
        "schema": VERSION,
        "formal": formal,
        "visual": visual,
        "kinematic": kinematic,
    }


def columns(condition):
    composites = scores(condition)
    fields = (
        ("percent", "score_percent"),
        ("status", "status"),
        ("earned", "earned"),
        ("eligible", "eligible"),
        ("measured", "measured"),
        ("coverage", "coverage_fraction"),
        ("lower_bound_percent", "lower_bound_percent"),
        ("upper_bound_percent", "upper_bound_percent"),
    )
    result = {
        f"{name}_fidelity_{suffix}": composites[name].get(key)
        for name in NAMES
        for suffix, key in fields
    }
    appearance = composites["visual"]["factors"]["appearance"]
    result.update(
        {"visual_appearance_" + suffix: appearance.get(key) for suffix, key in fields}
    )
    kinematic = composites["kinematic"]
    result.update(
        {
            "kinematic_" + field + "_joint_count": kinematic.get(field + "_joint_count")
            for field in ("reference", "candidate", "matched", "missing", "extra")
        }
    )
    result.update(
        {
            "kinematic_" + name + "_" + suffix: kinematic.get("groups", {})
            .get(name, {})
            .get(key)
            for name in ("moving_joints", "fixed_constraints")
            for suffix, key in fields
        }
    )
    return result
