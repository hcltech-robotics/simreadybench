"""Three independent scores and explicit measurement uncertainty."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

DIMENSIONS = ("formal", "visual", "kinematic")
REPORT_SCHEMA = "simready-evaluation-v1"
VISUAL_SCHEMA = "nv100-visual-surface-render-v1"


def _number(value, low=0, high=100):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def component(value=None, *, coverage=0, lower=0, upper=100, reason=None, **details):
    """A known zero is different from a measurement that could not be made."""
    if value is not None and not _number(value):
        raise ValueError("Finite percentage required")
    if not _number(coverage, 0, 1):
        raise ValueError("Finite coverage fraction required")
    if value is not None:
        lower = upper = value
    if not _number(lower) or not _number(upper) or lower > upper:
        raise ValueError("Ordered finite percentage bounds required")
    return {
        **details,
        "status": "measured" if value is not None else "unknown",
        "score_percent": value,
        "coverage_fraction": coverage,
        "lower_bound_percent": lower,
        "upper_bound_percent": upper,
        **({"reason": reason} if reason else {}),
    }


def _bounds(part):
    value = part.get("score_percent")
    if _number(value) and part.get("status") != "unknown":
        return value, value
    return part.get("lower_bound_percent", 0), part.get("upper_bound_percent", 100)


def visual_score(geometry, appearance):
    """The original geometric mean, with real geometry and four-view appearance."""
    factors = {"geometry": geometry, "rendered_appearance": appearance}
    known = all(_number(p.get("score_percent")) and p.get("status") != "unknown"
                and p.get("coverage_fraction", 1) == 1 for p in factors.values())
    bounds = [_bounds(part) for part in factors.values()]
    lower = math.sqrt(bounds[0][0] * bounds[1][0])
    upper = math.sqrt(bounds[0][1] * bounds[1][1])
    exact_zero = any(p.get("status") != "unknown" and p.get("score_percent") == 0
                     and p.get("coverage_fraction", 1) == 1 for p in factors.values())
    value = (0 if exact_zero else
             math.sqrt(geometry["score_percent"] * appearance["score_percent"]) if known else None)
    return component(value, coverage=sum(p.get("coverage_fraction", 0) for p in factors.values()) / 2,
                     lower=lower, upper=upper, schema=VISUAL_SCHEMA, factors=factors)


def reward_record(report):
    reward = {"contract_version": 1,
              "candidate_invalid": int(report.get("candidate_invalid", False))}
    for name in DIMENSIONS:
        part = report["scores"][name]
        value = part.get("score_percent")
        known = _number(value) and part.get("status") != "unknown"
        coverage = part.get("coverage_fraction", 0)
        low, high = _bounds(part)
        if not _number(coverage, 0, 1) or not _number(low) or not _number(high) or low > high:
            raise ValueError("Invalid score coverage or bounds")
        reward.update({name + "_known_credit": value / 100 if known else 0,
                       name + "_known": int(known),
                       name + "_measurement_coverage": coverage,
                       name + "_lower": low / 100,
                       name + "_upper": high / 100})
    reward["evaluation_complete"] = int(all(reward[n + "_known"] for n in DIMENSIONS))
    return reward


def write_outputs(report, output):
    """Replace both outputs atomically per file; no stale scalar reward survives."""
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != output:
        raise ValueError("Regular verifier output directory required")
    (output / "reward.txt").unlink(missing_ok=True)
    reward = reward_record(report)
    for name, value in (("report.json", report), ("reward.json", reward)):
        encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", dir=output, prefix=".output-", delete=False) as stream:
            stream.write(encoded)
            stream.flush()
            # NamedTemporaryFile starts at 0600. The separate verifier may run
            # as container root while Harbor collects these public outputs as
            # the host user; publish completed artifacts with readable mode.
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
            temporary = stream.name
        os.replace(temporary, output / name)
    return reward
