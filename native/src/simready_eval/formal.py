"""Neutral NVIDIA Robotics-Prop 4.0.0 mandatory and optional scoring.

The fixed public inventories and source identities are extracted from the audited
public evaluator. SDK availability is independent of candidate validity.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import math

from simready_eval.package import inspect_package
from simready_eval import validator

POLICY_ID = "NVIDIA-Robotics-Prop-4.0.0-mandatory-95-plus-5-v1"
SCHEMA = "simready-validator-score-v1"
PROFILE_FILE = Path(__file__).with_name("profiles") / "nvidia_profile.json"


def digest(v):
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def pinned_profile():
    return json.loads(PROFILE_FILE.read_text())


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _execution_errors(nvidia):
    value = nvidia.get("infrastructure_errors", [])
    return value if isinstance(value, list) else [None]


def scoring_policy(*, excluded_rules=None):
    p = pinned_profile()
    codes = [r["code"] for r in p["mandatory_requirements"]]
    if excluded_rules is None:
        excluded_rules = []
    if excluded_rules:
        raise ValueError(
            "Every pinned mandatory NVIDIA requirement must be scored; filtered profiles are not supported"
        )
    if (
        not isinstance(excluded_rules, list)
        or len(set(excluded_rules)) != len(excluded_rules)
        or (not set(excluded_rules) <= set(codes))
    ):
        raise ValueError("Explicit unique mandatory exclusions required")
    return {
        "schema": "simready-validator-scoring-policy-v1",
        "policy_id": POLICY_ID,
        "profile": {"id": p["profile_id"], "version": p["profile_version"]},
        "profile_inventory_sha256": digest(p),
        "scoring_unit": "one_mandatory_requirement_ID",
        "base_credit_percent": 95,
        "all_pass_bonus_percent": 5,
        "applicable_requirement_ids": [c for c in codes if c not in excluded_rules],
        "report_only_mandatory_ids": sorted(excluded_rules),
        "optional_requirements": "report_only",
        "applicability": "fixed_profile_and_task_before_candidate",
        "unknown": "null_score_fixed_denominator",
        "scope": "filtered_mandatory_profile"
        if excluded_rules
        else "all_mandatory_profile",
        "process_shaping_percent": 0,
        "stop_bonus_percent": 0,
        "early_stop_penalty_percent": 0,
    }


def validate_policy(p):
    if not isinstance(p, dict) or p != scoring_policy(
        excluded_rules=p.get("report_only_mandatory_ids")
    ):
        raise ValueError("Exact pinned versioned mandatory scoring policy required")
    if not p["applicable_requirement_ids"]:
        raise ValueError("Nonempty fixed mandatory scoring scope required")
    return deepcopy(p)


def pinned_optional_profile():
    return json.loads(
        (Path(__file__).with_name("profiles") / "nvidia_optional_profile.json").read_text()
    )


def _recorded_optional_requirements(nvidia, roster):
    execution = nvidia["rule_execution"]
    if "optional_requirements" in execution:
        return execution["optional_requirements"]
    from simready_eval.validator import requirement_outcomes

    buckets = execution.get("buckets")
    if not isinstance(buckets, list) or len(buckets) > 100:
        raise ValueError("Recorded SDK runtime buckets are required")
    for bucket in buckets:
        if not isinstance(bucket, dict) or any(
            not isinstance(bucket.get(key), list)
            for key in ("declared_requirement_codes", "enabled_rules", "method_calls")
        ):
            raise ValueError("Invalid recorded SDK runtime bucket")
        if any(
            not isinstance(value, str)
            for key in ("declared_requirement_codes", "enabled_rules")
            for value in bucket[key]
        ):
            raise ValueError("Invalid recorded SDK rule identities")
        if any(
            not isinstance(call, dict)
            or not isinstance(call.get("rule"), str)
            or not isinstance(call.get("method"), str)
            or type(call.get("count")) is not int
            or call["count"] < 0
            for call in bucket["method_calls"]
        ):
            raise ValueError("Invalid recorded SDK check calls")
    issues = [
        issue
        for key in ("issues", "soft_issues", "warnings", "infos")
        for issue in nvidia.get(key, [])
    ]
    if any(not isinstance(issue, dict) for issue in issues):
        raise ValueError("Invalid recorded NVIDIA findings")
    issues = [
        {
            "rule_id": issue.get("rule_id"),
            "rule": issue.get("rule"),
            "severity": issue.get("severity"),
        }
        for issue in issues
    ]
    return requirement_outcomes(roster, buckets, issues)


def score_optional_validation(nvidia, *, candidate_invalid=False):
    """Score optional requirement IDs independently from the mandatory grade."""
    mandatory = pinned_profile()
    profile = pinned_optional_profile()
    roster = profile["optional_requirements"]
    statuses = {row["code"]: "unknown" for row in roster}
    reason = None
    if candidate_invalid:
        statuses = {code: "invalid" for code in statuses}
        reason = "invalid_candidate_package"
    elif not isinstance(nvidia, dict):
        reason = "validator_unavailable"
    else:
        inventory = nvidia.get("inventory")
        inventory = inventory if isinstance(inventory, dict) else {}
        execution = nvidia.get("rule_execution")
        execution = execution if isinstance(execution, dict) else {}
        identity_fields = (
            "profile_id",
            "profile_version",
            "profile_manifest_sha256",
            "tier_manifest_hashes",
            "versions",
        )
        if profile["mandatory_inventory_sha256"] != digest(mandatory) or any(
            inventory.get(key) != mandatory[key] for key in identity_fields
        ):
            reason = "pinned_inventory_mismatch"
        elif "optional_requirements" in inventory and any(
            inventory.get(key) != profile[key]
            for key in (
                "optional_requirements",
                "optional_features",
                "optional_requirement_count",
            )
        ):
            reason = "optional_inventory_mismatch"
        elif (
            nvidia.get("engine_dispatch_completed") is not True
            or execution.get("completed") is not True
        ):
            reason = "validator_execution_incomplete"
        elif any(
            _mapping(execution.get("sdk_source_sha256")).get(key) != value
            for key, value in mandatory["SDK_dispatch_source_sha256"].items()
        ):
            reason = "validator_implementation_identity_mismatch"
        elif any(
            not isinstance(error, dict)
            or error.get("code") != "validator_rule_execution_error"
            for error in _execution_errors(nvidia)
        ):
            reason = "validator_infrastructure_error"
        else:
            try:
                rows = _recorded_optional_requirements(nvidia, roster)
                if not isinstance(rows, list) or len(rows) > len(roster):
                    raise ValueError("Invalid optional requirement execution")
                indexed = {}
                for row in rows:
                    if (
                        not isinstance(row, dict)
                        or not isinstance(row.get("code"), str)
                        or row.get("code") not in statuses
                        or row["code"] in indexed
                    ):
                        raise ValueError("Invalid optional requirement execution")
                    indexed[row["code"]] = row
                for requirement in roster:
                    code = requirement["code"]
                    row = indexed.get(code, {})
                    module = requirement["rule"].rsplit(".", 1)[0]
                    if _mapping(execution.get("sdk_source_sha256")).get(
                        module
                    ) != profile["rule_module_sha256"][module]:
                        reason = "validator_implementation_identity_mismatch"
                        continue
                    calls = row.get("method_calls")
                    if (
                        any(
                            row.get(key) != requirement[key]
                            for key in ("version", "rule")
                        )
                        or row.get("rule_enabled") is not True
                        or row.get("rule_invoked") is not True
                        or not isinstance(calls, dict)
                        or any(
                            not isinstance(key, str)
                            or type(value) is not int
                            or value < 0
                            for key, value in calls.items()
                        )
                        or not any(value > 0 for value in calls.values())
                    ):
                        continue
                    disposition = row.get("disposition")
                    if disposition == "failed":
                        statuses[code] = "fail"
                    elif disposition in ("no_failure_reported", "warning_reported"):
                        statuses[code] = "pass"
            except (ValueError, KeyError, TypeError):
                statuses = {code: "unknown" for code in statuses}
                reason = "invalid_optional_requirement_execution"
    count = len(roster)
    passed = sum(status == "pass" for status in statuses.values())
    failed = sum(status in ("fail", "invalid") for status in statuses.values())
    unknown = count - passed - failed
    return {
        "schema": "simready-validator-optional-score-v1",
        "profile": {"id": profile["profile_id"], "version": profile["profile_version"]},
        "profile_inventory_sha256": digest(profile),
        "scope": "all_optional_profile",
        "status": "invalid_candidate"
        if candidate_invalid
        else "unknown"
        if unknown
        else "scored",
        "score_percent": None if unknown else 100 * passed / count,
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "measured_count": passed + failed,
        "applicable_count": count,
        "profile_optional_count": count,
        "coverage_fraction": (passed + failed) / count,
        "lower_bound_percent": 100 * passed / count,
        "upper_bound_percent": 100 * (passed + unknown) / count,
        "unknown_reason": (reason or "optional_requirement_not_assessed")
        if unknown
        else None,
        "checks": [
            {
                "requirement_id": row["code"],
                "requirement_version": row["version"],
                "category": category(row["code"]),
                "status": statuses[row["code"]],
                "mandatory": False,
            }
            for row in roster
        ],
    }


def validate_optional_score(score):
    """Validate the optional denominator, outcomes and partial measurement bounds."""
    profile = pinned_optional_profile()
    roster = profile["optional_requirements"]
    rows = score.get("checks") if isinstance(score, dict) else None
    if (
        not isinstance(rows, list)
        or len(rows) != len(roster)
        or any(not isinstance(row, dict) for row in rows)
        or any(
            row.get("requirement_id") != requirement["code"]
            or row.get("requirement_version") != requirement["version"]
            or row.get("mandatory") is not False
            or row.get("status") not in ("pass", "fail", "unknown", "invalid")
            for row, requirement in zip(rows, roster)
        )
    ):
        raise ValueError("Exact pinned optional requirement outcomes are required")
    count = len(roster)
    passed = sum(row["status"] == "pass" for row in rows)
    failed = sum(row["status"] in ("fail", "invalid") for row in rows)
    unknown = count - passed - failed
    counts = {
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "measured_count": passed + failed,
        "applicable_count": count,
        "profile_optional_count": count,
    }
    if any(
        type(score.get(key)) is not int or score[key] != value
        for key, value in counts.items()
    ):
        raise ValueError("Optional requirement accounting is inconsistent")
    expected_status = "unknown" if unknown else "scored"
    invalid = sum(row["status"] == "invalid" for row in rows)
    if invalid and (invalid != count or score.get("status") != "invalid_candidate"):
        raise ValueError("Invalid optional outcomes require a wholly invalid candidate")
    if score.get("status") == "invalid_candidate" and (unknown == count or invalid == count):
        expected_status = "invalid_candidate"
    if (
        score.get("schema") != "simready-validator-optional-score-v1"
        or score.get("profile")
        != {"id": profile["profile_id"], "version": profile["profile_version"]}
        or score.get("profile_inventory_sha256") != digest(profile)
        or score.get("scope") != "all_optional_profile"
        or score.get("status") != expected_status
    ):
        raise ValueError("Pinned optional scoring scope is required")
    expected = {
        "score_percent": None if unknown else 100 * passed / count,
        "coverage_fraction": (passed + failed) / count,
        "lower_bound_percent": 100 * passed / count,
        "upper_bound_percent": 100 * (passed + unknown) / count,
    }
    for key, value in expected.items():
        actual = score.get(key)
        if value is None:
            valid = actual is None
        else:
            valid = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and math.isclose(actual, value, abs_tol=1e-10)
            )
        if not valid:
            raise ValueError("Optional score or measurement bounds are inconsistent")
    return score


def category(code):
    suffix = code.split(".", 3)[-1]
    if suffix.startswith(("RB.", "RB.COL.")):
        return "rigid_body_collision"
    if suffix.startswith("VG."):
        return "geometry_mesh"
    if suffix.startswith("VM."):
        return "materials_textures"
    if suffix.startswith("HI."):
        return "hierarchy"
    if suffix.startswith(("NP.", "AA.")):
        return "asset_paths_and_naming"
    if suffix.startswith("UN."):
        return "units"
    if suffix.startswith("SL."):
        return "semantics"
    if suffix.startswith("JT."):
        return "joints_articulations"
    if suffix.startswith(("NVM.", "PMT.")):
        return "nonvisual_materials"
    if suffix.startswith(("PHYSX.", "NEWTON.", "MUJOCO.", "RV.")):
        return "runtime_physics"
    if suffix.startswith("DISP."):
        return "display_colour"
    if suffix.startswith("GSP."):
        return "grasp_physics"
    if suffix.startswith("ISA."):
        return "runtime_composition"
    return "thumbnail_metadata"


def score_validation(nvidia, policy=None, *, candidate_invalid=False):
    """Pure projection of genuine SDK requirement outcomes; no candidate mutation."""
    policy = validate_policy(scoring_policy() if policy is None else policy)
    p = pinned_profile()
    roster = p["mandatory_requirements"]
    active = set(policy["applicable_requirement_ids"])
    statuses = {r["code"]: "unknown" for r in roster}
    reason = None
    official = None
    if candidate_invalid:
        statuses = {c: "invalid" for c in statuses}
        reason = "invalid_candidate_package"
    elif not isinstance(nvidia, dict):
        reason = "validator_unavailable"
    else:
        inv = nvidia.get("inventory")
        inv = inv if isinstance(inv, dict) else {}
        execution = nvidia.get("rule_execution")
        execution = execution if isinstance(execution, dict) else {}
        fields = (
            "profile_id",
            "profile_version",
            "mandatory_features",
            "mandatory_requirements",
            "mandatory_requirement_count",
            "profile_manifest_sha256",
            "tier_manifest_hashes",
            "versions",
        )
        if any((inv.get(k) != p[k] for k in fields)):
            reason = "pinned_inventory_mismatch"
        elif (
            nvidia.get("engine_dispatch_completed") is not True
            or execution.get("completed") is not True
        ):
            reason = "validator_execution_incomplete"
        elif any(
            (
                    _mapping(execution.get("sdk_source_sha256")).get(k) != v
                for k, v in {
                    **p["SDK_dispatch_source_sha256"],
                    **p["rule_module_sha256"],
                }.items()
            )
        ):
            reason = "validator_implementation_identity_mismatch"
        elif any(
            (
                not isinstance(e, dict)
                or e.get("code") != "validator_rule_execution_error"
                    for e in _execution_errors(nvidia)
            )
        ):
            reason = "validator_infrastructure_error"
        else:
            raw = execution.get("requirements", [])
            indexed = {}
            if not isinstance(raw, list) or len(raw) > 10000:
                reason = "invalid_requirement_execution"
            else:
                for r in raw:
                    if (
                        not isinstance(r, dict)
                            or not isinstance(r.get("code"), str)
                        or r.get("code") not in statuses
                        or r["code"] in indexed
                    ):
                        reason = "invalid_requirement_execution"
                        break
                    indexed[r["code"]] = r
                if reason is None:
                    for requirement in roster:
                        c = requirement["code"]
                        r = indexed.get(c, {})
                        if (
                            any(
                                (
                                    r.get(k) != requirement[k]
                                    for k in ("version", "rule")
                                )
                            )
                            or r.get("rule_invoked") is not True
                        ):
                            continue
                        calls = r.get("method_calls")
                        if (
                            r.get("rule_enabled") is not True
                            or not isinstance(calls, dict)
                            or any(
                                not isinstance(key, str)
                                or type(value) is not int
                                or value < 0
                                for key, value in calls.items()
                            )
                            or (
                                not any(
                                    (type(v) is int and v > 0 for v in calls.values())
                                )
                            )
                        ):
                            continue
                        disposition = r.get("disposition")
                        if disposition == "failed":
                            statuses[c] = "fail"
                        elif disposition in ("no_failure_reported", "warning_reported"):
                            statuses[c] = "pass"
                    official = (
                        nvidia.get("passed") is True
                        if not nvidia.get("infrastructure_errors")
                        and all((v in ("pass", "fail") for v in statuses.values()))
                        else None
                    )
                    if official is True and any(
                        (v != "pass" for v in statuses.values())
                    ):
                        statuses = {c: "unknown" for c in statuses}
                        official = None
                        reason = "inconsistent_official_result"
    checked = [statuses[c] for c in policy["applicable_requirement_ids"]]
    count = len(checked)
    passed = checked.count("pass")
    failed = checked.count("fail") + checked.count("invalid")
    unknown = checked.count("unknown")
    all_pass = not unknown and passed == count
    base = 95 * passed / count
    bonus = None if unknown else 5 if all_pass else 0
    score = None if unknown else base + bonus
    checks = []
    categories = {}
    for r in roster:
        c = r["code"]
        status = statuses[c]
        scoring = c in active
        group = category(c)
        row = {
            "requirement_id": c,
            "requirement_version": r["version"],
            "category": group,
            "status": status,
            "mandatory": True,
            "scoring": scoring,
            "weight": 1 if scoring else 0,
            "base_credit_percent": 95 / count
            if scoring and status == "pass"
            else None
            if scoring and status == "unknown"
            else 0,
            "status_basis": "pinned_NVIDIA_rule_execution_requirement_outcome",
            "corrective_action": None,
        }
        checks.append(row)
        if scoring:
            summary = categories.setdefault(
                group, {"applicable": 0, "passed": 0, "failed": 0, "unknown": 0}
            )
            summary["applicable"] += 1
            summary["passed"] += status == "pass"
            summary["failed"] += status in ("fail", "invalid")
            summary["unknown"] += status == "unknown"
    return {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "policy_sha256": digest(policy),
        "scope": policy["scope"],
        "status": "invalid_candidate"
        if candidate_invalid
        else "unknown"
        if unknown
        else "scored",
        "score_percent": score,
        "score_fraction": None if score is None else score / 100,
        "base_credit_percent": None if unknown else base,
        "completion_bonus": bonus,
        "score_lower_bound": base,
        "score_upper_bound": 95 * (passed + unknown) / count + (5 if not failed else 0),
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "applicable_count": count,
        "profile_mandatory_count": len(roster),
        "profile_coverage_fraction": count / len(roster),
        "all_scored_mandatory_pass": all_pass,
        "official_conformance": {
            "profile": policy["profile"],
            "passed": official,
            "filtered": False,
        },
        "unknown_reason": reason if unknown else None,
        "checks": checks,
        "categories": categories,
        "optional_requirements": "report_only",
        "optional_score": score_optional_validation(
            nvidia, candidate_invalid=candidate_invalid
        ),
    }




def evaluate(package_root, entrypoint="asset.usd"):
    """Inspect a candidate package, then invoke the genuine pinned NVIDIA SDK.

    Call from a bounded, isolated worker. The caller owns freezing submitted
    bytes and verifying their hashes after evaluation. This function requires no
    episode database, host service, reference fixture, or network connection.
    """
    preflight = inspect_package(package_root, entrypoint=entrypoint)
    invalid = preflight.get("proven_invalid") is True or (
        preflight.get("package_ok") is False
        and not preflight.get("infrastructure_errors")
    )
    raw = None
    errors = list(preflight.get("infrastructure_errors", []))
    if not invalid and not errors:
        raw = validator.validate_asset(Path(package_root) / entrypoint)
        errors.extend(raw.get("infrastructure_errors", []))
        if raw.get("outcome") == "invalid_submission" and not errors:
            invalid = True
    score = score_validation(raw, candidate_invalid=invalid)
    return {
        "schema_version": "simready-formal-evaluation-v1",
        "outcome": "invalid_submission" if invalid else "infrastructure_error"
        if score["status"] == "unknown" else "complete"
        if score["all_scored_mandatory_pass"] else "incomplete",
        "score_percent": score["score_percent"],
        "reward": score["score_percent"],
        "simready": score,
        "optional_score": score["optional_score"],
        "nvidia_validator_report": raw,
        "preflight": preflight,
        "infrastructure_errors": errors,
        "coverage": {
            "measured": score["passed_count"] + score["failed_count"],
            "total": score["applicable_count"],
            "execution_fraction": 1 - score["unknown_count"] / score["applicable_count"],
            "profile_fraction": score["profile_coverage_fraction"],
        },
    }
