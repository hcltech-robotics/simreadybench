"""Independent optional scores retain requirement identity and unknown outcomes."""

from copy import deepcopy
import hashlib
import importlib
import inspect
from pathlib import Path

import pytest

from simready_bench.engine.scoring import (
    pinned_optional_profile,
    pinned_profile,
    score_optional_validation,
    score_validation,
    validate_optional_score,
)
from simready_bench.engine.verifier import digest, scoring_policy
from test_validator_reward_v1 import official_report


def optional_report(*, failed=()):
    raw = official_report()
    profile = pinned_optional_profile()
    raw["inventory"].update(
        {
            key: deepcopy(profile[key])
            for key in (
                "optional_features",
                "optional_requirements",
                "optional_requirement_count",
            )
        }
    )
    raw["rule_execution"]["sdk_source_sha256"].update(profile["rule_module_sha256"])
    raw["rule_execution"]["optional_requirements"] = [
        {
            **row,
            "rule_enabled": True,
            "rule_invoked": True,
            "method_calls": {"CheckPrim": 1},
            "disposition": "failed" if row["code"] in failed else "no_failure_reported",
        }
        for row in profile["optional_requirements"]
    ]
    return raw


def bucket_report(*, failed=()):
    raw = optional_report(failed=failed)
    rows = raw["rule_execution"].pop("optional_requirements")
    raw["inventory"].pop("optional_requirements")
    raw["inventory"].pop("optional_features")
    raw["inventory"].pop("optional_requirement_count")
    rules = sorted({row["rule"] for row in rows})
    raw["rule_execution"]["buckets"] = [
        {
            "bucket": "neutral",
            "declared_requirement_codes": [row["code"] for row in rows],
            "enabled_rules": rules,
            "method_calls": [
                {"rule": rule, "method": "CheckPrim", "count": 1} for rule in rules
            ],
        }
    ]
    raw["soft_issues"] = [{"rule_id": code, "severity": "failure"} for code in failed]
    return raw


def test_optional_grade_is_separate_from_the_unchanged_mandatory_95_plus_five():
    roster = pinned_optional_profile()["optional_requirements"]
    failed = [row["code"] for row in roster[:3]]
    result = score_validation(optional_report(failed=failed), scoring_policy())
    assert result["score_percent"] == 100
    assert result["completion_bonus"] == 5
    assert result["passed_count"] == result["applicable_count"] == 46
    optional = result["optional_score"]
    assert optional["score_percent"] == pytest.approx(100 * 48 / 51)
    assert (
        optional["passed_count"],
        optional["failed_count"],
        optional["unknown_count"],
    ) == (48, 3, 0)
    assert optional["coverage_fraction"] == 1
    assert optional["measured_count"] == optional["applicable_count"] == 51


def test_recorded_runtime_buckets_backfill_existing_real_requirement_evidence():
    failed = [
        row["code"] for row in pinned_optional_profile()["optional_requirements"][:2]
    ]
    explicit = score_optional_validation(optional_report(failed=failed))
    historical = score_optional_validation(bucket_report(failed=failed))
    assert historical == explicit


def test_shared_rule_calls_do_not_assess_requirements_outside_their_declared_bucket():
    raw = bucket_report()
    raw["rule_execution"]["buckets"][0]["declared_requirement_codes"].pop()
    result = score_optional_validation(raw)
    assert result["score_percent"] is None
    assert (
        result["passed_count"],
        result["failed_count"],
        result["unknown_count"],
    ) == (50, 0, 1)
    assert result["measured_count"] == 50
    assert result["coverage_fraction"] == pytest.approx(50 / 51)
    assert result["lower_bound_percent"] == pytest.approx(100 * 50 / 51)
    assert result["upper_bound_percent"] == 100


@pytest.mark.parametrize(
    "change", ["not_invoked", "zero_calls", "missing", "exception"]
)
def test_unassessed_optional_requirements_are_unknown_and_never_count_as_failures(
    change,
):
    raw = optional_report()
    row = raw["rule_execution"]["optional_requirements"][0]
    if change == "not_invoked":
        row["rule_invoked"] = False
    elif change == "zero_calls":
        row["method_calls"] = {"CheckStage": 0}
    elif change == "missing":
        raw["rule_execution"]["optional_requirements"].pop(0)
    else:
        row["disposition"] = "infrastructure_error"
    optional = score_validation(raw, scoring_policy())["optional_score"]
    assert optional["score_percent"] is None
    assert optional["unknown_count"] == 1 and optional["failed_count"] == 0


def test_optional_rule_source_drift_does_not_change_the_mandatory_grade():
    raw = optional_report()
    additional = set(pinned_optional_profile()["rule_module_sha256"]) - set(
        pinned_profile()["rule_module_sha256"]
    )
    raw["rule_execution"]["sdk_source_sha256"][sorted(additional)[0]] = "0" * 64
    result = score_validation(raw, scoring_policy())
    assert result["score_percent"] == 100
    assert result["optional_score"]["score_percent"] is None
    assert (
        result["optional_score"]["unknown_reason"]
        == "validator_implementation_identity_mismatch"
    )


def test_missing_and_invalid_candidates_do_not_fabricate_optional_assessments():
    for invalid in (False, True):
        optional = score_optional_validation(None, candidate_invalid=invalid)
        assert optional["score_percent"] is None
        assert (
            optional["measured_count"]
            == optional["passed_count"]
            == optional["failed_count"]
            == 0
        )
        assert optional["unknown_count"] == optional["applicable_count"] == 51


@pytest.mark.parametrize("change", ["duplicate", "version", "bucket_calls"])
def test_malformed_or_mismatched_optional_execution_cannot_produce_a_complete_score(
    change,
):
    raw = bucket_report() if change == "bucket_calls" else optional_report()
    if change == "duplicate":
        raw["rule_execution"]["optional_requirements"][1] = deepcopy(
            raw["rule_execution"]["optional_requirements"][0]
        )
    elif change == "version":
        raw["rule_execution"]["optional_requirements"][0]["version"] = "99.0.0"
    else:
        raw["rule_execution"]["buckets"][0]["method_calls"][0]["count"] = True
    assert score_optional_validation(raw)["score_percent"] is None


def test_optional_pin_extends_the_exact_mandatory_profile_without_changing_its_identity():
    optional = pinned_optional_profile()
    assert optional["mandatory_inventory_sha256"] == digest(pinned_profile())
    assert (
        optional["profile_id"] == "Robotics-Prop"
        and optional["profile_version"] == "4.0.0"
    )
    codes = [row["code"] for row in optional["optional_requirements"]]
    assert len(codes) == len(set(codes)) == optional["optional_requirement_count"] == 51
    assert not set(codes) & {
        row["code"] for row in pinned_profile()["mandatory_requirements"]
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("score_percent", 100),
        ("coverage_fraction", 0.5),
        ("passed_count", 51),
        ("measured_count", 50),
        ("lower_bound_percent", 0),
        ("upper_bound_percent", 100),
    ],
)
def test_optional_measurements_reject_inconsistent_scores_counts_or_bounds(key, value):
    failure = pinned_optional_profile()["optional_requirements"][0]["code"]
    score = score_optional_validation(optional_report(failed=[failure]))
    assert validate_optional_score(score) is score
    score[key] = value
    with pytest.raises(ValueError):
        validate_optional_score(score)


def test_installed_sdk_optional_inventory_and_implementations_match_the_pinned_profile():
    pytest.importorskip("simready.validate")
    from simready_bench.engine.validator import derive_inventory

    inventory = derive_inventory()
    optional = pinned_optional_profile()
    for key in (
        "optional_features",
        "optional_requirements",
        "optional_requirement_count",
    ):
        assert inventory[key] == optional[key]
    assert inventory["optional_missing_rule_implementations"] == []
    for module, expected in optional["rule_module_sha256"].items():
        source = Path(inspect.getfile(importlib.import_module(module)))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


def test_real_sdk_dispatch_records_both_mandatory_and_optional_requirement_outcomes(
    tmp_path,
):
    pytest.importorskip("simready.validate")
    from simready_bench.engine.validator import validate_asset

    asset = tmp_path / "asset.usda"
    asset.write_text("""#usda 1.0
(
    defaultPrim = "Asset"
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "Asset" {
    def Cube "Body" {
        double size = 1
    }
}
""")
    report = validate_asset(asset)
    assert report["engine_dispatch_completed"] is True
    execution = report["rule_execution"]
    assert len(execution["requirements"]) == 46
    assert len(execution["optional_requirements"]) == 51
    score = score_optional_validation(report)
    assert score["measured_count"] > 0
    assert score["applicable_count"] == 51
    assert score["passed_count"] + score["failed_count"] + score["unknown_count"] == 51
    validate_optional_score(score)
