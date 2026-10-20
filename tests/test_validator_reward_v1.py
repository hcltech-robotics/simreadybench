"""Versioned mandatory-ID 95/5 objective, independent of physical or visual targets."""

from copy import deepcopy
import pytest
from simready_bench.engine.scoring import pinned_profile, score_validation
from simready_bench.engine.verifier import scoring_policy


def official_report(failed=(), optional=()):
    p = pinned_profile()
    failed = set(failed)
    return {
        "inventory": deepcopy(p),
        "engine_dispatch_completed": True,
        "passed": not failed,
        "outcome": "failure" if failed else "success",
        "infrastructure_errors": [],
        "issues": [{"rule_id": c, "severity": "failure"} for c in failed],
        "soft_issues": [{"rule_id": c, "severity": "failure"} for c in optional],
        "rule_execution": {
            "completed": True,
            "sdk_source_sha256": {
                **p["SDK_dispatch_source_sha256"],
                **p["rule_module_sha256"],
            },
            "requirements": [
                {
                    **r,
                    "rule_enabled": True,
                    "rule_invoked": True,
                    "method_calls": {"CheckPrim": 1},
                    "disposition": "failed"
                    if r["code"] in failed
                    else "no_failure_reported",
                }
                for r in p["mandatory_requirements"]
            ],
        },
    }


def test_official_mandatory_pass_is_exactly_100_even_with_optional_failures():
    policy = scoring_policy(excluded_rules=[])
    r = score_validation(official_report(optional=["optional.unrelated"]), policy)
    assert r["score_percent"] == 100 and r["completion_bonus"] == 5
    assert len(r["checks"]) == 46 and r["passed_count"] == r["applicable_count"] == 46
    assert r["official_conformance"]["passed"] is True


def test_individual_requirement_ids_count_once_and_incremental_repair_is_95_over_46():
    p = scoring_policy(excluded_rules=[])
    failures = ["com.nvidia.simready.RB.007", "com.nvidia.usd.VG.014"]
    raw = official_report(failed=failures)
    raw["issues"] *= 100
    before = score_validation(raw, p)
    after = score_validation(official_report(failed=failures[:1]), p)
    assert (
        before["score_percent"] == pytest.approx(95 * 44 / 46)
        and before["completion_bonus"] == 0
    )
    assert after["score_percent"] - before["score_percent"] == pytest.approx(95 / 46)
    assert before["applicable_count"] == after["applicable_count"] == 46
    assert after["categories"]["rigid_body_collision"]["failed"] == 1


def test_mandatory_semantics_naming_and_thumbnail_each_count_in_the_full_profile():
    p = scoring_policy()
    for code in (
        "com.nvidia.simready.SL.001",
        "com.nvidia.simready.NP.002",
        "com.nvidia.simready.SR.002",
    ):
        result = score_validation(official_report(failed=[code]), p)
        assert result["score_percent"] == pytest.approx(95 * 45 / 46)
        assert result["completion_bonus"] == 0 and result["applicable_count"] == 46
        assert result["official_conformance"]["passed"] is False
        assert (
            next((r for r in result["checks"] if r["requirement_id"] == code))[
                "scoring"
            ]
            is True
        )
    with pytest.raises(ValueError, match="filtered profiles"):
        scoring_policy(excluded_rules=["com.nvidia.simready.SL.001"])


def test_validator_exception_missing_execution_and_rule_exception_are_unknown():
    p = scoring_policy(excluded_rules=[])
    for mode in ("global", "dispatch", "rule", "missing"):
        raw = official_report()
        if mode == "global":
            raw["infrastructure_errors"] = [{"code": "validator_adapter_error"}]
        elif mode == "dispatch":
            raw["engine_dispatch_completed"] = False
        elif mode == "rule":
            raw["rule_execution"]["requirements"][0]["disposition"] = (
                "infrastructure_error"
            )
        else:
            raw["rule_execution"]["requirements"].pop()
        r = score_validation(raw, p)
        assert r["score_percent"] is None and r["completion_bonus"] is None
        assert r["status"] == "unknown" and r["applicable_count"] == 46


def test_candidate_invalid_is_zero_with_same_fixed_denominator():
    p = scoring_policy(excluded_rules=[])
    r = score_validation(None, p, candidate_invalid=True)
    assert (
        r["score_percent"] == 0
        and r["completion_bonus"] == 0
        and (r["applicable_count"] == 46)
    )
    assert r["official_conformance"]["passed"] is None


def test_unknown_candidate_does_not_make_a_requirement_inapplicable():
    p = scoring_policy(excluded_rules=[])
    raw = official_report()
    raw["rule_execution"]["requirements"][0]["rule_invoked"] = False
    r = score_validation(raw, p)
    assert (
        r["score_percent"] is None
        and r["applicable_count"] == 46
        and (r["unknown_count"] == 1)
    )
    assert r["score_lower_bound"] == pytest.approx(95 * 45 / 46)


def test_success_needs_executed_requirement_outcome_not_absence_of_issues():
    raw = official_report()
    raw["rule_execution"]["requirements"][0].pop("disposition")
    score = score_validation(raw, scoring_policy(excluded_rules=[]))
    assert score["score_percent"] is None and score["unknown_count"] == 1


def test_evaluator_wire_and_real_Harbor_reader_share_the_95_5_score(tmp_path):
    import time
    from types import SimpleNamespace
    from simready_bench.engine.evaluation import Evaluator
    from simready_bench.engine.scoring import build_wire, validate_wire
    from simready_bench.engine.verifier import write_verifier_output
    from harbor.verifier.verifier import Verifier
    from helpers import candidate_fixture as fixture

    actor, controller, cfg, _, _ = fixture(tmp_path)
    cfg["validator_scoring"] = scoring_policy(excluded_rules=[])
    snap = controller.freeze_candidate("ep")
    observation = {
        "source_record": {
            "sha256": next(
                (f.sha256 for f in snap.files if f.relative_path == snap.entrypoint)
            )
        },
        "nvidia_report": official_report(),
        "preflight": {"package_ok": True},
    }

    def observe(current, **kwargs):
        return {"snapshot_sha256": current.sha256, "observation": deepcopy(observation)}

    def forbidden_gpu(*args, **kwargs):
        raise AssertionError("GPU cannot gate mandatory validator score")

    evaluator = Evaluator(tmp_path / "validator-evaluator", observer=observe)
    report = evaluator.evaluate(
        snap, cfg, "ep", time.time() + 5, stop_reason="voluntary_good_enough"
    )
    assert report["reward"] == 100 and report["completion"]["earned_bonus"] == 5
    assert report["official_reward"] == 1
    assert "process_event" not in report
    assert evaluator.evaluate(snap, cfg, "ep", time.time() + 5) == report
    wire = build_wire(report, snapshot=snap)
    out = tmp_path / "verifier"
    assert write_verifier_output(wire, out) == 0
    reader = object.__new__(Verifier)
    reader.trial_paths = SimpleNamespace(reward_json_path=out / "reward.json")
    assert reader._parse_reward_json() == {"simready_reward": 100.0}
    changed = deepcopy(wire)
    changed["reward"] = 99
    with pytest.raises(ValueError):
        validate_wire(changed)
    changed_cfg = deepcopy(cfg)
    changed_cfg["validator_scoring"] = {**scoring_policy(), "all_pass_bonus_percent": 6}
    with pytest.raises(ValueError):
        evaluator.evaluate(snap, changed_cfg, "ep", time.time() + 5)


def test_installed_requirement_module_drift_is_not_accepted(tmp_path, monkeypatch):
    import inspect, importlib
    from simready_bench.engine.scoring import verify_validator_installation
    from simready_bench.engine import validator

    p = pinned_profile()
    monkeypatch.setattr(
        validator, "_sdk_source_identity", lambda: p["SDK_dispatch_source_sha256"]
    )
    monkeypatch.setattr(importlib, "import_module", lambda name: object())
    changed = tmp_path / "changed_module.py"
    changed.write_text("changed implementation")
    monkeypatch.setattr(inspect, "getfile", lambda module: str(changed))
    with pytest.raises(RuntimeError, match="requirement implementation changed"):
        verify_validator_installation(p)


def test_real_dispatch_identity_may_include_additional_optional_rule_modules():
    raw = official_report()
    raw["rule_execution"]["sdk_source_sha256"]["optional.module"] = "f" * 64
    assert score_validation(raw, scoring_policy())["score_percent"] == 100
    changed = deepcopy(raw)
    key = next(iter(pinned_profile()["rule_module_sha256"]))
    changed["rule_execution"]["sdk_source_sha256"][key] = "0" * 64
    assert score_validation(changed, scoring_policy())["score_percent"] is None


def test_mandatory_validator_execution_does_not_require_GPU_permission(tmp_path):
    import socket, sys
    from simready_bench.preparation import runtime_policy

    runtime = {
        "repository_root": str(tmp_path),
        "source_root": str(tmp_path),
        "validator_python": sys.executable,
        "expected_hostname": socket.gethostname(),
        "allowed_reference_root": str(tmp_path),
        "shared_profile": {},
        "execution_enabled": True,
        "gpu_enabled": False,
    }
    resolved = {
        "operator_policy": {"runtime": runtime},
        "normalized_manifest": {"scoring": scoring_policy()},
    }
    assert runtime_policy(resolved, execute=True)["gpu_enabled"] is False
    resolved["operator_policy"]["runtime"]["execution_enabled"] = False
    with pytest.raises(RuntimeError):
        runtime_policy(resolved, execute=True)


def test_prepared_public_dimensions_are_readable_by_actual_runner(tmp_path):
    from simready_bench.assets import public_instruction
    from simready_bench.local import _dimensions

    dimensions = [0.25, 0.3, 0.4]
    (tmp_path / "instruction.md").write_text(public_instruction(dimensions))
    assert _dimensions(tmp_path) == dimensions
