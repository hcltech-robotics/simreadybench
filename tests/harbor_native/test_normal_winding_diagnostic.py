"""Synthetic SDK evidence tests projection, not a live NVIDIA execution."""
from copy import deepcopy

import pytest

from simready_eval import formal, validator

CODE = "com.nvidia.usd.VG.029"
RULE = "usd_validation_nvidia.rules._geometry.NormalsWindingsChecker"
MODULE = RULE.rsplit(".", 1)[0]
PROFILE = formal.pinned_profile()
REQUIREMENT = next(r for r in PROFILE["mandatory_requirements"] if r["code"] == CODE)
ISSUE = {"rule_id": CODE, "requirement_version": "1.0.0", "rule": RULE,
         "severity": "error", "prim_path": "/Asset/Collider",
         "message": "Mesh '/Asset/Collider' has normals inconsistent with the face windings."}
BUCKETS = [{"declared_requirement_codes": [CODE], "enabled_rules": [RULE],
            "method_calls": [{"rule": RULE, "method": "CheckPrim", "count": 1}]}]


def project(issues, *, versions=None, sources=None):
    return validator.requirement_outcomes(
        [REQUIREMENT], BUCKETS, issues,
        versions=PROFILE["versions"] if versions is None else versions,
        sources=PROFILE["rule_module_sha256"] if sources is None else sources)[0]


def test_exact_pinned_candidate_diagnostic_fails_without_changing_raw_issue():
    issue = deepcopy(ISSUE)
    assert project([issue])["disposition"] == "failed"
    assert issue == ISSUE


@pytest.mark.parametrize("field,value", [
    ("message", "RuntimeError: failed to inspect normals"),
    ("message", "Mesh '/Other' has normals inconsistent with the face windings."),
    ("message", None), ("prim_path", None), ("prim_path", ""),
    ("requirement_version", "2.0.0"), ("rule", "other.Checker"),
])
def test_other_or_malformed_diagnostic_remains_execution_error(field, value):
    issue = {**ISSUE, field: value}
    assert project([issue])["disposition"] == "infrastructure_error"


@pytest.mark.parametrize("versions,sources", [
    ({}, PROFILE["rule_module_sha256"]),
    ({**PROFILE["versions"], "usd-validation-nvidia": "1.23.0"}, PROFILE["rule_module_sha256"]),
    (PROFILE["versions"], {}), (PROFILE["versions"], {MODULE: "0" * 64}),
])
def test_missing_or_other_sdk_identity_remains_execution_error(versions, sources):
    assert project([ISSUE], versions=versions, sources=sources)["disposition"] == "infrastructure_error"


def test_exception_in_same_requirement_takes_precedence():
    exception = {**ISSUE, "message": "RuntimeError: failed to inspect normals"}
    assert project([ISSUE, exception])["disposition"] == "infrastructure_error"
    unscoped = {**exception, "rule_id": None}
    assert project([ISSUE, unscoped])["disposition"] == "infrastructure_error"


def test_without_identity_and_without_diagnostic_no_reclassification():
    assert validator.requirement_outcomes([REQUIREMENT], BUCKETS, [ISSUE])[0]["disposition"] == "infrastructure_error"
    assert project([])["disposition"] == "no_failure_reported"


def test_live_report_preserves_diagnostic_and_only_removes_its_infrastructure_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pxr import Usd, UsdGeom
    asset = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(asset))
    UsdGeom.Cube.Define(stage, "/Cube")
    stage.GetRootLayer().Save()
    inventory = {**PROFILE, "missing_rule_implementations": []}
    execution = {"requirements": [project([ISSUE])], "sdk_source_sha256": PROFILE["rule_module_sha256"]}
    api = SimpleNamespace(features_summary_passed=lambda summary: False, destroy=lambda: None)
    monkeypatch.setattr(validator, "_dependencies", lambda: (api, {"physx_native_bindings": True}))
    monkeypatch.setattr(validator, "_inventory", lambda *a: inventory)
    monkeypatch.setattr(validator, "_run_profile", lambda *a: ([ISSUE], [], [ISSUE], execution, {"failed": True}))
    raw = validator.validate_asset(asset)
    assert raw["outcome"] == "failure"
    assert raw["infrastructure_errors"] == []
    assert raw["issues"] == [ISSUE]
    exception = {**ISSUE, "message": "RuntimeError: failed to inspect normals"}
    monkeypatch.setattr(validator, "_run_profile", lambda *a: ([ISSUE, exception], [], [ISSUE, exception], execution, {}))
    raw = validator.validate_asset(asset)
    assert raw["outcome"] == "infrastructure_error"
    assert raw["infrastructure_errors"] == [{"code": "validator_rule_execution_error", "issue": exception}]
    def broken_dispatch(*args):
        raise RuntimeError('actual dispatch exception')
    monkeypatch.setattr(validator, '_run_profile', broken_dispatch)
    raw = validator.validate_asset(asset)
    assert raw['infrastructure_errors'][0]['code'] == 'validator_adapter_error'
    assert raw['engine_dispatch_completed'] is False
    assert formal.score_validation(raw)['unknown_count'] == 46


def saved_evaluation(tmp_path):
    import hashlib
    from test_formal import recorded_report
    data = b'synthetic candidate hash check; never opened as USD'
    (tmp_path / 'asset.usd').write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    raw = recorded_report()
    rules = sorted({r['rule'] for r in PROFILE['mandatory_requirements']})
    buckets = [{'declared_requirement_codes': [r['code'] for r in PROFILE['mandatory_requirements']],
                'enabled_rules': rules,
                'method_calls': [{'rule': rule, 'method': 'CheckPrim', 'count': 1} for rule in rules]}]
    raw.update(asset_sha256=sha, issues=[deepcopy(ISSUE)], soft_issues=[], warnings=[], infos=[],
               infrastructure_errors=[{'code': 'validator_rule_execution_error', 'issue': deepcopy(ISSUE)}])
    raw['rule_execution']['buckets'] = buckets
    raw['rule_execution']['requirements'] = validator.requirement_outcomes(PROFILE['mandatory_requirements'], buckets, raw['issues'])
    raw['passed'] = False
    score = formal.score_validation(raw)
    return {'candidate': {'entrypoint': 'asset.usd', 'files': [{'path': 'asset.usd', 'sha256': sha, 'size_bytes': len(data)}]},
            'formal_report': {'nvidia_validator_report': raw, 'simready': score,
                              'infrastructure_errors': deepcopy(raw['infrastructure_errors']), 'coverage': {}},
            'scores': {'formal': {}, 'visual': {'score_percent': 50}, 'kinematic': {'score_percent': 25}}}


def regrader():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'native/tools/regrade_normals_diagnostic.py'
    spec = importlib.util.spec_from_file_location('normals_regrade_tool', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.regrade


def test_saved_reprojection_resolves_unknown_and_preserves_other_measurements(tmp_path):
    report = saved_evaluation(tmp_path)
    original = deepcopy(report)
    result = regrader()(report, tmp_path)
    assert result['formal_report']['simready']['unknown_count'] == 0
    assert result['formal_report']['simready']['failed_count'] == 1
    assert result['scores']['formal']['score_percent'] == pytest.approx(95 * 45 / 46)
    assert result['outcome'] == 'complete'
    assert report == original


@pytest.mark.parametrize('mode', ['hash_changed', 'missing_file', 'missing_buckets', 'other_exception', 'source_drift', 'malformed_errors'])
def test_saved_reprojection_refuses_unverifiable_or_other_failures(tmp_path, mode):
    report = saved_evaluation(tmp_path)
    raw = report['formal_report']['nvidia_validator_report']
    if mode == 'hash_changed':
        (tmp_path / 'asset.usd').write_bytes(b'changed')
    elif mode == 'missing_file':
        (tmp_path / 'asset.usd').unlink()
    elif mode == 'missing_buckets':
        raw['rule_execution']['buckets'] = []
    elif mode == 'other_exception':
        raw['issues'].append({**ISSUE, 'message': 'RuntimeError: execution failed'})
    elif mode == 'source_drift':
        raw['rule_execution']['sdk_source_sha256'][MODULE] = '0' * 64
    else:
        raw['infrastructure_errors'] = None
    original = deepcopy(report)
    with pytest.raises(ValueError):
        regrader()(report, tmp_path)
    assert report == original
