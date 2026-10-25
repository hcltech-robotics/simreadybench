"""Neutral formal scoring is fixed-denominator and requires genuine dispatch evidence."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import pytest

from simready_eval import formal, validator


def recorded_report(*, failed=(), optional_failed=()):
    """Synthetic recorded evidence exercises projection, never simulates the SDK."""
    profile = formal.pinned_profile()
    optional = formal.pinned_optional_profile()
    inventory = deepcopy(profile)
    inventory.update({key: deepcopy(optional[key]) for key in (
        "optional_features", "optional_requirements", "optional_requirement_count"
    )})
    def rows(roster, failures):
        return [{**row, "rule_enabled": True, "rule_invoked": True,
                 "method_calls": {"CheckPrim": 1},
                 "disposition": "failed" if row["code"] in failures else "no_failure_reported"}
                for row in roster]
    return {"inventory": inventory, "engine_dispatch_completed": True,
            "passed": not failed, "infrastructure_errors": [],
            "rule_execution": {"completed": True, "sdk_source_sha256": {
                **profile["SDK_dispatch_source_sha256"], **profile["rule_module_sha256"],
                **optional["rule_module_sha256"]},
                "requirements": rows(profile["mandatory_requirements"], failed),
                "optional_requirements": rows(optional["optional_requirements"], optional_failed)}}


def make_asset(root):
    path = root / "asset.usd"
    path.write_text('#usda 1.0\n(defaultPrim = "Asset"\n upAxis = "Z"\n metersPerUnit = 1)\ndef Xform "Asset" {\n def Cube "Body" {}\n}\n')
    return path


def test_fixed_mandatory_95_plus_5_and_optional_report_only():
    mandatory = formal.pinned_profile()["mandatory_requirements"]
    optional = formal.pinned_optional_profile()["optional_requirements"]
    score = formal.score_validation(recorded_report(optional_failed=[optional[0]["code"]]))
    assert score["score_percent"] == 100
    assert score["passed_count"] == score["applicable_count"] == 46
    assert score["optional_score"]["score_percent"] == pytest.approx(100 * 50 / 51)
    score = formal.score_validation(recorded_report(failed=[mandatory[0]["code"]]))
    assert score["score_percent"] == pytest.approx(95 * 45 / 46)
    assert score["completion_bonus"] == 0
    with pytest.raises(ValueError):
        formal.scoring_policy(excluded_rules=[mandatory[0]["code"]])


@pytest.mark.parametrize("mode", ["missing", "uninvoked", "source_drift", "exception", "bad_calls"])
def test_unknown_keeps_denominator_and_exact_bounds(mode):
    raw = recorded_report()
    row = raw["rule_execution"]["requirements"][0]
    if mode == "missing":
        raw["rule_execution"]["requirements"].pop(0)
    elif mode == "uninvoked":
        row["rule_invoked"] = False
    elif mode == "source_drift":
        raw["rule_execution"]["sdk_source_sha256"][next(iter(formal.pinned_profile()["rule_module_sha256"]))] = "0" * 64
    elif mode == "exception":
        row["disposition"] = "infrastructure_error"
    else:
        row["method_calls"]["CheckStage"] = True
    score = formal.score_validation(raw)
    assert score["score_percent"] is None and score["completion_bonus"] is None
    assert score["applicable_count"] == 46
    if mode != "source_drift":
        assert score["unknown_count"] == 1
        assert score["score_lower_bound"] == pytest.approx(95 * 45 / 46)
    assert score["score_upper_bound"] == 100


@pytest.mark.parametrize("raw", [None, {}, {"inventory": []}, {"inventory": {"x": 1}, "rule_execution": []}])
def test_unavailable_or_malformed_validator_evidence_is_unknown(raw):
    score = formal.score_validation(raw)
    assert score["score_percent"] is None and score["unknown_count"] == 46
    assert score["score_lower_bound"] == 0 and score["score_upper_bound"] == 100


def test_proven_invalid_is_zero_and_fixed_denominators():
    score = formal.score_validation(None, candidate_invalid=True)
    assert score["score_percent"] == score["optional_score"]["score_percent"] == 0
    assert score["failed_count"] == score["applicable_count"] == 46
    assert score["optional_score"]["failed_count"] == 51


def test_dependency_failure_is_infrastructure_and_does_not_change_candidate(tmp_path, monkeypatch):
    asset = make_asset(tmp_path)
    before = asset.read_bytes()
    def missing():
        raise ModuleNotFoundError("No module named 'simready'", name="simready")
    monkeypatch.setattr(validator, "_dependencies", missing)
    result = formal.evaluate(tmp_path)
    assert result["outcome"] == "infrastructure_error"
    assert result["simready"]["score_percent"] is None
    errors = result["nvidia_validator_report"]["infrastructure_errors"]
    assert any(row["code"] == "validator_dependency_unavailable" and row["module"] == "simready" for row in errors)
    assert asset.read_bytes() == before
    assert not {"episode_id", "snapshot_id", "hostname"} & result.keys()


def test_preflight_engine_error_is_unknown(tmp_path, monkeypatch):
    make_asset(tmp_path)
    monkeypatch.setattr(formal, "inspect_package", lambda *args, **kwargs: {
        "package_ok": True, "findings": [], "infrastructure_errors": ["package_checker_error:RuntimeError"]})
    monkeypatch.setattr(validator, "validate_asset", lambda *args, **kwargs: pytest.fail("Do not dispatch after failed preflight"))
    result = formal.evaluate(tmp_path)
    assert result["outcome"] == "infrastructure_error"
    assert result["simready"]["score_percent"] is None


def test_missing_entrypoint_is_zero_without_sdk(tmp_path, monkeypatch):
    monkeypatch.setattr(validator, "validate_asset", lambda *args, **kwargs: pytest.fail("Do not dispatch invalid candidate"))
    result = formal.evaluate(tmp_path)
    assert result["outcome"] == "invalid_submission"
    assert result["simready"]["score_percent"] == 0


def test_real_sdk_dispatch_when_installed(tmp_path):
    try:
        importlib.util.find_spec("simready.validate")
    except ModuleNotFoundError:
        pytest.skip("Pinned public NVIDIA validator is not installed in this interpreter")
    make_asset(tmp_path)
    before = hashlib.sha256((tmp_path / "asset.usd").read_bytes()).hexdigest()
    result = formal.evaluate(tmp_path)
    raw = result["nvidia_validator_report"]
    assert raw["engine_dispatch_completed"] is True, raw["infrastructure_errors"]
    assert len(raw["rule_execution"]["requirements"]) == 46
    assert len(raw["rule_execution"]["optional_requirements"]) == 51
    assert result["simready"]["score_percent"] is not None
    assert hashlib.sha256((tmp_path / "asset.usd").read_bytes()).hexdigest() == before


@pytest.mark.parametrize("kind", ["bad_sources", "bad_errors", "unhashable_code"])
def test_malformed_execution_containers_are_unknown(kind):
    raw = recorded_report()
    if kind == "bad_sources":
        raw["rule_execution"]["sdk_source_sha256"] = ["invalid"]
    elif kind == "bad_errors":
        raw["infrastructure_errors"] = None
    else:
        raw["rule_execution"]["requirements"][0]["code"] = []
        raw["rule_execution"]["optional_requirements"][0]["code"] = []
    score = formal.score_validation(raw)
    assert score["status"] == "unknown" and score["score_percent"] is None


def test_unknown_bounds_with_confirmed_failure_exclude_completion_bonus():
    raw = recorded_report(failed=[formal.pinned_profile()["mandatory_requirements"][0]["code"]])
    raw["rule_execution"]["requirements"][1]["rule_invoked"] = False
    score = formal.score_validation(raw)
    assert score["score_lower_bound"] == pytest.approx(95 * 44 / 46)
    assert score["score_upper_bound"] == pytest.approx(95 * 45 / 46)
    assert score["passed_count"] == 44 and score["failed_count"] == score["unknown_count"] == 1


def test_material_guard_detects_only_malformed_builtin_surface(tmp_path):
    from pxr import Usd, UsdGeom, UsdShade
    from simready_eval import compat
    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/Material")
    UsdGeom.Cube.Define(stage, "/Cube")
    material.GetPrim().CreateRelationship("outputs:surface").SetTargets(["/Cube"])
    assert compat.relationship_at_material_surface(material.GetPrim()) is True
    assert compat.relationship_at_material_surface(material.GetPrim(), "relatedAsset") is False
    assert compat.relationship_at_material_surface(stage.GetPrimAtPath("/Cube")) is False


def test_genuine_malformed_material_guard_keeps_native_findings(tmp_path):
    try:
        importlib.util.find_spec("simready.validate")
    except ModuleNotFoundError:
        pytest.skip("Pinned public NVIDIA validator is not installed in this interpreter")
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    from simready_eval import compat
    stage = Usd.Stage.CreateNew(str(tmp_path / "asset.usd"))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    cube = UsdGeom.Cube.Define(stage, "/Asset/Body")
    material = UsdShade.Material.Define(stage, "/Asset/Material")
    shader = UsdShade.Shader.Define(stage, "/Asset/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.GetPrim().CreateRelationship("outputs:surface").SetTargets([shader.GetPath().AppendProperty("outputs:surface")])
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    result = formal.evaluate(tmp_path)
    raw = result["nvidia_validator_report"]
    assert raw["engine_dispatch_completed"] is True, raw["infrastructure_errors"]
    assert raw["infrastructure_errors"] == []
    assert raw["rule_execution"]["compatibility_adapters"] == [compat.compatibility_identity()]
    checks = {row["requirement_id"]: row["status"] for row in result["simready"]["checks"]}
    assert checks["com.nvidia.simready.VM.PS.002"] == "fail"
    assert result["simready"]["unknown_count"] == 0
    assert all(checks["com.nvidia.simready.VM." + name] == "pass" for name in ("MAT.001", "TEX.001", "TEX.003"))


def test_pinned_inventory_files_keep_public_source_identity():
    profile = formal.pinned_profile()
    optional = formal.pinned_optional_profile()
    assert profile["mandatory_requirement_count"] == 46
    assert optional["optional_requirement_count"] == 51
    assert optional["mandatory_inventory_sha256"] == formal.digest(profile)
    assert set(validator._SDK_SOURCE_PINS) == set(profile["SDK_dispatch_source_sha256"])
    assert validator._SDK_SOURCE_PINS == profile["SDK_dispatch_source_sha256"]


def test_installed_genuine_inventory_and_source_identity():
    try:
        importlib.util.find_spec("simready.validate")
    except ModuleNotFoundError:
        pytest.skip("Pinned public NVIDIA validator is not installed in this interpreter")
    profile = formal.pinned_profile()
    optional = formal.pinned_optional_profile()
    inventory = validator.derive_inventory()
    for key in ("profile_id", "profile_version", "mandatory_features", "mandatory_requirements",
                "mandatory_requirement_count", "profile_manifest_sha256", "tier_manifest_hashes", "versions"):
        assert inventory[key] == profile[key]
    for key in ("optional_features", "optional_requirements", "optional_requirement_count"):
        assert inventory[key] == optional[key]
    assert validator._sdk_source_identity() == profile["SDK_dispatch_source_sha256"]
    import inspect
    for module, expected in {**profile["rule_module_sha256"], **optional["rule_module_sha256"]}.items():
        source = Path(inspect.getfile(importlib.import_module(module)))
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
