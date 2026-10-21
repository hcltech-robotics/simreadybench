"""Compatibility identity and explicitly enabled local NVIDIA regressions."""

from contextlib import nullcontext
import hashlib
import inspect
import os
from pathlib import Path

import pytest

from simready_bench.engine import nvidia_compat

native = pytest.mark.skipif(
    os.environ.get("SIMREADY_NATIVE_VALIDATOR_TESTS") != "1"
    or bool(os.environ.get("CI")),
    reason="Pinned NVIDIA checks run only in explicitly enabled local validation",
)


def test_compatibility_identity_binds_adapter_and_original_source():
    identity = nvidia_compat.compatibility_identity()
    assert identity["source_sha256"] == hashlib.sha256(
        Path(inspect.getfile(nvidia_compat)).read_bytes()
    ).hexdigest()
    assert identity["upstream_source_sha256"] == nvidia_compat.RULE_SOURCE_SHA256
    assert identity["id"] == "simready-nvidia-material-surface-attribute-v1"


def test_candidate_observer_seals_compatibility_source(tmp_path):
    import socket
    import sys
    from simready_bench.engine.evaluation import CandidateObserver

    observer = CandidateObserver(
        validator_python=sys.executable,
        source_root=Path(inspect.getfile(nvidia_compat)).parents[2],
        state_dir=tmp_path,
        expected_hostname=socket.gethostname(),
    )
    assert observer.identity["sources"]["nvidia_compat.py"] == (
        nvidia_compat.compatibility_identity()["source_sha256"]
    )
    assert observer.identity["compatibility_adapters"] == [
        nvidia_compat.compatibility_identity()
    ]


def material_asset(tmp_path, *, malformed=False):
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    path = tmp_path / "asset.usda"
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    shape = UsdGeom.Cube.Define(stage, "/Asset/Body")
    material = UsdShade.Material.Define(stage, "/Asset/Material")
    shader = UsdShade.Shader.Define(stage, "/Asset/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.3, 0.5, 0.7))
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    if malformed:
        material.GetPrim().CreateRelationship("outputs:surface").SetTargets(
            [shader.GetPath().AppendProperty("outputs:surface")]
        )
    else:
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
    material.GetPrim().CreateRelationship("relatedAsset").SetTargets([shape.GetPath()])
    stage.GetRootLayer().Save()
    return path


def run_checker(stage, *, adapted):
    import simready.validate as api
    from usd_validation_nvidia._compliance_runners import ComplianceCheckerEventType
    api.initialize([], [], [])
    try:
        from simready.foundation.tier_core.capabilities.visualization.materials.validation import (
            VisualMaterialsCapabilityChecker,
        )

        checker = VisualMaterialsCapabilityChecker()
        with nvidia_compat.material_surface_connections() if adapted else nullcontext():
            ComplianceCheckerEventType.STAGE.apply(checker, (stage,))
            for prim in stage.Traverse():
                ComplianceCheckerEventType.PRIM.apply(checker, (prim,))
        return list(checker._issues)
    finally:
        api.destroy()


@pytest.mark.licensed
@native
def test_native_checker_malformed_surface_has_only_preview_requirement_failure(tmp_path):
    from pxr import Usd

    path = material_asset(tmp_path, malformed=True)
    source = path.read_bytes()
    stage = Usd.Stage.Open(str(path))
    original = run_checker(stage, adapted=False)
    assert any(issue.severity.name == "ERROR" for issue in original)
    guarded = run_checker(stage, adapted=True)
    assert not any(issue.severity.name == "ERROR" for issue in guarded)
    mandatory = {
        "com.nvidia.simready.VM.MAT.001",
        "com.nvidia.simready.VM.PS.002",
        "com.nvidia.simready.VM.TEX.001",
        "com.nvidia.simready.VM.TEX.003",
    }
    assert {
        issue.requirement.code
        for issue in guarded
        if issue.severity.name == "FAILURE"
        and issue.requirement is not None
        and issue.requirement.code in mandatory
    } == {"com.nvidia.simready.VM.PS.002"}
    assert path.read_bytes() == source


@pytest.mark.licensed
@native
def test_native_valid_material_and_binding_relationship_keep_identical_findings(tmp_path):
    from pxr import Usd

    path = material_asset(tmp_path)
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/Asset/Material")
    assert not nvidia_compat.relationship_at_material_surface(prim)
    assert not nvidia_compat.relationship_at_material_surface(prim, "relatedAsset")

    def outcomes(issues):
        return [
            (issue.severity.name, getattr(issue.requirement, "code", None), issue.message)
            for issue in issues
        ]

    assert outcomes(run_checker(stage, adapted=False)) == outcomes(
        run_checker(stage, adapted=True)
    )


@pytest.mark.licensed
@native
def test_stronger_valid_attribute_retains_native_connection(tmp_path):
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(material_asset(tmp_path, malformed=True)))
    stage.SetEditTarget(stage.GetSessionLayer())
    material = UsdShade.Material(stage.GetPrimAtPath("/Asset/Material"))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Asset/Material/Shader"))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    assert not nvidia_compat.relationship_at_material_surface(material.GetPrim())
    issues = run_checker(stage, adapted=True)
    assert not any(issue.severity.name == "ERROR" for issue in issues)
    assert not any(
        getattr(issue.requirement, "code", None) == "com.nvidia.simready.VM.PS.002"
        for issue in issues
    )


@pytest.mark.licensed
@native
def test_native_profile_scores_malformed_surface_without_unknown_or_package_rejection(tmp_path):
    from simready_bench.engine.scoring import score_validation
    from simready_bench.engine.validator import validate_asset
    from simready_bench.engine.verifier import scoring_policy

    path = material_asset(tmp_path, malformed=True)
    source = path.read_bytes()
    report = validate_asset(path)
    score = score_validation(report, scoring_policy())
    assert report["engine_dispatch_completed"] is True
    assert report["infrastructure_errors"] == []
    assert report["rule_execution"]["compatibility_adapters"] == [
        nvidia_compat.compatibility_identity()
    ]
    checks = {row["requirement_id"]: row["status"] for row in score["checks"]}
    assert checks["com.nvidia.simready.VM.PS.002"] == "fail"
    assert all(
        checks["com.nvidia.simready.VM." + name] == "pass"
        for name in ("MAT.001", "TEX.001", "TEX.003")
    )
    assert score["status"] == "scored"
    assert score["applicable_count"] == 46 and score["unknown_count"] == 0
    assert score["score_percent"] > 0
    assert path.read_bytes() == source


@pytest.mark.licensed
@native
def test_unexpected_native_checker_error_remains_unknown(tmp_path, monkeypatch):
    from simready_bench.engine import validator
    from simready_bench.engine.scoring import score_validation
    from simready_bench.engine.verifier import scoring_policy

    def unexpected(*args):
        raise RuntimeError("Independent native checker error")

    original_inventory = validator._inventory

    def inventory_with_error(*args):
        inventory = original_inventory(*args)
        from simready.foundation.tier_core.capabilities.visualization.materials.validation import (
            VisualMaterialsCapabilityChecker,
        )

        monkeypatch.setattr(
            VisualMaterialsCapabilityChecker, "check_vm_tex_001_texture_size", unexpected
        )
        return inventory

    monkeypatch.setattr(validator, "_inventory", inventory_with_error)
    report = validator.validate_asset(material_asset(tmp_path))
    score = score_validation(report, scoring_policy())
    assert any(
        item["code"] == "validator_rule_execution_error"
        for item in report["infrastructure_errors"]
    )
    assert score["status"] == "unknown" and score["score_percent"] is None
    assert score["unknown_count"] == 4 and score["applicable_count"] == 46


@pytest.mark.licensed
@native
def test_native_connection_helpers_are_restored_after_an_exception():
    from simready.foundation.tier_core.capabilities.visualization.materials.validation import (
        VisualMaterialsCapabilityChecker,
    )

    before = (
        VisualMaterialsCapabilityChecker._connected_surface_terminals,
        VisualMaterialsCapabilityChecker._terminal_shader,
    )
    with pytest.raises(RuntimeError, match="outside checker"):
        with nvidia_compat.material_surface_connections():
            raise RuntimeError("outside checker")
    assert before == (
        VisualMaterialsCapabilityChecker._connected_surface_terminals,
        VisualMaterialsCapabilityChecker._terminal_shader,
    )


@pytest.mark.licensed
@native
def test_adapter_refuses_an_unpinned_native_implementation(monkeypatch):
    monkeypatch.setattr(nvidia_compat, "RULE_SOURCE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="compatibility source changed"):
        with nvidia_compat.material_surface_connections():
            pytest.fail("Unpinned source must not execute through the adapter")
