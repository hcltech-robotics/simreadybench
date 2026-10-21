"""Actual NVIDIA CPU label checks on independent fixtures, never model submissions."""

import importlib.util
import pytest

pytestmark = pytest.mark.licensed

pytest.importorskip("pxr")
if importlib.util.find_spec("simready") is None:
    pytest.skip("installed NVIDIA framework required", allow_module_level=True)


def test_modern_asset_and_material_label_presence_against_installed_nvidia_checker(
    tmp_path,
):
    from pxr import Usd, UsdGeom, UsdShade, UsdSemantics
    from simready_bench.engine.validator import validate_asset
    from simready_bench.engine.semantics import observe_labels, information

    stage = Usd.Stage.CreateNew(str(tmp_path / "asset.usda"))
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/Asset/Cube")
    material = UsdShade.Material.Define(stage, "/Asset/Looks/Material")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

    def failed_presence():
        result = validate_asset(str(tmp_path / "asset.usda"))
        assert result["engine_dispatch_completed"] is True
        return {
            row.get("rule_id")
            for row in result["issues"]
            if row.get("severity") in ("error", "failure")
        }

    before = failed_presence()
    assert observe_labels(stage)["entries"] == []
    assert {"com.nvidia.simready.SL.001", "com.nvidia.simready.SL.MAT.001"} <= before
    UsdSemantics.LabelsAPI.Apply(root.GetPrim(), "class").CreateLabelsAttr(
        ["fixture_cube"]
    )
    UsdSemantics.LabelsAPI.Apply(material.GetPrim(), "material").CreateLabelsAttr(
        ["plastic"]
    )
    stage.GetRootLayer().Save()
    after = failed_presence()
    reported = information(observe_labels(stage))
    assert reported["required"] is False and reported["scoring"] is False
    assert {label for r in reported["entries"] for label in r["labels"]} == {
        "fixture_cube",
        "plastic",
    }
    assert (
        "com.nvidia.simready.SL.001" not in after
        and "com.nvidia.simready.SL.MAT.001" not in after
    )
