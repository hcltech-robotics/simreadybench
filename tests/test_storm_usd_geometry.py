"""Executed with the full-imaging Python in the actual Daytona CPU image."""

import hashlib
import math
import pytest


def test_cross_root_material_binding_and_texture_paths_survive_normalization(tmp_path):
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdShade
    except ModuleNotFoundError:
        import pytest

        pytest.skip("The OpenUSD imaging runtime is required")
    from PIL import Image
    from simready_bench.storm import _stage

    path = tmp_path / "asset.usda"
    texture = tmp_path / "texture.png"
    Image.new("RGB", (8, 8), "orange").save(texture)
    source = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(source, 0.01)
    UsdGeom.SetStageUpAxis(source, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(source, "/Geom")
    cube = UsdGeom.Cube.Define(source, "/Geom/Cube")
    material = UsdShade.Material.Define(source, "/Looks/Material")
    shader = UsdShade.Shader.Define(source, "/Looks/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    tex = UsdShade.Shader.Define(source, "/Looks/Material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    source.GetRootLayer().Save()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    stage, root = _stage(path)
    cube = stage.GetPrimAtPath("/IllustrationAsset/Geom/Cube")
    bound = UsdShade.MaterialBindingAPI(cube).ComputeBoundMaterial()[0]
    assert bound and str(bound.GetPath()) == "/IllustrationAsset/Looks/Material"
    shader = bound.ComputeSurfaceSource()[0]
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    tex = UsdShade.Shader(
        stage.GetPrimAtPath("/IllustrationAsset/Looks/Material/Texture")
    )
    assert tex.GetInput("file").Get().resolvedPath == str(texture)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_MDL_only_material_gets_an_explicit_unscored_preview_approximation(tmp_path):
    try:
        from pxr import Gf, Sdf, Usd, UsdShade
    except ModuleNotFoundError:
        import pytest

        pytest.skip("The OpenUSD imaging runtime is required")
    from simready_bench.storm import _materials
    from simready_bench.captures import material_support

    stage = Usd.Stage.CreateInMemory()
    material = UsdShade.Material.Define(stage, "/Looks/MDL")
    shader = UsdShade.Shader.Define(stage, "/Looks/MDL/Shader")
    shader.CreateIdAttr("Test_MDL")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.2, 0.4, 0.6)
    )
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    records, approximations = _materials(stage)
    support = material_support(records, [])
    assert (
        support["MDL_only_materials"] == ["/Looks/MDL"]
        and support["warnings"]
        and support["scoring"] is False
    )
    assert len(approximations) == 1
    preview = material.ComputeSurfaceSource()[0]
    assert preview.GetIdAttr().Get() == "UsdPreviewSurface"
    assert preview.GetInput("diffuseColor").Get() == Gf.Vec3f(0.2, 0.4, 0.6)


def test_comparison_alignment_uses_the_evaluators_rigid_pose_after_unit_normalisation(
    tmp_path,
):
    from pxr import Gf, Usd, UsdGeom
    from simready_bench.storm import _stage

    path = tmp_path / "candidate.usda"
    original = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(original, 0.01)
    UsdGeom.SetStageUpAxis(original, UsdGeom.Tokens.y)
    UsdGeom.Cube.Define(original, "/Shape")
    original.GetRootLayer().Save()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    yaw = math.radians(70)
    rotation = [
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1],
    ]
    translation = [0.03, -0.04, 0.05]
    stage, root = _stage(
        path,
        alignment={"rotation": rotation, "translation_m": translation, "scale": 1.0},
    )
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(root.GetPrim())
    source = Gf.Vec3d(5, 8, 12)
    canonical = [0.05, -0.12, 0.08]
    expected = [
        sum(rotation[row][col] * canonical[col] for col in range(3)) + translation[row]
        for row in range(3)
    ]
    assert list(matrix.Transform(source)) == pytest.approx(expected)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "alignment",
    [
        {
            "rotation": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "translation_m": [0, 0, 0],
            "scale": 1,
        },
        {
            "rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            "translation_m": [0, 0, 0],
            "scale": 1,
        },
        {
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation_m": [0, 0, float("nan")],
            "scale": 1,
        },
    ],
)
def test_comparison_alignment_rejects_scaling_tilt_and_invalid_coordinates(alignment):
    from simready_bench.storm import _alignment_matrix

    with pytest.raises(ValueError):
        _alignment_matrix(alignment)


def test_reset_transform_stacks_receive_the_same_whole_object_alignment(tmp_path):
    from pxr import Gf, Usd, UsdGeom
    from simready_bench.storm import _alignment_matrix, _stage

    path = tmp_path / "reset-stack.usda"
    original = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(original, 0.01)
    UsdGeom.SetStageUpAxis(original, UsdGeom.Tokens.y)
    parent = UsdGeom.Xform.Define(original, "/Parent")
    parent.AddTranslateOp().Set((12, 23, 34))
    for name, offset, reset in (
        ("Ordinary", (1, 2, 3), False),
        ("Reset", (4, 5, 6), True),
        ("Reset/Nested", (7, 8, 9), True),
    ):
        cube = UsdGeom.Cube.Define(original, "/Parent/" + name)
        cube.AddTranslateOp().Set(offset)
        cube.SetResetXformStack(reset)
    original.GetRootLayer().Save()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    alignment = {
        "rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        "translation_m": [10, 20, 30],
        "scale": 1,
    }
    canonical = (
        Gf.Matrix4d().SetScale(0.01)
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90))
        * _alignment_matrix(alignment)
    )
    staged, _ = _stage(path, alignment=alignment)
    old_cache, new_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    point = Gf.Vec3d(1, 2, 3)
    for name in ("Ordinary", "Reset", "Reset/Nested"):
        old = original.GetPrimAtPath("/Parent/" + name)
        new = staged.GetPrimAtPath("/IllustrationAsset/Parent/" + name)
        expected = (old_cache.GetLocalToWorldTransform(old) * canonical).Transform(
            point
        )
        actual = new_cache.GetLocalToWorldTransform(new).Transform(point)
        assert list(actual) == pytest.approx(list(expected))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_comparison_images_require_the_evaluators_fitted_pose():
    from types import SimpleNamespace
    from simready_bench.captures import render_final

    result = render_final(
        {"settings": {"illustrations": {"renderer": "storm-llvmpipe"}}},
        {},
        {},
        SimpleNamespace(valid=True),
        None,
    )
    assert result["status"] == "unavailable"
    assert result["images"] == []
    assert result["warnings"] == ["Whole-object comparison alignment is unavailable."]
