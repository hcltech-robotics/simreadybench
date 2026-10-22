"""Executed with the full-imaging Python in the actual Daytona CPU image."""

import hashlib
import math
import pytest


@pytest.mark.parametrize("composition", ["sublayer", "reference"])
@pytest.mark.parametrize("endpoint", ["body0", "body1"])
def test_malformed_joint_attributes_are_omitted_only_from_rendering_layers(
    tmp_path, composition, endpoint
):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from PIL import Image
    from simready_bench.storm import _stage

    layer_path, path = tmp_path / "body.usda", tmp_path / "asset.usda"
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (2, 2), "orange").save(texture_path)
    layer = Usd.Stage.CreateNew(str(layer_path))
    body = UsdGeom.Xform.Define(layer, "/Body")
    cube = UsdGeom.Cube.Define(layer, "/Body/Cube")
    cube.AddTranslateOp().Set((1, 2, 3))
    joint = UsdPhysics.FixedJoint.Define(layer, "/Body/Fixed")
    joint.GetPrim().CreateAttribute("physics:" + endpoint, Sdf.ValueTypeNames.String).Set("/Body")
    material = UsdShade.Material.Define(layer, "/Body/Looks/Material")
    shader = UsdShade.Shader.Define(layer, "/Body/Looks/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(layer, "/Body/Looks/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    layer.SetDefaultPrim(body.GetPrim())
    layer.GetRootLayer().Save()
    source = Usd.Stage.CreateNew(str(path))
    if composition == "sublayer":
        source.GetRootLayer().subLayerPaths = ["body.usda"]
    else:
        root = UsdGeom.Xform.Define(source, "/Body")
        root.GetPrim().GetReferences().AddReference("body.usda")
    source.GetRootLayer().Save()
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, layer_path, texture_path)}
    rendered, _ = _stage(path)
    mesh = rendered.GetPrimAtPath("/IllustrationAsset/Body/Cube")
    assert mesh and mesh.IsA(UsdGeom.Cube)
    assert mesh.GetAttribute("xformOp:translate").Get() == (1, 2, 3)
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    assert material and material.ComputeSurfaceSource()[0].GetIdAttr().Get() == "UsdPreviewSurface"
    texture = UsdShade.Shader.Get(rendered, "/IllustrationAsset/Body/Looks/Material/Texture")
    assert texture.GetInput("file").Get().resolvedPath == str(texture_path)
    joint = rendered.GetPrimAtPath("/IllustrationAsset/Body/Fixed")
    assert joint and joint.IsA(UsdPhysics.FixedJoint)
    assert not joint.GetRelationship("physics:" + endpoint).GetPropertyStack()
    assert not joint.GetAttribute("physics:" + endpoint).GetPropertyStack()
    assert rendered.GetRootLayer().customLayerData["illustration_omitted_joint_properties"] == ["/Body/Fixed.physics:" + endpoint]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before} == before


@pytest.mark.parametrize(
    "variant", ["identity", "defaults", "scale", "rotation", "translation", "connected"]
)
def test_identity_uv_compatibility_preserves_coordinates_and_source(
    tmp_path, variant
):
    from pxr import Gf, Sdf, Usd, UsdShade
    from simready_bench.storm import _identity_texture_transforms, _stage

    path = tmp_path / "identity-uv.usda"
    original = Usd.Stage.CreateNew(str(path))
    material = UsdShade.Material.Define(original, "/Looks/Material")
    reader = UsdShade.Shader.Define(original, "/Looks/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    transform = UsdShade.Shader.Define(original, "/Looks/Material/Mapping")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    transform.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    if variant != "defaults":
        transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(2, 1) if variant == "scale" else Gf.Vec2f(1, 1)
        )
        transform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(
            45 if variant == "rotation" else 0
        )
        transform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(0.25, 0) if variant == "translation" else Gf.Vec2f(0, 0)
        )
    if variant == "connected":
        transform.GetInput("scale").ConnectToSource(reader.ConnectableAPI(), "result")
    for name in ("Albedo", "Normal", "ORM"):
        texture = UsdShade.Shader.Define(
            original, str(material.GetPath()) + "/" + name
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            transform.ConnectableAPI(), "result"
        )
    original.GetRootLayer().Save()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    stage, _ = _stage(path)
    changes = _identity_texture_transforms(stage)
    identity = variant in ("identity", "defaults")
    assert len(changes) == (3 if identity else 0)
    for name in ("Albedo", "Normal", "ORM"):
        texture = UsdShade.Shader.Get(
            stage, "/IllustrationAsset/Looks/Material/" + name
        )
        source = texture.GetInput("st").GetConnectedSource()
        assert source[0].GetPrim().GetName() == ("UV" if identity else "Mapping")
        assert str(source[1]) == "result"
    assert _identity_texture_transforms(stage) == []
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


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
