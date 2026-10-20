from pathlib import Path
import hashlib
import numpy as np

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    import pytest

    pytest.skip(
        "Full OpenUSD CPU runtime required; exercised separately in native Daytona",
        allow_module_level=True,
    )
from simready_bench.fidelity import compare, settings
from simready_bench.usd_geometry import observe
from test_fidelity import box


def write(
    path,
    *,
    extent=(1, 2, 3),
    name="Body",
    retessellated=False,
    units=1.0,
    up="Z",
    joint=True,
    material=True,
):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, up)
    body = UsdGeom.Xform.Define(stage, "/" + name)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    triangles = box(extent)
    if retessellated:
        triangles = np.array(
            [[a, b, (a + b + c) / 3] for a, b, c in triangles]
            + [[b, c, (a + b + c) / 3] for a, b, c in triangles]
            + [[c, a, (a + b + c) / 3] for a, b, c in triangles]
        )
    points = triangles.reshape(-1, 3)
    if up == "Y":
        points = points[:, [0, 2, 1]] * np.array([1, 1, -1])
    points = points / units
    mesh = UsdGeom.Mesh.Define(stage, "/" + name + "/Surface")
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(list(range(len(points))))
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    if material:
        mat = UsdShade.Material.Define(stage, "/Looks/Renamable")
        physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        physics.CreateStaticFrictionAttr(0.7)
        physics.CreateDynamicFrictionAttr(0.4)
        physics.CreateRestitutionAttr(0.2)
        physics.CreateDensityAttr(1234 * units**3)
        shader = UsdShade.Shader.Define(stage, "/Looks/Renamable/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.4, 0.3, 0.2)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        binding.Bind(mat)
        binding.Bind(mat, materialPurpose="physics")
    if joint:
        hinge = UsdPhysics.RevoluteJoint.Define(stage, "/AnyJointName")
        hinge.CreateBody0Rel().SetTargets([body.GetPath()])
        hinge.CreateAxisAttr("X")
        hinge.CreateLowerLimitAttr(0)
        hinge.CreateUpperLimitAttr(90)
    stage.SetDefaultPrim(body.GetPrim())
    stage.GetRootLayer().Save()


def test_native_equivalent_geometry_units_names_joints_and_materials(tmp_path):
    source = tmp_path / "source.usda"
    candidate = tmp_path / "equivalent.usda"
    write(source)
    write(candidate, retessellated=True, units=0.01, up="Y", name="Renamed")
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, candidate)
    }
    result = compare(observe(source), observe(candidate), settings({"samples": 256}))
    assert result["components"]["shape"]["score_percent"] == 100, result
    assert result["components"]["joints"]["score_percent"] == 100, result["components"][
        "joints"
    ]
    assert result["components"]["materials"]["score_percent"] == 100, result[
        "components"
    ]["materials"]
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, candidate)
    }


def test_native_valid_cube_missing_joints_and_materials_do_not_pass_reference(tmp_path):
    source = tmp_path / "source.usda"
    candidate = tmp_path / "cube.usda"
    write(source)
    write(candidate, extent=(1, 1, 1), joint=False, material=False)
    assert Usd.Stage.Open(str(candidate))
    reference = observe(source)
    cfg = settings({"samples": 256})
    result = compare(reference, observe(candidate), cfg)
    assert result["components"]["shape"]["score_percent"] < 50, result["components"][
        "shape"
    ]
    assert result["components"]["joints"]["score_percent"] == 0, result["components"][
        "joints"
    ]
    assert result["components"]["materials"]["score_percent"] == 0, result[
        "components"
    ]["materials"]
    assert (
        result["reference_eligibility_sha256"]
        == compare(reference, reference, cfg)["reference_eligibility_sha256"]
    )


def test_native_wrong_joint_anchor_axis_and_limits_are_reported(tmp_path):
    source = tmp_path / "source.usda"
    candidate = tmp_path / "wrong.usda"
    write(source)
    write(candidate)
    stage = Usd.Stage.Open(str(candidate))
    hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/AnyJointName"))
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.5, 0, 0))
    hinge.CreateAxisAttr("Z")
    hinge.CreateUpperLimitAttr(20)
    stage.GetRootLayer().Save()
    result = compare(observe(source), observe(candidate), settings({"samples": 128}))
    checks = {
        row["feature"]: row["passed"]
        for row in result["components"]["joints"]["checks"]
    }
    assert (
        checks["existence_type_connectivity"]
        and not checks["anchors"]
        and not checks["axes"]
        and not checks["limits"]
    ), checks


def test_native_disabled_constraints_and_extra_bodies_lose_credit(tmp_path):
    source = tmp_path / "source.usda"
    candidate = tmp_path / "disabled.usda"
    write(source)
    write(candidate)
    stage = Usd.Stage.Open(str(candidate))
    UsdPhysics.RevoluteJoint(
        stage.GetPrimAtPath("/AnyJointName")
    ).CreateJointEnabledAttr(False)
    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/Body")).CreateKinematicEnabledAttr(
        True
    )
    stage.GetRootLayer().Save()
    result = compare(observe(source), observe(candidate), settings({"samples": 128}))
    assert result["components"]["joints"]["score_percent"] < 100
    assert result["components"]["rigid_parts"]["score_percent"] < 100


def test_dependency_safety_rejects_private_source_reference_before_composition(
    tmp_path,
):
    from simready_bench.engine.package_checks import inspect_package

    package = tmp_path / "candidate"
    package.mkdir()
    (package / "asset.usda").write_text(
        '#usda 1.0\ndef Xform "Cheat" (references = @/reference/private-source.usda@) {}\n'
    )
    result = inspect_package(package, entrypoint="asset.usda")
    assert result["package_ok"] is False
    assert result["composition_attempted"] is False
    assert any(r["code"] == "external_dependency" for r in result["findings"])


def test_native_joint_child_targets_resolve_rigid_owners(tmp_path):
    source = tmp_path / "source.usda"
    candidate = tmp_path / "candidate.usda"
    write(source)
    write(candidate, name="Renamed")
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 30)
    for path, name in ((source, "Body"), (candidate, "Renamed")):
        stage = Usd.Stage.Open(str(path))
        transform = Gf.Matrix4d().SetRotate(rotation)
        transform.SetTranslateOnly(Gf.Vec3d(0.2, 0.1, 0))
        UsdGeom.Xformable(
            stage.GetPrimAtPath("/" + name + "/Surface")
        ).AddTransformOp().Set(transform)
        hinge = UsdPhysics.Joint(stage.GetPrimAtPath("/AnyJointName"))
        if path == source:
            hinge.GetBody0Rel().SetTargets(["/Body/Surface"])
        else:
            hinge.CreateLocalPos0Attr(Gf.Vec3f(0.2, 0.1, 0))
            hinge.CreateLocalRot0Attr(Gf.Quatf(rotation.GetQuat()))
        stage.GetRootLayer().Save()
    reference = observe(source)
    assert reference["joints"][0]["bodies"][0] == "/Body"
    result = compare(reference, observe(candidate), settings({"samples": 128}))
    assert result["components"]["joints"]["score_percent"] == 100, result["components"][
        "joints"
    ]


def test_native_nonplanar_quad_matches_original_corner_fan(tmp_path):
    paths = [tmp_path / "quad.usda", tmp_path / "triangles.usda"]
    for index, path in enumerate(paths):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        mesh = UsdGeom.Mesh.Define(stage, "/Surface")
        mesh.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateFaceVertexCountsAttr([4] if index == 0 else [3, 3])
        mesh.CreateFaceVertexIndicesAttr(
            [0, 1, 2, 3] if index == 0 else [0, 1, 2, 0, 2, 3]
        )
        stage.GetRootLayer().Save()
    result = compare(observe(paths[0]), observe(paths[1]), settings({"samples": 128}))
    assert (
        result["components"]["shape"]["score_percent"] == 100
        and result["components"]["shape"]["coverage_fraction"] == 1
    ), result
    assert any(
        "original-corner fan" in row["reason"]
        for row in result["reference_limitations"]
    )


def test_native_worker_rejects_private_reference_dependency_before_composition(
    tmp_path,
):
    if not Path("/reference").is_dir() or not Path("/candidate").is_dir():
        import pytest

        pytest.skip("Reference and candidate container mounts are required")
    from simready_bench.fidelity_worker import run

    source = Path("/reference/golden.usda")
    candidate = Path("/candidate/asset.usda")
    write(source)
    cfg = settings({"samples": 128})
    request = {
        "operation": "reference",
        "reference": str(source),
        "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "settings": cfg,
    }
    profile = run(request)
    candidate.write_text(
        '#usda 1.0\ndef Xform "Cheat" (references = @/reference/golden.usda@) {}\n'
    )
    result = run(
        {
            **request,
            "operation": "compare",
            "candidate": str(candidate),
            "profile_sha256": profile["profile_sha256"],
        }
    )
    assert result["candidate_invalid"] is True
    assert result["candidate_package_safety"]["composition_attempted"] is False
    for component in result["components"].values():
        if component["eligible"]:
            assert component["score_percent"] == 0, component


def test_native_texture_uv_and_subset_binding_are_cached_without_losing_corners(
    tmp_path,
):
    from PIL import Image
    from unittest.mock import patch

    source = tmp_path / "textured.usda"
    Image.new("RGB", (2, 2), (180, 60, 20)).save(tmp_path / "colour.png")
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    mesh = UsdGeom.Mesh.Define(stage, "/Surface")
    mesh.CreatePointsAttr(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]]
    )
    mesh.CreateFaceVertexCountsAttr([4, 4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 5, 2])
    mesh.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
    )
    uv.Set(
        [[0, 0], [1, 0], [1, 1], [0, 1], [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
    )
    textured = UsdShade.Material.Define(stage, "/Looks/Textured")
    shader = UsdShade.Shader.Define(stage, "/Looks/Textured/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    texture = UsdShade.Shader.Define(stage, "/Looks/Textured/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("colour.png")
    )
    reader = UsdShade.Shader.Define(stage, "/Looks/Textured/Reader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    textured.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(textured)
    green = UsdShade.Material.Define(stage, "/Looks/Green")
    surface = UsdShade.Shader.Define(stage, "/Looks/Green/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0, 1, 0)
    )
    green.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    subset = binding.CreateMaterialBindSubset("Patch", [1])
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(green)
    stage.GetRootLayer().Save()
    flattened = UsdGeom.Primvar.ComputeFlattened
    subsets = UsdShade.MaterialBindingAPI.GetMaterialBindSubsets
    calls = {"uv": 0, "subsets": 0}

    def count_uv(self, *args, **kwargs):
        calls["uv"] += 1
        return flattened(self, *args, **kwargs)

    def count_subsets(self, *args, **kwargs):
        calls["subsets"] += 1
        return subsets(self, *args, **kwargs)

    with (
        patch.object(UsdGeom.Primvar, "ComputeFlattened", count_uv),
        patch.object(
            UsdShade.MaterialBindingAPI, "GetMaterialBindSubsets", count_subsets
        ),
    ):
        result = observe(source)
        list(result["materials"])
    assert calls == {"uv": 1, "subsets": 1}, calls
    assert len(result["triangles"]) == 4
    for triangle, material in zip(result["triangles"][:2], result["materials"][:2]):
        assert np.allclose(material["visual:diffuseColor"]["uv"], triangle[:, :2])
    assert all(
        material["visual:diffuseColor"] == [0, 1, 0]
        for material in result["materials"][2:]
    )
    assert (
        compare(result, result, settings({"samples": 128}))["components"]["materials"][
            "score_percent"
        ]
        == 100
    )
    with patch.object(
        Image, "open", side_effect=OSError("Unreadable fixture texture")
    ) as failed:
        unsupported = observe(source)
        list(unsupported["materials"])
    assert failed.call_count == 1
    assert all("visual:diffuseColor" not in row for row in unsupported["materials"][:2])
    assert any(
        "Unsupported visual shader/texture input" in row["reason"]
        for row in unsupported["eligibility"]["unsupported"]
    )


def test_native_redundant_quad_keeps_original_fan_corner_uv(tmp_path):
    # A,B,B,C: the nonzero USD fan triangle uses the SECOND B corner.
    from PIL import Image

    Image.new("RGB", (2, 2), (180, 60, 20)).save(tmp_path / "colour.png")
    paths = [tmp_path / "quad.usda", tmp_path / "triangle.usda"]
    for i, path in enumerate(paths):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        mesh = UsdGeom.Mesh.Define(stage, "/Surface")
        mesh.CreatePointsAttr([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateFaceVertexCountsAttr([4] if i == 0 else [3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3] if i == 0 else [0, 2, 3])
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
        )
        uv.Set(
            [[0, 0], [0.2, 0.3], [0.8, 0.9], [0, 1]]
            if i == 0
            else [[0, 0], [0.8, 0.9], [0, 1]]
        )
        mat = UsdShade.Material.Define(stage, "/Looks/Material")
        shader = UsdShade.Shader.Define(stage, "/Looks/Material/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        texture = UsdShade.Shader.Define(stage, "/Looks/Material/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("colour.png")
        )
        reader = UsdShade.Shader.Define(stage, "/Looks/Material/Reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        stage.GetRootLayer().Save()
    source = observe(paths[0])
    candidate = observe(paths[1])
    assert (
        source["eligibility"]["supported_prims"]
        == source["eligibility"]["visible_prims"]
        == 1
    )
    assert len(source["triangles"]) == 1
    assert np.array_equal(source["triangles"], candidate["triangles"])
    assert np.allclose(
        source["materials"][0]["visual:diffuseColor"]["uv"],
        [[0, 0], [0.8, 0.9], [0, 1]],
    )
    assert np.array_equal(
        source["materials"][0]["visual:diffuseColor"]["uv"],
        candidate["materials"][0]["visual:diffuseColor"]["uv"],
    )
    assert (
        compare(source, candidate, settings({"samples": 128}))["components"]["shape"][
            "score_percent"
        ]
        == 100
    )


def test_native_thin_warped_quad_projection_keeps_distinct_source_vertices(tmp_path):
    points = [
        [0, 0, 0],
        [0, 0.001, 0.00008],
        [-7e-9, 0.001, 0.00008],
        [-7e-9, 0, -1.5e-8],
    ]
    paths = [tmp_path / "quad.usda", tmp_path / "fan.usda"]
    for i, path in enumerate(paths):
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        mesh = UsdGeom.Mesh.Define(stage, "/Surface")
        mesh.CreatePointsAttr(points)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateFaceVertexCountsAttr([4] if i == 0 else [3, 3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3] if i == 0 else [0, 1, 2, 0, 2, 3])
        stage.GetRootLayer().Save()
    source = observe(paths[0])
    candidate = observe(paths[1])
    assert source["eligibility"]["supported_prims"] == 1
    assert np.array_equal(source["triangles"], candidate["triangles"])
    assert len(source["triangles"]) == 2


def test_native_surface_storage_is_bounded_by_bytes_and_owners_stay_compact(tmp_path):
    source = tmp_path / "surface.usda"
    write(source, material=False)
    import pytest

    with pytest.raises(MemoryError, match="byte budget"):
        observe(source, maximum_surface_bytes=95)
    observed = observe(source)
    assert observed["triangle_bodies"].dtype == object
    assert observed["surface_storage_bytes"] <= 256 * 1024**2


def test_native_materials_are_sampled_lazily(tmp_path):
    from unittest.mock import patch

    source = tmp_path / "surface.usda"
    write(source)
    original = UsdShade.MaterialBindingAPI.ComputeBoundMaterial
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    with patch.object(UsdShade.MaterialBindingAPI, "ComputeBoundMaterial", counted):
        observed = observe(source)
        assert calls == []
        assert observed["materials"][0]["visual:assignment"] is True
        count = len(calls)
        assert observed["materials"][1]["visual:assignment"] is True
        assert len(calls) == count
