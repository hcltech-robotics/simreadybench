from pathlib import Path
import hashlib
import numpy as np
import pytest

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


@pytest.mark.parametrize("endpoint", ["body0", "body1"])
@pytest.mark.parametrize("joint_kind", ["revolute", "fixed"])
@pytest.mark.parametrize("malformation", ["attribute", "multiple", "unresolved"])
def test_malformed_joint_relationship_preserves_independent_measurements(
    tmp_path, endpoint, joint_kind, malformation
):
    source, candidate = tmp_path / "source.usda", tmp_path / "candidate.usda"
    write(source)
    write(candidate)
    stage = Usd.Stage.Open(str(candidate))
    prim = stage.GetPrimAtPath("/AnyJointName")
    if joint_kind == "fixed":
        prim.SetTypeName("PhysicsFixedJoint")
    name = "physics:" + endpoint
    prim.RemoveProperty(name)
    if malformation == "attribute":
        prim.CreateAttribute(name, Sdf.ValueTypeNames.String).Set("/Body")
    elif malformation == "multiple":
        prim.CreateRelationship(name).SetTargets(["/Body", "/Body/Surface"])
    else:
        prim.CreateRelationship(name).SetTargets(["/DoesNotExist"])
    stage.GetRootLayer().Save()
    before = hashlib.sha256(candidate.read_bytes()).hexdigest()
    reference, actual = observe(source), observe(candidate)
    assert actual["joints"] == []
    unobserved = actual["kinematic_observation"]
    assert unobserved["status"] == "unknown"
    assert unobserved["unobserved_joints"][0]["path"] == "/AnyJointName"
    assert all(
        joint.get(field) is None
        for joint in unobserved["unobserved_joints"]
        for field in ("bodies", "anchors", "axes", "frames", "motion_rms_m")
    )
    assert any(
        row.get("kinematics") is True and row["geometry"] is False
        for row in actual["eligibility"]["unsupported"]
    )
    for malformed_reference, first, second in (
        (False, reference, actual), (True, actual, reference)
    ):
        result = compare(first, second, settings({"samples": 128}))
        assert result["components"]["shape"]["score_percent"] == 100
        assert result["components"]["materials"]["score_percent"] == 100
        for component in ("joints", "kinematics"):
            measured = result["components"][component]
            assert measured["status"] == ("unknown" if malformed_reference else "measured")
            assert measured["score_percent"] == (None if malformed_reference else 0)
            assert measured["earned"] == (None if malformed_reference else 0)
            assert measured["lower_bound_percent"] == 0
            assert measured["upper_bound_percent"] == (100 if malformed_reference else 0)
            assert measured["reason"] == (
                "unsupported_reference_joint_connectivity" if malformed_reference
                else "candidate_malformed_joint_connectivity"
            )
        group = "fixed_constraints" if joint_kind == "fixed" else "moving_joints"
        assert result["components"]["kinematics"]["groups"][group]["status"] == (
            "unknown" if malformed_reference else "measured"
        )
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == before


def test_absent_joint_endpoint_remains_a_valid_world_attachment(tmp_path):
    path = tmp_path / "world-attached.usda"
    write(path)
    observation = observe(path)
    assert "kinematic_observation" not in observation
    assert observation["joints"][0]["bodies"] == ["/Body", None]
    result = compare(observation, observation, settings({"samples": 128}))
    assert result["components"]["joints"]["score_percent"] == 100
    assert result["components"]["kinematics"]["score_percent"] == 100


def test_candidate_ambiguous_joint_body_assignment_fails_without_motion_inference(
    tmp_path, monkeypatch
):
    from simready_bench import fidelity

    path = tmp_path / "ambiguous.usda"
    write(path)
    observation = observe(path)
    match = fidelity._body_match

    def ambiguous(reference, candidate, scale):
        mapping, metadata = match(reference, candidate, scale)
        metadata["ambiguous_reference_bodies"] = ["/Body"]
        return mapping, metadata

    monkeypatch.setattr(fidelity, "_body_match", ambiguous)
    result = compare(observation, observation, settings({"samples": 128}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["materials"]["score_percent"] == 100
    for name in ("joints", "kinematics"):
        assert result["components"][name]["status"] == "measured"
        assert result["components"][name]["score_percent"] == 0
    measured = result["components"]["kinematics"]["joints"][0]
    assert measured["reason"] == "candidate_ambiguous_correspondence"
    assert measured["motion_rms_m"] is None
    assert measured["candidate_joint"] is None
    assert measured.get("anchors") is None
    assert measured.get("axes") is None


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


@pytest.mark.parametrize(
    "missing", [None, "file", "st", "varname", "unconnected_st", "constant_st"]
)
def test_native_missing_texture_inputs_preserve_geometry_and_joint_metrics(
    tmp_path, missing
):
    from PIL import Image

    Image.new("RGB", (2, 2), (180, 60, 20)).save(tmp_path / "colour.png")
    paths = [tmp_path / "reference.usda", tmp_path / "candidate.usda"]
    for index, path in enumerate(paths):
        write(path)
        stage = Usd.Stage.Open(str(path))
        mesh = UsdGeom.Mesh.Get(stage, "/Body/Surface")
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying"
        )
        uv.Set([[0, 0], [1, 0], [0, 1]] * 12)
        texture = UsdShade.Shader.Define(stage, "/Looks/Renamable/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("colour.png")
        )
        reader = UsdShade.Shader.Define(stage, "/Looks/Renamable/Reader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )
        shader = UsdShade.Shader.Get(stage, "/Looks/Renamable/Surface")
        shader.GetInput("diffuseColor").ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        if index == 1 and missing:
            if missing == "varname":
                reader.GetPrim().RemoveProperty("inputs:varname")
            elif missing in ("unconnected_st", "constant_st"):
                texture.GetInput("st").DisconnectSource()
                if missing == "constant_st":
                    texture.GetInput("st").Set(Gf.Vec2f(0.25, 0.75))
            else:
                texture.GetPrim().RemoveProperty("inputs:" + missing)
        stage.GetRootLayer().Save()
    reference, candidate = map(observe, paths)
    measured = candidate["materials"][0]
    result = compare(reference, candidate, settings({"samples": 128}))
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["joints"]["score_percent"] == 100
    appearance = result["components"]["materials"]
    if missing is None:
        texture = measured["visual:diffuseColor"]
        assert texture["wrap"] == ["repeat", "repeat"]
        assert texture["sRGB"] is True
        assert np.array_equal(texture["scale"], [1, 1, 1])
        assert np.array_equal(texture["bias"], [0, 0, 0])
        assert appearance["score_percent"] == 100
        assert not candidate["eligibility"]["unsupported"]
    else:
        assert "visual:diffuseColor" not in measured
        assert appearance["features"]["visual:diffuseColor"]["earned"] == 0
        assert appearance["features"]["visual:diffuseColor"]["eligible"] == 128
        assert any(
            row["reason"] == "Unsupported visual shader/texture input: diffuseColor"
            and not row["geometry"]
            for row in candidate["eligibility"]["unsupported"]
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


def test_malformed_surface_retains_geometry_physics_and_kinematic_comparison(tmp_path):
    source = tmp_path / "source.usda"
    candidate = tmp_path / "candidate.usda"
    write(source)
    write(candidate)
    stage = Usd.Stage.Open(str(candidate))
    material = stage.GetPrimAtPath("/Looks/Renamable")
    material.RemoveProperty("outputs:surface")
    material.CreateRelationship("outputs:surface").SetTargets(
        ["/Looks/Renamable/Surface.outputs:surface"]
    )
    stage.GetRootLayer().Save()
    before = candidate.read_bytes()

    reference, observed = observe(source), observe(candidate)
    result = compare(reference, observed, settings({"samples": 128}))
    assert result["status"] == "measured"
    assert result["components"]["shape"]["score_percent"] == 100
    assert result["components"]["joints"]["score_percent"] == 100
    assert result["components"]["kinematics"]["score_percent"] == 100
    features = result["components"]["materials"]["features"]
    assert features["physics:staticFriction"]["earned"] == 128
    assert features["visual:assignment"]["earned"] == 0
    assert result["components"]["materials"]["appearance"]["score_percent"] == 0
    assert any(
        "surface is authored as a relationship" in row["reason"]
        and row["geometry"] is False
        for row in result["candidate_limitations"]
    )
    assert candidate.read_bytes() == before

    unsupported_reference = compare(observed, observed, settings({"samples": 128}))
    appearance = unsupported_reference["components"]["materials"]["appearance"]
    assert appearance["status"] == "not_applicable"
    assert appearance["score_percent"] is None


def test_formal_zero_does_not_suppress_valid_usd_reference_comparison(tmp_path):
    from simready_bench.composites import scores
    from simready_bench.engine.scoring import pinned_profile, score_validation
    from simready_bench.engine.verifier import scoring_policy
    from test_validator_reward_v1 import official_report

    source = tmp_path / "source.usda"
    candidate = tmp_path / "candidate.usda"
    write(source)
    write(candidate)
    formal = score_validation(
        official_report(
            failed=[row["code"] for row in pinned_profile()["mandatory_requirements"]]
        ),
        scoring_policy(),
    )
    assert formal["score_percent"] == 0
    fidelity = compare(observe(source), observe(candidate), settings({"samples": 128}))
    result = scores({"simready": formal, "fidelity": fidelity})
    assert result["formal"]["score_percent"] == 0
    assert result["formal"]["passed"] is False
    assert result["visual"]["score_percent"] == 100
    assert result["kinematic"]["score_percent"] == 100


def test_malformed_universal_surface_preserves_connected_render_context(tmp_path):
    candidate = tmp_path / "candidate.usda"
    write(candidate)
    stage = Usd.Stage.Open(str(candidate))
    material = stage.GetPrimAtPath("/Looks/Renamable")
    material.RemoveProperty("outputs:surface")
    material.CreateRelationship("outputs:surface").SetTargets(
        ["/Looks/Renamable/Surface.outputs:surface"]
    )
    material.CreateAttribute(
        "outputs:mdl:surface", Sdf.ValueTypeNames.Token
    ).SetConnections(["/Looks/Renamable/Surface.outputs:surface"])
    stage.GetRootLayer().Save()
    measured = observe(candidate)["materials"][0]
    assert measured["visual:assignment"] is True
    assert np.isclose(measured["physics:staticFriction"], 0.7)
    assert "visual:diffuseColor" not in measured
