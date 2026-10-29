import hashlib
import json
from pathlib import Path

import build123d as bd
import pytest
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

from cad_agent.usd_export import Part, Hinge, export_asset


def test_static_multicomponent(tmp_path):
    source = tmp_path / "reconstruction.py"
    source.write_text("# test reconstruction source\n")
    asset = tmp_path / "submission" / "asset.usd"
    receipt = export_asset(
        [Part("base", bd.Box(100, 60, 10), color=(0.8, 0.1, 0.2)),
         Part("top", bd.Box(20, 30, 40), translation_mm=(30, 0, 25),
              rotation_wxyz=(0.7071067811865476, 0, 0, 0.7071067811865476),
              color=(0.1, 0.2, 0.9))],
        asset, source=source, evidence_dir=tmp_path / "evidence")
    stage = Usd.Stage.Open(str(asset))
    assert stage.GetDefaultPrim().GetPath() == "/Asset"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    bound = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    assert tuple(bound.GetMin()) == pytest.approx((-0.05, -0.03, -0.005), abs=1e-8)
    assert tuple(bound.GetMax()) == pytest.approx((0.05, 0.03, 0.045), abs=1e-8)
    for name, color in [("base", (0.8, 0.1, 0.2)), ("top", (0.1, 0.2, 0.9))]:
        mesh = UsdGeom.Mesh.Get(stage, f"/Asset/{name}/Mesh")
        indices, counts, points = mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetPointsAttr().Get()
        assert len(indices) == sum(counts) and min(indices) >= 0 and max(indices) < len(points)
        assert all(n == 3 for n in counts)
        mat, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        shader = UsdShade.Shader.Get(stage, mat.GetPath().AppendChild("Shader"))
        assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx(color)
    assert receipt["asset_sha256"] == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(Path(p["step_path"]).exists() for p in receipt["parts"])
    assert json.loads((tmp_path / "evidence" / "generation.json").read_text())["kind"] == "cad_generation"


def test_hinge_local_frames_and_physics(tmp_path):
    source = tmp_path / "hinge.py"
    source.write_text("# articulated reconstruction\n")
    q = (0.7071067811865476, 0, 0, 0.7071067811865476)
    path = tmp_path / "submission" / "asset.usd"
    export_asset([
        Part("base", bd.Box(80, 40, 10), rigid=True, kinematic=True, mass_kg=1),
        Part("lid", bd.Box(80, 40, 4), translation_mm=(0, 20, 30),
             rotation_wxyz=q, rigid=True, mass_kg=0.2),
    ], path, hinges=[Hinge("lid_hinge", "base", "lid", (0, 20, 5), "X", (-10, 110))],
        source=source, evidence_dir=tmp_path / "evidence")
    stage = Usd.Stage.Open(str(path))
    joint = UsdPhysics.RevoluteJoint.Get(stage, "/Asset/Joints/lid_hinge")
    assert joint.GetBody0Rel().GetTargets() == ["/Asset/base"]
    assert joint.GetBody1Rel().GetTargets() == ["/Asset/lid"]
    assert joint.GetAxisAttr().Get() == "X"
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == (-10, 110)
    cache = UsdGeom.XformCache()
    for i, name in enumerate(("base", "lid")):
        body = stage.GetPrimAtPath(f"/Asset/{name}")
        assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        assert body.HasAPI(UsdPhysics.MassAPI)
        assert stage.GetPrimAtPath(f"/Asset/{name}/Mesh").HasAPI(UsdPhysics.CollisionAPI)
        anchor = (joint.GetLocalPos0Attr() if i == 0 else joint.GetLocalPos1Attr()).Get()
        frame = (joint.GetLocalRot0Attr() if i == 0 else joint.GetLocalRot1Attr()).Get()
        world = cache.GetLocalToWorldTransform(body)
        assert tuple(world.Transform(Gf.Vec3d(anchor))) == pytest.approx((0, 0.02, 0.005), abs=1e-8)
        world_axis = world.TransformDir(Gf.Quatd(frame).Transform(Gf.Vec3d(1, 0, 0)))
        assert tuple(world_axis) == pytest.approx((1, 0, 0), abs=1e-6)


def test_rejects_bad_inputs(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("# source")
    kwargs = dict(source=source, evidence_dir=tmp_path / "evidence")
    with pytest.raises(ValueError, match="name"):
        export_asset([Part("../oops", bd.Box(1, 1, 1))], tmp_path / "submission/asset.usd", **kwargs)
    with pytest.raises(ValueError, match="rigid"):
        export_asset([Part("a", bd.Box(1, 1, 1)), Part("b", bd.Box(1, 1, 1))],
                     tmp_path / "submission/asset.usd", hinges=[Hinge("h", "a", "b", (0, 0, 0))], **kwargs)
