"""Small build123d → USD bridge. CAD lengths are mm; authored USD uses meters."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import AGENT_ID, VERSION


@dataclass
class Part:
    name: str
    shape: object
    color: tuple = (0.5, 0.5, 0.5)
    translation_mm: tuple = (0, 0, 0)
    rotation_wxyz: tuple = (1, 0, 0, 0)
    rigid: bool = False
    kinematic: bool = False
    mass_kg: float = 1.0


@dataclass
class Hinge:
    name: str
    body0: str
    body1: str
    anchor_world_mm: tuple
    axis: str = "Z"
    limits_deg: tuple = (-180, 180)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _name(value):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) or value in {"Looks", "Joints"}:
        raise ValueError(f"Invalid or reserved part/joint name: {value!r}")


def _finite(values, length):
    if len(values) != length or not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"Expected {length} finite values")


def export_asset(parts, output, *, source, evidence_dir, hinges=(), tolerance_mm=0.1):
    """Export named local CAD solids, explicit rigid transforms and world-space hinges.

    Call from the reconstruction script with source=__file__. STEP intermediates
    and the generation receipt live outside the submission directory. This is
    generation evidence, not a substitute for auditing the native trajectory.
    """
    import build123d as bd

    parts, hinges = list(parts), list(hinges)
    output, source, evidence = Path(output).resolve(), Path(source).resolve(), Path(evidence_dir).resolve()
    if not source.is_file() or not parts:
        raise ValueError("A reconstruction source file and at least one CAD part are required")
    if evidence.is_relative_to(output.parent) or source.is_relative_to(output.parent):
        raise ValueError("Source and evidence must be outside the submission directory")
    if not math.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError("tolerance_mm must be positive")
    names = set()
    quaternions = {}
    for p in parts:
        _name(p.name)
        if p.name in names:
            raise ValueError("Duplicate part name")
        names.add(p.name)
        _finite(p.translation_mm, 3)
        _finite(p.rotation_wxyz, 4)
        _finite(p.color, 3)
        if any(c < 0 or c > 1 for c in p.color):
            raise ValueError("Color channels must be in [0, 1]")
        if abs(sum(v*v for v in p.rotation_wxyz) - 1) > 1e-6:
            raise ValueError("rotation_wxyz must be a unit quaternion")
        if p.kinematic and not p.rigid:
            raise ValueError("Kinematic parts must be rigid")
        if not math.isfinite(p.mass_kg) or p.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if not p.shape.is_valid or p.shape.volume <= 0:
            raise ValueError("A valid solid CAD shape is required")
        quaternions[p.name] = Gf.Quatd(p.rotation_wxyz[0], Gf.Vec3d(*p.rotation_wxyz[1:]))
    by_name = {p.name: p for p in parts}
    joint_names = set()
    for h in hinges:
        _name(h.name)
        if h.name in joint_names or h.body0 == h.body1:
            raise ValueError("Joint names and bodies must be distinct")
        joint_names.add(h.name)
        if any(n not in names or not by_name[n].rigid for n in (h.body0, h.body1)):
            raise ValueError("Hinges require two named rigid parts")
        _finite(h.anchor_world_mm, 3)
        _finite(h.limits_deg, 2)
        if h.axis not in ("X", "Y", "Z") or h.limits_deg[0] > h.limits_deg[1]:
            raise ValueError("Invalid hinge axis or limits")

    output.parent.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if hinges:
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    records = []
    for p in parts:
        body = UsdGeom.Xform.Define(stage, f"/Asset/{p.name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*(v * 0.001 for v in p.translation_mm)))
        body.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(quaternions[p.name])
        vertices, faces = p.shape.tessellate(tolerance_mm)
        points = [Gf.Vec3f(*(float(v) * 0.001 for v in vertex)) for vertex in vertices]
        if not faces or not points:
            raise ValueError("CAD tessellation produced an empty mesh")
        for face in faces:
            if len(set(face)) != 3 or min(face) < 0 or max(face) >= len(points):
                raise ValueError("Invalid CAD mesh topology")
            a, b, c = (points[i] for i in face)
            if Gf.Cross(b-a, c-a).GetLength() <= 1e-15:
                raise ValueError("Degenerate CAD mesh triangle")
        mesh = UsdGeom.Mesh.Define(stage, f"/Asset/{p.name}/Mesh")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr([i for face in faces for i in face])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
        material = UsdShade.Material.Define(stage, f"/Asset/Looks/{p.name}")
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*p.color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        if p.rigid:
            physics = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            physics.CreateKinematicEnabledAttr(p.kinematic)
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(p.mass_kg)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        step = evidence / f"{p.name}.step"
        bd.export_step(p.shape, str(step))
        records.append({"name": p.name, "step_path": str(step), "step_sha256": sha256(step),
                        "vertices": len(points), "triangles": len(faces)})
    for h in hinges:
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Asset/Joints/{h.name}")
        joint.CreateAxisAttr(h.axis)
        joint.CreateLowerLimitAttr(h.limits_deg[0])
        joint.CreateUpperLimitAttr(h.limits_deg[1])
        joint.CreateCollisionEnabledAttr(False)
        for i, name in enumerate((h.body0, h.body1)):
            p = by_name[name]
            inverse = quaternions[name].GetInverse()
            offset = Gf.Vec3d(*(a-b for a, b in zip(h.anchor_world_mm, p.translation_mm))) * 0.001
            getattr(joint, f"CreateBody{i}Rel")().SetTargets([f"/Asset/{name}"])
            getattr(joint, f"CreateLocalPos{i}Attr")().Set(Gf.Vec3f(inverse.Transform(offset)))
            getattr(joint, f"CreateLocalRot{i}Attr")().Set(Gf.Quatf(inverse))
    stage.GetRootLayer().Export(str(output))
    reopened = Usd.Stage.Open(str(output))
    if not reopened or not reopened.GetDefaultPrim():
        raise RuntimeError("Exported USD could not be reopened")
    from .workflow import bundle_identity
    receipt = {"kind": "cad_generation", "agent": AGENT_ID, "version": VERSION,
               "bundle": bundle_identity(), "source_path": str(source), "source_sha256": sha256(source),
               "asset_path": str(output), "asset_sha256": sha256(output),
               "parts": records, "hinges": [h.name for h in hinges], "meters_per_unit": 1.0}
    (evidence / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
