"""Small original articulated data for native integration, never NV100 roster data."""

from pathlib import Path


def articulated_box(path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    material = UsdShade.Material.Define(stage, "/Asset/Material")
    shader = UsdShade.Shader.Define(stage, "/Asset/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((.55, .14, .04))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for name, position in (("Base", -.75), ("Lid", .75)):
        body = UsdGeom.Cube.Define(stage, "/Asset/" + name)
        body.CreateSizeAttr(1)
        body.AddTranslateOp().Set(Gf.Vec3d(position, 0, 0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1)
        UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Asset/Hinge")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Asset/Base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Asset/Lid")])
    joint.CreateAxisAttr("Z")
    joint.CreateLocalPos0Attr(Gf.Vec3f(.75, 0, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-.75, 0, 0))
    joint.CreateLowerLimitAttr(0)
    joint.CreateUpperLimitAttr(90)
    stage.GetRootLayer().Save()
    return path


def cameras(size=32):
    import numpy as np

    views = []
    for name, position in (("observation", (5, -5, 4)), ("isometric", (5, 5, 4)),
                           ("front", (0, -6, 0)), ("side", (6, 0, 0)), ("top", (0, 0, 6))):
        backward = np.asarray(position, dtype=float)
        backward /= np.linalg.norm(backward)
        up = np.asarray((0, 1, 0) if name == "top" else (0, 0, 1), dtype=float)
        right = np.cross(up, backward)
        right /= np.linalg.norm(right)
        up = np.cross(backward, right)
        matrix = np.eye(4)
        matrix[:3, :3] = np.stack((right, up, backward), axis=1)
        matrix[:3, 3] = position
        views.append({"id": name, "projection": "orthographic",
                      "camera_to_world": matrix.tolist(), "orthographic_scale": 4,
                      "clip_start": .01, "clip_end": 100})
    return {"schema": "simready-cameras-v1", "resolution": [size, size],
            "observed_camera_id": "observation", "observation": views[0], "views": views[1:]}
