"""Headless mesh preview of the authored stage; no reference images or Blender."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Sdf, UsdGeom, UsdShade


def generate_thumbnail(stage, asset_path):
    triangles = []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        world = cache.GetLocalToWorldTransform(prim)
        points = np.array([tuple(world.Transform(Gf.Vec3d(p))) for p in mesh.GetPointsAttr().Get()])
        indices, counts = mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetFaceVertexCountsAttr().Get()
        color = (0.5, 0.5, 0.5)
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if material:
            shader = UsdShade.Shader.Get(stage, material.GetPath().AppendChild("Shader"))
            if shader and shader.GetInput("diffuseColor"):
                color = tuple(shader.GetInput("diffuseColor").Get())
        offset = 0
        for count in counts:
            face = list(indices[offset:offset+count])
            offset += count
            for i in range(1, count-1):
                triangles.append((points[[face[0], face[i], face[i+1]]], color))
    if not triangles:
        raise ValueError("Cannot render thumbnail for an empty USD stage")
    all_points = np.concatenate([t[0] for t in triangles])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    view = np.array([1.5, -2.0, 1.3]); view /= np.linalg.norm(view)
    right = np.cross(np.array([0., 0., 1.]), view); right /= np.linalg.norm(right)
    up = np.cross(view, right)
    basis = np.column_stack((right, -up, view))
    projected = [(np.dot(points-center, basis), color, points) for points, color in triangles]
    xy = np.concatenate([p[0][:, :2] for p in projected])
    bounds = np.max(np.abs(xy), axis=0)
    scale = 210.0 / (2 * max(float(bounds.max()), 1e-12))
    image = Image.new("RGB", (512, 512), (238, 240, 244))
    draw = ImageDraw.Draw(image)
    light = np.array([-0.4, -0.6, 1.]); light /= np.linalg.norm(light)
    for p, color, points in sorted(projected, key=lambda p: float(p[0][:, 2].mean())):
        normal = np.cross(points[1]-points[0], points[2]-points[0])
        norm = np.linalg.norm(normal)
        shade = 0.3 + 0.7 * abs(float(np.dot(normal / norm, light))) if norm else 0.3
        fill = tuple(round(255 * min(1., max(0., c * shade))) for c in color)
        draw.polygon([tuple(2 * (v[:2] * scale + 128)) for v in p], fill=fill)
    path = Path(asset_path).parent / ".thumbs/256x256" / (Path(asset_path).name + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((256, 256), Image.Resampling.LANCZOS).save(path, format="PNG")
    relative = path.relative_to(Path(asset_path).parent).as_posix()
    stage.GetDefaultPrim().SetAssetInfoByKey("previews:thumbnail", Sdf.AssetPath(relative))
    return path
