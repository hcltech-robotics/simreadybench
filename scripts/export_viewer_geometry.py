#!/usr/bin/env python3
"""Export a composed USD's visible default-time geometry to a sanitised GLB.

Install ``usd-core`` and ``numpy`` in an isolated environment. The output is a
neutral-material shape preview, not a replacement for the evaluated USD or its
render. No source names, paths, properties, materials, textures or code enter
the GLB. A separate, allowlisted receipt records hashes and conversion limits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simready_bench.usd_geometry import triangulate_polygon


def _positive(value, label, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"Invalid analytic {label}")
    return value


def _revolve(profile, segments):
    """Tessellate an actual analytic profile, including its cap centres."""
    angles = np.arange(segments) * 2 * math.pi / segments
    vertices = np.asarray(
        [[radius * math.cos(t), radius * math.sin(t), z]
         for radius, z in profile for t in angles], dtype=np.float64
    )
    faces = []
    for row in range(len(profile) - 1):
        for col in range(segments):
            a, b = row * segments + col, row * segments + (col + 1) % segments
            c, d = a + segments, b + segments
            faces.extend(((a, b, d), (a, d, c)))
    return vertices, np.asarray(faces, dtype=np.int64)


def _analytic(prim, segments):
    kind = prim.GetTypeName()
    schema = getattr(UsdGeom, kind)(prim)
    axis = "Z"
    if kind == "Cube":
        size = _positive(schema.GetSizeAttr().Get(), "size")
        vertices = np.asarray([[x, y, z] for x in (-.5, .5)
                               for y in (-.5, .5) for z in (-.5, .5)]) * size
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        faces = np.asarray([(q[0], q[i], q[i + 1]) for q in quads for i in (1, 2)])
    elif kind == "Plane":
        width = _positive(schema.GetWidthAttr().Get(), "width")
        length = _positive(schema.GetLengthAttr().Get(), "length")
        vertices = np.asarray([[-width / 2, -length / 2, 0],
                               [width / 2, -length / 2, 0],
                               [width / 2, length / 2, 0],
                               [-width / 2, length / 2, 0]])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]])
        axis = str(schema.GetAxisAttr().Get())
    else:
        radius = _positive(schema.GetRadiusAttr().Get(), "radius")
        if kind == "Sphere":
            profile = [(radius * math.cos(t), radius * math.sin(t))
                       for t in np.linspace(-math.pi / 2, math.pi / 2, segments // 2 + 1)]
        else:
            height = _positive(schema.GetHeightAttr().Get(), "height", zero=True)
            axis = str(schema.GetAxisAttr().Get())
            if kind == "Cylinder":
                profile = [(0, -height / 2), (radius, -height / 2),
                           (radius, height / 2), (0, height / 2)]
            elif kind == "Cone":
                profile = [(0, -height / 2), (radius, -height / 2), (0, height / 2)]
            elif kind == "Capsule":
                lower = np.linspace(-math.pi / 2, 0, segments // 4 + 1)
                upper = np.linspace(0, math.pi / 2, segments // 4 + 1)
                profile = [(radius * math.cos(t), -height / 2 + radius * math.sin(t))
                           for t in lower]
                profile += [(radius * math.cos(t), height / 2 + radius * math.sin(t))
                            for t in upper]
            else:
                raise ValueError("Unsupported analytic surface")
        vertices, faces = _revolve(profile, segments)
    if axis == "X":
        vertices = vertices[:, [2, 0, 1]]
    elif axis == "Y":
        vertices = vertices[:, [1, 2, 0]]
    elif axis != "Z":
        raise ValueError("Unsupported analytic axis")
    return vertices, faces


def _mesh(prim, warnings):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    holes = set(mesh.GetHoleIndicesAttr().Get() or [])
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or counts.ndim != 1 or indices.ndim != 1 or np.any(counts < 3)
            or counts.sum() != len(indices) or np.any(indices < 0)
            or np.any(indices >= len(points)) or any(i < 0 or i >= len(counts) for i in holes)):
        raise ValueError("Invalid authored mesh topology or positions")
    if mesh.GetSubdivisionSchemeAttr().Get() not in (None, "none"):
        warnings["subdivision_control_meshes"] += 1
    triangles = []
    offset = 0
    for face, count in enumerate(counts):
        face_indices = indices[offset:offset + count]
        offset += count
        if face in holes:
            continue
        if count == 3:
            triangles.append(face_indices)
            continue
        polygon = points[face_indices]
        # Redundant adjacent points are common in authored quads; deleting only
        # repeated positions preserves their surface in this untextured preview.
        keep = np.any(polygon != np.roll(polygon, 1, axis=0), axis=1)
        if not keep.all():
            polygon, face_indices = polygon[keep], face_indices[keep]
        if len(polygon) < 3:
            warnings["degenerate_faces_omitted"] += 1
            continue
        try:
            local = triangulate_polygon(polygon, nonplanarity_relative=1e-6)
        except ValueError as exc:
            if "nonplanar" in str(exc):
                local = [(0, i, i + 1) for i in range(1, len(polygon) - 1)]
                warnings["nonplanar_faces_fan_triangulated"] += 1
            elif "degenerate polygon" in str(exc):
                warnings["degenerate_faces_omitted"] += 1
                continue
            else:
                raise ValueError("Cannot triangulate an authored mesh polygon") from exc
        triangles.extend(face_indices[list(t)] for t in local)
    return points, np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def _alignment(value):
    if value is None:
        return None
    rotation = np.asarray(value["rotation"], dtype=float)
    translation = np.asarray(value["translation_m"], dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-7)
            or float(value.get("scale", 1)) != 1):
        raise ValueError("Alignment must be a finite rigid metre/Z-up transform without scaling")
    return {"rotation": rotation.tolist(), "translation_m": translation.tolist(), "scale": 1.0}


def geometry(stage, *, segments=64, maximum_triangles=2_000_000, alignment=None):
    """Return world-space metres/Y-up positions and triangles, with coverage."""
    if not 8 <= segments <= 256 or segments % 4:
        raise ValueError("Analytic segment count must be a multiple of four from 8 to 256")
    alignment = _alignment(alignment)
    meters = _positive(UsdGeom.GetStageMetersPerUnit(stage), "stage units")
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    if up_axis not in ("Y", "Z"):
        raise ValueError("Unsupported stage up axis")
    if stage.GetCompositionErrors():
        raise ValueError("USD composition has unresolved errors; refusing a partial preview")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices, faces = [], []
    warnings, kinds = Counter(), Counter()
    vertex_count = triangle_count = 0
    supported = {"Cube", "Sphere", "Capsule", "Cylinder", "Cone", "Plane"}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        image = UsdGeom.Imageable(prim)
        if (not image or image.ComputeVisibility() == "invisible"
                or str(image.ComputePurpose()) not in ("default", "render")):
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            raise ValueError("PointInstancer is unsupported; refusing a partial preview")
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if prim.HasAPI("SkelBindingAPI"):
            raise ValueError("Skinned geometry is unsupported; refusing a bind-pose preview")
        kind = prim.GetTypeName()
        if prim.IsA(UsdGeom.Mesh):
            local_points, triangles = _mesh(prim, warnings)
        elif kind in supported:
            local_points, triangles = _analytic(prim, segments)
            warnings["analytic_primitives_tessellated"] += 1
        else:
            # Avoid including an arbitrary source type name in a public receipt.
            raise ValueError("Unsupported visible surface; refusing a partial preview")
        if not len(triangles):
            continue
        world = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        points = (np.c_[local_points, np.ones(len(local_points))] @ world)[:, :3] * meters
        # The benchmark alignment acts in metre/Z-up coordinates, using
        # column-vector rotations. Convert to glTF metre/Y-up only afterwards.
        if up_axis == "Y":
            points = points[:, [0, 2, 1]] * [1, -1, 1]
        if alignment is not None:
            points = points @ np.asarray(alignment["rotation"]).T + alignment["translation_m"]
        points = points[:, [0, 2, 1]] * [1, 1, -1]
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite composed geometry")
        orientation = str(UsdGeom.Gprim(prim).GetOrientationAttr().Get())
        if (np.linalg.det(world[:3, :3]) < 0) != (orientation == "leftHanded"):
            triangles = triangles[:, [0, 2, 1]]
        areas = np.linalg.norm(np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                                       points[triangles[:, 2]] - points[triangles[:, 0]]), axis=1)
        valid = areas > np.finfo(float).tiny
        warnings["degenerate_triangles_omitted"] += int((~valid).sum())
        triangles = triangles[valid]
        if not len(triangles):
            continue
        triangle_count += len(triangles)
        if triangle_count > maximum_triangles:
            raise ValueError("Preview triangle limit exceeded; no decimation was performed")
        # Do not include unused or hole-only vertices in the bounds or payload.
        used, remapped = np.unique(triangles, return_inverse=True)
        vertices.append(points[used])
        faces.append(remapped.reshape(-1, 3) + vertex_count)
        vertex_count += len(used)
        kinds[kind] += 1
    if not faces:
        raise ValueError("No nonempty visible surface geometry")
    points = np.concatenate(vertices).astype("<f4")
    triangles = np.concatenate(faces).astype("<u4")
    if not np.isfinite(points).all():
        raise ValueError("Composed geometry exceeds GLB float32 range")
    bounds = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    if not any(high > low for low, high in zip(*bounds)):
        raise ValueError("Degenerate preview bounds")
    return points, triangles, {
        "vertices": len(points), "triangles": len(triangles), "bounds_m": bounds,
        "surface_types": dict(sorted(kinds.items())),
        "conversion_notes": {key: value for key, value in sorted(warnings.items()) if value},
    }


def write_glb(output, points, triangles):
    """Build an allowlisted, self-contained GLB: no source strings are copied."""
    position_data, index_data = points.tobytes(), triangles.tobytes()
    binary = position_data + index_data
    document = {
        "asset": {"version": "2.0", "generator": "SimReadyBench geometry preview"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1,
                                      "material": 0, "mode": 4}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [.65, .7, .76, 1],
                                                  "metallicFactor": 0, "roughnessFactor": .65},
                       "doubleSided": True}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(position_data),
                         "target": 34962},
                        {"buffer": 0, "byteOffset": len(position_data), "byteLength": len(index_data),
                         "target": 34963}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points),
                       "type": "VEC3", "min": points.min(axis=0).tolist(),
                       "max": points.max(axis=0).tolist()},
                      {"bufferView": 1, "componentType": 5125, "count": triangles.size,
                       "type": "SCALAR"}],
    }
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    payload = (struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(encoded) + 8 + len(binary))
               + struct.pack("<I4s", len(encoded), b"JSON") + encoded
               + struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def export(source, output, *, segments=64, maximum_triangles=2_000_000, alignment=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new path")
    stage = Usd.Stage.Open(str(source))
    if not stage:
        raise ValueError("Cannot open USD stage")
    alignment = _alignment(alignment)
    points, triangles, details = geometry(stage, segments=segments,
                                          maximum_triangles=maximum_triangles, alignment=alignment)
    output.parent.mkdir(parents=True, exist_ok=True)
    sha = write_glb(output, points, triangles)
    return {
        "schema": "simreadybench.geometry-preview.v1", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "glb_sha256": sha, "bytes": output.stat().st_size, "usd_version": ".".join(map(str, Usd.GetVersion())),
        "coordinates": "metres/Y-up", "time": "default", "materials": "neutral",
        "alignment_m_z_up": alignment,
        "composed_layer_sha256": sorted({hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                                         for layer in stage.GetUsedLayers() if layer.realPath}),
        "analytic_segments": segments, "animation": False,
        "limits": "Shape preview; original materials, textures, physics and animation are omitted. Subdivision meshes use authored control surfaces.",
        **details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path, help="Write a public-safe provenance receipt")
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--maximum-triangles", type=int, default=2_000_000)
    parser.add_argument("--alignment", type=Path, help="Recorded rigid rotation/translation JSON in metres/Z-up")
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        parser.error("Receipt already exists; choose a new path")
    receipt = export(args.source, args.output, segments=args.segments,
                     maximum_triangles=args.maximum_triangles,
                     alignment=json.loads(args.alignment.read_text()) if args.alignment else None)
    encoded = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
