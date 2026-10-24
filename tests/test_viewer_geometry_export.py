"""Geometry, provenance and privacy contract for browser-only shape exports."""

import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom

spec = importlib.util.spec_from_file_location(
    "viewer_export", Path(__file__).resolve().parents[1] / "scripts/export_viewer_geometry.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def read_glb(path):
    data = path.read_bytes()
    magic, version, size = struct.unpack("<4sII", data[:12])
    assert (magic, version, size) == (b"glTF", 2, len(data))
    length, kind = struct.unpack("<I4s", data[12:20])
    assert kind == b"JSON"
    document = json.loads(data[20:20 + length])
    offset = 20 + length
    binary_length, kind = struct.unpack("<I4s", data[offset:offset + 8])
    assert kind == b"BIN\x00"
    binary = data[offset + 8:offset + 8 + binary_length]
    return document, binary


def test_composed_instances_units_visibility_alignment_and_private_names(tmp_path):
    source = tmp_path / "private-source.usda"
    stage = Usd.Stage.CreateNew(str(source))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, .01)
    stage.CreateClassPrim("/SensitivePrototype")
    cube = UsdGeom.Cube.Define(stage, "/SensitivePrototype/PrivateName")
    cube.GetSizeAttr().Set(20)
    cube.GetPrim().SetCustomData({"private_instruction": "NEVER_PUBLISH_THIS"})
    for name, x in (("SecretOne", 0), ("SecretTwo", 100)):
        instance = UsdGeom.Xform.Define(stage, "/" + name)
        instance.GetPrim().GetReferences().AddInternalReference("/SensitivePrototype")
        instance.GetPrim().SetInstanceable(True)
        instance.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
    hidden = UsdGeom.Cube.Define(stage, "/Hidden")
    hidden.GetSizeAttr().Set(100000)
    hidden.GetVisibilityAttr().Set("invisible")
    proxy = UsdGeom.Cube.Define(stage, "/Proxy")
    proxy.GetSizeAttr().Set(100000)
    proxy.GetPurposeAttr().Set("proxy")
    stage.GetRootLayer().Save()
    output = tmp_path / "safe.glb"
    alignment = {"rotation": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                 "translation_m": [2, 3, 4], "scale": 1, "private_extra": "NEVER_PUBLISH_THIS"}
    receipt = exporter.export(source, output, alignment=alignment)
    assert receipt["triangles"] == 24
    assert np.allclose(receipt["bounds_m"], [[1.9, 3.9, -4.1], [2.1, 4.1, -2.9]])
    assert receipt["surface_types"] == {"Cube": 2}
    document, binary = read_glb(output)
    public = json.dumps(document) + json.dumps(receipt)
    for marker in ("SensitivePrototype", "SecretOne", "SecretTwo", "PrivateName", "NEVER_PUBLISH_THIS",
                   "private-source", "private_extra", str(tmp_path)):
        assert marker not in public
    assert "extras" not in document and "uri" not in public
    assert len(binary) == document["buffers"][0]["byteLength"]
    assert len(receipt["source_sha256"]) == len(receipt["glb_sha256"]) == 64


def test_y_up_is_not_rotated_and_mirrored_winding_is_preserved():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Y")
    mesh = UsdGeom.Mesh.Define(stage, "/triangle")
    mesh.GetPointsAttr().Set([(0, 0, 0), (2, 0, 0), (0, 3, 0)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.AddScaleOp().Set(Gf.Vec3f(-1, 1, 1))
    points, faces, receipt = exporter.geometry(stage)
    assert np.allclose(receipt["bounds_m"], [[-2, 0, 0], [0, 3, 0]])
    triangle = points[faces[0]]
    assert np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2] > 0


def test_concave_polygon_holes_and_unused_points_preserve_surface():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Y")
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh.GetPointsAttr().Set([(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0),
                             (1, 2, 0), (0, 2, 0), (999, 999, 999)])
    mesh.GetFaceVertexCountsAttr().Set([6, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 0, 1, 6])
    mesh.GetHoleIndicesAttr().Set([1])
    mesh.GetSubdivisionSchemeAttr().Set("none")
    points, faces, receipt = exporter.geometry(stage)
    triangles = points[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0]), axis=1).sum() / 2
    assert area == pytest.approx(3)
    assert receipt["triangles"] == 4
    assert receipt["bounds_m"] == [[0, 0, 0], [2, 2, 0]]


@pytest.mark.parametrize("kind,axis,expected", [
    ("Sphere", None, [2, 2, 2]), ("Cylinder", "X", [4, 2, 2]),
    ("Cone", "Y", [2, 4, 2]), ("Capsule", "Z", [2, 2, 6]),
    ("Plane", "Z", [2, 4, 0]),
])
def test_analytic_shapes_use_their_authored_dimensions(kind, axis, expected):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Y")
    schema = getattr(UsdGeom, kind).Define(stage, "/surface")
    if kind == "Plane":
        schema.GetWidthAttr().Set(2)
        schema.GetLengthAttr().Set(4)
    else:
        schema.GetRadiusAttr().Set(1)
        if kind != "Sphere":
            schema.GetHeightAttr().Set(4)
    if axis:
        schema.GetAxisAttr().Set(axis)
    points, faces, receipt = exporter.geometry(stage)
    assert np.allclose(np.ptp(points, axis=0), expected)
    assert len(faces) > 0 and receipt["conversion_notes"]["analytic_primitives_tessellated"] == 1


def test_incomplete_or_unsupported_geometry_fails_instead_of_showing_a_substitute(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    cube = UsdGeom.Cube.Define(stage, "/valid")
    stage.OverridePrim("/missing").GetReferences().AddReference(str(tmp_path / "missing.usda"))
    with pytest.raises(ValueError, match="composition"):
        exporter.geometry(stage)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/valid")
    UsdGeom.PointInstancer.Define(stage, "/unsupported")
    with pytest.raises(ValueError, match="PointInstancer"):
        exporter.geometry(stage)
    with pytest.raises(ValueError, match="No nonempty"):
        exporter.geometry(Usd.Stage.CreateInMemory())


def test_scaling_alignment_and_overwriting_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="without scaling"):
        exporter._alignment({"rotation": np.eye(3), "translation_m": [0, 0, 0], "scale": 2})
    output = tmp_path / "existing.glb"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        exporter.export(tmp_path / "irrelevant.usda", output)
    assert output.read_bytes() == b"existing"
