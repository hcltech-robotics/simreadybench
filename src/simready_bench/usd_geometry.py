"""Read composed default-time USD surfaces, joints and effective materials on CPU."""

from __future__ import annotations
import bisect, copy, hashlib, math
from pathlib import Path
import numpy as np
from simready_bench.fidelity import VERSION


class SampledMaterials:
    """Resolve only queried triangle materials, retaining original face corners."""

    def __init__(self, stage, groups, resolve):
        self.stage = stage
        self.groups = groups
        self.ends = [g[1] for g in groups]
        self.resolve = resolve
        self.cache = {}

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self.cache:
            start, _, prim, indices, metadata = self.groups[
                bisect.bisect_right(self.ends, index)
            ]
            row = metadata[index - start]
            corners = row[1:]
            value = self.resolve(prim, int(row[0]), corners, indices[corners])
            if len(self.cache) < 16384:
                self.cache[index] = value
            return value
        return self.cache[index]


def observe(path, *, maximum_triangles=None, maximum_surface_bytes=256 * 1024**2):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    if not stage:
        raise ValueError("Cannot compose USD")
    meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(stage))
    if (
        not math.isfinite(meters)
        or meters <= 0
        or not math.isfinite(kilograms)
        or kilograms <= 0
    ):
        raise ValueError("Positive stage units required")
    up = UsdGeom.GetStageUpAxis(stage)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def canonical(points):
        p = np.asarray(points, float) * meters
        return p[..., [0, 2, 1]] * np.array([1, -1, 1]) if up == "Y" else p

    def world(prim, points):
        p = np.asarray(points, float)
        w = np.asarray(cache.GetLocalToWorldTransform(prim), float)
        return canonical((np.c_[p, np.ones(len(p))] @ w)[:, :3])

    def owner(prim):
        while prim and not prim.IsPseudoRoot():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(prim.GetPath())
            prim = prim.GetParent()
        return "__static_world__"

    triangles = []
    owners = []
    groups = []
    total = 0
    storage_bytes = 0
    bodies = {}
    joints = []
    unsupported = []
    face_hashes = {}
    visible = 0
    supported = 0
    texture_cache = {}
    warned = {}
    uv_cache = {}
    texture_inputs = {}
    binding_faces = {}
    material_templates = {}

    def warning(prim, reason, geometry=False, face=None):
        key = (str(prim.GetPath()), reason, geometry)
        if key not in warned:
            warned[key] = {"prim": key[0], "reason": reason, "geometry": geometry}
            unsupported.append(warned[key])
        if face is not None:
            row = warned[key]
            row["face_count"] = row.get("face_count", 0) + 1
            examples = row.setdefault("example_faces", [])
            if len(examples) < 16:
                examples.append(int(face))
            face_hashes.setdefault(key, hashlib.sha256()).update(
                int(face).to_bytes(8, "little")
            )

    def texture_value(inp, prim, face, corners, vertices):
        input_key = str(inp.GetAttr().GetPath())
        if input_key not in texture_inputs:
            try:
                texture_inputs[input_key] = texture_descriptor(inp)
            except (OSError, ValueError, RuntimeError, IndexError):
                texture_inputs[input_key] = None
        descriptor = texture_inputs[input_key]
        if descriptor is None:
            return None
        name, template = descriptor
        key = (str(prim.GetPath()), name)
        if key not in uv_cache:
            uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar(name)
            values = uv.ComputeFlattened() if uv else None
            uv_cache[key] = (
                None
                if values is None
                else (np.asarray(values, float), str(uv.GetInterpolation()))
            )
        cached = uv_cache[key]
        if cached is None:
            return None
        values, interpolation = cached
        if interpolation == "faceVarying":
            coords = values[corners]
        elif interpolation in ("vertex", "varying"):
            coords = values[vertices]
        elif interpolation == "uniform":
            coords = np.repeat(values[face][None], 3, axis=0)
        elif interpolation == "constant":
            coords = np.repeat(values[0][None], 3, axis=0)
        else:
            return None
        if coords.shape != (3, 2) or not np.isfinite(coords).all():
            return None
        return {**template, "uv": coords}

    def texture_descriptor(inp):
        source = inp.GetConnectedSource()
        if not source:
            return None
        node = UsdShade.Shader(source[0].GetPrim())
        channel = str(source[1])
        if node.GetIdAttr().Get() != "UsdUVTexture":
            return None
        asset = node.GetInput("file").Get()
        resolved = getattr(asset, "resolvedPath", "")
        if not resolved:
            return None
        file = Path(resolved)
        source = Path(path).absolute()
        allowed = (
            Path("/reference")
            if source.is_relative_to("/reference")
            else Path("/candidate")
            if source.is_relative_to("/candidate")
            else source.parent
        )
        if not file.is_relative_to(allowed):
            return None
        st = node.GetInput("st").GetConnectedSource()
        if not st:
            return None
        reader = UsdShade.Shader(st[0].GetPrim())
        if reader.GetIdAttr().Get() != "UsdPrimvarReader_float2":
            return None
        name = str(reader.GetInput("varname").Get())
        colour_space = str(node.GetInput("sourceColorSpace").Get() or "auto")
        cache_key = (str(file), colour_space)
        if cache_key not in texture_cache:
            from PIL import Image

            if file.stat().st_size > 128 * 1024**2:
                return None
            with Image.open(file) as image:
                if image.width * image.height > 16 * 1024**2:
                    return None
                pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if (
                pixels.nbytes + sum(value.nbytes for value in texture_cache.values())
                > 512 * 1024**2
            ):
                return None
            texture_cache[cache_key] = pixels
        channels = {"rgb": [0, 1, 2], "r": [0], "g": [1], "b": [2], "a": [3]}.get(
            channel
        )
        if channels is None:
            return None
        wrap = [
            str(node.GetInput("wrap" + axis).Get() or "repeat") for axis in ("S", "T")
        ]
        if any(w not in ("repeat", "clamp") for w in wrap):
            return None
        scale = node.GetInput("scale").Get()
        bias = node.GetInput("bias").Get()
        return name, {
            "texture": texture_cache[cache_key],
            "channels": channels,
            "wrap": wrap,
            "sRGB": colour_space == "sRGB"
            or colour_space == "auto"
            and file.suffix.lower() in (".png", ".jpg", ".jpeg"),
            "scale": np.asarray(scale if scale is not None else [1] * 4)[channels],
            "bias": np.asarray(bias if bias is not None else [0] * 4)[channels],
        }

    def material(prim, face, corners, vertices):
        prim_key = str(prim.GetPath())
        if prim_key not in binding_faces:
            # Resolve subset membership once per mesh, preserving USD subset order.
            faces = {}
            for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
                for index in subset.GetIndicesAttr().Get() or []:
                    faces.setdefault(int(index), subset.GetPrim())
            binding_faces[prim_key] = faces
        target = binding_faces[prim_key].get(face, prim)
        key = (prim_key, str(target.GetPath()))
        if key not in material_templates:
            material_templates[key] = material_template(prim, target)
        base, inputs = material_templates[key]
        result = base.copy()
        for name, inp in inputs:
            try:
                value = texture_value(inp, prim, face, corners, vertices)
            except (OSError, ValueError, RuntimeError, IndexError):
                value = None
            if value is None:
                warning(prim, "Unsupported visual shader/texture input: " + name)
            else:
                result["visual:" + name] = value
        return result

    def material_template(prim, target):
        result = {}
        visual = None
        inputs = []
        binding = UsdShade.MaterialBindingAPI(prim)
        for purpose in ("", "physics"):
            mat, _ = UsdShade.MaterialBindingAPI(target).ComputeBoundMaterial(purpose)
            if not mat and target != prim:
                mat, _ = binding.ComputeBoundMaterial(purpose)
            if not mat:
                continue
            if purpose == "physics" and mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
                result["physics:assignment"] = True
                for attr in mat.GetPrim().GetAttributes():
                    name = attr.GetName()
                    value = attr.Get()
                    if name not in (
                        "physics:staticFriction",
                        "physics:dynamicFriction",
                        "physics:restitution",
                        "physics:density",
                        "physxMaterial:frictionCombineMode",
                        "physxMaterial:restitutionCombineMode",
                    ):
                        continue
                    if not attr.HasAuthoredValueOpinion() or value is None:
                        continue
                    if (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    ):
                        if value < 0 or name == "physics:restitution" and value > 1:
                            warning(
                                prim,
                                "Unusable authored physics material value: " + name,
                            )
                            continue
                        if name == "physics:density":
                            if value <= 0:
                                continue
                            value = float(value) * kilograms / meters**3
                        result[name] = float(value)
                    elif isinstance(value, str):
                        result[name] = value
            if purpose == "":
                visual = mat
        if visual:
            shader = visual.ComputeSurfaceSource()[0]
            if shader or visual.ComputeSurfaceSource("mdl")[0]:
                result["visual:assignment"] = True
            if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
                result["visual:assignment"] = True
                for name in ("diffuseColor", "roughness", "metallic", "opacity"):
                    inp = shader.GetInput(name)
                    if not inp:
                        continue
                    value = inp.Get()
                    if inp.HasConnectedSource():
                        inputs.append((name, inp))
                        continue
                    elif value is not None:
                        value = np.asarray(value, float)
                        if not np.isfinite(value).all():
                            warning(prim, "Nonfinite visual shader input: " + name)
                            continue
                        value = float(value) if value.ndim == 0 else value.tolist()
                    if value is not None:
                        result["visual:" + name] = value
            else:
                warning(
                    prim,
                    "Visual material has no supported universal UsdPreviewSurface; visual properties N/A",
                )
        return result, inputs

    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            enabled = UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get()
            bodies[str(prim.GetPath())] = {
                "position": world(prim, [[0, 0, 0]])[0].tolist(),
                "enabled": enabled,
                "kinematic": UsdPhysics.RigidBodyAPI(prim)
                .GetKinematicEnabledAttr()
                .Get(),
            }
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = []
            anchors = []
            axes = []
            frames = []
            axis_name = prim.GetAttribute("physics:axis").Get()
            axis = {"X": 0, "Y": 1, "Z": 2}.get(str(axis_name))
            for index in (0, 1):
                paths = prim.GetRelationship("physics:body" + str(index)).GetTargets()
                body = stage.GetPrimAtPath(paths[0]) if paths else None
                if paths and not body:
                    raise ValueError("Unresolved joint relationship target")
                # Connectivity belongs to the owning rigid body; the local
                # frame still uses the actual relationship prim's transform.
                resolved = owner(body) if body else "__static_world__"
                targets.append(None if resolved == "__static_world__" else resolved)
                transform = (
                    cache.GetLocalToWorldTransform(body) if body else Gf.Matrix4d(1)
                )
                pos = prim.GetAttribute("physics:localPos" + str(index)).Get()
                rotation = prim.GetAttribute("physics:localRot" + str(index)).Get()
                if pos is None or rotation is None:
                    raise ValueError("Resolved joint frame required")
                anchors.append(
                    canonical([transform.Transform(Gf.Vec3d(*pos))])[0].tolist()
                )
                rot = Gf.Rotation(rotation)
                basis = []
                for unit in np.eye(3):
                    direction = canonical(
                        [transform.TransformDir(rot.TransformDir(Gf.Vec3d(*unit)))]
                    )[0]
                    length = np.linalg.norm(direction)
                    if not math.isfinite(length) or length <= 0:
                        raise ValueError("Nondegenerate joint orientation required")
                    basis.append((direction / length).tolist())
                frames.append(basis)
                if axis is not None:
                    axes.append(basis[axis])
            attrs = {
                a.GetName(): {"value": a.Get(), "authored": a.HasAuthoredValueOpinion()}
                for a in prim.GetAttributes()
                if a.GetName().startswith("physics:")
                and "Limit" in a.GetName()
                or a.GetName() in ("physics:minDistance", "physics:maxDistance")
            }
            limits = _joint_range_limits(
                {"type": prim.GetTypeName(), "attrs": attrs}, meters
            )
            joints.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "bodies": targets,
                    "anchors": anchors,
                    "axes": axes,
                    "frames": frames,
                    "limits": limits,
                    "enabled": joint.GetJointEnabledAttr().Get(),
                }
            )
        image = UsdGeom.Imageable(prim)
        if (
            not image
            or image.ComputeVisibility() == "invisible"
            or str(image.ComputePurpose()) not in ("default", "render", "proxy")
        ):
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            visible += 1
            warning(prim, "PointInstancer expansion unsupported", True)
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        visible += 1
        mesh = UsdGeom.Mesh(prim)
        try:
            if mesh:
                points = np.asarray(mesh.GetPointsAttr().Get(), float)
                counts = np.asarray(
                    mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64
                )
                indices = np.asarray(
                    mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32
                )
                holes = set(mesh.GetHoleIndicesAttr().Get() or [])
                if mesh.GetSubdivisionSchemeAttr().Get() not in ("none", None):
                    warning(
                        prim,
                        "Compared authored polygon control surface; subdivision limit surface is not evaluated",
                    )
            elif prim.IsA(UsdGeom.Cube):
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                points = (
                    np.array(
                        [
                            [x, y, z]
                            for x in (-0.5, 0.5)
                            for y in (-0.5, 0.5)
                            for z in (-0.5, 0.5)
                        ]
                    )
                    * size
                )
                faces = [
                    [0, 1, 3, 2],
                    [4, 6, 7, 5],
                    [0, 4, 5, 1],
                    [2, 3, 7, 6],
                    [0, 2, 6, 4],
                    [1, 5, 7, 3],
                ]
                counts = [4] * 6
                indices = sum(faces, [])
                holes = set()
            else:
                warning(
                    prim,
                    "Unsupported analytic/curve surface: " + prim.GetTypeName(),
                    True,
                )
                continue
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not np.isfinite(points).all()
            ):
                raise ValueError("Finite mesh points required")
            counts = np.asarray(counts, dtype=np.int64)
            indices = np.asarray(indices, dtype=np.int32)
            if (
                counts.sum() != len(indices)
                or np.any(counts < 3)
                or np.any(indices < 0)
                or np.any(indices >= len(points))
            ):
                raise ValueError("Valid mesh topology required")
            capacity = int((counts - 2).sum()) - sum(int(counts[i] - 2) for i in holes)
            # Positions (9 float64), source face/corners (4 int32), owner pointer.
            allocated = capacity * (9 * 8 + 4 * 4 + 8)
            if storage_bytes + allocated > maximum_surface_bytes:
                raise MemoryError("CPU surface byte budget exceeded")
            block = np.empty((capacity, 3, 3), float)
            metadata = np.empty((capacity, 4), np.int32)
            written = 0
            points = world(prim, points)
            offset = 0
            own = owner(prim)
            for face, count in enumerate(counts):
                face_indices = indices[offset : offset + count]
                corners = list(range(offset, offset + count))
                offset += count
                if face in holes:
                    continue
                polygon = points[face_indices]
                # USD fan indices preserve which original corner supplies UVs,
                # including a redundant corner with a distinct primvar value.
                redundant_quad = count == 4 and np.any(
                    np.all(polygon == np.roll(polygon, -1, axis=0), axis=1)
                )
                try:
                    if redundant_quad:
                        local_triangles = [(0, 1, 2), (0, 2, 3)]
                        warning(
                            prim,
                            "Repeated-position quad uses original-corner fan; zero-area triangles contribute no surface",
                            face=face,
                        )
                    else:
                        local_triangles = triangulate_polygon(
                            polygon, nonplanarity_relative=1e-12
                        )
                except ValueError as exc:
                    if "nonplanar" in str(exc):
                        local_triangles = [(0, i, i + 1) for i in range(1, count - 1)]
                        warning(
                            prim,
                            "Nonplanar polygon uses deterministic original-corner fan triangulation; authored-vertex surface approximation",
                            face=face,
                        )
                    elif "degenerate polygon" in str(exc):
                        warning(
                            prim,
                            "Zero-area authored polygon contributes no surface area",
                        )
                        continue
                    else:
                        raise
                for local in local_triangles:
                    vertices = [face_indices[i] for i in local]
                    corner_indices = [corners[i] for i in local]
                    triangle = points[vertices]
                    if (
                        np.linalg.norm(
                            np.cross(
                                triangle[1] - triangle[0], triangle[2] - triangle[0]
                            )
                        )
                        <= 0
                    ):
                        warning(
                            prim,
                            "Zero-area authored triangle contributes no surface area",
                        )
                        continue
                    if (
                        maximum_triangles is not None
                        and total + written >= maximum_triangles
                    ):
                        raise MemoryError("CPU surface resource bound exceeded")
                    block[written] = triangle
                    metadata[written] = [face, *corner_indices]
                    written += 1
            if written:
                triangles.append(block[:written])
                owners.append(np.full(written, own, dtype=object))
                groups.append(
                    (total, total + written, prim, indices, metadata[:written])
                )
                total += written
                storage_bytes += allocated
                supported += 1
        except (ValueError, TypeError, IndexError, RuntimeError) as exc:
            # An unsupported source prim is frozen as partial coverage. Candidate
            # unsupported geometry cannot earn missing source criteria.
            warning(prim, str(exc), True)
    owners = (
        np.concatenate(owners)
        if len(owners) > 1
        else owners[0]
        if owners
        else np.empty(0, dtype=object)
    )
    for body in set(owners):
        bodies.setdefault(body, {"position": [0, 0, 0]})
    for key, value in face_hashes.items():
        warned[key]["face_indices_sha256"] = value.hexdigest()
        warned[key]["face_indices_encoding"] = "uint64 little-endian"
    triangles = (
        np.concatenate(triangles)
        if len(triangles) > 1
        else triangles[0]
        if triangles
        else np.empty((0, 3, 3))
    )
    if len(bodies) > 256 or len(joints) > 512:
        raise ValueError("CPU body/joint resource bound exceeded")
    return {
        "schema": VERSION,
        "triangles": triangles,
        "triangle_bodies": owners,
        "materials": SampledMaterials(stage, groups, material),
        "surface_storage_bytes": storage_bytes,
        "bodies": bodies,
        "joints": joints,
        "eligibility": {
            "visible_prims": visible,
            "supported_prims": supported,
            "unsupported": unsupported,
            "time": "default",
            "units": "metres/Z-up",
            "source_mass_COM_inertia": "not scored",
        },
    }


def triangulate_polygon(face, *, nonplanarity_relative=1e-3):
    """Ear clipping returns source face-corner indices; unsupported nonplanarity is explicit."""
    p = np.asarray(face, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("finite polygon with at least three corners required")
    scale = max(float(np.linalg.norm(np.ptp(p, axis=0))), 1e-12)
    normal = np.sum(np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]), axis=0)
    length = np.linalg.norm(normal)
    if length <= scale * scale * 1e-12:
        raise ValueError("degenerate polygon")
    if (
        float(np.max(np.abs((p - p[0]) @ (normal / length))))
        > nonplanarity_relative * scale
    ):
        raise ValueError("nonplanar polygon exceeds relative planarity tolerance 0.001")
    if len(p) == 3:
        return [(0, 1, 2)]
    plane = np.delete(p, int(np.argmax(np.abs(normal))), axis=1)
    cross = lambda a, b, c: float(
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    )
    area = sum(
        cross(np.zeros(2), plane[i], plane[(i + 1) % len(plane)])
        for i in range(len(plane))
    )
    sign = 1 if area > 0 else -1
    eps = scale * scale * 1e-12
    remaining = list(range(len(p)))
    triangles = []
    while len(remaining) > 3:
        found = False
        for i, current in enumerate(remaining):
            previous, next_ = remaining[i - 1], remaining[(i + 1) % len(remaining)]
            a, b, c = plane[previous], plane[current], plane[next_]
            if sign * cross(a, b, c) <= eps:
                continue
            if any(
                all(
                    sign * x >= -eps
                    for x in (
                        cross(a, b, plane[j]),
                        cross(b, c, plane[j]),
                        cross(c, a, plane[j]),
                    )
                )
                for j in remaining
                if j not in (previous, current, next_)
            ):
                continue
            triangles.append((previous, current, next_))
            remaining.pop(i)
            found = True
            break
        if not found:
            raise ValueError("polygon cannot be triangulated without changing outline")
    triangles.append(tuple(remaining))
    return triangles


def _joint_range_limits(prim, meters_per_unit):
    """Retain observed USD limit semantics without JSON nonfinite numbers.

    Schema fallbacks are accepted only when the observer actually resolved a
    value. A missing row or blocked Get()==None is not a schema-default hint.
    OpenUSD range low>high means locked; infinite limits can mean free motion.
    """
    kind = prim.get("type")
    interval = kind in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint")
    angular = kind in ("PhysicsRevoluteJoint", "PhysicsSphericalJoint")
    fields = (
        ("physics:lowerLimit", "physics:upperLimit")
        if interval
        else (
            ("physics:coneAngle0Limit", "physics:coneAngle1Limit")
            if kind == "PhysicsSphericalJoint"
            else ("physics:minDistance", "physics:maxDistance")
            if kind == "PhysicsDistanceJoint"
            else ()
        )
    )
    result = {
        "schema": "simready-joint-range-v1",
        "state": "not_applicable" if kind == "PhysicsFixedJoint" else "unknown",
        "units": "radians" if angular else "m",
        "source_units": "degrees" if angular else "stage_length",
        "bounds": {},
    }
    if not fields:
        return result
    values = []
    for field in fields:
        attr = prim.get("attrs", {}).get(field)
        value = attr.get("value") if isinstance(attr, dict) else None
        origin = {
            "kind": "unknown"
            if attr is None
            else "authored"
            if attr.get("authored")
            else "schema_default",
            "sources": copy.deepcopy((attr or {}).get("sources", [])),
            "time_samples": copy.deepcopy((attr or {}).get("time_samples", [])),
        }
        bound = {"state": "unknown", "value": None, "origin": origin}
        if (
            value is None
            or (attr or {}).get("blocked")
            or (attr or {}).get("value_blocked")
        ):
            bound["reason"] = "missing_or_blocked_limit_observation"
            values.append(None)
        elif type(value) not in (int, float) or math.isnan(value):
            bound.update(state="invalid", reason="invalid_limit_observation")
            values.append(None)
        elif interval and math.isinf(value):
            bound.update(
                state="unbounded", direction="negative" if value < 0 else "positive"
            )
            values.append(value)
        elif not interval and value < 0:
            bound.update(state="unbounded")
            values.append(value)
        elif not math.isfinite(value):
            bound.update(state="invalid", reason="invalid_limit_observation")
            values.append(None)
        else:
            bound.update(
                state="finite",
                value=float(value) * (math.pi / 180 if angular else meters_per_unit),
            )
            values.append(value)
        result["bounds"][field] = bound
    states = [row["state"] for row in result["bounds"].values()]
    if "invalid" in states:
        result["state"] = "invalid"
    elif "unknown" in states:
        result["state"] = "unknown"
    elif interval and values[0] > values[1]:
        result["state"] = "locked"
    elif interval and (values[0] == math.inf or values[1] == -math.inf):
        result["state"] = "invalid"
    elif all(state == "unbounded" for state in states):
        result["state"] = "unbounded"
    else:
        result["state"] = "bounded"
    return result
