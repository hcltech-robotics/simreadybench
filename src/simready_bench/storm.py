"""Full OpenUSD imaging worker. Runs only in the isolated CPU illustration image."""

from __future__ import annotations
import contextlib, hashlib, json, math, os, subprocess, sys, tempfile, time
from pathlib import Path
from simready_bench.captures import (
    check_software_GL,
    material_support,
    validate_settings,
)


def _display_ready():
    return Path("/tmp/.X11-unix/X97").exists()


@contextlib.contextmanager
def virtual_display():
    # Use a private Unix socket with TCP disabled.
    with tempfile.TemporaryDirectory(prefix="storm-runtime-") as temporary:
        previous = {
            k: os.environ.get(k)
            for k in ("DISPLAY", "XDG_RUNTIME_DIR", "XDG_CACHE_HOME", "HOME")
        }
        os.environ.update(
            DISPLAY=":97",
            XDG_RUNTIME_DIR=temporary,
            XDG_CACHE_HOME=temporary + "/cache",
            HOME=temporary,
        )
        Path(temporary + "/cache").mkdir(mode=0o700)
        with (Path(temporary) / "Xvfb.log").open("w+") as log:
            server = subprocess.Popen(
                [
                    "Xvfb",
                    ":97",
                    "-screen",
                    "0",
                    "1024x768x24",
                    "-nolisten",
                    "tcp",
                    "-ac",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                until = time.monotonic() + 15
                while not _display_ready():
                    if server.poll() is not None or time.monotonic() > until:
                        log.seek(0)
                        raise RuntimeError(
                            "Owned X server unavailable: " + log.read()[-4000:]
                        )
                    time.sleep(0.1)
                yield
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


def _GL_context():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
    from PySide6.QtOpenGL import (
        QOpenGLFramebufferObject,
        QOpenGLFramebufferObjectFormat,
    )
    from PySide6.QtCore import QSize
    from OpenGL import GL

    app = QApplication(["simready-Storm"])
    fmt = QSurfaceFormat()
    fmt.setSamples(0)
    fmt.setVersion(4, 5)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    fmt.setOption(QSurfaceFormat.DeprecatedFunctions)
    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    context = QOpenGLContext()
    context.setFormat(fmt)
    if not context.create() or not context.makeCurrent(surface):
        raise RuntimeError("Headless OpenGL context unavailable")
    fbo = QOpenGLFramebufferObject(QSize(1, 1), QOpenGLFramebufferObjectFormat())
    fbo.bind()
    read = lambda token: (GL.glGetString(token) or b"").decode("utf-8", "replace")
    proof = check_software_GL(
        read(GL.GL_VENDOR), read(GL.GL_RENDERER), read(GL.GL_VERSION)
    )
    proof["profile_mask"] = int(GL.glGetIntegerv(GL.GL_CONTEXT_PROFILE_MASK))
    print(json.dumps({"active_GL": proof}), flush=True)
    return (app, surface, context, fbo), proof


def _visual_composition(original):
    """Remove malformed joint attributes from independent rendering layers only."""
    from pxr import Ar, Sdf, Usd, UsdPhysics, UsdUtils

    omitted = {}
    for prim in original.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        for name in ("physics:body0", "physics:body1"):
            relationship = prim.GetRelationship(name)
            for spec in relationship.GetPropertyStack() if relationship else []:
                if isinstance(spec, Sdf.AttributeSpec):
                    omitted.setdefault(spec.layer.identifier, set()).add(str(spec.path))
    if not omitted:
        return original, []
    layers = list(original.GetUsedLayers())
    copies = {layer.identifier: Sdf.Layer.CreateAnonymous("illustration.usda") for layer in layers}
    identities = {
        identity: copies[layer.identifier]
        for layer in layers
        for identity in (layer.identifier, layer.realPath)
        if identity
    }
    records = []
    for layer in layers:
        copy = copies[layer.identifier]
        copy.TransferContent(layer)
        for path in sorted(omitted.get(layer.identifier, ())):
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(path, Sdf.Path.emptyPath)
            if not copy.Apply(edit):
                raise ValueError("Cannot isolate malformed joint property for illustrations")
            records.append({"property": path, "method": "omit malformed joint attribute from illustration copy"})

        def asset_path(value):
            anchored = Sdf.ComputeAssetPathRelativeToLayer(layer, value)
            resolved = str(Ar.GetResolver().Resolve(anchored))
            target = identities.get(anchored) or identities.get(resolved)
            return target.identifier if target else anchored

        UsdUtils.ModifyAssetPaths(copy, asset_path)
    stage = Usd.Stage.Open(
        copies[original.GetRootLayer().identifier],
        copies[original.GetSessionLayer().identifier],
        original.GetPathResolverContext(),
    )
    if not stage:
        raise ValueError("Cannot compose independent illustration layers")
    return stage, records


def _stage(path, *, alignment=None):
    from pxr import Gf, Usd, UsdGeom, UsdLux

    original = Usd.Stage.Open(str(path))
    if not original:
        raise ValueError("USD stage unavailable")
    # Work on an anonymous composed copy. Separate per-root references would
    # discard bindings/connections targeting a sibling root such as /Looks.
    visual, omitted = _visual_composition(original)
    stage = Usd.Stage.Open(visual.Flatten())
    if omitted:
        metadata = dict(stage.GetRootLayer().customLayerData)
        metadata["illustration_omitted_joint_properties"] = [item["property"] for item in omitted]
        stage.GetRootLayer().customLayerData = metadata
    roots = [p.GetPath() for p in stage.GetPseudoRoot().GetChildren()]
    if stage.GetPrimAtPath("/IllustrationAsset"):
        raise ValueError("Illustration staging namespace collision")
    root = UsdGeom.Xform.Define(stage, "/IllustrationAsset")
    for source in roots:
        editor = Usd.NamespaceEditor(stage)
        if (
            not editor.MovePrimAtPath(source, root.GetPath().AppendChild(source.name))
            or not editor.ApplyEdits()
        ):
            raise ValueError("Cannot preserve composed USD namespace for illustrations")
    stage.SetDefaultPrim(root.GetPrim())
    # Inspect/approximate materials in native instances too, in this copy only.
    for _ in range(64):
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances:
            break
        for prim in instances:
            prim.SetInstanceable(False)
    else:
        raise ValueError("Illustration instance nesting exceeds resource bound")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    # Canonical metres/Z-up, same rule for reference and authored candidate.
    meters = UsdGeom.GetStageMetersPerUnit(original)
    transform = Gf.Matrix4d().SetScale(meters)
    if UsdGeom.GetStageUpAxis(original) == UsdGeom.Tokens.y:
        transform = transform * Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), 90)
        )
    if alignment is not None:
        transform = transform * _alignment_matrix(alignment)
    cache = UsdGeom.XformCache()
    resets = [
        (UsdGeom.Xformable(prim), cache.GetLocalToWorldTransform(prim))
        for prim in stage.Traverse()
        if UsdGeom.Xformable(prim) and UsdGeom.Xformable(prim).GetResetXformStack()
    ]
    for xform, world in resets:
        xform.MakeMatrixXform().Set(world * transform)
        xform.SetResetXformStack(True)
    root.AddTransformOp().Set(transform)
    for prim in list(stage.Traverse()):
        if prim.HasAPI(UsdLux.LightAPI):
            prim.SetActive(False)
    return stage, root


def _alignment_matrix(alignment):
    """Apply the evaluator's whole-object pose in canonical metres/Z-up."""
    from pxr import Gf

    rotation, translation = alignment.get("rotation"), alignment.get("translation_m")
    if (
        alignment.get("scale") != 1
        or not isinstance(rotation, list)
        or len(rotation) != 3
    ):
        raise ValueError("Comparison alignment must preserve physical scale")
    if (
        any(not isinstance(row, list) or len(row) != 3 for row in rotation)
        or not isinstance(translation, list)
        or len(translation) != 3
    ):
        raise ValueError("A rigid comparison rotation and translation are required")
    values = [value for row in rotation for value in row] + translation
    if any(
        type(value) not in (int, float) or not math.isfinite(value) for value in values
    ):
        raise ValueError("Comparison alignment must be finite")
    for row in range(3):
        for column in range(3):
            dot = sum(rotation[k][row] * rotation[k][column] for k in range(3))
            if abs(dot - (row == column)) > 1e-7:
                raise ValueError("Comparison alignment must be a rigid rotation")
    expected = (
        (rotation[0][0], -rotation[1][0], 0),
        (rotation[1][0], rotation[0][0], 0),
        (0, 0, 1),
    )
    if any(
        abs(rotation[row][column] - expected[row][column]) > 1e-7
        for row in range(3)
        for column in range(3)
    ):
        raise ValueError("Comparison alignment must preserve the vertical axis")
    return Gf.Matrix4d(
        rotation[0][0],
        rotation[1][0],
        rotation[2][0],
        0,
        rotation[0][1],
        rotation[1][1],
        rotation[2][1],
        0,
        rotation[0][2],
        rotation[1][2],
        rotation[2][2],
        0,
        *translation,
        1,
    )


def _identity_texture_transforms(stage):
    """Bypass identity UV transforms in the anonymous illustration stage.

    Storm can generate references to an undefined transform accessor when
    several textures share a UsdTransform2d node. Removing an identity operation
    preserves the authored coordinates and avoids that shader compilation path.
    """
    from pxr import UsdShade

    changes = []
    for prim in stage.Traverse():
        texture = UsdShade.Shader(prim)
        if not texture or texture.GetIdAttr().Get() != "UsdUVTexture":
            continue
        coordinate = texture.GetInput("st")
        if not coordinate:
            continue
        visited = set()
        for _ in range(64):
            source = coordinate.GetConnectedSource()
            if not source or str(source[1]) != "result":
                break
            transform = UsdShade.Shader(source[0].GetPrim())
            if not transform or transform.GetIdAttr().Get() != "UsdTransform2d":
                break
            path = str(transform.GetPath())
            if path in visited:
                break
            visited.add(path)
            identity = True
            for name, expected in (
                ("rotation", 0.0),
                ("scale", (1.0, 1.0)),
                ("translation", (0.0, 0.0)),
            ):
                parameter = transform.GetInput(name)
                if parameter and parameter.HasConnectedSource():
                    identity = False
                    break
                value = parameter.Get() if parameter else None
                if value is not None and (
                    float(value) != expected
                    if name == "rotation"
                    else tuple(value) != expected
                ):
                    identity = False
                    break
            upstream_input = transform.GetInput("in")
            upstream = upstream_input.GetConnectedSource() if upstream_input else None
            if not identity or not upstream:
                break
            coordinate.ConnectToSource(*upstream)
            changes.append(
                {
                    "texture": str(prim.GetPath()),
                    "transform": path,
                    "method": "identity UsdTransform2d bypass",
                    "coordinates_preserved": True,
                }
            )
    return changes


def _materials(stage):
    from pxr import Ar, Gf, Sdf, Usd, UsdShade

    materials = []
    approximations = []
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdShade.Material):
            continue
        material = UsdShade.Material(prim)
        shaders = [
            UsdShade.Shader(p) for p in Usd.PrimRange(prim) if p.IsA(UsdShade.Shader)
        ]
        ids = [str(s.GetIdAttr().Get() or "") for s in shaders]
        surface = material.ComputeSurfaceSource()[0]
        preview = bool(surface and surface.GetIdAttr().Get() == "UsdPreviewSurface")
        mdl = bool(material.ComputeSurfaceSource("mdl")[0]) or any(
            s.GetSourceAsset("mdl") for s in shaders
        )
        materials.append(
            {
                "path": str(prim.GetPath()),
                "shader_ids": ids,
                "universal_surface": preview,
                "MDL_surface": mdl,
            }
        )
        if preview or not mdl:
            continue
        colour = Gf.Vec3f(0.55, 0.55, 0.55)
        texture = None
        for shader in shaders:
            for inp in shader.GetInputs():
                name = inp.GetBaseName().lower()
                value = inp.Get()
                if "diffuse" not in name and "base_color" not in name:
                    continue
                if isinstance(value, (Gf.Vec3f, Gf.Vec3d)):
                    colour = Gf.Vec3f(*value)
                if isinstance(value, Sdf.AssetPath) and value.path:
                    resolved = value.resolvedPath
                    if not resolved:
                        specs = inp.GetAttr().GetPropertyStack()
                        anchored = (
                            Sdf.ComputeAssetPathRelativeToLayer(
                                specs[0].layer, value.path
                            )
                            if specs
                            else value.path
                        )
                        resolved = str(Ar.GetResolver().Resolve(anchored))
                    if (
                        resolved
                        and Path(resolved).is_file()
                        and any(
                            Path(resolved).is_relative_to(r)
                            for r in (Path("/reference"), Path("/candidate"))
                        )
                    ):
                        texture = resolved
        shader = UsdShade.Shader.Define(
            stage, str(prim.GetPath()) + "/IllustrativePreview"
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colour)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
        if texture:
            tex = UsdShade.Shader.Define(
                stage, str(prim.GetPath()) + "/IllustrativeTexture"
            )
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(texture)
            )
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            uv = UsdShade.Shader.Define(stage, str(prim.GetPath()) + "/IllustrativeUV")
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.ConnectableAPI(), "result"
            )
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        approximations.append(
            {
                "material": str(prim.GetPath()),
                "method": "MDL diffuse input to illustrative UsdPreviewSurface",
                "texture_used": bool(texture),
            }
        )
    return materials, approximations


def render(request, output):
    from pxr import Gf, Usd, UsdAppUtils, UsdGeom, UsdImagingGL, UsdUtils
    from PIL import Image, ImageDraw

    settings = validate_settings(request["settings"])
    keepers, GL = _GL_context()
    plugins = list(UsdImagingGL.Engine.GetRendererPlugins())
    if "HdStormRendererPlugin" not in plugins:
        raise RuntimeError("Full OpenUSD Storm imaging plugin unavailable")
    stages = {}
    material_reports = {}
    warnings = []
    input_hashes = {}
    for role in ("reference", "candidate"):
        path = Path(request[role])
        expected = Path("/reference") if role == "reference" else Path("/candidate")
        if (
            not path.is_relative_to(expected)
            or path.resolve() != path
            or not path.is_file()
        ):
            raise ValueError("Confined USD input required")
        input_hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
        if role == "reference" and input_hashes[role] != request["reference_sha256"]:
            raise ValueError("Frozen reference identity differs")
        stage, root = _stage(
            path,
            alignment=request.get("candidate_alignment")
            if role == "candidate"
            else None,
        )
        stages[role] = (stage, root)
        _, _, unresolved = UsdUtils.ComputeAllDependencies(str(path))
        compatibility = _identity_texture_transforms(stage)
        materials, approximations = _materials(stage)
        support = material_support(materials, [str(x) for x in unresolved])
        support["approximations"] = approximations
        support["compatibility"] = compatibility
        support["omitted_joint_properties"] = list(
            stage.GetRootLayer().customLayerData.get("illustration_omitted_joint_properties", [])
        )
        material_reports[role] = support
        warnings.extend(role + ": " + w for w in support["warnings"])
    reference, root = stages["reference"]
    box = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=False
        )
        .ComputeWorldBound(root.GetPrim())
        .ComputeAlignedRange()
    )
    center = (box.GetMin() + box.GetMax()) * 0.5
    span = max(box.GetSize())
    if not math.isfinite(span) or span <= 0:
        raise ValueError("Reference bounds cannot frame illustrations")
    directions = {
        "isometric": (1, -1, 0.8),
        "front": (0, -1, 0),
        "side": (1, 0, 0),
        "top": (0, 0, 1),
    }
    rig = {
        "schema": "simready-illustration-rig-v1",
        "basis": "reference bounds in metres/Z-up",
        "center": list(center),
        "span": span,
        "projection": "orthographic",
        "aperture": span * 1.65,
        "lighting": "Storm fixed camera light, source lights disabled",
        "views": {},
        "width": settings["width"],
    }
    if request.get("candidate_alignment") is not None:
        rig["candidate_alignment"] = request["candidate_alignment"]
    for view in settings["views"]:
        direction = Gf.Vec3d(*directions[view]).GetNormalized()
        eye = center + direction * span * 3
        up = Gf.Vec3d(0, 1, 0) if view == "top" else Gf.Vec3d(0, 0, 1)
        matrix = Gf.Matrix4d().SetLookAt(eye, center, up).GetInverse()
        rig["views"][view] = {
            "camera_to_world": [list(row) for row in matrix],
            "clipping_range": [max(span * 0.001, 1e-6), span * 100],
        }
    rig_sha = hashlib.sha256(
        json.dumps(rig, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    images = []
    cells = {}
    for role, (stage, root) in stages.items():
        for view in settings["views"]:
            data = rig["views"][view]
            camera = UsdGeom.Camera.Define(stage, "/IllustrationCamera")
            camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            camera.CreateHorizontalApertureAttr(rig["aperture"] * 10)
            camera.CreateVerticalApertureAttr(rig["aperture"] * 10)
            camera.CreateClippingRangeAttr(Gf.Vec2f(*data["clipping_range"]))
            transform = Gf.Matrix4d(
                *[x for row in data["camera_to_world"] for x in row]
            )
            xf = UsdGeom.Xformable(camera)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(transform)
            recorder = UsdAppUtils.FrameRecorder("HdStormRendererPlugin", True, False)
            recorder.SetImageWidth(settings["width"])
            recorder.SetCameraLightEnabled(True)
            recorder.SetColorCorrectionMode("sRGB")
            recorder.SetIncludedPurposes(["default", "render", "proxy"])
            recorder.SetPrimaryCameraPrimPath(camera.GetPath())
            name = role + "-" + view + ".png"
            path = output / name
            if not recorder.Record(stage, camera, Usd.TimeCode.Default(), str(path)):
                raise RuntimeError("Storm did not record a PNG")
            actual = str(recorder.GetCurrentRendererId())
            recorder = None
            if actual != "HdStormRendererPlugin":
                raise RuntimeError("Unexpected rendering plugin")
            with Image.open(path) as im:
                im.load()
                cells[(role, view)] = im.convert("RGBA")
            images.append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "view": role + " / " + view,
                    "rig_sha256": rig_sha,
                }
            )
    width = settings["width"]
    sheet = Image.new(
        "RGB", (width * len(settings["views"]), 2 * (width + 28)), (232, 235, 239)
    )
    draw = ImageDraw.Draw(sheet)
    for row, role in enumerate(("reference", "candidate")):
        for col, view in enumerate(settings["views"]):
            x = col * width
            y = row * (width + 28)
            draw.text((x + 8, y + 7), role.title() + " / " + view, fill=(20, 25, 32))
            cell = cells[(role, view)]
            sheet.paste(cell, (x, y + 28), cell)
    name = "contact-sheet.png"
    sheet.save(output / name)
    images.insert(
        0,
        {
            "file": name,
            "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
            "view": "matched reference / reconstruction contact sheet",
            "rig_sha256": rig_sha,
        },
    )
    for role in ("reference", "candidate"):
        if (
            hashlib.sha256(Path(request[role]).read_bytes()).hexdigest()
            != input_hashes[role]
        ):
            raise ValueError("Illustrated source bytes changed")
    result = {
        "input_entrypoint_sha256": input_hashes,
        "status": "rendered",
        "renderer": "storm-llvmpipe",
        "renderer_plugin": "HdStormRendererPlugin",
        "GL": GL,
        "illustration_only": True,
        "scoring": False,
        "material_fidelity": "approximate",
        "material_reports": material_reports,
        "warnings": warnings,
        "rig": rig,
        "rig_sha256": rig_sha,
        "images": images,
        "source_bytes_modified": False,
        "original_input_thumbnail_modified": False,
        "GPU_device_required": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "rendered",
                "GL": GL,
                "warnings": warnings,
                "images": len(images),
                "rig_sha256": rig_sha,
            }
        )
    )
    return result


def main():
    request = Path(sys.argv[1])
    with virtual_display():
        render(json.loads(request.read_text()), request.parent)


if __name__ == "__main__":
    main()
