"""Unscored human illustrations, rendered after the actor has stopped."""

from __future__ import annotations
import hashlib, json, os, re, subprocess, time, uuid
from pathlib import Path

VIEWS = ("isometric", "front", "side", "top")


def validate_settings(value):
    if type(value) is not dict or set(value) - {
        "renderer",
        "width",
        "views",
        "seconds",
    }:
        raise ValueError("Exact illustration settings required")
    result = {
        "renderer": "none",
        "width": 512,
        "views": ["isometric", "front", "side"],
        "seconds": 180,
        **value,
    }
    if result["renderer"] not in ("none", "storm-llvmpipe"):
        raise ValueError("Known illustration renderer required")
    for key, lo, hi in (("width", 128, 1024), ("seconds", 10, 900)):
        if type(result[key]) is not int or not lo <= result[key] <= hi:
            raise ValueError("Bounded illustration " + key + " required")
    views = result["views"]
    if (
        type(views) is not list
        or not views
        or len(views) > 4
        or len(set(views)) != len(views)
        or any(v not in VIEWS for v in views)
    ):
        raise ValueError("Choose unique named illustration views")
    return result


def check_software_GL(vendor, renderer, version):
    if not isinstance(renderer, str) or "llvmpipe" not in renderer.lower():
        raise ValueError("Active OpenGL renderer is not verified llvmpipe")
    return {
        "vendor": vendor,
        "renderer": renderer,
        "version": version,
        "software_verified": True,
    }


def material_support(materials, unresolved):
    mdl = [
        m["path"]
        for m in materials
        if m.get("MDL_surface") and not m.get("universal_surface")
    ]
    warnings = []
    if mdl:
        warnings.append(
            "MDL-only materials use an illustrative Preview Surface approximation; MDL is not evaluated."
        )
    if unresolved:
        warnings.append("Unresolved asset dependencies may affect the illustrations.")
    return {
        "illustration_only": True,
        "scoring": False,
        "material_fidelity": "approximate",
        "MDL_only_materials": mdl,
        "unresolved_dependencies": sorted(set(unresolved)),
        "materials": materials,
        "warnings": warnings,
    }


def render_final(condition, cfg, runtime, snapshot, root, *, alignment=None):
    try:
        return _render_final(
            condition, cfg, runtime, snapshot, root, alignment=alignment
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "renderer": condition["settings"]
            .get("illustrations", {})
            .get("renderer", "none"),
            "illustration_only": True,
            "scoring": False,
            "images": [],
            "warnings": ["CPU illustration unavailable: " + type(exc).__name__],
        }


def _render_final(condition, cfg, runtime, snapshot, root, *, alignment=None):
    """Only read frozen inputs; renderer failure cannot change validation/reward."""
    settings = validate_settings(condition["settings"].get("illustrations", {}))
    base = {
        "renderer": settings["renderer"],
        "illustration_only": True,
        "scoring": False,
        "images": [],
        "warnings": [],
    }
    if settings["renderer"] == "none":
        return {**base, "status": "disabled"}
    if not snapshot.valid:
        return {
            **base,
            "status": "unavailable",
            "warnings": ["No valid frozen candidate to illustrate."],
        }
    if alignment is None:
        return {
            **base,
            "status": "unavailable",
            "warnings": ["Whole-object comparison alignment is unavailable."],
        }
    image = runtime.get("illustration_image")
    if not isinstance(image, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        return {
            **base,
            "status": "unavailable",
            "warnings": ["Operator has not bound an immutable CPU illustration image."],
        }
    if "validator_reference" not in cfg:
        return {
            **base,
            "status": "unavailable",
            "warnings": [
                "This illustration adapter requires a frozen NVIDIA source binding."
            ],
        }
    binding = cfg["validator_reference"]
    package = Path(runtime["allowed_reference_root"]) / binding["package"]
    source = package / binding["dependency_manifest"]["entrypoint"]
    candidate = Path(snapshot.path)
    output = Path(root) / "artifacts/illustrations"
    output.mkdir(mode=0o700, parents=True)
    for p in (package, source, candidate, Path(runtime["source_root"]), output):
        if p.resolve() != p or not p.exists():
            raise ValueError("Confined regular illustration inputs required")
    request = {
        "schema": "simready-Storm-illustrations-v1",
        "reference": "/reference/" + str(source.relative_to(package)),
        "candidate": "/candidate/" + snapshot.entrypoint,
        "settings": settings,
        "reference_sha256": binding["dependency_manifest"]["source_sha256"],
        "candidate_snapshot_sha256": snapshot.sha256,
    }
    if alignment is not None:
        request["candidate_alignment"] = alignment
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    name = "simready-illustration-" + uuid.uuid4().hex[:12]
    argv = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--pull=never",
        "--network",
        "none",
        "--read-only",
        "--user",
        str(os.getuid()) + ":" + str(os.getgid()),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--cpus",
        "2",
        "--memory",
        "2g",
        "--shm-size",
        "256m",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=536870912,mode=1777",
    ]
    for host, dest, readonly in (
        (package, "/reference", True),
        (candidate, "/candidate", True),
        (Path(runtime["source_root"]), "/product", True),
        (output, "/render", False),
    ):
        argv += [
            "--mount",
            "type=bind,src="
            + str(host)
            + ",dst="
            + dest
            + (",readonly" if readonly else ""),
        ]
    argv += [
        "--env",
        "PYTHONPATH=/product",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "LIBGL_ALWAYS_SOFTWARE=true",
        "--env",
        "GALLIUM_DRIVER=llvmpipe",
        "--env",
        "LP_NUM_THREADS=2",
        "--entrypoint",
        "/opt/storm/bin/python",
        image,
        "-B",
        "-m",
        "simready_bench.storm",
        "/render/request.json",
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=settings["seconds"]
        )
        (output / "renderer.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(
                "CPU illustration process failed; original renderer log retained privately"
            )
        receipt = json.loads((output / "receipt.json").read_text())
        check_software_GL(
            **{k: receipt["GL"][k] for k in ("vendor", "renderer", "version")}
        )
        if receipt.get("renderer_plugin") != "HdStormRendererPlugin":
            raise ValueError("Actual Storm renderer not confirmed")
        images = []
        for item in receipt["images"]:
            p = Path(item["file"])
            if len(p.parts) != 1 or p.suffix != ".png":
                raise ValueError("Confined renderer PNG required")
            path = output / p
            if (
                path.resolve() != path
                or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
            ):
                raise ValueError("Illustration file identity differs")
            images.append({**item, "file": str(path)})
        return {
            **base,
            **receipt,
            "images": images,
            "renderer_image": image,
            "seconds": time.monotonic() - started,
        }
    except (
        subprocess.TimeoutExpired,
        RuntimeError,
        ValueError,
        OSError,
        KeyError,
    ) as exc:
        try:
            subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, timeout=30
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        return {
            **base,
            "status": "unavailable",
            "warnings": ["CPU illustration unavailable: " + type(exc).__name__],
            "seconds": time.monotonic() - started,
        }
