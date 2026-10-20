"""Shared immutable Harbor images and private confined reference registry; no actor selector."""

from __future__ import annotations
from copy import deepcopy
import hashlib, json, math, re, stat, os
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

_ID = re.compile("^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_IMAGE = re.compile("^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
_SHA = re.compile("^[0-9a-f]{64}$")
_EXEC = {".py", ".sh", ".so", ".dll", ".exe", ".bat", ".ps1", ".js"}
_USD = {".usd", ".usda", ".usdc"}


def _hash(v):
    return hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _relative(value):
    if (
        not isinstance(value, str)
        or not value
        or any((c in value for c in ("\\", ":", "$", "~")))
    ):
        raise ValueError("confined relative POSIX path required")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or p.as_posix() != value:
        raise ValueError("confined relative POSIX path required")
    return p


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(
            "bounded registry identifier required; paths/environment selectors forbidden"
        )
    return value


def _sealed(value, key):
    out = deepcopy(value)
    digest = out.pop(key, None)
    if (
        not isinstance(digest, str)
        or not _SHA.fullmatch(digest)
        or _hash(out) != digest
    ):
        raise ValueError(key + " identity mismatch")
    return deepcopy(value)


def build_shared_profile(
    *,
    repository_root,
    actor_image,
    verifier_image,
    runtime_image,
    code_files,
    version="1.0.0",
):
    root = Path(repository_root).absolute()
    if root.resolve() != root:
        raise ValueError("repository symlink forbidden")
    files = {}
    for raw in code_files:
        relative = _relative(raw)
        p = root.joinpath(*relative.parts)
        if p.resolve() != p or not p.is_file():
            raise ValueError("regular confined code file required")
        files[str(relative)] = hashlib.sha256(p.read_bytes()).hexdigest()
    profile = {
        "schema": "simready-environment/v1",
        "version": _id(version),
        "actor_image": actor_image,
        "verifier_image": verifier_image,
        "runtime_image": runtime_image,
        "code_files": files,
        "actor_reference_access": "none",
        "asset_selection": "private_registry_asset_id_only",
        "task_inputs": "public supplied thumbnail and original brief mounted separately",
        "official_ready": False,
    }
    profile["profile_sha256"] = _hash(profile)
    return validate_shared_profile(profile)


def validate_shared_profile(profile):
    p = _sealed(profile, "profile_sha256")
    if p.get("schema") != "simready-environment/v1":
        raise ValueError("unknown shared profile")
    _id(p.get("version"))
    for key in ("actor_image", "verifier_image", "runtime_image"):
        if not isinstance(p.get(key), str) or not (
            _IMAGE.fullmatch(p[key]) or re.fullmatch("sha256:[0-9a-f]{64}", p[key])
        ):
            raise ValueError("immutable shared image pin required")
    if not isinstance(p.get("code_files"), dict) or not p["code_files"]:
        raise ValueError("shared code manifest required")
    for path, digest in p["code_files"].items():
        _relative(path)
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("code SHA256 required")
    if (
        p.get("actor_reference_access") != "none"
        or p.get("asset_selection") != "private_registry_asset_id_only"
    ):
        raise ValueError("private registry isolation required")
    return p


def validate_dependency_manifest(manifest):
    m = _sealed(manifest, "closure_sha256")
    if m.get("schema") != "simready-reference-closure/v1":
        raise ValueError("unknown dependency manifest")
    _id(m.get("package"))
    _relative(m.get("entrypoint"))
    if PurePosixPath(m["entrypoint"]).suffix.lower() not in _USD:
        raise ValueError("USD entrypoint required")
    files = m.get("files")
    if not isinstance(files, dict) or not files or len(files) > 10000:
        raise ValueError("bounded dependency manifest required")
    total = 0
    for raw, row in files.items():
        path = _relative(raw)
        if path.suffix.lower() in _EXEC or path.suffix.lower() == ".usdz":
            raise ValueError("executable/archive dependency unsupported")
        if (
            not isinstance(row, dict)
            or not _SHA.fullmatch(row.get("sha256", ""))
            or type(row.get("bytes")) is not int
            or (row["bytes"] < 0)
        ):
            raise ValueError("regular file SHA/size required")
        total += row["bytes"]
    if total > 1024**3:
        raise ValueError("reference package exceeds one GiB bound")
    if m["entrypoint"] not in files or files[m["entrypoint"]]["sha256"] != m.get(
        "source_sha256"
    ):
        raise ValueError("source entry digest mismatch")
    c = m.get("collector", {})
    if (
        c.get("method") != "OpenUSD_Sdf_asset_paths_all_package_layers"
        or c.get("network_resolution") is not False
        or c.get("complete_package_manifest") is not True
    ):
        raise ValueError("complete offline USD dependency inspection required")
    metadata = m.get("root_metadata", {})
    if metadata.get("up_axis") not in ("X", "Y", "Z") or any(
        (
            type(metadata.get(k)) not in (int, float)
            or not math.isfinite(metadata[k])
            or metadata[k] <= 0
            for k in ("meters_per_unit", "kilograms_per_unit")
        )
    ):
        raise ValueError("finite real root metadata required")
    return m


def _package_files(root):
    files = {}
    for p in sorted(root.rglob("*")):
        s = p.lstat()
        if stat.S_ISLNK(s.st_mode) or p.resolve() != p:
            raise ValueError("reference symlink forbidden")
        if stat.S_ISDIR(s.st_mode):
            continue
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
            raise ValueError("regular unlinked dependency required")
        files[p.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": s.st_size,
        }
        if len(files) > 10000 or sum((x["bytes"] for x in files.values())) > 1024**3:
            raise ValueError("bounded package required")
    return files


def verify_profile_code(profile, *, actual_source_root=None):
    """Verify the declared prepared code against actual trusted loaded project files."""
    p = validate_shared_profile(profile)
    loaded = Path(__file__).resolve().parent
    base = (
        Path(actual_source_root).absolute() if actual_source_root is not None else None
    )
    for relative, expected in p["code_files"].items():
        parts = PurePosixPath(relative).parts
        if base is not None:
            source = base.joinpath(*parts)
        elif parts[:2] == ("src", "simready_bench"):
            source = loaded.parents[1].joinpath(*parts[1:])
        else:
            raise ValueError(
                "profile code path needs explicit trusted actual source root"
            )
        if (
            source.resolve() != source
            or not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("actual loaded profile code changed or missing")
    return {
        "status": "verified_actual_project_code",
        "files": len(p["code_files"]),
        "manifest_sha256": _hash(p["code_files"]),
    }


def validate_task_reference_binding(
    task_config, *, allowed_reference_root, actual_source_root=None
):
    from simready_bench.assets import validate_reference_binding

    return validate_reference_binding(
        task_config,
        allowed_reference_root=allowed_reference_root,
        actual_source_root=actual_source_root,
    )


def collect_reference_dependencies(*, package, entrypoint, package_root):
    """Read-only CPU/OpenUSD inspection; scan paths before composing any stage."""
    _id(package)
    _relative(entrypoint)
    root = Path(package_root).absolute()
    if root.resolve() != root or not root.is_dir() or root.name != package:
        raise ValueError("regular explicit confined package root required")
    files = _package_files(root)
    if entrypoint not in files:
        raise ValueError("reference USD missing")
    from pxr import Sdf, Usd, UsdGeom
    from pxr import UsdUtils, UsdPhysics

    def check(raw, owner):
        if not raw:
            return raw
        if (
            not isinstance(raw, str)
            or not raw
            or raw.startswith(("/", "~"))
            or any(c in raw for c in ("\\", ":", "$", "[", "]"))
        ):
            raise ValueError("dependency must be local and confined")
        rel = PurePosixPath(raw)
        resolved = Path(os.path.normpath(str(owner.parent.joinpath(*rel.parts))))
        if resolved.resolve() != resolved or not resolved.is_relative_to(root):
            raise ValueError("dependency escape/symlink forbidden")
        if "<UDIM>" in raw:
            matches = list(
                resolved.parent.glob(
                    resolved.name.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")
                )
            )
            if not matches:
                raise ValueError("UDIM dependency missing")
        elif not resolved.is_file():
            raise ValueError("dependency missing")
        return raw

    for raw in files:
        path = root / raw
        suffix = path.suffix.lower()
        if suffix in _EXEC or suffix == ".usdz":
            raise ValueError("executable/archive dependency unsupported")
        if suffix in _USD:
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            if not layer:
                raise ValueError("USD layer unreadable")
            errors = []

            def visit(value):
                try:
                    return check(value, path)
                except ValueError as exc:
                    errors.append(str(exc))
                    return value

            UsdUtils.ModifyAssetPaths(layer, visit)
            if errors:
                raise ValueError("; ".join(sorted(set(errors))))
        elif suffix == ".mtlx":
            for el in ET.fromstring(path.read_bytes()).iter():
                for key, value in el.attrib.items():
                    if (
                        key in ("file", "href", "filename")
                        or key == "value"
                        and el.attrib.get("type") == "filename"
                    ):
                        check(value, path)
        elif suffix == ".mdl":
            text = path.read_text()
            for value in re.findall(
                r'(?:texture_\w+|light_profile|bsdf_measurement)\s*\(\s*"([^"\n]+)"',
                text,
            ):
                check(value, path)
            for value in re.findall(r"import\s+([^;]+);", text):
                if value.strip().startswith((".", '"')):
                    raise ValueError("unresolved local MDL import")
    stage = Usd.Stage.Open(str(root / entrypoint))
    if not stage or stage.GetCompositionErrors():
        raise ValueError("USD composition incomplete")
    m = {
        "schema": "simready-reference-closure/v1",
        "package": package,
        "entrypoint": entrypoint,
        "files": files,
        "source_sha256": files[entrypoint]["sha256"],
        "root_metadata": {
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "kilograms_per_unit": float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
            "default_prim": str(stage.GetDefaultPrim().GetPath())
            if stage.GetDefaultPrim()
            else None,
        },
        "collector": {
            "method": "OpenUSD_Sdf_asset_paths_all_package_layers",
            "network_resolution": False,
            "complete_package_manifest": True,
            "OpenUSD_version": str(Usd.GetVersion()),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "MDL_standard_library": "named imports only; implementation pinned by runtime image",
        },
    }
    m["closure_sha256"] = _hash(m)
    return validate_dependency_manifest(m)
