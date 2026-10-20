"""Offline candidate package and USD dependency inspection.

Invoke in a bounded worker. Raw reports remain evaluator-owned. No dependency is
composed until every authored asset path in every packaged layer is inspected.
"""

from __future__ import annotations
import hashlib
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile
import xml.etree.ElementTree as ET

VERSION = "package-checks-2"
LIMIT_BYTES = 256 * 1024 * 1024
LIMIT_FILES = 10000
USD_EXTS = {".usd", ".usda", ".usdc"}
EXEC_EXTS = {
    ".py",
    ".pyc",
    ".sh",
    ".bash",
    ".js",
    ".mjs",
    ".ts",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bat",
    ".cmd",
    ".ps1",
    ".jar",
    ".wasm",
}


def inspect_package(root, *, entrypoint="asset.usd"):
    root = Path(root).resolve()
    report = {
        "version": VERSION,
        "package_ok": True,
        "usd_ok": False,
        "geometry_ok": False,
        "composition_attempted": False,
        "metrics": {},
        "findings": [],
        "infrastructure_errors": [],
        "usd_diagnostics": [],
        "usd_diagnostics_truncated": False,
    }

    def fail(code, path=None, observed=None, gate="package"):
        report["findings"].append(
            {"code": code, "path": path, "observed": observed, "gate": gate}
        )
        if gate == "package":
            report["package_ok"] = False

    def infra(code):
        report["infrastructure_errors"].append(code)

    try:
        files = []
        total = 0
        for p in root.rglob("*"):
            s = p.lstat()
            if stat.S_ISDIR(s.st_mode):
                continue
            if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
                fail("nonregular_or_linked_file", p.relative_to(root).as_posix())
                continue
            files.append(p)
            total += s.st_size
            if len(files) > LIMIT_FILES or total > LIMIT_BYTES:
                fail("package_limit_exceeded")
                return report
        report["metrics"].update(package_bytes=total, package_files=len(files))
        ep = PurePosixPath(entrypoint)
        if (
            ep.is_absolute()
            or ".." in ep.parts
            or "\\" in entrypoint
            or (len(ep.parts) != 1)
        ):
            fail("invalid_entrypoint")
            return report
        asset = root / entrypoint
        if asset not in files or asset.suffix.lower() not in USD_EXTS | {".usdz"}:
            fail("missing_usd_entrypoint")
            return report
        if not report["package_ok"]:
            return report
        try:
            from pxr import Sdf, UsdUtils
        except ImportError:
            infra("openusd_unavailable")
            return report
        with tempfile.TemporaryDirectory(prefix="simready-package-") as scratch:
            layers = []
            scanfiles = list(files)
            expanded_total = total
            expanded_count = len(files)
            for archive in list(scanfiles):
                if archive.suffix.lower() != ".usdz":
                    continue
                base = Path(scratch) / hashlib.sha256(str(archive).encode()).hexdigest()
                base.mkdir()
                try:
                    with zipfile.ZipFile(archive) as z:
                        seen = set()
                        for item in z.infolist():
                            name = PurePosixPath(item.filename)
                            mode = item.external_attr >> 16
                            if (
                                name.is_absolute()
                                or ".." in name.parts
                                or "\\" in item.filename
                                or (":" in item.filename)
                                or (item.filename in seen)
                                or stat.S_ISLNK(mode)
                            ):
                                fail(
                                    "unsafe_archive_member",
                                    archive.relative_to(root).as_posix(),
                                )
                                break
                            seen.add(item.filename)
                            if item.is_dir():
                                continue
                            expanded_count += 1
                            expanded_total += item.file_size
                            if (
                                expanded_count > LIMIT_FILES
                                or expanded_total > LIMIT_BYTES
                            ):
                                fail("expanded_package_limit_exceeded")
                                break
                            if name.suffix.lower() == ".usdz":
                                infra("nested_usdz_not_supported")
                                continue
                            dst = base.joinpath(*name.parts)
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            with z.open(item) as stream, dst.open("wb") as out:
                                written = 0
                                while chunk := stream.read(1024 * 1024):
                                    written += len(chunk)
                                    if written > item.file_size:
                                        raise ValueError("archive size mismatch")
                                    out.write(chunk)
                            scanfiles.append(dst)
                except (zipfile.BadZipFile, ValueError, OSError):
                    fail("invalid_usdz_archive")

            def local_path(raw, owner):
                if not raw:
                    return
                if (
                    not isinstance(raw, str)
                    or len(raw) > 4096
                    or raw.startswith(("/", "~"))
                    or any((c in raw for c in ("\\", ":", "$")))
                ):
                    fail("external_dependency", str(owner.name))
                    return
                if "[" in raw or "]" in raw:
                    match = re.fullmatch("(.+\\.usdz)\\[([^\\[\\]]+)\\]", raw)
                    if not match:
                        fail("external_dependency", owner.name)
                        return
                    archive_name, member = match.groups()
                    local_path(archive_name, owner)
                    member_path = PurePosixPath(member)
                    if (
                        member_path.is_absolute()
                        or ".." in member_path.parts
                        or "\\" in member
                    ):
                        fail("external_dependency", owner.name)
                        return
                    try:
                        with zipfile.ZipFile(owner.parent / archive_name) as z:
                            if member not in z.namelist():
                                fail("missing_dependency", owner.name)
                    except (OSError, zipfile.BadZipFile):
                        fail("invalid_usdz_archive", owner.name)
                    return
                pp = PurePosixPath(raw)
                resolved = owner.parent.joinpath(*pp.parts).resolve()
                safe_root = root if owner.is_relative_to(root) else Path(scratch)
                if not resolved.is_relative_to(safe_root):
                    fail("external_dependency", str(owner.name))
                    return
                if pp.suffix.lower() in EXEC_EXTS:
                    fail("executable_dependency", str(owner.name))
                    return
                if "<UDIM>" in raw:
                    matches = list(
                        resolved.parent.glob(
                            resolved.name.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")
                        )
                    )
                    if not matches:
                        fail("missing_dependency", str(owner.name))
                elif not resolved.is_file():
                    fail("missing_dependency", str(owner.name))

            for p in scanfiles:
                if p.suffix.lower() in EXEC_EXTS:
                    fail("executable_dependency", p.name)
                    continue
                if p.suffix.lower() in USD_EXTS:
                    try:
                        layer = Sdf.Layer.OpenAsAnonymous(str(p))
                        if not layer:
                            fail("invalid_usd_layer", p.name)
                            continue
                        UsdUtils.ModifyAssetPaths(
                            layer, lambda value: local_path(value, p) or value
                        )
                        layers.append(layer)
                    except Exception:
                        fail("invalid_usd_layer", p.name)
                elif p.suffix.lower() == ".mtlx":
                    try:
                        for el in ET.fromstring(p.read_bytes()).iter():
                            for key, value in el.attrib.items():
                                if key in ("file", "href", "filename") or (
                                    key == "value"
                                    and el.attrib.get("type") == "filename"
                                ):
                                    local_path(value, p)
                    except (ET.ParseError, ValueError):
                        fail("invalid_materialx_document", p.name)
                elif p.suffix.lower() == ".mdl":
                    try:
                        text = p.read_text()
                        for raw in re.findall(
                            '(?:texture_\\w+|light_profile|bsdf_measurement)\\s*\\(\\s*"([^"\\n]+)"',
                            text,
                        ):
                            local_path(raw, p)
                        for raw in re.findall("import\\s+([^;]+);", text):
                            if raw.strip().startswith(".") or raw.strip().startswith(
                                '"'
                            ):
                                fail("unsupported_mdl_import", p.name)
                    except (UnicodeError, OSError):
                        fail("invalid_mdl_document", p.name)
            if not report["package_ok"] or report["infrastructure_errors"]:
                return report
            report["inspection_scope"] = "package_dependencies"
            return report
    except Exception as exc:
        infra("package_checker_error:" + type(exc).__name__)
    return report
