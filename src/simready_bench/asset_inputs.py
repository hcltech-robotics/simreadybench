"""Select local reference files without copying neighbouring project data."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET


_USD = {".usd", ".usda", ".usdc"}
_EXECUTABLE = {
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
_PRIVATE_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".ssh",
    ".aws",
    ".azure",
}
_PRIVATE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx", ".log"}
_PRIVATE_FILES = {
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
    "pyvenv.cfg",
    ".npmrc",
    ".pypirc",
    ".netrc",
}
_MAX_BYTES = 1024**3
_MAX_FILES = 10000
_MDL_STANDARD_MODULES = {
    "std",
    "limits",
    "anno",
    "state",
    "math",
    "tex",
    "scene",
    "debug",
    "df",
    "base",
}


def selected_file(path, root):
    """Require a confined regular asset file before any staging takes place."""
    path = Path(os.path.abspath(path))
    root = Path(os.path.abspath(root))
    if root.resolve() != root or not path.is_relative_to(root):
        raise ValueError("Selected asset files must remain inside their asset root")
    relative = path.relative_to(root)
    parts = [part.casefold() for part in relative.parts]
    if (
        any(
            part in _PRIVATE_DIRECTORIES
            or part.startswith((".env", ".git", ".hg", ".svn"))
            for part in parts
        )
        or any(
            suffix.casefold() in _EXECUTABLE | _PRIVATE_SUFFIXES
            for suffix in path.suffixes
        )
        or path.name.casefold().endswith(".env")
        or path.name.casefold() in _PRIVATE_FILES
    ):
        raise ValueError(
            "Asset selections cannot include credentials, project metadata or executable files"
        )
    if (
        path.resolve() != path
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_nlink != 1
    ):
        raise ValueError("Selected asset files must be regular files without links")
    if path.stat().st_size > _MAX_BYTES:
        raise ValueError("Asset file exceeds the 1 GiB package limit")
    return path


def _material_text(text):
    """Remove MDL comments while retaining quoted resource paths."""
    return re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
        lambda match: match.group() if match.group().startswith('"') else " ",
        text,
        flags=re.S,
    )


def _materialx_references(path):
    content = path.read_bytes()
    if re.search(
        rb"<!\s*(?:DOCTYPE|ENTITY)\b", content.replace(b"\0", b""), flags=re.I
    ):
        raise ValueError("MaterialX declarations require an explicit asset manifest")
    try:
        document = ET.fromstring(content)
    except ET.ParseError as error:
        raise ValueError(
            "Cannot inspect MaterialX dependencies in " + path.name
        ) from error
    references = []
    pending = [(document, "")]
    while pending:
        element, inherited_prefix = pending.pop()
        if "{http://www.w3.org/XML/1998/namespace}base" in element.attrib:
            raise ValueError("MaterialX xml:base requires an explicit asset manifest")
        prefix = element.attrib.get("fileprefix", inherited_prefix)
        if (
            element.tag.rsplit("}", 1)[-1] == "include"
            and element.get("href")
            and prefix
        ):
            raise ValueError(
                "MaterialX includes in a prefixed scope require an explicit asset manifest"
            )
        for key, value in element.attrib.items():
            if key == "value" and element.get("type") == "filename":
                value = prefix + value if value else ""
            elif key not in ("file", "href", "filename"):
                continue
            if any(character in value.replace("<UDIM>", "") for character in "<>[]{}$"):
                raise ValueError(
                    "MaterialX filename substitutions require an explicit asset manifest"
                )
            references.append(value)
        pending.extend((child, prefix) for child in element)
    return references


def _mdl_resources(text):
    resource = r"(?:texture_2d|texture_3d|texture_cube|texture_ptex|light_profile|bsdf_measurement)"
    literal = r'"((?:\\.|[^"\\])*)"'
    supported = re.compile(
        resource
        + r"\s*(?:\(\s*(?:name\s*:\s*)?"
        + literal
        + r"\s*(?=[,)])"
        + r"|[A-Za-z_][A-Za-z0-9_]*\s*=\s*"
        + literal
        + r"\s*(?=[,;)])|[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?="
        + resource
        + r"\s*\()|\(\s*\))"
    )
    references = []
    for token in re.finditer(r'"(?:\\.|[^"\\])*"|\b' + resource + r"\b", text):
        if token.group().startswith('"'):
            continue
        match = supported.match(text, token.start())
        if not match or re.search(r"\btypedef\s*$", text[: token.start()]):
            raise ValueError(
                "MDL resource expressions require an explicit asset manifest"
            )
        value = match.group(1) if match.group(1) is not None else match.group(2)
        if value is not None:
            references.append(json.loads('"' + value + '"'))
    return references


def _references(path):
    suffix = path.suffix.lower()
    if suffix in _USD:
        try:
            from pxr import Sdf, UsdUtils
        except ImportError as error:
            raise ValueError(
                "Install usd-core 26.8 to select USD dependencies, or supply an explicit asset manifest"
            ) from error
        try:
            layer = Sdf.Layer.OpenAsAnonymous(str(path))
            if not layer:
                raise ValueError("USD layer is unreadable")
            references = []

            def collect(value):
                if value:
                    references.append(value)
                return value

            UsdUtils.ModifyAssetPaths(layer, collect)
            return references
        except Exception as error:
            raise ValueError(
                "Cannot inspect USD dependency paths in " + path.name
            ) from error
    if suffix == ".mtlx":
        return _materialx_references(path)
    if suffix == ".mdl":
        try:
            text = _material_text(path.read_text(encoding="utf-8"))
            for declaration in re.findall(r"\bimport\s+([^;]+);", text):
                for imported in declaration.split(","):
                    imported = imported.strip()
                    parts = imported.removeprefix("::").split("::")
                    if not (
                        len(parts) in (1, 2)
                        and parts[0] in _MDL_STANDARD_MODULES
                        and all(
                            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|\*", part)
                            for part in parts
                        )
                    ):
                        raise ValueError(
                            "Custom MDL module imports require an explicit asset manifest with their module and resource files"
                        )
            return _mdl_resources(text)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Cannot inspect MDL dependencies in " + path.name
            ) from error
    return []


def dependency_files(source, root):
    """Follow authored USD, MaterialX and MDL asset paths without composition."""
    root = Path(os.path.abspath(root))
    pending = [selected_file(source, root)]
    selected = set()
    total = 0
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        selected.add(path)
        total += path.stat().st_size
        if len(selected) > _MAX_FILES or total > _MAX_BYTES:
            raise ValueError("Asset dependency selection exceeds 10000 files or 1 GiB")
        for value in _references(path):
            if not value:
                continue
            if (
                not isinstance(value, str)
                or len(value) > 4096
                or value.startswith(("/", "~"))
                or any(character in value for character in ("\\", ":", "$", "[", "]"))
            ):
                raise ValueError(
                    "USD dependencies must be local paths inside the asset root"
                )
            relative = PurePosixPath(value)
            resolved = Path(os.path.abspath(path.parent.joinpath(*relative.parts)))
            if not resolved.is_relative_to(root):
                raise ValueError(
                    "USD dependency escapes the asset root. Use --assets-root for shared asset files"
                )
            if "<UDIM>" in value:
                matches = sorted(
                    resolved.parent.glob(
                        resolved.name.replace("<UDIM>", "[0-9][0-9][0-9][0-9]")
                    )
                )
                if not matches:
                    raise ValueError("USD UDIM dependency is missing: " + value)
                pending.extend(selected_file(match, root) for match in matches)
            else:
                pending.append(selected_file(resolved, root))
    return sorted(selected)
