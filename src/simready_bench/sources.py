"""Pinned public dataset download and private archive import for product manifests."""

from __future__ import annotations
import hashlib, json, os, re, tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen
from simready_bench.archive_ingest import ingest_archive, INDEX


def validate_source(value, *, base_dir):
    if type(value) is not dict:
        raise ValueError("source must be an object")
    kind = value.get("kind")
    common = {"kind", "sha256", "destination"}
    fields = common | (
        {"path"}
        if kind == "local-archive"
        else {"repo_id", "revision", "filename"}
        if kind == "huggingface-archive"
        else {"revision", "packages"}
        if kind == "nvidia-hf-packages"
        else set()
    )
    if (
        kind not in ("local-archive", "huggingface-archive", "nvidia-hf-packages")
        or set(value) - fields
    ):
        raise ValueError("Unsupported archive source fields")
    result = dict(value)
    if kind != "nvidia-hf-packages" and not re.fullmatch(
        r"[a-f0-9]{64}", str(result.get("sha256", ""))
    ):
        raise ValueError("Pinned source SHA-256 required")
    from simready_bench.config import _relative

    result["destination"] = str(
        _relative(result.get("destination", "imported"), "source destination")
    )
    if kind == "local-archive":
        raw = result.get("path")
        if (
            not isinstance(raw, str)
            or not raw
            or any(c in raw for c in ("\x00", "\n", "\r", "$", "~"))
        ):
            raise ValueError("Literal archive path required")
        result["path"] = str((Path(base_dir) / raw).absolute())
    elif kind == "huggingface-archive":
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
            str(result.get("repo_id", "")),
        ):
            raise ValueError("Explicit Hugging Face dataset owner/name required")
        if not re.fullmatch(r"[a-f0-9]{40}", str(result.get("revision", ""))):
            raise ValueError("Immutable Hugging Face commit required")
        result["filename"] = str(
            _relative(result.get("filename"), "HF archive filename")
        )
    else:
        from simready_bench.engine.acquisition import REVISION

        if result.get("revision") != REVISION:
            raise ValueError(
                "NVIDIA package source requires audited immutable revision " + REVISION
            )
        packages = result.get("packages")
        if (
            not isinstance(packages, list)
            or not 1 <= len(packages) <= 100
            or len(set(packages)) != len(packages)
            or any(
                not isinstance(p, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", p)
                for p in packages
            )
        ):
            raise ValueError("Explicit unique NVIDIA package names required")
        if "sha256" in result:
            raise ValueError(
                "NVIDIA package integrity uses its official packaging BOM, not an archive hash"
            )
    return result


def _verify_import(destination, source):
    if destination.resolve() != destination or not destination.is_dir():
        raise ValueError("Private imported archive directory required")
    receipt_path = destination / "ingest-receipt.json"
    if receipt_path.is_symlink() or receipt_path.stat().st_size > 4 * 1024**2:
        raise ValueError("Bounded original ingest receipt required")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("archive_sha256") != source["sha256"]:
        raise ValueError("Existing archive source differs")
    index_path = destination / INDEX
    if index_path.is_symlink() or index_path.stat().st_size > 1024**2:
        raise ValueError("Bounded original archive index required")
    index_raw = index_path.read_bytes()
    if hashlib.sha256(index_raw).hexdigest() != receipt.get("index_sha256"):
        raise ValueError("Imported archive index changed")
    index = json.loads(index_raw)
    if index.get("schema") != "simready-reference-archive-v1" or index.get(
        "files"
    ) != receipt.get("files"):
        raise ValueError("Imported receipt differs from pinned archive index")
    expected = set(receipt.get("files", {})) | {INDEX, "ingest-receipt.json"}
    actual = set()
    for p in destination.rglob("*"):
        if p.is_symlink():
            raise ValueError("Imported archive links forbidden")
        if p.is_dir():
            continue
        relative = p.relative_to(destination).as_posix()
        actual.add(relative)
        if relative not in expected or not p.is_file() or p.stat().st_nlink != 1:
            raise ValueError("Imported archive closure changed")
        if relative in receipt["files"]:
            row = receipt["files"][relative]
            with p.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if p.stat().st_size != row["bytes"] or digest != row["sha256"]:
                raise ValueError("Imported archive bytes changed")
    if actual != expected:
        raise ValueError("Imported archive file roster changed")
    return {**receipt, "destination": str(destination), "reused_verified_import": True}


def import_source(source, *, base_dir, assets_root, opener=urlopen):
    source = validate_source(source, base_dir=base_dir)
    root = Path(assets_root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Existing trusted assets root required")
    destination = root.joinpath(*PurePosixPath(source["destination"]).parts)
    if source["kind"] != "nvidia-hf-packages" and (
        destination.exists() or destination.is_symlink()
    ):
        return _verify_import(destination, source)
    if destination.parent.resolve() != destination.parent:
        raise ValueError("Source destination parent must not contain symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if source["kind"] == "nvidia-hf-packages":
        from simready_bench.engine.acquisition import acquire

        if destination.is_symlink() or any(
            p.is_symlink() for p in destination.rglob("*")
        ):
            raise ValueError("NVIDIA import links forbidden")
        destination.mkdir(mode=0o700, exist_ok=True)
        previous = os.umask(0o077)
        try:
            receipts = [acquire(package, destination) for package in source["packages"]]
        finally:
            os.umask(previous)
        return {
            "schema": "simready-bench-nvidia-source-v1",
            "source": source,
            "destination": str(destination),
            "packages": receipts,
            "visibility": "private_ground_truth",
            "provider_calls": 0,
        }
    elif source["kind"] == "local-archive":
        result = ingest_archive(source["path"], destination, sha256=source["sha256"])
    else:
        url = (
            "https://huggingface.co/datasets/"
            + source["repo_id"]
            + "/resolve/"
            + source["revision"]
            + "/"
            + quote(source["filename"], safe="/")
        )
        fd, name = tempfile.mkstemp(prefix=".hf-archive-", dir=root)
        os.fchmod(fd, 0o600)
        try:
            size = 0
            with os.fdopen(fd, "wb") as target, opener(url, timeout=60) as incoming:
                if not incoming.geturl().startswith("https://"):
                    raise ValueError("HF archive redirect must retain HTTPS")
                while block := incoming.read(1024**2):
                    size += len(block)
                    if size > 512 * 1024**2:
                        raise ValueError("HF archive exceeds 512MiB bound")
                    target.write(block)
            result = ingest_archive(name, destination, sha256=source["sha256"])
        finally:
            Path(name).unlink(missing_ok=True)
    return {
        **result,
        "destination": str(destination),
        "reused_verified_import": False,
        "source": source,
    }
