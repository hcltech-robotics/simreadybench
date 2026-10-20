"""Pinned reference acquisition. All source data stays curator-owned."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from .downloads import download_file, read_json_page

REVISION = "f4860aa0c75607adb50a78bf96e77510bffc040f"
BASE = f"https://huggingface.co/datasets/nvidia/simready-assets/resolve/{REVISION}"
ROOT = Path(__file__).resolve().parents[2]


def safe_relative(value: str) -> Path:
    p = PurePosixPath(value)
    if (
        not value
        or p.is_absolute()
        or ".." in p.parts
        or ("\\" in value)
        or (":" in value)
    ):
        raise ValueError("unsafe relative package path")
    return Path(*p.parts)


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def verify_bom(directory: Path, bom: dict) -> list[dict]:
    verified = []
    for item in bom["items"]:
        p = directory / safe_relative(item["relative_path"])
        expected = item["hash"]["sha256"]
        if not p.is_file() or p.stat().st_size != item["size"] or sha256(p) != expected:
            raise ValueError(f"BOM mismatch: {item['relative_path']}")
        verified.append(
            {
                "path": item["relative_path"],
                "size": p.stat().st_size,
                "sha256": expected,
            }
        )
    return verified


def acquire(package: str, references: Path) -> dict:
    safe_relative(package)
    destination = references / package
    destination.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/nvidia/simready-assets/tree/{REVISION}/{package}?recursive=true&limit=1000"
    entries = []
    while api:
        page, links = read_json_page(api)
        entries.extend(page)
        api = None
        for part in links.split(","):
            if 'rel="next"' in part:
                api = part.split("<", 1)[1].split(">", 1)[0]
    paths = [e for e in entries if e["type"] == "file"]
    expected_bytes = sum((e["size"] for e in paths))
    if shutil.disk_usage(references).free < expected_bytes + 5 * 1024**3:
        raise OSError(
            "insufficient disk space for complete source package with 5 GiB reserve"
        )

    def fetch(entry):
        rel = safe_relative(entry["path"])
        if rel.parts[0] != package:
            raise ValueError("source tree escaped package")
        target = references / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == entry["size"]:
            return
        tmp = target.with_name(target.name + ".downloading")
        url = BASE + "/" + urllib.parse.quote(entry["path"], safe="/")
        download_file(url, tmp, entry["size"])
        os.replace(tmp, target)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, paths))
    bom_path = destination / ".metadata/com.nvidia.simready.packaging.bom.json"
    packaging = json.loads(
        (destination / "com.nvidia.simready.packaging.json").read_text()
    )
    expected_bom = next(
        (
            m["hash"]["sha256"]
            for m in packaging["metadata"]
            if m["name"] == bom_path.name
        )
    )
    if sha256(bom_path) != expected_bom:
        raise ValueError("BOM manifest hash mismatch")
    verified = verify_bom(destination, json.loads(bom_path.read_text()))
    receipt = {
        "package": package,
        "revision": REVISION,
        "tree_file_count": len(paths),
        "download_bytes": expected_bytes,
        "bom_sha256": sha256(bom_path),
        "verified_bom_files": verified,
        "attribution": "NVIDIA",
        "licence": packaging["license"],
        "source": f"https://huggingface.co/datasets/nvidia/simready-assets/tree/{REVISION}/{package}",
    }
    (destination / "acquisition.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
