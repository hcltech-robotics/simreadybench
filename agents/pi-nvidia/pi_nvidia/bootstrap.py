"""Locked public runtime installation; no provider keys or auth files are read."""

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).parent


def install(prefix):
    prefix = Path(prefix).resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This hosted runtime is pinned for Linux x86_64")
    pin = json.loads((ROOT / "node.json").read_text())
    node = prefix / "node/bin/node"
    if not node.exists():
        archive = prefix / "node.tar.xz"
        with urllib.request.urlopen(pin["url"], timeout=60) as response:
            archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["sha256"]:
            raise RuntimeError("Node checksum mismatch")
        with tarfile.open(archive) as tar:
            top = tar.getnames()[0].split("/", 1)[0]
            tar.extractall(prefix / "node-unpack", filter="data")
        shutil.move(prefix / "node-unpack" / top, prefix / "node")
    if subprocess.check_output([str(node), "--version"], text=True).strip() != "v" + pin["version"]:
        raise RuntimeError("Node version mismatch")
    env = dict(os.environ)
    env["PATH"] = str(node.parent) + os.pathsep + env.get("PATH", os.defpath)
    env["NPM_CONFIG_CACHE"] = str(prefix / "npm-cache")
    env["NPM_CONFIG_USERCONFIG"] = "/dev/null"
    empty = prefix / "empty-global.npmrc"
    empty.touch()
    env["NPM_CONFIG_GLOBALCONFIG"] = str(empty)
    digest = hashlib.sha256((ROOT / "package-lock.json").read_bytes()).hexdigest()
    marker = prefix / "npm-installed.sha256"
    if not marker.exists() or marker.read_text() != digest:
        for name in ("package.json", "package-lock.json"):
            shutil.copyfile(ROOT / name, prefix / name)
        subprocess.run([str(node.parent / "npm"), "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--prefix", str(prefix)],
                       env=env, stdout=sys.stderr, stderr=sys.stderr, check=True)
        marker.write_text(digest)
    for name, version in (("@earendil-works/pi-coding-agent", "1.4.0"), ("pi-acp", "0.0.34")):
        actual = json.loads((prefix / "node_modules" / name / "package.json").read_text())["version"]
        if actual != version:
            raise RuntimeError("Pinned Pi/ACP version mismatch")
    return node
