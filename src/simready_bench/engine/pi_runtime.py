"""Read-only integrity check for the fixed actor Pi closure (stdlib only)."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

EXPECTED_VERSIONS = {"node": "v22.23.3", "npm": "10.9.9", "pi": "0.85.1"}
EXPECTED_SHA256 = {
    "/opt/nvm/nvm.sh": "9290aec6cc2efc89273e16a77214eb37303caaff8c943102de5116fdef47ba86",
    "/opt/nvm/versions/node/v22.23.3/bin/node": "fde6a4bf8d0562f7751d1a2d6cb9b417c4cfe107bbcb0aa3e9a24e125e348f48",
    "/opt/simready/pi/node_modules/@earendil-works/pi-coding-agent/dist/cli.js": "8189b66abc4f9f431dbb70941dcba690d76d040de1fbfff212886be35a53639d",
    "/tmp/harbor-pi-mcp-adapter/node_modules/pi-mcp-adapter/index.ts": "608143846c31fd3b109dacf8c54095b5f9faffa27ae4b106f5bb0a3fcb6d5281",
}


def verify_file(path, expected_sha256, *, owner_uid=0):
    """Hash a genuine regular file protected from actor replacement."""
    path = Path(path).absolute()
    try:
        if path.resolve(strict=True) != path:
            raise RuntimeError("Frozen runtime file contains a symlink")
        for parent in path.parents:
            info = parent.stat()
            sticky_root_tmp = (
                parent == Path("/tmp")
                and info.st_uid == 0
                and info.st_mode & stat.S_ISVTX
            )
            if info.st_uid not in (0, owner_uid) or (
                info.st_mode & 0o022 and not sticky_root_tmp
            ):
                raise RuntimeError("Frozen runtime ancestor is actor-writable")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise RuntimeError("Frozen runtime input must be a regular file")
            if info.st_uid != owner_uid or info.st_mode & 0o022:
                raise RuntimeError(
                    "Frozen runtime file ownership or writable permissions changed"
                )
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeError("Frozen runtime file is unavailable") from error
    if digest.hexdigest() != expected_sha256:
        raise RuntimeError("Frozen runtime digest mismatch")
    return digest.hexdigest()


def check_runtime():
    if sys.platform != "linux" or os.getuid() != 1000:
        raise RuntimeError("Frozen actor runtime must execute as Linux UID 1000")
    expected_node = "/opt/nvm/versions/node/v22.23.3/bin/node"
    if Path(shutil.which("node") or "/missing").resolve() != Path(expected_node):
        raise RuntimeError("Frozen Node executable is not selected")
    hashes = {
        path: verify_file(path, digest) for path, digest in EXPECTED_SHA256.items()
    }
    versions = {}
    for command, expected in EXPECTED_VERSIONS.items():
        result = subprocess.run(
            [command, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
        )
        actual = result.stdout.strip()
        if actual != expected:
            raise RuntimeError("Frozen runtime version mismatch: " + command)
        versions[command] = actual
    return {
        "status": "frozen_pi_runtime_verified",
        "uid": os.getuid(),
        "versions": versions,
        "sha256": hashes,
    }


if __name__ == "__main__":
    print(json.dumps(check_runtime(), sort_keys=True))
