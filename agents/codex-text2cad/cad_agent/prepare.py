"""Bakeable installation and cheap per-trial configuration; no inference calls."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import shlex
import subprocess
import sys
import urllib.request
import venv

from .workflow import PACKAGE, verify_bundle, verify_generation


def runtime_env(prefix, state=None):
    prefix = Path(prefix)
    explicit_state = state is not None
    state = Path(state or Path.home() / ".cache/codex-text2cad-state")
    env = dict(os.environ)
    lib = str(prefix / "syslibs/usr/lib/x86_64-linux-gnu")
    env.update(CADGEN_TELEMETRY="0", DO_NOT_TRACK="1", CADGEN_UPDATE_CHECK="0",
               LD_LIBRARY_PATH=lib + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""))
    for name, directory in (("CADGEN_STATE_DIR", "state"), ("CADGEN_CACHE_DIR", "cache"),
                            ("CADGEN_ENGINE_DIR", "engine")):
        if explicit_state or name not in env:
            env[name] = str(state / directory)
    env.setdefault("XDG_CACHE_HOME", str(state / "xdg-cache"))
    return env


def _run(args, *, env=None):
    # Installation output goes to stderr, preserving ACP stdout for protocol.
    result = subprocess.run([str(x) for x in args], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    sys.stderr.write(re.sub(r"([?&]token=)[^\s]+", r"\1[REDACTED]", result.stdout))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, [str(x) for x in args])


def install(prefix):
    pins = verify_bundle()
    prefix = Path(prefix).resolve()
    prefix.mkdir(parents=True, exist_ok=True)
    if platform.machine() not in ("x86_64", "amd64") or platform.system() != "Linux":
        raise RuntimeError("Pinned runtime requires Linux amd64")
    if platform.libc_ver()[0] != "glibc" or tuple(map(int, platform.libc_ver()[1].split(".")[:2])) < (2, 38):
        raise RuntimeError("Pinned native libraries require glibc >= 2.38")
    lock = PACKAGE / "runtime.requirements.txt"
    expected = hashlib.sha256(lock.read_bytes()).hexdigest()
    marker = prefix / "installed.json"
    if marker.exists() and json.loads(marker.read_text()) == {"requirements_sha256": expected, "bundle": pins}:
        verify_bundle(prefix / "cad_agent")
        return
    bundled = prefix / "cad_agent"
    if bundled.resolve() != PACKAGE.resolve():
        shutil.copytree(PACKAGE, bundled, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    runtime = prefix / "runtime"
    if not (runtime / "bin/python").exists():
        venv.EnvBuilder(with_pip=True).create(runtime)
    _run([runtime / "bin/python", "-m", "pip", "install", "--require-hashes", "-r", lock])
    # No editable installation or dependency resolution inside the runtime.
    site = subprocess.check_output([str(runtime / "bin/python"), "-c", "import site; print(site.getsitepackages()[0])"], text=True).strip()
    Path(site, "cad-agent.pth").write_text(str(prefix) + "\n")
    for package in json.loads((PACKAGE / "system-runtime-packages.json").read_text())["packages"]:
        archive = prefix / package["url"].rsplit("/", 1)[1]
        if not archive.exists():
            with urllib.request.urlopen(package["url"], timeout=60) as response:
                archive.write_bytes(response.read())
        content = archive.read_bytes()
        if len(content) != package["size"] or hashlib.sha256(content).hexdigest() != package["sha256"]:
            raise RuntimeError(f"Invalid pinned library archive: {package['name']}")
        _run(["dpkg-deb", "-x", archive, prefix / "syslibs"])
    (prefix / "bin").mkdir(exist_ok=True)
    launcher = prefix / "bin/cad-python"
    launcher.write_text("#!/bin/sh\n" +
        "export LD_LIBRARY_PATH=" + shlex.quote(str(prefix / "syslibs/usr/lib/x86_64-linux-gnu")) + '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' +
        'export CADGEN_TELEMETRY=0 DO_NOT_TRACK=1 CADGEN_UPDATE_CHECK=0\n' +
        'export CADGEN_STATE_DIR="${CADGEN_STATE_DIR:-$HOME/.cache/codex-text2cad-state/state}"\n' +
        'export CADGEN_CACHE_DIR="${CADGEN_CACHE_DIR:-$HOME/.cache/codex-text2cad-state/cache}"\n' +
        'export CADGEN_ENGINE_DIR="${CADGEN_ENGINE_DIR:-$HOME/.cache/codex-text2cad-state/engine}"\n' +
        "exec " + shlex.quote(str(runtime / "bin/python")) + ' "$@"\n')
    launcher.chmod(0o755)
    preflight = prefix / "preflight"
    preflight.mkdir(exist_ok=True)
    shutil.copyfile(PACKAGE / "smoke.py", preflight / "smoke.py")
    env = runtime_env(prefix, preflight)
    _run([runtime / "bin/cadgen", "doctor"], env=env)
    _run([runtime / "bin/python", preflight / "smoke.py"], env=env)
    receipt = json.loads((preflight / "headless-receipt.json").read_text())
    if not receipt["valid"] or receipt["cadgen"] != "0.8.2" or receipt["build123d"] != "0.13.0":
        raise RuntimeError("CAD headless preflight failed")
    marker.write_text(json.dumps({"requirements_sha256": expected, "bundle": pins}, indent=2) + "\n")


def configure_trial(prefix, *, home=None, logs=None):
    """Always called, even when Harbor skips the bakeable install hook."""
    pins = verify_bundle()
    prefix = Path(prefix).resolve()
    installed = json.loads((prefix / "installed.json").read_text())
    if installed["bundle"] != pins:
        raise RuntimeError("Baked CAD bundle differs from agent bundle")
    verify_bundle(prefix / "cad_agent")
    home = Path(home or Path.home())
    target = home / ".agents/skills/cad"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PACKAGE / "vendor/cad", target, dirs_exist_ok=True)
    for rel, digest in pins["files"].items():
        if rel.startswith("vendor/cad/"):
            native = target / rel.removeprefix("vendor/cad/")
            if hashlib.sha256(native.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Native skill registration failed: {rel}")
    logs = Path(logs or "/logs/agent/cad")
    logs.mkdir(parents=True, exist_ok=True)
    env = runtime_env(prefix)
    # Import the real OCP-backed modeler and USD runtime before starting Codex.
    _run([prefix / "runtime/bin/python", "-c",
          "import build123d, cadgen; from pxr import Usd; assert build123d.Box(1,1,1).is_valid"], env=env)
    receipt = {"bundle": pins, "native_skill": str(target), "readiness": True,
               "skill_registration_verified": True, "workflow_injection": "configured",
               "generation_use": "not_yet_verified"}
    (logs / "readiness.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "trial", "verify", "mcp"))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--logs", default="/logs/agent/cad")
    args = parser.parse_args()
    if args.action == "install":
        install(args.prefix)
    elif args.action == "trial":
        configure_trial(args.prefix, logs=args.logs)
    elif args.action == "verify":
        result = verify_generation("/output/authoring/cad-evidence/generation.json", "/output/submission/asset.usd")
        logs = Path(args.logs)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "use.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    else:
        command = Path(args.prefix) / "runtime/bin/cadgen"
        os.execve(str(command), [str(command), "mcp"], runtime_env(args.prefix))


if __name__ == "__main__":
    main()
