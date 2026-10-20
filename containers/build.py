"""Build the actor, verifier and CPU runtime images."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROLES = ("actor", "verifier", "cpu")


def prepare_context(role, destination, *, wheel=None):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a new build context directory")
    destination.mkdir(parents=True, mode=0o700)
    if role in ("actor", "verifier"):
        source = ROOT / "actor"
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Container sources cannot contain symbolic links")
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        engine = ROOT.parent / "src/simready_bench/engine"
        for name in (
            "actor_mcp.py",
            "actor_relay.py",
            "pi_runtime.py",
            "verifier.py",
            "nvidia_profile.json",
        ):
            path = engine / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("Missing container client source: " + name)
            shutil.copyfile(path, destination / name)
    elif role == "cpu":
        wheel = Path(wheel).absolute() if wheel else None
        if (
            wheel is None
            or wheel.is_symlink()
            or not wheel.is_file()
            or not re.fullmatch(
                r"simready_bench-[A-Za-z0-9.]+-py3-none-any.whl", wheel.name
            )
        ):
            raise ValueError("Pass --wheel with the built SimReadyBench wheel")
        shutil.copyfile(ROOT / "daytona/Dockerfile.cpu", destination / "Dockerfile")
        shutil.copyfile(ROOT.parent / "locks/runtime.txt", destination / "runtime.txt")
        (destination / "wheels").mkdir()
        shutil.copyfile(wheel, destination / "wheels" / wheel.name)
    else:
        raise ValueError("Choose actor, verifier or cpu")
    return destination


def build_command(role, *, tag, context, builder=None):
    command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        "linux/amd64",
        "--progress",
        "plain",
    ]
    if builder:
        command += ["--builder", builder]
    if role in ("actor", "verifier"):
        command += ["--target", "runtime" if role == "actor" else "verifier"]
    return [*command, "--load", "--tag", tag, str(context)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=ROLES)
    parser.add_argument("--tag")
    parser.add_argument("--builder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args(argv)
    tag = args.tag or "simreadybench/" + args.role + ":1.0.0"
    if not re.fullmatch(r"[a-z0-9][a-z0-9._/:~-]{1,200}", tag):
        parser.error("Choose a literal Docker image tag")
    output = args.output.absolute()
    if output.exists() or output.resolve() != output or output.is_relative_to(ROOT):
        parser.error("Choose a fresh build directory outside containers/")
    if args.role == "cpu" and args.wheel is None:
        parser.error("CPU builds require --wheel")
    context = output / "context"
    command = build_command(args.role, tag=tag, context=context, builder=args.builder)
    output.mkdir(parents=True, mode=0o700)
    prepare_context(args.role, context, wheel=args.wheel)
    subprocess.run(command, check=True)
    image = subprocess.check_output(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True
    ).strip()
    print(json.dumps({"role": args.role, "tag": tag, "image": image}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
