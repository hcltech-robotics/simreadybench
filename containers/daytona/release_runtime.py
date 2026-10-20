"""Build and publish the complete versioned Daytona runtime image."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid

from prepare_snapshot import prepare


ROOT = Path(__file__).resolve().parents[2]


def run(arguments):
    subprocess.run([str(value) for value in arguments], check=True)


def capture(arguments):
    return subprocess.check_output([str(value) for value in arguments], text=True).strip()


def verify_runtime(image, images, output, *, timeout=600):
    """Load and execute every embedded image with the pinned consumer engine."""
    name = "simready-runtime-check-" + uuid.uuid4().hex
    started = time.monotonic()
    try:
        run(["docker", "run", "--detach", "--privileged", "--network", "none", "--cpus", "2", "--memory", "6g", "--name", name, image])
        while True:
            ready = subprocess.run(
                ["docker", "exec", name, "docker", "info"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
            )
            if ready.returncode == 0:
                break
            running = capture(["docker", "inspect", name, "--format", "{{.State.Running}}"])
            if running != "true" or time.monotonic() - started >= timeout:
                raise RuntimeError("The runtime consumer did not finish loading its images")
            time.sleep(2)
        for role, identifier in images.items():
            actual = capture(["docker", "exec", name, "docker", "image", "inspect", identifier, "--format", "{{.Id}}"])
            if actual != identifier:
                raise RuntimeError("The imported " + role + " image has another identity")
            run(["docker", "exec", name, "docker", "run", "--rm", "--network", "none", identifier, "/bin/true"])
        receipt = {"schema": "simready-bench-runtime-consumer-check/v1", "status": "passed", "images": images}
        (output / "consumer-check.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt
    finally:
        try:
            logs = subprocess.run(["docker", "logs", name], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30)
            (output / "consumer-check.log").write_bytes(logs.stdout)
        finally:
            subprocess.run(["docker", "rm", "--force", "--volumes", name], check=False, timeout=30)


def publish(*, repository, version, wheel, output, revision):
    if not re.fullmatch(r"ghcr\.io/[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*", repository):
        raise ValueError("Choose a literal GHCR owner and image name")
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9.-]+)?", version):
        raise ValueError("Choose a versioned runtime release")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Record the source commit used to build the runtime")
    wheel = Path(wheel).absolute()
    if not wheel.is_file() or wheel.is_symlink() or not re.fullmatch(r"simready_bench-[A-Za-z0-9.]+-py3-none-any\.whl", wheel.name):
        raise ValueError("Select a built SimReadyBench wheel")
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("Choose a fresh output directory below a regular parent")
    image = repository + ":" + version
    existing = subprocess.run(["docker", "manifest", "inspect", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if existing.returncode == 0:
        raise ValueError("This runtime version is already published. Choose a new version")
    output.mkdir(mode=0o700)
    tags = {role: "simreadybench/" + role + ":" + version for role in ("actor", "verifier", "cpu")}
    for role in ("actor", "verifier", "cpu"):
        arguments = [sys.executable, ROOT / "containers/build.py", role, "--tag", tags[role], "--output", output / role]
        if role == "cpu":
            arguments += ["--wheel", wheel]
        run(arguments)
    storm = output / "storm"
    run([sys.executable, ROOT / "containers/daytona/prepare_storm.py", "--output", storm])
    storm_tag = "simreadybench/cpu:storm-" + version
    run(["docker", "buildx", "build", "--platform", "linux/amd64", "--load", "--progress", "plain", "--build-arg", "CPU_BASE=" + tags["cpu"], "--tag", storm_tag, storm])
    snapshot = output / "snapshot"
    images = prepare(actor_image=tags["actor"], verifier_image=tags["verifier"], cpu_image=storm_tag, output=snapshot)
    run([
        "docker", "buildx", "build", "--platform", "linux/amd64", "--load", "--progress", "plain",
        "--file", snapshot / "snapshot.Dockerfile", "--tag", image,
        "--label", "org.opencontainers.image.source=https://github.com/hcltech-robotics/simreadybench",
        "--label", "org.opencontainers.image.revision=" + revision,
        "--label", "org.opencontainers.image.version=" + version,
        snapshot,
    ])
    verify_runtime(image, images, output)
    run(["docker", "push", image])
    digests = json.loads(capture(["docker", "image", "inspect", image, "--format", "{{json .RepoDigests}}"] ))
    pinned = [value for value in digests if re.fullmatch(re.escape(repository) + r"@sha256:[a-f0-9]{64}", value)]
    if len(pinned) != 1:
        raise ValueError("The registry did not return one immutable release digest")
    release = {
        "schema": "simready-bench-runtime-release/v1",
        "version": version,
        "platform": "linux/amd64",
        "snapshot_image": pinned[0],
        "images": images,
    }
    (output / "runtime-release.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "source_revision": revision,
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "image_tag": image,
        "release": release,
    }
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    publish(**vars(args))


if __name__ == "__main__":
    main()
