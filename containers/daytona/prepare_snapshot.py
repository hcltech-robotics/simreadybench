"""Prepare Docker images and their runtime mapping for a Daytona snapshot."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path


def _exported_images(archive, tags):
    index = None
    index_seen = False
    blobs = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            if member.name == "index.json":
                if index_seen or not member.isfile() or member.size > 65536:
                    raise ValueError("The exported archive requires one bounded OCI index.json")
                index_seen = True
                index = json.load(stream.extractfile(member))
            elif re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", member.name) and member.size <= 1024**2:
                if not member.isfile() or member.name in blobs:
                    raise ValueError("The exported archive contains a duplicate or invalid OCI blob")
                raw = stream.extractfile(member).read()
                blobs[member.name] = ("sha256:" + hashlib.sha256(raw).hexdigest(), len(raw))
    if not isinstance(index, dict) or index.get("schemaVersion") != 2 or not isinstance(index.get("manifests"), list):
        raise ValueError("The exported archive requires an OCI image index")
    names = {name: role for role, tag in tags.items() for name in (tag, "docker.io/" + tag)}
    selected = {}
    for descriptor in index["manifests"]:
        if not isinstance(descriptor, dict) or not isinstance(descriptor.get("annotations"), dict):
            raise ValueError("Each exported image requires its exact role tag")
        name = descriptor["annotations"].get("io.containerd.image.name")
        role = names.get(name) if isinstance(name, str) else None
        if role is None or role in selected:
            raise ValueError("The exported image index has an unexpected or duplicate role tag")
        digest = descriptor.get("digest")
        size = descriptor.get("size")
        if (
            descriptor.get("mediaType") not in (
                "application/vnd.oci.image.manifest.v1+json", "application/vnd.oci.image.index.v1+json",
                "application/vnd.docker.distribution.manifest.v2+json", "application/vnd.docker.distribution.manifest.list.v2+json",
            )
            or not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
            or type(size) is not int or not 0 < size <= 1024**2
            or blobs.get("blobs/sha256/" + digest.removeprefix("sha256:")) != (digest, size)
        ):
            raise ValueError("An exported image descriptor does not match its OCI manifest blob")
        selected[role] = digest
    if set(selected) != set(tags):
        raise ValueError("The exported archive is missing an actor, verifier or cpu role tag")
    return {role: selected[role] for role in tags}


def prepare(*, actor_image, verifier_image, cpu_image, output):
    root = Path(output).absolute()
    if root.exists() or root.resolve() != root:
        raise FileExistsError("Choose a new snapshot build directory")
    selected = {}
    for role, image in (
        ("actor", actor_image),
        ("verifier", verifier_image),
        ("cpu", cpu_image),
    ):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{1,255}", image):
            raise ValueError("Choose literal Docker image names")
        identifier = subprocess.check_output(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
        ).strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", identifier):
            raise ValueError("Docker did not return an immutable image identity")
        selected[role] = identifier
    tags = {role: "simreadybench/snapshot-" + role + ":" + identifier.removeprefix("sha256:") for role, identifier in selected.items()}
    for role, tag in tags.items():
        subprocess.run(["docker", "image", "tag", selected[role], tag], check=True)
    root.mkdir(parents=True, mode=0o700)
    recipe = Path(__file__).parent
    for name in ("bootstrap.sh", "snapshot.Dockerfile"):
        shutil.copyfile(recipe / name, root / name)
    process = subprocess.Popen(
        ["docker", "save", *tags.values()],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with (root / "images.tar.gz").open("xb") as output_stream:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=output_stream, mtime=0, compresslevel=1
            ) as compressed:
                shutil.copyfileobj(process.stdout, compressed, length=1024**2)
        error = process.stderr.read()
        if process.wait():
            raise RuntimeError(
                "Docker image export failed: "
                + error.decode("utf-8", "replace")[-1000:]
            )
    finally:
        process.stdout.close()
        process.stderr.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    runtime = _exported_images(root / "images.tar.gz", tags)
    for name, images in (("runtime-images.json", runtime), ("local-runtime-images.json", selected)):
        (root / name).write_text(json.dumps(images, indent=2) + "\n", encoding="utf-8")
    (root / "images.txt").write_text(
        "\n".join(dict.fromkeys(runtime.values())) + "\n", encoding="ascii"
    )
    return runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-image", required=True)
    parser.add_argument("--verifier-image", required=True)
    parser.add_argument("--cpu-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            prepare(
                actor_image=args.actor_image,
                verifier_image=args.verifier_image,
                cpu_image=args.cpu_image,
                output=args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
