"""Container images used by local and Daytona execution."""

import json
import re
from pathlib import Path

ACTOR_IMAGE = "sha256:54e91c261dec34326a3fd6b59fc1d4f01244871672a85fd0a80d521a1b909af5"
VERIFIER_IMAGE = (
    "sha256:5ac7024420678fe60e8566bc63730613192006169eefb2fb25b23cddfe6abd14"
)
CPU_IMAGE = "sha256:f92c3f24a7871648bc74c74eaa54443702b7003ec5c9cf66ab319c6b182007d8"
CPU_SNAPSHOT = "simready-bench-cpu"
DEFAULT_IMAGES = {"actor": ACTOR_IMAGE, "verifier": VERIFIER_IMAGE, "cpu": CPU_IMAGE}


def load_images(path=None):
    if path is None:
        return dict(DEFAULT_IMAGES)
    source = Path(path).absolute()
    if (
        source.resolve() != source
        or not source.is_file()
        or source.stat().st_size > 65536
    ):
        raise ValueError("Choose a regular runtime image configuration file")
    images = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(images, dict)
        or set(images) - {"actor", "verifier", "cpu", "snapshot"}
        or not set(DEFAULT_IMAGES) <= set(images)
    ):
        raise ValueError(
            "The runtime image configuration requires actor, verifier and cpu images"
        )
    for role in DEFAULT_IMAGES:
        if not isinstance(images[role], str) or not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[a-f0-9]{64}", images[role]
        ):
            raise ValueError("Each runtime image must use a SHA-256 digest")
    if "snapshot" in images and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", images["snapshot"]
    ):
        raise ValueError("Choose a literal Daytona snapshot name")
    return images
