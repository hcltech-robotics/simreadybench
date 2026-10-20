"""Import a published runtime into a Daytona organisation."""

from importlib.resources import files
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA = "simready-bench-runtime-release/v1"
RESOURCE = "resources/runtimes/daytona.json"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_IMAGE = re.compile(r"ghcr\.io/[a-z0-9._-]+/[a-z0-9._/-]+@sha256:[0-9a-f]{64}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def validate_release(value):
    """Validate the immutable outer image and the image IDs it contains."""
    if not isinstance(value, dict) or set(value) != {
        "schema", "version", "platform", "snapshot_image", "images"
    }:
        raise ValueError("A runtime release requires schema, version, platform, snapshot_image and images")
    if value["schema"] != SCHEMA or value["platform"] != "linux/amd64":
        raise ValueError("Choose a SimReadyBench linux/amd64 runtime release")
    if not isinstance(value["version"], str) or not _VERSION.fullmatch(value["version"]):
        raise ValueError("The runtime release requires a literal version")
    image = value["snapshot_image"]
    if not isinstance(image, str) or not _IMAGE.fullmatch(image) or any(
        part in ("", ".", "..") for part in image.split("@", 1)[0].split("/")[1:]
    ):
        raise ValueError("The public snapshot image must use a ghcr.io SHA-256 digest")
    images = value["images"]
    if not isinstance(images, dict) or set(images) != {"actor", "verifier", "cpu"} or any(
        not isinstance(image, str) or not _DIGEST.fullmatch(image)
        for image in images.values()
    ):
        raise ValueError("The runtime release requires exact actor, verifier and cpu image IDs")
    return {**value, "images": dict(images)}


def _regular_absolute(path):
    original = Path(path).absolute()
    if any(part.is_symlink() for part in (original, *original.parents)):
        raise ValueError("Choose a configuration path without symbolic links")
    normalised = Path(os.path.abspath(original))
    if normalised.resolve() != normalised:
        raise ValueError("Choose a configuration path below a regular directory")
    return normalised


def _read_json(path):
    source = _regular_absolute(path)
    if not source.is_file() or source.stat().st_size > 65536:
        raise ValueError("Choose a regular JSON configuration file below 64 KiB")
    return json.loads(source.read_text(encoding="utf-8"))


def load_release(path=None):
    """Read an explicit release or the release shipped with this installation."""
    if path is not None:
        return validate_release(_read_json(path))
    resource = files("simready_bench").joinpath(RESOURCE)
    if not resource.is_file():
        raise ValueError("The installed package has no published Daytona runtime. Pass --release-manifest PATH")
    raw = resource.read_bytes()
    if len(raw) > 65536:
        raise ValueError("The packaged runtime release exceeds 64 KiB")
    return validate_release(json.loads(raw))


def _confirm_snapshot(snapshot, *, name, image):
    if snapshot.name != name or snapshot.image_name != image:
        raise ValueError("The named Daytona snapshot uses another image. Choose a different --snapshot name")
    state = getattr(snapshot.state, "value", snapshot.state)
    if state != "active":
        raise RuntimeError("The Daytona snapshot is not active. Check its state and repeat setup when it is ready")


def _save_mapping(target, mapping):
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".runtime-images-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write((json.dumps(mapping, ensure_ascii=True, indent=2) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, target)
    except FileExistsError:
        if _read_json(target) != mapping:
            raise FileExistsError("Preserving a runtime mapping written during setup") from None
    finally:
        temporary.unlink(missing_ok=True)


def setup_daytona(*, output, release_manifest=None, snapshot=None, cpu=2, memory=6, disk=10, timeout=600):
    """Import or reuse a matching active snapshot and save its runtime mapping."""
    release = load_release(release_manifest)
    name = snapshot or "simready-bench-cpu-" + release["version"].replace("+", "-")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("Choose a literal Daytona --snapshot name")
    if any(type(value) is not int or value <= 0 for value in (cpu, memory, disk, timeout)):
        raise ValueError("CPU, memory, disk and timeout must be positive integers")
    target = _regular_absolute(output)
    mapping = {**release["images"], "snapshot": name}
    if target.exists() and _read_json(target) != mapping:
        raise FileExistsError("Preserving another runtime mapping. Choose a new --output path")
    key = os.environ.get("DAYTONA_API_KEY")
    if not key:
        raise ValueError("Set DAYTONA_API_KEY before importing the runtime")
    try:
        from daytona import (
            CreateSnapshotParams, Daytona, DaytonaConfig, DaytonaConflictError,
            DaytonaError, DaytonaNotFoundError, Resources,
        )
    except ImportError as error:
        raise ImportError("Install SimReadyBench with its daytona extra before importing the runtime") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    image = release["snapshot_image"]
    status = "reused"
    try:
        client = Daytona(DaytonaConfig(api_key=key))
        try:
            selected = client.snapshot.get(name)
        except DaytonaNotFoundError:
            try:
                selected = client.snapshot.create(
                    CreateSnapshotParams(
                        name=name, image=image,
                        resources=Resources(cpu=cpu, memory=memory, disk=disk),
                    ),
                    timeout=timeout,
                )
                status = "created"
            except DaytonaConflictError:
                selected = client.snapshot.get(name)
        _confirm_snapshot(selected, name=name, image=image)
    except DaytonaError as error:
        raise RuntimeError("Daytona runtime setup failed: " + str(error).replace(key, "[redacted]")) from error
    _save_mapping(target, mapping)
    return {
        "schema": "simready-bench-runtime-setup/v1",
        "release_version": release["version"],
        "snapshot": name,
        "snapshot_id": selected.id,
        "snapshot_image": image,
        "status": status,
        "runtime_images": str(target),
    }
