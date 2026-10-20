"Stage reference packages and generate Harbor tasks."

from __future__ import annotations
import copy, hashlib, json, os, shutil, socket
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(384)


def checked_resolved(resolved, policy):
    """Re-resolve declarative inputs before using a supplied plan as authority."""
    from simready_bench.config import resolve_experiment

    manifest = copy.deepcopy(resolved["normalized_manifest"])
    for asset in manifest["assets"]:
        for key in ("closure", "usd_path", "input_view_closure"):
            asset.pop(key, None)
    for skill in manifest["skill_sets"]:
        skill.pop("closure", None)
    fresh = resolve_experiment(manifest, base_dir=Path.cwd(), policy=policy)
    if (
        fresh["configuration_sha256"] != resolved["configuration_sha256"]
        or fresh != resolved
    ):
        raise ValueError(
            "Resolved configuration, trusted policy or source closure changed"
        )
    return fresh


def runtime_policy(resolved, *, execute=False):
    runtime = resolved["operator_policy"]["runtime"]
    required = {
        "repository_root",
        "source_root",
        "validator_python",
        "expected_hostname",
        "allowed_reference_root",
        "shared_profile",
    }
    if not required <= set(runtime):
        raise ValueError(
            "Operator must bind repository, source, validator, host, reference root and shared profile"
        )
    for key in (
        "repository_root",
        "source_root",
        "validator_python",
        "allowed_reference_root",
    ):
        path = Path(runtime[key]).absolute()
        if not path.exists() or (key != "validator_python" and path.resolve() != path):
            raise ValueError("Existing trusted runtime path required: " + key)
    if execute:
        if runtime.get("execution_enabled") is not True:
            raise RuntimeError(
                "Runtime execution must be enabled in the operator policy"
            )
        if socket.gethostname() != runtime["expected_hostname"]:
            raise RuntimeError("Execution host differs from trusted operator binding")
    return runtime


def confined_work_dir(path, *, fresh=False, ram_only=False):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError("No symlink work directory")
    if ram_only and (not path.is_relative_to("/dev/shm")):
        raise ValueError("Execution outputs must be RAM-only under /dev/shm")
    if path.exists() and (
        not path.is_dir()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_mode & 63
    ):
        raise ValueError(
            "Existing work directory must be operator-owned and private mode 0700"
        )
    if fresh and path.exists():
        raise FileExistsError("Preserve existing work directory")
    if shutil.disk_usage("/").free < 5 * 1024**3:
        raise RuntimeError("Five-GiB root filesystem reserve required")
    return path


def verify_closure(records):
    for record in records:
        path = Path(record["absolute_path"]).absolute()
        if path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError("Regular confined source files required")
        raw = path.read_bytes()
        if (
            len(raw) != record["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != record["sha256"]
        ):
            raise ValueError("Source closure changed")


def _copy_resolved(record, target):
    """Bind the independent copy to the already-resolved source bytes."""
    shutil.copyfile(record["absolute_path"], target)
    copied = {**record, "absolute_path": str(Path(target).absolute())}
    verify_closure([copied])


def _reference_input(asset, root, *, package_root=None):
    """Copy the declared private closure; never expose this directory to actors."""
    from simready_bench.harnesses import _relative

    verify_closure(asset["closure"])
    verify_closure(asset["input_view_closure"])
    package = Path(package_root) if package_root is not None else root / "package"
    package.mkdir(parents=True, mode=448)
    for record in asset["closure"]:
        relative = _relative(record["path"])
        target = package.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=448)
        _copy_resolved(record, target)
    source = package.joinpath(*_relative(asset["usd"]).parts)
    image = None
    if asset["input_view_closure"]:
        if len(asset["input_view_closure"]) != 1:
            raise ValueError(
                "Initial native Pi input supports exactly one supplied PNG; use no input for derived reference view"
            )
        image = root / "supplied-input.png"
        _copy_resolved(asset["input_view_closure"][0], image)
    verify_closure(asset["closure"])
    verify_closure(asset["input_view_closure"])
    return {
        "usd": str(source),
        "package_root": str(package),
        "input_png": str(image) if image else None,
        "asset_id": asset["id"],
        "task_id": "asset_" + hashlib.sha256(asset["id"].encode()).hexdigest()[:16],
        "public_task": {
            "brief": "Reconstruct the object shown in /task/thumbnail.png as a canonical metre, kilogram, Z-up USD asset.",
            "required_structure": "Match the visible object structure in the supplied image.",
            **(
                {"original_nominal_dimensions_m": asset["nominal_dimensions_m"]}
                if "nominal_dimensions_m" in asset
                else {}
            ),
        },
    }


def prepare_assets(resolved, *, work_dir, policy):
    from simready_bench.assets import prepare_validator_assets

    resolved = checked_resolved(resolved, policy)
    runtime = runtime_policy(resolved, execute=True)
    return prepare_validator_assets(resolved, work_dir=work_dir, runtime=runtime)


def check_prepared(root, resolved):
    root = Path(root).absolute()
    record = json.loads((root / "prepared.json").read_text())
    if root.resolve() != root or any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Regular prepared private tree required")
    if (
        record["configuration_sha256"] != resolved["configuration_sha256"]
        or record["policy_sha256"] != resolved["policy_sha256"]
    ):
        raise ValueError("Prepared configuration/policy differs")
    required = {
        "schema",
        "configuration_sha256",
        "policy_sha256",
        "profile_sha256",
        "conditions",
        "provider_calls",
        "official_success_claim",
        "files",
    }
    if (
        set(record) != required
        or record["schema"] != "simready-bench-prepared-v1"
        or record["provider_calls"] != 0
        or record["official_success_claim"] is not False
    ):
        raise ValueError("Exact prepared manifest schema required")
    profile = resolved["operator_policy"]["runtime"].get("shared_profile")
    if not isinstance(profile, dict):
        raise ValueError("Prepared task requires trusted immutable shared profile")
    if record["profile_sha256"] != profile["profile_sha256"]:
        raise ValueError("Prepared immutable profile differs")
    conditions = record["conditions"]
    if not isinstance(conditions, list) or [
        c.get("condition_id") for c in conditions
    ] != [c["condition_id"] for c in resolved["conditions"]]:
        raise ValueError("Exact ordered resolved condition identity required")
    for row, condition in zip(conditions, resolved["conditions"]):
        keys = {
            "condition_id",
            "asset_id",
            "task_path",
            "private_config",
            "task_id",
            "actor_image",
            "verifier_image",
        }
        task_id = (
            "asset_" + hashlib.sha256(condition["asset_id"].encode()).hexdigest()[:16]
        )
        if (
            set(row) != keys
            or row["asset_id"] != condition["asset_id"]
            or row["task_id"] != task_id
        ):
            raise ValueError("Prepared asset/task identity differs")
        base = root / "conditions" / condition["condition_id"]
        expected = {
            "task_path": base / "tasks" / task_id,
            "private_config": base / "private" / (task_id + ".json"),
        }
        for key, path in expected.items():
            if row[key] != str(path) or path.resolve() != path or not path.exists():
                raise ValueError("Prepared task/private paths must remain confined")
        if (
            row["actor_image"] != profile["actor_image"]
            or row["verifier_image"] != profile["verifier_image"]
        ):
            raise ValueError("Prepared image profile differs")
    actual = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "prepared.json"
    }
    if actual != record["files"]:
        raise ValueError("Prepared reference/task bytes changed")
    return record
