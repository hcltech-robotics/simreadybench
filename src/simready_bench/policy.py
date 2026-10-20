"""Create private operator policies for local and Daytona execution."""

from __future__ import annotations
import json, os, socket, sys, tempfile
from pathlib import Path

DEFAULT_ACTOR_IMAGE = (
    "sha256:54e91c261dec34326a3fd6b59fc1d4f01244871672a85fd0a80d521a1b909af5"
)
DEFAULT_VERIFIER_IMAGE = (
    "sha256:5ac7024420678fe60e8566bc63730613192006169eefb2fb25b23cddfe6abd14"
)
from simready_bench.runtime_images import CPU_IMAGE as DEFAULT_RUNTIME_IMAGE


def _literal(path):
    text = os.fspath(path)
    if (
        not isinstance(text, str)
        or not text
        or any(c in text for c in ("$", "~"))
        or any(ord(c) < 32 for c in text)
        or ".." in Path(text).parts
    ):
        raise ValueError("Literal filesystem path required")
    return Path(text).absolute()


def installed_profile(
    *,
    actor_image=DEFAULT_ACTOR_IMAGE,
    verifier_image=DEFAULT_VERIFIER_IMAGE,
    runtime_image=DEFAULT_RUNTIME_IMAGE,
):
    """Use the engine's real builder and verify its seal against loaded code."""
    import simready_bench
    from simready_bench.engine.environment import (
        build_shared_profile,
        verify_profile_code,
    )

    package = Path(simready_bench.__file__).absolute().parent
    if package.resolve() != package:
        raise ValueError("Installed package must use a regular path")
    code = sorted(package.rglob("*.py")) + [
        package / "engine/nvidia_profile.json",
        package / "engine/nvidia_optional_profile.json",
    ]
    if any(p.resolve() != p or not p.is_file() for p in code):
        raise ValueError("Installed evaluator files must be regular files")
    with tempfile.TemporaryDirectory(
        prefix="simready-profile-", dir="/dev/shm"
    ) as temporary:
        staging = Path(temporary)
        files = []
        for source in code:
            relative = Path("src") / source.relative_to(package.parent)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            target.chmod(0o600)
            files.append(relative.as_posix())
        profile = build_shared_profile(
            repository_root=staging,
            actor_image=actor_image,
            verifier_image=verifier_image,
            runtime_image=runtime_image,
            code_files=files,
            version=simready_bench.__version__,
        )
        verify_profile_code(profile)
    return profile, str(package.parent)


def init_policy(
    *,
    output,
    workspace,
    assets,
    validator_python=None,
    actor_image=DEFAULT_ACTOR_IMAGE,
    verifier_image=DEFAULT_VERIFIER_IMAGE,
    runtime_image=DEFAULT_RUNTIME_IMAGE,
    providers=None,
    maximum_spend_usd=0,
    execution_enabled=False,
    authority_state_dir=None,
    native_output=False,
):
    """Create a private policy; execution requires an explicit caller allowance."""
    output = _literal(output)
    workspace = _literal(workspace)
    assets = _literal(assets)
    if (
        output.exists()
        or output.is_symlink()
        or workspace.exists()
        or workspace.is_symlink()
    ):
        raise FileExistsError("Preserving existing output/workspace")
    if workspace.parent.resolve() != workspace.parent or not workspace.parent.is_dir():
        raise ValueError("Workspace parent must be an existing non-symlink directory")
    if assets.resolve() != assets or not assets.is_dir():
        raise ValueError("Existing non-symlink asset directory required")
    if workspace.is_relative_to(assets) or assets.is_relative_to(workspace):
        raise ValueError("Assets and operator workspace must be disjoint")
    if (
        output == workspace
        or output.is_relative_to(assets)
        or any(
            output.is_relative_to(workspace / name)
            for name in (
                "input_views",
                "skills",
                "private_manifests",
                "private/references",
                "private/authority",
            )
        )
    ):
        raise ValueError(
            "Operator policy must remain outside asset/public/authority roots"
        )
    if not output.is_relative_to(workspace) and (
        output.parent.resolve() != output.parent or not output.parent.is_dir()
    ):
        raise ValueError("Output parent must be an existing non-symlink directory")
    interpreter = _literal(
        sys.executable if validator_python is None else validator_python
    )
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("Existing executable validator Python required")
    profile, source_root = installed_profile(
        actor_image=actor_image,
        verifier_image=verifier_image,
        runtime_image=runtime_image,
    )
    workspace.mkdir(mode=0o700)
    for name in (
        "input_views",
        "skills",
        "private_manifests",
        "private",
        "private/references",
        "private/authority",
        "runtime",
    ):
        (workspace / name).mkdir(mode=0o700)
    value = {
        "roots": {
            "assets": str(assets),
            "input_views": str(workspace / "input_views"),
            "skills": str(workspace / "skills"),
            "private_manifests": str(workspace / "private_manifests"),
        },
        "limits": {
            "maximum_spend_usd": 0,
            "maximum_condition_spend_usd": 0,
            "maximum_matrix_runs": 125,
        },
        "providers": {},
        "runtime": {
            "repository_root": str(workspace),
            "source_root": source_root,
            "validator_python": str(interpreter),
            "expected_hostname": socket.gethostname(),
            "allowed_reference_root": str(workspace / "private/references"),
            "authority_state_dir": str(workspace / "private/authority"),
            "shared_profile": profile,
            "execution_enabled": False,
        },
    }
    if providers is not None:
        value["providers"] = providers
    if providers is not None:
        value["limits"].update(
            maximum_spend_usd=maximum_spend_usd,
            maximum_condition_spend_usd=maximum_spend_usd,
            maximum_matrix_runs=1,
            inference_requests=4,
            max_input_tokens=262144,
            max_output_tokens=4096,
        )
    if native_output:
        from simready_bench.engine.native_output import DEFAULT_AUTHORING_TURNS

        value["limits"].update(
            inference_requests=DEFAULT_AUTHORING_TURNS, max_input_tokens=None, max_output_tokens=None
        )
    value["runtime"].update(execution_enabled=execution_enabled)
    if authority_state_dir is not None:
        value["runtime"]["authority_state_dir"] = str(authority_state_dir)
    from simready_bench.config import validate_policy

    validate_policy(value, base_dir=output.parent)
    if output.is_relative_to(workspace):
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def init_daytona_policy(
    *,
    output,
    workspace,
    assets,
    providers,
    maximum_spend_usd,
    conditions,
    authoring_turns=64,
):
    """Save a Daytona policy using existing provider grants and asset files."""
    import copy
    from decimal import Decimal, ROUND_DOWN
    import math
    import time
    from urllib.parse import urlsplit

    from simready_bench.engine.native_output import authoring_protocol, authoring_seconds

    authoring_protocol(authoring_turns)
    if type(conditions) is not int or not 1 <= conditions <= 10000:
        raise ValueError("Conditions must be an integer between 1 and 10000")
    if (
        type(maximum_spend_usd) not in (int, float)
        or not math.isfinite(maximum_spend_usd)
        or not 0 < maximum_spend_usd <= 1000000000
    ):
        raise ValueError("An explicit positive bounded USD allowance is required")
    condition_spend = (Decimal(str(maximum_spend_usd)) / conditions).quantize(
        Decimal("0.000001"), rounding=ROUND_DOWN
    )
    if condition_spend <= 0:
        raise ValueError("The USD allowance must cover at least one microdollar per condition")
    if not isinstance(providers, dict) or not providers:
        raise ValueError("An existing provider registry and project allowance are required")
    providers = copy.deepcopy(providers)
    for provider in providers.values():
        if not isinstance(provider, dict):
            raise ValueError("An existing provider registry and project allowance are required")
        grant = provider.get("project_budget")
        if not isinstance(grant, dict):
            raise ValueError("An existing approved provider allowance is required")
        endpoint = provider.get("base_url")
        if not isinstance(endpoint, str):
            raise ValueError("An explicit provider endpoint is required")
        if "model_spec" in provider and not isinstance(provider["model_spec"], dict):
            raise ValueError("An operator model specification must be an object")
        origin = urlsplit(endpoint)
        if (
            grant.get("project") != "simready-bench"
            or grant.get("approved") is not True
            or not isinstance(grant.get("authorization_id"), str)
            or not grant["authorization_id"].strip()
            or grant.get("pricing_status") != "known"
            or type(grant.get("expires_at")) not in (int, float)
            or not math.isfinite(grant["expires_at"])
            or grant["expires_at"] <= time.time()
            or type(grant.get("maximum_requests")) is not int
            or grant["maximum_requests"] < conditions * authoring_turns
            or type(grant.get("maximum_spend_usd")) not in (int, float)
            or not math.isfinite(grant["maximum_spend_usd"])
            or grant["maximum_spend_usd"] < maximum_spend_usd
            or not isinstance(grant.get("model"), str)
            or not grant["model"]
            or grant.get("upstream_origin") != origin.scheme + "://" + origin.netloc
            or endpoint.rstrip("/") != grant["upstream_origin"] + "/v1"
            or not isinstance(provider.get("api_key_env"), str)
            or provider.get("model_spec", {}).get("model", grant["model"]) != grant["model"]
        ):
            raise ValueError("The requested run must fit its unchanged approved provider allowance")

    output, workspace, assets = map(_literal, (output, workspace, assets))
    if output.exists() or output.is_symlink() or workspace.exists() or workspace.is_symlink():
        raise FileExistsError("Preserving existing output/workspace")
    if workspace.parent.resolve() != workspace.parent or not workspace.parent.is_dir():
        raise ValueError("Workspace parent must be an existing non-symlink directory")
    if assets.resolve() != assets or not assets.is_dir():
        raise ValueError("Existing non-symlink asset directory required")
    if workspace.is_relative_to(assets) or assets.is_relative_to(workspace):
        raise ValueError("Assets and operator workspace must be disjoint")
    names = ("skills", "input_views", "private_manifests", "authority")
    if output == workspace or output.is_relative_to(assets) or any(
        output.is_relative_to(workspace / name) for name in names
    ):
        raise ValueError("Operator policy must remain outside asset/public/authority roots")
    if not output.is_relative_to(workspace) and (
        output.parent.resolve() != output.parent or not output.parent.is_dir()
    ):
        raise ValueError("Output parent must be an existing non-symlink directory")

    workspace.mkdir(mode=0o700)
    for name in names:
        (workspace / name).mkdir(mode=0o700)
    value = {
        "roots": {
            "assets": str(assets),
            **{name: str(workspace / name) for name in names if name != "authority"},
        },
        "providers": providers,
        "limits": {
            "maximum_spend_usd": maximum_spend_usd,
            "maximum_condition_spend_usd": float(condition_spend),
            "maximum_matrix_runs": conditions,
            "inference_requests": authoring_turns,
            "agent_seconds": authoring_seconds(authoring_turns),
            "max_input_tokens": None,
            "max_output_tokens": None,
        },
        "runtime": {"authority_state_dir": str(workspace / "authority")},
    }
    from simready_bench.config import validate_policy

    validate_policy(value, base_dir=output.parent)
    if output.is_relative_to(workspace):
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value
