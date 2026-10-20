"""Run asset reconstruction experiments from a reference set or USD package."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from decimal import Decimal, ROUND_DOWN
from importlib.resources import files
from pathlib import Path
from uuid import uuid4

from simready_bench.config import SCHEMA_VERSION, _json_object
from simready_bench.engine.native_output import (
    DEFAULT_AUTHORING_TURNS,
    authoring_protocol,
    authoring_seconds,
)


def _absolute(path):
    """Normalise relative segments without following filesystem links."""
    return Path(os.path.abspath(path))


def read_json(path):
    path = _absolute(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("A regular JSON file smaller than 16 MiB is required")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_json_object)


def _identifier(value):
    value = re.sub(r"[^A-Za-z0-9_.-]", "-", value)[:63]
    if not value or not value[0].isalnum():
        value = "asset-" + value[:57]
    return value


def reference_manifest(reference):
    """Return the experiment, its source directory and package-input mode."""
    if str(reference) in ("NV25", "nvidia-25"):
        value = json.loads(
            files("simready_bench")
            .joinpath("resources/benchmarks/nvidia-25.json")
            .read_text()
        )
        return value, Path.cwd(), False
    path = _absolute(reference)
    if path.is_symlink() or not path.is_file():
        raise ValueError(
            "Choose the NV25 public reference set, a USD file or a JSON asset list"
        )
    if path.suffix.lower() in (".usd", ".usda", ".usdc"):
        return {"assets": [str(path)]}, path.parent, True
    value = read_json(path)
    if isinstance(value, list):
        return {"assets": value}, path.parent, True
    if not isinstance(value, dict) or not isinstance(value.get("assets"), list):
        raise ValueError(
            "An asset list or experiment object with an assets array is required"
        )
    if value.get("source"):
        from simready_bench.sources import validate_source

        value["source"] = validate_source(value["source"], base_dir=path.parent)
    return value, path.parent, value.get("schema_version") != SCHEMA_VERSION


def _copy_file(source, destination, *, maximum=1024**3):
    if (
        source.is_symlink()
        or source.resolve() != source
        or not source.is_file()
        or source.stat().st_nlink != 1
    ):
        raise ValueError("Asset and skill inputs must be regular files without links")
    if source.stat().st_size > maximum:
        raise ValueError("Input file exceeds the package size limit")
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        if (
            hashlib.sha256(destination.read_bytes()).digest()
            != hashlib.sha256(source.read_bytes()).digest()
        ):
            raise ValueError("Input package paths collide")
        return
    shutil.copyfile(source, destination)
    destination.chmod(0o600)


def _package_files(root, output):
    """Keep relative USD composition and texture paths within their package."""
    total = 0
    selected = []
    for path in sorted(root.rglob("*")):
        if path == output or path.is_relative_to(output):
            continue
        if path.is_symlink():
            raise ValueError("Asset packages must not contain symbolic links")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Asset packages must contain regular files")
        total += path.stat().st_size
        if len(selected) >= 10000 or total > 1024**3:
            raise ValueError("Asset package exceeds 10000 files or 1 GiB")
        selected.append(path)
    return selected


def stage_assets(
    manifest,
    *,
    base_dir,
    destination,
    output_dir,
    assets_root=None,
    package_inputs=False,
    input_views_root=None,
):
    """Copy selected reference dependencies and explicitly supplied observations."""
    if manifest.get("source"):
        return
    from simready_bench.asset_inputs import dependency_files, selected_file
    from simready_bench.config import _id

    base = _absolute(assets_root) if assets_root else _absolute(base_dir)
    entries = []
    for entry in manifest["assets"]:
        bare = isinstance(entry, str)
        asset = {"usd": entry} if bare else copy.deepcopy(entry)
        if not isinstance(asset, dict) or not isinstance(asset.get("usd"), str):
            raise ValueError("Each asset needs a USD path")
        source = _absolute(base / asset["usd"])
        if source.suffix.lower() not in (".usd", ".usda", ".usdc"):
            raise ValueError("Asset paths must end in .usd, .usda or .usdc")
        entries.append((asset, source, bare))
    explicit = [asset["id"] for asset, _, _ in entries if "id" in asset]
    for identifier in explicit:
        _id(identifier, "Asset id")
    if len(explicit) != len(set(explicit)):
        raise ValueError(
            "Asset identifiers must be unique. Give each manifest entry its own id"
        )
    inferred = Counter(
        _identifier(source.stem) for asset, source, _ in entries if "id" not in asset
    )
    choices = []
    for asset, source, _ in entries:
        if "id" in asset:
            choices.append((None, None))
            continue
        identifier = _identifier(source.stem)
        relative = (
            source.relative_to(base)
            if source.is_relative_to(base)
            else Path(asset["usd"])
        )
        if inferred[identifier] > 1 or identifier in explicit:
            identifier = _identifier(relative.with_suffix("").as_posix())
        choices.append((identifier, relative))
    collisions = Counter(
        identifier for identifier, _ in choices if identifier is not None
    )
    used = set(explicit)
    plans = []
    for (asset, source, bare), (inferred_id, relative_path) in zip(entries, choices):
        identifier = asset.get("id", inferred_id)
        if "id" not in asset and (collisions[identifier] > 1 or identifier in explicit):
            digest = hashlib.sha256(relative_path.as_posix().encode()).hexdigest()[:8]
            identifier = identifier[:54] + "-" + digest
        if identifier in used and "id" not in asset:
            raise ValueError(
                "Asset identifiers collide. Give each manifest entry an explicit id"
            )
        used.add(identifier)
        # Inferred inputs follow authored dependencies. Explicit objects retain
        # exactly their declared file list relative to the selected asset root.
        infer_package = bare or (
            package_inputs
            and "dependencies" not in asset
            and not asset.get("source_view")
            and not asset.get("input_views")
        )
        package = source.parent if infer_package and not assets_root else base
        source = selected_file(source, package)
        relative = source.relative_to(package)
        prefix = Path("packages") / identifier if infer_package else Path()
        if infer_package:
            paths = dependency_files(source, package)
        else:
            dependencies = asset.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str) for item in dependencies
            ):
                raise ValueError(
                    "Asset dependencies must be an explicit list of file paths"
                )
            paths = [source, *[_absolute(base / item) for item in dependencies]]
            if asset.get("source_view"):
                paths.append(_absolute(base / asset["source_view"]))
            paths = sorted({selected_file(path, package) for path in paths})
        if len(paths) > 10000 or sum(path.stat().st_size for path in paths) > 1024**3:
            raise ValueError("Asset selection exceeds 10000 files or 1 GiB")
        staged = set()
        for path in paths:
            if not path.is_relative_to(package):
                raise ValueError("Asset dependencies must remain inside their package")
            name = prefix / path.relative_to(package)
            staged.add(name.as_posix())
        name = (prefix / relative).as_posix()
        asset.update(id=identifier, usd=name, dependencies=sorted(staged - {name}))
        if asset.get("source_view"):
            asset["source_view"] = (prefix / asset["source_view"]).as_posix()
        views = []
        if asset.get("input_views"):
            if not isinstance(asset["input_views"], list):
                raise ValueError("Input views must be a list of file paths")
            view_root = _absolute(input_views_root) if input_views_root else base
            for view in asset["input_views"]:
                if not isinstance(view, str):
                    raise ValueError("Input views must be file paths")
                path = selected_file(view_root / view, view_root)
                target = _absolute(Path(output_dir) / "input_views" / view)
                if not target.is_relative_to(
                    _absolute(Path(output_dir) / "input_views")
                ):
                    raise ValueError(
                        "Input view paths must remain inside their observation directory"
                    )
                if path.stat().st_size > 10 * 1024**2:
                    raise ValueError("Input view exceeds the 10 MiB limit")
                views.append((path, target))
        plans.append((asset, package, prefix, paths, views))
    # Resolve every selection before writing, so a later invalid asset cannot
    # leave an earlier package containing unintended files.
    targets = {}
    for _, package, prefix, paths, views in plans:
        selected = [
            (path, Path(destination) / prefix / path.relative_to(package))
            for path in paths
        ] + views
        for path, target in selected:
            if target in targets and targets[target] != path:
                if (
                    hashlib.sha256(targets[target].read_bytes()).digest()
                    != hashlib.sha256(path.read_bytes()).digest()
                ):
                    raise ValueError(
                        "Selected asset package paths collide. Give each package its own id"
                    )
            if (
                target.is_symlink()
                or target.exists()
                and (
                    not target.is_file()
                    or hashlib.sha256(target.read_bytes()).digest()
                    != hashlib.sha256(path.read_bytes()).digest()
                )
            ):
                raise ValueError(
                    "Selected asset package paths collide with an existing output file"
                )
            targets[target] = path
    for target, path in targets.items():
        _copy_file(path, target)
    manifest["assets"] = [asset for asset, _, _, _, _ in plans]
    manifest.pop("benchmark_freeze", None)


def stage_skills(paths, destination):
    result = []
    for index, raw in enumerate(paths or []):
        source = _absolute(raw)
        if source.is_symlink() or source.resolve() != source or not source.exists():
            raise ValueError(
                "Skill paths must be existing files or directories without links"
            )
        name = str(index + 1) + "-" + _identifier(source.name)
        target = Path(destination) / name
        if source.is_dir():
            package = _package_files(source, target)
            for path in package:
                _copy_file(
                    path, target / path.relative_to(source), maximum=256 * 1024**2
                )
        else:
            _copy_file(source, target, maximum=256 * 1024**2)
        result.append(name)
    return (
        [{"id": "selected", "paths": result}]
        if result
        else [{"id": "none", "paths": []}]
    )


def select_model(model, *, provider, policy=None, metadata=None):
    from simready_bench.models import model_configuration

    selected_provider = None
    if policy:
        matches = [
            copy.deepcopy(p)
            for p in policy.get("providers", {}).values()
            if p.get("project_budget", p.get("local_compute", {})).get("model") == model
        ]
        if len(matches) != 1:
            raise ValueError(
                "The policy must contain one provider for the selected model"
            )
        selected_provider = matches[0]
        provider = selected_provider["kind"]
    provider = provider or ("huggingface" if ":" in model else "openai")
    spec = (selected_provider or {}).get("model_spec")
    if spec:
        from simready_bench.engine.custom_model import validate_model_spec

        spec = validate_model_spec(spec, model=model)
        selected = {
            "id": "selected",
            "provider_ref": "selected",
            "model_id": model,
            "adapter": spec["api_style"],
            "metadata": {
                "contextWindow": spec["context_window"],
                "maxTokens": spec["maximum_output_tokens"] or spec["context_window"],
                "input": ["text", "image"],
                "reasoning": False,
            },
        }
    else:
        selected = model_configuration(
            provider, model, read_json(metadata) if metadata else None
        )
    return selected, selected_provider, provider


def compose_experiment(
    manifest,
    *,
    model,
    harness="pi",
    skill_sets=None,
    repetitions=4,
    maximum_spend_usd=0,
    authoring_turns=DEFAULT_AUTHORING_TURNS,
):
    protocol = authoring_protocol(authoring_turns)
    if type(repetitions) is not int or not 1 <= repetitions <= 1000:
        raise ValueError("Repetitions must be between 1 and 1000")
    result = copy.deepcopy(manifest)
    if not isinstance(result.get("assets"), list) or not result["assets"]:
        raise ValueError("At least one reference asset is required")
    template = json.loads(
        files("simready_bench")
        .joinpath("resources/benchmarks/nvidia-25.json")
        .read_text()
    )
    result.setdefault("schema_version", SCHEMA_VERSION)
    result["authoring_protocol"] = protocol
    result["scoring"] = result.get("scoring", template["scoring"])
    result["defaults"] = {
        **copy.deepcopy(template["defaults"]),
        **result.get("defaults", {}),
    }
    result["defaults"]["budgets"] = {
        **copy.deepcopy(template["defaults"]["budgets"]),
        **result["defaults"].get("budgets", {}),
    }
    count = len(result["assets"]) * repetitions
    # Decimal division keeps the sum of condition partitions within the cap.
    per_condition = (Decimal(str(maximum_spend_usd)) / count).quantize(
        Decimal("0.000001"), rounding=ROUND_DOWN
    )
    result["defaults"]["budgets"].update(
        maximum_spend_usd=float(per_condition),
        inference_requests=authoring_turns,
        agent_seconds=authoring_seconds(authoring_turns),
        max_input_tokens=None,
        max_output_tokens=None,
    )
    result["defaults"]["generation"] = {
        "output_limit": "provider-native",
        "max_output_tokens": None,
    }
    result["models"] = [copy.deepcopy(model)]
    result["models"][0]["generation"] = {
        **model.get("generation", {}),
        "output_limit": "provider-native",
        "max_output_tokens": None,
    }
    if model.get("generation", {}).get("reasoning_effort"):
        result["defaults"]["generation"]["reasoning_effort"] = model["generation"][
            "reasoning_effort"
        ]
    result["harnesses"] = [
        {
            "id": harness,
            "adapter": "harbor",
            "agent": harness,
            **({"kwargs": {"version": "1.19.3"}} if harness == "opencode" else {}),
        }
    ]
    result["skill_sets"] = skill_sets or [{"id": "none", "paths": []}]
    result["repetitions"] = repetitions
    result.setdefault("seed", 0)
    # Overrides from a matrix must not select the replaced models or harnesses.
    result["overrides"] = []
    return result


def _new_provider(
    kind, model, *, api_key_env, maximum_spend_usd, conditions,
    authoring_turns=DEFAULT_AUTHORING_TURNS,
):
    authoring_protocol(authoring_turns)
    from simready_bench.engine.hf_pricing import HF_PRICING
    from simready_bench.engine.request_cost import PRICING_BY_MODEL

    if kind not in ("openai", "huggingface"):
        raise ValueError(
            "Select an OpenAI or Hugging Face provider, or supply an operator policy"
        )
    origin = (
        "https://router.huggingface.co"
        if kind == "huggingface"
        else "https://api.openai.com"
    )
    known = model in (HF_PRICING if kind == "huggingface" else PRICING_BY_MODEL)
    if not known:
        raise ValueError("Supply --policy with model pricing for this model")
    return {
        "kind": kind,
        "base_url": origin + "/v1",
        "api_key_env": api_key_env
        or ("HF_TOKEN" if kind == "huggingface" else "OPENAI_API_KEY"),
        "project_budget": {
            "project": "simready-bench",
            "authorization_id": "benchmark-" + uuid4().hex,
            "approved": True,
            "upstream_origin": origin,
            "model": model,
            "expires_at": time.time() + max(
                86400, conditions * (authoring_seconds(authoring_turns) + 2400)
            ),
            "maximum_requests": authoring_turns * conditions,
            "maximum_spend_usd": maximum_spend_usd,
            "pricing_status": "known" if known else "unknown",
            "external_spend_cap_verified": False,
        },
    }


def native_run_allowance(model, *, provider, conditions, authoring_turns=DEFAULT_AUTHORING_TURNS):
    """Reserve enough accounting capacity for every permitted native request."""
    from simready_bench.engine.request_cost import (
        CostDenied, native_output_facts, upper_cost_micro_usd,
    )

    authoring_protocol(authoring_turns)
    try:
        facts = native_output_facts(model)
    except CostDenied as error:
        raise ValueError("Supply --policy with capacity and pricing for this model") from error
    if facts["route_identity_status"] != "resolved":
        raise ValueError("Supply --policy with a verified declaration for this model route")
    if provider == "huggingface":
        from simready_bench.engine.hf_pricing import hf_upper_cost_micro_usd

        per_request = hf_upper_cost_micro_usd(
            model, facts["output_token_upper_bound"], native_output_limits=True
        )
    elif provider == "openai":
        per_request = upper_cost_micro_usd(
            facts["maximum_input_tokens"], facts["output_token_upper_bound"],
            model=model, maximum_input_tokens=facts["maximum_input_tokens"],
            native_output_limits=True,
        )
    else:
        raise ValueError("Supply an operator policy for this model provider")
    return float(Decimal(per_request * authoring_turns * conditions) / Decimal(10**6))


def validate_native_policy_limits(policy, *, authoring_turns):
    """Check explicit operator restrictions before preparing native execution."""
    if policy is None:
        return
    limits = policy.get("limits", {})
    for name, requested in (
        ("inference_requests", authoring_turns),
        ("agent_seconds", authoring_seconds(authoring_turns)),
    ):
        if name in limits and requested > limits[name]:
            raise ValueError("The requested authoring turns exceed the policy " + name + " limit")
    for name in ("max_input_tokens", "max_output_tokens"):
        if limits.get(name) is not None:
            raise ValueError("Native authoring requires policy " + name + " to be null")


def prepare_policy(
    *,
    output_dir,
    provider,
    maximum_spend_usd,
    conditions,
    existing=None,
    deploy="local",
    runtime_images=None,
    authoring_turns=DEFAULT_AUTHORING_TURNS,
):
    authoring_protocol(authoring_turns)
    validate_native_policy_limits(existing, authoring_turns=authoring_turns)
    from simready_bench.policy import init_policy

    output = Path(output_dir)
    from simready_bench.runtime_images import DEFAULT_IMAGES

    images = runtime_images or DEFAULT_IMAGES
    if existing:
        policy = copy.deepcopy(existing)
        policy["providers"] = {"selected": provider}
        policy["roots"].update(
            assets=str(output / "assets"),
            skills=str(output / "skills"),
            input_views=str(output / "input_views"),
            private_manifests=str(output / "private_manifests"),
        )
        current = policy.get("runtime", {}).get("shared_profile")
        if (
            deploy == "local"
            and current
            and any(
                current.get(key) != images[name]
                for key, name in (
                    ("actor_image", "actor"),
                    ("verifier_image", "verifier"),
                    ("runtime_image", "cpu"),
                )
            )
        ):
            from simready_bench.policy import installed_profile

            profile, source_root = installed_profile(
                actor_image=images["actor"],
                verifier_image=images["verifier"],
                runtime_image=images["cpu"],
            )
            policy["runtime"].update(shared_profile=profile, source_root=source_root)
    elif deploy == "local":
        policy = init_policy(
            output=output / "operator.json",
            workspace=output / "operator",
            assets=output / "assets",
            providers={"selected": provider},
            maximum_spend_usd=maximum_spend_usd,
            execution_enabled=True,
            native_output=True,
            actor_image=images["actor"],
            verifier_image=images["verifier"],
            runtime_image=images["cpu"],
        )
        policy["roots"].update(
            skills=str(output / "skills"),
            input_views=str(output / "input_views"),
            private_manifests=str(output / "private_manifests"),
        )
    else:
        # The cloud executor creates host-bound runtime directories and profile.
        authority = output.parent / ".simready-bench-authority"
        authority.mkdir(parents=True, exist_ok=True, mode=0o700)
        policy = {
            "roots": {
                name: str(output / name)
                for name in ("assets", "skills", "input_views", "private_manifests")
            },
            "providers": {"selected": provider},
            "limits": {},
            "runtime": {"authority_state_dir": str(authority)},
        }
    policy["limits"].update(
        maximum_spend_usd=maximum_spend_usd,
        maximum_condition_spend_usd=float(
            (Decimal(str(maximum_spend_usd)) / conditions).quantize(
                Decimal("0.000001"), rounding=ROUND_DOWN
            )
        ),
        maximum_matrix_runs=conditions,
        inference_requests=authoring_turns,
        agent_seconds=authoring_seconds(authoring_turns),
        max_input_tokens=None,
        max_output_tokens=None,
    )
    policy["runtime"]["illustration_image"] = images["cpu"]
    policy["runtime"]["fidelity_image"] = images["cpu"]
    (output / "operator.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    (output / "operator.json").chmod(0o600)
    return policy


def prepare_source_views(manifest, policy, output_dir):
    """Render a reference PNG when an input package has no supplied thumbnail."""
    root = Path(policy["roots"]["assets"])
    image = policy["runtime"].get("illustration_image")
    source_root = policy["runtime"].get("source_root")
    for asset in manifest["assets"]:
        if asset.get("source_view") or asset.get("input_views"):
            continue
        source = root / asset["usd"]
        view = source.parent / ".simready-bench-reference.png"
        render_dir = Path(output_dir) / "reference-views" / asset["id"]
        render_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        request = {
            "reference": "/reference/" + asset["usd"],
            "candidate": "/candidate/" + asset["usd"],
            "reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "settings": {
                "renderer": "storm-llvmpipe",
                "width": 512,
                "views": ["isometric"],
                "seconds": 180,
            },
        }
        (render_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
        argv = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network",
            "none",
            "--read-only",
            "--user",
            str(os.getuid()) + ":" + str(os.getgid()),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--shm-size",
            "256m",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=536870912,mode=1777",
        ]
        for host, target, readonly in (
            (root, "/reference", True),
            (root, "/candidate", True),
            (Path(source_root), "/product", True),
            (render_dir, "/render", False),
        ):
            argv += [
                "--mount",
                "type=bind,src="
                + str(host)
                + ",dst="
                + target
                + (",readonly" if readonly else ""),
            ]
        argv += [
            "--env",
            "PYTHONPATH=/product",
            "--env",
            "LIBGL_ALWAYS_SOFTWARE=true",
            "--env",
            "GALLIUM_DRIVER=llvmpipe",
            "--entrypoint",
            "/opt/storm/bin/python",
            image,
            "-B",
            "-m",
            "simready_bench.storm",
            "/render/request.json",
        ]
        rendered = subprocess.run(argv, capture_output=True, text=True, timeout=200)
        (render_dir / "renderer.log").write_text(
            rendered.stdout + rendered.stderr, encoding="utf-8"
        )
        if rendered.returncode:
            raise RuntimeError(
                "Reference thumbnail rendering failed for " + asset["id"]
            )
        receipt = read_json(render_dir / "receipt.json")
        if receipt.get("status") != "rendered":
            raise RuntimeError("Reference thumbnail renderer did not complete")
        png = render_dir / "reference-isometric.png"
        expected = next(
            row["sha256"] for row in receipt["images"] if row["file"] == png.name
        )
        if hashlib.sha256(png.read_bytes()).hexdigest() != expected:
            raise ValueError("Reference thumbnail bytes changed")
        _copy_file(png, view)
        asset["source_view"] = view.relative_to(root).as_posix()
        asset.setdefault("dependencies", []).append(asset["source_view"])


def publish_report(work_dir, output_dir, *, results):
    from simready_bench.reporting import export_report

    output = Path(output_dir)
    temporary = output / ".report-export"
    report = export_report(work_dir, temporary, results=results)
    for child in temporary.iterdir():
        target = output / child.name
        if target.exists():
            raise FileExistsError("Preserving existing report output: " + str(target))
        child.rename(target)
    temporary.rmdir()
    for key in ("html", "scores", "results"):
        report[key] = str(output / Path(report[key]).name)
    report["directory"] = str(output)
    return report


def execute_local(manifest, *, policy, output_dir, work_dir=None, on_progress=None):
    """Acquire references, run the Harbor matrix and export scores and captures."""
    from simready_bench.config import resolve_experiment, validate_policy
    from simready_bench.preparation import write_json
    from simready_bench.sources import import_source
    from simready_bench.run_config import run_details
    import simready_bench.local as runner

    output = _absolute(output_dir)
    work = (
        _absolute(work_dir)
        if work_dir
        else Path("/dev/shm") / ("srb-" + uuid4().hex[:10])
    )
    if manifest.get("source"):
        try:
            import_source(
                manifest["source"], base_dir=output, assets_root=policy["roots"]["assets"]
            )
        except Exception as error:
            error._simready_source_acquisition_failure = True
            raise
    prepare_source_views(manifest, policy, output)
    (output / "experiment.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    trusted = validate_policy(policy, base_dir=output)
    resolved = resolve_experiment(manifest, base_dir=output, policy=trusted)
    details = run_details(
        resolved,
        manifest_sha256=hashlib.sha256(
            (output / "experiment.json").read_bytes()
        ).hexdigest(),
        manifest=manifest,
    )
    _save_json(
        output / "admission-plan.json",
        {
            "schema": "simready-bench-admission-plan-v1",
            "configuration_sha256": resolved["configuration_sha256"],
            "work_dir": str(work),
            "conditions": [item["condition_id"] for item in resolved["conditions"]],
        },
    )
    try:
        progress_options = (
            {"on_progress": on_progress} if on_progress is not None else {}
        )
        result = runner.run(resolved, work_dir=work, policy=trusted, **progress_options)
    except Exception as error:
        result = getattr(error, "result", None)
        if result is None:
            raise
        result = {**result, "run_details": details}
        report = publish_report(work, output, results=result)
        error.result = {**result, "report": report}
        raise
    write_json(work / "run-details.json", details)
    result = {**result, "run_details": details}
    return {**result, "report": publish_report(work, output, results=result)}


def _save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _aggregate_reports(
    reports, *, output_dir, planned_conditions, condition_sources=None
):
    """Combine already exported measurements without re-reading RAM snapshots."""
    from simready_bench.reporting import (
        read_generated_report,
        render_html,
        write_scores,
    )

    output = Path(output_dir)
    evidence = output / "evidence"
    evidence.mkdir(mode=0o700, exist_ok=True)
    rows = []
    scoring_notice = (
        "SimReady Robotics-Prop compliance and direct USD reconstruction fidelity."
    )
    for attempt, report in reports:
        portable = read_generated_report(report["results"])
        scoring_notice = portable.get("scoring_notice", scoring_notice)
        prefix = attempt.relative_to(output).as_posix()
        evidence_prefix = hashlib.sha256(prefix.encode()).hexdigest()[:24]
        relative_prefix = "" if prefix == "." else prefix + "/"
        for source in portable["results"]:
            if condition_sources is not None and condition_sources.get(
                source.get("condition_id")
            ) != str(Path(report["results"]).absolute()):
                continue
            row = copy.deepcopy(source)
            for kind in ("images", "traces"):
                for item in row.get(kind, []):
                    original = attempt / "evidence" / item["path"]
                    name = item["path"]
                    if attempt != output:
                        name = evidence_prefix + "-" + Path(name).name
                        if len(name.encode("utf-8")) > 255:
                            suffix = Path(item["path"]).suffix
                            if len(suffix.encode("utf-8")) > 16:
                                suffix = ""
                            name = (
                                evidence_prefix
                                + "-"
                                + hashlib.sha256(item["path"].encode()).hexdigest()
                                + suffix
                            )
                    _copy_file(original, evidence / name, maximum=32 * 1024**2)
                    item["path"] = name
            candidate = row.get("candidate") or {}
            if candidate.get("status") == "available":
                for key in ("manifest", "directory", "entrypoint"):
                    candidate[key] = relative_prefix + candidate[key]
                for member in candidate.get("files", []):
                    member["path"] = relative_prefix + member["path"]
            nvidia = row.get("nvidia_validator_report_file")
            if nvidia:
                nvidia["path"] = relative_prefix + nvidia["path"]
            if attempt != output:
                row["report_path"] = relative_prefix + "report.html"
            else:
                row.setdefault("report_path", "report.html")
            rows.append(row)
    completed = sum(row.get("status") == "completed" for row in rows)
    summary = {
        "schema": "simready-bench-portable-report-v1",
        "planned_conditions": planned_conditions,
        "attempted_conditions": len(rows),
        "completed_conditions": completed,
        "partial": completed < planned_conditions,
        "scoring_notice": scoring_notice,
        "results": rows,
    }
    _save_json(output / "results.json", summary)
    temporary = output / "scores.csv.tmp"
    write_scores(temporary, rows, mode="w")
    temporary.chmod(0o600)
    temporary.replace(output / "scores.csv")
    temporary = output / "report.html.tmp"
    temporary.write_text(render_html(summary), encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(output / "report.html")
    return {
        "directory": str(output),
        "html": str(output / "report.html"),
        "results": str(output / "results.json"),
        "scores": str(output / "scores.csv"),
    }


def remove_owned_tree(path, *, parent):
    """Remove an owned child tree, including read-only actor client directories."""
    import stat

    path, parent = _absolute(path), _absolute(parent)
    if (
        parent.resolve() != parent
        or path == parent
        or path.resolve() != path
        or not path.is_relative_to(parent)
    ):
        raise ValueError("Cleanup requires an owned child of its declared parent")
    if not path.exists():
        return
    for directory, children, filenames in os.walk(
        path, topdown=True, followlinks=False
    ):
        current = Path(directory)
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("Cleanup requires operator-owned directories")
        current.chmod(stat.S_IMODE(info.st_mode) | 0o700, follow_symlinks=False)
        for name in filenames:
            child = current / name
            info = child.lstat()
            if (
                stat.S_ISREG(info.st_mode)
                and info.st_uid == os.getuid()
                and info.st_nlink == 1
            ):
                child.chmod(stat.S_IMODE(info.st_mode) | 0o600, follow_symlinks=False)
    shutil.rmtree(path)


def execute_local_sequential(
    manifest, *, policy, output_dir, work_dir=None, on_progress=None
):
    """Evaluate each reference, export its artefacts and release temporary data."""
    output = _absolute(output_dir)
    work = (
        _absolute(work_dir)
        if work_dir
        else Path("/dev/shm") / ("srb-" + uuid4().hex[:8])
    )
    if work.exists() or work.is_symlink() or work.resolve() != work:
        raise FileExistsError("Choose a new sequential benchmark work directory")
    work.mkdir(parents=True, mode=0o700)
    attempts = output / "attempts"
    attempts.mkdir(mode=0o700)
    planned = (
        sum(
            len(
                manifest.get("reconstruction_selection", {}).get(
                    asset["id"], range(manifest["repetitions"])
                )
            )
            for asset in manifest["assets"]
        )
        * len(manifest["harnesses"])
        * len(manifest["skill_sets"])
    )
    references = Path(policy["runtime"]["allowed_reference_root"])
    results = []
    reports = []
    report = None
    for index, asset in enumerate(manifest["assets"]):
        selection = manifest.get("reconstruction_selection")
        if selection is not None and not selection[asset["id"]]:
            continue
        experiment = copy.deepcopy(manifest)
        experiment["assets"] = [copy.deepcopy(asset)]
        if selection is not None:
            experiment["reconstruction_selection"] = {
                asset["id"]: selection[asset["id"]]
            }
        source = experiment.get("source")
        imported = None
        if source and source["kind"] == "nvidia-hf-packages":
            if asset["id"] not in source["packages"]:
                raise ValueError(
                    "The reference set does not contain this asset package"
                )
            source["packages"] = [asset["id"]]
            imported = (
                Path(policy["roots"]["assets"]) / source["destination"] / asset["id"]
            )
        existed = imported.exists() if imported else True
        before = set(references.iterdir())
        attempt = attempts / (str(index).zfill(3) + "-" + asset["id"])
        attempt.mkdir(mode=0o700)
        trial_work = work / ("a" + str(index))
        try:
            progress_options = (
                {"on_progress": on_progress} if on_progress is not None else {}
            )
            value = execute_local(
                experiment,
                policy=policy,
                output_dir=attempt,
                work_dir=trial_work,
                **progress_options,
            )
        except Exception as error:
            value = getattr(error, "result", None)
            if value is not None and value.get("report"):
                results.extend(value.get("results", []))
                reports.append((attempt, value["report"]))
                report = _aggregate_reports(
                    reports, output_dir=output, planned_conditions=planned
                )
                error.result = {
                    "schema": "simready-bench-matrix-result-v1",
                    "planned_conditions": planned,
                    "completed_conditions": sum(
                        row.get("status") == "completed" for row in results
                    ),
                    "attempted_conditions": len(results),
                    "results": results,
                    "report": report,
                }
            raise
        results.extend(value["results"])
        reports.append((attempt, value["report"]))
        report = _aggregate_reports(
            reports, output_dir=output, planned_conditions=planned
        )
        if any(row.get("status") != "completed" for row in value["results"]):
            error = RuntimeError(
                "Reconstruction stopped because its provider or evaluator did not complete"
            )
            error.result = {
                "schema": "simready-bench-matrix-result-v1",
                "planned_conditions": planned,
                "completed_conditions": sum(
                    row.get("status") == "completed" for row in results
                ),
                "attempted_conditions": len(results),
                "results": results,
                "report": report,
            }
            raise error
        # Reports and frozen candidate copies now live outside the RAM tree.
        if trial_work.resolve() != trial_work or not trial_work.is_relative_to(work):
            raise ValueError(
                "Sequential cleanup must remain within its owned work directory"
            )
        remove_owned_tree(trial_work, parent=work)
        for package in set(references.iterdir()) - before:
            if (
                package.parent != references
                or package.resolve() != package
                or not package.name.startswith(("bv_", "bp_"))
            ):
                raise ValueError(
                    "Sequential reference cleanup requires an owned reference package"
                )
            remove_owned_tree(package, parent=references)
        if imported is not None and not existed:
            assets = Path(policy["roots"]["assets"])
            if imported.resolve() != imported or not imported.is_relative_to(assets):
                raise ValueError(
                    "Sequential input cleanup must remain in the asset root"
                )
            remove_owned_tree(imported, parent=assets)
        _save_json(
            work / "progress.json",
            {
                "schema": "simready-bench-matrix-result-v1",
                "planned_conditions": planned,
                "completed_conditions": sum(
                    row.get("status") == "completed" for row in results
                ),
                "attempted_conditions": len(results),
                "results": results,
                "partial": len(results) < planned,
            },
        )
    return {
        "schema": "simready-bench-matrix-result-v1",
        "planned_conditions": planned,
        "completed_conditions": sum(
            row.get("status") == "completed" for row in results
        ),
        "attempted_conditions": len(results),
        "results": results,
        "report": report,
    }


def publication_callback(output_dir, manifest, *, hf_repo=None, hf_run=None):
    """Open the run's saved Hugging Face destination before starting compute."""
    output = Path(output_dir)
    settings_path = output / "publication.json"
    saved = read_json(settings_path) if settings_path.exists() else {}
    if saved:
        if (
            hf_repo is not None
            and hf_repo != saved["repo"]
            or hf_run is not None
            and hf_run != saved["run"]
        ):
            raise ValueError(
                "Resume uses the saved Hugging Face repository and run name"
            )
        hf_repo, hf_run = saved["repo"], saved["run"]
    if hf_repo is None:
        if hf_run is not None:
            raise ValueError("--hf-run requires --hf-repo")
        return None
    from simready_bench.publishing import HuggingFacePublisher

    callback = HuggingFacePublisher(hf_repo, hf_run or output.name, manifest)
    callback.check_access()
    _save_json(settings_path, {"repo": hf_repo, "run": hf_run or output.name})
    return callback


def execute_local_reported(
    manifest, *, policy, output_dir, work_dir=None, on_report=None
):
    """Export and announce each reconstruction while the local run continues."""
    from simready_bench.controller_runner import ControllerReporter

    output = Path(output_dir)
    reporter = ControllerReporter(output, manifest)

    def progress(root, value):
        reporter(root, value)
        if on_report is not None:
            report = {
                "directory": str(reporter.output),
                "results": str(reporter.output / "results.json"),
                "html": str(reporter.output / "report.html"),
                "scores": str(reporter.output / "scores.csv"),
            }
            try:
                on_report(report)
            except Exception as error:
                _save_json(
                    output / "publication-status.json",
                    {"status": "pending", "error_type": type(error).__name__},
                )
                print(
                    "Hugging Face upload pending. Retrying after the next reconstruction.",
                    flush=True,
                )

    execute = execute_local_sequential if len(manifest["assets"]) > 1 else execute_local
    try:
        result = execute(
            manifest,
            policy=policy,
            output_dir=output,
            work_dir=work_dir,
            on_progress=progress,
        )
    except Exception as error:
        report = (getattr(error, "result", None) or {}).get("report")
        if report and on_report is not None:
            try:
                on_report(report, final=True)
            except Exception:
                pass
        raise
    if on_report is not None:
        try:
            on_report(result["report"], final=True)
        except Exception as error:
            error.result = result
            raise
    return result


def benchmark(
    reference="NV25",
    *,
    model,
    harness="pi",
    deploy="local",
    output_dir,
    skills=None,
    policy=None,
    maximum_spend_usd=None,
    repetitions=4,
    authoring_turns=None,
    provider=None,
    api_key_env=None,
    model_metadata=None,
    assets_root=None,
    work_dir=None,
    wheel=None,
    cpu_snapshot=None,
    reasoning=None,
    runtime_images=None,
    resume=False,
    hf_repo=None,
    hf_run=None,
    workers=1,
    max_retries=2,
    retry_delay_seconds=5,
    maximum_retry_delay_seconds=60,
    dispatch_pending=True,
):
    from simready_bench.recovery import validate_recovery

    validate_recovery(max_retries, retry_delay_seconds, maximum_retry_delay_seconds)
    if type(dispatch_pending) is not bool:
        raise ValueError("Dispatch pending must be a boolean")
    if not dispatch_pending and not resume:
        raise ValueError("Collecting existing executors requires resume")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("Workers must be an integer between 1 and 16")
    if deploy == "local" and workers > 1:
        raise ValueError("Parallel workers require Daytona deployment")
    if resume:
        from simready_bench.resume import resume_benchmark

        return resume_benchmark(
            reference,
            model=model,
            harness=harness,
            deploy=deploy,
            output_dir=output_dir,
            skills=skills,
            policy=policy,
            maximum_spend_usd=maximum_spend_usd,
            repetitions=repetitions,
            authoring_turns=authoring_turns,
            provider=provider,
            api_key_env=api_key_env,
            model_metadata=model_metadata,
            assets_root=assets_root,
            work_dir=work_dir,
            wheel=wheel,
            cpu_snapshot=cpu_snapshot,
            reasoning=reasoning,
            runtime_images=runtime_images,
            hf_repo=hf_repo,
            hf_run=hf_run,
            workers=workers,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            maximum_retry_delay_seconds=maximum_retry_delay_seconds,
            dispatch_pending=dispatch_pending,
        )
    authoring_turns = DEFAULT_AUTHORING_TURNS if authoring_turns is None else authoring_turns
    authoring_protocol(authoring_turns)
    output = _absolute(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new output directory")
    if deploy not in ("local", "daytona"):
        raise ValueError("Choose local or daytona deployment")
    original = read_json(policy) if policy else None
    validate_native_policy_limits(original, authoring_turns=authoring_turns)
    source, base, package_inputs = reference_manifest(reference)
    selected, connection, provider_kind = select_model(
        model, provider=provider, policy=original, metadata=model_metadata
    )
    from simready_bench.models import configure_reasoning

    configure_reasoning(selected, reasoning)
    from simready_bench.runtime_images import load_images

    images = load_images(runtime_images)
    if (
        runtime_images is None
        and original
        and original.get("runtime", {}).get("shared_profile")
    ):
        configured = original["runtime"]["shared_profile"]
        images = {
            "actor": configured["actor_image"],
            "verifier": configured["verifier_image"],
            "cpu": configured["runtime_image"],
        }
    if type(repetitions) is not int or not 1 <= repetitions <= 1000:
        raise ValueError("Repetitions must be between 1 and 1000")
    conditions = len(source["assets"]) * repetitions
    allowance_source = "explicit"
    if maximum_spend_usd is None:
        if original is None:
            maximum_spend_usd = native_run_allowance(
                model, provider=provider_kind, conditions=conditions,
                authoring_turns=authoring_turns,
            )
            allowance_source = "native-request-capacity"
        else:
            maximum_spend_usd = min(
                original["limits"]["maximum_spend_usd"],
                connection.get("project_budget", {}).get(
                    "maximum_spend_usd", original["limits"]["maximum_spend_usd"]
                ),
            )
            allowance_source = "operator-policy"
    if (
        type(maximum_spend_usd) not in (int, float)
        or not math.isfinite(maximum_spend_usd)
        or maximum_spend_usd <= 0
    ):
        raise ValueError("The run spend limit must be a positive finite amount")
    if (
        not conditions
        or conditions > 10000
        or type(repetitions) is not int
        or repetitions < 1
    ):
        raise ValueError("Choose between 1 and 10000 reconstruction attempts")
    if original:
        if maximum_spend_usd > original["limits"][
            "maximum_spend_usd"
        ] or (Decimal(str(maximum_spend_usd)) / conditions).quantize(
            Decimal("0.000001"), rounding=ROUND_DOWN
        ) > Decimal(str(original["limits"].get(
            "maximum_condition_spend_usd", original["limits"]["maximum_spend_usd"]
        ))):
            raise ValueError("The requested run exceeds the policy spend limits")
        if conditions > original["limits"].get("maximum_matrix_runs", 125):
            raise ValueError("The requested run exceeds the policy attempt limit")
    if connection is None:
        connection = _new_provider(
            provider_kind,
            model,
            api_key_env=api_key_env,
            maximum_spend_usd=maximum_spend_usd,
            conditions=conditions,
            authoring_turns=authoring_turns,
        )
    elif api_key_env:
        connection["api_key_env"] = api_key_env
    credential = connection.get("api_key_env")
    if credential and not re.fullmatch(r"[A-Z_][A-Z0-9_]{0,127}", credential):
        raise ValueError("--api-key-env must name an environment variable")
    if credential and not os.environ.get(credential):
        raise ValueError("Set " + credential + " before running the benchmark")
    output.mkdir(parents=True, mode=0o700)
    _save_json(
        output / "allowance.json",
        {
            "schema": "simready-bench-run-allowance-v1",
            "source": allowance_source,
            "authoring_turns": authoring_turns,
            "conditions": conditions,
            "maximum_requests": authoring_turns * conditions,
            "maximum_spend_usd": maximum_spend_usd,
        },
    )
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir(mode=0o700)
    if not assets_root and original:
        assets_root = original["roots"]["assets"]
    stage_assets(
        source,
        base_dir=base,
        destination=output / "assets",
        output_dir=output,
        assets_root=assets_root,
        package_inputs=package_inputs,
        input_views_root=original["roots"].get("input_views") if original else None,
    )
    selected_skills = stage_skills(skills, output / "skills")
    experiment = compose_experiment(
        source,
        model=selected,
        harness=harness,
        skill_sets=selected_skills,
        repetitions=repetitions,
        maximum_spend_usd=maximum_spend_usd,
        authoring_turns=authoring_turns,
    )
    effective_policy = prepare_policy(
        output_dir=output,
        provider=connection,
        maximum_spend_usd=maximum_spend_usd,
        conditions=conditions,
        existing=original,
        deploy=deploy,
        runtime_images=images,
        authoring_turns=authoring_turns,
    )
    if deploy == "local" and any(
        not asset.get("source_view") and not asset.get("input_views")
        for asset in experiment["assets"]
    ):
        if experiment.get("source"):
            from simready_bench.sources import import_source

            import_source(
                experiment["source"],
                base_dir=output,
                assets_root=effective_policy["roots"]["assets"],
            )
        prepare_source_views(experiment, effective_policy, output)
    (output / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    (output / "runtime-images.json").write_text(
        json.dumps(images, indent=2) + "\n", encoding="utf-8"
    )
    from simready_bench.resume import save_run_configuration

    save_run_configuration(
        output,
        reference=reference,
        deploy=deploy,
        manifest=experiment,
        policy=effective_policy,
        runtime_images=images,
        cpu_snapshot=cpu_snapshot,
        skills=skills,
    )
    on_report = publication_callback(output, experiment, hf_repo=hf_repo, hf_run=hf_run)
    try:
        if deploy == "daytona":
            if workers > 1:
                from simready_bench.parallel import run_parallel_daytona as execute

                worker_options = {
                    "workers": workers,
                    "max_retries": max_retries,
                    "retry_delay_seconds": retry_delay_seconds,
                    "maximum_retry_delay_seconds": maximum_retry_delay_seconds,
                }
            else:
                from simready_bench.daytona import run_daytona as execute

                worker_options = {}

            result = execute(
                experiment,
                policy=effective_policy,
                output_dir=output,
                assets_root=output / "assets",
                skills_root=output / "skills",
                work_dir=work_dir,
                wheel=wheel,
                cpu_snapshot=cpu_snapshot,
                runtime_images=images,
                **worker_options,
                **({"on_report": on_report} if on_report is not None else {}),
            )
        else:
            result = execute_local_reported(
                experiment,
                policy=effective_policy,
                output_dir=output,
                work_dir=work_dir,
                on_report=on_report,
            )
    except Exception:
        if not max_retries:
            raise
    else:
        if not max_retries or result.get("completed_conditions") == conditions:
            return result
    from simready_bench.resume import resume_benchmark

    return resume_benchmark(
        reference,
        model=model,
        harness=harness,
        deploy=deploy,
        output_dir=output,
        skills=skills,
        maximum_spend_usd=maximum_spend_usd,
        repetitions=repetitions,
        authoring_turns=authoring_turns,
        provider=provider,
        api_key_env=api_key_env,
        work_dir=work_dir,
        wheel=wheel,
        cpu_snapshot=cpu_snapshot,
        reasoning=reasoning,
        hf_repo=hf_repo,
        hf_run=hf_run,
        workers=workers,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        maximum_retry_delay_seconds=maximum_retry_delay_seconds,
        _initial_failure=True,
    )
