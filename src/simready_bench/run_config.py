"""Structured, credential-free execution configuration for result interpretation."""

from copy import deepcopy
from importlib.metadata import version, PackageNotFoundError
from simready_bench.config import _digest


def run_details(resolved, *, manifest_sha256, manifest):
    providers = resolved["operator_policy"]["providers"]
    conditions = []
    for condition in resolved["conditions"]:
        model = condition["model"]
        provider = providers.get(model.get("provider_ref"), {})
        conditions.append(
            {
                k: deepcopy(condition[k])
                for k in (
                    "condition_id",
                    "asset_id",
                    "harness_id",
                    "model_id",
                    "skill_set_id",
                    "repetition",
                )
                if k in condition
            }
        )
        row = conditions[-1]
        row.update(
            model=deepcopy(model),
            harness=deepcopy(condition["harness"]),
            skills=deepcopy(condition["skill_set"]),
            settings=deepcopy(condition["settings"]),
            reference_files=[
                {k: v for k, v in entry.items() if k != "absolute_path"}
                for entry in condition["asset"]["closure"]
            ],
            input_views=[
                {k: v for k, v in entry.items() if k != "absolute_path"}
                for entry in condition["asset"]["input_view_closure"]
            ],
            endpoint=provider.get("base_url"),
            credential_environment=provider.get("api_key_env"),
            operator_model_declaration=deepcopy(provider.get("model_spec")),
        )
    versions = {}
    for package in ("simready-bench", "harbor", "simready-validate", "usd-core"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "schema": "simready-bench-run-details-v1",
        "original_manifest_sha256": manifest_sha256,
        "original_manifest": deepcopy(manifest),
        "effective_configuration_sha256": resolved["configuration_sha256"],
        **(
            {"scoring_policy": deepcopy(resolved["normalized_manifest"]["scoring"])}
            if "scoring" in resolved["normalized_manifest"]
            else {}
        ),
        "effective_conditions_sha256": _digest(conditions),
        "conditions": conditions,
        "seed": resolved["normalized_manifest"]["seed"],
        "versions": versions,
        "profile_sha256": resolved["operator_policy"]
        .get("runtime", {})
        .get("shared_profile", {})
        .get("profile_sha256"),
    }
