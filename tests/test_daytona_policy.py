"""Create custom Daytona policies without a local evaluator installation."""

import builtins
import copy
import json
import os
from pathlib import Path
import re
import runpy
import stat
import subprocess
import sys
import time

import pytest

from simready_bench import policy
from simready_bench.config import validate_policy
from simready_bench.engine.native_output import authoring_seconds


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Daytona host uses Linux/WSL ownership")


def provider(*, requests=640, spend=100):
    return {
        "kind": "openai",
        "base_url": "https://models.example.org/v1",
        "api_key_env": "CUSTOM_MODEL_KEY",
        "model_spec": {
            "schema": "simready-operator-model-v1",
            "model": "custom-vision-model",
            "base_url": "https://models.example.org/v1",
            "api_style": "openai-completions",
            "context_window": 32768,
            "maximum_output_tokens": 8192,
            "pricing": {"input_per_million_usd": 1, "output_per_million_usd": 2},
            "pricing_source": "https://models.example.org/pricing",
            "checked": "2026-10-20",
        },
        "project_budget": {
            "project": "simready-bench",
            "authorization_id": "existing-custom-allowance",
            "approved": True,
            "upstream_origin": "https://models.example.org",
            "model": "custom-vision-model",
            "expires_at": time.time() + 86400,
            "maximum_requests": requests,
            "maximum_spend_usd": spend,
            "pricing_status": "known",
            "external_spend_cap_verified": False,
        },
    }


def arguments(tmp_path, **changes):
    assets = tmp_path / "references"
    assets.mkdir(exist_ok=True)
    (assets / "asset.usda").write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    return {
        "output": tmp_path / "operator.json",
        "workspace": tmp_path / "operator",
        "assets": assets,
        "providers": {"selected": provider()},
        "maximum_spend_usd": 100,
        "conditions": 10,
        **changes,
    }


def block_local_evaluator(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("nvvalidate", "pxr")) or name in (
            "simready_bench.local",
            "simready_bench.engine.environment",
            "simready_bench.engine.nvidia_validator",
        ):
            pytest.fail("Daytona policy imported a local evaluator: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(policy, "installed_profile", lambda **_: pytest.fail("Local profile requested"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("Host process requested"))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Host process requested"))


def test_cloud_policy_retains_assets_grant_and_native_64_limits(tmp_path, monkeypatch):
    options = arguments(tmp_path)
    original = copy.deepcopy(options["providers"])
    asset_bytes = (options["assets"] / "asset.usda").read_bytes()
    monkeypatch.setenv("CUSTOM_MODEL_KEY", "secret-value-never-saved")
    block_local_evaluator(monkeypatch)
    value = policy.init_daytona_policy(**options)
    assert json.loads(options["output"].read_text()) == value
    assert value["providers"] == original == options["providers"]
    assert value["roots"]["assets"] == str(options["assets"])
    assert (options["assets"] / "asset.usda").read_bytes() == asset_bytes
    assert value["limits"] == {
        "maximum_spend_usd": 100,
        "maximum_condition_spend_usd": 10,
        "maximum_matrix_runs": 10,
        "inference_requests": 64,
        "agent_seconds": authoring_seconds(64),
        "max_input_tokens": None,
        "max_output_tokens": None,
    }
    assert set(value["runtime"]) == {"authority_state_dir"}
    for name in ("skills", "input_views", "private_manifests", "authority"):
        directory = options["workspace"] / name
        assert directory.is_dir() and list(directory.iterdir()) == []
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(options["output"].stat().st_mode) == 0o600
    assert "secret-value-never-saved" not in options["output"].read_text()
    assert validate_policy(value, base_dir=tmp_path)["providers"] == original


@pytest.mark.parametrize("turns,conditions", [(1, 1), (16, 3), (128, 10000)])
def test_custom_turns_and_matrix_preserve_existing_grant(tmp_path, turns, conditions):
    options = arguments(tmp_path, authoring_turns=turns, conditions=conditions)
    options["providers"]["selected"] = provider(requests=turns * conditions)
    value = policy.init_daytona_policy(**options)
    assert value["limits"]["inference_requests"] == turns
    assert value["limits"]["agent_seconds"] == authoring_seconds(turns)
    assert value["limits"]["maximum_matrix_runs"] == conditions
    assert value["providers"] == options["providers"]
    from decimal import Decimal

    assert Decimal(str(value["limits"]["maximum_condition_spend_usd"])) * conditions <= 100


@pytest.mark.parametrize("field,value", [
    ("conditions", 0), ("conditions", 10001), ("conditions", True),
    ("authoring_turns", 0), ("authoring_turns", 129), ("authoring_turns", True),
    ("maximum_spend_usd", 0), ("maximum_spend_usd", -1),
    ("maximum_spend_usd", float("nan")), ("maximum_spend_usd", float("inf")),
    ("maximum_spend_usd", True),
])
def test_invalid_run_bounds_never_create_policy_or_workspace(tmp_path, field, value):
    options = arguments(tmp_path, **{field: value})
    with pytest.raises(ValueError):
        policy.init_daytona_policy(**options)
    assert not options["output"].exists() and not options["workspace"].exists()


@pytest.mark.parametrize("field,value", [
    ("project", "other-project"), ("approved", False), ("authorization_id", ""),
    ("expires_at", 0), ("maximum_requests", 639), ("maximum_spend_usd", 99),
    ("upstream_origin", "https://another.example.org"), ("model", "another-model"),
    ("pricing_status", "unknown"),
])
def test_existing_provider_allowance_cannot_be_expanded_or_rebound(tmp_path, field, value):
    options = arguments(tmp_path)
    options["providers"]["selected"]["project_budget"][field] = value
    with pytest.raises(ValueError, match="unchanged approved"):
        policy.init_daytona_policy(**options)
    assert not options["output"].exists() and not options["workspace"].exists()


@pytest.mark.parametrize("field,value", [("api_key_env", "secret-token-value"), ("api_key", "secret")])
def test_provider_credentials_must_be_environment_names(tmp_path, field, value):
    options = arguments(tmp_path)
    options["providers"]["selected"][field] = value
    with pytest.raises(ValueError):
        policy.init_daytona_policy(**options)
    assert not options["output"].exists()


def test_existing_output_and_workspace_are_preserved(tmp_path):
    options = arguments(tmp_path)
    options["output"].write_text("original policy")
    with pytest.raises(FileExistsError):
        policy.init_daytona_policy(**options)
    assert options["output"].read_text() == "original policy"
    options["output"].unlink()
    options["workspace"].mkdir()
    with pytest.raises(FileExistsError):
        policy.init_daytona_policy(**options)
    assert not options["output"].exists()


@pytest.mark.parametrize("target", ["assets", "authority", "skills", "input_views", "private_manifests"])
def test_policy_file_cannot_enter_exposed_or_authority_roots(tmp_path, target):
    options = arguments(tmp_path)
    options["output"] = (options["assets"] if target == "assets" else options["workspace"] / target) / "operator.json"
    with pytest.raises(ValueError, match="outside asset/public/authority"):
        policy.init_daytona_policy(**options)
    assert not options["workspace"].exists()


def test_documented_provider_script_uses_cloud_helper_without_local_imports(tmp_path, monkeypatch):
    options = arguments(tmp_path)
    spec = tmp_path / "model.json"
    spec.write_text(json.dumps(options["providers"]["selected"]["model_spec"]))
    document = (Path(__file__).resolve().parents[1] / "docs/configuration.md").read_text()
    source = next(block for block in re.findall(r"```python\n(.*?)```", document, re.S) if "init_daytona_policy" in block)
    script = tmp_path / "configure_provider.py"
    script.write_text(source)
    monkeypatch.setattr(sys, "argv", [str(script), "--model-spec", str(spec), "--api-key-env", "CUSTOM_MODEL_KEY",
        "--assets", str(options["assets"]), "--workspace", str(options["workspace"]), "--conditions", "3",
        "--authoring-turns", "16", "--max-spend-usd", "100", "--output", str(options["output"])])
    block_local_evaluator(monkeypatch)
    runpy.run_path(str(script), run_name="__main__")
    saved = json.loads(options["output"].read_text())
    assert saved["limits"]["inference_requests"] == 16
    assert saved["limits"]["maximum_matrix_runs"] == 3
    assert saved["providers"]["selected"]["project_budget"]["maximum_requests"] == 48
    assert saved["roots"]["assets"] == str(options["assets"])
