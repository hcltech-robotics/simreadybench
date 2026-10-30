import json

import pytest

from harbor.hosted.config import HostedJobConfig
from harbor.hosted.submit import dump_hosted_config
from pi_nvidia.hosted import configure
from test_agent import profile


def source(**changes):
    return {"PI_NVIDIA_PROFILE": json.dumps(profile()), "PI_NVIDIA_THINKING": "off",
            "HARBOR_ACP_REQUESTED_MODEL": "nvidia/fixture/vision", **changes}


def test_nonsecret_selector_survives_actual_hosted_serialization_and_controls_reference(tmp_path):
    # Upstream surprise: the old name contains KEY and is treated as sensitive.
    config = HostedJobConfig.model_validate({"agents": [{"name": "acp", "env": {
        "PI_NVIDIA_KEY_ENV": "NGC_API_KEY", "PI_NVIDIA_ENV_NAME": "NGC_API_KEY"}}]})
    env = dump_hosted_config(config)["agents"][0]["env"]
    assert env["PI_NVIDIA_KEY_ENV"] != "NGC_API_KEY"
    assert env["PI_NVIDIA_ENV_NAME"] == "NGC_API_KEY"
    # Observed Hub effective env omits the legacy selector entirely.
    del env["PI_NVIDIA_KEY_ENV"]
    child, *_ = configure(tmp_path / "runtime", tmp_path / "logs",
        {**source(), **env, "NGC_API_KEY": "dummy-key-a"})
    models = json.loads((tmp_path / "logs/pi/config/models.json").read_text())
    assert models["providers"]["harbor-endpoint"]["apiKey"] == "$NGC_API_KEY"
    assert child["NGC_API_KEY"] == "dummy-key-a"


@pytest.mark.parametrize("selector", [None, "", "NGC_API_KEY", "NVIDIA_API_KEY"])
def test_conflicting_credentials_are_rejected_without_values_or_files(tmp_path, selector):
    env = source(NGC_API_KEY="dummy-key-a", NVIDIA_API_KEY="dummy-key-b")
    if selector is not None:
        env["PI_NVIDIA_ENV_NAME"] = selector
    with pytest.raises(ValueError, match="Conflicting injected NVIDIA credentials") as error:
        configure(tmp_path / "runtime", tmp_path / "logs", env)
    assert "dummy-key-a" not in str(error.value) and "dummy-key-b" not in str(error.value)
    assert not (tmp_path / "logs").exists()


def test_explicit_selector_precedes_equal_hosted_token_and_aliases(tmp_path):
    child, *_ = configure(tmp_path / "runtime", tmp_path / "logs", source(
        PI_NVIDIA_ENV_NAME="NGC_API_KEY", NGC_API_KEY="dummy-key-a",
        NVIDIA_API_KEY="dummy-key-a", HOSTED_INFERENCE_TOKEN="dummy-key-a"))
    auth = json.loads((tmp_path / "logs/pi-auth-status.json").read_text())
    assert auth["selected_key_env"] == "NGC_API_KEY" and auth["credential_source"] == "NGC_API_KEY"
    assert child["NGC_API_KEY"] == "dummy-key-a"


def test_conflicting_hosted_token_is_not_allowed_to_replace_explicit_selection(tmp_path):
    with pytest.raises(ValueError, match="Conflicting injected NVIDIA credentials"):
        configure(tmp_path / "runtime", tmp_path / "logs", source(
            PI_NVIDIA_ENV_NAME="NGC_API_KEY", NGC_API_KEY="dummy-key-a",
            HOSTED_INFERENCE_TOKEN="dummy-key-b"))
