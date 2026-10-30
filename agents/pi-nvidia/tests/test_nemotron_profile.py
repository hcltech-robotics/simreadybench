import json
from pathlib import Path

import pytest

from pi_nvidia.accounting import snapshot
from pi_nvidia.hosted import configure
from pi_nvidia.profile import ModelProfile


ROOT = Path(__file__).parents[1]


def test_selected_profile_is_applied_per_entry_and_matches_native_low_receipt(tmp_path):
    from harbor.hosted.config import HostedJobConfig
    config = json.loads((ROOT / "examples/nemotron35vl-low-one-trial.json").read_text())
    HostedJobConfig.model_validate(config)
    entry = config["agents"][0]
    env, p, effort, sessions = configure(tmp_path / "runtime", tmp_path / "logs",
        {**entry["env"], "HARBOR_ACP_REQUESTED_MODEL": entry["model_name"]})
    assert effort == "low" and p.supports("low") and not p.supports("medium")
    models = json.loads((Path(env["PI_CODING_AGENT_DIR"]) / "models.json").read_text())
    provider = models["providers"]["harbor-endpoint"]
    assert provider["baseUrl"] == "https://integrate.api.nvidia.com/v1"
    assert provider["apiKey"] == "$NGC_API_KEY"
    assert provider["models"][0]["maxTokens"] == 8192
    assert provider["models"][0]["thinkingLevelMap"]["low"] == "low"
    rows = [{"type": "session", "id": "s"},
            {"type": "model_change", "provider": "harbor-endpoint", "modelId": p.id},
            {"type": "thinking_level_change", "thinkingLevel": "low"},
            {"type": "message", "id": "m", "message": {"role": "assistant",
             "usage": {"input": 200, "output": 100, "cacheRead": 800, "cacheWrite": 0, "reasoning": 60}}}]
    (sessions / "s.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    receipt = snapshot(sessions, "s", p, effort)
    assert receipt["status"] == "verified"
    assert receipt["usage"]["inputTokens"] == 1000
    assert receipt["reference_inference_estimate"]["usd"] == "0.000122"
    assert receipt["pricing"] == "unknown" and not receipt["billed"]
    assert snapshot(sessions, "s", p, "high")["status"] == "unavailable"
    assert "reference_inference_estimate" not in snapshot(sessions, "s", p, "high")
    with pytest.raises(ValueError, match="no clamping"):
        configure(tmp_path / "runtime", tmp_path / "bad",
            {**env, "PI_NVIDIA_THINKING": "medium"})
