"""The pinned manager cannot unset an existing task timeout via agent config."""
from types import SimpleNamespace
from types import MethodType

from harbor.models.trial.config import AgentConfig
from harbor.trial.trial import Trial
import pytest


@pytest.mark.parametrize("override", [None, 0])
def test_null_or_zero_override_does_not_disable_benchmark_task_timeout(override):
    trial = SimpleNamespace(
        config=SimpleNamespace(agent=AgentConfig(override_timeout_sec=override),
            agent_timeout_multiplier=None, timeout_multiplier=1),
        task=SimpleNamespace(config=SimpleNamespace(agent=SimpleNamespace(timeout_sec=231000))))
    trial._resolve_timeout_sec = MethodType(Trial._resolve_timeout_sec, trial)
    assert Trial._compute_agent_timeout_sec(trial) == 231000
    trial.task.config.agent.timeout_sec = None
    assert Trial._compute_agent_timeout_sec(trial) is None


def test_hosted_examples_remove_600_second_cap_without_claiming_to_unset_task_timeout():
    import json
    from pathlib import Path
    from harbor.hosted.config import HostedJobConfig
    from harbor.hosted.submit import dump_hosted_config
    for path in (Path(__file__).parents[1] / "examples").glob('*one-trial*.json'):
        config = json.loads(path.read_text())
        actual = dump_hosted_config(HostedJobConfig.model_validate(config))
        agent = actual["agents"][0]
        assert agent.get("max_timeout_sec") is None
        assert agent.get("override_timeout_sec") is None
        assert "PI_NVIDIA_TIME_LIMIT_SEC" not in agent["env"]
        assert "PI_NVIDIA_FINAL_RESERVE_SEC" not in agent["env"]
        assert actual["retry"]["max_retries"] == 0
