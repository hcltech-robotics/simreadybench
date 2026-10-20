import math
import os
import pytest

from simready_bench.local import reconstruction_status
from simready_bench.benchmark import compose_experiment, select_model, remove_owned_tree
from simready_bench.harnesses import agent_config


def test_provider_rejection_has_no_model_quality_score_even_with_a_zero_candidate_score():
    score = {"simready": {"score_percent": 0}}
    failures = [{"status": "completed", "upstream_status": 400} for _ in range(4)]
    assert reconstruction_status(score, failures) == "infrastructure_error"
    success = {"status": "completed", "upstream_status": 200}
    assert reconstruction_status(score, [success]) == "completed"
    for value in (None, True, math.nan):
        assert (
            reconstruction_status({"simready": {"score_percent": value}}, [success])
            == "infrastructure_error"
        )


def test_sol_default_reasoning_reaches_the_actual_pi_configuration():
    from simready_bench.engine.native_output import AUTHORING_PROTOCOL

    model = select_model("gpt-6.1-sol", provider="openai")[0]
    experiment = compose_experiment(
        {"assets": [{"id": "cup", "usd": "cup.usda"}]}, model=model
    )
    condition = {
        "harness": {"adapter": "pi"},
        "model": model,
        "settings": experiment["defaults"],
        "authoring_protocol": AUTHORING_PROTOCOL,
    }
    config = agent_config(condition, "11111111-1111-4111-8111-111111111111", [])
    assert config["kwargs"]["thinking"] == "medium"
    assert config["kwargs"]["model_metadata"]["maxTokens"] == 128000
    assert config["kwargs"]["model_metadata"]["contextWindow"] == 1050000
    assert config["kwargs"]["max_turns"] == 64


@pytest.mark.skipif(os.name == "nt", reason="Linux actor directory permissions")
def test_owned_cleanup_handles_read_only_actor_clients_without_following_links(
    tmp_path,
):
    outside = tmp_path / "retained"
    outside.mkdir()
    (outside / "asset.usd").write_text("retain")
    work = tmp_path / "work"
    work.mkdir()
    trial = work / "trial"
    client = trial / "public-client/simready_bench/engine"
    client.mkdir(parents=True)
    (client / "actor_mcp.py").write_text("client")
    (client / "actor_mcp.py").chmod(0o400)
    (client / "outside").symlink_to(outside, target_is_directory=True)
    for directory in (client, client.parent, client.parent.parent, trial):
        directory.chmod(0o500)
    remove_owned_tree(trial, parent=work)
    assert not trial.exists()
    assert (outside / "asset.usd").read_text() == "retain"
    with pytest.raises(ValueError):
        remove_owned_tree(work, parent=work)
