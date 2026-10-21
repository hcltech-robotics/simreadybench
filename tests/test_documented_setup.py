"""Check documented benchmark commands through the public CLI."""

from pathlib import Path
import importlib
import re
import shlex
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename,model,deployment",
    [
        ("README.md", "gpt-6.1-sol", "daytona"),
        ("docs/quickstart.md", "gpt-6.1-sol", "daytona"),
        ("docs/configuration.md", "google/gemma-4-31B-it:novita", "daytona"),
        ("docs/deployment.md", "gpt-6.1-sol", "daytona"),
    ],
)
def test_documented_reference_set_command_reaches_benchmark(
    filename, model, deployment, monkeypatch, capsys
):
    document = (ROOT / filename).read_text()
    block = next(
        block
        for block in re.findall(r"```(?:bash|sh)\n(.*?)```", document, re.S)
        if re.search(r"simready-bench benchmark (?:NV25|nvidia-25)\b", block)
        and (model in block or "MODEL_ID" in block)
    )
    command = shlex.split(block.replace("\\\n", "").replace("MODEL_ID", model))
    captured = {}

    def execute(reference, **options):
        captured.update(reference=reference, **options)
        return {"completed_conditions": 100, "planned_conditions": 100, "results": []}

    module = importlib.import_module("simready_bench.benchmark")
    monkeypatch.setattr(module, "benchmark", execute)
    from simready_bench.cli import main
    from simready_bench import supervision

    monkeypatch.setattr(
        supervision, "supervise",
        lambda argv, **_: main(supervision._child_arguments(argv)),
    )

    assert main(command[1:]) == 0
    assert captured["reference"] in ("NV25", "nvidia-25")
    assert captured["model"] == model and captured["harness"] == "pi"
    assert captured["deploy"] == deployment and captured["repetitions"] == 4
    assert captured["maximum_spend_usd"] is None
    assert captured["authoring_turns"] == 64
    assert captured["workers"] == (8 if filename == "docs/deployment.md" else 4)
    expected = (
        "../simreadybench-hf-results"
        if filename == "docs/configuration.md"
        else "../simreadybench-results"
    )
    assert captured["output_dir"] == Path(expected)
    assert capsys.readouterr().out == "100 of 100 reconstruction attempts completed.\n"


def test_documented_local_training_policy_and_episode_use_64_turns(
    tmp_path, monkeypatch
):
    import copy
    import json
    import sys

    from simready_bench import policy
    from simready_bench.engine.native_output import authoring_seconds

    configuration = (ROOT / "docs/configuration.md").read_text()
    provider_script = next(
        block for block in re.findall(r"```python\n(.*?)```", configuration, re.S)
        if "init_daytona_policy(" in block
    )
    training = (ROOT / "docs/reinforcement-learning.md").read_text()
    local_setup = next(
        block for block in re.findall(r"```python\n(.*?)```", training, re.S)
        if "init_policy(" in block
    )
    spec = next(
        json.loads(block) for block in re.findall(r"```json\n(.*?)```", training, re.S)
        if "simready-operator-model-v1" in block
    )
    model_path = tmp_path / "policy-model.json"
    model_path.write_text(json.dumps(spec))
    output = tmp_path / "training.json"
    calls = []

    def create_local(**settings):
        calls.append(copy.deepcopy(settings))
        return {"limits": {}, "providers": settings["providers"]}

    monkeypatch.setattr(policy, "init_policy", create_local)
    monkeypatch.setattr(
        policy, "init_daytona_policy",
        lambda **_: pytest.fail("Local training requested a cloud policy"),
    )
    monkeypatch.setattr(sys, "argv", [
        "configure_provider.py", "--model-spec", str(model_path),
        "--api-key-env", "POLICY_API_KEY", "--assets", str(tmp_path),
        "--workspace", str(tmp_path / "operator"), "--conditions", "100",
        "--max-spend-usd", "1", "--output", str(output),
    ])
    prefix = provider_script[:provider_script.index("init_daytona_policy(\n")]
    exec(compile(prefix + local_setup, "documented-local-policy", "exec"), {"__name__": "__main__"})
    saved = json.loads(output.read_text())
    assert len(calls) == 1 and calls[0]["native_output"] is True
    assert calls[0]["execution_enabled"] is True
    assert saved["limits"] == {
        "maximum_matrix_runs": 100,
        "maximum_condition_spend_usd": 0.01,
        "inference_requests": 64,
        "agent_seconds": authoring_seconds(64),
    }
    assert saved["providers"]["selected"]["project_budget"]["maximum_requests"] == 6400

    command = next(
        block.strip() for block in re.findall(r"```sh\n(.*?)```", training, re.S)
        if block.strip().startswith("simready-bench benchmark")
    )
    captured = {}

    def execute(reference, **options):
        captured.update(reference=reference, **options)
        return {"completed_conditions": 1, "planned_conditions": 1, "results": []}

    monkeypatch.setattr("simready_bench.benchmark.benchmark", execute)
    from simready_bench.cli import main

    assert main(shlex.split(command)[1:]) == 0
    assert captured["deploy"] == "local" and captured["authoring_turns"] == 64
    assert captured["repetitions"] == 1 and captured["maximum_spend_usd"] == 0.01
