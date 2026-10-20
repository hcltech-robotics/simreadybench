"""Generated reference views belong to the saved local run's immutable inputs."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simready_bench import benchmark as workflow
from simready_bench import resume
from test_resume import portable


@pytest.mark.parametrize("asset_count", [1, 2])
def test_local_reference_views_are_frozen_before_execution_and_resume(
    tmp_path, monkeypatch, asset_count
):
    sources = []
    for index in range(asset_count):
        package = tmp_path / ("source-" + str(index))
        package.mkdir()
        path = package / ("asset-" + str(index) + ".usda")
        path.write_text('#usda 1.0\ndef Cube "Asset" {}\n')
        sources.append(path)
    reference = sources[0]
    if asset_count > 1:
        reference = tmp_path / "references.json"
        reference.write_text(json.dumps([str(path) for path in sources]))
    source_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources]
    output = tmp_path / "run"
    rendered, executions = [], []
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    monkeypatch.setattr(workflow.os, "getuid", lambda: 1000, raising=False)
    monkeypatch.setattr(workflow.os, "getgid", lambda: 1000, raising=False)

    def prepare_policy(**options):
        policy = {
            "roots": {
                name: str(output / name)
                for name in ("assets", "skills", "input_views", "private_manifests")
            },
            "providers": {"selected": options["provider"]},
            "limits": {"maximum_spend_usd": asset_count * 10},
            "runtime": {
                "source_root": str(tmp_path / "product"),
                "illustration_image": "sha256:" + "a" * 64,
            },
        }
        (output / "operator.json").write_text(json.dumps(policy))
        return policy

    def render(argv, **options):
        assert argv[0:2] == ["docker", "run"]
        mount = next(item for item in argv if "dst=/render" in item)
        directory = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
        png = directory / "reference-isometric.png"
        png.write_bytes(b"reference thumbnail")
        (directory / "receipt.json").write_text(
            json.dumps(
                {
                    "status": "rendered",
                    "images": [
                        {
                            "file": png.name,
                            "sha256": hashlib.sha256(png.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
        )
        rendered.append(directory.name)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    saved_bytes = {}

    def execute(manifest, *, policy, output_dir, **options):
        executions.append(manifest)
        assert len(rendered) == asset_count
        saved = resume._read(output / "run.json")
        assert saved["experiment_sha256"] == resume._digest(manifest)
        assert saved["inputs"] == {
            name: resume._tree(output / name) for name in saved["inputs"]
        }
        for asset in manifest["assets"]:
            assert asset["source_view"] in asset["dependencies"]
            assert (output / "assets" / asset["source_view"]).is_file()
        saved_bytes.update(
            (name, (output / name).read_bytes())
            for name in ("run.json", "operator.json", "experiment.json")
        )
        workflow.prepare_source_views(manifest, policy, output)
        assert len(rendered) == asset_count
        assert saved["experiment_sha256"] == resume._digest(manifest)
        plan = resume.condition_plan(manifest)
        attempt = output / "attempts/000-reference"
        path = portable(
            attempt,
            [(identifier, condition, "completed") for identifier, condition in plan.items()],
        )
        report = workflow._aggregate_reports(
            [(attempt, {"results": str(path)})],
            output_dir=output,
            planned_conditions=len(plan),
        )
        return {
            "completed_conditions": len(plan),
            "results": resume._read(path)["results"],
            "report": report,
        }

    monkeypatch.setattr(workflow, "prepare_policy", prepare_policy)
    monkeypatch.setattr(workflow.subprocess, "run", render)
    monkeypatch.setattr(workflow, "execute_local_reported", execute)
    options = {
        "model": "gpt-6.1-sol",
        "harness": "pi",
        "deploy": "local",
        "output_dir": output,
        "maximum_spend_usd": asset_count * 10,
        "repetitions": 1,
    }
    workflow.benchmark(reference, **options)
    result = workflow.benchmark(reference, **options, resume=True)
    assert result["completed_conditions"] == asset_count
    assert len(executions) == 1
    assert len(rendered) == asset_count
    assert all((output / name).read_bytes() == raw for name, raw in saved_bytes.items())
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in sources] == source_hashes
