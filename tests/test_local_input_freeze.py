"""Generated reference views belong to the saved local run's immutable inputs."""

import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simready_bench import benchmark as workflow
from simready_bench import resume
from test_resume import portable


@pytest.mark.parametrize("tamper", [False, True])
def test_acquired_local_source_is_frozen_before_failure_and_saved_recovery(
    tmp_path, monkeypatch, tamper
):
    from simready_bench import sources

    template, _, _ = workflow.reference_manifest("NV25")
    template["assets"] = template["assets"][:1]
    template["source"]["packages"] = [template["assets"][0]["id"]]
    assert template["assets"][0].get("source_view")
    monkeypatch.setattr(workflow, "reference_manifest", lambda reference: (copy.deepcopy(template), tmp_path, False))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    output = tmp_path / "run"
    acquisitions = []

    def prepare_policy(**options):
        policy = {
            "roots": {name: str(output / name) for name in ("assets", "skills", "input_views", "private_manifests")},
            "providers": {"selected": options["provider"]},
            "limits": {"maximum_spend_usd": 10},
            "runtime": {},
        }
        (output / "operator.json").write_text(json.dumps(policy))
        return policy

    def acquire(source, *, base_dir, assets_root):
        acquisitions.append(source)
        asset = template["assets"][0]
        for name in {asset["usd"], asset["source_view"], *asset.get("dependencies", [])}:
            path = Path(assets_root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen source bytes")

    class VerifiedInputs(RuntimeError):
        pass

    publications = []

    def publish(*args, **kwargs):
        publications.append(args)
        if len(publications) == 2:
            raise VerifiedInputs("Saved recovery verified the acquired inputs")
        return None

    def execute(manifest, **options):
        assert len(acquisitions) == 1
        assert options["_inputs_prepared"] is True
        saved = resume._read(output / "run.json")
        assert saved["inputs"]["assets"]
        assert saved["inputs"] == {name: resume._tree(output / name) for name in saved["inputs"]}
        spec = resume._read(output / "run-spec.json")
        assert spec
        if tamper:
            (output / "assets" / manifest["assets"][0]["usd"]).write_bytes(b"changed source bytes")
        raise ModuleNotFoundError("Preparation dependency unavailable")

    monkeypatch.setattr(workflow, "prepare_policy", prepare_policy)
    monkeypatch.setattr(sources, "import_source", acquire)
    monkeypatch.setattr(workflow, "execute_local_reported", execute)
    monkeypatch.setattr(workflow, "publication_callback", publish)
    def recover(reference, **options):
        assert options.pop("_initial_failure") is True
        return resume._resume_once(reference, **options)

    monkeypatch.setattr(resume, "resume_benchmark", recover)
    expected = ValueError if tamper else VerifiedInputs
    message = "staged benchmark input files have changed" if tamper else "verified the acquired inputs"
    with pytest.raises(expected, match=message):
        workflow.benchmark("NV25", model="gpt-6.1-sol", deploy="local", output_dir=output,
                           repetitions=1, maximum_spend_usd=10, workers=24, max_retries=1)
    assert len(acquisitions) == 1


@pytest.mark.parametrize("prepared", [False, True])
def test_direct_local_execution_acquires_inputs_unless_prepared(tmp_path, monkeypatch, prepared):
    from simready_bench import config, local, sources

    events = []
    manifest = {"source": {"kind": "fixture"}}
    monkeypatch.setattr(sources, "import_source", lambda *args, **kwargs: events.append("source"))
    monkeypatch.setattr(workflow, "prepare_source_views", lambda *args: events.append("views"))
    monkeypatch.setattr(config, "validate_policy", lambda policy, **kwargs: policy)

    def resolve(*args, **kwargs):
        raise RuntimeError("Reached input resolution")

    monkeypatch.setattr(config, "resolve_experiment", resolve)
    with pytest.raises(RuntimeError, match="Reached input resolution"):
        workflow.execute_local(manifest, policy={"roots": {"assets": str(tmp_path)}},
                               output_dir=tmp_path, _inputs_prepared=prepared)
    assert events == ([] if prepared else ["source", "views"])


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
