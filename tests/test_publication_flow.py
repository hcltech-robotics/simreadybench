import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simready_bench import benchmark as workflow
from simready_bench.cli import main


def test_cli_passes_the_requested_dataset_and_run(tmp_path, monkeypatch):
    calls = []

    def execute(reference, **options):
        calls.append((reference, options))
        return {"completed_conditions": 0, "planned_conditions": 0, "results": []}

    monkeypatch.setattr(workflow, "benchmark", execute)
    assert (
        main(
            [
                "benchmark",
                "nvidia-25",
                "--model",
                "gpt-6.1-sol",
                "--output-dir",
                str(tmp_path),
                "--hf-repo",
                "owner/results",
                "--hf-run",
                "sol-medium",
            ]
        )
        == 0
    )
    assert calls[0][1]["hf_repo"] == "owner/results"
    assert calls[0][1]["hf_run"] == "sol-medium"


def test_resume_reuses_the_saved_publication_destination(tmp_path, monkeypatch):
    checked = []

    class Publisher:
        def __init__(self, repo, run, manifest):
            self.destination = (repo, run, manifest)

        def check_access(self):
            checked.append(self.destination)

    monkeypatch.setitem(
        sys.modules,
        "simready_bench.publishing",
        SimpleNamespace(HuggingFacePublisher=Publisher),
    )
    manifest = {"repetitions": 4}
    workflow.publication_callback(
        tmp_path, manifest, hf_repo="owner/results", hf_run="sol-medium"
    )
    resumed = workflow.publication_callback(tmp_path, manifest)
    assert resumed.destination == ("owner/results", "sol-medium", manifest)
    assert json.loads((tmp_path / "publication.json").read_text()) == {
        "repo": "owner/results",
        "run": "sol-medium",
    }
    assert len(checked) == 2
    with pytest.raises(ValueError, match="saved Hugging Face"):
        workflow.publication_callback(tmp_path, manifest, hf_repo="other/results")
    assert len(checked) == 2


def test_local_upload_failure_keeps_running_and_retries_without_new_compute(
    tmp_path, monkeypatch
):
    from simready_bench import controller_runner

    calls = []
    uploads = []
    report = {"directory": str(tmp_path), "results": str(tmp_path / "results.json")}

    class Reporter:
        def __init__(self, output, manifest):
            self.output = Path(output) / "reconstructions"

        def __call__(self, root, value):
            calls.append(value["completed_conditions"])

    def execute(manifest, *, on_progress, **options):
        on_progress(tmp_path, {"completed_conditions": 1})
        on_progress(tmp_path, {"completed_conditions": 2})
        return {"completed_conditions": 2, "planned_conditions": 2, "report": report}

    def publish(value, *, final=False):
        uploads.append((value, final))
        if len(uploads) == 1:
            raise ConnectionError("fixture upload interruption")

    monkeypatch.setattr(controller_runner, "ControllerReporter", Reporter)
    monkeypatch.setattr(workflow, "execute_local", execute)
    result = workflow.execute_local_reported(
        {"assets": [{"id": "box"}]}, policy={}, output_dir=tmp_path, on_report=publish
    )
    assert calls == [1, 2]
    assert len(uploads) == 3
    assert uploads[-1] == (report, True)
    assert result["completed_conditions"] == 2


def test_aggregating_a_single_asset_at_the_output_root_keeps_canonical_paths(tmp_path):
    row = {
        "condition_id": "condition-one",
        "status": "completed",
        "candidate": {
            "status": "available",
            "manifest": "candidates/one/manifest.json",
            "directory": "candidates/one",
            "entrypoint": "candidates/one/asset.usda",
            "files": [{"path": "candidates/one/asset.usda"}],
        },
        "nvidia_validator_report_file": {"path": "evidence/validator.json"},
    }
    source = tmp_path / "results.json"
    source.write_text(
        json.dumps({"schema": "simready-bench-portable-report-v1", "results": [row]})
    )
    report = workflow._aggregate_reports(
        [(tmp_path, {"results": str(source)})],
        output_dir=tmp_path,
        planned_conditions=1,
    )
    actual = json.loads(Path(report["results"]).read_text())["results"][0]
    assert actual["candidate"]["entrypoint"] == "candidates/one/asset.usda"
    assert actual["candidate"]["files"][0]["path"] == "candidates/one/asset.usda"
    assert actual["nvidia_validator_report_file"]["path"] == "evidence/validator.json"
