import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from simready_bench import benchmark as workflow
from simready_bench.cli import main


@pytest.mark.parametrize("workers", [None, 4, 25])
def test_cli_passes_worker_count_with_resume(tmp_path, monkeypatch, workers):
    seen = []

    def execute(reference, **options):
        seen.append(options)
        return {"completed_conditions": 0, "planned_conditions": 0, "results": []}

    monkeypatch.setattr(workflow, "benchmark", execute)
    arguments = [
        "benchmark",
        "--model",
        "gpt-6.1-sol",
        "--deploy",
        "daytona",
        "--output-dir",
        str(tmp_path),
        "--resume",
    ]
    if workers is not None:
        arguments.extend(["--workers", str(workers)])
    assert main(arguments) == 0
    assert seen[0]["workers"] == (workers or 1)
    assert seen[0]["resume"] is True
    assert seen[0]["max_retries"] == 2
    assert seen[0]["retry_delay_seconds"] == 5
    assert seen[0]["maximum_retry_delay_seconds"] == 60


@pytest.mark.parametrize("value", ["-1", "0", "2.5", "true"])
def test_cli_rejects_invalid_worker_count_before_dispatch(tmp_path, monkeypatch, value):
    def unexpected(*args, **kwargs):
        raise AssertionError("Invalid worker count reached benchmark dispatch")

    monkeypatch.setattr(workflow, "benchmark", unexpected)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "benchmark",
                "--model",
                "gpt-6.1-sol",
                "--deploy",
                "daytona",
                "--output-dir",
                str(tmp_path),
                "--workers",
                value,
            ]
        )
    assert error.value.code == 2


@pytest.mark.parametrize("workers", [False, True, -1, 0, 2.0, "2", None])
def test_api_rejects_invalid_worker_count_without_preparing_run(tmp_path, workers):
    output = tmp_path / "unallocated"
    with pytest.raises(ValueError, match="Workers must be a positive integer"):
        workflow.benchmark(
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            workers=workers,
        )
    assert not output.exists()


@pytest.mark.parametrize("resume", [False, True])
def test_local_parallel_workers_are_rejected_before_preparation(tmp_path, resume):
    output = tmp_path / "unallocated"
    with pytest.raises(ValueError, match="Parallel workers require Daytona"):
        workflow.benchmark(
            model="gpt-6.1-sol",
            output_dir=output,
            workers=2,
            resume=resume,
        )
    assert not output.exists()


@pytest.mark.parametrize("workers", [8, 25])
def test_resume_receives_worker_count_without_recomposing_experiment(
    tmp_path, monkeypatch, workers
):
    from simready_bench import resume

    observed = []
    sentinel = object()

    def continue_saved(reference, **options):
        observed.append((reference, options))
        return sentinel

    monkeypatch.setattr(resume, "resume_benchmark", continue_saved)
    assert (
        workflow.benchmark(
            "NV25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=tmp_path,
            workers=workers,
            resume=True,
        )
        is sentinel
    )
    assert observed[0][0] == "NV25"
    assert observed[0][1]["workers"] == workers
    assert observed[0][1]["max_retries"] == 2


def test_cli_passes_explicit_recovery_policy(tmp_path, monkeypatch):
    calls = []

    def execute(reference, **options):
        calls.append(options)
        return {"completed_conditions": 0, "planned_conditions": 0, "results": []}

    monkeypatch.setattr(workflow, "benchmark", execute)
    assert (
        main(
            [
                "benchmark",
                "--model",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--max-retries",
                "0",
                "--retry-delay-seconds",
                "7",
                "--maximum-retry-delay-seconds",
                "20",
            ]
        )
        == 0
    )
    assert calls[0]["max_retries"] == 0
    assert calls[0]["retry_delay_seconds"] == 7
    assert calls[0]["maximum_retry_delay_seconds"] == 20


@pytest.mark.parametrize(
    "options",
    [
        {"max_retries": False},
        {"max_retries": -1},
        {"max_retries": 11},
        {"retry_delay_seconds": -1},
        {"retry_delay_seconds": float("nan")},
        {"retry_delay_seconds": 20, "maximum_retry_delay_seconds": 10},
        {"dispatch_pending": False},
    ],
)
def test_recovery_policy_is_validated_before_preparing_a_run(tmp_path, options):
    output = tmp_path / "unallocated"
    with pytest.raises(ValueError):
        workflow.benchmark(model="fixture", output_dir=output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "deploy,workers", [("local", 1), ("daytona", 1), ("daytona", 4)]
)
@pytest.mark.parametrize("raises", [False, True])
def test_fresh_failure_enters_saved_recovery_without_new_authority(
    tmp_path,
    monkeypatch,
    deploy,
    workers,
    raises,
):
    from simready_bench import daytona, parallel, resume

    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    observed = []
    sentinel = object()

    def policy(**options):
        output = options["output_dir"]
        return {
            "roots": {
                name: str(output / name)
                for name in ("assets", "skills", "input_views", "private_manifests")
            },
            "providers": {"selected": options["provider"]},
            "limits": {},
            "runtime": {},
        }

    def execute(manifest, **options):
        observed.append(("initial", manifest, options))
        if raises:
            raise RuntimeError("Provider unavailable")
        return {"completed_conditions": 3, "planned_conditions": 100, "results": []}

    def continue_saved(reference, **options):
        observed.append(("recovery", reference, options))
        saved = json.loads((options["output_dir"] / "run.json").read_text())
        original = observed[0][2]["policy"]
        assert saved["operator_sha256"] == resume._digest(original)
        return sentinel

    monkeypatch.setattr(workflow, "prepare_policy", policy)
    monkeypatch.setattr(workflow, "prepare_source_views", lambda *args: None)
    monkeypatch.setattr(workflow, "publication_callback", lambda *args, **kwargs: None)
    monkeypatch.setattr(workflow, "execute_local_reported", execute)
    monkeypatch.setattr(daytona, "run_daytona", execute)
    monkeypatch.setattr(parallel, "run_parallel_daytona", execute)
    monkeypatch.setattr(resume, "resume_benchmark", continue_saved)
    output = tmp_path / "run"
    assert (
        workflow.benchmark(
            model="gpt-6.1-sol",
            deploy=deploy,
            workers=workers,
            output_dir=output,
            maximum_spend_usd=250,
            reasoning="medium",
        )
        is sentinel
    )
    assert [item[0] for item in observed] == ["initial", "recovery"]
    initial = observed[0][2]["policy"]["providers"]["selected"]["project_budget"]
    assert initial["maximum_spend_usd"] == 250
    assert initial["maximum_requests"] == (19200 if deploy == "daytona" else 6400)
    options = observed[1][2]
    assert options["output_dir"] == output
    assert options["max_retries"] == 2 and options["_initial_failure"] is True
    assert options["maximum_spend_usd"] == 250
    assert options["workers"] == workers


def test_workers_change_executor_only_and_preserve_scientific_configuration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key")
    calls = []
    reports = []

    def policy(**options):
        output = options["output_dir"]
        return {
            "roots": {
                name: str(output / name)
                for name in ("assets", "skills", "input_views", "private_manifests")
            },
            "providers": {"selected": options["provider"]},
            "limits": {},
            "runtime": {},
        }

    def executor(kind):
        def execute(manifest, **options):
            calls.append((kind, manifest, options))
            return {"completed_conditions": 0, "planned_conditions": 100, "results": []}

        return execute

    def publish(report, *, final=False):
        reports.append((report, final))

    monkeypatch.setattr(workflow, "prepare_policy", policy)
    monkeypatch.setattr(
        workflow, "publication_callback", lambda *args, **kwargs: publish
    )
    monkeypatch.setitem(
        sys.modules,
        "simready_bench.daytona",
        SimpleNamespace(
            run_daytona=executor("serial"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "simready_bench.parallel",
        SimpleNamespace(
            run_parallel_daytona=executor("parallel"),
        ),
    )
    wheel = tmp_path / "frozen.whl"
    work = tmp_path / "work"
    for workers in (1, 4):
        workflow.benchmark(
            "NV25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=tmp_path / ("run-" + str(workers)),
            maximum_spend_usd=250,
            reasoning="medium",
            workers=workers,
            max_retries=0,
            wheel=wheel,
            cpu_snapshot="frozen-snapshot",
            work_dir=work,
        )

    assert [kind for kind, _, _ in calls] == ["serial", "parallel"]
    assert calls[0][1] == calls[1][1]
    assert "workers" not in calls[0][1]
    assert "workers" not in calls[0][2]
    assert calls[1][2]["workers"] == 4
    from simready_bench.resume import condition_plan

    assert condition_plan(calls[0][1]) == condition_plan(calls[1][1])
    for _, manifest, options in calls:
        assert options["wheel"] == wheel
        assert options["cpu_snapshot"] == "frozen-snapshot"
        assert options["work_dir"] == work
        assert options["on_report"] is publish
        provider_budget = options["policy"]["providers"]["selected"]["project_budget"]
        assert provider_budget["maximum_spend_usd"] == 250
        assert provider_budget["maximum_requests"] == 6400
        assert manifest["defaults"]["budgets"]["maximum_spend_usd"] == 2.5
        saved = Path(options["output_dir"]) / "run.json"
        assert (
            json.loads(saved.read_text())["runtime_images"] == options["runtime_images"]
        )
    assert reports == []
