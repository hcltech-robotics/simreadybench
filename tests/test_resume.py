import hashlib
import json
import time
import zipfile
from dataclasses import replace

import pytest

from simready_bench import benchmark as workflow
from simready_bench import resume
from simready_bench.engine.inference_gateway import ProjectBudget
from simready_bench.local import ParentAuthorityLedger
from simready_bench.runtime_images import DEFAULT_IMAGES

pytestmark = pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="The launcher allowance ledger uses Linux ownership",
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def frozen_executor_wheel(path):
    with zipfile.ZipFile(path, 'w') as archive:
        member = zipfile.ZipInfo('simready_bench/config.py', date_time=(2020, 1, 1, 0, 0, 0))
        archive.writestr(member, b'REQUESTS = 64\n')
    return path


def portable(attempt, rows, configuration="c" * 64):
    attempt.mkdir(parents=True)
    candidate = attempt / "candidate"
    candidate.mkdir()
    usd = b'#usda 1.0\ndef Xform "Asset" {}\n'
    (candidate / "asset.usd").write_bytes(usd)
    members = [
        {
            "relative_path": "asset.usd",
            "size_bytes": len(usd),
            "sha256": hashlib.sha256(usd).hexdigest(),
        }
    ]
    snapshot = resume._digest({"entrypoint": "asset.usd", "files": members})
    document = {
        "schema": "simready-bench-submission-v1",
        "entrypoint": "asset.usd",
        "files": members,
        "snapshot_sha256": snapshot,
        "size_bytes": len(usd),
        "copied_independently": True,
    }
    write(candidate / "manifest.json", document)
    findings = attempt / "evidence/nvidia.json"
    write(findings, {"issues": []})
    records = []
    for identifier, condition, status in rows:
        row = {
            **condition,
            "condition_id": identifier,
            "status": status,
            "requested_model": "gpt-6.1-sol",
        }
        if status == "completed":
            row.update(
                trial_id=identifier,
                inference_calls_used=1,
                snapshot_sha256=snapshot,
                simready_reward=100,
                simready={
                    "schema": "simready-validator-score-v1",
                    "score_percent": 100,
                },
                candidate={
                    "status": "available",
                    "snapshot_sha256": snapshot,
                    "entrypoint": "candidate/asset.usd",
                    "manifest": "candidate/manifest.json",
                    "directory": "candidate",
                    "manifest_sha256": hashlib.sha256(
                        (candidate / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "files": [{"path": "candidate/asset.usd", **members[0]}],
                },
                nvidia_validator_report_file={
                    "path": "evidence/nvidia.json",
                    "sha256": hashlib.sha256(findings.read_bytes()).hexdigest(),
                },
            )
        else:
            row.update(
                trial_id=None,
                inference_calls_used=None,
                simready=None,
                candidate={"status": "unavailable"},
            )
        records.append(row)
    path = attempt / "results.json"
    write(
        path,
        {
            "schema": "simready-bench-portable-report-v1",
            "configuration_sha256": configuration,
            "results": records,
        },
    )
    return path


def stopped_run(tmp_path, monkeypatch, *, fourth="blocked", unmatched=False):
    output = tmp_path / "results"
    output.mkdir()
    wheel = output / ".executor-package/simready_bench-original.whl"
    wheel.parent.mkdir()
    frozen_executor_wheel(wheel)
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir()
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    source, _, _ = workflow.reference_manifest("nvidia-25")
    model = workflow.select_model("gpt-6.1-sol", provider="openai")[0]
    model["generation"] = {"reasoning_effort": "medium"}
    manifest = workflow.compose_experiment(
        source, model=model, maximum_spend_usd=250, repetitions=4, authoring_turns=32
    )
    budget = ProjectBudget(
        project="simready-bench",
        authorization_id="original-project",
        approved=True,
        upstream_origin="https://api.openai.com",
        model="gpt-6.1-sol",
        expires_at=time.time() + 3600,
        maximum_requests=3200,
        maximum_spend_usd=250,
        pricing_status="known",
    )
    from dataclasses import asdict

    policy = {
        "roots": {
            name: str(output / name)
            for name in ("assets", "skills", "input_views", "private_manifests")
        },
        "providers": {
            "selected": {
                "kind": "openai",
                "api_key_env": "TEST_RESUME_KEY",
                "base_url": "https://api.openai.com/v1",
                "project_budget": asdict(budget),
            }
        },
        "limits": {
            "maximum_spend_usd": 250,
            "maximum_condition_spend_usd": 2.5,
            "maximum_matrix_runs": 100,
        },
        "runtime": {"authority_state_dir": str(authority)},
    }
    write(output / "experiment.json", manifest)
    write(output / "operator.json", policy)
    write(output / "runtime-images.json", DEFAULT_IMAGES)
    resume.save_run_configuration(
        output,
        reference="nvidia-25",
        deploy="daytona",
        manifest=manifest,
        policy=policy,
        runtime_images=DEFAULT_IMAGES,
    )
    plan = resume.condition_plan(manifest)
    first_asset = manifest["assets"][0]["id"]
    first = [
        (identifier, condition, "completed" if condition["repetition"] < 3 else fourth)
        for identifier, condition in plan.items()
        if condition["asset_id"] == first_asset
    ]
    path = portable(output / "attempts/000-reference", first)
    if fourth != "blocked":
        value = json.loads(path.read_text())
        value["results"][3].update(trial_id="started-fourth", inference_calls_used=1)
        write(path, value)
    outer = ParentAuthorityLedger(authority)
    outer.reserve(
        budget,
        allocation_id="outer-allocation",
        requests=3200,
        maximum_spend_usd=250,
        work_root=output,
    )
    remote_root = output / "daytona-executor/authority"
    remote_root.mkdir(parents=True, mode=0o700)
    remote = ParentAuthorityLedger(remote_root)
    child = replace(
        budget,
        authorization_id="cloud-outer-allocation",
        upstream_origin="http://127.0.0.1:1234",
    )
    for identifier, _, _ in first:
        allocation = resume._digest(
            {
                "work_root": "/dev/shm/bench/a0",
                "configuration": "c" * 64,
                "condition": identifier,
            }
        )
        remote.reserve(
            child,
            allocation_id=allocation,
            requests=32,
            maximum_spend_usd=2.5,
            work_root="/dev/shm/bench/a0",
        )
    if unmatched:
        remote.reserve(
            child,
            allocation_id="unexplained-child",
            requests=32,
            maximum_spend_usd=2.5,
            work_root="/dev/shm/bench/a1",
        )
    write(output / "daytona-executor/controller-status.json", {"status": "failed"})
    write(
        output / "daytona-delivery.json",
        {
            "collection": "completed",
            "cleanup_confirmed": True,
            "sandbox_id": "deleted-executor",
            "parent_allocation": {"allocation_id": "outer-allocation"},
            "planned_requests": 3200,
            "maximum_spend_usd": 250,
            "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "snapshot": "simready-bench-cpu",
            "runtime_images": DEFAULT_IMAGES,
        },
    )

    class Deleted:
        def get(self, identifier):
            from daytona.common.errors import DaytonaNotFoundError

            raise DaytonaNotFoundError("deleted")

    monkeypatch.setattr(resume, "_daytona_client", Deleted)
    monkeypatch.setenv("TEST_RESUME_KEY", "fixture-key")
    return output, manifest, policy, outer


def test_resume_uses_original_parent_and_finishes_exactly_one_result_per_condition(
    tmp_path, monkeypatch
):
    output, manifest, policy, ledger = stopped_run(tmp_path, monkeypatch)
    called = []

    def execute(selected, **options):
        called.append(selected)
        assert options["policy"]["providers"] == policy["providers"]
        assert (
            options["policy"]["runtime"]["authority_state_dir"]
            == policy["runtime"]["authority_state_dir"]
        )
        batch = options["output_dir"]
        assert batch == output / "executions" / str(len(called)).zfill(3)
        assert options["runtime_images"] == DEFAULT_IMAGES
        conditions = resume.condition_plan(selected)
        assert len(conditions) == 1
        assert not set(list(resume.condition_plan(manifest))[:3]) & conditions.keys()
        parent = ProjectBudget(**policy["providers"]["selected"]["project_budget"])
        ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
            parent,
            allocation_id="shard-" + batch.name,
            requests=32,
            maximum_spend_usd=2.5,
            work_root=batch,
        )
        for index, asset in enumerate(selected["assets"]):
            rows = [
                (identifier, condition, "completed")
                for identifier, condition in conditions.items()
                if condition["asset_id"] == asset["id"]
            ]
            if rows:
                portable(
                    options["output_dir"]
                    / ("attempts/" + str(index).zfill(3) + "-asset"),
                    rows,
                    "d" * 64,
                )
        return {"status": "completed"}

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    result = workflow.benchmark(
        "nvidia-25",
        model="gpt-6.1-sol",
        harness="pi",
        deploy="daytona",
        output_dir=output,
        repetitions=4,
        resume=True,
    )
    assert len(called) == 97
    assert len({identity for shard in called for identity in resume.condition_plan(shard)}) == 97
    assert result["completed_conditions"] == len(result["results"]) == 100
    assert len({row["condition_id"] for row in result["results"]}) == 100
    assert result["partial"] is False
    assert resume._remaining(policy, manifest) == (0, 0)
    with ledger._db() as db:
        assert db.execute("SELECT id FROM parents").fetchall() == [
            ("original-project",)
        ]
        assert db.execute("SELECT requests,usd FROM retirements").fetchall() == [
            (96, "7.5")
        ]


def test_started_unscored_condition_keeps_its_full_ceiling(tmp_path, monkeypatch):
    from simready_bench.recovery import RecoveryLimitReached

    output, manifest, policy, ledger = stopped_run(
        tmp_path, monkeypatch, fourth="infrastructure_error"
    )
    admitted = set(list(resume.condition_plan(manifest))[:4])
    dispatched = []

    def execute(selected, **options):
        conditions = resume.condition_plan(selected)
        assert len(conditions) == 1 and not admitted & conditions.keys()
        dispatched.append(set(conditions))
        batch = options["output_dir"]
        parent = ProjectBudget(**policy["providers"]["selected"]["project_budget"])
        ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
            parent,
            allocation_id="shard-" + batch.name,
            requests=32,
            maximum_spend_usd=2.5,
            work_root=batch,
        )
        portable(
            options["output_dir"] / "attempts/remaining",
            [
                (identity, condition, "completed")
                for identity, condition in conditions.items()
            ],
        )

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    with pytest.raises(RecoveryLimitReached) as caught:
        workflow.benchmark(
            "nvidia-25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            resume=True,
            max_retries=0,
        )
    assert len(dispatched) == 96
    assert len(set().union(*dispatched)) == 96
    assert caught.value.result["completed_conditions"] == 99
    assert resume._remaining(policy, manifest) == (0, 0)
    with ledger._db() as db:
        assert db.execute("SELECT requests,usd FROM retirements").fetchall() == [
            (128, "10.0")
        ]


def test_unexplained_issued_child_refuses_retirement(tmp_path, monkeypatch):
    output, _, _, ledger = stopped_run(tmp_path, monkeypatch, unmatched=True)
    with pytest.raises(ValueError, match="no matching condition"):
        workflow.benchmark(
            "nvidia-25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            resume=True,
        )
    with ledger._db() as db:
        assert db.execute("SELECT * FROM retirements").fetchall() == []


def test_existing_executor_and_changed_configuration_cannot_resume(
    tmp_path, monkeypatch
):
    output, _, _, ledger = stopped_run(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="must remain unchanged"):
        workflow.benchmark(
            "nvidia-25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            resume=True,
            reasoning="high",
        )

    class Existing:
        def get(self, identifier):
            return object()

    monkeypatch.setattr(resume, "_daytona_client", Existing)
    with pytest.raises(ValueError, match="still exists"):
        workflow.benchmark(
            "nvidia-25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            resume=True,
        )
    with ledger._db() as db:
        assert db.execute("SELECT * FROM retirements").fetchall() == []


def test_cli_passes_resume_to_saved_benchmark(tmp_path, monkeypatch):
    from simready_bench.cli import main

    seen = []

    def execute(reference, **options):
        seen.append(options)
        return {"completed_conditions": 0, "planned_conditions": 0, "results": []}

    monkeypatch.setattr(workflow, "benchmark", execute)
    assert (
        main(
            [
                "benchmark",
                "--model",
                "gpt-6.1-sol",
                "--deploy",
                "daytona",
                "--resume",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert seen[0]["resume"] is True


def test_resume_reconnects_before_reading_reports_and_reclaiming_allowance(
    tmp_path, monkeypatch
):
    output, manifest, _, ledger = stopped_run(tmp_path, monkeypatch)
    pending = tmp_path / "uncollected-attempts"
    (output / "attempts").rename(pending)
    delivery_path = output / "daytona-delivery.json"
    delivery = json.loads(delivery_path.read_text())
    delivery.update(
        collection=None,
        cleanup_confirmed=False,
        controller_started=True,
        command_id="original-command",
    )
    write(delivery_path, delivery)
    events = []
    dispatched = []

    def reconnect(selected, **options):
        assert selected == manifest and options["output_dir"] == output
        events.append("reconnected-original")
        pending.rename(output / "attempts")
        delivery.update(collection="completed", cleanup_confirmed=True)
        write(delivery_path, delivery)

    def execute(selected, **options):
        events.append("new-batch")
        dispatched.append((options["output_dir"], set(resume.condition_plan(selected))))
        assert len(resume.condition_plan(selected)) == 1
        raise RuntimeError("stop after scheduling")

    monkeypatch.setattr("simready_bench.daytona.reconnect_daytona", reconnect)
    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    from simready_bench.recovery import RecoveryLimitReached

    with pytest.raises(RecoveryLimitReached, match="bounded recovery policy") as caught:
        workflow.benchmark(
            "nvidia-25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            resume=True,
            retry_delay_seconds=0,
            maximum_retry_delay_seconds=0,
        )
    assert events == ["reconnected-original", "new-batch", "new-batch", "new-batch"]
    assert all(dispatch == dispatched[0] for dispatch in dispatched)
    assert str(caught.value.__cause__) == "stop after scheduling"
    with ledger._db() as db:
        assert db.execute("SELECT requests,usd FROM retirements").fetchall() == [
            (96, "7.5")
        ]


def test_resume_unknown_observation_cannot_retire_or_start_another_batch(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import DaytonaObservationError

    output, _, _, ledger = stopped_run(tmp_path, monkeypatch)
    path = output / "daytona-delivery.json"
    delivery = json.loads(path.read_text())
    delivery.update(collection=None, cleanup_confirmed=False)
    write(path, delivery)

    def unavailable(*args, **kwargs):
        raise DaytonaObservationError("Original executor is not confirmed terminal")

    monkeypatch.setattr("simready_bench.daytona.reconnect_daytona", unavailable)
    with pytest.raises(DaytonaObservationError):
        workflow.benchmark(
            "nvidia-25",
            model="gpt-6.1-sol",
            deploy="daytona",
            output_dir=output,
            resume=True,
        )
    assert not (output / "executions").exists()
    with ledger._db() as db:
        assert db.execute("SELECT * FROM retirements").fetchall() == []
