"""Stopped local runs recover unused ceilings without refunding admitted work."""

from dataclasses import asdict
from decimal import Decimal
import os
from pathlib import Path
import tempfile
import time

import pytest

from simready_bench import benchmark as workflow
from simready_bench import local, resume
from simready_bench.engine.inference_gateway import ProjectBudget
from simready_bench.runtime_images import DEFAULT_IMAGES
from test_resume import portable, write

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Local execution uses Linux ownership")


@pytest.fixture
def interrupted_local_run(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir()
    state = tmp_path / "authority"
    state.mkdir(mode=0o700)
    source, _, _ = workflow.reference_manifest("nvidia-25")
    source["assets"] = source["assets"][:1]
    model = workflow.select_model("gpt-6.1-sol", provider="openai")[0]
    manifest = workflow.compose_experiment(
        source, model=model, maximum_spend_usd=10, authoring_turns=32,
    )
    budget = ProjectBudget(
        project="simready-bench", authorization_id="local-original", approved=True,
        upstream_origin="https://api.openai.com", model="gpt-6.1-sol",
        expires_at=time.time() + 3600, maximum_requests=128,
        maximum_spend_usd=10, pricing_status="known",
    )
    policy = {
        "roots": {name: str(output / name) for name in ("assets", "skills", "input_views", "private_manifests")},
        "providers": {"selected": {"kind": "openai", "api_key_env": "TEST_LOCAL_KEY", "base_url": "https://api.openai.com/v1", "project_budget": asdict(budget)}},
        "limits": {"maximum_spend_usd": 10, "maximum_condition_spend_usd": 2.5, "maximum_matrix_runs": 4},
        "runtime": {"authority_state_dir": str(state)},
    }
    write(output / "experiment.json", manifest)
    write(output / "operator.json", policy)
    write(output / "runtime-images.json", DEFAULT_IMAGES)
    resume.save_run_configuration(output, reference="nvidia-25", deploy="local", manifest=manifest, policy=policy, runtime_images=DEFAULT_IMAGES)
    plan = resume.condition_plan(manifest)
    conditions = [
        {**condition, "condition_id": identifier, "model": model,
         "settings": {"budgets": manifest["defaults"]["budgets"]}}
        for identifier, condition in plan.items()
    ]
    resolved = {
        "conditions": conditions, "configuration_sha256": "c" * 64,
        "aggregate_budget": {"maximum_spend_usd": 10}, "operator_policy": policy,
    }
    monkeypatch.setenv("TEST_LOCAL_KEY", "fixture-key")
    monkeypatch.setattr(local, "checked_resolved", lambda value, policy: value)
    monkeypatch.setattr(local, "runtime_policy", lambda *args, **kwargs: None)
    monkeypatch.setattr(local, "_provider_authorities", lambda value: None)
    monkeypatch.setattr(local, "prepare", lambda value, **kwargs: {"conditions": value["conditions"]})
    monkeypatch.setattr(local, "bounded_controller_class", lambda: lambda root: object())
    with tempfile.TemporaryDirectory(prefix="srl-", dir="/dev/shm") as temporary:
        yield output, manifest, policy, budget, resolved, plan, Path(temporary) / "work"


def interrupt_after_three(fixture, monkeypatch, *, fourth_admitted=False):
    output, manifest, policy, budget, resolved, plan, work = fixture
    calls = []

    async def condition(entry, *args):
        calls.append(entry["condition_id"])
        if fourth_admitted and len(calls) == 4:
            raise ConnectionError("Uncertain admitted reconstruction")
        return {**plan[entry["condition_id"]], "condition_id": entry["condition_id"], "status": "completed", "trial_id": entry["condition_id"]}

    def progress(root, value):
        if not fourth_admitted and len(value["results"]) == 3:
            raise ConnectionError("Interrupted before the fourth admission")

    original_write = local._write_progress

    def write_progress(root, value, results, allowance):
        if fourth_admitted and len(results) == 4:
            raise OSError("The fourth admission has no durable progress result")
        return original_write(root, value, results, allowance)

    monkeypatch.setattr(local, "_condition", condition)
    monkeypatch.setattr(local, "_write_progress", write_progress)
    with pytest.raises(local.RunExecutionError) as stopped:
        local.run(resolved, work_dir=work, policy=policy, on_progress=progress)
    return stopped.value.result, calls


def test_three_completed_and_one_unrun_resume_inside_the_original_ceiling(
    interrupted_local_run, monkeypatch
):
    output, manifest, policy, budget, resolved, plan, work = interrupted_local_run
    result, original_calls = interrupt_after_three(interrupted_local_run, monkeypatch)
    assert len(original_calls) == 3
    assert result["results"][-1]["status"] == "blocked"
    authority = local.ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    with authority._db() as db:
        assert db.execute("SELECT requests,usd FROM retirements").fetchall() == [(0, "0")]
        retired = db.execute("SELECT allocation FROM retirements").fetchone()[0]
    assert resume._remaining(policy, manifest) == (32, Decimal("2.5"))
    portable(output / "attempts/000-reference", [
        (identifier, condition, "completed" if identifier in original_calls else "blocked")
        for identifier, condition in plan.items()
    ])
    resumed_calls = []

    def execute(selected, **options):
        remaining = resume.condition_plan(selected)
        assert set(remaining) == set(plan) - set(original_calls)
        for identifier in remaining:
            authority.reserve(budget, allocation_id="resumed-fourth", requests=32, maximum_spend_usd=2.5, work_root=options["output_dir"])
            resumed_calls.append(identifier)
        portable(options["output_dir"] / "attempts/000-reference", [
            (identifier, condition, "completed") for identifier, condition in remaining.items()
        ], configuration="d" * 64)

    monkeypatch.setattr(workflow, "execute_local_reported", execute)
    finished = workflow.benchmark("nvidia-25", model="gpt-6.1-sol", deploy="local", output_dir=output, repetitions=4, resume=True)
    assert finished["completed_conditions"] == 4
    assert len(resumed_calls) == 1
    assert resume._remaining(policy, manifest) == (0, Decimal("0.0"))
    with pytest.raises(ValueError, match="retired allocation"):
        authority.reserve(budget, allocation_id=retired, requests=32, maximum_spend_usd=2.5, work_root=work)
    with pytest.raises(ValueError, match="allowance exhausted"):
        authority.reserve(budget, allocation_id="cannot-respend-completed", requests=32, maximum_spend_usd=2.5, work_root=work / "another")
    with authority._db() as db:
        assert db.execute("SELECT id FROM parents").fetchall() == [(budget.authorization_id,)]
        assert len(db.execute("SELECT id FROM allocations").fetchall()) == 5


def test_an_admitted_but_unscored_fourth_reconstruction_retains_its_ceiling(
    interrupted_local_run, monkeypatch
):
    output, manifest, policy, budget, resolved, plan, work = interrupted_local_run
    result, calls = interrupt_after_three(interrupted_local_run, monkeypatch, fourth_admitted=True)
    assert len(calls) == 4
    assert result["results"][-1]["status"] == "blocked"
    with local.sqlite3.connect(work / "matrix.sqlite3") as db:
        assert len(db.execute("SELECT id FROM conditions").fetchall()) == 4
    authority = local.ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    with authority._db() as db:
        assert db.execute("SELECT * FROM retirements").fetchall() == []
    assert resume._remaining(policy, manifest) == (0, Decimal("0.0"))
    with pytest.raises(ValueError, match="allowance exhausted"):
        authority.reserve(budget, allocation_id="uncertain-retry", requests=32, maximum_spend_usd=2.5, work_root=work / "retry")


def test_a_second_invocation_cannot_retire_an_existing_work_directory(
    interrupted_local_run, monkeypatch
):
    output, manifest, policy, budget, resolved, plan, work = interrupted_local_run
    work.mkdir(mode=0o700)
    authority = local.ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    identifier = next(iter(plan))
    allocation = local.digest({"work_root": str(work), "configuration": resolved["configuration_sha256"], "condition": identifier})
    authority.reserve(budget, allocation_id=allocation, requests=32, maximum_spend_usd=2.5, work_root=work)
    with pytest.raises(local.RunExecutionError, match="existing work directory"):
        local.run(resolved, work_dir=work, policy=policy)
    with authority._db() as db:
        assert db.execute("SELECT * FROM retirements").fetchall() == []
