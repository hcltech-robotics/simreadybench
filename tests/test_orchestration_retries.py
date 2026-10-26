import copy
import json
import os
import time

import pytest

from simready_bench import daytona, orchestration, parallel, recovery, resume
from simready_bench.runtime_images import DEFAULT_IMAGES
from test_daytona_recovery import setup_run, source_failure
from test_parallel import manifest
from test_recovery_publication import retained_report
from test_resume import frozen_executor_wheel, portable, write


def select_conditions(original, conditions):
    selected = copy.deepcopy(original)
    selected["reconstruction_selection"] = {
        asset["id"]: [] for asset in original["assets"]
    }
    for condition in conditions:
        selected["reconstruction_selection"][condition["asset_id"]].append(
            condition["repetition"]
        )
    return selected


def test_unstarted_tail_membership_does_not_consume_transport_retries(
    tmp_path, monkeypatch, non_hf_policy
):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    conditions = list(resume.condition_plan(selected).items())
    output = tmp_path / "run"
    output.mkdir()
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir()
        roots[name] = str(output / name)
    policy = {**non_hf_policy, "roots": roots}
    wheel = output / "frozen.whl"
    frozen_executor_wheel(wheel)
    tail_id, tail = conditions[3]
    for number, (identity, actor) in zip(("005", "009", "010"), conditions[:3]):
        batch = output / "executions" / number
        shard = select_conditions(selected, [actor, tail])
        write(batch / "experiment.json", shard)
        write(
            batch / "daytona-delivery.json",
            {
                "schema": "simready-bench-daytona-delivery-v2",
                "cleanup_confirmed": True,
                "collection": "completed",
            },
        )
        write(
            batch / "retirement.json",
            {
                "requests": 32,
                "admitted_conditions": [identity],
                "unreported_admitted_conditions": [],
            },
        )
        portable(
            batch / "attempts/final",
            [
                (identity, actor, "infrastructure_error"),
                (tail_id, tail, "blocked"),
            ],
        )
    monkeypatch.setattr(resume, "_retire", lambda *_a, **_k: None)
    monkeypatch.setattr(resume, "_remaining", lambda *_: (32, 2.5))
    monkeypatch.setattr(resume, "_daytona_client", object)
    dispatched = []

    def execute(shard, **options):
        plan = resume.condition_plan(shard)
        dispatched.append(set(plan))
        assert set(plan) == {tail_id}
        portable(
            options["output_dir"] / "attempts/final",
            [
                (identity, condition, "completed")
                for identity, condition in plan.items()
            ],
        )

    monkeypatch.setattr(daytona, "run_daytona", execute)
    with pytest.raises(
        recovery.RecoveryLimitReached, match="Condition infrastructure retries"
    ) as caught:
        orchestration.coordinate_daytona(
            selected,
            policy=policy,
            output_dir=output,
            workers=1,
            wheel=wheel,
            cpu_snapshot=None,
            runtime_images=DEFAULT_IMAGES,
            max_retries=0,
        )
    assert dispatched == [{tail_id}]
    assert caught.value.result["completed_conditions"] == 1
    state = json.loads((output / "coordinator-state.json").read_text())
    assert set(state["blocked_conditions"]) == {identity for identity, _ in conditions[:3]}
    assert all(
        item["reason"] == "condition_infrastructure_retries_exhausted"
        for item in state["blocked_conditions"].values()
    )
    assert all(tail_id in state["dispatches"][batch] for batch in ("005", "009", "010"))


@pytest.mark.skipif(os.name == "nt", reason="Native allowance ownership requires Linux")
def test_exhausted_zero_admission_transport_scope_does_not_abort_live_collection(
    tmp_path, monkeypatch
):
    output, original, policy, parent, wheel = setup_run(tmp_path, monkeypatch)
    frozen_executor_wheel(wheel)
    conditions = list(resume.condition_plan(original).items())
    failed_id, failed = conditions[0]
    live_id, live = conditions[1]
    selected = select_conditions(original, [failed, live])
    for _ in range(3):
        batch, shard, _ = parallel._stage_shards(
            select_conditions(original, [failed]),
            policy,
            output,
            1,
            assets_root=policy["roots"]["assets"],
            skills_root=policy["roots"]["skills"],
            wheel=wheel,
            cpu_snapshot=None,
            runtime_images=DEFAULT_IMAGES,
        )[0]
        source_failure(batch, shard, policy, parent)
    live_batch, _, _ = parallel._stage_shards(
        select_conditions(original, [live]),
        policy,
        output,
        1,
        assets_root=policy["roots"]["assets"],
        skills_root=policy["roots"]["skills"],
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
    )[0]
    dispatched = []

    def execute(shard, **options):
        assert set(resume.condition_plan(shard)) == {live_id}
        dispatched.append(options["output_dir"])
        time.sleep(0.05)
        options["on_report"](retained_report(shard, options, policy))

    monkeypatch.setattr(daytona, "run_daytona", execute)
    options = dict(
        policy=policy,
        output_dir=output,
        workers=2,
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
    )
    with pytest.raises(
        recovery.RecoveryLimitReached, match="transport retries"
    ) as caught:
        orchestration.coordinate_daytona(selected, **options)
    assert dispatched == [live_batch]
    assert caught.value.result["completed_conditions"] == 1
    assert caught.value.result["results"][0]["condition_id"] == live_id
    state = json.loads((output / "coordinator-state.json").read_text())
    assert state["active_batches"] == []
    assert state["blocked_conditions"] == {
        failed_id: {
            "reason": "unadmitted_transport_retries_exhausted",
            "failed_batches": ["001", "002", "003"],
            "maximum_retries": 2,
        }
    }
    result = orchestration.coordinate_daytona(
        selected, **options, dispatch_pending=False
    )
    assert result["completed_conditions"] == 1
    assert dispatched == [live_batch]
    assert (
        json.loads((output / "coordinator-state.json").read_text())[
            "blocked_conditions"
        ]
        == state["blocked_conditions"]
    )
