import json
import os
from pathlib import Path
import threading
from dataclasses import replace

import pytest

from simready_bench import benchmark as workflow
from simready_bench import parallel, resume
from simready_bench.engine.inference_gateway import ProjectBudget
from simready_bench.local import ParentAuthorityLedger
from test_automatic_recovery import arguments
from test_parallel import manifest
from test_resume import frozen_executor_wheel, portable, stopped_run, write


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Linux allowance ownership")


def committing_publisher():
    committed, publications = {}, []

    def publish(report, **_):
        rows = json.loads(Path(report["results"]).read_text())["results"]
        current = {row["condition_id"]: row["status"] for row in rows}
        assert set(committed) <= set(current)
        assert all(
            current[identity] == "completed"
            for identity, status in committed.items()
            if status == "completed"
        )
        committed.update(current)
        publications.append(set(current))
        if len(publications) == 1:
            raise RuntimeError("Remote commit succeeded before the connection failed")

    return publish, publications


def retained_report(shard, options, policy):
    conditions = resume.condition_plan(shard)
    batch = options["output_dir"]
    parent = ProjectBudget(**policy["providers"]["selected"]["project_budget"])
    allocation = "shard-" + batch.name
    requests, dollars = 32 * len(conditions), 2.5 * len(conditions)
    ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
        parent,
        allocation_id=allocation,
        requests=requests,
        maximum_spend_usd=dollars,
        work_root=batch,
    )
    child_root = batch / "daytona-executor/authority"
    child_root.mkdir(parents=True, mode=0o700)
    child = replace(
        parent,
        authorization_id="cloud-" + allocation,
        upstream_origin="http://127.0.0.1:1234",
        maximum_requests=requests,
        maximum_spend_usd=dollars,
    )
    work_root = "/dev/shm/publication/" + batch.name
    for identity in conditions:
        ParentAuthorityLedger(child_root).reserve(
            child,
            allocation_id=resume._digest({
                "work_root": work_root,
                "configuration": "c" * 64,
                "condition": identity,
            }),
            requests=32,
            maximum_spend_usd=2.5,
            work_root=work_root,
        )
    path = portable(
        batch / "attempts/reference",
        [(identity, condition, "completed") for identity, condition in conditions.items()],
    )
    write(batch / "daytona-executor/controller-status.json", {"status": "completed"})
    write(batch / "daytona-delivery.json", {
        "schema": "simready-bench-daytona-delivery-v2",
        "collection": "completed",
        "cleanup_confirmed": True,
        "sandbox_id": "deleted-publication-" + batch.name,
        "parent_allocation": {"allocation_id": allocation},
        "planned_requests": requests,
        "maximum_spend_usd": dollars,
    })
    return {"directory": str(path.parent), "results": str(path)}


def deleted_client(monkeypatch):
    class Deleted:
        def get(self, identifier):
            from daytona.common.errors import DaytonaNotFoundError

            raise DaytonaNotFoundError("deleted")

    monkeypatch.setattr(resume, "_daytona_client", Deleted)


def test_parallel_keeps_a_valid_report_when_publication_commits_then_raises(
    tmp_path, monkeypatch
):
    selected = manifest()
    selected["assets"] = selected["assets"][:2]
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    wheel = tmp_path / "original.whl"
    frozen_executor_wheel(wheel)
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    provider = workflow._new_provider(
        "openai", "gpt-6.1-sol", api_key_env=None,
        maximum_spend_usd=20, conditions=8, authoring_turns=32,
    )
    policy = {
        "roots": roots,
        "providers": {"selected": provider},
        "runtime": {"authority_state_dir": str(authority)},
        "limits": {"maximum_spend_usd": 20, "maximum_condition_spend_usd": 2.5,
                   "maximum_matrix_runs": 8, "inference_requests": 32},
    }
    publish, publications = committing_publisher()
    first = threading.Event()
    dispatched = []

    def execute(shard, **options):
        dispatched.append(set(resume.condition_plan(shard)))
        if options["output_dir"].name != "001":
            assert first.wait(timeout=10)
        report = retained_report(shard, options, policy)
        options["on_report"](report)
        if options["output_dir"].name == "001":
            first.set()

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    deleted_client(monkeypatch)
    result = parallel.run_parallel_daytona(
        selected,
        policy=policy,
        output_dir=tmp_path,
        workers=2,
        wheel=wheel,
        on_report=publish,
    )
    assert result["completed_conditions"] == 8
    assert len(publications[0]) == 1 and publications[0] < publications[1]
    assert len(publications[-1]) == 8
    assert all(before <= after for before, after in zip(publications, publications[1:]))
    assert len(dispatched) == 8 and all(len(conditions) == 1 for conditions in dispatched)
    assert set().union(*dispatched) == set(resume.condition_plan(selected))
    assert resume._remaining(policy, selected) == (0, 0)


def test_resume_keeps_a_valid_report_when_publication_commits_then_raises(
    tmp_path, monkeypatch
):
    output, original, policy, _ = stopped_run(tmp_path, monkeypatch)
    publish, publications = committing_publisher()
    dispatched = []
    monkeypatch.setattr(
        workflow, "publication_callback", lambda *args, **kwargs: publish
    )

    def execute(shard, **options):
        dispatched.append(set(resume.condition_plan(shard)))
        options["on_report"](retained_report(shard, options, policy))

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    result = resume.resume_benchmark(**arguments(output, max_retries=0))
    assert result["completed_conditions"] == 100
    assert publications[0] < publications[1]
    assert all(before <= after for before, after in zip(publications, publications[1:]))
    assert publications[-1] == set(resume.condition_plan(original))
    assert len(dispatched) == 97 and all(len(conditions) == 1 for conditions in dispatched)
    assert sum(map(len, dispatched)) == len(set().union(*dispatched)) == 97
    assert resume._remaining(policy, original) == (0, 0)
