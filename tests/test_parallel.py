import copy
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from simready_bench import benchmark as workflow
from simready_bench import parallel, resume
from simready_bench.engine.inference_gateway import ProjectBudget
from simready_bench.local import ParentAuthorityLedger
from simready_bench.runtime_images import DEFAULT_IMAGES
from test_resume import portable, stopped_run, write


def manifest():
    source, _, _ = workflow.reference_manifest("NV25")
    model = workflow.select_model("gpt-6.1-sol", provider="openai")[0]
    return workflow.compose_experiment(
        source, model=model, repetitions=4, maximum_spend_usd=250, authoring_turns=32
    )


def test_partition_preserves_conditions_seeds_and_scientific_settings():
    original = manifest()
    original_copy = copy.deepcopy(original)
    first = original["assets"][0]["id"]
    original["reconstruction_selection"] = {
        asset["id"]: ([3] if asset["id"] == first else [0, 1, 2, 3])
        for asset in original["assets"]
    }
    expected = resume.condition_plan(original)
    shards = parallel.partition_manifest(original, 8)
    joined = {}
    for shard in shards:
        conditions = resume.condition_plan(shard)
        assert not joined.keys() & conditions.keys()
        joined.update(conditions)
        scientific = copy.deepcopy(shard)
        scientific.pop("reconstruction_selection")
        assert scientific == original_copy
    assert joined == expected
    assert sum(len(resume.condition_plan(shard)) * 32 for shard in shards) == 97 * 32
    assert sum(len(resume.condition_plan(shard)) * 2.5 for shard in shards) == 97 * 2.5


def test_partition_can_parallelise_repetitions_of_one_asset():
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    assert len(parallel.partition_manifest(selected, 16)) == 4


@pytest.mark.parametrize("workers", [0, 17, True, 1.5, "4", None])
def test_invalid_worker_limits(workers):
    with pytest.raises(ValueError, match="integer between 1 and 16"):
        parallel.partition_manifest(manifest(), workers)


def test_coordinator_lock_excludes_other_threads_and_is_reentrant(tmp_path):
    def compete():
        with pytest.raises(ValueError, match="Another benchmark coordinator"):
            with parallel.coordinator_lock(tmp_path):
                pytest.fail("Concurrent ownership was granted")

    with parallel.coordinator_lock(tmp_path):
        with parallel.coordinator_lock(tmp_path):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(compete).result(timeout=10)
    with parallel.coordinator_lock(tmp_path):
        pass


def test_parallel_aggregate_is_not_counted_as_an_independent_execution(tmp_path):
    portable(tmp_path / "run", [])
    root = tmp_path / "run"
    (root / "executions").mkdir()
    assert len(resume._reports(root)) == 1
    write(
        root / "parallel-plan-001.json", {"schema": "simready-bench-parallel-plan-v1"}
    )
    assert resume._reports(root) == []


def test_new_parallel_run_publishes_cumulative_reports_from_one_writer(
    tmp_path, monkeypatch
):
    selected = manifest()
    selected["assets"] = selected["assets"][:2]
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    wheel = tmp_path / "frozen.whl"
    wheel.write_bytes(b"frozen evaluator")
    barrier = threading.Barrier(2)
    counts = []

    def execute(shard, **options):
        path = portable(
            options["output_dir"] / "attempts/reference",
            [
                (identity, condition, "completed")
                for identity, condition in resume.condition_plan(shard).items()
            ],
        )
        barrier.wait(timeout=10)
        options["on_report"](
            {
                "directory": str(path.parent),
                "results": str(path),
                "html": str(path.parent / "report.html"),
                "scores": str(path.parent / "scores.csv"),
            }
        )

    def publish(report, *, final=False):
        value = json.loads(__import__("pathlib").Path(report["results"]).read_text())
        assert value["planned_conditions"] == 8
        assert len(value["results"]) == len(
            {row["condition_id"] for row in value["results"]}
        )
        counts.append((value["completed_conditions"], final))

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    result = parallel.run_parallel_daytona(
        selected,
        policy={"roots": roots},
        output_dir=tmp_path,
        workers=2,
        wheel=wheel,
        on_report=publish,
    )
    assert result["completed_conditions"] == 8
    assert [count for count, _ in counts] == sorted(count for count, _ in counts)
    assert counts[-1] == (8, True)


linux_only = pytest.mark.skipif(
    os.name == "nt", reason="The allowance ledger uses Linux ownership"
)


@linux_only
def test_parallel_resume_has_one_publisher_and_original_frozen_allowance(
    tmp_path, monkeypatch
):
    output, original, policy, ledger = stopped_run(tmp_path, monkeypatch)
    wheel = tmp_path / "frozen.whl"
    wheel.write_bytes(b"frozen evaluator")
    barrier = threading.Barrier(8)
    seen, publications = [], []
    active, maximum = 0, 0
    state_lock = threading.Lock()

    def publish(report, *, final=False):
        nonlocal active, maximum
        with state_lock:
            active += 1
            maximum = max(maximum, active)
        try:
            rows = json.loads(
                __import__("pathlib").Path(report["results"]).read_text()
            )["results"]
            identities = {row["condition_id"] for row in rows}
            assert len(identities) == len(rows)
            first = set(list(resume.condition_plan(original))[:3])
            assert first <= identities
            publications.append((identities, final))
            time.sleep(0.01)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(
        workflow, "publication_callback", lambda *args, **kwargs: publish
    )

    def execute(selected, **options):
        conditions = resume.condition_plan(selected)
        batch = options["output_dir"]
        assert (output / "parallel-plan-001.json").is_file()
        plan = json.loads((output / "parallel-plan-001.json").read_text())
        assert len(plan["shards"]) == 8
        assert options["wheel"] == wheel
        assert options["runtime_images"] == DEFAULT_IMAGES
        assert options["cpu_snapshot"] == "simready-bench-cpu"
        assert options["policy"]["providers"] == policy["providers"]
        assert options["policy"]["runtime"] == policy["runtime"]
        parent = ProjectBudget(**policy["providers"]["selected"]["project_budget"])
        ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
            parent,
            allocation_id="shard-" + batch.name,
            requests=32 * len(conditions),
            maximum_spend_usd=2.5 * len(conditions),
            work_root=batch,
        )
        path = portable(
            batch / "attempts/reference",
            [
                (identity, condition, "completed")
                for identity, condition in conditions.items()
            ],
        )
        with state_lock:
            seen.append(set(conditions))
        barrier.wait(timeout=10)
        options["on_report"](
            {
                "directory": str(path.parent),
                "results": str(path),
                "html": str(path.parent / "report.html"),
                "scores": str(path.parent / "scores.csv"),
            },
            final=True,
        )
        return {"status": "completed"}

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    result = resume.resume_benchmark(
        "NV25",
        model="gpt-6.1-sol",
        harness="pi",
        deploy="daytona",
        output_dir=output,
        workers=8,
        wheel=wheel,
    )
    assert result["completed_conditions"] == len(result["results"]) == 100
    assert len(seen) == 8 and sum(map(len, seen)) == 97
    assert maximum == 1
    assert all(a[0] <= b[0] for a, b in zip(publications, publications[1:]))
    assert len(publications[-1][0]) == 100 and publications[-1][1] is True
    assert resume._remaining(policy, original) == (0, 0)
    with ledger._db() as db:
        assert db.execute("SELECT id FROM parents").fetchall() == [
            ("original-project",)
        ]


@linux_only
def test_worker_failure_keeps_other_workers_and_their_reports(tmp_path, monkeypatch):
    output, original, _, _ = stopped_run(tmp_path, monkeypatch)
    wheel = tmp_path / "frozen.whl"
    wheel.write_bytes(b"frozen evaluator")
    barrier = threading.Barrier(4)
    seen = []
    failed = set()

    def execute(selected, **options):
        conditions = resume.condition_plan(selected)
        batch = options["output_dir"]
        seen.append(batch.name)
        barrier.wait(timeout=10)
        if batch.name == "001":
            failed.update(conditions)
            raise RuntimeError("One worker is unavailable")
        portable(
            batch / "attempts/reference",
            [
                (identity, condition, "completed")
                for identity, condition in conditions.items()
            ],
        )

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    with pytest.raises(RuntimeError, match="One worker is unavailable") as caught:
        resume.resume_benchmark(
            "NV25",
            model="gpt-6.1-sol",
            harness="pi",
            deploy="daytona",
            output_dir=output,
            workers=4,
            wheel=wheel,
            max_retries=0,
        )
    result = caught.value.result
    assert set(seen) == {"001", "002", "003", "004"}
    assert result["completed_conditions"] == 100 - len(failed)
    assert len(result["results"]) == len(
        {row["condition_id"] for row in result["results"]}
    )
    assert not failed & {
        row["condition_id"]
        for row in result["results"]
        if row.get("status") == "completed"
    }


@linux_only
def test_unknown_worker_checks_other_workers_before_forbidding_dispatch(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import DaytonaObservationError

    output, original, _, _ = stopped_run(tmp_path, monkeypatch)
    for number, shard in zip(("001", "002"), parallel.partition_manifest(original, 2)):
        batch = output / "executions" / number
        write(batch / "experiment.json", shard)
        write(
            batch / "daytona-delivery.json",
            {
                "snapshot": "simready-bench-cpu",
                "runtime_images": DEFAULT_IMAGES,
                "collection": None,
                "cleanup_confirmed": False,
            },
        )
    barrier = threading.Barrier(2)
    observed = []

    def reconnect(selected, **options):
        observed.append(options["output_dir"].name)
        barrier.wait(timeout=10)
        if options["output_dir"].name == "001":
            raise DaytonaObservationError("Worker state is unknown")

    monkeypatch.setattr("simready_bench.daytona.reconnect_daytona", reconnect)
    monkeypatch.setattr(
        "simready_bench.daytona.run_daytona",
        lambda *args, **kwargs: pytest.fail("A replacement worker was started"),
    )
    with pytest.raises(DaytonaObservationError):
        resume.resume_benchmark(
            "NV25",
            model="gpt-6.1-sol",
            harness="pi",
            deploy="daytona",
            output_dir=output,
            workers=8,
        )
    assert set(observed) == {"001", "002"}
    assert sorted(path.name for path in (output / "executions").iterdir()) == [
        "001",
        "002",
    ]


@linux_only
def test_parallel_resume_never_reaggregates_itself_or_drops_prior_artefacts(
    tmp_path, monkeypatch
):
    from simready_bench.publishing import _signature
    from simready_bench.reporting import read_generated_report

    output, original, policy, _ = stopped_run(tmp_path, monkeypatch)
    (output / "attempts").rename(tmp_path / "initial-report-fixture")
    planned = resume.condition_plan(original)
    prior = dict(list(planned.items())[:8])
    parent = ProjectBudget(**policy["providers"]["selected"]["project_budget"])
    child = replace(
        parent,
        authorization_id="cloud-outer-allocation",
        upstream_origin="http://127.0.0.1:1234",
    )
    remote = ParentAuthorityLedger(output / "daytona-executor/authority")
    originals = {}
    for index, asset in enumerate(original["assets"][:2]):
        selected = [
            (identity, condition, "completed")
            for identity, condition in prior.items()
            if condition["asset_id"] == asset["id"]
        ]
        path = portable(output / "attempts" / format(index, "03d"), selected)
        document = json.loads(path.read_text())
        for name, raw in (
            ("capture.png", b"retained capture"),
            ("trace.json", b"{}\n"),
        ):
            (path.parent / "evidence" / name).write_bytes(raw)
        for row in document["results"]:
            row["images"] = [
                {
                    "path": "capture.png",
                    "view": "isometric",
                    "role": "candidate",
                    "sha256": hashlib.sha256(b"retained capture").hexdigest(),
                }
            ]
            row["traces"] = [
                {"path": "trace.json", "sha256": hashlib.sha256(b"{}\n").hexdigest()}
            ]
        write(path, document)
        for row in read_generated_report(path)["results"]:
            originals[row["condition_id"]] = _signature([row])
        if index:
            for identity, _, _ in selected:
                remote.reserve(
                    child,
                    allocation_id=resume._digest(
                        {
                            "work_root": "/dev/shm/bench/a1",
                            "configuration": "c" * 64,
                            "condition": identity,
                        }
                    ),
                    requests=32,
                    maximum_spend_usd=2.5,
                    work_root="/dev/shm/bench/a1",
                )

    aggregate = workflow._aggregate_reports
    stable_paths, counts = {}, []

    def checked_aggregate(reports, **options):
        assert options["planned_conditions"] == 100
        assert all(root != output for root, _ in reports)
        result = aggregate(reports, **options)
        value = read_generated_report(result["results"])
        counts.append(value["completed_conditions"])
        rows = {row["condition_id"]: row for row in value["results"]}
        assert prior.keys() <= rows.keys()
        for identity in prior:
            row = rows[identity]
            assert _signature([row]) == originals[identity]
            paths = [item["path"] for item in row["images"] + row["traces"]]
            if identity in stable_paths:
                assert paths == stable_paths[identity]
            stable_paths[identity] = paths
            for item in row["images"] + row["traces"]:
                assert (
                    hashlib.sha256(
                        (output / "evidence" / item["path"]).read_bytes()
                    ).hexdigest()
                    == item["sha256"]
                )
        return result

    monkeypatch.setattr(workflow, "_aggregate_reports", checked_aggregate)
    barrier = threading.Barrier(4)

    def execute(selected, **options):
        path = portable(
            options["output_dir"] / "attempts/reference",
            [
                (identity, condition, "completed")
                for identity, condition in resume.condition_plan(selected).items()
            ],
        )
        barrier.wait(timeout=10)
        report = {
            "directory": str(path.parent),
            "results": str(path),
            "html": str(path.parent / "report.html"),
            "scores": str(path.parent / "scores.csv"),
        }
        options["on_report"](report, final=True)

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    wheel = tmp_path / "frozen.whl"
    wheel.write_bytes(b"frozen evaluator")
    result = resume.resume_benchmark(
        "NV25",
        model="gpt-6.1-sol",
        harness="pi",
        deploy="daytona",
        output_dir=output,
        workers=4,
        wheel=wheel,
    )
    assert result["completed_conditions"] == 100
    assert counts[0] == 8 and counts[-1] == 100
    assert counts == sorted(counts)
