import json
import os
from pathlib import Path
import threading

import pytest

from simready_bench import benchmark as workflow
from simready_bench import parallel, recovery, resume
from test_automatic_recovery import arguments
from test_parallel import manifest
from test_resume import portable, stopped_run


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
    wheel.write_bytes(b"original evaluator")
    publish, publications = committing_publisher()
    first = threading.Event()

    def execute(shard, **options):
        path = portable(
            options["output_dir"] / "attempts/reference",
            [
                (identity, condition, "completed")
                for identity, condition in resume.condition_plan(shard).items()
            ],
        )
        report = {"directory": str(path.parent), "results": str(path)}
        if options["output_dir"].name == "001":
            options["on_report"](report)
            first.set()
        else:
            assert first.wait(timeout=10)
            options["on_report"](report)

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
    assert len(publications[0]) == 4 and len(publications[1]) == 8


def test_resume_keeps_a_valid_report_when_publication_commits_then_raises(
    tmp_path, monkeypatch
):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    publish, publications = committing_publisher()
    monkeypatch.setattr(
        workflow, "publication_callback", lambda *args, **kwargs: publish
    )

    def execute(shard, **options):
        selected = list(resume.condition_plan(shard).items())[1:3]
        for index, (identity, condition) in enumerate(selected):
            path = portable(
                options["output_dir"] / ("attempts/00" + str(index)),
                [(identity, condition, "completed")],
            )
            report = {"directory": str(path.parent), "results": str(path)}
            options["on_report"](report)

    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    with pytest.raises(recovery.RecoveryLimitReached) as caught:
        resume.resume_benchmark(**arguments(output, max_retries=0))
    assert caught.value.result["completed_conditions"] == 5
    assert publications[0] < publications[1]
