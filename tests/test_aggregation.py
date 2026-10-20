import hashlib
import json

from simready_bench import benchmark, orchestration, resume
from simready_bench.reporting import read_generated_report
from test_parallel import manifest
from test_resume import portable, write


def capture_report(attempt, condition, *, image_name="capture.png"):
    path = portable(attempt, [(*condition, "completed")])
    value = json.loads(path.read_text())
    for kind, name, raw in (
        ("images", image_name, b"retained capture"),
        ("traces", "trace.json", b"{}\n"),
    ):
        (attempt / "evidence" / name).write_bytes(raw)
        value["results"][0][kind] = [
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
        ]
    write(path, value)
    return {"directory": str(attempt), "results": str(path)}


def test_baseline_evidence_is_idempotent_across_twenty_index_restarts(tmp_path):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    output = tmp_path / "run"
    condition = next(iter(resume.condition_plan(selected).items()))
    attempt = output / "attempts/reference"
    report = capture_report(attempt, condition)
    first = benchmark._aggregate_reports(
        [(attempt, report)], output_dir=output, planned_conditions=4
    )
    original = read_generated_report(first["results"])
    files = {path.name: path.read_bytes() for path in (output / "evidence").iterdir()}

    for _ in range(20):
        index = orchestration.ReportIndex(output, selected)
        index.backfill([])
        result = index.aggregate(final=True)
        assert read_generated_report(result["results"]) == original
        assert {
            path.name: path.read_bytes() for path in (output / "evidence").iterdir()
        } == files


def test_child_evidence_names_are_bounded_and_stable(tmp_path):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    conditions = list(resume.condition_plan(selected).items())
    output = tmp_path / "run"
    reports = []
    for number, condition in enumerate(conditions[:2]):
        attempt = output / "attempts" / str(number)
        reports.append(
            (attempt, capture_report(attempt, condition, image_name="a" * 240 + ".png"))
        )
    first = benchmark._aggregate_reports(
        reports, output_dir=output, planned_conditions=4
    )
    original = read_generated_report(first["results"])
    names = [row["images"][0]["path"] for row in original["results"]]
    assert len(set(names)) == 2
    for name in names:
        assert len(name.encode("utf-8")) <= 255
        assert name.endswith(".png")
        assert (output / "evidence" / name).read_bytes() == b"retained capture"
    repeated = benchmark._aggregate_reports(
        reports, output_dir=output, planned_conditions=4
    )
    assert read_generated_report(repeated["results"]) == original
