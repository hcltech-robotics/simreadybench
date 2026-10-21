import hashlib
import json
import os

import pytest

from simready_bench import recovery, resume
from test_automatic_recovery import arguments
from test_resume import stopped_run


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Linux allowance ownership")


def test_resume_reuses_retained_original_wheel_without_building_current_code(
    tmp_path, monkeypatch
):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    original = output / ".executor-package/simready_bench-original.whl"
    observed = []

    def execute(manifest, **options):
        observed.append(options["wheel"])
        raise RuntimeError("Verified original package")

    monkeypatch.setattr(
        "simready_bench.daytona.build_wheel",
        lambda *args: pytest.fail("Changed evaluator was rebuilt"),
    )
    monkeypatch.setattr("simready_bench.daytona.run_daytona", execute)
    with pytest.raises(RuntimeError, match="Verified original package"):
        resume.resume_benchmark(**arguments(output, max_retries=0))
    assert observed and all(path == original for path in observed)


@pytest.mark.parametrize(
    "mutation", ["retained", "explicit", "missing", "conflict", "missing_hash"]
)
def test_changed_or_missing_original_evaluator_is_rejected_before_dispatch(
    tmp_path, monkeypatch, mutation
):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    wheel = output / ".executor-package/simready_bench-original.whl"
    options = {}
    if mutation == "retained":
        wheel.write_bytes(b"changed evaluator code")
    elif mutation == "explicit":
        supplied = tmp_path / "changed.whl"
        supplied.write_bytes(b"changed evaluator code")
        options["wheel"] = supplied
    elif mutation == "missing":
        wheel.unlink()
    elif mutation == "conflict":
        (output / "parallel-plan-001.json").write_text(
            json.dumps({"wheel_sha256": "f" * 64})
        )
    else:
        path = output / "daytona-delivery.json"
        record = json.loads(path.read_text())
        record.pop("wheel_sha256")
        path.write_text(json.dumps(record))
    monkeypatch.setattr(
        "simready_bench.daytona.build_wheel",
        lambda *args: pytest.fail("Changed evaluator was rebuilt"),
    )
    monkeypatch.setattr(
        "simready_bench.daytona.run_daytona",
        lambda *args, **kw: pytest.fail("Unexpected dispatch"),
    )
    with pytest.raises(ValueError, match="evaluator"):
        resume.resume_benchmark(**arguments(output, max_retries=0, **options))


def test_collect_only_does_not_require_an_original_wheel(tmp_path, monkeypatch):
    output, _, _, _ = stopped_run(tmp_path, monkeypatch)
    (output / ".executor-package/simready_bench-original.whl").unlink()
    result = resume.resume_benchmark(**arguments(output, dispatch_pending=False))
    assert result["completed_conditions"] == 3


def test_parallel_retained_package_and_explicit_matching_package_are_accepted(tmp_path):
    original = tmp_path / ".parallel-package-original/original.whl"
    original.parent.mkdir()
    original.write_bytes(b"frozen parallel evaluator")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    (tmp_path / "parallel-plan-001.json").write_text(
        json.dumps({"wheel_sha256": digest})
    )
    assert recovery.frozen_wheel(tmp_path) == original
    supplied = tmp_path / "supplied.whl"
    supplied.write_bytes(original.read_bytes())
    assert recovery.frozen_wheel(tmp_path, supplied) == supplied


def test_unadmitted_bootstrap_reuses_its_first_retained_package(tmp_path):
    assert recovery.frozen_wheel(tmp_path) is None
    wheel = tmp_path / ".executor-package/original.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"first evaluator package")
    assert recovery.frozen_wheel(tmp_path) == wheel


def test_saved_parallel_identity_requires_its_retained_package_before_admission(
    tmp_path,
):
    (tmp_path / "parallel-plan-001.json").write_text(
        json.dumps({"wheel_sha256": "a" * 64})
    )
    with pytest.raises(ValueError, match="missing"):
        recovery.frozen_wheel(tmp_path)
