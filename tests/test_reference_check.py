import pytest
from simready_bench.reference import check_reference


def test_reference_acceptance_requires_real_completed_profile_and_no_infrastructure(
    tmp_path, monkeypatch
):
    package = tmp_path / "source"
    package.mkdir()
    source = package / "asset.usda"
    source.write_text("#usda 1.0")
    monkeypatch.setattr(
        "simready_bench.reference.collect_reference_dependencies",
        lambda **kwargs: {"closure_sha256": "f" * 64},
    )
    import simready_bench.engine.validator

    report = {
        "passed": True,
        "engine_dispatch_completed": True,
        "infrastructure_errors": [],
    }
    monkeypatch.setattr(
        simready_bench.engine.validator, "validate_asset", lambda source: dict(report)
    )
    assert check_reference(source, package_root=package)["accepted"] is True
    report["passed"] = False
    assert check_reference(source, package_root=package)["accepted"] is False
    report.update(passed=True, infrastructure_errors=[{"code": "runtime_missing"}])
    assert check_reference(source, package_root=package)["accepted"] is False
    assert source.read_text() == "#usda 1.0"


def test_profile_and_source_confinement_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="supports"):
        check_reference(
            tmp_path / "a.usda", package_root=tmp_path, profile={"id": "invented"}
        )
    with pytest.raises(ValueError, match="Regular source"):
        check_reference(tmp_path.parent / "a.usda", package_root=tmp_path)
