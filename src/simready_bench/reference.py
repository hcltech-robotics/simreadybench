"""Read-only selected-profile validation, separate from scoring measurement support."""

from pathlib import Path
import hashlib
from simready_bench.engine.environment import collect_reference_dependencies

PROFILE = {"id": "Robotics-Prop", "version": "4.0.0"}


def check_reference(source, *, package_root, profile=None):
    """Inspect dependencies before composing USD; never author, repair or stamp it."""
    from simready_bench.engine.profiles import selection
    selected = selection(profile)
    source = Path(source).absolute()
    package = Path(package_root).absolute()
    if (
        source.resolve() != source
        or package.resolve() != package
        or not source.is_file()
        or not source.is_relative_to(package)
    ):
        raise ValueError("Regular source USD inside its package root required")
    manifest = collect_reference_dependencies(
        package=package.name,
        entrypoint=source.relative_to(package).as_posix(),
        package_root=package,
    )
    from simready_bench.engine.validator import validate_asset

    report = validate_asset(source, **({"profile": selected} if selected != PROFILE else {}))
    # An infrastructure error cannot be accepted as a successful profile check.
    accepted = (
        report.get("passed") is True
        and report.get("engine_dispatch_completed") is True
        and not report.get("infrastructure_errors")
    )
    result = {
        "schema": "simready-bench-reference-check-v1",
        "profile": selected,
        "accepted": accepted,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dependency_manifest": manifest,
        "conformance": report,
        "source_modified": False,
        "provider_calls": 0,
    }
    return result
