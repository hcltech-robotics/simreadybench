"""Private CPU geometry worker: no rendering, simulation or inference."""

import hashlib, json, sys
from pathlib import Path
from simready_bench.fidelity import compare, digest, reference_profile, settings
from simready_bench.usd_geometry import observe


def run(request):
    cfg = settings(request["settings"])
    source = Path(request["reference"])
    if (
        not source.is_relative_to("/reference")
        or source.resolve() != source
        or not source.is_file()
    ):
        raise ValueError("Confined reference USD required")
    if hashlib.sha256(source.read_bytes()).hexdigest() != request["reference_sha256"]:
        raise ValueError("Frozen source entrypoint differs")
    reference = observe(source)
    profile = reference_profile(reference, cfg)
    identity = digest(profile)
    if request["operation"] == "reference":
        return {
            "status": "frozen",
            "profile": profile,
            "profile_sha256": identity,
            "source_sha256": request["reference_sha256"],
        }
    if identity != request["profile_sha256"]:
        raise ValueError("Pre-authoring source eligibility differs")
    candidate = None
    invalid = request.get("candidate_invalid", False)
    if request.get("candidate") and not invalid:
        path = Path(request["candidate"])
        if (
            not path.is_relative_to("/candidate")
            or path.resolve() != path
            or not path.is_file()
        ):
            raise ValueError("Confined frozen candidate required")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        from simready_bench.engine.package_checks import inspect_package

        safety = inspect_package(
            "/candidate", entrypoint=str(path.relative_to("/candidate"))
        )
        if safety.get("infrastructure_errors"):
            raise RuntimeError("Candidate dependency safety unavailable")
        if safety.get("package_ok") is not True:
            invalid = True
        if not invalid:
            try:
                candidate = observe(path)
            except ValueError:
                invalid = True
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ValueError("Candidate bytes changed")
    result = compare(reference, candidate, cfg, candidate_invalid=invalid)
    result.update(
        reference_profile_frozen_before_authoring=True,
        reference_sha256=request["reference_sha256"],
        candidate_invalid=invalid,
    )
    if request.get("candidate") and "safety" in locals():
        result["candidate_package_safety"] = {
            "passed": safety.get("package_ok") is True,
            "findings": safety.get("findings"),
            "composition_attempted": safety.get("composition_attempted"),
        }
    if hashlib.sha256(source.read_bytes()).hexdigest() != request["reference_sha256"]:
        raise ValueError("Source bytes changed")
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1])
    result = run(json.loads(path.read_text()))
    (path.parent / "receipt.json").write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )
