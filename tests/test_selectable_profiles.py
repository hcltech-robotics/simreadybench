"""Portable policy tests plus opt-in real NVIDIA worker acceptance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import pytest

from simready_bench.engine.profiles import inventories, installed_policy, selection
from simready_bench.engine.scoring import (
    build_wire, pinned_optional_profile, score_validation, validate_optional_score,
)
from simready_bench.engine.verifier import (
    digest, pinned_profile, scoring_policy, validate_policy, validate_wire,
    write_verifier_output,
)


def fixture_policy(*, optional_count=2):
    """A synthetic unit-test roster, never presented as an installed profile."""
    mandatory, optional = pinned_profile(), pinned_optional_profile()
    for inv in (mandatory, optional):
        inv.update(profile_id="Unit-Test-Only", profile_version="1.0.0")
    mandatory["mandatory_requirements"] = mandatory["mandatory_requirements"][:3]
    mandatory["mandatory_requirement_count"] = 3
    optional["optional_requirements"] = optional["optional_requirements"][:optional_count]
    optional["optional_requirement_count"] = optional_count
    optional["mandatory_inventory_sha256"] = digest(mandatory)
    return scoring_policy(inventory=mandatory, optional_inventory=optional)


def fixture_report(policy):
    mandatory, optional = inventories(policy)
    inventory = deepcopy(mandatory)
    inventory.update({k: v for k, v in optional.items() if k.startswith("optional_")})
    def rows(roster):
        return [{**row, "rule_enabled": True, "rule_invoked": True,
                 "method_calls": {"CheckStage": 1}, "disposition": "no_failure_reported"}
                for row in roster]
    return {
        "inventory": inventory, "engine_dispatch_completed": True, "passed": True,
        "infrastructure_errors": [], "issues": [], "soft_issues": [],
        "rule_execution": {
            "completed": True,
            "sdk_source_sha256": {**mandatory["SDK_dispatch_source_sha256"],
                                  **mandatory["rule_module_sha256"], **optional["rule_module_sha256"]},
            "requirements": rows(mandatory["mandatory_requirements"]),
            "optional_requirements": rows(optional["optional_requirements"]),
        },
    }


def test_frozen_default_policy_and_inventories_are_unchanged():
    expected = json.loads(Path("src/simready_bench/resources/benchmarks/nvidia-25.json").read_text())["scoring"]
    assert scoring_policy() == expected
    assert len(scoring_policy()["applicable_requirement_ids"]) == 46
    assert inventories()[1]["optional_requirement_count"] == 51
    assert selection({"id": "NVIDIA-Robotics-Prop-4.0.0"}) == selection()


def test_profile_specific_rosters_scores_and_portable_metadata():
    from simready_bench.reporting import compact_report, render_html
    from simready_bench.assets import profile_contract
    policy = fixture_policy()
    assert validate_policy(policy) == policy
    raw = fixture_report(policy)
    score = score_validation(raw, policy)
    assert score["score_percent"] == 100 and score["applicable_count"] == 3
    assert score["optional_score"]["score_percent"] == 100
    assert score["scoring_policy"] == policy
    raw["passed"] = False
    raw["rule_execution"]["requirements"][0]["disposition"] = "failed"
    assert score_validation(raw, policy)["score_percent"] == pytest.approx(95 * 2 / 3)
    # The same evidence cannot acquire the frozen default's denominator or identity.
    assert score_validation(raw, scoring_policy())["score_percent"] is None
    assert "3 mandatory" in profile_contract(policy)
    assert "2 optional" in profile_contract(policy)
    portable = compact_report({"schema": "simready-bench-portable-report-v1", "results": [{"simready": score}]})
    assert portable["results"][0]["simready"]["scoring_policy"] == policy
    assert "SimReady Unit-Test-Only optional" in render_html(portable)


@pytest.mark.parametrize("mutation", ["count", "duplicate", "rule", "hash", "profile", "default"])
def test_invalid_portable_inventory_cannot_be_selected(mutation):
    policy = fixture_policy()
    inv = policy["profile_inventory"]
    if mutation == "count":
        inv["mandatory_requirement_count"] += 1
    elif mutation == "duplicate":
        inv["mandatory_requirements"][1] = inv["mandatory_requirements"][0]
    elif mutation == "rule":
        inv["mandatory_requirements"][0]["rule"] = None
    elif mutation == "hash":
        inv["rule_module_sha256"]["fake"] = "bad"
    elif mutation == "profile":
        policy["profile"]["version"] = "2.0.0"
    else:
        inv["profile_id"], inv["profile_version"] = "Robotics-Prop", "4.0.0"
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_optional_empty_scope_is_not_applicable_and_unknown_stays_unknown():
    policy = fixture_policy(optional_count=0)
    score = score_validation(fixture_report(policy), policy)
    assert score["score_percent"] == 100
    optional = score["optional_score"]
    assert optional["status"] == "not_applicable" and optional["score_percent"] is None
    assert optional["applicable_count"] == 0
    validate_optional_score(optional, policy=policy)
    raw = fixture_report(policy)
    raw["rule_execution"]["requirements"].pop()
    score = score_validation(raw, policy)
    assert score["score_percent"] is None and score["applicable_count"] == 3


def test_settings_must_match_the_explicit_scoring_profile(setup):
    from simready_bench.config import validate_manifest
    root, manifest, operator = setup
    manifest["scoring"] = fixture_policy()
    with pytest.raises(ValueError, match="match.*scoring policy"):
        validate_manifest(manifest, base_dir=root, policy=operator)
    manifest["defaults"]["profile"] = manifest["scoring"]["profile"]
    assert validate_manifest(manifest, base_dir=root, policy=operator)["scoring"] == manifest["scoring"]


def test_cli_freezes_policy_and_forwards_benchmark_selection(tmp_path, monkeypatch):
    from simready_bench.cli import main
    import simready_bench.engine.profiles as profiles
    import simready_bench.benchmark as benchmark_module
    policy = fixture_policy()
    def resolve(profile):
        assert profile == policy["profile"]
        return policy
    monkeypatch.setattr(profiles, "installed_policy", resolve)
    output = tmp_path / "policy.json"
    assert main(["profiles", "--id", "Unit-Test-Only", "--version", "1.0.0",
                 "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == policy
    observed = {}
    def run(reference, **kwargs):
        observed.update(kwargs)
        return {"results": []}
    monkeypatch.setattr(benchmark_module, "benchmark", run)
    assert main(["benchmark", "--model", "unit-test", "--validation-policy", str(output),
                 "--output-dir", str(tmp_path / "run")]) == 0
    assert observed["validation_policy"] == output


def test_profile_cannot_change_on_resume_or_enter_the_frozen_leaderboard(tmp_path):
    from simready_bench.benchmark import benchmark
    path = tmp_path / "selected.json"
    path.write_text(json.dumps(fixture_policy()))
    run = tmp_path / "run"
    run.mkdir()
    (run / "experiment.json").write_text(json.dumps({"scoring": scoring_policy()}))
    with pytest.raises(ValueError, match="preserve the saved validation profile"):
        benchmark(model="unit-test", output_dir=run, resume=True, validation_policy=path)
    with pytest.raises(ValueError, match="public leaderboard"):
        benchmark(model="unit-test", output_dir=tmp_path / "new", validation_policy=path,
                  hf_repo="example/dataset")
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("custom", [False, True])
def test_verifier_image_modules_validate_with_only_stdlib(tmp_path, custom):
    """Exercise the actual copied image client without coordinator dependencies."""
    import importlib.util
    import shutil
    import subprocess
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("container_build", root / "containers/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    context = build.prepare_context("verifier", tmp_path / "context")
    package = tmp_path / "runtime/simready_bench/engine"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    for name in ("verifier.py", "profiles.py", "scoring.py",
                 "nvidia_profile.json", "nvidia_optional_profile.json"):
        shutil.copyfile(context / name, package / name)
    policy = fixture_policy() if custom else scoring_policy()
    score = score_validation(fixture_report(policy), policy)
    wire = {
        "protocol": "usd-reconstruction/v1", "schema_version": "simready-verifier/v1",
        "episode_id": "unit", "snapshot_id": "unit", "snapshot_sha256": "a" * 64,
        "outcome": "complete", "reward": 100, "simready": score,
        "acceptance": {"accepted": True}, "readiness": {}, "errors": [],
    }
    script = """
import json, sys
sys.path.insert(0, sys.argv[1])
from simready_bench.engine.verifier import validate_wire
validate_wire(json.load(sys.stdin))
assert not any(name.startswith(('pxr', 'simready.validate', 'harbor')) for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script,
                             str(tmp_path / "runtime")], input=json.dumps(wire),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr


@pytest.mark.licensed
@pytest.mark.parametrize("profile,mandatory,optional", [
    ({"id": "Robotics-Prop", "version": "4.0.0"}, 46, 51),
    ({"id": "Robotics-Prop", "version": "3.0.0"}, 34, 30),
    ({"id": "Open-Taxonomy-COCO", "version": "0.1.0"}, 13, 0),
])
def test_real_selected_profile_worker_evaluation_feedback_and_wire(tmp_path, profile, mandatory, optional):
    pytest.importorskip("simready.validate")
    from helpers import candidate_fixture
    from pxr import Usd, UsdGeom
    from simready_bench.engine.evaluation import CandidateObserver, Evaluator
    from simready_bench.reporting import compact_report

    policy = installed_policy(profile)
    actor, controller, cfg, _, _ = candidate_fixture(tmp_path)
    source = actor / "output/asset.usd"
    source.unlink()
    stage = Usd.Stage.CreateNew(str(source))
    prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.Cube.Define(stage, "/Asset/Body")
    stage.GetRootLayer().Save()
    cfg["validator_scoring"] = policy
    snap = controller.freeze_candidate("ep")
    observer = CandidateObserver(
        validator_python=sys.executable, source_root=Path(__file__).resolve().parents[1] / "src",
        state_dir=tmp_path / "worker", expected_hostname=socket.gethostname(),
        execution_enabled=True, scoring_policy=policy,
    )
    evaluator = Evaluator(tmp_path / "evaluation", observer=observer)
    report = evaluator.evaluate(snap, cfg, "ep", time.time() + 90, final=False)
    score = report["simready"]
    assert score["score_percent"] is not None, score.get("unknown_reason")
    assert score["applicable_count"] == mandatory
    assert len(score["optional_score"]["checks"]) == optional
    assert report["feedback"]["simready"] == score
    assert report["nvidia_validator_report"]["profile_id"] == profile["id"]
    assert report["nvidia_validator_report"]["profile_version"] == profile["version"]
    assert report["validator_implementation_identity"]["profile_inventory_sha256"] == policy["profile_inventory_sha256"]
    cached = evaluator.evaluate(snap, cfg, "ep", time.time() + 90, final=False)
    assert cached["observation_cache_hit"] is True
    assert cached["simready"] == score
    report = evaluator.evaluate(snap, cfg, "ep", time.time() + 90, final=True)
    assert report["simready"] == score
    assert evaluator.evaluate(snap, cfg, "ep", time.time() + 90) == report
    wire = build_wire(report, snapshot=snap)
    validate_wire(wire)
    assert write_verifier_output(wire, tmp_path / "verifier") == 0
    assert json.loads((tmp_path / "verifier/reward.json").read_text())["simready_reward"] == score["score_percent"]
    portable = compact_report({"schema": "simready-bench-portable-report-v1", "results": [{"simready": score}]})
    assert portable["results"][0]["simready"]["official_conformance"]["profile"] == profile
    tampered = deepcopy(wire)
    tampered["reward"] = 100
    with pytest.raises(ValueError):
        validate_wire(tampered)


@pytest.mark.licensed
def test_real_reference_admission_uses_selected_profile_without_modification(tmp_path):
    pytest.importorskip("simready.validate")
    from pxr import Usd, UsdGeom, UsdSemantics
    from simready_bench.reference import check_reference
    package = tmp_path / "reference"
    (package / "usd").mkdir(parents=True)
    source = package / "usd/reference.usda"
    stage = Usd.Stage.CreateNew(str(source))
    prim = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
    stage.SetDefaultPrim(prim)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.Cube.Define(stage, "/Asset/Body")
    stage.GetRootLayer().customLayerData = {
        "SimReady_Metadata": {"fixture": "profile switching"}, "asset_name": "reference",
        "asset_type": "prop", "source_file": "fixture", "usd_date_generated": "2026-11-06",
    }
    UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr(["chair"])
    stage.GetRootLayer().Save()
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    profile = {"id": "Open-Taxonomy-COCO", "version": "0.1.0"}
    receipt = check_reference(source, package_root=package, profile=profile)
    assert receipt["accepted"] is True
    assert receipt["profile"] == profile
    assert receipt["source_modified"] is False and receipt["provider_calls"] == 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    # Exercise the actual reference-admission subprocess and installed-policy seal.
    from simready_bench.assets import _metadata
    metadata = _metadata(source, package, {
        "validator_python": sys.executable,
        "source_root": str(Path(__file__).resolve().parents[1] / "src"),
    }, tmp_path / "source-conformance.json", profile=profile,
        scoring_policy=installed_policy(profile))
    assert metadata["accepted"] is True and metadata["profile"] == profile
    assert metadata["visible_bounds_m"] == [[-1, -1, -1], [1, 1, 1]]
    from simready_bench.assets import validate_reference_binding
    from simready_bench.policy import installed_profile
    shared, _ = installed_profile()
    cfg = {
        "validator_scoring": installed_policy(profile),
        "shared_environment_profile": shared,
        "validator_reference": {
            "asset_id": "reference", "package": package.name,
            "dependency_manifest": metadata["dependency_manifest"],
            "conformance_receipt": {
                "path": str(tmp_path / "source-conformance.json"),
                "sha256": hashlib.sha256((tmp_path / "source-conformance.json").read_bytes()).hexdigest(),
            },
        },
    }
    assert validate_reference_binding(cfg, allowed_reference_root=tmp_path)["status"] == "verified"
    stale = deepcopy(cfg)
    stale["validator_scoring"]["optional_inventory"]["optional_features"].append({
        "id": "not-installed", "version": "1.0.0",
    })
    with pytest.raises(ValueError, match="inventory binding"):
        validate_reference_binding(stale, allowed_reference_root=tmp_path)
    # The same reference lacks Robotics-Prop's physics, mesh and provenance requirements.
    assert check_reference(source, package_root=package)["accepted"] is False
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    with pytest.raises(ValueError, match="not found"):
        installed_policy({"id": "Invented-Profile", "version": "1.0.0"})
    with pytest.raises(ValueError, match="not found"):
        installed_policy({"id": "Robotics-Prop", "version": "99.0.0"})
