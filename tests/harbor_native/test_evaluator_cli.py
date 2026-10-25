"""Real installed evaluator CLI, immutable candidate boundary and frozen fixtures."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest
from PIL import Image

from simready_eval import cli, correspondence, render
from simready_eval.usd import observe


def require_validator():
    try:
        installed = importlib.util.find_spec("simready.validate")
    except ModuleNotFoundError:
        installed = None
    if installed is None:
        pytest.skip("Pinned public NVIDIA validator is unavailable in this interpreter")


def test_render_worker_can_write_measured_large_scene_without_raising_other_limits(tmp_path):
    # The real heat-gun scene is 335 MiB. Sparse files verify the actual kernel
    # file limit without another expensive mesh conversion or large disk write.
    script = "import sys; f=open(sys.argv[1], 'wb'); f.truncate(351193703); f.close()"
    ordinary = cli._run_process([sys.executable, "-c", script, str(tmp_path / "ordinary")],
                                tmp_path / "ordinary.log", timeout=5)
    rendered = cli._run_process([sys.executable, "-c", script, str(tmp_path / "rendered")],
                               tmp_path / "rendered.log", timeout=5,
                               file_limit_bytes=cli.RENDER_FILE_LIMIT_BYTES)
    assert ordinary["status"] == "unknown"
    assert rendered["status"] == "complete"
    assert (tmp_path / "rendered").stat().st_size == 351193703


def test_worker_file_budgets_cover_scene_and_trusted_archive_expansion(tmp_path, monkeypatch):
    limits = []
    def process(command, log, **kwargs):
        limits.append((kwargs["file_limit_bytes"], kwargs["address_space_bytes"]))
        Path(command[-1]).write_text('{}')
        return {"status": "complete"}
    monkeypatch.setattr(cli, "_run_process", process)
    for mode in ("formal", "reference", "compare", "render"):
        cli._run_worker(mode, {}, tmp_path / mode, time.monotonic() + 10)
    assert limits == [(cli.MAX_BYTES, 8 * 1024**3)] + [
        (cli.REFERENCE_FILE_LIMIT_BYTES, 8 * 1024**3)] * 2 + [
        (cli.RENDER_FILE_LIMIT_BYTES, cli.RENDER_ADDRESS_SPACE_BYTES)]


def test_trusted_reference_worker_can_expand_a_member_above_128_mib(tmp_path):
    size = 128 * 1024**2 + 1
    script = 'import sys; f=open(sys.argv[1], "wb"); f.truncate(int(sys.argv[2])); f.close()'
    path = tmp_path / 'expanded'
    result = cli._run_process([sys.executable, '-c', script, str(path), str(size)],
        tmp_path / 'archive.log', timeout=5, file_limit_bytes=cli.REFERENCE_FILE_LIMIT_BYTES)
    assert result['status'] == 'complete'
    assert path.stat().st_size == size


def test_renderer_address_space_allows_measured_large_mesh_within_container_envelope(tmp_path):
    log = tmp_path / 'limits.log'
    result = cli._run_process([sys.executable, '-c',
        'import json,resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_AS)))'],
        log, timeout=5, address_space_bytes=cli.RENDER_ADDRESS_SPACE_BYTES)
    assert result['status'] == 'complete'
    assert json.loads(log.read_text()) == [12 * 1024**3] * 2
    assert cli.RENDER_ADDRESS_SPACE_BYTES < 16 * 1024**3


def camera_manifest():
    return {"schema": "simready-cameras-v1", "resolution": [32, 32],
            "observed_camera_id": "front", "views": [
        {"id": name, "projection": "orthographic", "camera_to_world": matrix,
         "orthographic_scale": 4, "clip_start": .01, "clip_end": 100}
        for name, matrix in [
            ("front", [[1,0,0,0],[0,1,0,0],[0,0,1,6],[0,0,0,1]]),
            ("back", [[-1,0,0,0],[0,1,0,0],[0,0,-1,-6],[0,0,0,1]]),
            ("right", [[0,0,1,6],[1,0,0,0],[0,1,0,0],[0,0,0,1]]),
            ("left", [[0,0,-1,-6],[-1,0,0,0],[0,1,0,0],[0,0,0,1]])]]}


def cube(path, size=2, color=(.6,.12,.05)):
    from pxr import Sdf, Usd, UsdGeom, UsdShade
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    body = UsdGeom.Cube.Define(stage, "/Body")
    stage.SetDefaultPrim(body.GetPrim())
    body.CreateSizeAttr(size)
    material = UsdShade.Material.Define(stage, "/Material")
    shader = UsdShade.Shader.Define(stage, "/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


@pytest.fixture(scope="module")
def frozen_fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli-frozen-fixtures")
    (root / "reference").mkdir()
    (root / "comparison").mkdir()
    cube(root / "reference/asset.usd")
    cameras = camera_manifest()
    (root / "comparison/cameras.json").write_text(json.dumps(cameras))
    if not shutil.which("blender"):
        pytest.skip("Public Blender is required for genuine CLI integration fixtures")
    receipt = render.render_asset(root / "reference/asset.usd", cameras, root / "reference-render")
    assert receipt["status"] == "measured", receipt
    for row in receipt["images"]:
        shutil.copyfile(row["file"], root / "comparison" / (row["camera_id"] + ".png"))
        shutil.copyfile(row["mask_file"], root / "comparison" / (row["camera_id"] + "-mask.png"))
    config = {"schema": "simready-evaluator-config-v1", "task_id": "fixture-cube",
              "entrypoint": "asset.usd", "reference": "reference/asset.usd",
              "cameras": "comparison/cameras.json", "eligibility": "reference/eligibility.json",
              "fidelity_settings": {"enabled": True, "samples": 128}, "final_seconds": 600}
    profile = correspondence.reference_profile(observe(root / config["reference"]), config["fidelity_settings"])
    files = [{"path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size,
              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for directory in (root / "reference", root / "comparison")
             for p in sorted(directory.rglob("*")) if p.is_file()]
    eligibility = {"schema": "simready-reference-eligibility-v1", "files": files,
                   "profile": profile, "profile_sha256": correspondence.digest(profile),
                   "render_recipe_sha256": receipt["recipe_sha256"],
                   "render_camera_sha256": receipt["camera_sha256"],
                   "render_bridge_code_sha256": receipt["bridge_code_sha256"],
                   "renderer_binary_sha256": receipt["renderer_binary_sha256"]}
    (root / config["eligibility"]).write_text(json.dumps(eligibility))
    return root, config


def submission(tmp_path, size=2, color=(.6,.12,.05)):
    root = tmp_path / "submission"
    root.mkdir()
    cube(root / "asset.usd", size=size, color=color)
    return root


def test_missing_submission_is_proven_invalid_with_three_known_zeros(tmp_path, frozen_fixtures):
    fixtures, config = frozen_fixtures
    root = tmp_path / "missing"
    result = cli.evaluate_submission(root, fixtures, config, tmp_path / "out")
    assert result["candidate_invalid"] is True
    assert all(result["scores"][name]["score_percent"] == 0 for name in ("formal", "visual", "kinematic"))
    reward = json.loads((tmp_path / "out/reward.json").read_text())
    assert reward["candidate_invalid"] == reward["evaluation_complete"] == 1


@pytest.mark.parametrize("kind", ["bytes", "profile", "unsafe_path", "missing_mask", "undeclared_dependency"])
def test_bad_fixture_evidence_is_unknown_before_candidate_work(tmp_path, frozen_fixtures, monkeypatch, kind):
    fixtures, config = frozen_fixtures
    local = tmp_path / "fixtures"
    shutil.copytree(fixtures, local)
    ep = local / config["eligibility"]
    eligibility = json.loads(ep.read_text())
    if kind == "bytes":
        (local / "reference/asset.usd").write_text("tampered")
    elif kind == "profile":
        eligibility["profile"]["shape_triangle_count"] += 1
        eligibility["profile_sha256"] = correspondence.digest(eligibility["profile"])
    elif kind == "unsafe_path":
        eligibility["files"][0]["path"] = "../escape"
    elif kind == "missing_mask":
        eligibility["files"] = [row for row in eligibility["files"] if row["path"] != "comparison/front-mask.png"]
    else:
        (local / "reference/unrecorded.usda").write_text('#usda 1.0\ndef Xform "Extra" {}\n')
    ep.write_text(json.dumps(eligibility))
    monkeypatch.setattr(cli, "freeze_candidate", lambda *args, **kwargs: pytest.fail("Candidate work before fixture eligibility"))
    result = cli.evaluate_submission(tmp_path / "submission", local, config, tmp_path / "out")
    assert result["candidate_invalid"] is False
    assert all(result["scores"][name]["score_percent"] is None for name in ("formal", "visual", "kinematic"))
    assert result["errors"]


def test_good_bad_and_unrelated_parent_files_use_real_sdk_geometry_and_renderer(tmp_path, frozen_fixtures):
    require_validator()
    fixtures, config = frozen_fixtures
    good = submission(tmp_path)
    before = hashlib.sha256((good / "asset.usd").read_bytes()).hexdigest()
    (tmp_path / "agent.py").write_text("raise AssertionError('Must never execute')")
    (tmp_path / "reference.usd").write_text("unrelated parent artifact")
    first = cli.evaluate_submission(good, fixtures, config, tmp_path / "good-result")
    assert first["scores"]["formal"]["score_percent"] is not None, first["errors"]
    assert first["scores"]["geometry"]["score_percent"] == pytest.approx(100)
    assert first["scores"]["appearance"]["score_percent"] == pytest.approx(100, abs=.1)
    assert first["scores"]["visual"]["score_percent"] == pytest.approx(100, abs=.1)
    assert first["scores"]["kinematic"]["score_percent"] is None  # Static reference has no joint criteria.
    assert first["scores"]["formal"]["coverage_fraction"] == 1
    assert hashlib.sha256((good / "asset.usd").read_bytes()).hexdigest() == before
    assert len(first["candidate"]["files"]) == 1
    bad_parent = tmp_path / "bad"
    bad_parent.mkdir()
    bad = submission(bad_parent, size=.5, color=(0,0,1))
    worse = cli.evaluate_submission(bad, fixtures, config, tmp_path / "bad-result")
    assert worse["scores"]["geometry"]["score_percent"] < first["scores"]["geometry"]["score_percent"]
    assert worse["scores"]["appearance"]["score_percent"] < first["scores"]["appearance"]["score_percent"]
    assert worse["scores"]["visual"]["score_percent"] < first["scores"]["visual"]["score_percent"]


def test_renderer_failure_preserves_formal_geometry_and_kinematic_evidence(tmp_path, frozen_fixtures, monkeypatch):
    require_validator()
    fixtures, config = frozen_fixtures
    root = submission(tmp_path)
    original = cli._run_worker
    def unavailable(mode, *args, **kwargs):
        if mode == "render":
            return {"status": "unknown", "reason": "renderer_unavailable"}
        return original(mode, *args, **kwargs)
    monkeypatch.setattr(cli, "_run_worker", unavailable)
    result = cli.evaluate_submission(root, fixtures, config, tmp_path / "out")
    assert result["scores"]["formal"]["score_percent"] is not None
    assert result["scores"]["geometry"]["score_percent"] == pytest.approx(100)
    assert result["scores"]["appearance"]["score_percent"] is None
    assert result["scores"]["visual"]["score_percent"] is None


def test_frozen_candidate_copy_rejects_links_and_detects_changes(tmp_path):
    root = submission(tmp_path)
    frozen = tmp_path / "frozen"
    result = cli.freeze_candidate(root, frozen)
    assert result["status"] == "frozen"
    assert cli.verify_candidate(frozen, result) is True
    (frozen / "asset.usd").chmod(0o644)
    (frozen / "asset.usd").write_text("changed")
    assert cli.verify_candidate(frozen, result) is False
    (root / "link").symlink_to(root / "asset.usd")
    invalid = cli.freeze_candidate(root, tmp_path / "invalid")
    assert invalid["status"] == "invalid" and invalid["proven_invalid"] is True


def test_deadline_kills_entire_process_group(tmp_path):
    started = time.monotonic()
    result = cli._run_process([sys.executable, "-c", "import time; time.sleep(60)"],
                              tmp_path / "worker.log", timeout=.1)
    assert result["status"] == "unknown" and result["reason"] == "worker_timeout"
    assert time.monotonic() - started < 5


def test_module_cli_writes_typed_outputs_for_missing_submission(tmp_path, frozen_fixtures):
    fixtures, config = frozen_fixtures
    configfile = tmp_path / "evaluator.json"
    configfile.write_text(json.dumps(config))
    process = subprocess.run([sys.executable, "-m", "simready_eval.cli", "evaluate",
        "--submission", str(tmp_path / "missing"), "--fixtures", str(fixtures),
        "--config", str(configfile), "--output", str(tmp_path / "result")],
        capture_output=True, timeout=30)
    assert process.returncode == 0, process.stderr.decode()
    assert json.loads((tmp_path / "result/report.json").read_text())["candidate_invalid"] is True


def test_invalid_candidate_keeps_fixed_formal_accounting(tmp_path, frozen_fixtures):
    fixtures, config = frozen_fixtures
    result = cli.evaluate_submission(tmp_path / "missing", fixtures, config, tmp_path / "out")
    score = result["scores"]["formal"]["evidence"]
    assert score["applicable_count"] == score["failed_count"] == 46
    assert score["optional_score"]["applicable_count"] == 51


def test_coherent_fixture_replacement_during_candidate_work_is_rejected(tmp_path, frozen_fixtures, monkeypatch):
    fixtures, config = frozen_fixtures
    local = tmp_path / "fixtures"
    shutil.copytree(fixtures, local)
    root = submission(tmp_path)
    original = cli._run_worker
    def replace_fixture(mode, *args, **kwargs):
        result = original(mode, *args, **kwargs)
        if mode == "formal":
            eligibility_path = local / config["eligibility"]
            eligibility = json.loads(eligibility_path.read_text())
            # A coherent replacement still passes its own manifest hashes.
            cameras = json.loads((local / config["cameras"]).read_text())
            cameras["views"][0]["orthographic_scale"] = 5
            eligibility["render_camera_sha256"] = render._identity(cameras)
            (local / config["cameras"]).write_text(json.dumps(cameras))
            for row in eligibility["files"]:
                if row["path"] == config["cameras"]:
                    path = local / row["path"]
                    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    row["size_bytes"] = path.stat().st_size
            eligibility_path.write_text(json.dumps(eligibility))
        return result
    monkeypatch.setattr(cli, "_run_worker", replace_fixture)
    result = cli.evaluate_submission(root, local, config, tmp_path / "out")
    assert all(result["scores"][name]["score_percent"] is None for name in ("formal", "visual", "kinematic"))
    assert any("changed" in error.get("message", "") for error in result["errors"])


def test_configuration_preserves_trusted_final_allowance_above_default():
    config = {"schema": "simready-evaluator-config-v1", "task_id": "extended",
              "entrypoint": "asset.usd", "reference": "reference/asset.usd",
              "cameras": "comparison/cameras.json", "eligibility": "reference/eligibility.json",
              "fidelity_settings": {"enabled": True}, "final_seconds": 900}
    assert cli._configuration(config)["final_seconds"] == 900


@pytest.mark.parametrize("field", ["render_recipe_sha256", "render_camera_sha256",
                                 "render_bridge_code_sha256", "renderer_binary_sha256"])
@pytest.mark.parametrize("change", ["missing", "mismatch"])
def test_reference_render_identity_required_before_candidate_work(
    tmp_path, frozen_fixtures, monkeypatch, field, change
):
    fixtures, config = frozen_fixtures
    local = tmp_path / "fixtures"
    shutil.copytree(fixtures, local)
    path = local / config["eligibility"]
    eligibility = json.loads(path.read_text())
    if change == "missing":
        eligibility.pop(field)
    else:
        eligibility[field] = "0" * 64
    path.write_text(json.dumps(eligibility))
    monkeypatch.setattr(cli, "freeze_candidate", lambda *args, **kwargs: pytest.fail(
        "Candidate work started before frozen render identity verification"))
    result = cli.evaluate_submission(tmp_path / "submission", local, config, tmp_path / "out")
    assert result["candidate_invalid"] is False
    assert all(result["scores"][name]["score_percent"] is None
               for name in ("formal", "visual", "kinematic"))
    assert any("render" in error.get("message", "").lower() for error in result["errors"])


def test_nested_reference_with_sibling_materials_reaches_candidate_boundary(tmp_path, frozen_fixtures):
    from pxr import Sdf, Usd
    fixtures, original_config = frozen_fixtures
    config = deepcopy(original_config)
    local = tmp_path / "fixtures"
    shutil.copytree(fixtures, local)
    source = local / "reference/asset.usd"
    nested = local / "reference/simready_usd/sm_nested.usd"
    nested.parent.mkdir()
    source.rename(nested)
    materials = local / "reference/Materials"
    materials.mkdir()
    Image.new("RGB", (2, 2), (120, 30, 20)).save(materials / "texture.png")
    stage = Usd.Stage.Open(str(nested))
    stage.GetDefaultPrim().CreateAttribute("sourceTexture", Sdf.ValueTypeNames.Asset, custom=True).Set(
        "../Materials/texture.png")
    stage.GetRootLayer().Save()
    config["reference"] = "reference/simready_usd/sm_nested.usd"
    eligibility_path = local / config["eligibility"]
    eligibility = json.loads(eligibility_path.read_text())
    eligibility["profile"] = correspondence.reference_profile(observe(nested), config["fidelity_settings"])
    eligibility["profile_sha256"] = correspondence.digest(eligibility["profile"])
    eligibility["files"] = [{"path": path.relative_to(local).as_posix(), "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for directory in (local / "reference", local / "comparison")
        for path in sorted(directory.rglob("*")) if path.is_file() and path != eligibility_path]
    eligibility_path.write_text(json.dumps(eligibility))
    before = nested.read_bytes()
    result = cli.evaluate_submission(tmp_path / "missing", local, config, tmp_path / "out")
    assert result["candidate_invalid"] is True, result["errors"]
    assert result["scores"]["formal"]["score_percent"] == 0
    assert nested.read_bytes() == before
