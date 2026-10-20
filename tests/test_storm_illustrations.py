import copy, hashlib, json
from pathlib import Path
import pytest


def test_manifest_renderer_selection_is_separate_from_scoring_and_model_budgets(setup):
    from simready_bench.config import resolve_experiment

    root, manifest, policy = setup
    manifest["defaults"]["illustrations"] = {
        "renderer": "storm-llvmpipe",
        "width": 384,
        "views": ["isometric", "front"],
        "seconds": 180,
    }
    result = resolve_experiment(manifest, base_dir=root, policy=policy)
    settings = result["conditions"][0]["settings"]
    assert settings["illustrations"] == manifest["defaults"]["illustrations"]
    assert settings["budgets"]["maximum_spend_usd"] == 0
    for value in (
        {"renderer": "shell-command"},
        {"renderer": "storm-llvmpipe", "width": 100000},
        {"renderer": "storm-llvmpipe", "views": ["$(command)"]},
    ):
        manifest["defaults"]["illustrations"] = value
        with pytest.raises(ValueError):
            resolve_experiment(manifest, base_dir=root, policy=policy)


def test_active_GL_renderer_must_prove_software_execution():
    from simready_bench.captures import check_software_GL

    assert check_software_GL("Mesa", "llvmpipe (LLVM 20.1.8, 256 bits)", "4.5")[
        "software_verified"
    ]
    for renderer in ("NVIDIA RTX 6000", "softpipe", "unknown", ""):
        with pytest.raises(ValueError):
            check_software_GL("Mesa", renderer, "4.5")


def test_direct_X_server_is_cleaned_up_without_waiting_for_ready_signals(
    monkeypatch, tmp_path
):
    from simready_bench import storm as storm_worker

    class Server:
        stopped = False

        def poll(self):
            return None

        def terminate(self):
            self.stopped = True

        def wait(self, timeout):
            return 0

    server = Server()
    commands = []

    def start(argv, **kwargs):
        commands.append(argv)
        return server

    monkeypatch.setattr(storm_worker.subprocess, "Popen", start)
    monkeypatch.setattr(storm_worker, "_display_ready", lambda: True)
    with storm_worker.virtual_display():
        assert storm_worker.os.environ["DISPLAY"] == ":97"
    assert server.stopped and commands[0][0] == "Xvfb"
    assert "-nolisten" in commands[0] and "tcp" in commands[0]


def test_MDL_only_and_missing_assets_are_explicit_illustration_warnings():
    from simready_bench.captures import material_support

    records = [
        {
            "path": "/M/mdl",
            "shader_ids": ["mdl"],
            "universal_surface": False,
            "MDL_surface": True,
        },
        {
            "path": "/M/preview",
            "shader_ids": ["UsdPreviewSurface", "UsdUVTexture"],
            "universal_surface": True,
            "MDL_surface": True,
        },
    ]
    result = material_support(records, ["missing.png"])
    assert result["MDL_only_materials"] == ["/M/mdl"]
    assert result["unresolved_dependencies"] == ["missing.png"]
    assert result["illustration_only"] and result["scoring"] is False
    assert result["material_fidelity"] == "approximate"
    assert len(result["warnings"]) == 2


def test_failed_illustration_cannot_abort_or_mutate_a_scored_condition(monkeypatch):
    from simready_bench import captures as illustrations

    condition = {
        "settings": {"illustrations": {"renderer": "storm-llvmpipe"}},
        "score": 80.54347826086956,
    }
    original = copy.deepcopy(condition)

    def broken(*args):
        raise RuntimeError("private renderer failure details")

    monkeypatch.setattr(illustrations, "_render_final", broken)
    result = illustrations.render_final(condition, {}, None, None, None)
    assert (
        condition == original
        and result["status"] == "unavailable"
        and result["scoring"] is False
    )
    assert "private renderer failure details" not in json.dumps(result)


def test_report_retains_matched_captures_without_changing_score(tmp_path):
    from PIL import Image
    from simready_bench.reporting import export_report

    work = tmp_path / "work"
    work.mkdir()
    image = work / "contact.png"
    Image.new("RGB", (32, 32), "brown").save(image)
    illustration = {
        "status": "rendered",
        "renderer": "storm-llvmpipe",
        "illustration_only": True,
        "scoring": False,
        "warnings": ["MDL-only material approximated"],
        "rig_sha256": "a" * 64,
        "images": [
            {
                "file": str(image),
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "view": "matched contact sheet",
            }
        ],
    }
    evaluation = {
        "simready": {"schema": "simready-validator-score-v1", "score_percent": 0}
    }
    original = copy.deepcopy(evaluation)
    result = {
        "schema": "simready-bench-matrix-result-v1",
        "results": [
            {
                "condition_id": "one",
                "asset_id": "cup",
                "model_id": "control",
                "status": "completed",
                "evaluation": evaluation,
                "illustrations": illustration,
            }
        ],
    }
    report = export_report(work, tmp_path / "report", results=result)
    data = json.loads(Path(report["results"]).read_text())["results"][0]
    assert evaluation == original and data["simready_reward"] == 0
    assert data["illustrations"]["scoring"] is False
    assert len(data["images"]) == 1
    page = Path(report["html"]).read_text()
    assert "Matched captures" in page and "Matched contact sheet" in page
    assert "evidence/" + data["images"][0]["path"] in page
    assert "MDL-only material approximated" not in page
