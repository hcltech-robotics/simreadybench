"""Native job failures retain stopped on-disk submissions without scoring them."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from simready_bench import captures, fidelity_control, local, reporting
from simready_bench.engine import environment, evaluation, harbor, harbor_services
from test_budget_completion import docker_boundary, evaluate_final, native_result
from test_confirmed_submit_stop import setup


pytestmark = pytest.mark.skipif(os.name == "nt", reason="Frozen submission ownership requires Linux")


@pytest.mark.parametrize(
    "failure", ["before_stop", "after_stop", "after_export", "cleanup", "invalid", "none"]
)
def test_raised_job_retains_a_portable_frozen_candidate(tmp_path, monkeypatch, failure):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    from harbor.models.task.config import TaskConfig

    async def exercise():
        controller, bridge, env, _ = setup(tmp_path, invalid=failure == "invalid")
        bridge.uninstall()
        docker_boundary(env, monkeypatch)
        work = tmp_path / "work"
        work.mkdir()
        root = work / "c0"
        source = tmp_path / "task"
        public = source / "environment/public"
        public.mkdir(parents=True)
        (public / "instruction.md").write_text(
            "Nominal dimensions in canonical metres XYZ: [1, 1, 1]\n"
        )
        (public / "thumbnail.png").write_bytes(b"input fixture")
        (source / "task.toml").write_text("")
        private = tmp_path / "config.json"
        private.write_text("{}")
        condition = {
            "condition_id": "condition-" + "a" * 64,
            "asset_id": "box", "harness_id": "pi", "model_id": "fixture",
            "skill_set_id": "none", "harness": {"adapter": "pi"},
            "skill_set": {},
            "settings": {
                "budgets": {
                    "agent_seconds": 60, "final_seconds": 10,
                    "validation_seconds": 5, "preview_seconds": 5,
                    "inference_requests": 3, "preview_requests": 1,
                    "validation_calls": 1,
                }
            },
        }
        job_directory = work / "native-jobs" / ("simready-bench-" + "a" * 12)
        job_directory.mkdir(parents=True)
        (job_directory / "trial.log").write_text("injected native job evidence\n")
        closed = []
        authored = b'#usda 1.0\n(defaultPrim="Box")\ndef Xform "Box" { def Cube "Visual" { double size=2 } }\n'

        class Services:
            def __init__(self, *_args, **_kwargs):
                self.verifiers = {}
                self.container_checks = {}
                self.transport = SimpleNamespace(
                    records=lambda _episode: [{"status": "completed", "upstream_status": 200}]
                )

            async def start(self):
                pass

            async def close(self):
                # Export must precede teardown of the services holding the snapshot.
                assert env.actor_stopped and env.final_snapshot is not None
                if failure != "invalid":
                    assert (root / "artifacts/submission/files/asset.usd").read_bytes() == authored
                closed.append(True)
                if failure == "cleanup":
                    raise OSError("Injected teardown failure")

        class NativeJob:
            job_dir = job_directory

            def add_hook(self, *_args):
                pass

            async def run(self):
                if failure != "invalid":
                    (env.actor_root / "output/asset.usd").write_bytes(authored)
                    (env.actor_root / "output/texture.bin").write_bytes(b"authored dependency")
                if failure in ("after_stop", "after_export", "none"):
                    await bridge.request_stop("episode", "inference_budget_exhausted")
                if failure not in ("after_export", "none"):
                    raise ConnectionError("Injected native job failure after authoring")
                final = evaluate_final(tmp_path, env.final_snapshot)
                path = controller.state_dir / "final-service-reports/episode.json"
                path.parent.mkdir()
                path.write_text(json.dumps(final))
                return native_result(None)

        async def create(_config):
            return NativeJob()

        def compare(*_args):
            if failure == "after_export":
                raise ConnectionError("Injected post-export comparison failure")
            return {"alignment": None}

        monkeypatch.setattr(Job, "create", create)
        monkeypatch.setattr(JobConfig, "model_validate", lambda _value: {})
        monkeypatch.setattr(TaskConfig, "model_validate_toml", lambda _value: SimpleNamespace())
        monkeypatch.setattr(harbor, "NativeLifecycle", lambda *_args, **_kwargs: bridge)
        monkeypatch.setattr(harbor_services, "ReferenceTaskInput", lambda *_args: None)
        monkeypatch.setattr(environment, "validate_task_reference_binding", lambda *_a, **_k: None)
        monkeypatch.setattr(evaluation, "CandidateObserver", lambda **_kwargs: None)
        monkeypatch.setattr(evaluation, "Evaluator", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(local, "runtime_policy", lambda *_a, **_k: {
            "allowed_reference_root": str(tmp_path), "validator_python": "fixture",
            "source_root": str(tmp_path), "expected_hostname": "fixture",
        })
        monkeypatch.setattr(local, "_controller_images", lambda *_args: [])
        monkeypatch.setattr(local, "stage_skills", lambda *_args: [])
        monkeypatch.setattr(local, "agent_config", lambda *_args: {})
        monkeypatch.setattr(local, "_gateway", lambda *_a, **_k: None)
        monkeypatch.setattr(local, "native_services_class", lambda: Services)
        monkeypatch.setattr(local.shutil, "disk_usage", lambda _path: SimpleNamespace(free=10 * 1024**3))
        monkeypatch.setattr(fidelity_control, "prepare_reference", lambda *_args: {})
        monkeypatch.setattr(fidelity_control, "evaluate_final", compare)
        monkeypatch.setattr(captures, "render_final", lambda *_a, **_k: {"status": "unavailable"})
        result = await local._condition(
            condition, {"task_path": source, "private_config": private},
            {"configuration_sha256": "c" * 64}, root, controller,
        )
        assert closed == [True]
        assert json.loads((root / "result.json").read_text()) == result
        assert result["actor_stopped"] is True
        if failure == "none":
            assert result["status"] == "completed" and result["official_reward"] == 1
        else:
            assert result["status"] == "infrastructure_error"
            assert result["error_type"] == "ConnectionError"
            assert result["evaluation"] is None
            assert result["official_reward"] is result["diagnostic_reward"] is None
            assert result["model_quality_evidence"] is False
            assert result["harbor_trace_paths"] == [str(job_directory / "trial.log")]
        if failure == "cleanup":
            assert result["cleanup_error_type"] == "OSError"
        if failure == "invalid":
            assert result["snapshot_sha256"] is None
            assert result.get("submission") is None
            return
        manifest = json.loads(Path(result["submission"]["manifest"]).read_text())
        assert {row["relative_path"] for row in manifest["files"]} == {"asset.usd", "texture.bin"}
        assert result["snapshot_sha256"] == env.final_snapshot.sha256 == manifest["snapshot_sha256"]
        assert not os.path.samefile(
            Path(env.final_snapshot.path) / "asset.usd",
            Path(result["submission"]["directory"]) / "asset.usd",
        )
        durable = tmp_path / "report"
        reporting.export_report(work, durable, results={"results": [result]})
        portable = json.loads((durable / "results.json").read_text())["results"][0]
        assert portable["candidate"]["status"] == "available"
        assert portable["candidate"]["snapshot_sha256"] == env.final_snapshot.sha256
        assert (durable / portable["candidate"]["entrypoint"]).read_bytes() == authored
        assert portable["status"] == result["status"]
        if failure != "none":
            assert portable["simready_reward"] is None

    asyncio.run(exercise())
