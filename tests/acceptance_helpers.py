"""External transports for installed-wheel process-death acceptance.

Run as a standalone script. Product modules are imported only after the installed
wheel directory is added to sys.path. No product entrypoint is replaced.
"""

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class Transports:
    def __init__(self, root, fault):
        self.root = root
        self.fault = fault
        self.remote = root / "remote"
        self.remote.mkdir(parents=True, exist_ok=True)
        self.state_file = root / "transport-state.json"
        self.state = read(self.state_file, {
            "creates": 0, "generation": 0, "count": 0, "deletes": 0,
            "downloads": 0, "commit_attempts": 0, "commits": 0,
            "faults": [], "deleted": False,
        })

    def save(self):
        write(self.state_file, self.state)

    def fail(self, boundary):
        if self.fault == boundary and boundary not in self.state["faults"]:
            self.state["faults"].append(boundary)
            self.save()
            # Immediate process death bypasses product finally blocks and worker
            # thread handling, so the next invocation must recover durable state.
            os._exit(77)

    def once(self, name):
        if self.fault == name and name not in self.state["faults"]:
            self.state["faults"].append(name)
            self.save()
            raise ConnectionResetError("Acceptance transport interrupted")

    def install(self):
        import daytona
        import huggingface_hub

        transports = self

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def create(self, params, **kwargs):
                assert not transports.state["creates"], "Duplicate executor ownership"
                transports.state.update(creates=1, deleted=False)
                transports.save()
                return transports.sandbox()

            def get(self, identity, **kwargs):
                assert identity == "acceptance-executor"
                if transports.state["deleted"]:
                    from daytona.common.errors import DaytonaNotFoundError
                    raise DaytonaNotFoundError("Executor deleted")
                return transports.sandbox()

            def delete(self, sandbox, **kwargs):
                assert sandbox.id == "acceptance-executor"
                assert transports.state["downloads"] >= 1
                transports.once("delete_transient")
                transports.state["deletes"] += 1
                transports.state["deleted"] = True
                transports.save()
                transports.fail("deletion_acknowledgement")

        class Hub:
            def whoami(self):
                return {"auth": {"accessToken": {"role": "write"}}}

            def repo_info(self, *args, **kwargs):
                return SimpleNamespace(private=True, sha="head-" + str(transports.state["commits"]))

            def hf_hub_download(self, repo, filename, **kwargs):
                target = transports.root / "hub" / filename
                if not target.is_file():
                    from huggingface_hub.errors import EntryNotFoundError
                    raise EntryNotFoundError("No fixture object")
                return str(target)

            def create_commit(self, **kwargs):
                transports.state["commit_attempts"] += 1
                transports.save()
                transports.once("publication_transient")
                if transports.fault == "publication_permanent":
                    raise ConnectionResetError("Acceptance publication unavailable")
                assert kwargs["parent_commit"] == "head-" + str(transports.state["commits"])
                for operation in kwargs["operations"]:
                    target = transports.root / "hub" / operation.path_in_repo
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = operation.path_or_fileobj
                    target.write_bytes(source if isinstance(source, bytes) else Path(source).read_bytes())
                transports.state["commits"] += 1
                transports.save()
                transports.fail("remote_commit")
                transports.fail("missing_delivery")
                return SimpleNamespace(oid="head-" + str(transports.state["commits"]))

        daytona.Daytona = Client
        huggingface_hub.HfApi = Hub

    def sandbox(self):
        transports = self

        class Process:
            def create_session(self, *args, **kwargs):
                pass

            def execute_session_command(self, session, request, **kwargs):
                if session == "simready-benchmark":
                    transports.run_remote()
                return SimpleNamespace(cmd_id="acceptance-command")

            def get_session_command(self, *args, **kwargs):
                transports.run_remote()
                return SimpleNamespace(exit_code=0)

            def exec(self, command, **kwargs):
                if "ready.json" in command:
                    return SimpleNamespace(exit_code=0, result='{"port":32000}')
                if "stat -c %g" in command:
                    return SimpleNamespace(exit_code=0, result="1000")
                if "cat /work/controller-status.json" in command:
                    return SimpleNamespace(exit_code=0, result=json.dumps(read(transports.remote / "work/controller-status.json", {})))
                return SimpleNamespace(exit_code=0, result="")

        class FileSystem:
            def upload_file(self, source, target, **kwargs):
                path = transports.remote / target.lstrip("/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(source if isinstance(source, bytes) else Path(source).read_bytes())

            def download_file(self, source, destination=None, **kwargs):
                if source == "/work/controller-status.json":
                    # The executor survives coordinator death, including a lost
                    # acknowledgement for the initial asynchronous command.
                    transports.run_remote()
                if source == "/tmp/simready-benchmark-results.tar.gz":
                    transports.state["downloads"] += 1
                    transports.save()
                    transports.once("collection_transient")
                    if transports.fault == "collection_permanent":
                        raise OSError(28, "Acceptance collection disk full")
                    if transports.fault in {"receipt_corrupt", "receipt_corrupt_legacy"}:
                        for candidate in (transports.remote / "work/output").rglob("asset.usd"):
                            candidate.write_bytes(b"changed after its frozen receipt\n")
                    with tarfile.open(destination, "w:gz") as archive:
                        for path in sorted((transports.remote / "work").iterdir()):
                            if path.name in {"output", "run-result.json", "controller-status.json", "authority"}:
                                archive.add(path, arcname="work/" + path.name)
                    transports.fail("collection")
                    return None
                path = transports.remote / source.lstrip("/")
                if not path.is_file():
                    raise FileNotFoundError(source)
                raw = path.read_bytes()
                if isinstance(destination, (str, Path)):
                    Path(destination).write_bytes(raw)
                    return None
                return raw

        return SimpleNamespace(id="acceptance-executor", process=Process(), fs=FileSystem())

    def run_remote(self):
        from simready_bench.engine.controller import Controller
        from simready_bench.engine.inference_gateway import ProjectBudget
        from simready_bench.engine.scoring import score_validation
        from simready_bench.engine.verifier import scoring_policy
        from simready_bench.local import ParentAuthorityLedger, complete_trial_result, export_submission
        from simready_bench.reporting import export_report
        from simready_bench.resume import _digest, condition_plan

        envelope = read(self.remote / "work/controller.json")
        manifest = envelope["manifest"]
        identity, condition = next(iter(condition_plan(manifest).items()))
        work_root = self.remote / "authoring"
        actor = work_root / "actor"
        (actor / "output").mkdir(parents=True, exist_ok=True)
        controller = Controller(work_root / "trusted")
        if not (self.remote / "registered.json").is_file():
            controller.register_episode(identity, actor, task_id=identity, trial_id=identity,
                                        inference_limit=64, deadline_seconds=600)
            controller.start_episode(identity)
            write(self.remote / "registered.json", True)
        budget = ProjectBudget(**envelope["provider"]["project_budget"])
        configuration = "c" * 64
        allocation = _digest({"work_root": "/dev/shm/bench/a1", "configuration": configuration, "condition": identity})
        (self.remote / "work/authority").mkdir(mode=0o700, exist_ok=True)
        ParentAuthorityLedger(self.remote / "work/authority").reserve(
            budget, allocation_id=allocation,
            requests=manifest["defaults"]["budgets"]["inference_requests"],
            maximum_spend_usd=manifest["defaults"]["budgets"]["maximum_spend_usd"],
            work_root="/dev/shm/bench/a1",
        )
        if not self.state["generation"]:
            asyncio.run(self.generate(controller, identity))
        self.fail("admission")
        (actor / "output/asset.usd").write_bytes(b"invalid USD from fixture actor\n")
        snapshot = controller.finalize_submission(identity, reason="submitted")
        assert snapshot.valid and snapshot.sha256
        submission = work_root / "artifacts/submission"
        submission.parent.mkdir(mode=0o700, exist_ok=True)
        if not submission.is_dir():
            exported = export_submission(snapshot, submission)
            write(self.remote / "submission.json", exported)
        exported = read(self.remote / "submission.json")
        self.fail("candidate_sealing")
        report = self.remote / "work/output"
        if not (report / "results.json").is_file():
            score = score_validation(None, scoring_policy(), candidate_invalid=True)
            row = {
                **condition, "condition_id": identity, "trial_id": identity,
                "status": "completed", "requested_model": "gpt-6.1-sol",
                "inference_calls_used": 1, "validation_calls_used": 0,
                "snapshot_sha256": snapshot.sha256, "submission": exported,
                "evaluation": {"simready": score}, "simready": score,
                "official_reward": 0, "simready_reward": 0,
            }
            if self.fault == "metadata_hold":
                row["gateway_records"] = [
                    *read(self.remote / "gateway-records.json"),
                    {"gateway_request_id": "metadata-admission", "status": "denied",
                     "error_code": "hf_pricing_metadata_changed"},
                ]
                complete_trial_result(row, SimpleNamespace(trial_results=[SimpleNamespace(exception_info=None)]), normal_stop=True)
                assert row["infrastructure_hold"] == "hf_pricing_metadata_changed"
            result = {"schema": "simready-bench-matrix-result-v1",
                      "configuration_sha256": configuration,
                      "planned_conditions": 1, "completed_conditions": int(row["status"] == "completed"), "results": [row]}
            export_report(work_root, report, results=result)
            if self.fault != "receipt_corrupt_legacy":
                from simready_bench.controller_runner import ControllerReporter

                reporter = ControllerReporter(report, manifest)
                try:
                    reporter(work_root, result)
                finally:
                    reporter.close()
            portable = read(report / "results.json")
            write(self.remote / "work/run-result.json", portable)
            write(self.remote / "work/controller-status.json", {
                "status": "completed" if row["status"] == "completed" else "incomplete"})
        self.fail("evaluation")

    async def generate(self, controller, identity):
        import aiohttp
        from aiohttp import web
        from simready_bench.engine.inference_gateway import GatewayConfig, InferenceGateway, ProjectBudget
        from simready_bench.engine.request_cost import native_output_facts

        @asynccontextmanager
        async def serve(app):
            runner = web.AppRunner(app, access_log=None) if isinstance(app, web.Application) else app
            await runner.setup()
            site = runner.tcp_site("127.0.0.1", 0) if hasattr(runner, "tcp_site") else web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            try:
                yield "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
            finally:
                await runner.cleanup()

        app = web.Application()

        async def counter(request):
            self.state["count"] += 1
            self.save()
            return web.json_response({"object": "response.input_tokens", "input_tokens": 8})

        async def generate(request):
            self.state["generation"] += 1
            self.save()
            return web.json_response({"model": "gpt-6.1-sol", "status": "completed", "output": [],
                                      "usage": {"input_tokens": 8, "output_tokens": 1}})

        app.router.add_post("/v1/responses/input_tokens", counter)
        app.router.add_post("/v1/responses", generate)
        async with serve(app) as upstream:
            budget = ProjectBudget("simready-bench", "acceptance-gateway", True, upstream,
                                   "gpt-6.1-sol", time.time() + 120, 64, 2.5, "known")
            gateway = InferenceGateway(controller, GatewayConfig(
                upstream + "/v1", budget.model, "loopback-fixture", budget=budget,
                native_output_limits=True, max_output_tokens=None,
                max_input_tokens=native_output_facts(budget.model)["maximum_input_tokens"],
            ))
            token = gateway.issue_episode_token(identity)
            async with serve(gateway.runner()) as origin, aiohttp.ClientSession() as client:
                async with client.post(origin + "/v1/responses", json={"model": budget.model, "input": "Author the fixture asset"},
                                       headers={"Authorization": "Bearer " + token}) as response:
                    assert response.status == 200, await response.text()
                    await response.read()
                # HTTP EOF precedes the gateway's final accounting write. Keep
                # the real server alive until its terminal record is durable.
                until = asyncio.get_running_loop().time() + 3
                while asyncio.get_running_loop().time() < until:
                    records = gateway.records(identity)
                    if len(records) == 1 and records[0].get("status") == "completed" and records[0].get("upstream_status") == 200:
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("Gateway final accounting did not complete")
            write(self.remote / "gateway-records.json", records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("installed", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("fault")
    options = parser.parse_args()
    sys.path.insert(0, str(options.installed))
    import simready_bench.cli
    assert Path(simready_bench.cli.__file__).is_relative_to(options.installed)
    transports = Transports(options.root, options.fault)
    transports.install()
    os.environ["OPENAI_API_KEY"] = "acceptance-loopback-only"
    os.environ["DAYTONA_API_KEY"] = "acceptance-local-only"
    reference = options.root / "reference.usda"
    reference.write_text('#usda 1.0\ndef Xform "Reference" {}\n')
    output = options.root / "run"
    wheel = next((options.installed.parent / "wheel").glob("*.whl"))
    arguments = ["benchmark", str(reference), "--model", "gpt-6.1-sol", "--deploy", "daytona",
                 "--output-dir", str(output), "--repetitions", "1", "--max-spend-usd", "10",
                 "--wheel", str(wheel),
                 "--max-retries", "2", "--retry-delay-seconds", "0", "--maximum-retry-delay-seconds", "0",
                 "--hf-repo", "acceptance/results", "--hf-run", "fixture"]
    if output.exists():
        arguments.append("--resume")
    code = simready_bench.cli.main(arguments)
    transports.save()
    result = read(output / "results.json", {})
    write(options.root / "acceptance-result.json", {
        "exit_code": code, "product_cli": simready_bench.cli.__file__,
        "results": result, "transport": transports.state,
    })
    return code


if __name__ == "__main__":
    raise SystemExit(main())
