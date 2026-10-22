"""Harbor task input, authoring tools and separate final verification."""

from __future__ import annotations
import asyncio, hashlib, hmac, json, secrets, shutil, tempfile, time
import os
import re
from dataclasses import dataclass
from pathlib import Path
from aiohttp import web
from simready_bench.engine.services import NativeServices, NativeServiceTransport
from simready_bench.engine.inference_gateway import InferenceGateway
from simready_bench.engine.harbor import NativeActorStopAdapter
from simready_bench.engine.tools import ActorTools, _png, _result
from simready_bench.engine.task_contract import PUBLIC_FILES
from simready_bench.engine.native_output import EXTENSION_FILENAME
from simready_bench.engine.verifier import PROTOCOL_VERSION
from simready_bench.engine.evaluation import Evaluator
from simready_bench.engine.verifier import validate_report


def _validation_failure(controller, error, *, episode_id, snapshot, phase, report=None, job_id=None):
    """Write bounded private diagnostics before returning a stable actor error."""
    from simready_bench.engine.scoring import public_report

    causes, seen = [], set()
    current = error
    while current is not None and id(current) not in seen and len(causes) < 8:
        seen.add(id(current))
        message = public_report({"message": str(current)[:2048]})["message"]
        message = re.sub(r"[A-Za-z]:[\\/][^\s\"'<>]+", "<private host path>", message)
        message = re.sub(r"\b(?:hf_|sk-|ghp_)[A-Za-z0-9_-]+", "<redacted secret>", message)
        message = re.sub(r"(?i)(?:authorization|proxy-authorization|token|api[_-]?key|secret|password)\s*[:=]\s*(?:(?:basic|bearer)\s+)?[^\s\"'<>]+", "<redacted secret>", message)
        message = re.sub(r"(?i)(https?://)[^/\s@]+:[^/\s@]+@", r"\1<redacted secret>@", message)
        causes.append({"type": type(current).__name__, "message": message[:1024]})
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    failure_id = secrets.token_hex(16)
    record = {
        "schema": "simready-validation-failure-v1", "failure_id": failure_id,
        "episode_id": episode_id, "job_id": job_id, "phase": phase,
        "snapshot_id": getattr(snapshot, "snapshot_id", None),
        "snapshot_sha256": getattr(snapshot, "sha256", None),
        "causes": causes, "recorded_at": time.time(),
        "partial_report_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True, default=str).encode()
        ).hexdigest() if report is not None else None,
    }
    directory = controller.state_dir / "validation-failures"
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / (failure_id + ".json")
    with path.open("x") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)
    return failure_id


class _DiagnosticObserver:
    """Retain the cause before the scorer converts unavailable observation to unknown."""

    def __init__(self, observer, controller):
        self.observer, self.controller = observer, controller
        self.failures = {}

    def __call__(self, snapshot, *, episode_id, deadline_at):
        key = (episode_id, snapshot.snapshot_id)
        self.failures.pop(key, None)
        try:
            return self.observer(snapshot, episode_id=episode_id, deadline_at=deadline_at)
        except Exception as error:
            failure_id = _validation_failure(self.controller, error, episode_id=episode_id,
                                            snapshot=snapshot, phase="candidate_observation")
            self.failures[key] = (error, failure_id)
            raise


@dataclass(frozen=True)
class ReferenceTaskInput:
    task_path: Path
    thumbnail: Path
    public_text: str
    task_config: dict
    public_dir: Path

    def __post_init__(self):
        for key in ("task_path", "thumbnail", "public_dir"):
            p = Path(getattr(self, key)).absolute()
            if p.resolve() != p:
                raise ValueError("No symlink task binding")
            object.__setattr__(self, key, p)
        cfg = json.loads(json.dumps(self.task_config, allow_nan=False))
        if cfg.get("protocol") != PROTOCOL_VERSION or "validator_scoring" not in cfg:
            raise ValueError("Pinned NVIDIA scoring policy required")
        if not self.task_path.is_dir() or not self.public_dir.is_dir():
            raise ValueError("Existing generated task required")
        public_files = {p.name for p in self.public_dir.iterdir()}
        if public_files not in (PUBLIC_FILES, PUBLIC_FILES - {EXTENSION_FILENAME}):
            raise ValueError("Exact public task allowlist required")
        if set(cfg.get("public_file_hashes", {})) != public_files:
            raise ValueError("Exact frozen public file hashes required")
        for p in self.public_dir.iterdir():
            if p.is_symlink() or not p.is_file() or p.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Bounded regular public files required")
            if (
                hashlib.sha256(p.read_bytes()).hexdigest()
                != cfg["public_file_hashes"][p.name]
            ):
                raise ValueError("Public file binding changed")
        if (
            self.thumbnail != self.public_dir / "thumbnail.png"
            or self.public_text != (self.public_dir / "instruction.md").read_text()
        ):
            raise ValueError("Public input must match generated frozen files")
        if "validator_scoring" in cfg:
            from simready_bench.engine.scoring import validate_policy

            validate_policy(cfg["validator_scoring"])
            public = json.loads((self.public_dir / "scoring-policy.json").read_text())
            if public.get("validator_scoring") != cfg["validator_scoring"]:
                raise ValueError("Public validator scoring policy differs")
        _png(self.thumbnail.read_bytes())
        object.__setattr__(self, "task_config", cfg)

    @property
    def public_repair_contract(self):
        value = {
            "protocol": PROTOCOL_VERSION,
            "nominal_dimensions_m": list(self.task_config["nominal_dimensions_m"]),
            "probes": [],
        }
        return value


class ReferenceValidationAdapter:
    kind = "validation"

    def __init__(self, services):
        self.services = services
        self.controller = services.lifecycle.controller

    def __call__(self, job):
        record = self.services.input_for_episode(job.episode_id)
        report = self.services.evaluator.evaluate(
            job.snapshot,
            record.task_config,
            job.episode_id,
            job.deadline_at,
            final=False,
        )
        return report


class ReferenceActorTools(ActorTools):
    def __init__(self, services):
        self.services = services
        super().__init__(
            services.lifecycle.controller,
            stop_actor=NativeActorStopAdapter(services.lifecycle),
        )
        self.validator = ReferenceValidationAdapter(services)
        from simready_bench.engine.validation_reports import ReportStore

        self.validator_reports = ReportStore(
            services.evaluator.root / "public-validator-reports"
        )

    def describe(self, token):
        result = super().describe(token)
        binding = self._binding(token)
        if (
            binding is not None
            and "validator_scoring"
            in self.services.input_for_episode(binding.episode_id).task_config
        ):
            from simready_bench.engine.validation_reports import TOOL
            from simready_bench.engine.tools import _result

            data = result.structured_content
            return _result({"tools": [*data["tools"], TOOL]})
        return result

    def invoke(self, token, operation, arguments):
        if operation != "get_validation_report":
            return super().invoke(token, operation, arguments)
        from simready_bench.engine.tools import _error

        binding = self._binding(token)
        if binding is None:
            return _error("unauthorized")
        try:
            return _result(self.validator_reports.read(binding.episode_id, arguments))
        except (ValueError, OSError):
            return _error("validation_report_unavailable_or_invalid_arguments")

    def _adapter(self, adapter, kind):
        if kind == "validation":
            if (
                type(adapter) is not ReferenceValidationAdapter
                or adapter.controller is not self.controller
            ):
                raise RuntimeError("Exact reference validation adapter required")
            return adapter
        return super()._adapter(adapter, kind)

    def _validation_job(self, *, binding, job, payload, arguments):
        raw = None
        phase = "interactive_evaluation"
        try:
            raw = self._adapter(self.validator, "validation")(job)
            phase = "interactive_report_delivery"
            self._private(job.job_id, raw)
            delivery = self.validator_reports.register(
                job.episode_id, raw.get("nvidia_validator_report")
            )
            score = raw["feedback"]["simready"]
            summary = {
                key: score[key]
                for key in (
                    "schema",
                    "policy_id",
                    "status",
                    "score_percent",
                    "base_credit_percent",
                    "completion_bonus",
                    "passed_count",
                    "failed_count",
                    "unknown_count",
                    "applicable_count",
                    "official_conformance",
                )
                if key in score
            }
            summary["checks"] = [
                {
                    "requirement_id": check["requirement_id"],
                    "status": check["status"],
                    "finding_count": len(check.get("findings", [])),
                }
                for check in score["checks"]
            ]
            feedback = {
                "schema_version": raw["feedback"]["schema_version"],
                "snapshot_sha256": raw.get("snapshot_sha256"),
                "reward": raw.get("reward"),
                "simready": summary,
                "nvidia_validator_report": delivery,
            }
            payload.update(feedback=feedback, status="completed")
            return _result(payload)
        except Exception as error:
            failure_id = _validation_failure(
                self.controller, error, episode_id=job.episode_id,
                snapshot=job.snapshot, phase=phase, report=raw, job_id=job.job_id,
            )
            return _result(
                {**payload, "status": "infrastructure_error", "reward": None, "failure_id": failure_id},
                error=True,
            )


class ReferenceServiceTransport(NativeServiceTransport):
    def __init__(self, services):
        super().__init__(
            services.lifecycle.controller,
            services.config,
            tools=services.tools,
            on_stop_requested=services.lifecycle.request_stop,
        )


class ReferenceVerifierTransport(InferenceGateway):
    def __init__(self, services, episode_id, token):
        super().__init__(services.lifecycle.controller, services.config)
        self.services = services
        self.episode_id = episode_id
        self.token_hash = hashlib.sha256(token.encode()).digest()
        self.lock = asyncio.Lock()

    async def handle(self, request):
        token = request.headers.get("Authorization", "")
        token = token[7:] if token.startswith("Bearer ") else ""
        if len(token) > 512 or not hmac.compare_digest(
            hashlib.sha256(token.encode()).digest(), self.token_hash
        ):
            return web.json_response({"error": "unauthorized"}, status=401)
        if (
            request.method != "POST"
            or request.path != "/internal/final"
            or request.query_string
        ):
            return web.json_response({"error": "endpoint_not_authorized"}, status=400)
        try:
            async with asyncio.timeout(5):
                if (
                    request.content_length is None
                    or request.content_length > 1024
                    or await request.json() != {}
                ):
                    raise ValueError()
        except (ValueError, asyncio.TimeoutError):
            return web.json_response({"error": "invalid_arguments"}, status=400)
        lifecycle = self.services.lifecycle
        env = lifecycle._episodes.get(self.episode_id)
        episode = lifecycle.controller.get_episode(self.episode_id)
        if (
            env is None
            or not env.actor_stopped
            or episode.status != "closed"
            or (env.final_snapshot is None)
        ):
            return web.json_response(
                {"error": "final_snapshot_unavailable"}, status=409
            )
        async with self.lock:
            snapshot = env.final_snapshot
            record = self.services.input_for_episode(self.episode_id)
            reason = {
                "submitted": "voluntary_good_enough",
                "agent_completed": "voluntary_good_enough",
                "inference_budget_exhausted": "budget_exhausted",
                "inference_attempt_budget_exhausted": "infrastructure_error",
                "project_request_budget_exhausted": "budget_exhausted",
                "project_attempt_budget_exhausted": "infrastructure_error",
                "project_spend_budget_exhausted": "budget_exhausted",
                "walltime_exhausted": "deadline",
                "cancelled": "user_stop",
            }.get(episode.termination_reason, "unknown")
            service_deadline = time.time() + min(
                self.services.verifier_timeout_seconds, 580
            )
            phase, report = "final_evaluation", None
            try:
                async with asyncio.timeout(max(0.001, service_deadline - time.time())):
                    if "validator_scoring" in record.task_config and (
                        not snapshot.valid
                    ):
                        from simready_bench.engine.scoring import (
                            record_native_invalid_snapshot,
                        )

                        await asyncio.to_thread(
                            record_native_invalid_snapshot,
                            self.services.evaluator,
                            controller=lifecycle.controller,
                            environment=env,
                            snapshot=snapshot,
                            task_config=record.task_config,
                            task_input=record,
                        )
                    from simready_bench.engine.validation_recovery import evaluate_final_snapshot

                    report = await evaluate_final_snapshot(
                        self.services.evaluator, lifecycle.controller,
                        snapshot, record.task_config, self.episode_id,
                        deadline_at=service_deadline - 2, stop_reason=reason,
                    )
                    from simready_bench.engine.scoring import build_wire

                    phase = "final_wire_validation"
                    wire = build_wire(report, snapshot=snapshot)
                    validate_report(wire)
                directory = lifecycle.controller.state_dir / "final-service-reports"
                phase = "final_report_persistence"
                directory.mkdir(mode=448, exist_ok=True)
                p = directory / (self.episode_id + ".json")
                tmp = p.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
                )
                tmp.chmod(384)
                tmp.replace(p)
            except Exception as error:
                failure_id = _validation_failure(
                    lifecycle.controller, error, episode_id=self.episode_id,
                    snapshot=snapshot, phase=phase, report=report,
                )
                return web.json_response(
                    {"error": "final_validation_failed", "failure_id": failure_id}, status=503
                )
            return web.json_response(wire)


class ReferenceNativeServices(NativeServices):
    def __init__(
        self, lifecycle, config, *, task_inputs, evaluator, verifier_timeout_seconds=600
    ):
        if type(evaluator) is not Evaluator:
            raise ValueError("Exact NVIDIA evaluator required")
        if evaluator.observer is not None and not isinstance(evaluator.observer, _DiagnosticObserver):
            evaluator.observer = _DiagnosticObserver(evaluator.observer, lifecycle.controller)
        if any((type(r) is not ReferenceTaskInput for r in task_inputs)):
            raise ValueError("Frozen task inputs required")
        super().__init__(
            lifecycle,
            config,
            task_inputs=task_inputs,
            evaluator=evaluator,
            verifier_timeout_seconds=verifier_timeout_seconds,
        )
        self.tools = ReferenceActorTools(self)
        self.transport = ReferenceServiceTransport(self)

    def input_for_episode(self, episode_id):
        env = self.lifecycle._episodes[episode_id]
        return self.public_input(env.environment_dir)

    def actor_mounts_for_directory(self, environment_dir, actor_root):
        if self.runner is None or self.socket_dir is None:
            raise RuntimeError("Native Unix services must start before actor creation")
        record = self.public_input(environment_dir)
        ReferenceTaskInput(
            record.task_path,
            record.thumbnail,
            record.public_text,
            record.task_config,
            record.public_dir,
        )
        public = actor_root / "public"
        public.mkdir(mode=448)
        for source in record.public_dir.iterdir():
            shutil.copyfile(source, public / source.name)
            (public / source.name).chmod(292)
        public.chmod(365)
        tool_mounts = []
        if getattr(self.lifecycle, "tool_protocol", "legacy") == "bounded-v1":
            from simready_bench.engine.tool_protocol import stage_tool_protocol

            tool_mounts.append(stage_tool_protocol(actor_root / "tool-protocol"))
        return [
            {
                "type": "bind",
                "source": str(public),
                "target": "/task",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
            {
                "type": "bind",
                "source": str(self.socket_dir),
                "target": "/run/simready-services",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
        ] + self.public_client_mounts(actor_root) + tool_mounts

    async def ensure_verifier(self, episode_id):
        if episode_id in self.verifiers:
            return
        env = self.lifecycle._episodes[episode_id]
        if not env.actor_stopped or env.final_snapshot is None:
            raise RuntimeError("Stop/freeze before separate verifier")
        directory = Path(tempfile.mkdtemp(prefix="simready-reference-verifier-"))
        directory.chmod(448)
        token = secrets.token_urlsafe(32)
        (directory / "token").write_text(token)
        (directory / "token").chmod(384)
        entrypoint = directory / "test.sh"
        entrypoint.write_text(
            "#!/bin/sh\nset -eu\nexec python3 -I -B -c 'import sys;sys.path.insert(0,\"/opt/simready-source\");from simready_bench.engine.verifier import main;main()' --socket /run/simready-verifier/service.sock --output-dir /logs/verifier\n"
        )
        entrypoint.chmod(365)
        runner = ReferenceVerifierTransport(self, episode_id, token).runner()
        try:
            await runner.setup()
            await runner.unix_site(directory / "service.sock").start()
        except BaseException:
            await runner.cleanup()
            raise
        self.verifiers[episode_id] = (directory, runner)

    def verifier_mounts(self, trial_paths):
        mounts = super().verifier_mounts(trial_paths)
        private = next(
            (
                Path(m["source"])
                for m in mounts
                if m["target"] == "/run/simready-verifier"
            )
        )
        entrypoint = private / "test.sh"
        if (
            entrypoint.is_symlink()
            or not entrypoint.is_file()
            or entrypoint.stat().st_mode & 146
        ):
            raise RuntimeError("Trusted executable verifier entrypoint required")
        return mounts + [
            {
                "type": "bind",
                "source": str(Path(__file__).absolute().parents[2]),
                "target": "/opt/simready-source",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
            {
                "type": "bind",
                "source": str(entrypoint),
                "target": "/tests/test.sh",
                "read_only": True,
                "bind": {"create_host_path": False},
            },
        ]
