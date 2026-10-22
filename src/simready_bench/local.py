"""Local Harbor execution with persistent request and spend limits."""

from __future__ import annotations
import asyncio, hashlib, json, math, os, re, shutil, sqlite3, subprocess, tempfile, time
from dataclasses import asdict, replace
from functools import lru_cache
from decimal import Decimal
from pathlib import Path
from simready_bench.preparation import (
    checked_resolved,
    runtime_policy,
    confined_work_dir,
    prepare_assets,
    check_prepared,
    write_json,
    digest,
)
from simready_bench.harnesses import stage_skills, agent_config, native_services_class


class MatrixLedger:
    """Reserve configured condition ceilings atomically; uncertainty never refunds.

    This is an additional aggregate guard. Actual monetary reservations remain
    owned by the engine gateway and approved operator ProjectBudget authority.
    """

    def __init__(self, path, configuration_sha256, maximum_spend_usd):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=448)
        self.binding = configuration_sha256
        self.maximum = Decimal(str(maximum_spend_usd))
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS binding(identity TEXT PRIMARY KEY,maximum TEXT)"
            )
            rows = db.execute("SELECT identity,maximum FROM binding").fetchall()
            if rows and rows != [(self.binding, str(self.maximum))]:
                raise ValueError(
                    "Existing allowance ledger configuration/cap differs; never reset it"
                )
            db.execute(
                "INSERT OR IGNORE INTO binding VALUES (?,?)",
                (self.binding, str(self.maximum)),
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS conditions(id TEXT PRIMARY KEY,ceiling TEXT,status TEXT,result TEXT)"
            )
        self.path.chmod(384)

    def _db(self):
        return sqlite3.connect(self.path, timeout=10)

    def reserve(self, identifier, amount):
        value = Decimal(str(amount))
        if not value.is_finite() or value < 0:
            raise ValueError("Finite nonnegative condition ceiling required")
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT ceiling,status,result FROM conditions WHERE id=?", (identifier,)
            ).fetchone()
            if row:
                if Decimal(row[0]) != value:
                    raise ValueError("Condition ceiling changed")
                return {
                    "status": row[1],
                    "result": json.loads(row[2]) if row[2] else None,
                }
            total = sum(
                (Decimal(x[0]) for x in db.execute("SELECT ceiling FROM conditions")),
                Decimal(0),
            )
            if total + value > self.maximum:
                raise ValueError("Aggregate matrix allowance exhausted")
            db.execute(
                "INSERT INTO conditions VALUES (?,?,?,NULL)",
                (identifier, str(value), "admitted"),
            )
        return {"status": "new", "result": None}

    def complete(self, identifier, result):
        with self._db() as db:
            if (
                db.execute(
                    "UPDATE conditions SET status=?,result=? WHERE id=? AND status=?",
                    (
                        "completed",
                        json.dumps(result, sort_keys=True, allow_nan=False),
                        identifier,
                        "admitted",
                    ),
                ).rowcount
                != 1
            ):
                raise ValueError("Condition result already frozen or not admitted")

    def records(self):
        with self._db() as db:
            return [
                {
                    "condition_id": r[0],
                    "reserved_ceiling_usd": float(Decimal(r[1])),
                    "status": r[2],
                }
                for r in db.execute(
                    "SELECT id,ceiling,status FROM conditions ORDER BY id"
                )
            ]


class ParentAuthorityLedger:
    """One persistent operator-owned parent allowance across all work directories.

    Planned ceilings are reserved before deriving child authority. A deleted
    executor can retire its reservation using collected condition ceilings.
    """

    def __init__(self, state_dir):
        self.root = Path(state_dir).absolute()
        if (
            self.root.resolve() != self.root
            or not self.root.is_dir()
            or self.root.stat().st_uid != os.getuid()
            or self.root.stat().st_mode & 63
        ):
            raise ValueError(
                "Existing private operator-owned authority_state_dir required"
            )
        self.path = self.root / "parent-allowances.sqlite3"
        try:
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 384
            )
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        if self.path.is_symlink() or (
            self.path.exists()
            and (
                not self.path.is_file()
                or self.path.stat().st_nlink != 1
                or self.path.stat().st_uid != os.getuid()
                or self.path.stat().st_mode & 63
            )
        ):
            raise ValueError("Private nonlinked parent allowance ledger required")
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS parents(id TEXT PRIMARY KEY,fingerprint TEXT,maximum_requests INTEGER,maximum_usd TEXT,expires REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS allocations(id TEXT PRIMARY KEY,parent TEXT,requests INTEGER,usd TEXT,work_root TEXT)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS retirements(allocation TEXT PRIMARY KEY,requests INTEGER,usd TEXT,proof_sha256 TEXT)"
            )
        self.path.chmod(384)

    def _db(self):
        return sqlite3.connect(self.path, timeout=15)

    def reserve(self, budget, *, allocation_id, requests, maximum_spend_usd, work_root):
        from dataclasses import asdict

        amount = Decimal(str(maximum_spend_usd))
        parent_amount = Decimal(str(budget.maximum_spend_usd))
        if (
            budget.project != "simready-bench"
            or budget.approved is not True
            or budget.expires_at <= time.time()
            or (not budget.authorization_id)
        ):
            raise ValueError(
                "Existing approved unexpired simready-bench parent authority required"
            )
        if (
            type(requests) is not int
            or requests <= 0
            or (not amount.is_finite())
            or (amount <= 0)
        ):
            raise ValueError("Positive bounded planned parent allocation required")
        fingerprint = digest(asdict(budget))
        work = str(Path(work_root).absolute())
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            parent = db.execute(
                "SELECT fingerprint FROM parents WHERE id=?", (budget.authorization_id,)
            ).fetchone()
            if parent and parent[0] != fingerprint:
                raise ValueError(
                    "Parent authorization fingerprint changed; no cap reset"
                )
            existing = db.execute(
                "SELECT parent,requests,usd,work_root FROM allocations WHERE id=?",
                (allocation_id,),
            ).fetchone()
            wanted = (budget.authorization_id, requests, str(amount), work)
            if existing:
                if db.execute(
                    "SELECT 1 FROM retirements WHERE allocation=?", (allocation_id,)
                ).fetchone():
                    raise ValueError(
                        "A retired allocation cannot authorise another executor"
                    )
                if existing != wanted:
                    raise ValueError("Existing allocation identity changed")
                return {"allocation_id": allocation_id, "status": "already_reserved"}
            rows = db.execute(
                "SELECT COALESCE(r.requests,a.requests),COALESCE(r.usd,a.usd) FROM allocations a LEFT JOIN retirements r ON r.allocation=a.id WHERE a.parent=?",
                (budget.authorization_id,),
            ).fetchall()
            if (
                sum((r[0] for r in rows)) + requests > budget.maximum_requests
                or sum((Decimal(r[1]) for r in rows), Decimal(0)) + amount
                > parent_amount
            ):
                raise ValueError("Persistent parent request/spend allowance exhausted")
            db.execute(
                "INSERT OR IGNORE INTO parents VALUES (?,?,?,?,?)",
                (
                    budget.authorization_id,
                    fingerprint,
                    budget.maximum_requests,
                    str(parent_amount),
                    budget.expires_at,
                ),
            )
            db.execute(
                "INSERT INTO allocations VALUES (?,?,?,?,?)", (allocation_id, *wanted)
            )
        return {"allocation_id": allocation_id, "status": "reserved"}

    def retire(self, *, allocation_id, requests, maximum_spend_usd, proof_sha256):
        """Seal a stopped executor and retain conservative executed ceilings."""
        amount = Decimal(str(maximum_spend_usd))
        if (
            type(requests) is not int
            or requests < 0
            or not amount.is_finite()
            or amount < 0
            or not re.fullmatch(r"[0-9a-f]{64}", proof_sha256)
        ):
            raise ValueError(
                "Bounded executed ceilings and a collection proof are required"
            )
        wanted = (requests, str(amount), proof_sha256)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            allocation = db.execute(
                "SELECT requests,usd FROM allocations WHERE id=?", (allocation_id,)
            ).fetchone()
            if (
                not allocation
                or requests > allocation[0]
                or amount > Decimal(allocation[1])
            ):
                raise ValueError("Retirement cannot exceed its existing allocation")
            existing = db.execute(
                "SELECT requests,usd,proof_sha256 FROM retirements WHERE allocation=?",
                (allocation_id,),
            ).fetchone()
            if existing and existing != wanted:
                raise ValueError(
                    "A retired allocation cannot change its executed ceilings"
                )
            db.execute(
                "INSERT OR IGNORE INTO retirements VALUES (?,?,?,?)",
                (allocation_id, *wanted),
            )
        return {"allocation_id": allocation_id, "status": "retired"}


@lru_cache(maxsize=1)
def bounded_controller_class():
    """Narrower per-condition deadlines, preserving authoritative DB admission."""
    import threading
    from dataclasses import replace
    from simready_bench.engine.controller import Controller

    class BoundedController(Controller):
        call_limits = {"validation": 120, "preview": 90, "inference": 1800}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.call_deadline = threading.local()

        def _snapshot(self, db, row, *, deadline_at=None):
            bound = getattr(self.call_deadline, "deadline", None)
            if bound is not None:
                deadline_at = (
                    min(deadline_at, bound) if deadline_at is not None else bound
                )
            return super()._snapshot(db, row, deadline_at=deadline_at)

        def _reserve(self, episode_id, request_id, kind, snapshot=None):
            bound = time.time() + self.call_limits[kind]
            old = getattr(self.call_deadline, "deadline", None)
            self.call_deadline.deadline = bound
            try:
                job = super()._reserve(episode_id, request_id, kind, snapshot)
            finally:
                self.call_deadline.deadline = old
            deadline = min(job.deadline_at, bound)
            if deadline < job.deadline_at:
                with self._transaction() as db:
                    db.execute(
                        "UPDATE jobs SET deadline_at=MIN(deadline_at,?) WHERE job_id=?",
                        (deadline, job.job_id),
                    )
                job = replace(job, deadline_at=deadline)
            return job

    return BoundedController


def prepare(resolved, *, work_dir, policy):
    resolved = checked_resolved(resolved, policy)
    root = confined_work_dir(work_dir, ram_only=True)
    root.mkdir(parents=True, exist_ok=True, mode=448)
    output = root / "prepared"
    if output.exists():
        return check_prepared(output, resolved)
    return prepare_assets(resolved, work_dir=output, policy=policy)


def _provider_authorities(resolved):
    """Validate parent scopes before any condition is admitted or model called."""
    from simready_bench.engine.inference_gateway import ProjectBudget

    providers = resolved["operator_policy"]["providers"]
    totals = {}
    request_totals = {}
    for c in resolved["conditions"]:
        ref = c["model"]["provider_ref"]
        provider = providers[ref]
        if provider["kind"] in ("openai", "huggingface"):
            auth = ProjectBudget(**provider.get("project_budget", {}))
            if (
                not auth.approved
                or auth.expires_at <= time.time()
                or auth.model != c["model"]["model_id"]
                or (auth.project != "simready-bench")
            ):
                raise ValueError(
                    "Existing approved unexpired project/model authority required"
                )
            key = (ref, auth.authorization_id)
            totals[key] = totals.get(key, Decimal(0)) + Decimal(
                str(c["settings"]["budgets"]["maximum_spend_usd"])
            )
            request_totals[key] = (
                request_totals.get(key, 0)
                + c["settings"]["budgets"]["inference_requests"]
            )
            if (
                totals[key] > Decimal(str(auth.maximum_spend_usd))
                or request_totals[key] > auth.maximum_requests
            ):
                raise ValueError(
                    "Matrix partitions exceed approved parent provider request/spend allowance"
                )


def _gateway(condition, resolved, *, allocation_id=None, shared_route_throttle=None):
    from simready_bench.engine.inference_gateway import GatewayConfig, ProjectBudget

    model = condition["model"]
    budgets = condition["settings"]["budgets"]
    from simready_bench.engine.native_output import native_output

    native = native_output(condition)
    transport = (
        {
            "native_output_limits": True,
            "native_output_cap": False,
            "max_output_tokens": None,
            "strip_native_output_fields": condition["harness"]["adapter"] == "harbor",
        }
        if native
        else {}
    )
    provider = resolved["operator_policy"]["providers"][model["provider_ref"]]
    if provider.get("model_spec") is not None:
        transport["model_spec"] = provider["model_spec"]
    if provider.get("hf_router_relay") is not None:
        transport["hf_router_relay"] = provider["hf_router_relay"]
    if shared_route_throttle is not None:
        transport["shared_route_throttle"] = shared_route_throttle
    keyref = provider.get("api_key_env")
    key = os.environ.get(keyref, "") if keyref else ""
    projectref = provider.get("project_id_env")
    project = os.environ.get(projectref) if projectref else None
    if provider["kind"] in ("openai", "huggingface"):
        if not key:
            raise ValueError(
                "Configured existing credential ENV reference is unavailable"
            )
        parent = ProjectBudget(**provider["project_budget"])
        ceiling = budgets["maximum_spend_usd"]
        if ceiling <= 0:
            raise ValueError(
                "Default paid allowance is zero; explicit condition partition required"
            )
        auth = replace(
            parent,
            authorization_id=parent.authorization_id
            + "-bp-"
            + (allocation_id or condition["condition_id"])[-24:],
            maximum_requests=min(
                parent.maximum_requests, budgets["inference_requests"]
            ),
            maximum_spend_usd=ceiling,
        )
        if not native:
            transport.update(
                max_output_tokens=condition["settings"]
                .get("generation", {})
                .get("max_output_tokens", budgets["max_output_tokens"])
                if condition["harness"]["adapter"] == "harbor"
                else budgets["max_output_tokens"],
                native_output_cap=condition["harness"]["adapter"] == "harbor",
            )
        return GatewayConfig(
            provider["base_url"],
            model["model_id"],
            key,
            api_style=model["adapter"],
            budget=auth,
            **transport,
            request_timeout_seconds=min(budgets["agent_seconds"], 1800),
            upstream_project_id=project,
            max_input_tokens=budgets["max_input_tokens"],
        )
    raise ValueError("Use an OpenAI-compatible provider with configured pricing")


def _dimensions(public):
    match = re.search(
        "^Nominal dimensions in canonical metres XYZ: (.+)$",
        (public / "instruction.md").read_text(),
        re.MULTILINE,
    )
    if not match:
        raise ValueError("Original public nominal dimensions required")
    dims = json.loads(match.group(1))
    return dims


def export_submission(snapshot, output):
    """Independent bounded public-candidate artifact; no DB, token or reference."""
    from dataclasses import asdict, replace
    from simready_bench.engine.snapshots import verify_frozen_snapshot
    from simready_bench.harnesses import _relative

    verify_frozen_snapshot(snapshot)
    output = Path(output).absolute()
    if (
        output.exists()
        or output.is_symlink()
        or output.parent.resolve() != output.parent
    ):
        raise ValueError("Fresh nonsymlink submission export required")
    output.mkdir(mode=448)
    files = output / "files"
    files.mkdir(mode=448)
    for item in snapshot.files:
        relative = _relative(item.relative_path)
        source = Path(snapshot.path).joinpath(*relative.parts)
        target = files.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=448)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            copied = 0
            hashed = hashlib.sha256()
            while block := incoming.read(1024 * 1024):
                copied += len(block)
                if copied > item.size_bytes:
                    raise ValueError("Frozen candidate changed during export")
                hashed.update(block)
                outgoing.write(block)
        if (
            copied != item.size_bytes
            or hashed.hexdigest() != item.sha256
            or os.path.samefile(source, target)
        ):
            raise ValueError("Independent submission copy failed")
        target.chmod(292)
    verify_frozen_snapshot(snapshot)
    copied_snapshot = replace(snapshot, path=str(files))
    verify_frozen_snapshot(copied_snapshot)
    record = {
        "schema": "simready-bench-submission-v1",
        "entrypoint": snapshot.entrypoint,
        "snapshot_sha256": snapshot.sha256,
        "files": [asdict(f) for f in snapshot.files],
        "size_bytes": snapshot.size_bytes,
        "copied_independently": True,
    }
    write_json(output / "snapshot-manifest.json", record)
    return {
        "directory": str(files),
        "manifest": str(output / "snapshot-manifest.json"),
        "copied_independently": True,
        "snapshot_sha256": snapshot.sha256,
    }


class ConditionExecutionError(RuntimeError):
    """Known independent condition rejection; never an automatic retry signal."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _controller_images(task, runtime):
    images = [task.verifier.environment.docker_image]
    images.insert(0, task.environment.docker_image)
    return images


class LocalRuntimePreflightError(RuntimeError):
    """The local Docker runtime cannot execute the declared matrix."""


def _redacted_runtime_diagnostic(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value or "")
    for name, secret in os.environ.items():
        if len(secret) >= 4 and re.search(r"TOKEN|KEY|SECRET|PASSWORD", name, re.I):
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", text)
    text = re.sub(r"(?i)Bearer\s+[^\s]+", "Bearer [redacted]", text)
    return text[:4096]


def preflight_runtime_images(runtime, root):
    """Check declared Docker images once, before reserving condition authority."""
    runtime = runtime or {}
    profile = runtime.get("shared_profile", {})
    images = {}
    for role, image in (
        ("actor", profile.get("actor_image")),
        ("verifier", profile.get("verifier_image")),
        ("fidelity", runtime.get("fidelity_image")),
        ("illustration", runtime.get("illustration_image")),
    ):
        if image:
            images.setdefault(image, []).append(role)
    if not images:
        return
    directory = Path(root) / "private"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    diagnostic = directory / "runtime-preflight.json"
    checks = []
    for image, roles in images.items():
        command = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
        check = {"roles": roles, "image": image, "command": command}
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            check.update(returncode=result.returncode, available=result.returncode == 0,
                         stdout=_redacted_runtime_diagnostic(result.stdout),
                         stderr=_redacted_runtime_diagnostic(result.stderr))
        except (OSError, subprocess.TimeoutExpired) as exc:
            check.update(returncode=None, available=False, error_type=type(exc).__name__,
                         stderr=_redacted_runtime_diagnostic(getattr(exc, "stderr", None) or str(exc)))
        checks.append(check)
    write_json(diagnostic, {"schema": "simready-bench-local-runtime-preflight-v1", "checks": checks})
    unavailable = [role for check in checks if not check["available"] for role in check["roles"]]
    if unavailable:
        raise LocalRuntimePreflightError(
            "Local Docker runtime images are unavailable for " + ", ".join(unavailable)
            + ". Command, exit status and redacted output are retained in " + str(diagnostic)
        )


def _native_failure(job_result, *, normal_stop=False):
    """Keep provider and harness failures distinct from configured actor limits."""
    trials = getattr(job_result, "trial_results", None) or []
    if len(trials) != 1:
        return None
    info = getattr(trials[0], "exception_info", None)
    name = getattr(info, "exception_type", None)
    if not name:
        return None
    if normal_stop and name in {
        "NonZeroAgentExitCodeError",
        "AgentTimeoutError",
        "NativeActorStop",
    }:
        return None
    return {
        "error_type": name,
        "error_message": "Harbor " + name,
        "failure_scope": "infrastructure",
        "automatic_retry": False,
    }


def _controlled_final_stream_cancel(records, *, normal_stop):
    """Keep the frozen candidate score after one verified stop interrupts a stream."""
    if not normal_stop or not records:
        return False
    row = records[-1]
    return (
        row.get("status") == "cancelled"
        and sum(record.get("status") == "cancelled" for record in records) == 1
        and row.get("observed_upstream_status") == 200
        and row.get("upstream_status") in (None, 200)
        and row.get("error_code") is None
        and not (
            (
                set(row.get("provider_response_statuses", ()))
                | set(row.get("observed_provider_response_statuses", ()))
            )
            & {"failed", "cancelled"}
        )
    )


def reconstruction_status(evaluation, records, *, normal_stop=False, recovery=None):
    final_stream_cancel = _controlled_final_stream_cancel(
        [row for row in records if recovery is None or row.get("gateway_request_id") not in recovery["resolved_request_ids"]],
        normal_stop=normal_stop,
    )
    successful = False
    for index, row in enumerate(records):
        if row.get("status") == "completed" and row.get("upstream_status") == 200:
            if set(row.get("provider_response_statuses", ())) & {"failed", "cancelled"}:
                return "infrastructure_error"
            successful = True
        elif not (
            recovery is not None and row.get("gateway_request_id") in recovery["resolved_request_ids"]
            or
            final_stream_cancel and index == len(records) - 1
            or normal_stop
            and row.get("error_code")
            in {
                "episode_admission_closed",
                "inference_deadline_exceeded",
                "inference_budget_exhausted",
                "inference_attempt_budget_exhausted",
                "project_request_budget_exhausted",
                "project_attempt_budget_exhausted",
                "project_spend_budget_exhausted",
            }
        ):
            return "infrastructure_error"
    score = (evaluation or {}).get("simready", {}).get("score_percent")
    return (
        "completed"
        if successful and type(score) in (int, float) and math.isfinite(score)
        else "infrastructure_error"
    )


def complete_trial_result(result, job_result, *, normal_stop):
    from simready_bench.engine.recovery_evidence import (
        verified_request_recovery, native_terminal_failure, infrastructure_hold,
    )

    recovery = (
        verified_request_recovery(result, normal_stop=normal_stop)
        if len(getattr(job_result, "trial_results", None) or []) == 1 else None
    )
    status = reconstruction_status(
        result.get("evaluation"),
        result.get("gateway_records", []),
        normal_stop=normal_stop,
        recovery=recovery,
    )
    failure = _native_failure(job_result, normal_stop=normal_stop)
    if failure is not None:
        result.update(failure)
        status = "infrastructure_error"
    native_error = native_terminal_failure(result, normal_stop=normal_stop)
    if native_error is not None:
        result.update(error_type="NativeActorError", error_code=native_error["error_code"],
                      failure_scope="infrastructure", automatic_retry=False,
                      termination_reason="inference_failed")
        status = "infrastructure_error"
    hold = infrastructure_hold(result, normal_stop=normal_stop)
    if hold is not None:
        result.update(infrastructure_hold=hold["code"], infrastructure_hold_evidence=hold,
                      automatic_retry=False)
    result["status"] = status
    if recovery is not None and status == "completed":
        result["request_recovery"] = recovery
    if status != "completed":
        result.update(
            official_reward=None,
            diagnostic_reward=None,
            model_quality_evidence=False,
        )
    return result


def _condition_failure(condition, root, exc):
    result = {
        "schema": "simready-bench-condition-result-v1",
        "condition_id": condition["condition_id"],
        "asset_id": condition["asset_id"],
        "harness_id": condition["harness_id"],
        "model_id": condition["model_id"],
        "requested_model": condition.get("model", {}).get("model_id"),
        "skill_set_id": condition["skill_set_id"],
        "status": "condition_error"
        if isinstance(exc, ConditionExecutionError)
        else "infrastructure_error",
        "error_type": type(exc).__name__,
        "official_reward": None,
        "diagnostic_reward": None,
        "evaluation": None,
        "model_quality_evidence": False,
        "harbor_trace_paths": [
            str(path)
            for directory in (
                root,
                root.parent / "native-jobs"
                / ("simready-bench-" + condition["condition_id"][-12:]),
            )
            for path in directory.rglob("*")
            if path.is_file()
            and (path.suffix in (".jsonl", ".log") or path.name == "trajectory.json")
        ],
        "repetition": condition.get("repetition", 0),
        "seed": condition.get("seed"),
    }
    if isinstance(exc, ConditionExecutionError):
        result.update(
            error_code=exc.code,
            error_message=str(exc),
            failure_scope="condition",
            automatic_retry=False,
        )
    return result


def _retain_failed_submission(result, root, controller, bridge, submission):
    if len(bridge._episodes) != 1:
        return
    episode_id, environment = next(iter(bridge._episodes.items()))
    snapshot = environment.final_snapshot
    if not environment.actor_stopped or snapshot is None:
        return
    episode = controller.get_episode(episode_id)
    if episode.status != "closed":
        return
    result.update(
        trial_id=episode_id,
        termination_reason=episode.termination_reason,
        snapshot_sha256=snapshot.sha256,
        actor_stopped=True,
        actor_stop_audit=environment._actor_stop_audit,
        inference_calls_used=episode.inference_calls_used,
        validation_calls_used=episode.validation_calls_used,
        preview_calls_used=episode.preview_calls_used,
    )
    if bridge.completed_actor_stop(environment):
        result["actor_stop_receipt"] = asdict(bridge._controlled_receipts[episode_id])
    if snapshot.valid:
        (root / "artifacts").mkdir(mode=448, exist_ok=True)
        result["submission"] = submission or export_submission(
            snapshot, root / "artifacts/submission"
        )


async def _condition(condition, prepared, resolved, root, controller, *, shared_route_throttle=None):
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    from harbor.models.task.config import TaskConfig
    from simready_bench.engine.harbor import NativeLifecycle
    from simready_bench.engine.harbor_services import ReferenceTaskInput
    from simready_bench.engine.evaluation import Evaluator, CandidateObserver
    from simready_bench.engine.environment import validate_task_reference_binding

    runtime = runtime_policy(resolved, execute=True)
    budgets = condition["settings"]["budgets"]
    controller.call_limits = {
        "validation": budgets["validation_seconds"],
        "preview": budgets["preview_seconds"],
        "inference": min(budgets["agent_seconds"], 1800),
    }
    root.mkdir(mode=448)
    task = root / "task"
    shutil.copytree(prepared["task_path"], task)
    public = task / "environment/public"
    _dimensions(public)
    cfg = json.loads(Path(prepared["private_config"]).read_text())
    validate_task_reference_binding(
        cfg, allowed_reference_root=runtime["allowed_reference_root"]
    )
    from simready_bench.fidelity_control import prepare_reference, evaluate_final

    fidelity_reference = await asyncio.to_thread(
        prepare_reference, condition, cfg, runtime, root
    )
    tc = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    for image in _controller_images(tc, runtime):
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    skillroot = root / "skills"
    skilldirs = stage_skills(condition, skillroot)
    bridge = NativeLifecycle(
        controller,
        root / "actors",
        deadline_seconds=budgets["agent_seconds"],
        inference_limit=budgets["inference_requests"],
        preview_limit=budgets["preview_requests"],
        validation_limit=budgets["validation_calls"],
        tool_protocol=condition.get("tool_protocol", "legacy"),
    )
    cpu = CandidateObserver(
        validator_python=runtime["validator_python"],
        source_root=runtime["source_root"],
        state_dir=root / "private/cpu",
        expected_hostname=runtime["expected_hostname"],
        execution_enabled=True,
    )
    evaluator = Evaluator(
        root / "private/evaluation",
        observer=cpu,
        allowed_reference_root=runtime["allowed_reference_root"],
    )
    allocation_id = digest(
        {
            "work_root": str(root.parent),
            "configuration": resolved["configuration_sha256"],
            "condition": condition["condition_id"],
        }
    )
    services = native_services_class()(
        bridge,
        _gateway(condition, resolved, allocation_id=allocation_id, shared_route_throttle=shared_route_throttle),
        task_inputs=[
            ReferenceTaskInput(
                task,
                public / "thumbnail.png",
                (public / "instruction.md").read_text(),
                cfg,
                public,
            )
        ],
        evaluator=evaluator,
        verifier_timeout_seconds=budgets["final_seconds"],
        skill_root=skillroot if skilldirs else None,
        budgets=budgets,
        audit_dir=root / "private/container-audit",
        reference_root=runtime["allowed_reference_root"],
    )
    bridge.install()
    watch = None
    failed_result = None
    submission = None
    started = time.monotonic()

    async def guard():
        while True:
            root_low = shutil.disk_usage("/").free < 5 * 1024**3
            work_low = shutil.disk_usage(root).free < 1024**3
            if root_low or work_low:
                for eid in list(bridge._episodes):
                    await bridge.request_stop(eid, "infrastructure_disk_guard")
                raise RuntimeError("Five-GiB root reserve crossed" if root_low else "One-GiB work storage reserve crossed")
            for episode_id, environment in list(bridge._episodes.items()):
                verifier = services.verifiers.get(episode_id)
                if verifier is not None and (
                    not services.container_checks.get("verifier")
                ):
                    expected_source = str(verifier[0])
                    for path in (controller.state_dir / "native-environments").glob(
                        "*.actual-isolation.json"
                    ):
                        proof = json.loads(path.read_text())
                        if proof.get("role") == "verifier":
                            try:
                                await services.record_container_audit(
                                    environment,
                                    proof["container_id"],
                                    "verifier",
                                    expected_source=expected_source,
                                )
                            except (RuntimeError, ValueError):
                                pass
            await asyncio.sleep(0.25)

    try:
        await services.start()
        try:
            native_agent = agent_config(condition, bridge.bridge_id, skilldirs)
            if condition.get("tool_protocol", "legacy") != "legacy":
                if condition["harness"]["adapter"] != "pi":
                    raise ValueError("Bounded tool protocol requires the pinned Pi harness")
                native_agent.setdefault("kwargs", {})["tool_protocol"] = condition["tool_protocol"]
        except ValueError as exc:
            raise ConditionExecutionError(
                "native_agent_configuration_rejected",
                "Native agent configuration is incompatible with this condition; no inference was dispatched.",
            ) from exc
        jobconfig = {
            "job_name": "simready-bench-" + condition["condition_id"][-12:],
            "jobs_dir": str(root.parent / "native-jobs"),
            "n_concurrent_trials": 1,
            "n_attempts": 1,
            "retry": {"max_retries": 0},
            "environment": {
                "import_path": "simready_bench.engine.harbor:BenchmarkDockerEnvironment",
                "kwargs": {"simready_bridge_id": bridge.bridge_id},
            },
            "agents": [native_agent],
            "tasks": [{"path": str(task)}],
        }
        for name, value in runtime.get("container_resources", {}).items():
            jobconfig["environment"]["override_" + name] = value
        job = await Job.create(JobConfig.model_validate(jobconfig))
        bridge.attach(job)
        watch = asyncio.create_task(guard())
        async with asyncio.timeout(
            budgets["agent_seconds"] + budgets["final_seconds"] + 180
        ):
            native_result = await job.run()
        if watch.done():
            await watch
        if len(bridge._episodes) != 1:
            raise RuntimeError("Exactly one actual native Harbor trial required")
        episode_id, environment = next(iter(bridge._episodes.items()))
        episode = controller.get_episode(episode_id)
        if (
            not environment.actor_stopped
            or episode.status != "closed"
            or environment.final_snapshot is None
        ):
            raise RuntimeError("Native stop/freeze did not complete")
        (root / "artifacts").mkdir(mode=448, exist_ok=True)
        submission = (
            export_submission(environment.final_snapshot, root / "artifacts/submission")
            if environment.final_snapshot.valid
            else None
        )
        finalpath = (
            controller.state_dir / "final-service-reports" / (episode_id + ".json")
        )
        final = json.loads(finalpath.read_text()) if finalpath.is_file() else None
        records = list(services.transport.records(episode_id))
        if not any(
            (
                row.get("status") == "completed" and row.get("upstream_status") == 200
                for row in records
            )
        ):
            final = None
        measured = []
        for row in records:
            value = row.get("actual_cost_usd", row.get("cost_usd"))
            if type(value) in (int, float):
                measured.append(value)
        inference_cost = sum(measured) if len(measured) == len(records) else None
        result = {
            "schema": "simready-bench-condition-result-v1",
            "condition_id": condition["condition_id"],
            "asset_id": condition["asset_id"],
            "harness_id": condition["harness_id"],
            "model_id": condition["model_id"],
            "requested_model": condition.get("model", {}).get("model_id"),
            "skill_set_id": condition["skill_set_id"],
            "harbor_job_directory": str(job.job_dir),
            "trial_id": episode_id,
            "termination_reason": episode.termination_reason,
            "snapshot_sha256": environment.final_snapshot.sha256,
            "actor_stopped": environment.actor_stopped,
            "inference_calls_used": episode.inference_calls_used,
            "validation_calls_used": episode.validation_calls_used,
            "preview_calls_used": episode.preview_calls_used,
            "elapsed_seconds": time.monotonic() - started,
            "inference_cost_usd": inference_cost,
            "compute_cost_usd": None,
            "acceptance_preset": condition["settings"].get(
                "acceptance_preset", "nvidia-mandatory-95-5-v1"
            ),
            "evaluation": final,
            "gateway_records": records,
            "model_quality_evidence": True,
            "official_reward": final.get("official_reward") if final else None,
            "diagnostic_reward": final.get("diagnostic_reward") if final else None,
            "status": reconstruction_status(final, records),
            "effective_settings": condition["settings"],
            "effective_harness": condition["harness"],
            "effective_skills": condition["skill_set"],
            "submission": submission,
            "container_audits": dict(services.container_checks),
            "harbor_results": [
                str(p) for p in sorted(job.job_dir.rglob("result.json"))
            ],
            "harbor_trace_paths": [
                str(p)
                for p in sorted(job.job_dir.rglob("*"))
                if p.is_file()
                and (
                    p.suffix in (".jsonl", ".log")
                    or p.name in ("trajectory.json", "console.txt")
                )
            ],
            "repetition": condition.get("repetition", 0),
            "seed": condition.get("seed"),
        }
        result["fidelity"] = await asyncio.to_thread(
            evaluate_final,
            condition,
            cfg,
            runtime,
            environment.final_snapshot,
            root,
            fidelity_reference,
        )
        from simready_bench.captures import render_final

        result["illustrations"] = await asyncio.to_thread(
            render_final,
            condition,
            cfg,
            runtime,
            environment.final_snapshot,
            root,
            alignment=result["fidelity"].get("alignment"),
        )
        normal_stop = bridge.completed_actor_stop(environment)
        result["actor_stop_audit"] = environment._actor_stop_audit
        if normal_stop:
            result["actor_stop_receipt"] = asdict(
                bridge._controlled_receipts[episode_id]
            )
        complete_trial_result(result, native_result, normal_stop=normal_stop)
        write_json(root / "result.json", result)
        return result
    except BaseException as exc:
        failed_result = _condition_failure(condition, root, exc)
        if isinstance(exc, Exception):
            return failed_result
        raise
    finally:
        if watch is not None:
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)
        for eid, env in list(bridge._episodes.items()):
            if not env.actor_stopped or env.final_snapshot is None:
                try:
                    await bridge.request_stop(eid, "cancelled")
                except Exception:
                    pass
        if failed_result is not None:
            try:
                _retain_failed_submission(
                    failed_result, root, controller, bridge, submission
                )
            except Exception as exc:
                failed_result["candidate_collection_error_type"] = type(exc).__name__
        try:
            await services.close()
        except Exception as exc:
            if failed_result is None:
                raise
            failed_result["cleanup_error_type"] = type(exc).__name__
        finally:
            bridge.uninstall()
            if failed_result is not None:
                write_json(root / "result.json", failed_result)


def _write_progress(root, resolved, results, allowance):
    record = {
        "schema": "simready-bench-matrix-result-v1",
        "configuration_sha256": resolved["configuration_sha256"],
        "planned_conditions": len(resolved["conditions"]),
        "completed_conditions": sum((r.get("status") == "completed" for r in results)),
        "attempted_conditions": len(results),
        "results": results,
        "allowance_ledger": allowance,
        "partial": True,
    }
    temporary = root / "progress.json.tmp"
    with temporary.open("w") as stream:
        json.dump(record, stream, sort_keys=True, allow_nan=False)
    temporary.chmod(384)
    temporary.replace(root / "progress.json")
    return record


_worker_stop = None


def _initialise_condition_worker(stop):
    global _worker_stop
    _worker_stop = stop


def _condition_process(condition, prepared, resolved, root, controller_root, throttle):
    """Keep Harbor's native bridge hooks within one condition process."""
    async def execute():
        if _worker_stop is not None and _worker_stop.is_set():
            raise RuntimeError("The local coordinator stopped before condition execution")
        controller = bounded_controller_class()(controller_root)
        options = {"shared_route_throttle": throttle} if throttle is not None else {}
        task = asyncio.create_task(_condition(condition, prepared, resolved, root, controller, **options))

        async def watch_stop():
            while True:
                if _worker_stop is not None and _worker_stop.is_set():
                    task.cancel()
                    return
                await asyncio.sleep(0.25)

        watcher = asyncio.create_task(watch_stop())
        try:
            return await task
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)

    return asyncio.run(execute())


async def _condition_results(resolved, prepared, root, controller, ledger, workers, throttles):
    """Admit at most one condition per worker and retain each completed outcome."""
    conditions = iter(enumerate(resolved["conditions"]))

    def admit(index, condition):
        if shutil.disk_usage(root).free < 1024**3:
            raise RuntimeError("At least 1 GiB of free work storage is required before admitting another condition")
        admission = ledger.reserve(condition["condition_id"], condition["settings"]["budgets"]["maximum_spend_usd"])
        if admission["status"] == "completed":
            return admission["result"]
        if admission["status"] != "new":
            raise RuntimeError("Previously admitted incomplete condition requires operator investigation; no automatic paid retry")
        return None

    if workers == 1:
        for index, condition in conditions:
            result = admit(index, condition)
            cached = result is not None
            if result is None:
                throttle = throttles.get((condition["model"]["provider_ref"], condition["model"]["model_id"]))
                options = {"shared_route_throttle": throttle} if throttle is not None else {}
                try:
                    result = await _condition(condition, prepared[condition["condition_id"]], resolved, root / ("c" + str(index)), controller, **options)
                except Exception as exc:
                    result = _condition_failure(condition, root / ("c" + str(index)), exc)
            yield index, condition, result, cached
        return

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    pool = ProcessPoolExecutor(max_workers=min(workers, len(resolved["conditions"])), mp_context=context, initializer=_initialise_condition_worker, initargs=(stop,))
    loop = asyncio.get_running_loop()
    pending = {}

    def submit_one():
        try:
            index, condition = next(conditions)
        except StopIteration:
            return
        result = admit(index, condition)
        if result is None:
            future = loop.run_in_executor(pool, _condition_process, condition, prepared[condition["condition_id"]], resolved, root / ("c" + str(index)), root / "controller", throttles.get((condition["model"]["provider_ref"], condition["model"]["model_id"])))
        else:
            future = loop.create_future()
            future.set_result(result)
        pending[future] = index, condition, result is not None

    try:
        for _ in range(min(workers, len(resolved["conditions"]))):
            submit_one()
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for future in sorted(done, key=lambda item: pending[item][0]):
                index, condition, cached = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _condition_failure(condition, root / ("c" + str(index)), exc)
                yield index, condition, result, cached
                submit_one()
    finally:
        stop.set()
        for future in pending:
            future.cancel()
        # Child cancellation runs the native actor stop and service cleanup
        # before the route throttle or work tree can be released.
        await asyncio.to_thread(pool.shutdown, wait=True, cancel_futures=True)


async def _run(resolved, *, work_dir, policy, on_progress=None, _ownership=None, workers=1):
    resolved = checked_resolved(resolved, policy)
    runtime = runtime_policy(resolved, execute=True)
    _provider_authorities(resolved)
    root = confined_work_dir(work_dir, fresh=True, ram_only=True)
    root.mkdir(mode=448, parents=True)
    if _ownership is not None:
        _ownership["work_root"] = str(root)
    if len(str(root)) + 65 > 106:
        raise ValueError("Short RAM work directory required for private Unix sockets")
    preflight_runtime_images(runtime, root)
    authorized = [
        c
        for c in resolved["conditions"]
        if c["model"]["adapter"] in ("openai-responses", "openai-completions")
    ]
    if authorized:
        authority = resolved["operator_policy"]["runtime"].get("authority_state_dir")
        if not authority:
            raise ValueError(
                "Provider execution requires persistent trusted authority_state_dir outside work directories"
            )
        state = Path(authority).absolute()
        forbidden = [
            root,
            *[Path(p) for p in resolved["operator_policy"]["roots"].values()],
        ]
        if any(
            (
                state == p or state.is_relative_to(p) or p.is_relative_to(state)
                for p in forbidden
            )
        ):
            raise ValueError(
                "Authority state must remain separate from work and input/public roots"
            )
        parent_ledger = ParentAuthorityLedger(state)
        from simready_bench.engine.inference_gateway import ProjectBudget

        for c in authorized:
            provider = resolved["operator_policy"]["providers"][
                c["model"]["provider_ref"]
            ]
            allocation = digest(
                {
                    "work_root": str(root),
                    "configuration": resolved["configuration_sha256"],
                    "condition": c["condition_id"],
                }
            )
            if provider["kind"] in ("openai", "huggingface"):
                parent = ProjectBudget(**provider["project_budget"])
                parent_ledger.reserve(
                    parent,
                    allocation_id=allocation,
                    requests=c["settings"]["budgets"]["inference_requests"],
                    maximum_spend_usd=c["settings"]["budgets"]["maximum_spend_usd"],
                    work_root=root,
                )
    old_tmp = os.environ.get("TMPDIR")
    old_temp = tempfile.tempdir
    (root / "tmp").mkdir(mode=448, exist_ok=True)
    os.environ["TMPDIR"] = str(root / "tmp")
    tempfile.tempdir = str(root / "tmp")
    try:
        record = prepare(resolved, work_dir=root, policy=policy)
        prepared = {c["condition_id"]: c for c in record["conditions"]}
        ledger = MatrixLedger(
            root / "matrix.sqlite3",
            resolved["configuration_sha256"],
            resolved["aggregate_budget"]["maximum_spend_usd"],
        )
        controller = bounded_controller_class()(root / "controller")
        results = []
        from simready_bench.local_route_throttle import local_route_throttles

        with local_route_throttles(resolved) as throttles:
            outcomes = _condition_results(resolved, prepared, root, controller, ledger, workers, throttles)
            try:
                order = {condition["condition_id"]: index for index, condition in enumerate(resolved["conditions"])}
                async for index, c, result, cached in outcomes:
                    if not cached:
                        ledger.complete(c["condition_id"], result)
                    results.append(result)
                    results.sort(key=lambda item: order[item["condition_id"]])
                    progress = _write_progress(root, resolved, results, ledger.records())
                    if on_progress is not None:
                        on_progress(root, progress)
            finally:
                await outcomes.aclose()
        summary = {
            "schema": "simready-bench-matrix-result-v1",
            "configuration_sha256": resolved["configuration_sha256"],
            "harbor_jobs_directory": str(root / "native-jobs"),
            "harbor_viewer_command": __import__("shlex").join(
                ["harbor", "view", str(root / "native-jobs")]
            ),
            "planned_conditions": len(resolved["conditions"]),
            "completed_conditions": sum(
                (r.get("status") == "completed" for r in results)
            ),
            "attempted_conditions": len(results),
            "results": results,
            "allowance_ledger": ledger.records(),
            "official_success_count": sum(
                (r.get("official_reward") == 1 for r in results)
            ),
            "official_reward_unknown_count": sum(
                (r.get("official_reward") is None for r in results)
            ),
        }
        write_json(root / "results.json", summary)
        return summary
    finally:
        tempfile.tempdir = old_temp
        if old_tmp is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmp


class RunExecutionError(RuntimeError):
    """A failed invocation with reviewable real progress and explicit unrun rows."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def _blocked_result(resolved, work_dir, exc):
    from simready_bench.reporting import read_results

    try:
        saved = read_results(work_dir)
        previous = (
            saved.get("results", [])
            if saved.get("configuration_sha256") == resolved["configuration_sha256"]
            else []
        )
    except (ValueError, OSError, KeyError, TypeError):
        previous = []
    reason = (
        "infrastructure_disk_guard"
        if "GiB" in str(exc) or "root reserve" in str(exc)
        else "runtime_execution_not_enabled"
        if "execution_enabled" in str(exc)
        else "execution_host_not_bound"
        if "host differs" in str(exc)
        else "infrastructure_preflight_or_run_failure"
    )
    by_id = {row.get("condition_id"): row for row in previous}
    rows = []
    for condition in resolved["conditions"]:
        old = by_id.get(condition["condition_id"])
        rows.append(
            old
            if old is not None
            else {
                "schema": "simready-bench-condition-result-v1",
                "condition_id": condition["condition_id"],
                "asset_id": condition["asset_id"],
                "harness_id": condition["harness_id"],
                "model_id": condition["model_id"],
                "requested_model": condition.get("model", {}).get("model_id"),
                "skill_set_id": condition["skill_set_id"],
                "status": "blocked",
                "blocker": reason,
                "error_type": type(exc).__name__,
                "trial_id": None,
                "inference_calls_used": None,
                "validation_calls_used": None,
                "preview_calls_used": None,
                "official_reward": None,
                "diagnostic_reward": None,
                "inference_cost_usd": None,
                "model_quality_evidence": False,
                "evaluation": None,
                "acceptance_preset": condition["settings"].get(
                    "acceptance_preset", "nvidia-mandatory-95-5-v1"
                ),
                "repetition": condition.get("repetition", 0),
                "seed": condition.get("seed"),
            }
        )
        if old is None and isinstance(exc, LocalRuntimePreflightError):
            rows[-1].update(error_code="local_runtime_image_unavailable", error_message=str(exc))
    return {
        "schema": "simready-bench-matrix-result-v1",
        "configuration_sha256": resolved["configuration_sha256"],
        "planned_conditions": len(rows),
        "completed_conditions": sum((r.get("status") == "completed" for r in rows)),
        "attempted_conditions": sum((r.get("status") != "blocked" for r in rows)),
        "blocked_conditions": sum((r.get("status") == "blocked" for r in rows)),
        "results": rows,
        "status": "blocked" if not previous else "interrupted",
        "blocker": reason,
        "partial": True,
        "official_success_count": sum((r.get("official_reward") == 1 for r in rows)),
        "official_reward_unknown_count": sum(
            (r.get("official_reward") is None for r in rows)
        ),
    }


def _retire_unrun_local(resolved, result, ownership):
    """Release only unused ceilings after this invocation has left its run loop."""
    from simready_bench.resume import _unrun

    if not ownership.get("work_root"):
        return
    root = Path(ownership["work_root"])
    conditions = {
        condition["condition_id"]: condition for condition in resolved["conditions"]
    }
    rows = {row["condition_id"]: row for row in result.get("results", [])}
    if len(rows) != len(result.get("results", [])) or set(rows) != set(conditions):
        raise ValueError("Stopped local results must identify every resolved condition")
    unused = {identifier for identifier, row in rows.items() if _unrun(row)}
    if not unused:
        return
    matrix = root / "matrix.sqlite3"
    matrix_sha256 = None
    if matrix.exists():
        if (
            matrix.resolve() != matrix
            or not matrix.is_file()
            or matrix.stat().st_nlink != 1
        ):
            raise ValueError("The stopped local admission ledger must be a regular file")
        with sqlite3.connect(matrix.as_uri() + "?mode=ro", uri=True) as db:
            binding = db.execute("SELECT identity,maximum FROM binding").fetchall()
            admitted = {row[0] for row in db.execute("SELECT id FROM conditions")}
        expected_binding = (
            resolved["configuration_sha256"],
            str(Decimal(str(resolved["aggregate_budget"]["maximum_spend_usd"]))),
        )
        if binding != [expected_binding] or not admitted <= set(conditions):
            raise ValueError(
                "The stopped local admission ledger differs from its configuration"
            )
        unused -= admitted
        matrix_sha256 = hashlib.sha256(matrix.read_bytes()).hexdigest()
    if not unused:
        return
    operator = resolved["operator_policy"]
    state = operator["runtime"].get("authority_state_dir")
    if not state:
        return
    authority = ParentAuthorityLedger(state)
    for identifier in sorted(unused):
        condition = conditions[identifier]
        provider = operator["providers"][condition["model"]["provider_ref"]]
        if provider["kind"] not in ("openai", "huggingface"):
            continue
        allocation = digest(
            {
                "work_root": str(root),
                "configuration": resolved["configuration_sha256"],
                "condition": identifier,
            }
        )
        with authority._db() as db:
            held = db.execute(
                "SELECT parent,requests,usd,work_root FROM allocations WHERE id=?",
                (allocation,),
            ).fetchone()
        if held is None:
            continue
        limits = condition["settings"]["budgets"]
        if held != (
            provider["project_budget"]["authorization_id"],
            limits["inference_requests"],
            str(Decimal(str(limits["maximum_spend_usd"]))),
            str(root),
        ):
            raise ValueError(
                "An unused local ceiling differs from its original allocation"
            )
        proof = {
            "schema": "simready-bench-unused-local-allocation-v1",
            "allocation_id": allocation,
            "configuration_sha256": resolved["configuration_sha256"],
            "condition_id": identifier,
            "work_root": str(root),
            "run_loop_stopped": True,
            "condition_unadmitted": True,
            "matrix_sha256": matrix_sha256,
        }
        receipts = authority.root / "unused-local-allocations"
        receipts.mkdir(mode=0o700, exist_ok=True)
        if receipts.resolve() != receipts or not receipts.is_dir():
            raise ValueError("Local allocation proofs must remain in the private ledger")
        receipt = receipts / (allocation + ".json")
        if receipt.exists():
            if receipt.resolve() != receipt or json.loads(receipt.read_text()) != proof:
                raise ValueError("An unused local allocation proof cannot change")
        else:
            write_json(receipt, proof)
        authority.retire(
            allocation_id=allocation,
            requests=0,
            maximum_spend_usd=0,
            proof_sha256=digest(proof),
        )


def run(resolved, *, work_dir, policy, on_progress=None, workers=1):
    from simready_bench.parallel import validate_workers
    validate_workers(workers)
    ownership = {}
    try:
        result = asyncio.run(
            _run(
                resolved,
                work_dir=work_dir,
                policy=policy,
                on_progress=on_progress,
                _ownership=ownership,
                workers=workers,
            )
        )
    except BaseException as exc:
        result = _blocked_result(resolved, work_dir, exc)
        _retire_unrun_local(resolved, result, ownership)
        if not isinstance(exc, Exception):
            raise
        raise RunExecutionError(str(exc), result) from exc
    _retire_unrun_local(resolved, result, ownership)
    return result
