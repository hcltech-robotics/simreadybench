"""Execute and collect a complete reconstruction benchmark on Daytona."""

from __future__ import annotations

import base64
import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import sys
import tarfile
import time
import zipfile
from collections import Counter
from decimal import Decimal
from importlib import metadata
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .runtime_images import CPU_IMAGE, CPU_SNAPSHOT, DEFAULT_IMAGES

EXECUTOR_HEALTH_SCHEMA = "simready-bench-executor-health-v1"
ORPHAN_STOP_MINUTES = 10
EXECUTOR_STALL_SECONDS = 3900


def _health_policy(policy):
    seconds = policy.get("runtime", {}).get("executor_stall_seconds", EXECUTOR_STALL_SECONDS)
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 1800 <= seconds <= 86400:
        raise ValueError("Executor stall seconds must be between 1800 and 86400")
    return {"schema": EXECUTOR_HEALTH_SCHEMA, "stall_seconds": seconds, "orphan_stop_minutes": ORPHAN_STOP_MINUTES}


def _executor_activity(sandbox):
    """Read provider admissions from the relay rather than actor-controlled logs."""
    raw = sandbox.fs.download_file("/run/simready-benchmark/ledger.json", 10)
    if len(raw) > 32 * 1024**2:
        raise ValueError("The executor activity ledger exceeds its size limit")
    value = json.loads(raw)
    counts = [value.get(name, 0) for name in ("generation_requests", "input_counter_requests")]
    if any(type(number) is not int or number < 0 for number in counts):
        raise ValueError("The executor activity counters are invalid")
    return counts


def _stop_stalled_executor(sandbox, record, output, *, reason, save, exit_code=137):
    """Stop the owned executor before collecting a stalled attempt and retrying."""
    _executor_command(
        sandbox,
        'set -eu; docker ps -q | xargs -r docker kill >/dev/null; test -z "$(docker ps -q)"; echo simready-containers-stopped',
        clean=str, timeout=60,
    )
    proof = {"schema": "simready-bench-executor-terminal-v1", "sandbox_id": record["sandbox_id"],
             "run_id": record["run_id"], "observed_at": time.time(), "controller_exit": exit_code,
             "reason": reason, "all_containers_stopped": True}
    proof_path = Path(output) / "executor-terminal-proof.json"
    _json(proof_path, proof)
    record.update(controller_exit=exit_code, controller_status="failed", terminal_confirmed=True,
                  termination_reason=reason, executor_terminal_proof_sha256=hashlib.sha256(proof_path.read_bytes()).hexdigest())
    save()


def _actor_resources(value=None):
    """Retain explicit actor bounds across the executor boundary."""
    if value is None:
        return {"cpus": 2, "memory_mb": 4096}
    if (
        not isinstance(value, dict)
        or set(value) != {"cpus", "memory_mb"}
        or type(value["cpus"]) is not int
        or not 1 <= value["cpus"] <= 64
        or type(value["memory_mb"]) is not int
        or not 128 <= value["memory_mb"] <= 1048576
    ):
        raise ValueError("Actor resources require explicit CPU and memory MiB bounds")
    return dict(value)


def _json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def build_wheel(directory):
    """Package the installed evaluator and its declared runtime dependencies."""
    import simready_bench

    distribution = metadata.distribution("simready-bench")
    version = distribution.version
    if version != simready_bench.__version__:
        raise ValueError(
            "Reinstall SimReadyBench before packaging the current executor"
        )
    if not re.fullmatch(r"[A-Za-z0-9.]+", version):
        raise ValueError("A literal package version is required")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    wheel = destination / ("simready_bench-" + version + "-py3-none-any.whl")
    if wheel.exists():
        raise FileExistsError("Preserving an existing executor wheel")
    records = []
    size = 0
    with zipfile.ZipFile(wheel, "x", compression=zipfile.ZIP_DEFLATED) as archive:

        def add(name, raw):
            nonlocal size
            size += len(raw)
            if size > 1024**3 or len(records) >= 10000:
                raise ValueError("Executor package exceeds its size limit")
            archive.writestr(name, raw)
            records.append(
                (
                    name,
                    "sha256="
                    + base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
                    .rstrip(b"=")
                    .decode(),
                    len(raw),
                )
            )

        for module in (simready_bench,):
            root = Path(module.__file__).absolute().parent
            if root.resolve() != root:
                raise ValueError("Executor package cannot contain symbolic links")
            for path in sorted(root.rglob("*")):
                if "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                    raise ValueError("Executor package must contain regular files")
                if path.is_file():
                    add(
                        root.name + "/" + path.relative_to(root).as_posix(),
                        path.read_bytes(),
                    )
        info = "simready_bench-" + version + ".dist-info/"
        metadata_text = distribution.read_text("METADATA") or distribution.read_text(
            "PKG-INFO"
        )
        if metadata_text is None:
            raise ValueError("Installed distribution metadata is required")
        add(info + "METADATA", metadata_text.encode("utf-8"))
        add(
            info + "WHEEL",
            b"Wheel-Version: 1.0\nGenerator: SimReadyBench\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        entrypoints = distribution.read_text("entry_points.txt")
        if entrypoints:
            add(info + "entry_points.txt", entrypoints.encode("utf-8"))
        record = io.StringIO(newline="")
        writer = csv.writer(record)
        writer.writerows([*records, (info + "RECORD", "", "")])
        archive.writestr(info + "RECORD", record.getvalue())
    wheel.chmod(0o600)
    return wheel


def _check_wheel(path):
    path = Path(path).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or not re.fullmatch(r"simready_bench-[A-Za-z0-9.]+-py3-none-any.whl", path.name)
    ):
        raise ValueError("A SimReadyBench wheel is required")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt executor wheel")
        for module in (
            "benchmark.py",
            "daytona.py",
            "provider_relay.py",
            "engine/scoring.py",
        ):
            if "simready_bench/" + module not in archive.namelist():
                raise ValueError(
                    "Executor wheel does not include the benchmark command"
                )
    return path


def _archive_inputs(roots, destination):
    total = 0
    count = 0
    with tarfile.open(destination, "w:gz") as archive:
        for name, raw in roots.items():
            root = Path(raw).absolute()
            if root.resolve() != root or not root.is_dir():
                raise ValueError("Staged input directories must be regular directories")
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or (
                    not path.is_dir()
                    and (not path.is_file() or path.stat().st_nlink != 1)
                ):
                    raise ValueError(
                        "Staged inputs cannot contain links or special files"
                    )
                if not path.is_file():
                    continue
                total += path.stat().st_size
                count += 1
                if total > 1024**3 or count > 10000:
                    raise ValueError(
                        "Staged input package exceeds 10000 files or 1 GiB"
                    )
                archive.add(
                    path,
                    arcname=name + "/" + path.relative_to(root).as_posix(),
                    recursive=False,
                )
    Path(destination).chmod(0o600)


def _extract_inputs(archive, destination):
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0]
                not in ("assets", "skills", "input_views", "private_manifests")
                or not member.isfile()
            ):
                raise ValueError("Invalid staged input archive")
            target = Path(destination).joinpath(*name.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o600)


def _extract_results(archive, destination):
    destination = Path(destination).absolute()
    count = size = 0
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != "work"
            ):
                raise ValueError("Invalid result archive path")
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    "Result archives cannot contain links or special files"
                )
            count += 1
            size += member.size
            if count > 100000 or size > 20 * 1024**3:
                raise ValueError("Result archive exceeds its size limit")
            target = destination.joinpath(*name.parts[1:])
            if target.resolve() != target or not target.is_relative_to(destination):
                raise ValueError("Result archive escaped its output directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with stream.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o600)


def _plan(manifest, policy):
    models = manifest.get("models", [])
    if len(models) != 1 or models[0].get("adapter") not in (
        "openai-responses",
        "openai-completions",
    ):
        raise ValueError("Choose one model for the Daytona benchmark command")
    provider = copy.deepcopy(policy["providers"][models[0]["provider_ref"]])
    if provider.get("kind") not in ("openai", "huggingface"):
        raise ValueError(
            "Daytona deployment requires an OpenAI-compatible model provider"
        )
    from simready_bench.engine.inference_gateway import ProjectBudget

    budget = ProjectBudget(**provider["project_budget"])
    conditions = (
        sum(
            len(
                manifest.get("reconstruction_selection", {}).get(
                    asset["id"], range(manifest["repetitions"])
                )
            )
            for asset in manifest["assets"]
        )
        * len(manifest["harnesses"])
        * len(manifest["skill_sets"])
    )
    settings = manifest["defaults"]["budgets"]
    requests = conditions * settings["inference_requests"]
    spend = Decimal(str(settings["maximum_spend_usd"])) * conditions
    if (
        not conditions
        or conditions > policy["limits"]["maximum_matrix_runs"]
        or requests > budget.maximum_requests
        or spend > Decimal(str(budget.maximum_spend_usd))
        or spend > Decimal(str(policy["limits"]["maximum_spend_usd"]))
    ):
        raise ValueError("The requested matrix exceeds its parent allowance")
    if (
        budget.model != models[0]["model_id"]
        or budget.expires_at <= time.time()
        or not budget.approved
        or budget.pricing_status != "known"
    ):
        raise ValueError("An approved, unexpired model allowance is required")
    if provider["base_url"].rstrip("/") != budget.upstream_origin.rstrip("/") + "/v1":
        raise ValueError("The provider endpoint differs from its parent allowance")
    return provider, budget, requests, float(spend)


def _collect_portable(portable, output):
    """Retain exported attempts even when aggregation stopped before its report."""

    def promote(source, target):
        if not target.exists():
            source.rename(target)
            return
        if target.resolve() != target or source.is_dir() != target.is_dir():
            raise FileExistsError("Preserving existing report output: " + str(target))
        if source.is_dir():
            for child in source.iterdir():
                promote(child, target / child.name)
            source.rmdir()
        elif (
            hashlib.sha256(source.read_bytes()).digest()
            == hashlib.sha256(target.read_bytes()).digest()
        ):
            source.unlink()
        else:
            raise FileExistsError("Preserving existing report output: " + str(target))

    if not portable.is_dir():
        return False
    for child in portable.iterdir():
        if child.name not in (
            "report.html",
            "results.json",
            "scores.csv",
            "evidence",
            "candidates",
            "run-details.json",
            "attempts",
            "reconstructions",
            "telemetry",
            "accounting",
        ):
            continue
        target = output / child.name
        if child.name == "reconstructions" and target.exists():
            if target.resolve() != target or not target.is_dir():
                raise ValueError(
                    "A reconstruction collection must remain an owned directory"
                )
            for member in child.iterdir():
                destination = target / member.name
                if member.name in {"report.html", "results.json", "scores.csv"}:
                    if (
                        destination.is_symlink()
                        or destination.exists()
                        and not destination.is_file()
                    ):
                        raise ValueError(
                            "A reconstruction index must remain a regular file"
                        )
                    member.replace(destination)
                else:
                    promote(member, destination)
            child.rmdir()
        else:
            promote(child, target)
    return (output / "report.html").is_file()


def remote_provider(provider, *, port, allocation_id, requests, maximum_spend_usd):
    """Bind the controller to a local relay without extending its authority."""
    result = copy.deepcopy(provider)
    origin = "http://127.0.0.1:" + str(port)
    result.pop("hf_router_relay", None)
    if provider["base_url"].rstrip("/") == "https://router.huggingface.co/v1":
        result["hf_router_relay"] = {
            "schema": "simready-hf-router-relay/v1",
            "model": provider["project_budget"]["model"],
            "upstream_base_url": "https://router.huggingface.co/v1",
            "relay_base_url": origin + "/v1",
        }
    result.update(
        kind="openai", base_url=origin + "/v1", api_key_env="SIMREADY_RELAY_KEY"
    )
    result["project_budget"].update(
        authorization_id="cloud-" + allocation_id,
        upstream_origin=origin,
        maximum_requests=requests,
        maximum_spend_usd=maximum_spend_usd,
    )
    if "model_spec" in result:
        result["model_spec"]["base_url"] = origin + "/v1"
    elif provider["kind"] == "huggingface":
        from simready_bench.engine.hf_pricing import (
            HF_CATALOG_URL,
            HF_PRICING,
            HF_PRICING_CHECKED,
        )

        model = result["project_budget"]["model"]
        price = HF_PRICING[model]
        result["model_spec"] = {
            "schema": "simready-operator-model-v1",
            "model": model,
            "base_url": origin + "/v1",
            "api_style": "openai-completions",
            "context_window": price.context_length,
            "maximum_output_tokens": None,
            "pricing": {
                "input_per_million_usd": price.input_per_million_usd,
                "output_per_million_usd": price.output_per_million_usd,
            },
            "pricing_source": HF_CATALOG_URL,
            "checked": HF_PRICING_CHECKED,
        }
    return result


def _matrix_complete(result, manifest):
    per_asset = (
        manifest["repetitions"]
        * len(manifest["models"])
        * len(manifest["harnesses"])
        * len(manifest["skill_sets"])
    )
    selection = manifest.get("reconstruction_selection", {})
    expected = {
        asset["id"]: len(selection.get(asset["id"], range(manifest["repetitions"])))
        * per_asset
        // manifest["repetitions"]
        for asset in manifest["assets"]
    }
    expected = {asset: count for asset, count in expected.items() if count}
    combinations = (
        len(manifest["models"])
        * len(manifest["harnesses"])
        * len(manifest["skill_sets"])
    )
    identities = Counter(
        {
            (asset["id"], repetition): combinations
            for asset in manifest["assets"]
            for repetition in selection.get(asset["id"], range(manifest["repetitions"]))
        }
    )
    rows = result.get("results", [])
    planned = sum(expected.values())

    def measured(row):
        score = row.get("simready_reward")
        if score is None:
            score = (
                (row.get("evaluation") or {}).get("simready", {}).get("score_percent")
            )
        return (
            type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 100
        )

    return (
        len(rows) == planned
        and result.get("planned_conditions") == planned
        and result.get("completed_conditions") == planned
        and not result.get("partial", False)
        and all(row.get("status") == "completed" and measured(row) for row in rows)
        and Counter(row.get("asset_id") for row in rows) == expected
        and all(type(row.get("repetition")) is int for row in rows)
        and Counter((row.get("asset_id"), row.get("repetition")) for row in rows)
        == identities
    )


def _bootstrap_command(log_file="/var/log/simready-dockerd.log"):
    """Preserve image entry-point loading before the public socket is ready."""
    return shlex.join([
        "sh", "-c",
        "if ! pidof dockerd >/dev/null 2>&1 && ! pidof dockerd-entrypoint.sh >/dev/null 2>&1; then "
        "dockerd-entrypoint.sh dockerd >" + shlex.quote(str(log_file)) + " 2>&1 </dev/null & fi",
    ])


def _relay_command(cpu_image=CPU_IMAGE):
    argv = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        "simready-model-relay",
        "--network",
        "host",
        "--user",
        "0:0",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--cpus",
        "1",
        "--memory",
        "256m",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=33554432,mode=1777",
    ]
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        argv += ["--env", name]
    return [
        *argv,
        "--env",
        "SSL_CERT_FILE=/run/host-ca.pem",
        "--mount",
        "type=bind,src=/run/simready-benchmark,dst=/run/simready-benchmark",
        "--mount",
        "type=bind,src=/etc/ssl/cert.pem,dst=/run/host-ca.pem,readonly",
        "--entrypoint",
        "python",
        cpu_image,
        "-B",
        "/run/simready-benchmark/relay.py",
        "/run/simready-benchmark/provider.json",
    ]


def execute_remote(path):
    """Installed controller entry point within the owned Daytona executor."""
    from simready_bench.benchmark import execute_local_sequential
    from simready_bench.controller_runner import ControllerReporter, Telemetry
    from simready_bench.policy import init_policy

    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    actor_resources = _actor_resources(envelope.get("container_resources"))
    root = Path("/work")
    output = root / "output"
    output.mkdir(mode=0o700)
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    _extract_inputs(root / "inputs.tar.gz", root)
    authority = root / "authority"
    authority.mkdir(mode=0o700)
    from simready_bench.local import ParentAuthorityLedger

    # A source download may fail before the first authoring admission. Its
    # empty native ledger is still collected as an authoritative zero scope.
    ParentAuthorityLedger(authority)
    os.environ["SIMREADY_RELAY_KEY"] = envelope["marker"]
    provider = envelope["provider"]
    if provider.get("project_id_env") and envelope.get("project_id"):
        os.environ[provider["project_id_env"]] = envelope["project_id"]
    policy = init_policy(
        output=root / "remote-operator.json",
        workspace=root / "operator",
        assets=root / "assets",
        actor_image=envelope["actor_image"],
        verifier_image=envelope["verifier_image"],
        providers={envelope["provider_ref"]: provider},
        maximum_spend_usd=envelope["maximum_spend_usd"],
        execution_enabled=True,
        native_output=True,
        authority_state_dir=authority,
    )
    policy["limits"].update(envelope["limits"])
    policy["runtime"].update(
        container_resources=actor_resources,
        illustration_image=envelope["cpu_image"],
        fidelity_image=envelope["cpu_image"],
    )
    policy["roots"].update(
        {
            name: str(root / name)
            for name in ("skills", "input_views", "private_manifests")
        }
    )
    _json(root / "remote-operator.json", policy)
    work = Path(envelope["work_dir"])
    status = {"status": "running", "started_at": time.time(), "work_dir": str(work)}
    _json(root / "controller-status.json", status)
    telemetry = reporter = None
    try:
        telemetry = Telemetry(output_dir=output)
        reporter = ControllerReporter(output, envelope["manifest"], telemetry=telemetry)
        reporter.emit(
            "run_started",
            phase="run",
            status="running",
            planned_conditions=reporter.planned,
        )
        result = execute_local_sequential(
            envelope["manifest"],
            policy=policy,
            output_dir=output,
            work_dir=work,
            on_progress=reporter,
        )
        _json(root / "run-result.json", result)
        completed = _matrix_complete(result, envelope["manifest"])
        status.update(
            status="completed" if completed else "incomplete",
            completed_at=time.time(),
            completed_conditions=result.get("completed_conditions"),
            planned_conditions=result.get("planned_conditions"),
        )
        reporter.emit(
            "run_finished",
            phase="run",
            status=status["status"],
            completed_conditions=status["completed_conditions"],
            planned_conditions=status["planned_conditions"],
        )
        return 0 if completed else 2
    except Exception as error:
        if getattr(error, "result", None) is not None:
            _json(root / "run-result.json", error.result)
        status.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error)[:2000],
            completed_at=time.time(),
        )
        from simready_bench.engine.downloads import retry_receipt

        transport = retry_receipt(error)
        if transport is not None:
            status["source_transport_exhaustion"] = transport
        if getattr(error, "_simready_source_acquisition_failure", False) is True:
            status["source_acquisition_failure"] = True
        if reporter is not None:
            reporter.emit(
                "run_failed",
                phase="run",
                status="failed",
                error_type=type(error).__name__,
            )
        raise
    finally:
        if telemetry is not None:
            telemetry.close()
        _json(root / "controller-status.json", status)


class DaytonaObservationError(RuntimeError):
    """The saved executor could not be authoritatively observed."""


def _redactor(policy=None, *secrets):
    provider_secrets = []
    for provider in (policy or {}).get("providers", {}).values():
        provider_secrets.append(os.environ.get(provider.get("api_key_env", ""), ""))
    values = [
        value
        for value in (
            *secrets,
            os.environ.get("DAYTONA_API_KEY", ""),
            *provider_secrets,
        )
        if value
    ]

    def clean(value):
        value = str(value)
        for secret in values:
            value = value.replace(secret, "<redacted>")
        value = re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1<redacted>", value)
        value = re.sub(
            r"\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,})\b", "<redacted>", value
        )
        return value

    return clean


def _executor_command(sandbox, text, *, clean, timeout=30):
    response = sandbox.process.exec("sh -c " + shlex.quote(text), timeout=timeout)
    if response.exit_code:
        raise RuntimeError(
            "Daytona executor command failed: " + clean(response.result)[-2000:]
        )
    return response.result.strip()


def _controller_status(sandbox):
    raw = sandbox.fs.download_file("/work/controller-status.json", 30)
    if len(raw) > 65536:
        raise ValueError("The controller status exceeds its size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("The controller status must be an object")
    return value


class DaytonaReportError(RuntimeError):
    """Compute completed, but its retained report could not be published."""

    def __init__(self, result):
        self.result = result
        super().__init__(
            "The reconstruction report is retained but publication failed. Resume with --resume to retry publication."
        )


def _notify_report(on_report, report, record, *, save, final=False):
    if on_report is None:
        return True
    collected = set(record.get("collected_reconstruction_ids", []))
    published = set(record.get("reported_reconstruction_ids", []))
    if not final and not collected - published:
        return True
    try:
        on_report(report, final=final)
    except Exception as error:
        record["report_callback_error"] = type(error).__name__
        save()
        return False
    record["reported_reconstruction_ids"] = sorted(collected)
    record.pop("report_callback_error", None)
    save()
    return True


def _collect_reconstruction(sandbox, output, identity, expected):
    from simready_bench.controller_runner import (
        REPORT_DIRECTORIES,
        REPORT_FILES,
        verify_reconstruction,
    )

    if not re.fullmatch(r"condition-[a-z0-9-]{1,118}", identity):
        raise ValueError("A literal resolved reconstruction identity is required")
    root = output / "reconstructions"
    root.mkdir(mode=0o700, exist_ok=True)
    if root.resolve() != root or not root.is_dir():
        raise ValueError("A reconstruction collection must remain an owned directory")
    target = root / identity
    if target.exists():
        return verify_reconstruction(target, identity, expected)
    incoming = output / (".reconstruction-" + uuid4().hex)
    incoming.mkdir(mode=0o700)
    archive = incoming / "download.tar.gz"
    remote = "/tmp/simready-reconstruction-" + identity + "-" + uuid4().hex + ".tar.gz"
    paths = [
        identity + "/" + name for name in sorted(REPORT_FILES | REPORT_DIRECTORIES)
    ]
    command = (
        "set --; for p in "
        + shlex.join(paths)
        + '; do if [ -e "/work/output/reconstructions/$p" ]; then set -- "$@" "$p"; fi; done; tar -C /work/output/reconstructions -czf '
        + shlex.quote(remote)
        + ' -- "$@"; sha256sum '
        + shlex.quote(remote)
    )
    try:
        checksum = _executor_command(sandbox, command, clean=str, timeout=120).split()[
            0
        ]
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("The reconstruction archive has no SHA-256 digest")
        sandbox.fs.download_file(remote, str(archive), 300)
        with archive.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != checksum:
                raise ValueError("The downloaded reconstruction archive hash differs")
        count = size = 0
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream:
                name = PurePosixPath(member.name)
                if (
                    name.as_posix() != member.name.rstrip("/")
                    or name.is_absolute()
                    or ".." in name.parts
                    or len(name.parts) < 2
                    or name.parts[0] != identity
                ):
                    raise ValueError("Noncanonical reconstruction archive path")
                relative = name.parts[1:]
                if not (
                    (len(relative) == 1 and relative[0] in REPORT_FILES)
                    or relative[0] in REPORT_DIRECTORIES
                ) or not (member.isfile() or member.isdir()):
                    raise ValueError("Unexpected reconstruction archive member")
                count += 1
                size += member.size
                if count > 20000 or size > 1024**3:
                    raise ValueError(
                        "The reconstruction archive exceeds its size limit"
                    )
                destination = incoming.joinpath(*name.parts)
                if (
                    destination.resolve() != destination
                    or not destination.is_relative_to(incoming)
                ):
                    raise ValueError("The reconstruction archive escaped its directory")
                if member.isdir():
                    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    with (
                        stream.extractfile(member) as source,
                        destination.open("xb") as copied,
                    ):
                        shutil.copyfileobj(source, copied)
                    destination.chmod(0o600)
        verify_reconstruction(incoming / identity, identity, expected)
        (incoming / identity).rename(target)
        return verify_reconstruction(target, identity, expected)
    finally:
        if (
            incoming.resolve() == incoming
            and incoming.parent == output
            and incoming.name.startswith(".reconstruction-")
        ):
            shutil.rmtree(incoming)
        try:
            _executor_command(
                sandbox, "rm -f " + shlex.quote(remote), clean=str, timeout=10
            )
        except Exception:
            pass


def _collect_live_reports(
    sandbox, output, progress, record, *, save, on_report=None, maximum_items=2
):
    from simready_bench.benchmark import _aggregate_reports
    from simready_bench.controller_runner import verify_reconstruction

    if progress.get("schema") != "simready-bench-portable-report-v1":
        return
    collected = set(record.get("collected_reconstruction_ids", []))
    added = 0
    for row in progress.get("results", []):
        identity = row.get("condition_id")
        if row.get("status") != "completed" or identity in collected:
            continue
        if added >= maximum_items:
            break
        try:
            report = _collect_reconstruction(sandbox, output, identity, row)
        except Exception as error:
            record["reconstruction_collection_error"] = type(error).__name__
            save()
            break
        collected.add(identity)
        added += 1
        record["collected_reconstruction_ids"] = sorted(collected)
        with (Path(report["directory"]) / "artifacts.json").open("rb") as source:
            record.setdefault("collected_reconstruction_manifests", {})[identity] = (
                hashlib.file_digest(source, "sha256").hexdigest()
            )
        record.pop("reconstruction_collection_error", None)
        save()
    if not collected:
        return
    reports = []
    for row in progress["results"]:
        identity = row.get("condition_id")
        if identity in collected:
            path = output / "reconstructions" / identity
            reports.append((path, verify_reconstruction(path, identity, row)))
    report = _aggregate_reports(
        reports,
        output_dir=output / "reconstructions",
        planned_conditions=progress["planned_conditions"],
    )
    _notify_report(on_report, report, record, save=save)


def _monitor_executor(sandbox, record, output, *, save, on_report=None):
    """Observe the original session or trusted controller status without restarting it."""
    from simready_bench.controller_runner import forward_telemetry

    failures = 0
    next_progress = time.monotonic()
    health = record.get("health_policy", {})
    guarded = health.get("schema") == EXECUTOR_HEALTH_SCHEMA
    last_activity = record.get("last_activity_at", time.time())
    while True:
        observed = False
        command_id = record.get("command_id")
        if command_id:
            try:
                command = sandbox.process.get_session_command(
                    record.get("session_name", "simready-benchmark"),
                    command_id,
                    request_timeout=30,
                )
                observed = True
                failures = 0
                if command.exit_code is not None:
                    if guarded and command.exit_code != 0:
                        _stop_stalled_executor(
                            sandbox, record, output, reason="controller_exited",
                            exit_code=command.exit_code, save=save,
                        )
                    record.update(
                        controller_exit=command.exit_code, terminal_confirmed=True
                    )
                    save()
                    forward_telemetry(sandbox, output)
                    return
            except Exception:
                pass
        try:
            controller = _controller_status(sandbox)
        except Exception:
            controller = None
        if controller and controller.get("status") == "running":
            observed = True
            failures = 0
        if controller and controller.get("status") in (
            "completed",
            "failed",
            "incomplete",
        ):
            record.update(
                controller_status=controller["status"], terminal_confirmed=True
            )
            save()
            forward_telemetry(sandbox, output)
            return
        if not observed:
            failures += 1
        deadline = record.get("deadline_at")
        if guarded and deadline is not None and time.time() >= deadline:
            _stop_stalled_executor(sandbox, record, output, reason="executor_deadline_exceeded", save=save)
            return
        if guarded and failures >= 12:
            _stop_stalled_executor(sandbox, record, output, reason="executor_stalled", save=save)
            return
        if failures >= 12 or deadline is not None and time.time() >= deadline:
            raise DaytonaObservationError(
                "The saved Daytona executor is not confirmed terminal. Resume monitoring with --resume."
            )
        if time.monotonic() >= next_progress:
            forward_telemetry(sandbox, output)
            if guarded:
                try:
                    activity = _executor_activity(sandbox)
                    if activity != record.get("activity_counters"):
                        record.update(activity_counters=activity, last_activity_at=time.time())
                        last_activity = record["last_activity_at"]
                        save()
                except Exception:
                    pass
                try:
                    state = json.loads(_executor_command(sandbox, "docker inspect --format '{{json .State}}' simready-benchmark-controller", clean=str))
                    if state.get("Status") in ("exited", "dead") and state.get("Running") is False and type(state.get("ExitCode")) is int:
                        _stop_stalled_executor(sandbox, record, output, reason="controller_exited", exit_code=state["ExitCode"], save=save)
                        return
                except Exception:
                    pass
            try:
                try:
                    raw = sandbox.fs.download_file(
                        "/work/output/reconstructions/results.json", 30
                    )
                except Exception:
                    raw = sandbox.fs.download_file("/work/output/results.json", 30)
                if len(raw) > 128 * 1024**2:
                    raise ValueError(
                        "The generated progress report exceeds its size limit"
                    )
                value = json.loads(raw)
                _collect_live_reports(
                    sandbox, output, value, record, save=save, on_report=on_report
                )
                _json(output / "daytona-progress.json", value)
                previous = record.get("completed_conditions")
                before = (previous, record.get("attempted_conditions"))
                record.update(
                    completed_conditions=value.get("completed_conditions"),
                    attempted_conditions=value.get("attempted_conditions"),
                )
                if guarded and before != (record["completed_conditions"], record["attempted_conditions"]):
                    last_activity = time.time()
                    record["last_activity_at"] = last_activity
                save()
                if record["completed_conditions"] != previous:
                    print(
                        "Completed "
                        + str(record["completed_conditions"])
                        + "/"
                        + str(record.get("planned_conditions"))
                        + " reconstructions.",
                        flush=True,
                    )
            except Exception:
                pass
            if guarded and time.time() - last_activity >= health["stall_seconds"]:
                _stop_stalled_executor(sandbox, record, output, reason="executor_stalled", save=save)
                return
            next_progress = time.monotonic() + 60
        time.sleep(10)


def _verify_final_reconstruction_receipts(output, record):
    from simready_bench.controller_runner import verify_reconstruction
    from simready_bench.reporting import read_generated_report

    root = output / "reconstructions"
    index = root / "results.json"
    if not index.is_file():
        return
    rows = read_generated_report(index)["results"]
    completed = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        identity = row.get("condition_id")
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"condition-[a-z0-9-]{1,118}", identity)
            or identity in completed
        ):
            raise ValueError("The final reconstruction index has invalid identities")
        completed[identity] = row
    previous = record.get("collected_reconstruction_manifests", {})
    if not set(previous) <= completed.keys():
        raise ValueError(
            "The final collection omits a previously retained reconstruction"
        )
    verified = {}
    for identity, row in completed.items():
        directory = root / identity
        verify_reconstruction(directory, identity, row)
        with (directory / "artifacts.json").open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if identity in previous and previous[identity] != digest:
            raise ValueError("A retained reconstruction artefact manifest has changed")
        verified[identity] = digest
    record["collected_reconstruction_ids"] = sorted(verified)
    record["collected_reconstruction_manifests"] = verified


def _collect_executor(sandbox, record, output, *, clean, save):
    """Download only final output, controller records and the allowance ledger."""
    incoming = output / (".daytona-collection-" + uuid4().hex)
    incoming.mkdir(mode=0o700)
    downloaded = incoming / "download.tar.gz"
    try:
        _executor_command(
            sandbox,
            'if [ -d /work ] && [ -f /run/simready-benchmark/ledger.json ]; then cp /run/simready-benchmark/ledger.json /work/relay-ledger.json; chmod 600 /work/relay-ledger.json; fi; set --; for p in work/output work/run-result.json work/controller-status.json work/controller.stdout.log work/controller.stderr.log work/authority work/relay-ledger.json; do if [ -e /"$p" ]; then set -- "$@" "$p"; fi; done; if [ "$#" -eq 0 ]; then tar -C / -czf /tmp/simready-benchmark-results.tar.gz -T /dev/null; else tar -C / -czf /tmp/simready-benchmark-results.tar.gz "$@"; fi',
            clean=clean,
            timeout=300,
        )
        sandbox.fs.download_file(
            "/tmp/simready-benchmark-results.tar.gz", str(downloaded)
        )
        staged = incoming / "daytona-executor"
        staged.mkdir(mode=0o700)
        _extract_results(downloaded, staged)
        relay_path = staged / "relay-ledger.json"
        if relay_path.exists():
            if relay_path.stat().st_size > 32 * 1024**2:
                raise ValueError("The retained relay ledger exceeds its size bound")
            relay = json.loads(relay_path.read_bytes())
            fields = {
                "model",
                "generation_requests",
                "input_counter_requests",
                "maximum_requests",
                "maximum_condition_requests",
                "expires_at",
                "events",
                "condition_generation_requests",
                "condition_input_counter_requests",
                "condition_limits",
            }
            if (
                set(relay) != fields
                or relay["maximum_requests"] != record["planned_requests"]
                or any(
                    key in record and record[key] != relay[field]
                    for key, field in (
                        ("model", "model"),
                        ("maximum_condition_requests", "maximum_condition_requests"),
                        ("parent_expires_at", "expires_at"),
                    )
                )
            ):
                raise ValueError("The retained relay ledger changed its original scope")
            for field, multiplier in (
                ("generation_requests", 1),
                ("input_counter_requests", 4),
            ):
                if (
                    type(relay[field]) is not int
                    or not 0 <= relay[field] <= multiplier * relay["maximum_requests"]
                ):
                    raise ValueError("The retained relay request counters are invalid")
            digest = hashlib.sha256(relay_path.read_bytes()).hexdigest()
            accounting = staged / "output/accounting"
            accounting.mkdir(parents=True, mode=0o700, exist_ok=True)
            retained = accounting / ("provider-relay-" + digest + ".json")
            shutil.copyfile(relay_path, retained)
            retained.chmod(0o600)
            record["relay_ledger_sha256"] = digest
        for name in (
            "controller.stdout.log",
            "controller.stderr.log",
            "controller-status.json",
            "run-result.json",
        ):
            path = staged / name
            if path.is_file():
                path.write_text(
                    clean(path.read_text(encoding="utf-8", errors="replace")),
                    encoding="utf-8",
                )
        # Retain a redacted archive as well as redacted extracted controller logs.
        sanitized = incoming / "daytona-results.tar.gz"
        with tarfile.open(sanitized, "w:gz") as archive:
            for child in sorted(staged.iterdir()):
                archive.add(child, arcname="work/" + child.name)
        downloaded.unlink()
        remote = output / "daytona-executor"
        archive_path = output / "daytona-results.tar.gz"
        previous = [path for path in (remote, archive_path) if path.exists()]
        if previous:
            history = (
                output
                / "collections"
                / incoming.name.removeprefix(".daytona-collection-")
            )
            history.mkdir(parents=True, mode=0o700)
            for path in previous:
                if path.resolve() != path or not path.is_relative_to(output):
                    raise ValueError(
                        "Previous collection paths must remain inside the benchmark output"
                    )
                path.rename(history / path.name)
        staged.rename(remote)
        sanitized.rename(archive_path)
        if _collect_portable(remote / "output", output):
            record["report"] = str(output / "report.html")
        _verify_final_reconstruction_receipts(output, record)
        record.update(collection="completed")
        record.pop("collection_error", None)
        save()
        incoming.rmdir()
    except BaseException as error:
        record["collection_error"] = type(error).__name__
        save()
        raise


def _delete_executor(client, sandbox, record, *, save):
    from daytona.common.errors import DaytonaNotFoundError

    try:
        record["cleanup_confirmed"] = False
        client.delete(sandbox, timeout=60, wait=True)
        until = time.monotonic() + 30
        while time.monotonic() < until:
            try:
                client.get(sandbox.id)
            except DaytonaNotFoundError:
                record["cleanup_confirmed"] = True
                record.pop("cleanup_error", None)
                save()
                return
            time.sleep(0.5)
    except Exception as error:
        record["cleanup_error"] = type(error).__name__
        save()
        raise DaytonaObservationError(
            "Executor deletion is not confirmed. Reconnect with --resume."
        ) from error
    raise DaytonaObservationError(
        "Executor deletion is not confirmed. Reconnect with --resume."
    )


def _collected_result(manifest, output, record, *, clean):
    remote = output / "daytona-executor"
    result_path = remote / "run-result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if not result.get("results") and (output / "results.json").is_file():
        from simready_bench.reporting import read_generated_report

        result = read_generated_report(output / "results.json")
    controller_path = remote / "controller-status.json"
    controller = (
        json.loads(controller_path.read_text()) if controller_path.is_file() else {}
    )
    record["completed_conditions"] = result.get("completed_conditions", 0)
    if "attempted_conditions" in result:
        record["attempted_conditions"] = result["attempted_conditions"]
    if result.get("planned_conditions") is not None:
        record["planned_conditions"] = result["planned_conditions"]
    record["status"] = (
        "completed"
        if (
            (
                record.get("controller_exit") == 0
                or controller.get("status") == "completed"
            )
            and _matrix_complete(result, manifest)
            and (output / "report.html").is_file()
        )
        else "failed"
    )
    if record["status"] == "failed":
        record["error"] = clean(
            controller.get("error")
            or "The saved reconstruction batch did not complete."
        )[:2000]
    report = {"directory": str(output)}
    for key, name in (
        ("html", "report.html"),
        ("results", "results.json"),
        ("scores", "scores.csv"),
    ):
        if (output / name).is_file():
            report[key] = str(output / name)
    return {**result, "report": report, "deployment": record}


def _cold_retirement(client, sandbox, record, output, policy, manifest, *, save):
    """Retire an accepted create RPC only when provider staging never began."""
    from dataclasses import asdict
    from decimal import Decimal
    from daytona.common.errors import DaytonaNotFoundError, DaytonaFileNotFoundError
    from simready_bench.engine.inference_gateway import ProjectBudget
    from simready_bench.local import ParentAuthorityLedger
    from simready_bench.resume import _digest, _read, _write

    if record.get("controller_started") or record.get("phase") not in (
        "creating",
        "bootstrap",
    ):
        raise DaytonaObservationError(
            "A staged executor requires its original controller state"
        )
    allocation = record["parent_allocation"]["allocation_id"]
    expected_allocation = hashlib.sha256(
        (str(output) + "\n" + record["run_id"]).encode()
    ).hexdigest()
    parent = ProjectBudget(
        **policy["providers"][manifest["models"][0]["provider_ref"]]["project_budget"]
    )
    authority = ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    with authority._db() as db:
        held = db.execute(
            "SELECT parent,requests,usd,work_root FROM allocations WHERE id=?",
            (allocation,),
        ).fetchone()
        fingerprint = db.execute(
            "SELECT fingerprint FROM parents WHERE id=?", (parent.authorization_id,)
        ).fetchone()
    if (
        allocation != expected_allocation
        or not held
        or held[0] != parent.authorization_id
        or held[1] != record["planned_requests"]
        or Decimal(held[2]) != Decimal(str(record["maximum_spend_usd"]))
        or held[3] != str(output)
        or not fingerprint
        or fingerprint[0] != _digest(asdict(parent))
    ):
        raise ValueError(
            "A create-gap retirement requires its original parent allocation"
        )
    intent_path = output / "cold-bootstrap-intent.json"
    if intent_path.exists():
        intent = _read(intent_path)
        if (
            intent.get("sandbox_id") != record["sandbox_id"]
            or intent.get("allocation_id") != allocation
            or intent.get("original_parent_fingerprint") != fingerprint[0]
        ):
            raise ValueError("The saved create-gap proof changed its original scope")
    else:
        if sandbox is None:
            raise DaytonaObservationError(
                "A deleted create RPC needs its retained zero-staging proof"
            )
        name = "simready-benchmark-" + record["run_id"][:12]
        if (
            getattr(sandbox, "name", None) != name
            or getattr(sandbox, "labels", {}).get("project") != "simreadybench"
            or sandbox.labels.get("role") != "benchmark-executor"
        ):
            raise ValueError(
                "The accepted create RPC differs from its exact saved name"
            )
        for path in ("/work", "/run/simready-benchmark"):
            try:
                sandbox.fs.list_files(path)
            except (DaytonaFileNotFoundError, FileNotFoundError):
                pass
            else:
                raise DaytonaObservationError(
                    "Provider staging began before host interruption"
                )
        intent = {
            "schema": "simready-bench-cold-bootstrap-intent-v1",
            "sandbox_id": sandbox.id,
            "sandbox_name": name,
            "run_id": record["run_id"],
            "allocation_id": allocation,
            "original_parent_fingerprint": fingerprint[0],
            "work_directory_absent": True,
            "provider_staging_absent": True,
            "observed_at": time.time(),
        }
        _write(intent_path, intent)
    if sandbox is not None:
        _delete_executor(client, sandbox, record, save=save)
    try:
        client.get(record["sandbox_id"])
    except DaytonaNotFoundError:
        pass
    else:
        raise DaytonaObservationError("The accepted create RPC has not been deleted")
    proof = {
        **intent,
        "schema": "simready-bench-cold-bootstrap-retirement-v1",
        "executor_deleted": True,
        "requests": 0,
        "maximum_spend_usd": "0",
        "admitted_conditions": [],
    }
    proof_path = output / "retirement.json"
    if proof_path.exists():
        if _read(proof_path) != proof:
            raise ValueError("A create-gap retirement cannot change")
    else:
        _write(proof_path, proof)
    authority.retire(
        allocation_id=allocation,
        requests=0,
        maximum_spend_usd=Decimal(0),
        proof_sha256=_digest(proof),
    )
    record.update(
        status="failed",
        phase="cold_bootstrap_retired",
        collection="not_started",
        controller_started=False,
        terminal_confirmed=True,
        cleanup_confirmed=True,
    )
    save()
    return {
        "planned_conditions": record["planned_conditions"],
        "completed_conditions": 0,
        "results": [],
        "report": {"directory": str(output)},
        "deployment": record,
    }


def _lost_executor_result(manifest, output, record, *, on_report, save):
    from simready_bench.resume import condition_plan
    from simready_bench.reporting import read_generated_report, render_html, write_scores

    retained = set(record.get("collected_reconstruction_ids", []))
    rows = [
        {**values, "condition_id": identity, "status": "infrastructure_error",
         "failure_stage": "executor", "error_type": "ExecutorLost",
         "error_code": "executor_missing", "model_quality_evidence": False,
         "requested_model": manifest["models"][0]["model_id"]}
        for identity, values in condition_plan(manifest).items()
        if identity not in retained
    ]
    destination = output / "attempts/lost-executor"
    if (destination / "results.json").exists():
        result = read_generated_report(destination / "results.json")
        report = {"directory": str(destination), "results": str(destination / "results.json"),
                  "html": str(destination / "report.html"), "scores": str(destination / "scores.csv")}
    else:
        result = {"schema": "simready-bench-portable-report-v1",
                  "planned_conditions": record["planned_conditions"], "completed_conditions": 0,
                  "attempted_conditions": len(rows), "results": rows}
        if destination.parent.resolve() != destination.parent:
            raise ValueError("A lost executor report must remain in its owned output")
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        incoming = destination.with_name(".lost-executor-" + uuid4().hex)
        incoming.mkdir(mode=0o700)
        _json(incoming / "results.json", result)
        write_scores(incoming / "scores.csv", rows)
        (incoming / "report.html").write_text(render_html(result), encoding="utf-8")
        incoming.rename(destination)
        report = {"directory": str(destination), "results": str(destination / "results.json"),
                  "html": str(destination / "report.html"), "scores": str(destination / "scores.csv")}
    if not _notify_report(on_report, report, record, save=save, final=True):
        raise DaytonaReportError({**result, "report": report, "deployment": record})
    return {**result, "report": report, "deployment": record}


def reconnect_daytona(manifest, *, policy, output_dir, client=None, on_report=None):
    """Monitor and collect the saved executor without allocating or restarting work."""
    from daytona.common.errors import DaytonaNotFoundError

    output = Path(os.path.abspath(output_dir))
    delivery = output / "daytona-delivery.json"
    if (
        output.resolve() != output
        or delivery.resolve() != delivery
        or not delivery.is_file()
        or delivery.stat().st_size > 65536
    ):
        raise ValueError("A saved Daytona execution is required")
    record = json.loads(delivery.read_text())
    if client is None:
        from daytona import Daytona

        client = Daytona()
    clean = _redactor(policy)

    def save():
        _json(delivery, record)

    if not record.get("sandbox_id"):
        run_id = record.get("run_id", "")
        if (
            not re.fullmatch("[0-9a-f]{32}", run_id)
            or record.get("phase") != "creating"
        ):
            raise ValueError("The saved Daytona execution has no executor identity")
        name = "simready-benchmark-" + run_id[:12]
        if record.get("sandbox_name", name) != name:
            raise ValueError("The saved deterministic executor name changed")
        try:
            accepted = client.get(name)
        except Exception as error:
            raise DaytonaObservationError(
                "The original create RPC is not yet resolved by its exact name"
            ) from error
        if getattr(accepted, "name", None) != name:
            raise ValueError("The observed create RPC differs from its saved name")
        record["sandbox_id"] = accepted.id
        save()
        return _cold_retirement(
            client, accepted, record, output, policy, manifest, save=save
        )
    try:
        sandbox = client.get(record["sandbox_id"])
    except DaytonaNotFoundError:
        if (
            record.get("phase") in ("creating", "bootstrap")
            and (output / "cold-bootstrap-intent.json").is_file()
        ):
            return _cold_retirement(
                client, None, record, output, policy, manifest, save=save
            )
        if (
            record.get("phase") == "cold_bootstrap_retired"
            and (output / "retirement.json").is_file()
        ):
            return {
                "completed_conditions": 0,
                "results": [],
                "report": {"directory": str(output)},
                "deployment": record,
            }
        if record.get("collection") != "completed":
            if record.get("health_policy", {}).get("schema") == EXECUTOR_HEALTH_SCHEMA:
                record.update(
                    collection="lost", terminal_confirmed=True, cleanup_confirmed=True,
                    status="failed", termination_reason="executor_missing",
                    missing_executor_proof={"schema": "simready-bench-daytona-not-found-v1", "sandbox_id": record["sandbox_id"], "observed_at": time.time()},
                )
                save()
                return _lost_executor_result(manifest, output, record, on_report=on_report, save=save)
            raise DaytonaObservationError(
                "The previous executor is deleted but its output was not collected."
            )
        record["cleanup_confirmed"] = True
        result = _collected_result(manifest, output, record, clean=clean)
        save()
        if not _notify_report(
            on_report, result["report"], record, save=save, final=True
        ):
            raise DaytonaReportError(result)
        return result
    except Exception as error:
        raise DaytonaObservationError(
            "The saved Daytona executor could not be observed. Reconnect with --resume."
        ) from error
    if sandbox.id != record["sandbox_id"]:
        raise ValueError("The observed executor differs from the saved identity")
    if record.get("phase") in ("creating", "bootstrap") and not record.get(
        "controller_started"
    ):
        return _cold_retirement(
            client, sandbox, record, output, policy, manifest, save=save
        )
    try:
        if record.get("collection") != "completed":
            _monitor_executor(sandbox, record, output, save=save, on_report=on_report)
            _collect_executor(sandbox, record, output, clean=clean, save=save)
        else:
            controller = output / "daytona-executor/controller-status.json"
            terminal = (
                json.loads(controller.read_text()) if controller.is_file() else {}
            )
            if (
                not record.get("terminal_confirmed")
                and record.get("controller_exit") is None
                and terminal.get("status") not in ("completed", "failed", "incomplete")
            ):
                raise DaytonaObservationError(
                    "The collected executor is not confirmed terminal."
                )
        result = _collected_result(manifest, output, record, clean=clean)
        save()
        _delete_executor(client, sandbox, record, save=save)
        if not _notify_report(
            on_report, result["report"], record, save=save, final=True
        ):
            raise DaytonaReportError(result)
        return result
    except BaseException as error:
        if (
            not record.get("terminal_confirmed")
            and record.get("collection") != "completed"
        ):
            record["status"] = "unobserved"
            record["observation_error"] = clean(error)[:2000]
        save()
        raise


def run_daytona(
    manifest,
    *,
    policy,
    output_dir,
    assets_root,
    skills_root,
    work_dir=None,
    wheel=None,
    cpu_snapshot=None,
    runtime_images=None,
    on_report=None,
):
    """Run the full matrix remotely, retain its report and remove the executor."""
    from daytona import (
        CreateSandboxFromSnapshotParams,
        Daytona,
        DaytonaConfig,
        SessionExecuteRequest,
    )
    from simready_bench.local import ParentAuthorityLedger
    from . import provider_relay

    output = Path(os.path.abspath(output_dir))
    if output.resolve() != output or not output.is_dir():
        raise ValueError("An existing staged output directory is required")
    delivery = output / "daytona-delivery.json"
    if delivery.exists():
        raise FileExistsError("Preserving an existing Daytona run")
    actor_resources = _actor_resources(
        policy.get("runtime", {}).get("container_resources")
    )
    images = dict(DEFAULT_IMAGES if runtime_images is None else runtime_images)
    for role in DEFAULT_IMAGES:
        if role not in images or not re.fullmatch(
            r"(?:[A-Za-z0-9][A-Za-z0-9._:/-]*@)?sha256:[0-9a-f]{64}", images[role]
        ):
            raise ValueError(
                "Runtime images must provide actor, verifier and cpu image digests"
            )
    snapshot = cpu_snapshot or images.get("snapshot") or CPU_SNAPSHOT
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", snapshot):
        raise ValueError("A literal Daytona snapshot name is required")
    remote_work = str(work_dir) if work_dir else "/dev/shm/bench"
    path = PurePosixPath(remote_work)
    if (
        not path.is_absolute()
        or path.parts[:3] != ("/", "dev", "shm")
        or ".." in path.parts
        or len(remote_work) > 40
    ):
        raise ValueError(
            "Choose a short absolute Daytona work directory under /dev/shm"
        )
    provider, parent, requests, spend = _plan(manifest, policy)
    run_id = uuid4().hex
    allocation = hashlib.sha256((str(output) + "\n" + run_id).encode()).hexdigest()
    budgets = manifest["defaults"]["budgets"]
    conditions = requests // budgets["inference_requests"]
    render_seconds = (
        manifest.get("defaults", {}).get("illustrations", {}).get("seconds", 300)
    )
    planned_seconds = (
        conditions
        * (
            budgets.get("agent_seconds", 1800)
            + budgets.get("final_seconds", 600)
            + render_seconds
            + 300
        )
        + 600
    )
    execution_seconds = min(planned_seconds, parent.expires_at - time.time() - 600)
    if execution_seconds <= 0:
        raise ValueError(
            "The parent authority needs at least ten minutes remaining for execution and collection"
        )
    deadline_at = time.time() + execution_seconds
    ttl_minutes = math.ceil((execution_seconds + 600) / 60)
    key = os.environ.get("DAYTONA_API_KEY")
    provider_key = os.environ.get(provider.get("api_key_env", ""))
    if not key or not provider_key:
        raise ValueError("Set DAYTONA_API_KEY and the selected model provider key")
    project_id = os.environ.get(provider.get("project_id_env", ""))
    package_root = output / ".executor-package"
    wheel_path = _check_wheel(wheel or build_wheel(package_root))
    package = output / ".executor-inputs.tar.gz"
    roots = {
        "assets": assets_root,
        "skills": skills_root,
        "input_views": output / "input_views",
        "private_manifests": output / "private_manifests",
    }
    _archive_inputs(roots, package)
    marker = uuid4().hex + uuid4().hex
    record = {
        "schema": "simready-bench-daytona-delivery-v2",
        "status": "starting",
        "phase": "creating",
        "run_id": run_id,
        "sandbox_name": "simready-benchmark-" + run_id[:12],
        "snapshot": snapshot,
        "runtime_images": images,
        "container_resources": actor_resources,
        "planned_requests": requests,
        "model": parent.model,
        "maximum_condition_requests": budgets["inference_requests"],
        "parent_expires_at": parent.expires_at,
        "maximum_spend_usd": spend,
        "cleanup_confirmed": False,
        "planned_conditions": conditions,
        "deadline_at": deadline_at,
        "ttl_minutes": ttl_minutes,
        "health_policy": _health_policy(policy),
        "last_activity_at": time.time(),
        "wheel_sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
    }
    clean = _redactor(policy, key, provider_key, marker)

    def save():
        _json(delivery, record)

    client = Daytona(DaytonaConfig(api_key=key))
    # Reserve all planned requests before deriving a child or creating compute.
    record["parent_allocation"] = ParentAuthorityLedger(
        policy["runtime"]["authority_state_dir"]
    ).reserve(
        parent,
        allocation_id=allocation,
        requests=requests,
        maximum_spend_usd=spend,
        work_root=output,
    )
    if record["parent_allocation"]["status"] != "reserved":
        raise ValueError("A Daytona executor requires a new parent allocation")
    sandbox = None
    started = time.monotonic()
    collected = False
    result = None
    publication_failed = False
    save()

    def command(text, timeout=30):
        return _executor_command(sandbox, text, clean=clean, timeout=timeout)

    def collect():
        nonlocal collected
        _collect_executor(sandbox, record, output, clean=clean, save=save)
        collected = True

    try:
        sandbox = client.create(
            CreateSandboxFromSnapshotParams(
                name="simready-benchmark-" + run_id[:12],
                snapshot=snapshot,
                public=False,
                ttl_minutes=ttl_minutes,
                auto_stop_interval=ORPHAN_STOP_MINUTES,
                auto_delete_interval=0,
                ephemeral=True,
                labels={"project": "simreadybench", "role": "benchmark-executor"},
            ),
            timeout=90,
        )
        record["sandbox_id"] = sandbox.id
        record["phase"] = "bootstrap"
        save()
        sandbox.process.create_session("simready-dockerd")
        sandbox.process.execute_session_command(
            "simready-dockerd",
            SessionExecuteRequest(
                command=_bootstrap_command(),
                run_async=True,
            ),
            timeout=10,
        )
        command(
            "timeout 510 sh -c 'until docker info >/dev/null 2>&1; do sleep 2; done'",
            timeout=520,
        )
        command("mount -t tmpfs -o size=2147483648,nosuid,nodev,noexec tmpfs /dev/shm")
        command(
            "mkdir -m 700 /work /run/simready-benchmark; mkdir -m 700 /work/tmp; chown 1000:1000 /work /work/tmp"
        )
        sandbox.fs.upload_file(str(wheel_path), "/work/" + wheel_path.name)
        sandbox.fs.upload_file(str(package), "/work/inputs.tar.gz")
        sandbox.fs.upload_file(
            str(Path(provider_relay.__file__)), "/run/simready-benchmark/relay.py"
        )
        relay_config = {
            "model": parent.model,
            "marker": marker,
            "api_key": provider_key,
            "base_url": provider["base_url"],
            "project_id": project_id,
            "maximum_requests": requests,
            "maximum_condition_requests": manifest["defaults"]["budgets"][
                "inference_requests"
            ],
            "expires_at": parent.expires_at,
        }
        sandbox.fs.upload_file(
            json.dumps(relay_config).encode(), "/run/simready-benchmark/provider.json"
        )
        command("chmod 600 /run/simready-benchmark/*; test -f /etc/ssl/cert.pem")
        sandbox.process.create_session("simready-model-relay")
        sandbox.process.execute_session_command(
            "simready-model-relay",
            SessionExecuteRequest(
                command=shlex.join(_relay_command(images["cpu"])), run_async=True
            ),
            timeout=10,
        )
        ready = command(
            "timeout 25 sh -c 'until test -f /run/simready-benchmark/ready.json; do sleep 1; done'; cat /run/simready-benchmark/ready.json",
            timeout=30,
        )
        port = json.loads(ready)["port"]
        selected = remote_provider(
            provider,
            port=port,
            allocation_id=allocation,
            requests=requests,
            maximum_spend_usd=spend,
        )
        envelope = {
            "manifest": manifest,
            "provider": selected,
            "provider_ref": manifest["models"][0]["provider_ref"],
            "marker": marker,
            "project_id": project_id,
            "maximum_spend_usd": spend,
            "limits": policy["limits"],
            "container_resources": actor_resources,
            "cpu_image": images["cpu"],
            "actor_image": images["actor"],
            "verifier_image": images["verifier"],
            "work_dir": remote_work,
        }
        sandbox.fs.upload_file(json.dumps(envelope).encode(), "/work/controller.json")
        command(
            "chown 1000:1000 /work/*; chmod 600 /work/*.whl /work/*.json /work/*.gz; chmod 700 /work/tmp"
        )
        socket_gid = command("stat -c %g /run/simready-docker-load.sock")
        if not socket_gid.isdecimal():
            raise RuntimeError("The executor Docker socket group is unavailable")
        base = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network",
            "host",
            "--user",
            "1000:1000",
            "--group-add",
            socket_gid,
            "--env",
            "TMPDIR=/work/tmp",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--pids-limit",
            "2048",
            "--cpus",
            "2",
            "--memory",
            "3g",
            "--mount",
            "type=bind,src=/work,dst=/work",
            "--mount",
            "type=bind,src=/dev/shm,dst=/dev/shm",
            "--mount",
            "type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock",
        ]
        command(
            shlex.join(
                [
                    *base,
                    "--entrypoint",
                    "python",
                    images["cpu"],
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-index",
                    "--target",
                    "/work/installed",
                    "/work/" + wheel_path.name,
                ]
            ),
            timeout=120,
        )
        from simready_bench.controller_runner import DAYTONA_TELEMETRY_ENV

        telemetry_env = [
            part for name in DAYTONA_TELEMETRY_ENV for part in ("--env", name)
        ]
        controller_base = base.copy()
        controller_base.remove("--rm")
        argv = [
            *controller_base,
            *telemetry_env,
            "--name",
            "simready-benchmark-controller",
            "--env",
            "PYTHONPATH=/work/installed",
            "--entrypoint",
            "python",
            images["cpu"],
            "-u",
            "-B",
            "-m",
            "simready_bench.controller_runner",
            "/work/controller.json",
        ]
        sandbox.process.create_session("simready-benchmark")
        invocation = shlex.join(argv)
        record.update(phase="controller", controller_started=True)
        save()
        child = sandbox.process.execute_session_command(
            "simready-benchmark",
            SessionExecuteRequest(
                command="sh -c " + shlex.quote(invocation), run_async=True
            ),
            timeout=10,
        )
        record.update(
            status="running", session_name="simready-benchmark", command_id=child.cmd_id
        )
        save()
        _monitor_executor(sandbox, record, output, save=save, on_report=on_report)
        collect()
        result = _collected_result(manifest, output, record, clean=clean)
        if record["status"] != "completed":
            raise RuntimeError(
                "Daytona benchmark failed: "
                + record["error"]
                + ". Executor output: "
                + str(output / "daytona-executor")
            )
    except BaseException as error:
        unobserved = record.get("controller_started") and not record.get(
            "terminal_confirmed"
        )
        record.update(
            status="unobserved" if unobserved else "failed",
            error_type=type(error).__name__,
            error=clean(error)[:2000],
        )
        # A monitoring failure does not establish controller termination.
        if (
            sandbox is not None
            and not collected
            and not record.get("controller_started")
        ):
            try:
                collect()
            except Exception as collection_error:
                record["collection_error"] = type(collection_error).__name__
        raise
    finally:
        if (
            sandbox is not None
            and collected
            and (
                record.get("terminal_confirmed") or not record.get("controller_started")
            )
        ):
            try:
                _delete_executor(client, sandbox, record, save=save)
            except DaytonaObservationError:
                record["cleanup_confirmed"] = False
        record["seconds"] = time.monotonic() - started
        save()
        if collected and result is not None:
            publication_failed = not _notify_report(
                on_report, result["report"], record, save=save, final=True
            )
    if not record["cleanup_confirmed"]:
        raise DaytonaObservationError(
            "The benchmark output is collected but executor deletion is not confirmed. Reconnect with --resume."
        )
    if publication_failed:
        raise DaytonaReportError(result)
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--execute":
        raise SystemExit(execute_remote(sys.argv[2]))
    raise SystemExit("Use simready-bench benchmark --deploy daytona")
