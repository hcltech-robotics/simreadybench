"""Live reconstruction reports and Daytona application telemetry."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout, suppress
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from urllib import request as http_request
from urllib.parse import urlparse


DAYTONA_TELEMETRY_ENV = (
    "DAYTONA_OTEL_ENDPOINT",
    "DAYTONA_SANDBOX_ID",
    "DAYTONA_ORGANIZATION_ID",
    "DAYTONA_REGION_ID",
    "DAYTONA_SANDBOX_SNAPSHOT",
)


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


REPORT_FILES = frozenset(
    {"report.html", "results.json", "scores.csv", "artifacts.json"}
)
REPORT_DIRECTORIES = frozenset({"evidence", "candidates"})


def seal_reconstruction(directory, condition_id):
    directory = Path(directory).absolute()
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError("An owned reconstruction directory is required")
    files = []
    size = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Reconstruction artefacts cannot contain links")
        if path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative == "artifacts.json":
            continue
        if path.stat().st_nlink != 1 or not path.is_file():
            raise ValueError(
                "Reconstruction artefacts must be independent regular files"
            )
        size += path.stat().st_size
        if len(files) >= 10000 or size > 1024**3:
            raise ValueError(
                "The reconstruction artefact closure exceeds its size limit"
            )
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        files.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}
        )
    _write_json(
        directory / "artifacts.json",
        {
            "schema": "simready-bench-reconstruction-artifacts-v1",
            "condition_id": condition_id,
            "files": files,
        },
    )


def verify_reconstruction(directory, condition_id, expected=None):
    from simready_bench.reporting import FIDELITY_COMPONENTS, read_generated_report
    from pathlib import PurePosixPath

    directory = Path(directory).absolute()
    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError("An owned reconstruction directory is required")
    manifest = directory / "artifacts.json"
    if manifest.is_symlink() or manifest.stat().st_size > 4 * 1024**2:
        raise ValueError("A bounded reconstruction artefact manifest is required")
    closure = json.loads(manifest.read_text())
    if (
        closure.get("schema") != "simready-bench-reconstruction-artifacts-v1"
        or closure.get("condition_id") != condition_id
    ):
        raise ValueError("The reconstruction artefact identity differs")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Reconstruction artefacts cannot contain links")
        if path.is_file():
            relative = path.relative_to(directory).as_posix()
            parts = PurePosixPath(relative).parts
            if not (
                (len(parts) == 1 and parts[0] in REPORT_FILES)
                or parts[0] in REPORT_DIRECTORIES
            ):
                raise ValueError("Unexpected file in reconstruction artefact closure")
            actual.add(relative)
    expected_files = {"artifacts.json"}
    records = {}
    size = 0
    for item in closure["files"]:
        relative = PurePosixPath(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != item["path"]
            or item["path"] in expected_files
        ):
            raise ValueError("Noncanonical or repeated reconstruction artefact path")
        expected_files.add(item["path"])
        records[item["path"]] = item
        path = directory.joinpath(*relative.parts)
        size += item["size_bytes"]
        if (
            len(expected_files) > 10001
            or size > 1024**3
            or path.stat().st_nlink != 1
            or path.stat().st_size != item["size_bytes"]
        ):
            raise ValueError("Reconstruction artefact size differs")
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() != item["sha256"]:
                raise ValueError("Reconstruction artefact hash differs")
    if (
        actual != expected_files
        or not {"report.html", "results.json", "scores.csv"} <= actual
    ):
        raise ValueError("The reconstruction artefact roster differs")
    portable = read_generated_report(directory / "results.json")
    if len(portable["results"]) != 1:
        raise ValueError(
            "One completed reconstruction per artefact closure is required"
        )
    row = portable["results"][0]
    score = _number(row.get("simready_reward"))
    if (
        row.get("condition_id") != condition_id
        or row.get("status") != "completed"
        or score is None
        or not 0 <= score <= 100
    ):
        raise ValueError("The reconstruction has no completed measured result")
    if expected is not None:
        for key in (
            "condition_id",
            "asset_id",
            "repetition",
            "requested_model",
            "harness_id",
            "skill_set_id",
            "simready_reward",
            *(
                "fidelity_" + component + "_percent"
                for component in FIDELITY_COMPONENTS
            ),
        ):
            if row.get(key) != expected.get(key):
                raise ValueError(
                    "The collected reconstruction differs from its progress index"
                )

    def linked(path, sha256=None, size=None):
        item = records.get(path)
        if (
            item is None
            or sha256 is not None
            and item["sha256"] != sha256
            or size is not None
            and item["size_bytes"] != size
        ):
            raise ValueError("A reconstruction report link is missing or changed")

    for kind in ("images", "traces"):
        for item in row.get(kind, []):
            linked(
                "evidence/" + item["path"], item.get("sha256"), item.get("size_bytes")
            )
    nvidia = row.get("nvidia_validator_report_file")
    if nvidia:
        linked(nvidia["path"], nvidia.get("sha256"), nvidia.get("size_bytes"))
    candidate = row.get("candidate") or {}
    if candidate.get("status") == "available":
        linked(candidate["entrypoint"])
        linked(candidate["manifest"], candidate["manifest_sha256"])
        for item in candidate["files"]:
            linked(item["path"], item["sha256"], item["size_bytes"])
    return {
        "directory": str(directory),
        "html": str(directory / "report.html"),
        "results": str(directory / "results.json"),
        "scores": str(directory / "scores.csv"),
    }


class TelemetryOutbox:
    def __init__(self, output_dir, endpoint):
        self.root = Path(output_dir) / "telemetry"
        self.root.mkdir(mode=0o700)
        self.endpoint = endpoint.rstrip("/")
        self.batches = {}
        self.lock = threading.Lock()

    def save(self, signal, payload):
        identity = hashlib.sha256(signal.encode() + b"\x00" + payload).hexdigest()
        with self.lock:
            if identity not in self.batches:
                name = identity + ".pb"
                (self.root / name).write_bytes(payload)
                self.batches[identity] = {
                    "id": identity,
                    "signal": signal,
                    "file": name,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "delivered": False,
                }
                self._index()
        return identity

    def delivered(self, identity):
        with self.lock:
            self.batches[identity]["delivered"] = True
            self._index()

    def _index(self):
        _write_json(
            self.root / "outbox.json",
            {
                "endpoint": self.endpoint,
                "batches": list(self.batches.values()),
            },
        )


class _DurableExporter:
    def __init__(self, exporter, outbox, signal, encode):
        self.exporter, self.outbox, self.signal, self.encode = (
            exporter,
            outbox,
            signal,
            encode,
        )

    def export(self, value, *args, **kwargs):
        identity = self.outbox.save(self.signal, self.encode(value).SerializeToString())
        result = self.exporter.export(value, *args, **kwargs)
        if result.name == "SUCCESS":
            self.outbox.delivered(identity)
        return result

    def __getattr__(self, name):
        return getattr(self.exporter, name)


def forward_telemetry(sandbox, output_dir, *, maximum_batches=8):
    """Forward queued application telemetry from the host when sandbox delivery fails."""
    output = Path(output_dir)
    try:
        raw = sandbox.fs.download_file("/work/output/telemetry/outbox.json", 10)
        if len(raw) > 16 * 1024**2:
            return 0
        index = json.loads(raw)
    except Exception:
        return 0
    endpoint = index.get("endpoint", "").rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        return 0
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return 0
    checkpoint = output / "daytona-telemetry.json"
    delivered = (
        set(json.loads(checkpoint.read_text()).get("delivered", []))
        if checkpoint.is_file()
        else set()
    )
    forwarded = 0
    for batch in index.get("batches", []):
        if forwarded >= maximum_batches:
            break
        if batch.get("delivered") or batch.get("id") in delivered:
            continue
        identity, signal, name = (
            batch.get("id", ""),
            batch.get("signal"),
            batch.get("file", ""),
        )
        if (
            len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
            or name != identity + ".pb"
            or signal not in {"logs", "traces", "metrics"}
        ):
            continue
        try:
            payload = sandbox.fs.download_file("/work/output/telemetry/" + name, 10)
            if (
                len(payload) > 4 * 1024**2
                or hashlib.sha256(payload).hexdigest() != batch["sha256"]
            ):
                continue
            request = http_request.Request(
                endpoint + "/v1/" + signal,
                data=payload,
                headers={"Content-Type": "application/x-protobuf"},
                method="POST",
            )
            with http_request.urlopen(request, timeout=5) as response:
                if response.status != 200:
                    break
            delivered.add(identity)
            _write_json(checkpoint, {"delivered": sorted(delivered)})
            forwarded += 1
        except Exception:
            break
    return forwarded


class Telemetry:
    def __init__(self, environment=None, *, output_dir=None):
        env = os.environ if environment is None else environment
        self.providers = []
        self.logger = None
        endpoint = env.get("DAYTONA_OTEL_ENDPOINT")
        if not endpoint:
            return
        from opentelemetry.exporter.otlp.proto.http._log_exporter import (
            OTLPLogExporter,
            encode_logs,
        )
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
            encode_metrics,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
            encode_spans,
        )
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        sandbox = env.get("DAYTONA_SANDBOX_ID", "unknown")
        attributes = {
            "service.name": "sandbox-" + sandbox,
            "service.instance.id": sandbox,
        }
        for source, target in (
            ("DAYTONA_SANDBOX_ID", "daytona_sandbox_id"),
            ("DAYTONA_ORGANIZATION_ID", "daytona_organization_id"),
            ("DAYTONA_REGION_ID", "daytona_region_id"),
            ("DAYTONA_SANDBOX_SNAPSHOT", "daytona_snapshot"),
        ):
            if env.get(source):
                attributes[target] = env[source]
        resource = Resource.create(attributes)
        endpoint = endpoint.rstrip("/")
        outbox = (
            TelemetryOutbox(output_dir, endpoint) if output_dir is not None else None
        )

        def exporter(kind, cls, encode):
            instance = cls(endpoint=endpoint + "/v1/" + kind, timeout=3)
            return (
                _DurableExporter(instance, outbox, kind, encode)
                if outbox is not None
                else instance
            )

        traces = TracerProvider(resource=resource)
        traces.add_span_processor(
            BatchSpanProcessor(
                exporter("traces", OTLPSpanExporter, encode_spans),
                schedule_delay_millis=1000,
                export_timeout_millis=3000,
            )
        )
        logs = LoggerProvider(resource=resource)
        logs.add_log_record_processor(
            BatchLogRecordProcessor(
                exporter("logs", OTLPLogExporter, encode_logs),
                schedule_delay_millis=1000,
                export_timeout_millis=3000,
            )
        )
        reader = PeriodicExportingMetricReader(
            exporter("metrics", OTLPMetricExporter, encode_metrics),
            export_interval_millis=60000,
            export_timeout_millis=3000,
        )
        meters = MeterProvider(resource=resource, metric_readers=[reader])
        self.providers = [traces, logs, meters]
        self.tracer = traces.get_tracer("simready_bench.controller")
        meter = meters.get_meter("simready_bench.controller")
        self.completed = meter.create_counter(
            "simready.reconstructions.completed", unit="{reconstruction}"
        )
        self.score = meter.create_histogram("simready.score", unit="%")
        self.fidelity = meter.create_histogram("simready.fidelity", unit="%")
        self.logger = logs.get_logger("simready_bench.controller")

    def emit(self, event):
        if self.logger is None:
            return
        from opentelemetry.trace import Status, StatusCode, use_span
        from opentelemetry._logs import SeverityNumber

        attributes = {
            "simready." + key: value
            for key, value in event.items()
            if type(value) in (str, int, float, bool)
        }
        end = int(event["timestamp"] * 1e9)
        elapsed = max(0, _number(event.get("elapsed_seconds")) or 0)
        span = self.tracer.start_span(
            event["event"],
            attributes=attributes,
            start_time=end - int(elapsed * 1e9),
        )
        if event.get("status") in {"failed", "infrastructure_error", "condition_error"}:
            span.set_status(Status(StatusCode.ERROR))
        with use_span(span, end_on_exit=False):
            self.logger.emit(
                timestamp=end,
                severity_number=SeverityNumber.INFO,
                body=json.dumps(event, sort_keys=True, allow_nan=False),
                attributes=attributes,
            )
        span.end(end_time=end)
        if event["event"] == "reconstruction_completed":
            labels = {
                key: event[key]
                for key in ("asset_id", "model", "harness", "repetition")
                if key in event
            }
            self.completed.add(1, labels)
            if event.get("score_percent") is not None:
                self.score.record(event["score_percent"], labels)
            for component in ("shape", "rigid_parts", "joints", "materials"):
                value = event.get("fidelity_" + component + "_percent")
                if value is not None:
                    self.fidelity.record(value, {**labels, "component": component})

    def close(self):
        for provider in self.providers:
            with suppress(Exception):
                provider.force_flush(timeout_millis=3000)
            with suppress(Exception):
                provider.shutdown()


class ControllerReporter:
    def __init__(self, output_dir, manifest, *, telemetry=None, stream=None):
        self.output = Path(output_dir) / "reconstructions"
        self.output.mkdir(mode=0o700)
        selection = manifest.get("reconstruction_selection", {})
        self.planned = (
            sum(
                len(selection.get(asset["id"], range(manifest["repetitions"])))
                for asset in manifest["assets"]
            )
            * len(manifest["models"])
            * len(manifest["harnesses"])
            * len(manifest["skill_sets"])
        )
        self.telemetry = telemetry
        self.stream = sys.stdout if stream is None else stream
        self.reports = []
        self.seen = set()
        self.completed = 0

    def emit(self, name, **values):
        event = {"event": name, "timestamp": time.time(), **values}
        print(
            json.dumps(event, sort_keys=True, allow_nan=False),
            file=self.stream,
            flush=True,
        )
        if self.telemetry is not None:
            self.telemetry.emit(event)

    def __call__(self, work_dir, progress):
        from simready_bench.benchmark import _aggregate_reports
        from simready_bench.reporting import (
            FIDELITY_COMPONENTS,
            export_report,
            read_generated_report,
        )

        for row in progress.get("results", []):
            identity = row["condition_id"]
            if identity in self.seen:
                continue
            if not identity.startswith("condition-") or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in identity
            ):
                raise ValueError(
                    "A reconstruction report requires its resolved condition identity"
                )
            destination = self.output / identity
            report = export_report(
                work_dir,
                destination,
                results={
                    "schema": "simready-bench-matrix-result-v1",
                    "planned_conditions": 1,
                    "attempted_conditions": 1,
                    "completed_conditions": int(row.get("status") == "completed"),
                    "results": [row],
                },
            )
            portable = read_generated_report(report["results"])["results"][0]
            seal_reconstruction(destination, identity)
            self.reports.append((destination, report))
            _aggregate_reports(
                self.reports, output_dir=self.output, planned_conditions=self.planned
            )
            self.seen.add(identity)
            self.completed += row.get("status") == "completed"
            self.emit(
                "reconstruction_completed"
                if row.get("status") == "completed"
                else "reconstruction_failed",
                phase="reconstruction",
                condition_id=identity,
                trial_id=row.get("trial_id"),
                asset_id=row.get("asset_id"),
                repetition=row.get("repetition"),
                model=row.get("requested_model") or row.get("model_id"),
                harness=row.get("harness_id"),
                skills=row.get("skill_set_id"),
                status=row.get("status"),
                score_percent=_number(portable.get("simready_reward")),
                **{
                    "fidelity_" + component + "_percent": _number(
                        portable.get("fidelity_" + component + "_percent")
                    )
                    for component in FIDELITY_COMPONENTS
                },
                elapsed_seconds=_number(row.get("elapsed_seconds")),
                completed_conditions=self.completed,
                planned_conditions=self.planned,
                report=str(destination / "report.html"),
            )


class _Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile, self.lock = stream, logfile, threading.Lock()

    def write(self, value):
        with self.lock:
            self.logfile.write(value)
            self.logfile.flush()
            return self.stream.write(value)

    def flush(self):
        with self.lock:
            self.logfile.flush()
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def main(path):
    from simready_bench.daytona import execute_remote

    root = Path(path).absolute().parent
    with (
        (root / "controller.stdout.log").open("a", buffering=1) as stdout,
        (root / "controller.stderr.log").open("a", buffering=1) as stderr,
    ):
        with (
            redirect_stdout(_Tee(sys.stdout, stdout)),
            redirect_stderr(_Tee(sys.stderr, stderr)),
        ):
            try:
                return execute_remote(path)
            except Exception:
                traceback.print_exc()
                return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
