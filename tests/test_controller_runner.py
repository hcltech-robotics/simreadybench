import io
import json
import time

from simready_bench.controller_runner import (
    ControllerReporter,
    DAYTONA_TELEMETRY_ENV,
    Telemetry,
)


def manifest():
    return {
        "assets": [{"id": "box"}],
        "repetitions": 4,
        "models": [{"id": "selected"}],
        "harnesses": [{"id": "pi"}],
        "skill_sets": [{"id": "none"}],
    }


def measured(work, index, score=100):
    trace = work / ("session-" + str(index) + ".jsonl")
    trace.write_text('{"type":"message","role":"assistant","content":"authored"}\n')
    return {
        "condition_id": "condition-" + str(index),
        "trial_id": "native-" + str(index),
        "asset_id": "box",
        "repetition": index,
        "model_id": "selected",
        "requested_model": "gpt-6.1-sol",
        "harness_id": "pi",
        "skill_set_id": "none",
        "status": "completed",
        "elapsed_seconds": 24,
        "evaluation": {"simready": {"score_percent": score, "checks": []}},
        "harbor_trace_paths": [str(trace)],
        "fidelity": {
            "status": "measured",
            "components": {
                "shape": {"score_percent": 72.5},
                "materials": {"score_percent": 0},
            },
        },
        "gateway_records": [{"Authorization": "secret-not-logged"}],
    }


def test_each_reconstruction_has_a_portable_report_before_the_next_completes(tmp_path):
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    output.mkdir()
    stream = io.StringIO()
    reporter = ControllerReporter(output, manifest(), stream=stream)
    first = measured(work, 0)
    reporter(work, {"results": [first]})
    readouts = output / "reconstructions"
    assert (readouts / "condition-0/report.html").is_file()
    progress = json.loads((readouts / "results.json").read_text())
    assert progress["completed_conditions"] == 1 and progress["planned_conditions"] == 4
    assert progress["results"][0]["simready_reward"] == 100
    trace = progress["results"][0]["traces"][0]["path"]
    assert (readouts / "evidence" / trace).read_bytes() == (
        work / "session-0.jsonl"
    ).read_bytes()
    first_bytes = (readouts / "condition-0/results.json").read_bytes()
    second = measured(work, 1, 0)
    reporter(work, {"results": [first, second]})
    progress = json.loads((readouts / "results.json").read_text())
    assert progress["completed_conditions"] == 2
    assert [row["repetition"] for row in progress["results"]] == [0, 1]
    assert [row["simready_reward"] for row in progress["results"]] == [100, 0]
    assert (readouts / "condition-0/results.json").read_bytes() == first_bytes
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(events) == 2
    assert events[0]["fidelity_shape_percent"] == 72.5
    assert events[0]["fidelity_materials_percent"] == 0
    assert events[0]["fidelity_joints_percent"] is None
    assert events[1]["score_percent"] == 0
    assert "secret-not-logged" not in stream.getvalue()


def test_failed_reconstruction_does_not_emit_a_completed_score(tmp_path):
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir()
    output.mkdir()
    stream = io.StringIO()
    reporter = ControllerReporter(output, manifest(), stream=stream)
    row = measured(work, 0)
    row.update(status="infrastructure_error", evaluation=None)
    reporter(work, {"results": [row]})
    event = json.loads(stream.getvalue())
    assert event["event"] == "reconstruction_failed"
    assert event["score_percent"] is None and event["completed_conditions"] == 0


def test_controller_heartbeat_reports_active_asset_and_progress(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    stream = io.StringIO()
    reporter = ControllerReporter(output, manifest(), stream=stream)
    reporter.asset_started({"id": "box"}, 1, 1)
    reporter.start_heartbeat(interval_seconds=0.01)
    time.sleep(0.04)
    reporter.close()

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    heartbeat = next(event for event in events if event["event"] == "run_heartbeat")
    assert heartbeat["active_asset_id"] == "box"
    assert heartbeat["active_asset_index"] == 1
    assert heartbeat["active_asset_count"] == 1
    assert heartbeat["completed_conditions"] == 0
    assert heartbeat["planned_conditions"] == 4
    assert heartbeat["active_asset_elapsed_seconds"] >= 0
    assert reporter._heartbeat_thread is not None
    assert not reporter._heartbeat_thread.is_alive()


def test_controller_streams_stdout_stderr_and_keeps_original_exit_status(
    tmp_path, monkeypatch, capsys
):
    from simready_bench import controller_runner, daytona
    import sys

    def execute(path):
        print("reconstruction box 1 completed")
        print("provider route closed", file=sys.stderr)
        return 2

    monkeypatch.setattr(daytona, "execute_remote", execute)
    assert controller_runner.main(tmp_path / "controller.json") == 2
    captured = capsys.readouterr()
    assert "box 1 completed" in captured.out
    assert "provider route closed" in captured.err
    assert "box 1 completed" in (tmp_path / "controller.stdout.log").read_text()
    assert "provider route closed" in (tmp_path / "controller.stderr.log").read_text()


def test_telemetry_can_be_disabled_without_a_collector_and_never_forwards_cloud_credentials():
    telemetry = Telemetry({})
    telemetry.emit({"event": "reconstruction_completed", "timestamp": 1})
    telemetry.close()
    assert "DAYTONA_SANDBOX_ID" in DAYTONA_TELEMETRY_ENV
    assert "DAYTONA_SANDBOX_SNAPSHOT" in DAYTONA_TELEMETRY_ENV
    assert "DAYTONA_API_KEY" not in DAYTONA_TELEMETRY_ENV


def test_real_otlp_exporters_send_reconstruction_logs_spans_and_scores(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    received = {}

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            received.setdefault(self.path, []).append(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        telemetry = Telemetry(
            {
                "DAYTONA_OTEL_ENDPOINT": "http://127.0.0.1:" + str(server.server_port),
                "DAYTONA_SANDBOX_ID": "owned-executor",
                "DAYTONA_ORGANIZATION_ID": "organisation",
                "DAYTONA_SANDBOX_SNAPSHOT": "simready-bench-cpu",
            },
            output_dir=tmp_path,
        )
        event = {
            "event": "reconstruction_completed",
            "timestamp": 100,
            "asset_id": "box",
            "repetition": 0,
            "model": "gpt-6.1-sol",
            "harness": "pi",
            "status": "completed",
            "elapsed_seconds": 24,
            "score_percent": 100,
            "fidelity_shape_percent": 72.5,
            "fidelity_materials_percent": 0,
        }
        telemetry.emit(event)
        telemetry.emit(
            {
                "event": "run_heartbeat",
                "timestamp": 130,
                "elapsed_seconds": 30,
                "completed_conditions": 1,
                "planned_conditions": 4,
            }
        )
        telemetry.close()
        assert {"/v1/logs", "/v1/traces", "/v1/metrics"} <= received.keys()
        traces = ExportTraceServiceRequest.FromString(received["/v1/traces"][0])
        resources = traces.resource_spans[0]
        attributes = {
            item.key: item.value.string_value for item in resources.resource.attributes
        }
        assert attributes["service.name"] == "sandbox-owned-executor"
        assert attributes["daytona_organization_id"] == "organisation"
        assert attributes["daytona_snapshot"] == "simready-bench-cpu"
        spans = {span.name: span for span in resources.scope_spans[0].spans}
        span = spans["reconstruction_completed"]
        assert span.name == "reconstruction_completed"
        assert span.end_time_unix_nano - span.start_time_unix_nano == 24_000_000_000
        heartbeat_span = spans["run_heartbeat"]
        assert heartbeat_span.end_time_unix_nano == heartbeat_span.start_time_unix_nano
        logs = ExportLogsServiceRequest.FromString(received["/v1/logs"][0])
        record = logs.resource_logs[0].scope_logs[0].log_records[0]
        assert json.loads(record.body.string_value) == event
        assert record.trace_id == span.trace_id and record.span_id == span.span_id
        metrics = ExportMetricsServiceRequest.FromString(received["/v1/metrics"][0])
        values = {
            metric.name: metric
            for resource in metrics.resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        assert set(values) == {
            "simready.reconstructions.completed",
            "simready.score",
            "simready.fidelity",
        }
        assert values["simready.score"].histogram.data_points[0].sum == 100
        assert sorted(
            point.sum for point in values["simready.fidelity"].histogram.data_points
        ) == [0, 72.5]
        outbox = json.loads((tmp_path / "telemetry/outbox.json").read_text())
        assert {batch["signal"] for batch in outbox["batches"]} == {
            "logs",
            "traces",
            "metrics",
        }
        assert all(batch["delivered"] for batch in outbox["batches"])
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_failed_sandbox_exports_are_durably_forwarded_by_host_once(
    tmp_path, monkeypatch
):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from types import SimpleNamespace
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk._logs.export import LogRecordExportResult
    from opentelemetry.sdk.metrics.export import MetricExportResult
    from opentelemetry.sdk.trace.export import SpanExportResult
    from simready_bench.controller_runner import forward_telemetry

    monkeypatch.setattr(
        OTLPSpanExporter, "export", lambda *args, **kwargs: SpanExportResult.FAILURE
    )
    monkeypatch.setattr(
        OTLPLogExporter, "export", lambda *args, **kwargs: LogRecordExportResult.FAILURE
    )
    monkeypatch.setattr(
        OTLPMetricExporter, "export", lambda *args, **kwargs: MetricExportResult.FAILURE
    )
    received = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.headers["Content-Type"] == "application/x-protobuf"
            received.append(
                (self.path, self.rfile.read(int(self.headers["Content-Length"])))
            )
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        telemetry = Telemetry(
            {
                "DAYTONA_OTEL_ENDPOINT": "http://127.0.0.1:" + str(server.server_port),
                "DAYTONA_SANDBOX_ID": "owned-executor",
            },
            output_dir=tmp_path,
        )
        telemetry.emit(
            {
                "event": "reconstruction_completed",
                "timestamp": 100,
                "asset_id": "box",
                "repetition": 0,
                "status": "completed",
                "score_percent": 91,
            }
        )
        telemetry.close()
        index = json.loads((tmp_path / "telemetry/outbox.json").read_text())
        assert index["batches"] and not any(
            batch["delivered"] for batch in index["batches"]
        )
        assert received == []

        def download(remote, timeout):
            return (tmp_path / remote.removeprefix("/work/output/")).read_bytes()

        sandbox = SimpleNamespace(fs=SimpleNamespace(download_file=download))
        assert forward_telemetry(sandbox, tmp_path) == len(index["batches"])
        assert {path for path, payload in received} == {
            "/v1/logs",
            "/v1/traces",
            "/v1/metrics",
        }
        assert all(payload for path, payload in received)
        assert forward_telemetry(sandbox, tmp_path) == 0
        assert len(received) == len(index["batches"])
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
