import json

from simready_bench.daytona_telemetry import activity, deliver, payloads


def test_activity_uses_only_provider_metadata():
    record = {"sandbox_id": "worker-one", "model": "model", "last_observed_at": 95}
    ledger = {
        "generation_requests": 2,
        "maximum_requests": 64,
        "secret": "token",
        "events": [
            {
                "started_at": 90,
                "finished_at": 98,
                "status": "completed",
                "http_status": 200,
                "prompt": "private prompt",
                "headers": "secret",
            },
            {"started_at": 99, "status": "admitted", "scope": "condition-one"},
        ],
    }
    summary, events = activity(record, ledger, now=100)
    assert summary["inflight_requests"] == 1
    assert summary["oldest_request_age_seconds"] == 1
    assert summary["coordinator_observation_age_seconds"] == 5
    assert events[0]["elapsed_seconds"] == 8
    assert events[1]["condition_scope_sha256"] != "condition-one"
    signals, seen = payloads(record, ledger, {"run_id": "canonical"}, now=100)
    encoded = json.dumps(signals)
    assert "private prompt" not in encoded and "secret" not in encoded
    assert '"name": "simready.model.inflight_requests"' in encoded
    assert len(signals["traces"]["resourceSpans"][0]["scopeSpans"][0]["spans"]) == 1
    repeated, _ = payloads(record, ledger, {"run_id": "canonical"}, now=101, seen=seen)
    assert "traces" not in repeated
    assert len(repeated["logs"]["resourceLogs"][0]["scopeLogs"][0]["logRecords"]) == 1


class Response:
    status = 200

    def __init__(self, payload=b"{}"):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, maximum):
        return self.payload


def queued(tmp_path):
    root = tmp_path / "relay-telemetry"
    root.mkdir()
    path = root / "pending-one.json"
    path.write_text(json.dumps({"signal": "logs", "payload": {"resourceLogs": []}}))
    return path


def test_delivery_retries_the_original_batch_after_network_failure(tmp_path):
    batch = queued(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("offline")

    state = deliver(tmp_path, "https://otel-collector.app.daytona.io", opener=fail)
    assert batch.exists() and state["status"] == "retrying"
    calls = []

    def accept(req, **kwargs):
        calls.append(req)
        return Response()

    state = deliver(
        tmp_path,
        "https://otel-collector.app.daytona.io",
        organization_id="owned-organisation",
        opener=accept,
    )
    assert not batch.exists()
    assert state["signals"] == {"logs": 1}
    assert state["pending_batches"] == 0
    assert calls[0].full_url.endswith("/v1/logs")
    assert calls[0].get_header("Content-type") == "application/json"
    assert calls[0].get_header("Organization-id") == "owned-organisation"


def test_http_success_with_rejected_records_is_retained(tmp_path):
    batch = queued(tmp_path)
    state = deliver(
        tmp_path,
        "https://otel-collector.app.daytona.io",
        opener=lambda *args, **kwargs: Response(b'{"partialSuccess":{"rejectedLogRecords":"1"}}'),
    )
    assert state["status"] == "retrying"
    assert batch.exists()
