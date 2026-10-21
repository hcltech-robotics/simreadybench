"""Deliver worker activity through the coordinator's telemetry connection."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from urllib import request
from urllib.parse import urlparse


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _attributes(value):
    result = []
    for key, item in value.items():
        if type(item) is bool:
            encoded = {"boolValue": item}
        elif type(item) is int:
            encoded = {"intValue": str(item)}
        elif type(item) is float and math.isfinite(item):
            encoded = {"doubleValue": item}
        elif type(item) is str:
            encoded = {"stringValue": item}
        else:
            continue
        result.append({"key": key, "value": encoded})
    return result


def _resource(record, environment):
    sandbox = record["sandbox_id"]
    return {
        "attributes": _attributes(
            {
                "service.name": "sandbox-" + sandbox,
                "service.instance.id": sandbox,
                "daytona_sandbox_id": sandbox,
                "daytona_organization_id": environment.get("organization_id"),
                "daytona_region_id": environment.get("region_id"),
                "simready.run_id": environment.get("run_id", record.get("run_id")),
                "simready.model": record.get("model"),
                "simready.observer": "coordinator",
            }
        )
    }


def activity(record, ledger, *, now):
    """Select counters and request metadata without prompts or credentials."""
    events = []
    for ordinal, raw in enumerate(ledger.get("events", []), start=1):
        started = raw.get("started_at")
        if type(started) not in (int, float) or not math.isfinite(started):
            continue
        item = {"request_ordinal": ordinal, "started_at": started}
        for key in ("status", "http_status", "error_type", "retry_delay_seconds"):
            if type(raw.get(key)) in (str, int, float, bool):
                item[key] = raw[key]
        finished = raw.get("finished_at")
        if type(finished) in (int, float) and math.isfinite(finished):
            item.update(finished_at=finished, elapsed_seconds=max(0, finished - started))
        scope = raw.get("scope")
        if isinstance(scope, str):
            item["condition_scope_sha256"] = hashlib.sha256(scope.encode()).hexdigest()
        events.append(item)
    inflight = [
        event
        for event in events
        if event.get("status") == "admitted" and "finished_at" not in event
    ]
    latest = max((event.get("finished_at", event["started_at"]) for event in events), default=None)
    summary = {
        "event": "worker_heartbeat",
        "timestamp": now,
        "sandbox_id": record["sandbox_id"],
        "status": record.get("controller_status", "running"),
        "model": record.get("model"),
        "completed_conditions": record.get("completed_conditions", 0),
        "planned_conditions": record.get("planned_conditions", 0),
        "generation_requests": ledger.get("generation_requests", 0),
        "maximum_requests": ledger.get("maximum_requests", 0),
        "inflight_requests": len(inflight),
        "oldest_request_age_seconds": max(
            (max(0, now - event["started_at"]) for event in inflight), default=0
        ),
        "provider_activity_age_seconds": max(0, now - latest) if latest is not None else None,
        "coordinator_observation_age_seconds": max(0, now - record["last_observed_at"])
        if record.get("last_observed_at")
        else None,
    }
    return summary, events


def payloads(record, ledger, environment, *, now, seen=()):
    summary, events = activity(record, ledger, now=now)
    resource = _resource(record, environment)
    timestamp = str(int(now * 1e9))
    logs = [
        {
            "timeUnixNano": timestamp,
            "severityNumber": 9,
            "severityText": "INFO",
            "body": {"stringValue": json.dumps(summary, sort_keys=True)},
            "attributes": _attributes({"simready." + key: value for key, value in summary.items()}),
        }
    ]
    spans = []
    identities = set(seen)
    for event in events:
        identity = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
        if identity in identities:
            continue
        identities.add(identity)
        name = "model_request_started" if "finished_at" not in event else "model_request_finished"
        body = {"event": name, "sandbox_id": record["sandbox_id"], **event}
        end = event.get("finished_at", now)
        correlation = hashlib.sha256(
            (
                record["sandbox_id"]
                + ":"
                + str(event["started_at"])
                + ":"
                + str(event["request_ordinal"])
            ).encode()
        ).hexdigest()
        trace_id, span_id = correlation[:32], correlation[32:48]
        entry = {
            "timeUnixNano": str(int(end * 1e9)),
            "severityNumber": 17 if event.get("http_status", 0) >= 400 else 9,
            "body": {"stringValue": json.dumps(body, sort_keys=True)},
            "traceId": trace_id,
            "spanId": span_id,
            "attributes": _attributes({"simready." + key: value for key, value in body.items()}),
        }
        logs.append(entry)
        if "finished_at" in event:
            spans.append(
                {
                    "traceId": trace_id,
                    "spanId": span_id,
                    "name": "simready.model_request",
                    "kind": 3,
                    "startTimeUnixNano": str(int(event["started_at"] * 1e9)),
                    "endTimeUnixNano": str(int(end * 1e9)),
                    "attributes": entry["attributes"],
                    "status": {
                        "code": 2
                        if event.get("http_status", 0) >= 400
                        or event.get("status") in ("failed", "cancelled")
                        else 1
                    },
                }
            )
    metrics = []
    for name, key, unit in (
        ("simready.worker.completed_conditions", "completed_conditions", "{condition}"),
        ("simready.worker.planned_conditions", "planned_conditions", "{condition}"),
        ("simready.model.generation_requests", "generation_requests", "{request}"),
        ("simready.model.maximum_requests", "maximum_requests", "{request}"),
        ("simready.model.inflight_requests", "inflight_requests", "{request}"),
        ("simready.model.oldest_request_age", "oldest_request_age_seconds", "s"),
        ("simready.model.provider_activity_age", "provider_activity_age_seconds", "s"),
        ("simready.coordinator.observation_age", "coordinator_observation_age_seconds", "s"),
    ):
        number = summary.get(key)
        if type(number) not in (int, float) or not math.isfinite(number):
            continue
        metrics.append(
            {
                "name": name,
                "unit": unit,
                "gauge": {"dataPoints": [{"timeUnixNano": timestamp, "asDouble": number}]},
            }
        )
    result = {
        "logs": {
            "resourceLogs": [
                {
                    "resource": resource,
                    "scopeLogs": [
                        {"scope": {"name": "simready_bench.observer"}, "logRecords": logs}
                    ],
                }
            ]
        },
        "metrics": {
            "resourceMetrics": [
                {
                    "resource": resource,
                    "scopeMetrics": [
                        {"scope": {"name": "simready_bench.observer"}, "metrics": metrics}
                    ],
                }
            ]
        },
    }
    if spans:
        result["traces"] = {
            "resourceSpans": [
                {
                    "resource": resource,
                    "scopeSpans": [{"scope": {"name": "simready_bench.observer"}, "spans": spans}],
                }
            ]
        }
    return result, sorted(identities)[-2048:]


def deliver(
    output_dir, endpoint, *, organization_id=None, maximum_batches=12, opener=request.urlopen
):
    """Retain failed batches and confirm collector acceptance before removing them."""
    root = Path(output_dir) / "relay-telemetry"
    root.mkdir(mode=0o700, exist_ok=True)
    parsed = urlparse(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("An HTTPS telemetry collector without credentials is required")
    state_path = root / "delivery.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"delivered_batches": 0, "signals": {}}
    )
    for path in sorted(root.glob("pending-*.json"))[:maximum_batches]:
        batch = json.loads(path.read_text())
        signal = batch["signal"]
        if signal not in {"logs", "metrics", "traces"}:
            raise ValueError("Unsupported telemetry signal")
        try:
            req = request.Request(
                endpoint.rstrip("/") + "/v1/" + signal,
                data=json.dumps(batch["payload"], allow_nan=False).encode(),
                headers={
                    "Content-Type": "application/json",
                    **({"organization-id": organization_id} if organization_id else {}),
                },
                method="POST",
            )
            with opener(req, timeout=5) as response:
                body = response.read(8192)
                if response.status != 200:
                    raise RuntimeError("The collector did not accept the telemetry batch")
                result = json.loads(body) if body else {}
                partial = result.get("partialSuccess") or result.get("partial_success") or {}
                if any(
                    int(partial.get(key, 0))
                    for key in (
                        "rejectedLogRecords",
                        "rejectedSpans",
                        "rejectedDataPoints",
                        "rejected_log_records",
                        "rejected_spans",
                        "rejected_data_points",
                    )
                ):
                    raise RuntimeError("The collector rejected telemetry records")
            path.unlink()
            state.update(
                status="delivered",
                last_delivered_at=time.time(),
                last_signal=signal,
                http_status=200,
                error_type=None,
            )
            state["delivered_batches"] += 1
            state["signals"][signal] = state["signals"].get(signal, 0) + 1
        except (OSError, RuntimeError, ValueError) as error:
            state.update(
                status="retrying", last_attempt_at=time.time(), error_type=type(error).__name__
            )
            _save(state_path, state)
            return state
    state["pending_batches"] = sum(1 for _ in root.glob("pending-*.json"))
    _save(state_path, state)
    return state


def forward_relay_telemetry(sandbox, record, output_dir, *, environment=None, now=None):
    """Export current relay activity without changing the executor or its protocol."""
    now = time.time() if now is None else now
    output = Path(output_dir)
    root = output / "relay-telemetry"
    root.mkdir(mode=0o700, exist_ok=True)
    metadata_path = root / "destination.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        index = json.loads(sandbox.fs.download_file("/work/output/telemetry/outbox.json", 10))
        metadata = {"endpoint": index["endpoint"]}
        _save(metadata_path, metadata)
    ledger = json.loads(sandbox.fs.download_file("/run/simready-benchmark/ledger.json", 10))
    seen_path = root / "seen.json"
    previous = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    signals, seen = payloads(
        record,
        ledger,
        environment
        or {
            "organization_id": getattr(sandbox, "organization_id", None),
            "region_id": getattr(sandbox, "target", None),
        },
        now=now,
        seen=previous.get("events", ()),
    )
    for signal, payload in signals.items():
        batch = {"signal": signal, "payload": payload}
        identity = hashlib.sha256(json.dumps(batch, sort_keys=True).encode()).hexdigest()
        _save(root / ("pending-" + identity + ".json"), batch)
    _save(seen_path, {"events": seen, "observed_at": now})
    return deliver(
        output,
        metadata["endpoint"],
        organization_id=(environment or {}).get(
            "organization_id", getattr(sandbox, "organization_id", None)
        ),
    )
