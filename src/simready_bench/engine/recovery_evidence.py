"""Bind recovered transport failures to native actor and immutable measurement evidence."""

import hashlib
import json
from pathlib import Path
import re

PIN_HOLD_CODES = frozenset({
    "hf_pricing_metadata_changed", "hf_pricing_metadata_invalid", "hf_cost_model_not_authorized",
    "hf_model_route_binding_required", "hf_model_route_evidence_required", "hf_model_declaration_route_mismatch",
    "native_output_route_not_authorized", "cost_model_not_authorized", "cost_settlement_original_route_mismatch",
})
NATIVE_EVIDENCE_HOLD_CODE = "native_terminal_evidence_unavailable"


def _native_session_paths(result):
    return [Path(path) for path in result.get("harbor_trace_paths", [])
            if Path(path).suffix == ".jsonl" and Path(path).parent.name == "sessions"
            and Path(path).parent.parent.name == "pi"]


def _controlled_stream_interruption(result):
    rows = result.get("gateway_records") or []
    if not rows:
        return False
    row = rows[-1]
    return (
        row.get("status") == "cancelled" and row.get("observed_upstream_status") == 200
        and row.get("upstream_status") in (None, 200) and row.get("error_code") is None
        and not (set(row.get("provider_response_statuses", ())) |
                 set(row.get("observed_provider_response_statuses", ()))) & {"failed", "cancelled"}
    )


def _native_messages(result):
    from harbor.agents.installed.pi import Pi

    sessions = _native_session_paths(result)
    if len(sessions) != 1:
        return None
    path = sessions[0]
    if path.is_symlink() or path.stat().st_size > 128 * 1024 * 1024:
        return None
    raw = path.read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    active = Pi._active_session_events(list(enumerate(events, 1)))
    messages = [event["message"] for _, event in active
                if event.get("type") == "message" and event.get("message", {}).get("role") == "assistant"]
    return messages, hashlib.sha256(raw).hexdigest()


def _native_final(result, normal_stop):
    retained = _native_messages(result)
    if retained is None:
        return None
    messages, digest = retained
    if not messages:
        return None
    final = messages[-1]
    interrupted = final.get("stopReason") == "aborted" or (
        final.get("stopReason") == "error" and final.get("errorMessage") == "This operation was aborted"
    )
    if normal_stop and interrupted and (not final.get("content") or _controlled_stream_interruption(result)):
        messages.pop()
        if not messages:
            return None
        final = messages[-1]
    if final.get("stopReason") not in {"stop", "toolUse"} or final.get("errorMessage"):
        return None
    response_id = final.get("responseId")
    if not isinstance(response_id, str) or not response_id:
        return None
    return {"response_id": response_id, "native_session_sha256": digest}


def native_terminal_failure(result, *, normal_stop):
    """Read the native error field, never assistant prose or HTTP success alone."""
    expected = bool(_native_session_paths(result))
    unavailable = {"error_code": NATIVE_EVIDENCE_HOLD_CODE, "native_session_sha256": None}
    try:
        retained = _native_messages(result)
        if retained is None or not retained[0]:
            return unavailable if expected else None
        messages, digest = retained
        final = messages[-1]
        reason, message = final.get("stopReason"), final.get("errorMessage")
        if normal_stop and (not final.get("content") or _controlled_stream_interruption(result)) and (
            reason == "aborted" or reason == "error" and message == "This operation was aborted"
        ):
            return None
        if reason not in {"error", "aborted"}:
            return None
        code = None
        if isinstance(message, str) and len(message) <= 8192:
            text = re.sub(r"^(?:HTTP\s+)?[45][0-9]{2}[: ]+", "", message).strip()
            if text in PIN_HOLD_CODES:
                code = text
            else:
                try:
                    value = json.loads(text)
                    error = value.get("error", {})
                    candidate = error.get("code") if isinstance(error, dict) else error
                    if candidate in PIN_HOLD_CODES:
                        code = candidate
                except (ValueError, TypeError, AttributeError):
                    pass
        return {"error_code": code or "native_actor_terminal_error", "native_session_sha256": digest}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return unavailable if expected else None


def infrastructure_hold(result, *, normal_stop):
    for row in result.get("gateway_records", []):
        if row.get("error_code") in PIN_HOLD_CODES:
            return {"code": row["error_code"], "source": "trusted_gateway_record",
                    "gateway_request_id": row.get("gateway_request_id")}
    native = native_terminal_failure(result, normal_stop=normal_stop)
    if native is not None and native["error_code"] in PIN_HOLD_CODES | {NATIVE_EVIDENCE_HOLD_CODE}:
        return {"code": native["error_code"], "source": "native_terminal_error",
                "native_session_sha256": native["native_session_sha256"]}
    return None


def _submission_matches(result, report):
    submission = result.get("submission")
    snapshot_hash = result.get("snapshot_sha256")
    if not isinstance(submission, dict):
        return report.get("outcome") == "invalid_submission" and snapshot_hash is None
    if submission.get("snapshot_sha256") != snapshot_hash or submission.get("copied_independently") is not True:
        return False
    manifest = json.loads(Path(submission["manifest"]).read_text())
    if manifest.get("snapshot_sha256") != snapshot_hash or manifest.get("copied_independently") is not True:
        return False
    directory = Path(submission["directory"])
    for item in manifest["files"]:
        relative = Path(item["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = directory / relative
        if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
            return False
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                return False
    return bool(manifest["files"])


def verified_request_recovery(result, *, normal_stop):
    """A later HTTP success is insufficient without all three terminal bindings."""
    try:
        report = result.get("evaluation") or {}
        audit = result.get("actor_stop_audit") or {}
        state = audit.get("state", {})
        receipt = result.get("actor_stop_receipt") or {}
        episode = result.get("trial_id")
        if (
            not episode or result.get("actor_stopped") is not True
            or state.get("Running") is not False or state.get("Paused") is not False
            or state.get("OOMKilled") is not False
            or report.get("episode_id") != episode
            or report.get("snapshot_sha256") != result.get("snapshot_sha256")
            or not report.get("evaluation_binding")
            or report.get("outcome") not in {"complete", "incomplete", "invalid_submission"}
            or not _submission_matches(result, report)
        ):
            return None
        if normal_stop and (
            receipt.get("episode_id") != episode
            or receipt.get("snapshot_id") != report.get("snapshot_id")
            or receipt.get("snapshot_sha256") != result.get("snapshot_sha256")
            or receipt.get("container_id") != audit.get("container_id")
            or receipt.get("termination_reason") != result.get("termination_reason")
        ):
            return None
        native = _native_final(result, normal_stop)
        if native is None:
            return None
        rows = result.get("gateway_records", [])
        successes = [index for index, row in enumerate(rows) if
                     row.get("status") == "completed" and row.get("upstream_status") == 200
                     and not set(row.get("provider_response_statuses", ())) & {"failed", "cancelled"}]
        if not successes:
            return None
        index = successes[-1]
        success = rows[index]
        if native["response_id"] not in success.get("provider_response_ids", ()):
            return None
        resolved = []
        for row in rows[:index]:
            if row.get("status") == "completed" and row.get("upstream_status") == 200:
                if set(row.get("provider_response_statuses", ())) & {"failed", "cancelled"}:
                    return None
                continue
            if row.get("upstream_status") not in (None, 200, 408, 429, 500, 502, 503, 504):
                return None
            if row.get("error_code") not in {
                None, "upstream_transport_error", "upstream_stream_incomplete",
                "inference_deadline_exceeded", "hf_pricing_metadata_unavailable", "hf_pricing_route_unavailable",
            }:
                return None
            if not row.get("gateway_request_id"):
                return None
            resolved.append(row["gateway_request_id"])
        if not resolved or not success.get("gateway_request_id"):
            return None
        return {"schema": "simready-request-recovery-v1", "disposition": "verified_recovered",
                "resolved_request_ids": resolved, "successful_request_id": success["gateway_request_id"],
                "episode_id": episode, "snapshot_id": report["snapshot_id"],
                "snapshot_sha256": report["snapshot_sha256"], "evaluation_binding": report["evaluation_binding"], **native}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
