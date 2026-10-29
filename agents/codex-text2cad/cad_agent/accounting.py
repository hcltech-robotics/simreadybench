"""Auditable ACP compatibility and allowlisted evidence, independent of scoring."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _session_files(root):
    root = Path(root)
    if (root.name != "sessions" or root.parent.name != ".codex-text2cad" or root.is_symlink()
            or root.parent.is_symlink()):
        raise ValueError("Only the task-specific .codex-text2cad/sessions directory is allowed")
    resolved = root.resolve()
    for path in sorted(root.rglob("rollout-*.jsonl")) if root.exists() else []:
        if not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(resolved):
            yield path


def _usage(raw):
    names = ("input_tokens", "cached_input_tokens", "output_tokens")
    if not isinstance(raw, dict) or any(type(raw.get(k)) is not int or raw[k] < 0 for k in names):
        raise ValueError("Missing or invalid native cumulative counters")
    result = {k: raw[k] for k in names}
    if result["cached_input_tokens"] > result["input_tokens"]:
        raise ValueError("Cached input exceeds inclusive input")
    for key in ("reasoning_output_tokens", "total_tokens"):
        if type(raw.get(key)) is int and raw[key] >= 0:
            result[key] = raw[key]
    return result


def native_snapshot(root, session_id, model, effort):
    """Take the final cumulative checkpoint, never sum cumulative checkpoints."""
    failure = {"status": "unverified", "reason": "missing_native_session_usage"}
    snapshots = []
    for path in _session_files(root):
        session, turns, previous, latest, last = None, [], None, None, None
        reset = False
        with path.open() as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # A partially flushed final line is not a receipt.
                payload = event.get("payload") or {}
                if event.get("type") == "session_meta":
                    session = payload.get("id")
                elif event.get("type") == "turn_context":
                    turns.append({"model": payload.get("model"),
                                  "effort": payload.get("effort", payload.get("reasoning_effort"))})
                elif event.get("type") == "event_msg" and payload.get("type") == "token_count":
                    info = payload.get("info") or {}
                    try:
                        counters = _usage(info.get("total_token_usage"))
                    except ValueError:
                        continue
                    if previous and any(counters[k] < previous[k] for k in ("input_tokens", "cached_input_tokens", "output_tokens")):
                        reset = True
                    previous, latest, last = counters, counters, info.get("last_token_usage")
        if session != session_id:
            continue
        if reset:
            return {"status": "unverified", "reason": "native_cumulative_counter_reset"}
        if not turns or any(t != {"model": model, "effort": effort} for t in turns):
            return {"status": "unverified", "reason": "native_model_or_reasoning_mismatch", "turns": turns}
        if latest:
            snapshots.append({"status": "verified", "basis": "native_session_cumulative",
                "scope": "selected_root_session_only", "input_semantics": "inclusive_of_cached_input",
                "totals": latest, "last_usage": last, "turns": turns,
                "native_file": str(path.relative_to(root)), "native_sha256": _digest(path.read_bytes())})
    if len(snapshots) > 1:
        return {"status": "unverified", "reason": "ambiguous_multiple_rollouts_for_session"}
    return snapshots[0] if snapshots else failure


class AccountingBridge:
    def __init__(self, codex_home, logs, model, effort):
        self.home, self.logs = Path(codex_home), Path(logs)
        if self.home.name != ".codex-text2cad":
            raise ValueError("Expected task-specific .codex-text2cad home")
        self.model, self.effort = model, effort
        self.pending, self.receipts, self.sessions = {}, {}, {}
        self.lock = threading.RLock()
        self.logs.mkdir(parents=True, exist_ok=True)

    def request(self, message):
        with self.lock:
            result = copy.deepcopy(message)
            method, params = message.get("method"), message.get("params") or {}
            if method == "session/set_model":
                requested = params.get("modelId", "").removeprefix("openai/")
                if "[" not in requested:
                    requested += f"[{self.effort}]"
                if requested != f"{self.model}[{self.effort}]":
                    raise ValueError("Requested model/reasoning differs from configured evaluation")
                result["params"]["modelId"] = requested
            if method == "session/prompt":
                receipt = self.receipts.get(params.get("sessionId"))
                if receipt != f"{self.model}[{self.effort}]":
                    raise ValueError("Actual session model/reasoning receipt is missing or mismatched")
            if "id" in message and "method" in message:
                self.pending[message["id"]] = {"method": method, "params": result.get("params") or {}}
            return result

    def response(self, message):
        with self.lock:
            if "result" not in message and "error" not in message:
                return message  # Server→client request IDs have a separate namespace.
            request = self.pending.pop(message.get("id"), None)
            if not request or "result" not in message:
                return message
            result = copy.deepcopy(message)
            payload, method = result["result"], request["method"]
            if method == "session/new":
                sid = payload.get("sessionId")
                self.receipts[sid] = (payload.get("models") or {}).get("currentModelId")
            elif method == "session/set_model":
                # Successful acknowledgement of an explicit model[effort] request.
                self.receipts[request["params"].get("sessionId")] = request["params"]["modelId"]
            elif method == "session/prompt":
                sid = request["params"].get("sessionId")
                snapshot = native_snapshot(self.home / "sessions", sid, self.model, self.effort)
                # Give Codex's journal writer time to finish the just-completed response.
                deadline = time.monotonic() + 1
                while snapshot.get("reason") == "missing_native_session_usage" and time.monotonic() < deadline:
                    time.sleep(0.05)
                    snapshot = native_snapshot(self.home / "sessions", sid, self.model, self.effort)
                snapshot["upstream_usage"] = copy.deepcopy(payload.get("usage"))
                self.sessions[sid] = snapshot
                if snapshot["status"] == "verified":
                    total = snapshot["totals"]
                    usage = {"inputTokens": total["input_tokens"], "cachedReadTokens": total["cached_input_tokens"],
                             "outputTokens": total["output_tokens"],
                             "totalTokens": total.get("total_tokens", total["input_tokens"] + total["output_tokens"])}
                    if "reasoning_output_tokens" in total:
                        usage["thoughtTokens"] = total["reasoning_output_tokens"]
                    payload["usage"] = usage
                payload.setdefault("_meta", {})["cad_agent_accounting"] = snapshot
            self.save()
            return result

    def save(self):
        seen = set()
        for path in _session_files(self.home / "sessions"):
            try:
                with path.open() as handle:
                    header = json.loads(handle.readline())
                if header.get("type") == "session_meta":
                    seen.add(header["payload"]["id"])
            except (OSError, ValueError, KeyError):
                continue
        (self.logs / "accounting.json").write_text(json.dumps({"sessions": self.sessions,
            "session_reasoning_receipts": self.receipts, "pricing": "not_derived",
            "other_native_session_ids": sorted(seen - self.sessions.keys()),
            "note": "Root-session cumulative totals; subagent sessions are retained but not summed."}, indent=2) + "\n")

    def finalize(self):
        with self.lock:
            for sid, previous in self.sessions.items():
                snapshot = native_snapshot(self.home / "sessions", sid, self.model, self.effort)
                snapshot["upstream_usage"] = previous.get("upstream_usage")
                self.sessions[sid] = snapshot
            self.save()


def _redact(data, secrets):
    text = data.decode("utf-8")
    for secret in secrets:
        for value in {secret, json.dumps(secret)[1:-1]}:
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r'([?&]token=)[^\s"\\]+', r'\1[REDACTED]', text)
    text = re.sub(r'("(?:[a-z_]*api_key|access_token|refresh_token|password|secret)"\s*:\s*)"[^"\n]*"', r'\1"[REDACTED]"', text, flags=re.I)
    return text.encode()


def retain_evidence(codex_home, authoring, logs, environ=None):
    """Copy only rollout JSONL and receipt-linked CAD files; never auth/config homes."""
    home, authoring, logs = Path(codex_home), Path(authoring), Path(logs)
    env = os.environ if environ is None else environ
    secrets = {v for k, v in env.items() if isinstance(v, str) and len(v) >= 4 and
               re.search(r'(?:TOKEN|SECRET|PASSWORD|API_KEY)', k)}
    entries = [(p, Path("native-sessions") / p.relative_to(home / "sessions")) for p in _session_files(home / "sessions")]
    receipt = authoring / "cad-evidence/generation.json"
    if receipt.is_file() and not receipt.is_symlink() and not authoring.is_symlink():
        data = json.loads(receipt.read_text())
        candidates = [(receipt, {".json"}), (Path(data.get("source_path", "")), {".py"})]
        candidates.extend((Path(p.get("step_path", "")), {".step", ".stp"}) for p in data.get("parts", []))
        for path, suffixes in candidates:
            if (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(authoring.resolve())
                    and path.suffix.lower() in suffixes):
                entries.append((path, Path("authoring") / path.resolve().relative_to(authoring.resolve())))
    manifest = []
    for source, relative in entries:
        # Cap diagnostics rather than following arbitrary references or copying a home.
        if source.stat().st_size > 128 * 1024 * 1024:
            manifest.append({"file": str(relative), "status": "skipped_size_limit"})
            continue
        raw = source.read_bytes()
        if relative.parts[0] == "native-sessions":
            try:
                header = json.loads(raw.splitlines()[0])
                if header.get("type") != "session_meta" or not isinstance(header.get("payload", {}).get("id"), str):
                    continue
            except (ValueError, IndexError):
                continue
        retained = _redact(raw, secrets)
        target = logs / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(retained)
        manifest.append({"file": str(relative), "source_sha256": _digest(raw),
                         "retained_sha256": _digest(retained), "redacted": retained != raw})
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "retention.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def normalize_atif(trajectory, accounting):
    """Explicit post-download repair for Harbor 0.24's missing ATIF cache totals."""
    trajectory, accounting = Path(trajectory), Path(accounting)
    data = json.loads(trajectory.read_text())
    receipt = json.loads(accounting.read_text())
    if receipt.get("other_native_session_ids"):
        raise ValueError("Additional native sessions require explicit accounting reconciliation")
    records = receipt["sessions"]
    record = records.get(data.get("session_id"))
    if not record or record.get("status") != "verified":
        raise ValueError("No verified cumulative native usage for this ATIF session")
    basis = {"basis": record["basis"], "input_semantics": "inclusive_of_cached_input",
             "accounting_sha256": _digest(accounting.read_bytes()), "cost_policy": "upstream_unchanged"}
    marker = (data.get("extra") or {}).get("cad_agent_accounting")
    if marker:
        if marker != basis:
            raise ValueError("ATIF already normalized from a different accounting receipt")
        return
    backup = trajectory.with_name(trajectory.stem + ".upstream.json")
    if backup.exists():
        raise ValueError("Upstream ATIF backup already exists without matching normalization")
    backup.write_bytes(trajectory.read_bytes())
    total = record["totals"]
    final = data["final_metrics"] = data.get("final_metrics") or {}
    final.update(total_prompt_tokens=total["input_tokens"], total_cached_tokens=total["cached_input_tokens"],
                 total_completion_tokens=total["output_tokens"])
    for step in reversed(data.get("steps", [])):
        if step.get("source") == "agent":
            step["metrics"] = step.get("metrics") or {}
            step["metrics"].update(prompt_tokens=total["input_tokens"],
                cached_tokens=total["cached_input_tokens"], completion_tokens=total["output_tokens"])
            break
    data.setdefault("extra", {})["cad_agent_accounting"] = basis
    trajectory.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("accounting", type=Path)
    args = parser.parse_args()
    normalize_atif(args.trajectory, args.accounting)
