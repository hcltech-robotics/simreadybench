"""Small guarded stdio proxy around the registry-listed pi-acp bridge."""

import time
STARTED_NS = time.monotonic_ns()

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from uuid import uuid4

from .accounting import snapshot
from .bootstrap import install
from .hosted import configure
from .runtime_inputs import attach_observation, save_runtime

ROOT = Path(__file__).parent


@dataclass(frozen=True)
class Continuation:
    request: dict


class Bridge:
    def __init__(self, profile, effort, sessions, logs, *, runtime_file=None):
        self.profile, self.effort, self.sessions, self.logs = profile, effort, Path(sessions), Path(logs)
        self.public = "nvidia/" + profile.id
        self.native = "harbor-endpoint/" + profile.id
        self.pending, self.ready = {}, set()
        self.origins, self.recovered, self.cancelled = {}, set(), set()
        self.runtime_file = Path(runtime_file) if runtime_file else None
        self.image_sessions = set()
        self.lock = threading.RLock()

    def request(self, message):
        with self.lock:
            result = copy.deepcopy(message)
            method, params = result.get("method"), result.get("params") or {}
            if method == "session/cancel":
                self.cancelled.add(params.get("sessionId"))
            if method == "session/new" and params.get("mcpServers"):
                raise ValueError("Task MCP servers cannot be silently ignored by this plain bridge")
            if method in {"session/load", "session/fork"}:
                raise ValueError("Session resume/fork accounting is not verified for this hosted variant")
            if method == "session/set_model":
                if params.get("modelId") not in {self.public, self.native, self.profile.id}:
                    raise ValueError("Model selection differs from this configured agent entry")
                params["modelId"] = self.native
            if method == "session/set_mode" and params.get("modeId") != self.effort:
                raise ValueError("Thinking selection differs from this configured agent entry")
            if method == "session/set_config_option":
                if params.get("configId") == "model":
                    if params.get("value") not in {self.public, self.native, self.profile.id}:
                        raise ValueError("Model configuration mismatch")
                    params["value"] = self.native
                elif params.get("configId") == "thought_level":
                    if params.get("value") != self.effort:
                        raise ValueError("Thinking configuration mismatch")
                else:
                    raise ValueError("Unverified session configuration option")
            if method == "session/prompt":
                if params.get("sessionId") not in self.ready:
                    raise ValueError("Matching native model/thinking receipt is required before prompting")
                if result.get("id") not in self.origins:
                    self.cancelled.discard(params["sessionId"])
                if self.runtime_file and params["sessionId"] not in self.image_sessions:
                    runtime = json.loads(self.runtime_file.read_text())
                    try:
                        params["prompt"], image_receipt = attach_observation(params["prompt"], runtime)
                    except (ValueError, OSError) as error:
                        (self.logs / "pi-input-image.json").write_text(json.dumps({
                            "status": "error", "reason": str(error)}) + "\n")
                        raise ValueError("Frozen observation attachment failed; inspect pi-input-image.json") from error
                    self.image_sessions.add(params["sessionId"])
                    runtime["expected_image_sha256"] = image_receipt["source_sha256"]
                    save_runtime(self.runtime_file, runtime)
                    (self.logs / "pi-input-image.json").write_text(json.dumps(image_receipt, indent=2) + "\n")
                caps = (f"This entry's output cap is {self.profile.maxTokens} tokens per inference response, including reasoning; "
                        f"it is not an authoring-turn or validator-call budget. Reasoning level: {self.effort}. "
                        "The original runner wall-clock deadline covers the whole task and any bounded continuation; it is never reset.\n")
                if self.runtime_file:
                    caps += "The frozen task observation was hash-verified and attached to the initial session prompt; use that image directly.\n"
                params["prompt"] = [{"type": "text", "text": caps + (ROOT / "feedback.txt").read_text()}] + list(params["prompt"])
            if "id" in result and method:
                self.pending[result["id"]] = (method, params)
            return result

    def _configuration(self, payload):
        models, modes = payload.get("models") or {}, payload.get("modes") or {}
        if models.get("currentModelId") and models["currentModelId"] != self.native:
            raise ValueError("Pi reported a different actual model")
        if modes.get("currentModeId") and modes["currentModeId"] != self.effort:
            raise ValueError("Pi reported a different actual thinking level")
        if models:
            models["currentModelId"] = self.public
            models["availableModels"] = [{"modelId": self.public, "name": self.public}]
        for option in payload.get("configOptions") or []:
            if option.get("id") == "model":
                if option.get("currentValue") != self.native:
                    raise ValueError("Pi model receipt mismatch")
                option["currentValue"] = self.public
                option["options"] = [{"value": self.public, "name": self.public}]
            if option.get("id") == "thought_level":
                if option.get("currentValue") != self.effort:
                    raise ValueError("Pi thinking receipt mismatch")
                option["options"] = [{"value": self.effort, "name": self.effort}]

    def response(self, message):
        with self.lock:
            result = copy.deepcopy(message)
            if result.get("method") == "session/update":
                update = (result.get("params") or {}).get("update") or {}
                if update.get("sessionUpdate") == "usage_update" and update.get("cost") is not None:
                    update.setdefault("_meta", {})["unverified_upstream_cost"] = update.pop("cost")
                return result
            if "result" not in result and "error" not in result:
                return result
            request = self.pending.pop(result.get("id"), None)
            root_id = self.origins.pop(result.get("id"), result.get("id"))
            if "id" in result:
                result["id"] = root_id
            if not request:
                return result
            method, params = request
            if method == "session/prompt":
                receipt = snapshot(self.sessions, params["sessionId"], self.profile, self.effort)
                receipt["length_recovery_count"] = int(root_id in self.recovered)
                receipt["length_recovery_limit"] = 1
                if self.runtime_file:
                    wire = self.logs / "pi-image-wire.jsonl"
                    entries = [json.loads(line) for line in wire.read_text().splitlines() if line] if wire.exists() else []
                    first = next((e for e in entries if e.get("expected_sha256")), {})
                    receipt["image_delivery"] = {"initial_payload_verified": first.get("matched") is True,
                        "receipt": "pi-image-wire.jsonl", "source_sha256": first.get("expected_sha256")}
                if "error" in result:
                    receipt["completion"] = "upstream_error"
                self.logs.mkdir(parents=True, exist_ok=True)
                (self.logs / "pi-accounting.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if "result" not in result:
                return result
            payload = result["result"]
            if method == "initialize":
                payload["agentInfo"] = {"name": "pi-nvidia-feedback", "version": "0.2.3"}
                payload["authMethods"] = []  # Headless injected auth; never terminal login.
                payload.setdefault("agentCapabilities", {})["loadSession"] = False
            if method in {"session/new", "session/set_model", "session/set_mode", "session/set_config_option"}:
                self._configuration(payload)
                if method == "session/new":
                    if (payload.get("modes") or {}).get("currentModeId") != self.effort:
                        raise ValueError("Pi actual thinking receipt is missing")
                    if (payload.get("models") or {}).get("currentModelId") != self.public:
                        raise ValueError("Pi actual model receipt is missing")
                    self.ready.add(payload["sessionId"])
            if method == "session/prompt":
                if params["sessionId"] in self.cancelled:
                    payload["stopReason"] = "cancelled"
                payload.setdefault("_meta", {})["pi_nvidia_accounting"] = receipt
                payload.pop("usage", None)
                if receipt["status"] == "verified":
                    payload["usage"] = receipt["usage"]
                if receipt.get("provider_error") and payload.get("stopReason") != "cancelled":
                    return {"jsonrpc": "2.0", "id": root_id, "error": {
                        "code": -32000, "message": "Pi reported a provider error; inspect native session diagnostics"}}
                if self.runtime_file and not receipt["image_delivery"]["initial_payload_verified"] and params["sessionId"] not in self.cancelled:
                    return {"jsonrpc": "2.0", "id": root_id, "error": {"code": -32000,
                        "message": "Initial frozen observation was not verified in the provider payload; inspect image receipts"}}
                last = receipt.get("last_assistant") or {}
                if payload.get("stopReason") == "end_turn" and last.get("stop_reason") == "length":
                    payload["stopReason"] = "max_tokens"
                    if receipt["status"] == "verified" and last.get("empty_length_stop") and root_id not in self.recovered:
                        self.recovered.add(root_id)
                        internal_id = "pi-nvidia-length-" + uuid4().hex
                        self.origins[internal_id] = root_id
                        prompt = "Your prior response reached its output limit without text or a tool call. Continue the same task with concrete tool actions: preserve existing work, write and reopen the first complete /output/submission/asset.usd candidate, then use the genuine feedback validator and repair actionable failures. Do not restart exploration. The original deadline and five-validator-call allowance still apply. This is the single automatic length recovery; do not rely on another follow-up."
                        follow_up = self.request({"jsonrpc": "2.0", "id": internal_id,
                            "method": "session/prompt", "params": {"sessionId": params["sessionId"],
                            "prompt": [{"type": "text", "text": prompt}]}})
                        receipt.update(length_recovery_count=1, completion="continuation_requested")
                        (self.logs / "pi-accounting.json").write_text(json.dumps(receipt, indent=2) + "\n")
                        with (self.logs / "pi-length-recovery.jsonl").open("a") as log:
                            log.write(json.dumps({"reason": "empty_length_stop", "root_request_id": root_id,
                                "limit": 1, "request": follow_up}) + "\n")
                        return Continuation(follow_up)
            return result


def proxy(command, env, bridge):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               env=env, start_new_session=True)
    output_lock = threading.Lock()
    input_lock = threading.Lock()
    errors = []
    def send(value):
        with input_lock:
            process.stdin.write(json.dumps(value).encode() + b"\n")
            process.stdin.flush()
    def emit(value):
        with output_lock:
            sys.stdout.buffer.write(json.dumps(value).encode() + b"\n")
            sys.stdout.buffer.flush()
    def failure(message):
        if isinstance(message, dict) and "id" in message:
            emit({"jsonrpc": "2.0", "id": message["id"], "error": {
                "code": -32000, "message": "Pi compatibility check failed; inspect agent diagnostics"}})
    def input_loop():
        try:
            for raw in sys.stdin.buffer:
                message = json.loads(raw)
                try:
                    transformed = bridge.request(message)
                except ValueError:
                    failure(message)
                    continue
                send(transformed)
        except (ValueError, OSError):
            errors.append("ACP input failed")
        finally:
            process.stdin.close()
    threading.Thread(target=input_loop, daemon=True).start()
    try:
        for raw in process.stdout:
            message = json.loads(raw)
            try:
                response = bridge.response(message)
                if isinstance(response, Continuation):
                    with bridge.lock:
                        if response.request["params"]["sessionId"] in bridge.cancelled:
                            emit(bridge.response({"id": response.request["id"], "result": {"stopReason": "cancelled"}}))
                        else:
                            send(response.request)
                else:
                    emit(response)
            except ValueError:
                failure(message)
        code = process.wait()
        if errors:
            raise RuntimeError(errors[0])
        return code
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def main():
    prefix = Path(os.environ.get("PI_NVIDIA_PREFIX", Path.home() / ".cache/pi-nvidia"))
    logs = Path(os.environ.get("PI_NVIDIA_LOGS", "/logs/agent"))
    # Fail incomplete model profiles before any installation, not after launch.
    env, profile, effort, sessions = configure(prefix, logs, os.environ, started_ns=STARTED_NS)
    node = install(prefix)
    bridge = Bridge(profile, effort, sessions, logs, runtime_file=env["PI_NVIDIA_RUNTIME_FILE"])
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    return proxy([str(node), str(prefix / "node_modules/pi-acp/dist/index.js")], env, bridge)


if __name__ == "__main__":
    raise SystemExit(main())
