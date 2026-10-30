"""Small guarded stdio proxy around the registry-listed pi-acp bridge."""

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from .accounting import snapshot
from .bootstrap import install
from .hosted import configure

ROOT = Path(__file__).parent


class Bridge:
    def __init__(self, profile, effort, sessions, logs):
        self.profile, self.effort, self.sessions, self.logs = profile, effort, Path(sessions), Path(logs)
        self.public = "nvidia/" + profile.id
        self.native = "harbor-endpoint/" + profile.id
        self.pending, self.ready = {}, set()
        self.lock = threading.RLock()

    def request(self, message):
        with self.lock:
            result = copy.deepcopy(message)
            method, params = result.get("method"), result.get("params") or {}
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
                params["prompt"] = [{"type": "text", "text": (ROOT / "feedback.txt").read_text()}] + list(params["prompt"])
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
            if not request or "result" not in result:
                return result
            method, params = request
            payload = result["result"]
            if method == "initialize":
                payload["agentInfo"] = {"name": "pi-nvidia-feedback", "version": "0.2.0"}
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
                receipt = snapshot(self.sessions, params["sessionId"], self.profile, self.effort)
                self.logs.mkdir(parents=True, exist_ok=True)
                (self.logs / "pi-accounting.json").write_text(json.dumps(receipt, indent=2) + "\n")
                payload.setdefault("_meta", {})["pi_nvidia_accounting"] = receipt
                payload.pop("usage", None)
                if receipt["status"] == "verified":
                    payload["usage"] = receipt["usage"]
                if receipt.get("provider_error") and payload.get("stopReason") != "cancelled":
                    raise ValueError("Pi reported a provider error; see the retained native session")
            return result


def proxy(command, env, bridge):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               env=env, start_new_session=True)
    output_lock = threading.Lock()
    errors = []
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
                process.stdin.write(json.dumps(transformed).encode() + b"\n")
                process.stdin.flush()
        except (ValueError, OSError):
            errors.append("ACP input failed")
        finally:
            process.stdin.close()
    threading.Thread(target=input_loop, daemon=True).start()
    try:
        for raw in process.stdout:
            message = json.loads(raw)
            try:
                emit(bridge.response(message))
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
    env, profile, effort, sessions = configure(prefix, logs, os.environ)
    node = install(prefix)
    bridge = Bridge(profile, effort, sessions, logs)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    return proxy([str(node), str(prefix / "node_modules/pi-acp/dist/index.js")], env, bridge)


if __name__ == "__main__":
    raise SystemExit(main())
