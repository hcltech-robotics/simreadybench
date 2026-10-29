"""Hosted ACP transport around the same pinned Codex CAD workflow.

The selected skill, legacy model IDs and usage envelopes have explicit adapters.
Original usage is retained in _meta. MCP, images, permissions and cancellation
are forwarded; no provider price is invented.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.request

from .prepare import install, configure_trial
from .accounting import AccountingBridge, retain_evidence
from .workflow import PACKAGE, bundle_identity, render_instruction, verify_generation


def transform_request(message, prefix):
    if message.get("method") != "session/prompt":
        return message
    result = dict(message)
    params = dict(message["params"])
    # Keep original prompt blocks, including images and resource links, intact.
    params["prompt"] = [{"type": "text", "text": render_instruction("", prefix)}] + list(params["prompt"])
    result["params"] = params
    return result


def child_environment(prefix, environ=None):
    env = dict(os.environ if environ is None else environ)
    effort = env.get("CAD_AGENT_REASONING_EFFORT", "medium")
    if effort not in {"minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError("Unsupported CAD_AGENT_REASONING_EFFORT")
    config = {"model_reasoning_effort": effort,
              "mcp_servers": {"cad": {"command": str(Path(prefix) / "bin/cad-python"),
                 "args": ["-m", "cad_agent.prepare", "mcp", "--prefix", str(prefix)],
                 "required": True, "startup_timeout_sec": 300}}}
    model = env.get("HARBOR_ACP_REQUESTED_MODEL")
    if model:
        config["model"] = model.split("/", 1)[-1]
    token = env.get("HOSTED_INFERENCE_TOKEN") or env.get("OPENAI_API_KEY")
    if token:
        env["CODEX_API_KEY"] = token
    endpoint = env.get("HOSTED_INFERENCE_URL")
    if endpoint:
        config.update(model_provider="harbor")
        config["model_providers"] = {"harbor": {"name": "Harbor", "base_url": endpoint,
            "env_key": "CODEX_API_KEY", "wire_api": "responses"}}
    env.update(CODEX_CONFIG=json.dumps(config),
        CODEX_PATH=str(Path(prefix) / "toolchain/node_modules/.bin/codex"),
        CODEX_HOME=str(Path.home() / ".codex-text2cad"), NO_BROWSER="1",
        INITIAL_AGENT_MODE="agent-full-access", DEFAULT_AUTH_REQUEST='{"methodId":"api-key"}')
    env["PATH"] = str(Path(prefix) / "node/bin") + os.pathsep + env.get("PATH", os.defpath)
    return env


def install_node(prefix):
    pin = json.loads((PACKAGE / "node.json").read_text())
    target = Path(prefix) / "node"
    binary = target / "bin/node"
    if binary.exists():
        actual = subprocess.check_output([str(binary), "--version"], text=True).strip()
        if actual == "v" + pin["version"]:
            return target / "bin"
        raise RuntimeError("Baked Node version does not match pin")
    archive = Path(prefix) / "node.tar.xz"
    with urllib.request.urlopen(pin["url"], timeout=60) as response:
        archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != pin["sha256"]:
        raise RuntimeError("Node archive checksum mismatch")
    with tarfile.open(archive) as tar:
        top = tar.getnames()[0].split("/", 1)[0]
        tar.extractall(Path(prefix) / "node-unpack", filter="data")
    shutil.move(Path(prefix) / "node-unpack" / top, target)
    return target / "bin"


def install_toolchain(prefix):
    """npm's lock pins both the adapter and its actual selected Codex binary."""
    prefix = Path(prefix)
    node_bin = install_node(prefix)
    env = dict(os.environ)
    env["PATH"] = str(node_bin) + os.pathsep + env.get("PATH", os.defpath)
    env.setdefault("NPM_CONFIG_CACHE", str(prefix / "npm-cache"))
    target = prefix / "toolchain"
    target.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((PACKAGE / "package-lock.json").read_bytes()).hexdigest()
    marker = target / "installed.sha256"
    if not marker.exists() or marker.read_text() != digest:
        for name in ("package.json", "package-lock.json"):
            shutil.copyfile(PACKAGE / name, target / name)
        subprocess.run([str(node_bin / "npm"), "ci", "--no-audit", "--no-fund", "--prefix", str(target)],
                       check=True, env=env, stdout=sys.stderr, stderr=sys.stderr)
        marker.write_text(digest)
    actual = subprocess.check_output([str(target / "node_modules/.bin/codex"), "--version"], env=env, text=True).strip()
    if actual != "codex-cli 0.164.0":
        raise RuntimeError(f"Unexpected Codex toolchain: {actual}")
    return actual


def proxy(command, env, prefix, logs):
    bridge = AccountingBridge(Path(env["CODEX_HOME"]), logs,
                              env.get("HARBOR_ACP_REQUESTED_MODEL", "").removeprefix("openai/"),
                              json.loads(env["CODEX_CONFIG"])["model_reasoning_effort"])
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    errors = []
    output_lock = threading.Lock()

    def emit(message):
        with output_lock:
            sys.stdout.buffer.write(json.dumps(message).encode() + b"\n")
            sys.stdout.buffer.flush()

    def forward_input():
        message = None
        try:
            for raw in sys.stdin.buffer:
                message = json.loads(raw)
                transformed = bridge.request(transform_request(message, prefix))
                if message.get("method") == "session/prompt":
                    with (logs / "workflow-applied.jsonl").open("a") as f:
                        f.write(json.dumps({"request_id": message.get("id"), "session_id": message["params"].get("sessionId"),
                            "skill_commit": bundle_identity()["skill_commit"],
                            "instruction_sha256": hashlib.sha256(transformed["params"]["prompt"][0]["text"].encode()).hexdigest()}) + "\n")
                process.stdin.write(json.dumps(transformed).encode() + b"\n")
                process.stdin.flush()
        except (ValueError, OSError, KeyError) as exc:
            errors.append(str(exc))
            if isinstance(message, dict) and "id" in message:
                emit({"jsonrpc": "2.0", "id": message["id"], "error": {
                    "code": -32000, "message": "CAD agent compatibility check failed: " + str(exc)}})
        finally:
            process.stdin.close()

    thread = threading.Thread(target=forward_input, daemon=True)
    thread.start()
    try:
        for line in process.stdout:
            emit(bridge.response(json.loads(line)))
        code = process.wait()
        if errors:
            raise RuntimeError("ACP input forwarding failed: " + errors[0])
        return code
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        bridge.finalize()


def main():
    # Stable cache can be baked; configure_trial still executes on every start.
    prefix = Path(os.environ.get("CAD_AGENT_PREFIX", Path.home() / ".cache/codex-text2cad"))
    prefix = prefix.resolve()
    logs = Path(os.environ.get("CAD_AGENT_LOGS", "/logs/agent/cad"))
    install(prefix)
    configure_trial(prefix, logs=logs)
    actual = install_toolchain(prefix)
    env = child_environment(prefix)
    Path(env["CODEX_HOME"]).mkdir(mode=0o700, parents=True, exist_ok=True)
    (logs / "transport.json").write_text(json.dumps({"transport": "acp", "codex": actual,
        "adapter": "1.12.0", "requested_model": os.environ.get("HARBOR_ACP_REQUESTED_MODEL"),
        "reasoning_effort": json.loads(env["CODEX_CONFIG"])["model_reasoning_effort"]}, indent=2) + "\n")
    stop_retention = threading.Event()

    def retain():
        try:
            retain_evidence(Path(env["CODEX_HOME"]), Path("/output/authoring"), logs)
        except (OSError, ValueError, TypeError):
            (logs / "retention-error.json").write_text('{"status":"incomplete"}\n')

    def watch_evidence():
        while not stop_retention.wait(2):
            retain()

    watcher = threading.Thread(target=watch_evidence, daemon=True)
    watcher.start()
    try:
        code = proxy([str(prefix / "toolchain/node_modules/.bin/codex-acp")], env, prefix, logs)
    finally:
        stop_retention.set()
        watcher.join(timeout=10)
        result = verify_generation("/output/authoring/cad-evidence/generation.json", "/output/submission/asset.usd")
        (logs / "use.json").write_text(json.dumps(result, indent=2) + "\n")
        retain()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
