"""Protocol checks use initialization or an echo process, never inference."""

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

import pytest

from cad_agent.acp import child_environment


def test_proxy_transforms_actual_stdio_without_a_model(tmp_path):
    root = Path(__file__).parents[1]
    code = """
import os,sys
from pathlib import Path
from cad_agent.acp import proxy
raise SystemExit(proxy([sys.executable, '-c', 'import sys; [print(x.strip(), flush=True) for x in sys.stdin]'],
                       dict(os.environ), '/fixture/agent', Path(sys.argv[1])))
"""
    request = {"jsonrpc": "2.0", "id": 1, "method": "session/prompt", "params": {
        "sessionId": "s", "prompt": [{"type": "text", "text": "original request"}]}}
    env = dict(os.environ, PYTHONPATH=str(root))
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            input=json.dumps(request)+"\n", capture_output=True, text=True,
                            env=env, check=True, timeout=30)
    message = json.loads(result.stdout)
    assert message["params"]["prompt"][-1]["text"] == "original request"
    assert "<selected-cad-skill>" in message["params"]["prompt"][0]["text"]
    assert json.loads((tmp_path / "workflow-applied.jsonl").read_text())["session_id"] == "s"


@pytest.mark.skipif(not os.environ.get("CAD_AGENT_TEST_PREFIX"), reason="requires prepared local toolchain")
def test_real_pinned_acp_initialize_without_credentials(tmp_path):
    prefix = Path(os.environ["CAD_AGENT_TEST_PREFIX"])
    env = child_environment(prefix, {"PATH": os.defpath})
    env["CODEX_HOME"] = str(tmp_path / "codex-home")
    Path(env["CODEX_HOME"]).mkdir()
    process = subprocess.Popen([str(prefix / "toolchain/node_modules/.bin/codex-acp")],
        env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "local-no-model-check", "version": "1"}}}) + "\n")
        process.stdin.flush()
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=30), "ACP initialize timed out"
        response = json.loads(process.stdout.readline())
        assert response["id"] == 1 and "result" in response, response
        assert response["result"]["agentInfo"]["version"] == "1.12.0"
        assert response["result"]["protocolVersion"] == 1
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
