"""Real Pi + real registry ACP bridge; only inference HTTP is a loopback mock."""

import asyncio
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires pinned real runtime")
@pytest.mark.parametrize("selector,key_source", [
    ("NGC_API_KEY", "NGC_API_KEY"), ("NGC_API_KEY", "NVIDIA_API_KEY"),
    ("NVIDIA_API_KEY", "NGC_API_KEY"), ("NVIDIA_API_KEY", "NVIDIA_API_KEY"),
    (None, "NGC_API_KEY"), (None, "NVIDIA_API_KEY"),
    (None, "HOSTED_INFERENCE_TOKEN"), ("", "NGC_API_KEY"), (None, None)])
@pytest.mark.parametrize("offline", [True, False])
def test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, key_source, selector, offline):
    from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
    from acp.schema import ClientCapabilities
    from harbor.agents.installed.acp_runner import HarborAcpClient
    from harbor.hosted.config import HostedJobConfig
    from harbor.hosted.submit import dump_hosted_config

    requests = []
    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            assert self.headers.get("Authorization") == "Bearer offline-dummy-not-real"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "mock complete"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                           "prompt_tokens_details": {"cached_tokens": 20}, "completion_tokens_details": {"reasoning_tokens": 4}}},
            ]:
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prefix = Path(os.environ["PI_TEST_PREFIX"])
    profile = json.loads((ROOT / "pi_nvidia/nemotron-35-super-vl.json").read_text())
    logs = tmp_path / "logs"
    script = tmp_path / "launch.py"
    script.write_text(f'''
import os
from pathlib import Path
import pi_nvidia.hosted as hosted
from pi_nvidia.acp import Bridge, proxy
# Test-only endpoint replacement, never a production environment knob.
hosted.ENDPOINT = "http://127.0.0.1:{server.server_port}/v1"
env,p,e,s=hosted.configure({str(prefix)!r},{str(logs)!r},os.environ)
raise SystemExit(proxy([{str(prefix / "node/bin/node")!r},{str(prefix / "node_modules/pi-acp/dist/index.js")!r}],env,Bridge(p,e,s,Path({str(logs)!r}))))
''')
    guard = tmp_path / "fetch-guard.mjs"
    guard.write_text('''
const originalFetch = globalThis.fetch;
globalThis.fetch = (url, init) => {
  const hostname = new URL(typeof url === "string" ? url : url.url ?? url).hostname;
  if (hostname !== "127.0.0.1") throw new Error("Non-loopback fetch forbidden in mock test");
  return originalFetch(url, init);
};
''')
    declared = {"PI_NVIDIA_PROFILE": json.dumps(profile), "PI_NVIDIA_THINKING": "low"}
    if selector is not None:
        declared["PI_NVIDIA_ENV_NAME"] = selector
    config = HostedJobConfig.model_validate({"agents": [{"name": "acp", "env": declared}]})
    normalized = dump_hosted_config(config)["agents"][0]["env"]
    assert normalized == declared  # Real Harbor serialization retains these public options.
    env = {"PATH": os.defpath, "PYTHONPATH": str(ROOT), "NODE_OPTIONS": "--import=" + str(guard),
           **normalized, "HARBOR_ACP_REQUESTED_MODEL": "nvidia/" + profile["id"]}
    if key_source:
        env[key_source] = "offline-dummy-not-real"
    if offline:
        env["PI_OFFLINE"] = "1"
    client_logs = tmp_path / "client"
    client_logs.mkdir()
    client = HarborAcpClient(logs_dir=client_logs, permission_mode="allow")

    async def exercise():
        async with spawn_agent_process(client, sys.executable, str(script), env=env, cwd=str(tmp_path)) as (conn, proc):
            await conn.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities())
            session = await asyncio.wait_for(conn.new_session(cwd=str(tmp_path), mcp_servers=[]), 15)
            assert next(o.current_value for o in session.config_options if o.id == "model") == "nvidia/" + profile["id"]
            assert session.modes.current_mode_id == "low"
            result = await asyncio.wait_for(conn.prompt(session_id=session.session_id,
                prompt=[text_block("Reply briefly; this is a local transport fixture, do not run tools.")]), 15)
            assert result.stop_reason == "end_turn"
            assert result.usage.input_tokens == 100 and result.usage.output_tokens == 10
            assert result.usage.cached_read_tokens == 20

    try:
        if key_source is None:
            from acp import RequestError
            with pytest.raises(RequestError, match="Authentication required"):
                asyncio.run(exercise())
        else:
            asyncio.run(exercise())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    auth = json.loads((logs / "pi-auth-status.json").read_text())
    assert auth["selected_key_env"] == (selector or "NVIDIA_API_KEY")
    assert auth["credential_present"] == (key_source is not None)
    assert auth["credential_source"] == key_source
    if key_source is None:
        assert requests == []
        return
    assert len(requests) == 1
    path, _, body = requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == profile["id"] and body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 8192 and body["stream"] is True
    assert body["stream_options"]["include_usage"] is True
    assert any("validate_candidate" in str(message) for message in body["messages"])
    assert all("offline-dummy-not-real" not in p.read_text()
               for p in logs.rglob("*") if p.is_file())
