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
def test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, key_source, selector, offline,
        recovery_case="normal", effort="low", output_cap=8192):
    from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
    from acp.schema import ClientCapabilities
    from harbor.agents.installed.acp_runner import HarborAcpClient
    from harbor.hosted.config import HostedJobConfig
    from harbor.hosted.submit import dump_hosted_config

    requests = []
    second_request, release = threading.Event(), threading.Event()
    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, self.headers.get("Authorization"), body))
            assert self.headers.get("Authorization") == "Bearer offline-dummy-not-real"
            number = len(requests)
            if number == 2:
                second_request.set()
                if recovery_case in {"cancel", "deadline"}:
                    release.wait(timeout=5)
            truncated = (number == 1 and recovery_case != "normal") or recovery_case == "exhausted"
            partial = recovery_case == "partial"
            output = 8192 if truncated else 10
            prompt = 100 if number == 1 else 120
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = {"role": "assistant"}
            delta.update({"reasoning_content": "synthetic private reasoning"} if truncated and not partial
                         else {"content": "partial answer" if partial else "mock complete"})
            finish = "length" if truncated else ("content_filter" if recovery_case == "error" else "stop")
            try:
                for chunk in [
                    {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output,
                               "prompt_tokens_details": {"cached_tokens": 20},
                               "completion_tokens_details": {"reasoning_tokens": output if truncated and not partial else 4}}},
                ]:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Expected when cancellation interrupts the local HTTP stream.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    prefix = Path(os.environ["PI_TEST_PREFIX"])
    profile = json.loads((ROOT / "pi_nvidia/nemotron-35-super-vl.json").read_text())
    profile["maxTokens"] = output_cap
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
    declared = {"PI_NVIDIA_PROFILE": json.dumps(profile), "PI_NVIDIA_THINKING": effort}
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
            assert session.modes.current_mode_id == effort
            task = asyncio.create_task(conn.prompt(session_id=session.session_id,
                prompt=[text_block("Reply briefly; this is a local transport fixture, do not run tools.")]))
            if recovery_case in {"cancel", "deadline"}:
                assert await asyncio.to_thread(second_request.wait, 5)
                if recovery_case == "deadline":
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(task, 0.1)
                    await conn.cancel(session_id=session.session_id)
                    return
                await conn.cancel(session_id=session.session_id)
            result = await asyncio.wait_for(task, 15)
            expected = {"exhausted": "max_tokens", "partial": "max_tokens", "cancel": "cancelled"}.get(recovery_case, "end_turn")
            assert result.stop_reason == expected
            if recovery_case != "cancel":
                count = 1 if recovery_case in {"normal", "partial"} else 2
                assert result.usage.input_tokens == (100 if count == 1 else 220)
                assert result.usage.output_tokens == {"exhausted": 16384, "partial": 8192, "recovered": 8202}.get(recovery_case, 10)
                assert result.usage.cached_read_tokens == 20 * count

    try:
        if key_source is None or recovery_case == "error":
            from acp import RequestError
            with pytest.raises(RequestError, match="Authentication required" if key_source is None else "provider error"):
                asyncio.run(exercise())
        else:
            asyncio.run(exercise())
    finally:
        release.set()
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
    assert len(requests) == (1 if recovery_case in {"normal", "partial"} else 2)
    path, _, body = requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == profile["id"] and body["reasoning_effort"] == profile["thinkingLevelMap"][effort]
    assert body["max_tokens"] == output_cap and body["stream"] is True
    assert body["stream_options"]["include_usage"] is True
    assert any("validate_candidate" in str(message) for message in body["messages"])
    if recovery_case not in {"normal", "partial"}:
        assert any("single automatic length recovery" in str(message) for message in requests[1][2]["messages"])
        recovery_log = (logs / "pi-length-recovery.jsonl").read_text().splitlines()
        assert len(recovery_log) == 1 and json.loads(recovery_log[0])["reason"] == "empty_length_stop"
        assert json.loads((logs / "pi-accounting.json").read_text())["length_recovery_count"] == 1
        if recovery_case != "deadline":
            receipt = json.loads((logs / "pi-accounting.json").read_text())
            assert receipt["length_recovery_count"] == 1 and receipt["length_recovery_limit"] == 1
    assert all("offline-dummy-not-real" not in p.read_text()
               for p in logs.rglob("*") if p.is_file())


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires pinned real runtime")
@pytest.mark.parametrize("case", ["recovered", "exhausted", "partial", "cancel", "deadline", "error"])
def test_real_pi_length_recovery_is_bounded_and_keeps_usage(tmp_path, case):
    test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, "NGC_API_KEY", None, False, recovery_case=case)


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires pinned real runtime")
def test_proposed_none_reasoning_16384_request_is_supported_locally(tmp_path):
    test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, "NGC_API_KEY", None, False,
        effort="off", output_cap=16384)
