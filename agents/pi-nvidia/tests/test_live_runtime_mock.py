"""Real Pi + real registry ACP bridge; only inference HTTP is a loopback mock."""

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from test_feedback_protocol import feedback

ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires pinned real runtime")
@pytest.mark.parametrize("selector,key_source", [
    ("NGC_API_KEY", "NGC_API_KEY"), ("NGC_API_KEY", "NVIDIA_API_KEY"),
    ("NVIDIA_API_KEY", "NGC_API_KEY"), ("NVIDIA_API_KEY", "NVIDIA_API_KEY"),
    (None, "NGC_API_KEY"), (None, "NVIDIA_API_KEY"),
    (None, "HOSTED_INFERENCE_TOKEN"), ("", "NGC_API_KEY"), (None, None)])
@pytest.mark.parametrize("offline", [True, False])
def test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, key_source, selector, offline,
        recovery_case="normal", effort="low", output_cap=8192, runtime_case=None, feedback_url=None):
    from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
    from acp.schema import ClientCapabilities, ImageContentBlock
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
            if runtime_case == "provider_503" and number == 1:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"synthetic service unavailable","type":"server_error"}}')
                return
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
            if runtime_case in {"turn_limit", "turn_limit_length", "turn_limit_validation"}:
                if runtime_case == "turn_limit_length" and number == 64:
                    delta = {"role": "assistant", "reasoning_content": "synthetic empty length stop"}
                    finish = "length"
                else:
                    delta = {"role": "assistant", "tool_calls": [{"index": i, "id": f"limit-{number}-{i}",
                        "type": "function", "function": {"name": "write", "arguments": json.dumps({
                            "path": str(tmp_path / "candidate" / ("asset.usd" if i == 0 else "batch-receipt.txt")),
                            "content": f"#usda 1.0\n# saved turn {number}\n"})}} for i in range(2)]}
                    finish = "tool_calls"
            if runtime_case == "turn_limit_validation" and number == 64:
                delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": "final-validation",
                    "type": "function", "function": {"name": "bash", "arguments": json.dumps({
                        "command": shlex.join(["python", "-m", "simready_eval.feedback", "validate_candidate",
                            "--package", str(tmp_path / "candidate"), "--request-id", "candidate-final", "--url", feedback_url])})}}]}
                finish = "tool_calls"
            if runtime_case == "workflow" and number in (1, 2):
                args = {"path": str(tmp_path / "candidate/asset.usd"), "content": '#usda 1.0\n( metersPerUnit = 1; upAxis = "Z" )\ndef Xform "Asset" {}\n'} if number == 1 else {
                    "command": shlex.join(["python", "-m", "simready_eval.feedback", "validate_candidate",
                        "--package", str(tmp_path / "candidate"), "--request-id", "candidate-1", "--url", feedback_url])}
                delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": f"workflow-{number}",
                    "type": "function", "function": {"name": "write" if number == 1 else "bash", "arguments": json.dumps(args)}}]}
                finish = "tool_calls"
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
import json,time
from pathlib import Path
import pi_nvidia.hosted as hosted
import pi_nvidia.acp as shim
from pi_nvidia.acp import Bridge, proxy
# Test-only endpoint replacement, never a production environment knob.
hosted.ENDPOINT = "http://127.0.0.1:{server.server_port}/v1"
env,p,e,s=hosted.configure({str(prefix)!r},{str(logs)!r},os.environ)
if {runtime_case!r}:
    runtime=Path(env["PI_NVIDIA_RUNTIME_FILE"])
    data=json.loads(runtime.read_text());data["submission_entrypoint"]={str(tmp_path / "candidate/asset.usd")!r}
    runtime.write_text(json.dumps(data))
if {runtime_case!r} == "reserve":
    runtime=Path(env["PI_NVIDIA_RUNTIME_FILE"])
    data=json.loads(runtime.read_text())
    (Path({str(logs)!r})/"pi-turn-budget.json").write_text(json.dumps(dict(limit=64,started=60,completed=60,exhausted=False)))
    candidate=Path({str(tmp_path / "candidate/asset.usd")!r});candidate.parent.mkdir();candidate.write_text("#usda 1.0")
    data["submission_entrypoint"]=str(candidate)
    runtime.write_text(json.dumps(data))
if {runtime_case!r} == "wire_missing":
    original=shim.attach_observation
    def drop_image(prompt,config):
        blocks,receipt=original(prompt,config)
        return [b for b in blocks if b["type"]!="image"],receipt
    shim.attach_observation=drop_image
bridge=Bridge(p,e,s,Path({str(logs)!r}),runtime_file=env["PI_NVIDIA_RUNTIME_FILE"] if {runtime_case!r} else None)
raise SystemExit(proxy([{str(prefix / "node/bin/node")!r},{str(prefix / "node_modules/pi-acp/dist/index.js")!r}],env,bridge))
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
    image_data = None
    if runtime_case:
        image_path = Path(os.environ["NV100_OBSERVATION_PNG"])
        image_data = image_path.read_bytes()
        declared.update(PI_NVIDIA_OBSERVATION_PATH=str(image_path),
            PI_NVIDIA_OBSERVATION_SHA256="0" * 64 if runtime_case == "wrong_hash" else hashlib.sha256(image_data).hexdigest())
    config = HostedJobConfig.model_validate({"agents": [{"name": "acp", "env": declared}]})
    normalized = dump_hosted_config(config)["agents"][0]["env"]
    assert normalized == declared  # Real Harbor serialization retains these public options.
    env = {"PATH": os.defpath, "PYTHONPATH": str(ROOT), "NODE_OPTIONS": "--import=" + str(guard),
           **normalized, "HARBOR_ACP_REQUESTED_MODEL": "nvidia/" + profile["id"]}
    if key_source:
        env[key_source] = "offline-dummy-not-real"
    if offline:
        env["PI_OFFLINE"] = "1"
    if runtime_case in {"workflow", "turn_limit_validation"}:
        env["PATH"] = os.environ["PI_TASK_PYTHON_DIR"] + os.pathsep + os.defpath
    client_logs = tmp_path / "client"
    client_logs.mkdir()
    client = HarborAcpClient(logs_dir=client_logs, permission_mode="allow")

    async def exercise():
        async with spawn_agent_process(client, sys.executable, str(script), env=env, cwd=str(tmp_path)) as (conn, proc):
            await conn.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities())
            session = await asyncio.wait_for(conn.new_session(cwd=str(tmp_path), mcp_servers=[]), 15)
            assert next(o.current_value for o in session.config_options if o.id == "model") == "nvidia/" + profile["id"]
            assert session.modes.current_mode_id == effort
            prompt = [text_block("Reply briefly; this is a local transport fixture, do not run tools.")]
            if runtime_case == "preprovided":
                prompt.append(ImageContentBlock(type="image", data=base64.b64encode(image_data).decode(), mime_type="image/png"))
            task = asyncio.create_task(conn.prompt(session_id=session.session_id, prompt=prompt))
            if recovery_case in {"cancel", "deadline"}:
                assert await asyncio.to_thread(second_request.wait, 5)
                if recovery_case == "deadline":
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(task, 0.1)
                    await conn.cancel(session_id=session.session_id)
                    return
                await conn.cancel(session_id=session.session_id)
            result = await asyncio.wait_for(task, 45 if runtime_case in {"turn_limit", "turn_limit_length", "turn_limit_validation"} else 15)
            expected = {"exhausted": "max_tokens", "partial": "max_tokens", "cancel": "cancelled"}.get(recovery_case, "end_turn")
            if runtime_case in {"turn_limit", "turn_limit_length", "turn_limit_validation"}:
                expected = "max_turn_requests"
            assert result.stop_reason == expected
            if recovery_case != "cancel":
                count = 64 if runtime_case in {"turn_limit", "turn_limit_length", "turn_limit_validation"} else 3 if runtime_case == "workflow" else 1 if recovery_case in {"normal", "partial"} else 2
                assert result.usage.input_tokens == (100 + 120 * (count - 1))
                assert result.usage.output_tokens == (640 if count == 64 else 30 if count == 3 else {"exhausted": 16384, "partial": 8192, "recovered": 8202}.get(recovery_case, 10))
                assert result.usage.cached_read_tokens == 20 * count

    try:
        if key_source is None or recovery_case == "error" or runtime_case in {"wrong_hash", "wire_missing", "provider_503"}:
            from acp import RequestError
            match = "Authentication required" if key_source is None else "provider error" if recovery_case == "error" else ".*"
            with pytest.raises(RequestError, match=match):
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
    if runtime_case in {"wrong_hash", "wire_missing"}:
        assert requests == []
        if runtime_case == "wrong_hash":
            assert json.loads((logs / "pi-input-image.json").read_text())["status"] == "error"
        else:
            assert json.loads((logs / "pi-image-wire.jsonl").read_text().splitlines()[0])["matched"] is False
        return
    if runtime_case == "provider_503":
        assert len(requests) == 1
        assert json.loads((logs / "pi-accounting.json").read_text())["status"] == "unavailable"
        return
    assert len(requests) == (64 if runtime_case in {"turn_limit", "turn_limit_length", "turn_limit_validation"} else 3 if runtime_case == "workflow" else 1 if recovery_case in {"normal", "partial"} else 2)
    path, _, body = requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == profile["id"] and body["reasoning_effort"] == profile["thinkingLevelMap"][effort]
    assert body["max_tokens"] == output_cap and body["stream"] is True
    assert body["stream_options"]["include_usage"] is True
    assert any("validate_candidate" in str(message) for message in body["messages"])
    if runtime_case:
        def image_urls(value):
            if isinstance(value, list):
                return sum((image_urls(v) for v in value), [])
            if isinstance(value, dict):
                return ([value["image_url"]["url"]] if value.get("type") == "image_url" else
                        sum((image_urls(v) for v in value.values()), []))
            return []
        for _, _, request in requests:
            images = image_urls(request["messages"])
            assert len(images) == 1 and base64.b64decode(images[0].split(",", 1)[1]) == image_data
        image_receipt = json.loads((logs / "pi-input-image.json").read_text())
        assert image_receipt["delivery"] == ("already_supplied" if runtime_case == "preprovided" else "attached")
        assert image_receipt["source_sha256"] == image_receipt["payload_sha256"] == hashlib.sha256(image_data).hexdigest()
        wire = [json.loads(line) for line in (logs / "pi-image-wire.jsonl").read_text().splitlines()]
        assert len(wire) == len(requests) and all(row["matched"] for row in wire)
        budget = [json.loads(line) for line in (logs / "pi-budget-notices.jsonl").read_text().splitlines()]
        assert len(budget) == len(requests) and all(1 <= row["remaining_including_current"] <= 64 for row in budget)
        assert budget[-1]["remaining_including_current"] <= budget[0]["remaining_including_current"]
        if runtime_case == "reserve":
            assert budget[0]["phase"] == "final_revalidation" and budget[0]["remaining_including_current"] == 4
            assert any("FINAL REVALIDATION WINDOW" in str(m) for m in body["messages"])
        if runtime_case in {"turn_limit", "turn_limit_length", "turn_limit_validation"}:
            assert [row["turn"] for row in budget] == list(range(1,65))
            assert [row["remaining_including_current"] for row in budget] == list(range(64,0,-1))
            last_write = 63 if runtime_case in {"turn_limit_length", "turn_limit_validation"} else 64
            assert (tmp_path / "candidate/asset.usd").read_text() == f"#usda 1.0\n# saved turn {last_write}\n"
            assert (tmp_path / "candidate/batch-receipt.txt").read_text() == (tmp_path / "candidate/asset.usd").read_text()
            receipt = json.loads((logs / "pi-accounting.json").read_text())
            assert receipt["turn_budget"]["started"] == receipt["turn_budget"]["completed"] == 64
            assert receipt["turn_budget"]["exhausted"] is True and receipt["length_recovery_count"] == 0
            assert receipt["completion"] == "model_turn_budget_exhausted"
        if runtime_case == "workflow":
            assert budget[0]["phase"] == "build_first_candidate"
            assert budget[1]["phase"] == "validate_or_repair"
        assert json.loads((logs / "pi-accounting.json").read_text())["image_delivery"]["initial_payload_verified"] is True
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


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX") or not os.environ.get("NV100_OBSERVATION_PNG"),
                   reason="requires pinned runtime and original public observation")
@pytest.mark.parametrize("case", ["initial", "preprovided", "wrong_hash", "wire_missing", "reserve", "continuation", "provider_503"])
def test_initial_image_and_turn_budget_reach_real_pi_provider_payload(tmp_path, case):
    test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, "NGC_API_KEY", None, False,
        effort="off", output_cap=16384, runtime_case=case,
        recovery_case="recovered" if case == "continuation" else "normal")


@pytest.mark.skipif(not all(os.environ.get(k) for k in ("PI_TEST_PREFIX", "NV100_OBSERVATION_PNG", "PI_TASK_PYTHON_DIR", "NV100_EVAL_WHEEL")),
                   reason="requires pinned runtime, original task image and genuine feedback client")
@pytest.mark.parametrize("case", ["workflow", "turn_limit_validation"])
def test_real_candidate_write_then_genuine_feedback_with_one_of_five_calls(tmp_path, feedback, case):
    from test_feedback_protocol import outcome
    service = feedback.FeedbackService(tmp_path / "feedback-state", evaluator=lambda p: outcome((p / "asset.usd").is_file()))
    server = service.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, "NGC_API_KEY", None, False,
            effort="off", output_cap=16384, runtime_case=case, feedback_url=f"http://127.0.0.1:{server.server_port}")
        assert service.ledger.health() == {"status": "ok", "limit": 5, "used": 1, "remaining": 4}
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2); service.close()


@pytest.mark.parametrize("case", ["turn_limit", "turn_limit_length"])
@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX") or not os.environ.get("NV100_OBSERVATION_PNG"),
                   reason="requires pinned runtime and original public observation")
def test_actual_64_turn_limit_completes_last_tool_batch_without_request_65(tmp_path, case):
    test_real_pi_auth_gate_and_streamed_request_with_dummy_key(tmp_path, "NGC_API_KEY", None, True,
        effort="off", output_cap=16384, runtime_case=case)
