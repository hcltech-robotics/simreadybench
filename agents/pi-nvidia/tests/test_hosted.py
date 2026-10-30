import asyncio
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from pi_nvidia.acp import Bridge
from pi_nvidia.hosted import configure
from pi_nvidia.profile import ModelProfile
from test_agent import profile

ROOT = Path(__file__).parents[1]


def test_profile_auth_reference_and_baked_reconfiguration(tmp_path):
    source = {"PI_NVIDIA_PROFILE": json.dumps(profile()), "PI_NVIDIA_THINKING": "off",
              "HARBOR_ACP_REQUESTED_MODEL": "nvidia/fixture/vision"}
    env, p, effort, sessions = configure(tmp_path / "runtime", tmp_path / "logs", source)
    assert source.get("PI_CODING_AGENT_DIR") is None
    config = Path(env["PI_CODING_AGENT_DIR"])
    models = json.loads((config / "models.json").read_text())
    assert models["providers"]["harbor-endpoint"]["apiKey"] == "$NVIDIA_API_KEY"
    assert p.id == "fixture/vision" and effort == "off"
    assert "--no-skills --no-extensions --no-prompt-templates" in (config / "pi-plain").read_text()
    newer = copy.deepcopy(source)
    data = profile(id="fixture/other")
    newer.update(PI_NVIDIA_PROFILE=json.dumps(data), HARBOR_ACP_REQUESTED_MODEL="nvidia/fixture/other")
    configure(tmp_path / "runtime", tmp_path / "logs", newer)
    assert "fixture/other" in (config / "models.json").read_text()
    assert "fixture/vision" not in (config / "models.json").read_text()
    for bad in ({"HOSTED_INFERENCE_URL": "https://gateway.example/v1"}, {"PI_NVIDIA_THINKING": "high"}):
        with pytest.raises(ValueError):
            configure(tmp_path / "runtime", tmp_path / "logs", {**source, **bad})


@pytest.mark.skipif(not os.environ.get("PI_TASK_PYTHON_DIR"), reason="requires original feedback-client task Python")
def test_source_venv_does_not_hide_the_genuine_task_feedback_client(tmp_path):
    import shutil
    task_bin = os.environ["PI_TASK_PYTHON_DIR"]
    source = {"PI_NVIDIA_PROFILE": json.dumps(profile()), "PI_NVIDIA_THINKING": "off",
              "HARBOR_ACP_REQUESTED_MODEL": "nvidia/fixture/vision",
              "PATH": str(Path(sys.prefix) / "bin") + os.pathsep + task_bin + os.pathsep + os.defpath,
              "VIRTUAL_ENV": sys.prefix}
    env, *_ = configure(tmp_path / "runtime", tmp_path / "logs", source)
    assert shutil.which("python", path=env["PATH"]) == str(Path(task_bin) / "python")
    assert "VIRTUAL_ENV" not in env
    result = subprocess.run(["python", "-m", "simready_eval.feedback", "--help"],
                            env=env, text=True, capture_output=True, check=True, timeout=10)
    assert "validate_candidate" in result.stdout


def test_receipt_gate_model_remap_and_cost_unknown(tmp_path):
    b = Bridge(ModelProfile.model_validate(profile()), "off", tmp_path / "pi/sessions", tmp_path)
    with pytest.raises(ValueError, match="receipt"):
        b.request({"method": "session/prompt", "params": {"sessionId": "s", "prompt": []}})
    b.request({"id": 1, "method": "session/new", "params": {"mcpServers": []}})
    result = b.response({"id": 1, "result": {"sessionId": "s", "models": {"currentModelId": b.native}, "modes": {"currentModeId": "off"}}})
    assert result["result"]["models"]["currentModelId"] == b.public
    request = b.request({"id": 2, "method": "session/set_model", "params": {"sessionId": "s", "modelId": b.public}})
    assert request["params"]["modelId"] == b.native
    with pytest.raises(ValueError):
        b.request({"method": "session/set_mode", "params": {"modeId": "high"}})
    update = b.response({"method": "session/update", "params": {"sessionId": "s", "update": {"sessionUpdate": "usage_update", "used": 7000, "size": 32000, "cost": {"amount": 0, "currency": "USD"}}}})
    assert "cost" not in update["params"]["update"]
    assert update["params"]["update"]["_meta"]["unverified_upstream_cost"]["amount"] == 0


def launch_script(tmp_path):
    prefix = Path(os.environ["PI_TEST_PREFIX"])
    fake = tmp_path / "fake-pi"
    fake.write_text(f'#!/bin/sh\nexec {sys.executable} {Path(__file__).with_name("fake_pi.py")} "$@"\n')
    fake.chmod(0o755)
    script = tmp_path / "launch"
    script.write_text(f'''#!{sys.executable}
import os
from pathlib import Path
from pi_nvidia.acp import Bridge, proxy
from pi_nvidia.hosted import configure
prefix=Path({str(prefix)!r})
env,p,e,s=configure(prefix,Path({str(tmp_path / "logs")!r}),os.environ)
env["PI_ACP_PI_COMMAND"]={str(fake)!r}
env["FIXTURE_CAPTURE"]={str(tmp_path / "captured.json")!r}
raise SystemExit(proxy([str(prefix/"node/bin/node"),str(prefix/"node_modules/pi-acp/dist/index.js")],env,Bridge(p,e,s,Path({str(tmp_path / "logs")!r}))))
''')
    script.chmod(0o755)
    return script


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires pinned Pi/registry bridge")
@pytest.mark.parametrize("provider_error", [False, True])
def test_actual_harbor_runner_registry_bridge_and_mock_pi(tmp_path, monkeypatch, provider_error):
    from harbor.agents.installed.acp_runner import run
    from harbor.agents.installed.acp import AcpAgent
    from harbor.models.agent.acp_source import AcpSourceManifest
    from harbor.models.agent.context import AgentContext
    monkeypatch.setenv("PI_NVIDIA_PROFILE", json.dumps(profile()))
    monkeypatch.setenv("PI_NVIDIA_THINKING", "off")
    monkeypatch.setenv("HARBOR_ACP_REQUESTED_MODEL", "nvidia/fixture/vision")
    monkeypatch.delenv("HOSTED_INFERENCE_URL", raising=False)
    if provider_error:
        monkeypatch.setenv("FIXTURE_PROVIDER_ERROR", "1")
    # No real key exists or is supplied; only the Pi backend is a local RPC fixture.
    script = launch_script(tmp_path)
    logs = tmp_path / "logs"
    code = asyncio.run(run(["--instruction", "ORIGINAL benchmark observation /task/observation.png",
                            "--logs-dir", str(logs), "--launcher", str(script)]))
    summary = json.loads((logs / "acp-summary.json").read_text())
    captured = json.loads((tmp_path / "captured.json").read_text())
    assert "ORIGINAL benchmark" in captured["message"] and "validate_candidate" in captured["message"]
    receipt = json.loads((logs / "pi-accounting.json").read_text())
    if provider_error:
        assert code == 1 and "error" in summary
        assert receipt["status"] == "unavailable" and receipt["reason"] == "native_provider_error"
        return
    assert code == 0, summary
    assert receipt["call_count"] == 2  # Early agent_end did not prematurely finish.
    assert summary["prompt_response"]["usage"]["inputTokens"] == 360
    assert summary["latest_usage_update"]["used"] == 7000  # Occupancy is separate.
    a = AcpAgent(logs_dir=logs, model_name="nvidia/fixture/vision", source={
        "repo_url": "https://github.com/hcltech-robotics/simreadybench.git", "ref": "89350e067fc4e6411347edd51bf4a1695ad188fb",
        "source_dir": "agents/pi-nvidia", "manifest_path": "harbor-agent.json"})
    a._source_manifest = AcpSourceManifest.model_validate_json((ROOT / "harbor-agent.json").read_text())
    a._load_summary = lambda: summary
    a._write_trajectory = lambda **kw: None
    context = AgentContext()
    a.populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (360, 160, 20)
    assert context.cost_usd is None
    events = [json.loads(line) for line in (logs / "acp-events.jsonl").read_text().splitlines()]
    trajectory = a._convert_events_to_trajectory(summary=summary, events=events)
    assert trajectory.final_metrics.total_prompt_tokens == 360
    assert trajectory.final_metrics.total_cost_usd is None
    assert any(step.tool_calls for step in trajectory.steps)


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires pinned Pi/registry bridge")
def test_actual_registry_bridge_keeps_images_and_handles_cancel(tmp_path, monkeypatch):
    from acp import PROTOCOL_VERSION, text_block, spawn_agent_process
    from acp.schema import ClientCapabilities, FileSystemCapabilities, ImageContentBlock
    from harbor.agents.installed.acp_runner import HarborAcpClient
    monkeypatch.setenv("PI_NVIDIA_PROFILE", json.dumps(profile()))
    monkeypatch.setenv("PI_NVIDIA_THINKING", "off")
    monkeypatch.setenv("HARBOR_ACP_REQUESTED_MODEL", "nvidia/fixture/vision")
    monkeypatch.setenv("FIXTURE_WAIT_FOR_ABORT", "1")
    monkeypatch.delenv("HOSTED_INFERENCE_URL", raising=False)
    script = launch_script(tmp_path)
    client_logs = tmp_path / "client"
    client_logs.mkdir()
    client = HarborAcpClient(logs_dir=client_logs, permission_mode="allow")
    async def exercise():
        async with spawn_agent_process(client, str(script), env=dict(os.environ), cwd=str(tmp_path)) as (conn, proc):
            init = await conn.initialize(protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities(fs=FileSystemCapabilities(read_text_file=True, write_text_file=True)))
            assert init.agent_capabilities.prompt_capabilities.image
            session = await conn.new_session(cwd=str(tmp_path), mcp_servers=[])
            task = asyncio.create_task(conn.prompt(session_id=session.session_id,
                prompt=[text_block("Original image observation"), ImageContentBlock(type="image", data="aGVsbG8=", mime_type="image/png")]))
            for _ in range(100):
                if (tmp_path / "captured.json").exists():
                    break
                await asyncio.sleep(0.02)
            captured = json.loads((tmp_path / "captured.json").read_text())
            assert captured["images"] == [{"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"}]
            assert "Original image observation" in captured["message"] and "validate_candidate" in captured["message"]
            await conn.cancel(session_id=session.session_id)
            result = await asyncio.wait_for(task, timeout=10)
            assert result.stop_reason == "cancelled"
            assert result.usage is None
    asyncio.run(exercise())


def test_accounting_duplicate_ids_missing_usage_and_receipt_mismatch(tmp_path):
    from pi_nvidia.accounting import snapshot
    sessions = tmp_path / "pi/sessions"
    sessions.mkdir(parents=True)
    p = ModelProfile.model_validate(profile())
    rows = [{"type": "session", "id": "s"},
            {"type": "model_change", "provider": "harbor-endpoint", "modelId": p.id},
            {"type": "thinking_level_change", "thinkingLevel": "off"},
            {"type": "message", "id": "m", "message": {"role": "assistant", "usage": {"input": 20, "output": 10, "cacheRead": 5, "cacheWrite": 0}}}]
    path = sessions / "s.jsonl"
    def save(data):
        path.write_text("".join(json.dumps(e) + "\n" for e in data))
    save(rows + [rows[-1]])
    result = snapshot(sessions, "s", p, "off")
    assert result["status"] == "verified" and result["call_count"] == 1
    assert result["usage"]["inputTokens"] == 25
    conflict = copy.deepcopy(rows[-1])
    conflict["message"]["usage"]["input"] = 99
    save(rows + [conflict])
    assert snapshot(sessions, "s", p, "off")["status"] == "unavailable"
    rows[-1]["message"].pop("usage")
    save(rows)
    assert snapshot(sessions, "s", p, "off")["status"] == "unavailable"
    assert snapshot(sessions, "s", p, "high")["reason"] == "native_thinking_receipt_mismatch"


def test_hosted_manifest_matches_installed_harbor_contract():
    from harbor.models.agent.acp_source import AcpSourceManifest
    m = AcpSourceManifest.model_validate_json((ROOT / "harbor-agent.json").read_text())
    assert m.protocol == "acp" and m.runtime.entrypoint == ["python", "-m", "pi_nvidia.acp"]


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires installed pinned runtime")
def test_real_manifest_entrypoint_initializes_without_credentials_or_model_call(tmp_path):
    from acp import PROTOCOL_VERSION, spawn_agent_process
    from acp.schema import ClientCapabilities
    from harbor.agents.installed.acp_runner import HarborAcpClient
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {"PATH": os.defpath, "PYTHONPATH": str(ROOT),
        "PI_NVIDIA_PREFIX": os.environ["PI_TEST_PREFIX"], "PI_NVIDIA_LOGS": str(logs),
        "PI_NVIDIA_PROFILE": json.dumps(profile()), "PI_NVIDIA_THINKING": "off",
        "HARBOR_ACP_REQUESTED_MODEL": "nvidia/fixture/vision"}
    client = HarborAcpClient(logs_dir=logs, permission_mode="allow")
    async def exercise():
        async with spawn_agent_process(client, sys.executable, "-m", "pi_nvidia.acp", env=env, cwd=str(tmp_path)) as (conn, proc):
            result = await conn.initialize(protocol_version=PROTOCOL_VERSION, client_capabilities=ClientCapabilities())
            assert result.agent_info.name == "pi-nvidia-feedback"
            assert result.agent_info.version == "0.2.0"
            assert result.auth_methods == []
            assert not result.agent_capabilities.load_session
    asyncio.run(exercise())
