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
raise SystemExit(proxy([sys.executable, sys.argv[2]],
                       dict(os.environ), '/fixture/agent', Path(sys.argv[1])))
"""
    server = tmp_path / "fixture_acp.py"
    server.write_text('''import sys,json,os
from pathlib import Path
for raw in sys.stdin:
 r=json.loads(raw)
 if r['method']=='session/new':
  result={'sessionId':'s','models':{'currentModelId':'gpt-6.1-terra[low]'}}
 else:
  path=Path(os.environ['CODEX_HOME'])/'sessions/rollout-fixture.jsonl'
  path.parent.mkdir(parents=True,exist_ok=True)
  events=[{'type':'session_meta','payload':{'id':'s'}},
          {'type':'turn_context','payload':{'model':'gpt-6.1-terra','effort':'low'}}]
  previous=(0,0,0)
  for inp,cached,out in [(1000,600,100),(3000,2100,300)]:
   last=dict(zip(['input_tokens','cached_input_tokens','output_tokens'],[inp-previous[0],cached-previous[1],out-previous[2]]))
   events.append({'type':'event_msg','payload':{'type':'token_count','info':{'total_token_usage':{
       'input_tokens':inp,'cached_input_tokens':cached,'output_tokens':out},'last_token_usage':last,'model_context_window':1050000}}})
   previous=(inp,cached,out)
  path.write_text(''.join(json.dumps(e)+'\\n' for e in events))
  result={'receivedPrompt':r['params']['prompt'],'usage':{'inputTokens':500,'cachedReadTokens':1500,'outputTokens':200}}
 print(json.dumps({'jsonrpc':'2.0','id':r['id'],'result':result}),flush=True)
''')
    request = {"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": {
        "sessionId": "s", "prompt": [{"type": "text", "text": "original request"}]}}
    env = child_environment(tmp_path, dict(os.environ, PYTHONPATH=str(root),
        HARBOR_ACP_REQUESTED_MODEL="openai/gpt-6.1-terra", CAD_AGENT_REASONING_EFFORT="low"))
    env['CODEX_HOME'] = str(tmp_path / ".codex-text2cad")
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(server)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
    process.stdin.write(json.dumps({'jsonrpc':'2.0','id':1,'method':'session/new','params':{}})+'\n')
    process.stdin.flush()
    assert json.loads(process.stdout.readline())['result']['sessionId']=='s'
    stdout, stderr = process.communicate(json.dumps(request)+'\n',timeout=30)
    assert process.returncode == 0, stderr
    messages = [json.loads(line) for line in stdout.splitlines()]
    message = next(m for m in messages if m.get('id')==2)
    assert message["result"]["receivedPrompt"][-1]["text"] == "original request"
    assert "<selected-cad-skill>" in message["result"]["receivedPrompt"][0]["text"]
    assert "http://feedback:8080/health" in message["result"]["receivedPrompt"][0]["text"]
    assert "Reserve the last available call" in message["result"]["receivedPrompt"][0]["text"]
    assert message["result"]["usage"]["inputTokens"] == 3000
    assert message["result"]["usage"]["cachedReadTokens"] == 2100
    assert message["result"]["_meta"]["cad_agent_accounting"]["upstream_usage"]["inputTokens"] == 500
    cost_update = next(m for m in messages if m.get('method')=='session/update')
    assert cost_update['params']['update']['cost']['amount'] == pytest.approx(0.002505)
    assert messages.index(cost_update) < messages.index(message)
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
