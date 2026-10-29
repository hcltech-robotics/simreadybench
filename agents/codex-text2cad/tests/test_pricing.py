import json
from pathlib import Path
from decimal import Decimal

import pytest

from cad_agent.pricing import estimate_cost
from cad_agent.accounting import AccountingBridge, native_snapshot

POLICY = "openai-standard-global-30m"


def journal(home, calls, model="gpt-6.1-terra", sid="s"):
    folder = home / "sessions"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rollout-pricing.jsonl"
    events = [{"type": "session_meta", "payload": {"id": sid}}]
    total = dict(input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning_output_tokens=0, cache_write_input_tokens=0)
    for call in calls:
        context = {"model": model, "effort": "low", **call.get("context", {})}
        events.append({"type": "turn_context", "payload": context})
        usage = {k: call.get(k, 0) for k in total}
        for key in total:
            total[key] += usage[key]
        event = {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": dict(total), "last_token_usage": usage, "model_context_window": 1050000}}}
        events.append(event)
        if call.get("duplicate"):
            events.append(event)
    path.write_text("".join(json.dumps(e)+"\n" for e in events))
    return path, total


def live_counts_calls():
    return [dict(input_tokens=70000, cached_input_tokens=65000, output_tokens=400, reasoning_output_tokens=100, duplicate=True)] * 10 + [
        dict(input_tokens=69755, cached_input_tokens=63850, output_tokens=715, reasoning_output_tokens=177)]


@pytest.mark.parametrize("model,expected", [("gpt-6.1-terra", "0.1151725"), ("gpt-6.1-sol", "0.230345")])
@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_live_fixture_counts_are_priced_per_request_not_cumulative_context(tmp_path, model, expected, effort):
    home = tmp_path / ".codex-text2cad"
    calls = [{**call, "context": {"effort": effort}} for call in live_counts_calls()]
    _, total = journal(home, calls, model)
    assert total["input_tokens"] == 769755 and total["cached_input_tokens"] == 713850
    assert total["output_tokens"] == 4715 and total["reasoning_output_tokens"] == 1177
    snapshot = native_snapshot(home / "sessions", "s", model, effort)
    assert snapshot["status"] == "verified"
    assert all(turn == {"model": model, "effort": effort} for turn in snapshot["turns"])
    assert native_snapshot(home / "sessions", "s", model, "xhigh")["status"] == "unverified"
    result = estimate_cost(snapshot, model, POLICY)
    assert result["status"] == "estimated" and result["billed"] is False
    assert Decimal(str(result["amount_usd"])) == Decimal(expected)
    assert result["request_count"] == 11 and result["long_context_requests"] == 0
    assert result["totals"] == snapshot["totals"]


def test_mixed_context_tier_regional_and_cache_write(tmp_path):
    home = tmp_path / ".codex-text2cad"
    calls = [dict(input_tokens=100000, cached_input_tokens=80000, output_tokens=1000),
             dict(input_tokens=300000, cached_input_tokens=200000, output_tokens=2000, cache_write_input_tokens=10000,
                  context={"service_tier": "priority", "regional_processing": True, "prompt_cache_retention": "12h"})]
    journal(home, calls)
    result = estimate_cost(native_snapshot(home / "sessions", "s", "gpt-6.1-terra", "low"), "gpt-6.1-terra", POLICY)
    assert Decimal(str(result["amount_usd"])) == Decimal("0.612")
    assert result["long_context_requests"] == 1
    assert result["requests"][1]["tier"] == "priority"


@pytest.mark.parametrize("change", ["unknown_model", "unknown_tier", "missing_last_usage", "unreconciled", "unknown_policy", "invalid_retention_type"])
def test_unknown_pricing_or_usage_does_not_invent_a_cost(tmp_path, change):
    home = tmp_path / ".codex-text2cad"
    path, _ = journal(home, [dict(input_tokens=1000, cached_input_tokens=500, output_tokens=100)])
    snapshot = native_snapshot(home / "sessions", "s", "gpt-6.1-terra", "low")
    model, policy = "gpt-6.1-terra", POLICY
    if change == "unknown_model": model = "unpriced-model"
    if change == "unknown_tier": snapshot["requests"][0]["context"]["service_tier"] = "unpriced-tier"
    if change == "missing_last_usage":
        events = [json.loads(line) for line in path.read_text().splitlines()]
        del events[-1]["payload"]["info"]["last_token_usage"]
        path.write_text("".join(json.dumps(e)+"\n" for e in events))
        snapshot = native_snapshot(home / "sessions", "s", model, "low")
    if change == "unreconciled": snapshot["totals"]["input_tokens"] += 1
    if change == "unknown_policy": policy = "unpriced-policy"
    if change == "invalid_retention_type": snapshot["requests"][0]["context"]["prompt_cache_retention"] = {}
    result = estimate_cost(snapshot, model, policy)
    assert result["status"] == "unavailable" and "amount_usd" not in result


def start_bridge(home, logs):
    b = AccountingBridge(home, logs, "gpt-6.1-terra", "low", pricing_policy=POLICY)
    b.request({"id": 1, "method": "session/new", "params": {}})
    b.response({"id": 1, "result": {"sessionId": "s", "models": {"currentModelId": "gpt-6.1-terra[low]"}}})
    return b


def test_harbor_receives_cumulative_usd_and_late_updates_keep_it(tmp_path):
    from harbor.agents.installed.acp import AcpAgent
    from harbor.models.agent.context import AgentContext
    from harbor.models.agent.acp_source import AcpSourceManifest
    from harbor.agents.installed.acp_runner import HarborAcpClient
    from acp.schema import UsageUpdate
    import asyncio
    home = tmp_path / ".codex-text2cad"
    journal(home, live_counts_calls())
    bridge = start_bridge(home, tmp_path / "logs")
    bridge.response({"method": "session/update", "params": {"sessionId": "s", "update": {
        "sessionUpdate": "usage_update", "used": 70470, "size": 1050000}}})
    bridge.request({"id": 2, "method": "session/prompt", "params": {"sessionId": "s"}})
    response = bridge.response({"id": 2, "result": {"stopReason": "end_turn", "usage": {
        "inputTokens": 5905, "cachedReadTokens": 63850, "outputTokens": 715}}})
    notifications = bridge.drain_notifications()
    cost_update = notifications[-1]["params"]["update"]
    assert cost_update["cost"]["currency"] == "USD"
    assert cost_update["cost"]["amount"] == pytest.approx(0.1151725)
    late = bridge.response({"method": "session/update", "params": {"sessionId": "s", "update": {
        "sessionUpdate": "usage_update", "used": 70470, "size": 1050000}}})
    assert late["params"]["update"]["cost"] == cost_update["cost"]
    agent = AcpAgent(logs_dir=tmp_path / "harbor", model_name="openai/gpt-6.1-terra", source={
        "repo_url": "https://github.com/hcltech-robotics/simreadybench.git", "ref": "8d6a88938955c844fb2f87c71a7e2613475856c3",
        "source_dir": "agents/codex-text2cad", "manifest_path": "harbor-agent.json"})
    agent._source_manifest = AcpSourceManifest.model_validate_json((Path(__file__).parents[1] / "harbor-agent.json").read_text())
    client_logs = tmp_path / "client"
    client_logs.mkdir()
    client = HarborAcpClient(logs_dir=client_logs, permission_mode="allow")
    asyncio.run(client.session_update("s", UsageUpdate.model_validate(late["params"]["update"])))
    summary = {"instruction": "fixture", "session": {"sessionId": "s"}, "prompt_response": response["result"], "latest_usage_update": client.latest_usage_update}
    agent._load_summary = lambda: summary
    agent._write_trajectory = lambda **kwargs: None
    context = AgentContext(); agent.populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (769755, 713850, 4715)
    assert context.cost_usd == pytest.approx(0.1151725)
    events=[json.loads(line) for line in (client_logs/'acp-events.jsonl').read_text().splitlines()]
    trajectory = agent._convert_events_to_trajectory(summary=summary, events=events)
    assert trajectory.final_metrics.total_cost_usd == pytest.approx(0.1151725)
    assert trajectory.final_metrics.total_prompt_tokens == 769755
    assert '"billed": false' in json.dumps(trajectory.to_json_dict())
    assert cost_update["cost"]["_meta"]["billed"] is False
    assert client.latest_usage_update['cost']['_meta']['status']=='estimated'


def test_multiple_prompts_and_upstream_cost_are_not_double_counted_or_overwritten(tmp_path):
    home = tmp_path / ".codex-text2cad"
    calls = live_counts_calls()
    journal(home, calls)
    b = start_bridge(home, tmp_path / "logs")
    for rid in (2, 3):
        b.request({'id':rid,'method':'session/prompt','params':{'sessionId':'s'}})
        b.response({'id':rid,'result':{'usage':{'inputTokens':5905,'cachedReadTokens':63850,'outputTokens':715}}})
        assert b.drain_notifications()[-1]['params']['update']['cost']['amount']==pytest.approx(0.1151725)
    calls += [dict(input_tokens=1000,cached_input_tokens=500,output_tokens=100)]
    journal(home,calls)
    b.request({'id':4,'method':'session/prompt','params':{'sessionId':'s'}})
    b.response({'id':4,'result':{'usage':{'inputTokens':500,'cachedReadTokens':500,'outputTokens':100}}})
    assert b.drain_notifications()[-1]['params']['update']['cost']['amount']==pytest.approx(0.1161975)
    upstream={'amount':0.3,'currency':'USD','_meta':{'source':'upstream_fixture'}}
    b.response({'method':'session/update','params':{'sessionId':'s','update':{'sessionUpdate':'usage_update','used':1100,'size':1050000,'cost':upstream}}})
    b.request({'id':5,'method':'session/prompt','params':{'sessionId':'s'}})
    b.response({'id':5,'result':{'usage':{'inputTokens':500,'cachedReadTokens':500,'outputTokens':100}}})
    assert not b.drain_notifications()
    assert b.costs['s']==upstream
    b.pricing_policy='unknown-policy'
    b.request({'id':6,'method':'session/prompt','params':{'sessionId':'s'}})
    b.response({'id':6,'result':{'usage':{'inputTokens':500,'cachedReadTokens':500,'outputTokens':100}}})
    assert not b.drain_notifications() and b.costs['s']==upstream


def test_raw_response_ids_deduplicate_and_conflicts_do_not_price(tmp_path):
    home=tmp_path/'.codex-text2cad'
    path,_=journal(home,[dict(input_tokens=1000,cached_input_tokens=500,output_tokens=100)])
    raw={'type':'event_msg','payload':{'type':'raw_response_completed','response_id':'response-1',
        'token_usage':{'input_tokens':1000,'cached_input_tokens':500,'output_tokens':100},
        'usage_metadata':{'metadata':{'service_tier':'default'}}}}
    with path.open('a') as f:
        f.write(json.dumps(raw)+'\n');f.write(json.dumps(raw)+'\n')
    result=estimate_cost(native_snapshot(home/'sessions','s','gpt-6.1-terra','low'),'gpt-6.1-terra',POLICY)
    assert result['status']=='estimated' and result['request_count']==1
    raw['payload']['token_usage']['output_tokens']=101
    with path.open('a') as f:f.write(json.dumps(raw)+'\n')
    result=estimate_cost(native_snapshot(home/'sessions','s','gpt-6.1-terra','low'),'gpt-6.1-terra',POLICY)
    assert result['status']=='unavailable'
