import json
from pathlib import Path

import pytest

from cad_agent.accounting import AccountingBridge, native_snapshot, retain_evidence, normalize_atif


def write_session(home, sid="session-1", effort="low", totals=(3000, 2100, 300)):
    root = home / ".codex-text2cad/sessions/2026/10/29"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"rollout-2026-10-29-{sid}.jsonl"
    inp, cache, out = totals
    events = [
        {"type": "session_meta", "payload": {"id": sid}},
        {"type": "turn_context", "payload": {"model": "gpt-6.1-terra", "effort": effort}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 100},
            "last_token_usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 100}}}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": inp, "cached_input_tokens": cache, "output_tokens": out},
            "last_token_usage": {"input_tokens": 2000, "cached_input_tokens": 1500, "output_tokens": 200}}}},
    ]
    path.write_text("".join(json.dumps(e)+"\n" for e in events))
    return path


def bridge(tmp_path):
    home = tmp_path / "home"
    write_session(home)
    return AccountingBridge(home / ".codex-text2cad", tmp_path / "logs",
                            "gpt-6.1-terra", "low"), home


def observe_new(b):
    b.request({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {}})
    b.response({"jsonrpc": "2.0", "id": 1, "result": {
        "sessionId": "session-1", "models": {"currentModelId": "gpt-6.1-terra[low]"}}})


@pytest.mark.parametrize("model", ["gpt-6.1-terra", "gpt-6.1-sol"])
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh"])
def test_models_require_the_selected_effort_receipt(tmp_path, model, effort):
    b = AccountingBridge(tmp_path / ".codex-text2cad", tmp_path / "logs", model, effort)
    b.request({"id": 1, "method": "session/new", "params": {}})
    b.response({"id": 1, "result": {"sessionId": "s", "models": {"currentModelId": model + f"[{effort}]"}}})
    forwarded = b.request({"id": 2, "method": "session/prompt", "params": {"sessionId": "s"}})
    assert forwarded["params"]["sessionId"] == "s"


def test_bidirectional_rpc_ids_do_not_consume_client_request(tmp_path):
    b, _ = bridge(tmp_path)
    observe_new(b)
    b.request({"id": 2, "method": "session/prompt", "params": {"sessionId": "session-1"}})
    server_request = {"id": 2, "method": "terminal/create", "params": {"command": "fixture"}}
    assert b.response(server_request) == server_request
    client_response = {"id": 2, "result": {"terminalId": "t"}}
    assert b.request(client_response) == client_response
    response = b.response({"id": 2, "result": {"usage": {"inputTokens": 500, "cachedReadTokens": 1500, "outputTokens": 200}}})
    assert response["result"]["usage"]["inputTokens"] == 3000


def test_cumulative_inclusive_cache_and_no_double_count(tmp_path):
    b, home = bridge(tmp_path)
    observe_new(b)
    upstream = {"inputTokens": 500, "cachedReadTokens": 1500, "outputTokens": 200, "totalTokens": 2200}
    for rid in (2, 3):
        b.request({"id": rid, "method": "session/prompt", "params": {"sessionId": "session-1", "prompt": []}})
        result = b.response({"id": rid, "result": {"stopReason": "end_turn", "usage": upstream, "_meta": {"preserve": True}}})
        assert result["result"]["usage"] == {"inputTokens": 3000, "cachedReadTokens": 2100, "outputTokens": 300, "totalTokens": 3300}
        meta = result["result"]["_meta"]
        assert meta["preserve"] and meta["cad_agent_accounting"]["upstream_usage"] == upstream
        assert meta["cad_agent_accounting"]["input_semantics"] == "inclusive_of_cached_input"
    saved = json.loads((tmp_path / "logs/accounting.json").read_text())
    assert saved["sessions"]["session-1"]["totals"]["input_tokens"] == 3000
    assert len(saved["sessions"]) == 1
    assert "cost_usd" not in saved["sessions"]["session-1"]


def test_actual_low_gate_and_legacy_model_compatibility(tmp_path):
    b, _ = bridge(tmp_path)
    request = b.request({"id": 2, "method": "session/set_model", "params": {
        "sessionId": "session-1", "modelId": "openai/gpt-6.1-terra"}})
    assert request["params"]["modelId"] == "gpt-6.1-terra[low]"
    # Requested environment alone never qualifies as an actual receipt.
    with pytest.raises(ValueError, match="reasoning"):
        b.request({"id": 3, "method": "session/prompt", "params": {"sessionId": "session-1"}})
    observe_new(b)
    b.request({"id": 4, "method": "session/prompt", "params": {"sessionId": "session-1"}})
    wrong = AccountingBridge(tmp_path / "home/.codex-text2cad", tmp_path / "wrong", "gpt-6.1-terra", "low")
    wrong.request({"id": 1, "method": "session/new", "params": {}})
    wrong.response({"id": 1, "result": {"sessionId": "s", "models": {"currentModelId": "gpt-6.1-terra[medium]"}}})
    with pytest.raises(ValueError, match="reasoning"):
        wrong.request({"id": 2, "method": "session/prompt", "params": {"sessionId": "s"}})


def test_native_turn_mismatch_and_counter_reset_are_not_verified(tmp_path):
    home = tmp_path / "home"
    path = write_session(home, effort="medium")
    snap = native_snapshot(home / ".codex-text2cad/sessions", "session-1", "gpt-6.1-terra", "low")
    assert snap["status"] == "unverified" and "reasoning" in snap["reason"]
    path = write_session(home)
    with path.open("a") as f:
        f.write(json.dumps({"type": "event_msg", "payload": {"type": "token_count", "info": {
            "total_token_usage": {"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}}}})+"\n")
    assert native_snapshot(home / ".codex-text2cad/sessions", "session-1", "gpt-6.1-terra", "low")["status"] == "unverified"


def test_additional_native_sessions_are_flagged_not_double_counted(tmp_path):
    b, home = bridge(tmp_path)
    observe_new(b)
    write_session(home, sid="child-session")
    b.request({"id": 2, "method": "session/prompt", "params": {"sessionId": "session-1"}})
    b.response({"id": 2, "result": {"usage": {"inputTokens": 500, "cachedReadTokens": 1500, "outputTokens": 200}}})
    record = json.loads((tmp_path / "logs/accounting.json").read_text())
    assert record["other_native_session_ids"] == ["child-session"]
    assert record["sessions"]["session-1"]["totals"]["input_tokens"] == 3000
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(json.dumps({"session_id": "session-1"}))
    with pytest.raises(ValueError, match="Additional native sessions"):
        normalize_atif(trajectory, tmp_path / "logs/accounting.json")


def test_missing_or_other_home_usage_is_not_fabricated(tmp_path):
    b, home = bridge(tmp_path)
    observe_new(b)
    for p in (home / ".codex-text2cad/sessions").rglob("*.jsonl"):
        p.unlink()
    b.request({"id": 2, "method": "session/prompt", "params": {"sessionId": "session-1"}})
    upstream = {"inputTokens": 5, "outputTokens": 2}
    response = b.response({"id": 2, "result": {"usage": upstream}})
    assert response["result"]["usage"] == upstream
    assert response["result"]["_meta"]["cad_agent_accounting"]["status"] == "unverified"
    with pytest.raises(ValueError, match="codex-text2cad"):
        native_snapshot(home / ".codex/sessions", "s", "gpt-6.1-terra", "low")


def test_retention_allowlist_symlinks_and_secret_redaction(tmp_path):
    home = tmp_path / "home"
    session = write_session(home)
    with session.open("a") as f:
        f.write(json.dumps({"type": "response_item", "payload": {"text": "test-provider-credential", "api_key": "other-credential"}})+"\n")
    (home / ".codex-text2cad/auth.json").write_text("test-provider-credential")
    unrelated = home / ".codex/sessions"
    unrelated.mkdir(parents=True)
    (unrelated / "rollout-unrelated.jsonl").write_text("not this trial")
    (session.parent / "rollout-link.jsonl").symlink_to(home / ".codex-text2cad/auth.json")
    author = tmp_path / "authoring"
    evidence = author / "cad-evidence"
    evidence.mkdir(parents=True)
    (author / "reconstruct.py").write_text("# test-provider-credential\n")
    (author / "secret.env").write_text("test-provider-credential")
    (evidence / "part.step").write_text("STEP geometry")
    (evidence / "generation.json").write_text(json.dumps({"source_path": str(author / "reconstruct.py"),
        "parts": [{"step_path": str(evidence / "part.step")}]}))
    logs = tmp_path / "logs"
    retain_evidence(home / ".codex-text2cad", author, logs, {"OPENAI_API_KEY": "test-provider-credential"})
    kept = [p for p in logs.rglob("*") if p.is_file()]
    names = [p.name for p in kept]
    assert "reconstruct.py" in names and "part.step" in names and "generation.json" in names
    assert "auth.json" not in names and "secret.env" not in names and "rollout-unrelated.jsonl" not in names and "rollout-link.jsonl" not in names
    assert all(b"test-provider-credential" not in p.read_bytes() and b"other-credential" not in p.read_bytes() for p in kept)


def test_atif_normalization_preserves_upstream_and_is_idempotent(tmp_path):
    trajectory = tmp_path / "trajectory.json"
    original = {"session_id": "s", "steps": [{"source": "user"}, {"source": "agent", "metrics": {"prompt_tokens": 5}}],
        "final_metrics": {"total_prompt_tokens": 5, "total_completion_tokens": 2, "total_cost_usd": None}}
    trajectory.write_text(json.dumps(original))
    accounting = tmp_path / "accounting.json"
    accounting.write_text(json.dumps({"sessions": {"s": {"status": "verified", "basis": "native_session_cumulative", "totals": {
        "input_tokens": 3000, "cached_input_tokens": 2100, "output_tokens": 300}}}}))
    normalize_atif(trajectory, accounting)
    normalized = trajectory.read_bytes()
    assert json.loads(trajectory.with_name("trajectory.upstream.json").read_text()) == original
    assert json.loads(normalized)["final_metrics"]["total_cached_tokens"] == 2100
    assert json.loads(normalized)["final_metrics"]["total_cost_usd"] is None
    normalize_atif(trajectory, accounting)
    assert trajectory.read_bytes() == normalized
