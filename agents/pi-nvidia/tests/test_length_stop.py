import json
import pytest

from pi_nvidia.accounting import snapshot
from pi_nvidia.acp import Bridge
from pi_nvidia.profile import ModelProfile
from test_agent import profile


def fixture(tmp_path, *, stop="length", content=None):
    sessions = tmp_path / "pi/sessions"
    sessions.mkdir(parents=True)
    p = ModelProfile.model_validate(profile())
    rows = [{"type": "session", "id": "s"},
        {"type": "model_change", "provider": "harbor-endpoint", "modelId": p.id},
        {"type": "thinking_level_change", "thinkingLevel": "off"},
        {"type": "message", "id": "m1", "message": {"role": "assistant", "stopReason": stop,
         "content": content if content is not None else [{"type": "thinking", "thinking": "private fixture reasoning"}],
         "usage": {"input": 100, "cacheRead": 20, "cacheWrite": 0, "output": 8192}}}]
    path = sessions / "s.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    b = Bridge(p, "off", sessions, tmp_path)
    b.ready.add("s")
    b.request({"id": 7, "method": "session/prompt", "params": {"sessionId": "s", "prompt": [{"type": "text", "text": "Original task"}]}})
    return b, rows, path


def test_length_receipt_records_only_stop_and_content_flags(tmp_path):
    b, _, _ = fixture(tmp_path)
    receipt = snapshot(b.sessions, "s", b.profile, b.effort)
    assert receipt["last_assistant"] == {"stop_reason": "length", "has_text": False,
                                        "has_tool_call": False, "empty_length_stop": True}
    assert "private fixture reasoning" not in json.dumps(receipt)


def test_empty_length_gets_one_continuation_then_max_tokens_with_all_usage(tmp_path):
    b, rows, path = fixture(tmp_path)
    response = b.response({"id": 7, "result": {"stopReason": "end_turn"}})
    continuation = response.request
    assert continuation["method"] == "session/prompt" and continuation["params"]["sessionId"] == "s"
    assert continuation["id"] != 7
    rows.append({"type": "message", "id": "m2", "message": {
        "role": "assistant", "stopReason": "length", "content": [],
        "usage": {"input": 200, "cacheRead": 40, "cacheWrite": 0, "output": 8192}}})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    final = b.response({"id": continuation["id"], "result": {"stopReason": "end_turn"}})
    assert final["id"] == 7 and final["result"]["stopReason"] == "max_tokens"
    assert final["result"]["usage"]["inputTokens"] == 360
    assert final["result"]["usage"]["outputTokens"] == 16384
    assert final["result"]["_meta"]["pi_nvidia_accounting"]["length_recovery_count"] == 1


@pytest.mark.parametrize("content", [[{"type": "text", "text": "Partial answer"}],
                                     [{"type": "toolCall", "id": "partial", "name": "bash", "arguments": {}}]])
def test_nonempty_length_is_incomplete_without_auto_continuation(tmp_path, content):
    b, *_ = fixture(tmp_path, content=content)
    final = b.response({"id": 7, "result": {"stopReason": "end_turn"}})
    assert final["result"]["stopReason"] == "max_tokens"


def test_cancelled_length_never_continues(tmp_path):
    b, *_ = fixture(tmp_path)
    final = b.response({"id": 7, "result": {"stopReason": "cancelled"}})
    assert final["result"]["stopReason"] == "cancelled"


def test_cancel_notification_prevents_racing_length_continuation(tmp_path):
    b, *_ = fixture(tmp_path)
    b.request({"method": "session/cancel", "params": {"sessionId": "s"}})
    final = b.response({"id": 7, "result": {"stopReason": "end_turn"}})
    assert final["result"]["stopReason"] == "cancelled"


def test_unverified_usage_does_not_start_another_model_request(tmp_path):
    b, rows, path = fixture(tmp_path)
    rows[-1]["message"].pop("usage")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    final = b.response({"id": 7, "result": {"stopReason": "end_turn"}})
    assert final["result"]["stopReason"] == "max_tokens" and "usage" not in final["result"]


def test_continuation_error_is_returned_under_original_request_id(tmp_path):
    b, *_ = fixture(tmp_path)
    follow = b.response({"id": 7, "result": {"stopReason": "end_turn"}}).request
    result = b.response({"id": follow["id"], "error": {"code": -32000, "message": "fixture provider failure"}})
    assert result["id"] == 7 and "error" in result
