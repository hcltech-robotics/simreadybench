"""Execute Pi's round feedback hooks without model calls or transcript writes."""

import json
import shutil
import subprocess

import pytest

from simready_bench.engine.pi import PiAgent


NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js executes Pi extension hooks")


def exercise(tmp_path, turns):
    extension = tmp_path / "rounds.mjs"
    extension.write_text(PiAgent._build_max_turns_extension(turns), encoding="ascii")
    script = tmp_path / "exercise.mjs"
    script.write_text(
        r'''
import install from "./rounds.mjs";
const hooks = new Map();
install({ on: (name, callback) => hooks.set(name, callback) });
let aborts = 0;
const ctx = { abort: () => { aborts += 1; } };
const original = [
  { role: "user", content: [
    { type: "text", text: "Build this object" },
    { type: "image", data: "AQID", mimeType: "image/png" },
  ], timestamp: 1 },
  { role: "assistant", content: [
    { type: "toolCall", id: "call-1", name: "bash", arguments: { command: "inspect" } },
  ], timestamp: 2 },
  { role: "toolResult", toolCallId: "call-1", toolName: "bash", content: [
    { type: "text", text: "dimensions" },
    { type: "image", data: "BAUG", mimeType: "image/png" },
  ], isError: false, timestamp: 3 },
  { role: "custom", customType: "other-extension", content: "Keep this", timestamp: 4 },
];
const before = JSON.stringify(original);
const rounds = [];
let previous;
for (let i = 0; i < Number(process.argv[2]); i += 1) {
  const first = hooks.get("context")({ messages: original }, ctx).messages;
  const repeated = hooks.get("context")({ messages: first }, ctx).messages;
  const refreshed = hooks.get("context")({ messages: previous ?? original }, ctx).messages;
  const notices = (messages) => messages.filter(
    (message) => message.role === "custom" && message.customType === "simready-authoring-round",
  );
  rounds.push({
    notice: notices(first)[0],
    noticeCount: notices(repeated).length,
    repeatedContent: notices(repeated)[0].content,
    refreshedContent: notices(refreshed)[0].content,
    preserved: JSON.stringify(first.slice(0, -1)) === before,
    originalUnchanged: JSON.stringify(original) === before,
    abortsBeforeEnd: aborts,
  });
  previous = first;
  hooks.get("turn_end")({}, ctx);
  rounds.at(-1).abortsAfterEnd = aborts;
}
console.log(JSON.stringify({
  hooks: [...hooks.keys()],
  prompt: hooks.get("before_agent_start")({ systemPrompt: "Original prompt" }, ctx).systemPrompt,
  rounds,
  aborts,
}));
''',
        encoding="ascii",
    )
    result = subprocess.run(
        [NODE, str(script), str(turns)],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return json.loads(result.stdout)


def test_feedback_is_current_on_first_penultimate_and_final_round(tmp_path):
    result = exercise(tmp_path, 64)
    rounds = result["rounds"]
    assert rounds[0]["notice"]["content"] == (
        "Authoring round 1 of 64. 63 model rounds remain after this round."
    )
    assert rounds[-2]["notice"]["content"] == (
        "Authoring round 63 of 64. 1 model round remains after this round."
    )
    assert rounds[-1]["notice"]["content"] == (
        "Authoring round 64 of 64. 0 model rounds remain after this round. "
        "This is your final model round. Save the USD asset and required files, "
        "then submit now. No further model calls will be available."
    )
    assert result["prompt"].startswith("Original prompt\n\n")
    assert "hard budget of 64 model turns" in result["prompt"]
    assert all(row["abortsBeforeEnd"] == 0 for row in rounds)
    assert all(row["abortsAfterEnd"] == 0 for row in rounds[:-1])
    assert rounds[-1]["abortsAfterEnd"] == result["aborts"] == 1


def test_context_refresh_keeps_one_notice_without_consuming_rounds(tmp_path):
    result = exercise(tmp_path, 3)
    assert set(result["hooks"]) == {"before_agent_start", "context", "turn_end"}
    for row in result["rounds"]:
        assert row["noticeCount"] == 1
        assert row["repeatedContent"] == row["notice"]["content"]
        assert row["refreshedContent"] == row["notice"]["content"]
        assert row["preserved"] and row["originalUnchanged"]
        assert row["notice"]["display"] is False
    assert result["aborts"] == 1


def test_one_round_warns_before_the_first_call_and_stops_after_it(tmp_path):
    result = exercise(tmp_path, 1)
    row = result["rounds"][0]
    assert row["notice"]["content"].startswith(
        "Authoring round 1 of 1. 0 model rounds remain after this round."
    )
    assert "submit now" in row["notice"]["content"]
    assert row["abortsBeforeEnd"] == 0
    assert row["abortsAfterEnd"] == result["aborts"] == 1


def exercise_stops(tmp_path, stops, *, turns=64, enabled=True):
    extension = tmp_path / "stops.mjs"
    extension.write_text(
        PiAgent._build_max_turns_extension(turns, continue_on_length=enabled),
        encoding="ascii",
    )
    script = tmp_path / "stops-test.mjs"
    script.write_text(
        '''import install from "./stops.mjs";
const hooks = new Map();
const followUps = [];
install({ on: (name, handler) => hooks.set(name, handler),
  sendUserMessage: (content, options) => followUps.push({content, options}) });
let aborts = 0;
const ctx = { abort: () => { aborts += 1; } };
const counts = [];
for (const message of JSON.parse(process.argv[2])) {
  hooks.get("turn_end")({message, toolResults: []}, ctx);
  counts.push(followUps.length);
}
console.log(JSON.stringify({followUps, counts, aborts}));
''',
        encoding="ascii",
    )
    result = subprocess.run(
        [NODE, str(script), json.dumps(stops)], check=True,
        capture_output=True, text=True, timeout=15,
    )
    return json.loads(result.stdout)


def assistant_stop(reason, content=None):
    return {"role": "assistant", "stopReason": reason, "content": content or []}


def test_text_truncation_queues_one_native_follow_up_per_turn(tmp_path):
    result = exercise_stops(tmp_path, [assistant_stop("length")] * 3)
    assert result["counts"] == [1, 2, 3]
    assert result["aborts"] == 0
    for follow_up in result["followUps"]:
        assert follow_up["options"] == {"deliverAs": "followUp"}
        assert "current files and tool results" in follow_up["content"]


@pytest.mark.parametrize("reason", ["error", "aborted", "stop", "toolUse"])
def test_continuation_does_not_change_other_terminal_reasons(tmp_path, reason):
    result = exercise_stops(tmp_path, [assistant_stop(reason)])
    assert result["followUps"] == []
    assert result["aborts"] == 0


def test_truncated_tool_call_keeps_native_rejection_and_continuation(tmp_path):
    result = exercise_stops(tmp_path, [assistant_stop("length", [
        {"type": "toolCall", "id": "partial", "name": "bash", "arguments": {}},
    ])])
    assert result["followUps"] == []


def test_continuation_uses_same_hard_turn_budget(tmp_path):
    result = exercise_stops(tmp_path, [assistant_stop("length")] * 64)
    assert len(result["followUps"]) == 63
    assert result["counts"][-2:] == [63, 63]
    assert result["aborts"] == 1


def test_existing_protocol_does_not_continue_length(tmp_path):
    result = exercise_stops(tmp_path, [assistant_stop("length")], enabled=False)
    assert result["followUps"] == []


@pytest.mark.parametrize("value", [None, "true", 1])
def test_constructor_rejects_non_boolean_continuation(value):
    with pytest.raises(ValueError, match="continue_on_length must be a boolean"):
        PiAgent(continue_on_length=value)


def test_continuation_keyword_is_consumed_before_harbor_options(monkeypatch):
    from harbor.agents.installed.pi import Pi
    from types import SimpleNamespace

    metadata = {
        "input": ["text", "image"], "contextWindow": 8192,
        "maxTokens": 4096, "reasoning": True, "cost_status": "unknown",
    }
    seen = []
    monkeypatch.setattr(Pi, "__init__", lambda self, *args, **kwargs: seen.append(kwargs))
    monkeypatch.setattr(PiAgent, "_check_reasoning", lambda self: None)
    monkeypatch.setattr(Pi, "preflight", classmethod(lambda cls, options, env: seen.append(options)))
    monkeypatch.setattr(PiAgent, "parse_options", classmethod(
        lambda cls, options, env: SimpleNamespace(thinking="off")
    ))
    agent = PiAgent(model_metadata=metadata, continue_on_length=True)
    assert "const continueOnLength = true" in agent._build_max_turns_extension(64)
    PiAgent.preflight({"model_metadata": metadata, "continue_on_length": True})
    assert len(seen) == 2
    assert all("continue_on_length" not in options for options in seen)
