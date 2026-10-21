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
