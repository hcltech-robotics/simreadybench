import base64
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from pi_nvidia import runtime_inputs


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


def image_config(tmp_path):
    image = tmp_path / "observation.png"
    image.write_bytes(PNG)
    return {"observation_path": str(image), "expected_image_sha256": hashlib.sha256(PNG).hexdigest()}


def test_frozen_image_attached_exactly_once_and_original_blocks_preserved(tmp_path):
    config = image_config(tmp_path)
    original = [{"type": "text", "text": "Original frozen instruction. α"}]
    blocks, receipt = runtime_inputs.attach_observation(original, config)
    assert blocks[0] == original[0] and len(original) == 1
    image = blocks[-1]
    assert image["type"] == "image" and image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"]) == PNG
    assert receipt["source_sha256"] == receipt["payload_sha256"] == config["expected_image_sha256"]
    same, already = runtime_inputs.attach_observation(blocks, config)
    assert same == blocks and already["delivery"] == "already_supplied"
    assert "data" not in receipt and "data" not in already


def test_wrong_hash_missing_image_and_symlink_fail_before_attachment(tmp_path):
    config = image_config(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        runtime_inputs.attach_observation([], {**config, "expected_image_sha256": "0" * 64})
    link = tmp_path / "link.png"
    link.symlink_to(config["observation_path"])
    with pytest.raises(ValueError, match="regular"):
        runtime_inputs.attach_observation([], {**config, "observation_path": str(link)})
    with pytest.raises(ValueError, match="regular"):
        runtime_inputs.attach_observation([], {**config, "observation_path": str(tmp_path / "missing.png")})


def test_budget_has_exactly_64_turns_and_no_authoring_clock(tmp_path):
    path = runtime_inputs.prepare_runtime(tmp_path, {})
    data = json.loads(path.read_text())
    assert data["model_turn_limit"] == 64 and data["final_validation_turns"] == 4
    assert data["first_candidate_target_turn"] == 8
    assert not any("deadline" in key or "sec" in key or "monotonic" in key for key in data)
    # Old deployments must not silently retain the obsolete authoring policy.
    with pytest.raises(ValueError, match="obsolete"):
        runtime_inputs.prepare_runtime(tmp_path, {"PI_NVIDIA_TIME_LIMIT_SEC": "600"})


def test_actual_node_notice_counts_current_turn_and_final_validation_reserve(tmp_path):
    module = Path(__file__).parents[1] / "pi_nvidia/runtime.mjs"
    script = tmp_path / "notice.mjs"
    script.write_text(f'''
import {{makeNotice}} from {json.dumps(module.as_uri())};
const config={{model_turn_limit:64,final_validation_turns:4,first_candidate_target_turn:8}};
console.log(JSON.stringify([1,9,61,64].map(n => makeNotice(config,n,n>9))));
''')
    result = subprocess.run(["/workspace/pi-nvidia-runtime/node/bin/node", str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    first, overdue, final, last = json.loads(result.stdout)
    assert first["turn"] == 1 and first["remaining_including_current"] == 64
    assert first["remaining_after_current"] == 63
    assert "turn 1 of 64" in first["message"] and "64 turns remain including this turn" in first["message"]
    assert overdue["candidate_target_overdue"] is True
    assert final["remaining_including_current"] == 4 and final["phase"] == "final_revalidation"
    assert last["remaining_after_current"] == 0 and last["phase"] == "final_revalidation"
    assert "five" in last["message"] and "last turn" in last["message"]
    assert "seconds" not in last["message"] and "deadline" not in last["message"]


def test_turn_end_finishes_tool_batch_and_aborts_only_at_64_and_guards_65(tmp_path):
    module = Path(__file__).parents[1] / "pi_nvidia/runtime.mjs"
    logs = tmp_path / "logs"
    config = logs / "pi/config/runtime.json"
    runtime_inputs.prepare_runtime(config.parent, {})
    script = tmp_path / "turns.mjs"
    script.write_text(f'''
import runtime from {json.dumps(module.as_uri())};
import {{readFileSync}} from "node:fs";
process.env.PI_NVIDIA_RUNTIME_FILE={json.dumps(str(config))};
const hooks={{}};runtime({{on:(name,fn)=>hooks[name]=fn,registerTool:()=>{{}},exec:async()=>({{code:0,stdout:"{{}}"}})}});
let aborts=0;const ctx={{abort:()=>aborts++,cwd:"/task"}};
if (!hooks.tool_call({{toolName:"write",input:{{path:"/output/submission/asset.usd"}}}},ctx)?.block)
 throw new Error("Direct final writes must be blocked");
if (hooks.tool_call({{toolName:"write",input:{{path:"/tmp/author.py"}}}},ctx))
 throw new Error("Separate authoring must remain available");
for(let i=1;i<=64;i++){{
 hooks.turn_start({{turnIndex:i-1}},ctx);
 hooks.context({{messages:[]}},ctx);
 await hooks.turn_end({{message:{{stopReason:"toolUse"}},toolResults:[{{role:"toolResult"}}]}},ctx);
 if(aborts !== (i===64?1:0)) throw new Error("Wrong boundary at "+i);
}}
const receipt=JSON.parse(readFileSync({json.dumps(str(logs / 'pi-turn-budget.json'))},"utf8"));
// A subsequent ACP prompt must not reset the same-trial budget.
hooks.turn_start({{turnIndex:0}},ctx);
hooks.before_provider_request({{payload:{{messages:[]}}}},ctx);
console.log(JSON.stringify({{aborts,receipt}}));
''')
    result = subprocess.run(["/workspace/pi-nvidia-runtime/node/bin/node", str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["aborts"] >= 2
    assert data["receipt"]["started"] == data["receipt"]["completed"] == 64
    assert data["receipt"]["exhausted"] is True
    notices = [json.loads(line) for line in (logs / "pi-budget-notices.jsonl").read_text().splitlines()]
    assert [row["remaining_including_current"] for row in notices] == list(range(64,0,-1))


def test_bridge_injects_the_frozen_image_once_without_resetting_turn_budget(tmp_path):
    from pi_nvidia.acp import Bridge
    from pi_nvidia.profile import ModelProfile
    from test_agent import profile
    config = image_config(tmp_path)
    runtime = runtime_inputs.prepare_runtime(tmp_path / "config", {
        "PI_NVIDIA_OBSERVATION_PATH": config["observation_path"],
        "PI_NVIDIA_OBSERVATION_SHA256": config["expected_image_sha256"]})
    budget = json.loads(runtime.read_text())["model_turn_limit"]
    bridge = Bridge(ModelProfile.model_validate(profile()), "off", tmp_path / "pi/sessions", tmp_path,
        runtime_file=runtime)
    bridge.ready.add("s")
    def request(i):
        return bridge.request({"id": i, "method": "session/prompt", "params": {
            "sessionId": "s", "prompt": [{"type": "text", "text": "Frozen task"}]}})
    first = request(1)
    assert "build_candidate" in first["params"]["prompt"][0]["text"]
    assert "Valid empty stages are allowed" in first["params"]["prompt"][0]["text"]
    second = request(2)
    assert len([b for b in first["params"]["prompt"] if b["type"] == "image"]) == 1
    assert not any(b["type"] == "image" for b in second["params"]["prompt"])
    assert json.loads(runtime.read_text())["model_turn_limit"] == budget == 64
    receipt = json.loads((tmp_path / "pi-input-image.json").read_text())
    assert receipt["source_sha256"] == receipt["payload_sha256"] == config["expected_image_sha256"]
