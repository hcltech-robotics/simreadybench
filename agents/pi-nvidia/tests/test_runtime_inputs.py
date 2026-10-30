import base64
import hashlib
import json
import time
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


def test_budget_starts_before_runtime_install_and_is_labeled_conservatively(tmp_path):
    started = time.monotonic_ns() - 20_000_000_000
    path = runtime_inputs.prepare_runtime(tmp_path, {}, started_ns=started)
    data = json.loads(path.read_text())
    assert data["configured_limit_sec"] == 600 and data["final_reserve_sec"] == 90
    assert int(data["started_monotonic_ns"]) == started
    assert int(data["working_deadline_monotonic_ns"]) - started == 585_000_000_000
    assert data["deadline_basis"] == "conservative_adapter_clock"
    assert data["outer_deadline_known"] is False
    with pytest.raises(ValueError):
        runtime_inputs.prepare_runtime(tmp_path, {"PI_NVIDIA_TIME_LIMIT_SEC": "601"}, started_ns=started)


def test_actual_node_budget_notice_moves_to_final_validation_without_extending_time(tmp_path):
    module = Path(__file__).parents[1] / "pi_nvidia/runtime.mjs"
    script = tmp_path / "notice.mjs"
    script.write_text(f'''
import {{makeNotice}} from {json.dumps(module.as_uri())};
const config={{configured_limit_sec:600,final_reserve_sec:90,first_candidate_target_sec:120,
 started_monotonic_ns:"1000000000000",working_deadline_monotonic_ns:"1585000000000",
 deadline_basis:"conservative_adapter_clock",working_deadline_utc:"fixture"}};
console.log(JSON.stringify([
 makeNotice(config,1010000000000n,false),
 makeNotice(config,1130000000000n,false),
 makeNotice(config,1510000000000n,true),
 makeNotice(config,1586000000000n,true)]));
''')
    result = subprocess.run(["/workspace/pi-nvidia-runtime/node/bin/node", str(script)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    first, overdue, final, expired = json.loads(result.stdout)
    assert first["remaining_sec"] == 575 and first["phase"] == "build_first_candidate"
    assert overdue["candidate_target_overdue"] is True
    assert final["remaining_sec"] == 75 and final["phase"] == "final_revalidation"
    assert expired["remaining_sec"] == 0 and expired["phase"] == "finish_now"
    assert "conservative" in final["message"] and "five" in final["message"]


def test_bridge_injects_the_frozen_image_once_without_resetting_working_deadline(tmp_path):
    from pi_nvidia.acp import Bridge
    from pi_nvidia.profile import ModelProfile
    from test_agent import profile
    config = image_config(tmp_path)
    runtime = runtime_inputs.prepare_runtime(tmp_path / "config", {
        "PI_NVIDIA_OBSERVATION_PATH": config["observation_path"],
        "PI_NVIDIA_OBSERVATION_SHA256": config["expected_image_sha256"]})
    deadline = json.loads(runtime.read_text())["working_deadline_monotonic_ns"]
    bridge = Bridge(ModelProfile.model_validate(profile()), "off", tmp_path / "pi/sessions", tmp_path,
        runtime_file=runtime)
    bridge.ready.add("s")
    def request(i):
        return bridge.request({"id": i, "method": "session/prompt", "params": {
            "sessionId": "s", "prompt": [{"type": "text", "text": "Frozen task"}]}})
    first = request(1)
    second = request(2)
    assert len([b for b in first["params"]["prompt"] if b["type"] == "image"]) == 1
    assert not any(b["type"] == "image" for b in second["params"]["prompt"])
    assert json.loads(runtime.read_text())["working_deadline_monotonic_ns"] == deadline
    receipt = json.loads((tmp_path / "pi-input-image.json").read_text())
    assert receipt["source_sha256"] == receipt["payload_sha256"] == config["expected_image_sha256"]
