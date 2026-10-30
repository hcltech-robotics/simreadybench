"""Real pinned Pi metadata parsing, with network and credential access excluded."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from test_agent import agent, connection, profile


@pytest.mark.skipif(not os.environ.get("PI_TEST_PREFIX"), reason="requires offline Pi 1.4.0 installation")
@pytest.mark.parametrize("thinking", ["off", "high"])
def test_actual_pi_loads_explicit_model_metadata_without_auth_or_network(tmp_path, thinking):
    prefix = Path(os.environ["PI_TEST_PREFIX"])
    package = prefix / "node_modules/@earendil-works/pi-coding-agent"
    assert json.loads((package / "package.json").read_text())["version"] == "1.4.0"
    data = profile()
    if thinking == "high":
        data["reasoning"] = True
        data["thinkingLevelMap"] = {k: None for k in ("off", "minimal", "low", "medium", "high", "xhigh", "max")}
        data["thinkingLevelMap"]["high"] = "large"
        data["compat"]["thinkingFormat"] = "openai"
        data["compat"]["supportsReasoningEffort"] = True
    a = agent(tmp_path / "logs", data, thinking=thinking)
    path = tmp_path / "models.json"
    path.write_text(json.dumps(a._build_custom_models_json(connection(), "fixture/vision")))
    script = tmp_path / "verify.mjs"
    script.write_text(f'''
globalThis.fetch = () => {{ throw new Error("Network forbidden in offline check"); }};
const {{ ModelRuntime }} = await import({json.dumps((package / "dist/index.js").as_uri())});
const {{ InMemoryCredentialStore }} = await import({json.dumps((prefix / "node_modules/@earendil-works/pi-ai/dist/auth/credential-store.js").as_uri())});
const {{ getSupportedThinkingLevels, clampThinkingLevel }} = await import({json.dumps((prefix / "node_modules/@earendil-works/pi-ai/dist/models.js").as_uri())});
const runtime = await ModelRuntime.create({{credentials: new InMemoryCredentialStore(),
  modelsPath: {json.dumps(str(path))}, allowModelNetwork: false, refreshOnCreate: false}});
const model = runtime.getModel("harbor-endpoint", "fixture/vision");
if (!model) throw new Error(runtime.getError() || "Model unavailable");
console.log(JSON.stringify({{input:model.input, contextWindow:model.contextWindow,
  maxTokens:model.maxTokens, reasoning:model.reasoning, compat:model.compat,
  thinking:getSupportedThinkingLevels(model), effective:clampThinkingLevel(model,{json.dumps(thinking)})}}));
''')
    result = subprocess.run(["node", str(script)], check=True, text=True,
                            capture_output=True, timeout=30)
    data = json.loads(result.stdout)
    assert data["input"] == ["text", "image"]
    assert (data["contextWindow"], data["maxTokens"]) == (32000, 4000)
    assert data["thinking"] == [thinking] and data["effective"] == thinking
    assert data["compat"]["maxTokensField"] == "max_tokens"
