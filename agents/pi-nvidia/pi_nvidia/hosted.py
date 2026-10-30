"""Profile preparation shared by every hosted start, including baked-cache reuse."""

import json
import os
from pathlib import Path
import shlex
import sys

from .profile import ENDPOINT, ModelProfile


def configure(prefix, logs, environ):
    env = dict(environ)
    profile = ModelProfile.model_validate_json(env.get("PI_NVIDIA_PROFILE", "{}"))
    effort = env.get("PI_NVIDIA_THINKING")
    if not profile.supports(effort):
        raise ValueError("Set an explicitly supported PI_NVIDIA_THINKING; no clamping")
    if "image" not in profile.input:
        raise ValueError("This hosted image task requires verified image input")
    requested = env.get("HARBOR_ACP_REQUESTED_MODEL")
    if requested != "nvidia/" + profile.id:
        raise ValueError("Hosted model_name and PI_NVIDIA_PROFILE.id must agree")
    if env.get("HOSTED_INFERENCE_URL"):
        raise ValueError("Select direct NVIDIA credentials; this package has no verified gateway route")
    # Only future authorized runs consume the existing injected credential. No
    # value enters a file, command argument, diagnostic or model-profile receipt.
    if env.get("HOSTED_INFERENCE_TOKEN"):
        env["NVIDIA_API_KEY"] = env["HOSTED_INFERENCE_TOKEN"]
    logs, prefix = Path(logs).resolve(), Path(prefix).resolve()
    config = logs / "pi/config"
    config.mkdir(parents=True, exist_ok=True)
    models = {"providers": {"harbor-endpoint": {"baseUrl": ENDPOINT,
        "api": "openai-completions", "apiKey": "$NVIDIA_API_KEY", "authHeader": True,
        "models": [profile.pi_model()]}}}
    (config / "models.json").write_text(json.dumps(models, indent=2) + "\n")
    (config / "settings.json").write_text(json.dumps({"defaultProvider": "harbor-endpoint",
        "defaultModel": profile.id, "defaultThinkingLevel": effort, "quietStartup": True,
        "enableSkillCommands": False, "packages": [], "skills": [], "extensions": []}) + "\n")
    sessions = logs / "pi/sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    # pi-acp owns ~/.pi/pi-acp separately from Pi's agent directory. Isolate it
    # too, so this bridge never scans another agent's home or authentication.
    home = logs / "pi/home"
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o700)
    launcher = config / "pi-plain"
    command = [str(prefix / "node/bin/node"), str(prefix / "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"),
        "--provider", "harbor-endpoint", "--model", profile.id, "--thinking", effort,
        "--session-dir", str(sessions), "--no-skills", "--no-extensions", "--no-prompt-templates"]
    launcher.write_text("#!/bin/sh\nexec " + shlex.join(command) + ' "$@"\n')
    launcher.chmod(0o755)
    env.update(PI_CODING_AGENT_DIR=str(config), PI_ACP_PI_COMMAND=str(launcher),
        PI_CODING_AGENT_SESSION_DIR=str(sessions), PI_ACP_ENABLE_EMBEDDED_CONTEXT="true", HOME=str(home))
    # Harbor prepends the source project's .venv/bin to start this Python module.
    # It must not replace the task's Python (which owns simready_eval.feedback)
    # inside Pi's authoring shell. Keep the rest of the task's original path.
    agent_bins = {str(Path(sys.prefix) / "bin")} if sys.prefix != sys.base_prefix else set()
    paths = [p for p in env.get("PATH", os.defpath).split(os.pathsep) if p not in agent_bins]
    if env.get("VIRTUAL_ENV") == sys.prefix:
        env.pop("VIRTUAL_ENV")
    env["PATH"] = str(prefix / "node/bin") + os.pathsep + str(prefix / "node_modules/.bin") + os.pathsep + os.pathsep.join(paths)
    return env, profile, effort, sessions
