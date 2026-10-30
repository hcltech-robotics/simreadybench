#!/usr/bin/env python3
"""Offline Pi RPC fixture used behind the real registry pi-acp bridge."""
import json
import os
from pathlib import Path
import sys
import time

root = Path(os.environ["PI_CODING_AGENT_SESSION_DIR"])
path = root / "fixture.jsonl"
model = {"provider": "harbor-endpoint", "id": "fixture/vision", "name": "Fixture",
         "contextWindow": 32000, "maxTokens": 4000, "input": ["text", "image"]}
rows = [{"type": "session", "id": "fixture-session"},
        {"type": "model_change", "id": "model", "provider": "harbor-endpoint", "modelId": "fixture/vision"},
        {"type": "thinking_level_change", "id": "effort", "thinkingLevel": "off"}]
def emit(data):
    print(json.dumps(data), flush=True)
def save():
    path.write_text("".join(json.dumps(e) + "\n" for e in rows))
save()
for raw in sys.stdin:
    command = json.loads(raw)
    kind = command["type"]
    data = {}
    if kind == "get_state":
        data = {"sessionId": "fixture-session", "sessionFile": str(path), "model": model, "thinkingLevel": "off"}
    elif kind == "get_available_models":
        data = {"models": [model]}
    elif kind == "get_available_thinking_levels":
        data = {"levels": ["off"]}
    elif kind == "get_session_stats":
        data = {"contextUsage": {"tokens": 7000, "contextWindow": 32000}}
    elif kind == "get_commands":
        data = {"commands": []}
    elif kind == "prompt":
        Path(os.environ["FIXTURE_CAPTURE"]).write_text(json.dumps(command))
        data = {"disposition": "started"}
    emit({"type": "response", "id": command.get("id"), "command": kind, "success": True, "data": data})
    if kind == "abort":
        emit({"type": "agent_settled"})
    if kind == "prompt":
        emit({"type": "agent_start"})
        emit({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "Fixture started"}})
        if os.environ.get("FIXTURE_WAIT_FOR_ABORT"):
            continue
        for n in (1, 2):
            if n == 2:
                # Early agent_end is not settled: retry/continuation still runs.
                emit({"type": "agent_end", "messages": []})
                time.sleep(0.1)
            message = {"role": "assistant", "content": [{"type": "text", "text": f"Fixture call {n}"}],
                "model": model["id"], "provider": model["provider"], "stopReason": "stop",
                "usage": {"input": 100, "cacheRead": 80, "cacheWrite": 0, "output": 10, "totalTokens": 190,
                          "cost": {"total": 0}}}
            if os.environ.get("FIXTURE_PROVIDER_ERROR"):
                message["stopReason"] = "error"
            rows.append({"type": "message", "id": f"message-{len(rows)}", "message": message})
            save()
            emit({"type": "message_end", "message": message})
            call_id = f"feedback-{n}"
            emit({"type": "tool_execution_start", "toolCallId": call_id, "toolName": "bash",
                  "args": {"command": f"python -m simready_eval.feedback validate_candidate --package /output/submission --request-id candidate-{n}"}})
            emit({"type": "tool_execution_end", "toolCallId": call_id, "toolName": "bash", "isError": False,
                  "result": {"content": [{"type": "text", "text": json.dumps({"synthetic_feedback_fixture": True, "passing": n == 2})}]}})
        emit({"type": "agent_end", "messages": []})
        emit({"type": "agent_settled"})
