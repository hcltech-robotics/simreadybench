"""Real Linux subprocess faults at the declared Pi authoring tool boundary."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from simready_bench.engine.tool_protocol import TOOL_LIMITS, stage_tool_protocol, tool_sources

pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("node") is None, reason="Linux Node tool process boundary")


@pytest.fixture
def tool_runtime(tmp_path):
    # Standard regression needs only the exported tool interface. The pinned Pi
    # package can replace it unchanged for the installed-runtime acceptance run.
    index = os.environ.get("SIMREADY_PI_TEST_INDEX")
    if not index:
        stock = tmp_path / "stock.mjs"
        stock.write_text('''import { readFile } from "node:fs/promises";
export function createBashToolDefinition(cwd, options) { return { name: "bash", parameters: {},
  async execute(_id, args) { const data = []; const result = await options.operations.exec(args.command, cwd,
    { onData: chunk => data.push(chunk), env: process.env });
    return { content: [{ type: "text", text: Buffer.concat(data).toString() }], details: result }; } }; }
export function createReadToolDefinition(cwd) { return { name: "read", parameters: {},
  async execute(_id, args) { return { content: [{ type: "text", text: await readFile(args.path, "utf8") }] }; } }; }
export const createWriteToolDefinition = createReadToolDefinition;
export const createEditToolDefinition = createReadToolDefinition;
export const createFindToolDefinition = createReadToolDefinition;
export const createGrepToolDefinition = createReadToolDefinition;
export const createLsToolDefinition = createReadToolDefinition;
''')
        index = str(stock)
    for name, source in tool_sources(pi_index=index).items():
        (tmp_path / name).write_text(source)
    return tmp_path


def probe(runtime, name, args, **limits):
    script = runtime / "probe.mjs"
    request = {"name": name, "args": args, "cwd": str(runtime), "callId": "test"}
    script.write_text('import { runTool } from "./extension.mjs";\n' +
        "const result = await runTool(" + json.dumps(request) + ", " + json.dumps(limits) + ");\n" +
        "console.log(JSON.stringify(result));\n")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_bash_wall_timeout_returns_actual_partial_output_and_kills_background_child(tool_runtime):
    result = probe(tool_runtime, "bash", {"command": "echo before-timeout; sleep 60 & echo $! > child.pid; wait"}, wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 1)
    assert result["details"]["termination"] == "timeout"
    assert "before-timeout" in result["content"][0]["text"]
    pid = (tool_runtime / "child.pid").read_text().strip()
    stat = Path("/proc") / pid / "stat"
    assert not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z ")


def test_success_also_retires_background_children(tool_runtime):
    result = probe(tool_runtime, "bash", {"command": "sleep 60 >/dev/null 2>&1 & echo $! > child.pid; echo done"}, wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 5)
    assert result["details"]["termination"] == "completed"
    pid = (tool_runtime / "child.pid").read_text().strip()
    stat = Path("/proc") / pid / "stat"
    assert not stat.exists() or stat.read_text().split(") ", 1)[1].startswith("Z ")


def test_success_retires_a_descendant_which_starts_a_new_session(tool_runtime):
    result = probe(tool_runtime, "bash", {"command": "setsid sh -c 'echo $$ > detached.pid; sleep 60' >/dev/null 2>&1 & sleep 0.1; echo done"},
                   wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 5)
    assert result["details"]["termination"] == "completed"
    pid = (tool_runtime / "detached.pid").read_text().strip()
    assert not (Path("/proc") / pid).exists()


def test_output_flood_has_a_hard_byte_bound(tool_runtime):
    result = probe(tool_runtime, "bash", {"command": "yes flood"}, wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 5, streamBytes=4096)
    assert result["details"]["termination"] == "output_limit"
    assert len(result["content"][0]["text"].encode()) < 4200


def test_fifo_read_obeys_the_same_tool_deadline(tool_runtime):
    fifo = tool_runtime / "blocked"
    os.mkfifo(fifo)
    result = probe(tool_runtime, "read", {"path": str(fifo)}, wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 1)
    assert result["details"]["termination"] == "timeout"


def test_process_fanout_is_bounded(tool_runtime):
    result = probe(tool_runtime, "bash", {"command": "for i in $(seq 1 20); do sleep 60 & done; wait"}, wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 5, processLimit=8)
    assert result["details"]["termination"] == "process_limit"


def test_process_fanout_in_separate_sessions_is_bounded(tool_runtime):
    result = probe(tool_runtime, "bash", {"command": "for i in $(seq 1 20); do setsid sleep 60 >/dev/null 2>&1 & done; wait"},
                   wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 5, processLimit=8)
    assert result["details"]["termination"] == "process_limit"


def test_non_shell_tool_result_has_the_same_output_bound(tool_runtime):
    path = tool_runtime / "large.txt"
    path.write_text("one candidate diagnostic line\n" * 10000)
    result = probe(tool_runtime, "read", {"path": str(path)},
                   wallSeconds=30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 5, resultBytes=4096)
    assert result["details"]["termination"] == "output_limit"


def test_actor_cannot_extend_the_frozen_tool_wall_limit(tool_runtime):
    wall_limit = 30 if os.environ.get("SIMREADY_PI_TEST_INDEX") else 1
    result = probe(tool_runtime, "bash", {"command": "sleep 60", "timeout": 600}, wallSeconds=wall_limit)
    assert result["details"]["termination"] == "timeout"
    assert result["details"]["wall_limit_seconds"] == wall_limit


def test_staged_tool_protocol_is_read_only_and_declares_every_limit(tmp_path):
    mount = stage_tool_protocol(tmp_path / "contract")
    assert mount["read_only"] is True
    assert TOOL_LIMITS == {"wall_seconds": 120, "stream_bytes": 1048576, "result_bytes": 4194304, "processes": 64, "input_bytes": 8388608}
    for name in ("extension.mjs", "worker.mjs", "supervisor.py"):
        assert not (Path(mount["source"]) / name).stat().st_mode & 0o222
