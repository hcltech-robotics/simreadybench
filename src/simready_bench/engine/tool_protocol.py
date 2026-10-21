"""Declared Pi tool subprocess limits for newly frozen authoring conditions."""

from pathlib import Path

TOOL_PROTOCOLS = frozenset({"legacy", "bounded-v1"})
TOOL_DIRECTORY = "/opt/simready-tool-protocol"
TOOL_EXTENSION_PATH = TOOL_DIRECTORY + "/extension.mjs"
TOOL_LIMITS = {
    "wall_seconds": 120,
    "stream_bytes": 1024 * 1024,
    "result_bytes": 4 * 1024 * 1024,
    "processes": 64,
    "input_bytes": 8 * 1024 * 1024,
}


def validate_tool_protocol(value):
    if value not in TOOL_PROTOCOLS:
        raise ValueError("Unknown authoring tool protocol")
    return value


# Each stock tool executes in its own Linux process group. The parent retains
# the actual partial output and kills the group on every terminal path, including
# successful completion, so commands cannot leave ordinary background children.
EXTENSION_SOURCE = r'''import { spawn } from "node:child_process";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as stock from "__PI_INDEX__";

const limits = __LIMITS__;
const factories = {
  bash: "createBashToolDefinition", read: "createReadToolDefinition",
  write: "createWriteToolDefinition", edit: "createEditToolDefinition",
  find: "createFindToolDefinition", grep: "createGrepToolDefinition", ls: "createLsToolDefinition",
};

function groupProcesses(group) {
  let count = 0;
  for (const pid of readdirSync("/proc")) {
    if (!/^[0-9]+$/.test(pid)) continue;
    try {
      const text = readFileSync(`/proc/${pid}/stat`, "utf8");
      const fields = text.slice(text.lastIndexOf(")") + 2).split(" ");
      if (Number(fields[2]) === group && fields[0] !== "Z") count += 1;
    } catch (error) {
      if (error.code !== "ENOENT" && error.code !== "ESRCH" && error.code !== "EACCES") throw error;
    }
  }
  return count;
}

export function runTool(request, { signal, wallSeconds = limits.wall_seconds,
  streamBytes = limits.stream_bytes, resultBytes = limits.result_bytes,
  processLimit = limits.processes } = {}) {
  return new Promise((resolve, reject) => {
    if (!factories[request.name]) { reject(new Error("Unknown bounded tool")); return; }
    const input = JSON.stringify(request);
    if (Buffer.byteLength(input) > limits.input_bytes) { reject(new Error("Tool input limit exceeded")); return; }
    const requested = request.name === "bash" ? request.args.timeout : undefined;
    if (requested !== undefined && (!Number.isFinite(requested) || requested <= 0)) {
      reject(new Error("Positive finite bash timeout required")); return;
    }
    const duration = Math.min(wallSeconds, requested ?? wallSeconds);
    const started = Date.now();
    const child = spawn("python3", [fileURLToPath(new URL("supervisor.py", import.meta.url)),
      String(processLimit), process.execPath, fileURLToPath(new URL("worker.mjs", import.meta.url))], {
      cwd: request.cwd, detached: true, stdio: ["pipe", "pipe", "pipe"], env: process.env,
    });
    let partial = [], partialSize = 0, wire = "", stderrSize = 0, wireSize = 0;
    let finalResult, termination, spawnError, hardKillTimer, toolError = false, settled = false;
    const killGroup = () => {
      if (!child.pid || child.exitCode !== null || child.signalCode !== null) return;
      // The Linux subreaper first retires descendants which started a separate
      // session, then exits. The group kill is a bounded last cleanup step.
      child.kill("SIGTERM");
      hardKillTimer ??= setTimeout(() => { try { process.kill(-child.pid, "SIGKILL"); }
        catch (error) { if (error.code !== "ESRCH") spawnError ??= error; } }, 2500);
    };
    const stop = (reason) => { termination ??= reason; killGroup(); };
    const aborted = () => stop("cancelled");
    const timer = setTimeout(() => stop("timeout"), duration * 1000);
    const monitor = setInterval(() => {
      try { if (child.pid && groupProcesses(child.pid) > processLimit) stop("process_limit"); }
      catch (error) { spawnError = error; stop("process_monitor_failed"); }
    }, 50);
    signal?.addEventListener("abort", aborted, { once: true });
    if (signal?.aborted) aborted();
    child.stdout.on("data", (chunk) => {
      wireSize += chunk.length;
      if (wireSize > resultBytes + 2 * streamBytes) { stop("output_limit"); return; }
      wire += chunk.toString("utf8");
      let boundary;
      while ((boundary = wire.indexOf("\n")) >= 0) {
        const line = wire.slice(0, boundary); wire = wire.slice(boundary + 1);
        let event;
        try { event = JSON.parse(line); } catch { stop("invalid_tool_result"); return; }
        if (event.type === "output") {
          const output = Buffer.from(event.data, "base64");
          const available = Math.max(0, streamBytes - partialSize);
          partial.push(output.subarray(0, available)); partialSize += output.length;
          if (partialSize > streamBytes) stop("output_limit");
        } else if (event.type === "termination" && event.reason === "process_limit") {
          stop("process_limit");
        } else if (event.type === "result") {
          if (Buffer.byteLength(JSON.stringify(event.result)) > resultBytes) stop("output_limit");
          else finalResult = event.result;
        } else if (event.type === "error") {
          toolError = true;
          finalResult = { content: [{ type: "text", text: String(event.message).slice(0, 4096) }], isError: true };
        } else stop("invalid_tool_result");
      }
    });
    child.stderr.on("data", (chunk) => {
      stderrSize += chunk.length;
      if (stderrSize > streamBytes) stop("output_limit");
    });
    child.once("error", (error) => { spawnError = error; });
    child.once("close", (code) => {
      if (settled) return;
      settled = true; clearTimeout(timer); clearTimeout(hardKillTimer); clearInterval(monitor);
      signal?.removeEventListener("abort", aborted);
      if (spawnError) { reject(spawnError); return; }
      if (!termination && (!finalResult || code !== 0 || wire.trim())) termination = "tool_process_failed";
      const details = { protocol: "bounded-v1", termination: termination ?? (toolError ? "tool_error" : "completed"),
        wall_limit_seconds: duration, stream_limit_bytes: streamBytes,
        result_limit_bytes: resultBytes, process_limit: processLimit,
        elapsed_seconds: (Date.now() - started) / 1000 };
      if (termination) {
        resolve({ content: [{ type: "text", text: Buffer.concat(partial).toString("utf8") +
          `\nTool stopped: ${termination}.` }], details, isError: true });
      } else resolve({ ...finalResult, details: { ...finalResult.details, ...details } });
    });
    child.stdin.on("error", () => {});
    child.stdin.end(input);
  });
}

export default function (pi) {
  for (const [name, factory] of Object.entries(factories)) {
    const definition = stock[factory](process.cwd());
    pi.registerTool({ ...definition,
      description: definition.description + ` Each invocation has a ${limits.wall_seconds}-second wall limit, a ${limits.stream_bytes}-byte shell output limit, a ${limits.result_bytes}-byte result limit and a ${limits.processes}-process limit. Its descendants are stopped when the tool returns.`,
      async execute(callId, args, signal, _onUpdate, ctx) {
        return runTool({ name, callId, args, cwd: ctx.cwd, model: ctx.model,
          sessionId: ctx.sessionManager.getSessionId(), sessionFile: ctx.sessionManager.getSessionFile(),
          thinkingLevel: ctx.thinkingLevel }, { signal });
      },
    });
  }
}
'''

SUPERVISOR_SOURCE = r'''"""Linux subreaper for exactly one bounded authoring tool."""
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    raise OSError(ctypes.get_errno(), "Tool subreaper setup failed")
stopping = False


def requested(_signal, _frame):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM, requested)
signal.signal(signal.SIGINT, requested)
owner = os.getpid()
maximum = int(sys.argv[1])
process = subprocess.Popen(sys.argv[2:])


def descendants():
    records = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            text = (path / "stat").read_text()
            fields = text[text.rfind(")") + 2:].split()
            records[int(path.name)] = (int(fields[1]), fields[19], fields[0])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    owned = {owner}
    while True:
        found = {pid for pid, row in records.items() if row[0] in owned}
        if found <= owned:
            break
        owned |= found
    return {pid: records[pid] for pid in owned - {owner} if pid in records}


try:
    while process.poll() is None and not stopping:
        current = descendants()
        if 1 + sum(row[2] != "Z" for row in current.values()) > maximum:
            print(json.dumps({"type": "termination", "reason": "process_limit"}), flush=True)
            stopping = True
            break
        time.sleep(0.02)
finally:
    # Subreaping keeps daemonised/detached grandchildren owned by this process.
    # Re-scan after kills to include children reparented during the transition.
    end = time.monotonic() + 2
    while time.monotonic() < end:
        current = descendants()
        for pid, row in current.items():
            if row[2] == "Z":
                continue
            try:
                text = Path("/proc", str(pid), "stat").read_text()
                fields = text[text.rfind(")") + 2:].split()
                if fields[19] == row[1]:
                    os.kill(pid, signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError):
                pass
        process.poll()
        while True:
            try:
                waited, _ = os.waitpid(-1, os.WNOHANG)
                if not waited:
                    break
            except ChildProcessError:
                break
        if not descendants():
            break
        time.sleep(0.01)
sys.exit(process.returncode if process.returncode is not None and process.returncode >= 0 else 1)
'''

WORKER_SOURCE = r'''import { spawn } from "node:child_process";

const emit = (event) => process.stdout.write(JSON.stringify(event) + "\n");
const factories = {
  bash: "createBashToolDefinition", read: "createReadToolDefinition",
  write: "createWriteToolDefinition", edit: "createEditToolDefinition",
  find: "createFindToolDefinition", grep: "createGrepToolDefinition", ls: "createLsToolDefinition",
};
let input = "";
for await (const chunk of process.stdin) input += chunk;
try {
  const request = JSON.parse(input);
  const index = "__PI_INDEX__";
  const stock = await import(index.endsWith("/core/tools/index.js")
    ? index.replace(/index\.js$/, request.name + ".js") : index);
  const operations = { exec(command, cwd, { onData, env }) {
    return new Promise((resolve, reject) => {
      // Inherit the tool worker's process group so the parent owns the shell and
      // its descendants, including a command which closes its output pipes.
      const child = spawn("/bin/bash", ["-c", command], { cwd, env, detached: false,
        stdio: ["ignore", "pipe", "pipe"] });
      const data = (chunk) => { emit({ type: "output", data: chunk.toString("base64") }); onData(chunk); };
      child.stdout.on("data", data); child.stderr.on("data", data);
      child.once("error", reject); child.once("close", (code) => resolve({ exitCode: code }));
    });
  } };
  const definition = stock[factories[request.name]](request.cwd,
    request.name === "bash" ? { operations } : undefined);
  const result = await definition.execute(request.callId, request.args, undefined, undefined,
    { cwd: request.cwd, model: request.model, thinkingLevel: request.thinkingLevel,
      sessionManager: { getSessionId: () => request.sessionId, getSessionFile: () => request.sessionFile } });
  emit({ type: "result", result });
} catch (error) { emit({ type: "error", message: error.message }); }
'''


def tool_sources(*, pi_index="/opt/simready/pi/node_modules/@earendil-works/pi-coding-agent/dist/core/tools/index.js"):
    import json

    return {
        "extension.mjs": EXTENSION_SOURCE.replace("__PI_INDEX__", pi_index).replace("__LIMITS__", json.dumps(TOOL_LIMITS)),
        "worker.mjs": WORKER_SOURCE.replace("__PI_INDEX__", pi_index),
        "supervisor.py": SUPERVISOR_SOURCE,
    }


def stage_tool_protocol(directory):
    """Stage a read-only extension and worker outside the public task contract."""
    directory = Path(directory)
    directory.mkdir(mode=0o700)
    for name, source in tool_sources().items():
        path = directory / name
        path.write_text(source)
        path.chmod(0o444)
    directory.chmod(0o555)
    return {
        "type": "bind", "source": str(directory), "target": TOOL_DIRECTORY,
        "read_only": True, "bind": {"create_host_path": False},
    }
