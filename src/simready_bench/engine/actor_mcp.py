"""Actor-only bounded MCP stdio proxy. Standard library; public schemas only."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, build_opener, ProxyHandler

MAX_REQUEST = 65536
MAX_RESPONSE = 64 * 1024 * 1024
BASE_URL = "http://127.0.0.1:32190"
REPORT_ROOT = Path("/tmp/simready-validation-reports")
TOOLS = [
    {
        "name": "get_task_thumbnail",
        "description": "Read the supplied original task thumbnail pixels and public brief.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "validate_candidate",
        "description": "Validate the frozen candidate against NVIDIA Robotics-Prop 4.0.0 and return its score plus the complete report file path. Each new request uses one validation call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "minLength": 1, "maxLength": 256}
            },
            "required": ["request_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_asset",
        "description": "End authoring, stop actor writes and freeze the current candidate. Returns only submission acknowledgement and hash.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "minLength": 1, "maxLength": 256}
            },
            "required": ["request_id"],
            "additionalProperties": False,
        },
    },
]
_REPORT_TOOL = {
    "name": "get_validation_report",
    "description": "Read an explicitly requested range of a prior NVIDIA report. The complete file is saved at file_path. Prefer rg and targeted local reads. Read-only and free of validation quota.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "report_id": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 65536,
                "default": 32768,
            },
        },
        "required": ["report_id"],
        "additionalProperties": False,
    },
}
OPTIONAL_TOOLS = [_REPORT_TOOL]


def error_result(status):
    return {
        "content": [{"type": "text", "text": json.dumps({"status": status})}],
        "structuredContent": {"status": status},
        "isError": True,
    }


def materialize_validation_report(result, token, base_url):
    """Save private-transport pages locally without sending their text to the model."""
    descriptor = (
        result.get("structuredContent", {})
        .get("feedback", {})
        .get("nvidia_validator_report")
    )
    if not isinstance(descriptor, dict) or not descriptor.get("available"):
        return result
    sha = descriptor.get("report_id")
    size = descriptor.get("bytes")
    if (
        not isinstance(sha, str)
        or not re.fullmatch("[a-f0-9]{64}", sha)
        or descriptor.get("sha256") != sha
        or (type(size) is not int)
        or (size < 1)
        or (
            descriptor.get("file_path")
            != "/tmp/simready-validation-reports/" + sha + ".json"
        )
    ):
        raise ValueError("Invalid complete report descriptor")
    root = REPORT_ROOT.absolute()
    if root.resolve() != root:
        raise ValueError("Nonsymlink report directory required")
    root.mkdir(mode=448, exist_ok=True)
    info = root.stat()
    if not root.is_dir() or info.st_uid != os.getuid() or info.st_mode & 63:
        raise ValueError("Owned private report directory required")
    path = root / (sha + ".json")
    if path.exists() or path.is_symlink():
        descriptor_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor_fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != size
                or (hashlib.file_digest(stream, "sha256").hexdigest() != sha)
            ):
                raise ValueError("Local report changed")
    else:
        with tempfile.TemporaryDirectory(dir=root, prefix="staging-") as temporary:
            staged = Path(temporary) / "report.json"
            digest = hashlib.sha256()
            offset = 0
            with staged.open("xb") as stream:
                while offset < size:
                    arguments = {"report_id": sha, "offset": offset, "limit": 65536}
                    request = Request(
                        base_url + "/v1/tools/get_validation_report",
                        data=json.dumps(arguments).encode(),
                        headers={
                            "Authorization": "Bearer " + token,
                            "Content-Type": "application/json",
                        },
                    )
                    with build_opener(ProxyHandler({})).open(
                        request, timeout=125
                    ) as reply:
                        raw = reply.read(1024 * 1024 + 1)
                    if len(raw) > 1024 * 1024:
                        raise ValueError("Report page transport bound")
                    page = json.loads(raw).get("structuredContent", {})
                    text = page.get("text")
                    if (
                        page.get("report_id") != sha
                        or page.get("sha256") != sha
                        or page.get("offset") != offset
                        or (page.get("total_bytes") != size)
                        or (page.get("total_characters") != size)
                        or (page.get("validation_calls_consumed") != 0)
                        or (page.get("report_truncated") is not False)
                        or (not isinstance(text, str))
                        or (not 1 <= len(text) <= 65536)
                    ):
                        raise ValueError("Invalid complete report page")
                    data = text.encode("ascii")
                    end = offset + len(data)
                    if (
                        end > size
                        or page.get("next_offset") != (end if end < size else None)
                        or page.get("complete") != (end == size)
                    ):
                        raise ValueError("Incomplete report page sequence")
                    stream.write(data)
                    digest.update(data)
                    offset = end
            if digest.hexdigest() != sha:
                raise ValueError("Complete report hash mismatch")
            staged.chmod(292)
            os.replace(staged, path)
    descriptor.update(
        file_path=str(path), file_available=True, validation_calls_consumed=0
    )
    result["content"] = [
        {
            "type": "text",
            "text": json.dumps(
                result["structuredContent"], separators=(",", ":"), allow_nan=False
            ),
        }
    ]
    return result


def dispatch(message, token, base_url=BASE_URL):
    identifier = message.get("id")
    method = message.get("method")
    if "id" not in message:
        return None
    response = {"jsonrpc": "2.0", "id": identifier}
    if method == "initialize":
        protocol = message.get("params", {}).get("protocolVersion")
        if protocol not in ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"):
            protocol = "2025-11-25"
        result = {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "simready-bench", "version": "0.2.0"},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        request = Request(
            base_url + "/v1/tools/tool_definitions",
            data=b"{}",
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
        )
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=10) as reply:
                raw = reply.read(MAX_REQUEST + 1)
            if len(raw) > MAX_REQUEST:
                raise ValueError("Public tool schema exceeds transport bound")
            payload = json.loads(raw)
            definitions = payload.get("structuredContent", {}).get("tools")
            if (
                not isinstance(definitions, list)
                or len(definitions) not in (len(TOOLS), len(TOOLS) + 1)
                or {t.get("name") for t in definitions if isinstance(t, dict)}
                not in (
                    {t["name"] for t in TOOLS},
                    {t["name"] for t in TOOLS + OPTIONAL_TOOLS},
                )
            ):
                raise ValueError("Trusted public tool roster unavailable")
            result = {"tools": definitions}
        except (HTTPError, OSError, ValueError, TypeError, AttributeError):
            response["error"] = {
                "code": -32603,
                "message": "Task tool schemas unavailable",
            }
            return response
    elif method == "tools/call":
        parameters = message.get("params", {})
        name = parameters.get("name")
        arguments = parameters.get("arguments", {})
        if name not in {
            tool["name"] for tool in TOOLS + OPTIONAL_TOOLS
        } or not isinstance(arguments, dict):
            result = error_result("invalid_arguments")
        else:
            data = json.dumps(arguments, allow_nan=False).encode()
            if len(data) > MAX_REQUEST:
                result = error_result("invalid_arguments")
            else:
                request = Request(
                    base_url + "/v1/tools/" + name,
                    data=data,
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                    },
                )
                try:
                    with build_opener(ProxyHandler({})).open(
                        request, timeout=125
                    ) as reply:
                        raw = reply.read(MAX_RESPONSE + 1)
                    if len(raw) > MAX_RESPONSE:
                        raise ValueError("Response exceeds public transport bound")
                    result = json.loads(raw)
                    if not isinstance(result, dict) or not isinstance(
                        result.get("content"), list
                    ):
                        raise ValueError("Invalid tool response")
                    if name == "validate_candidate":
                        result = materialize_validation_report(result, token, base_url)
                except HTTPError:
                    result = error_result("unauthorized")
                except (OSError, ValueError):
                    result = error_result("infrastructure_error")
    else:
        response["error"] = {"code": -32601, "message": "Method not found"}
        return response
    response["result"] = result
    return response


def main():
    parser = argparse.ArgumentParser(description="Private episode MCP stdio proxy")
    parser.parse_args()
    token = os.environ.get("SIMREADY_EPISODE_TOKEN", "")
    if not 16 <= len(token) <= 512:
        raise SystemExit("An episode-scoped token is required")
    while True:
        line = sys.stdin.buffer.readline(MAX_REQUEST + 1)
        if not line:
            return
        if len(line) > MAX_REQUEST:
            raise SystemExit("MCP request exceeds public transport bound")
        try:
            message = json.loads(line)
            if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                raise ValueError("Invalid JSON-RPC envelope")
            response = dispatch(message, token)
        except (ValueError, TypeError, AttributeError):
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Invalid request"},
            }
        if response is not None:
            sys.stdout.write(
                json.dumps(response, separators=(",", ":"), allow_nan=False) + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
