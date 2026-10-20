"""Complete episode-scoped NVIDIA report files with bounded, read-only pages."""

from pathlib import Path
import hashlib, json, os, re, stat, tempfile, threading

AGENT_REPORT_ROOT = "/tmp/simready-validation-reports"
TOOL = {
    "name": "get_validation_report",
    "description": "Read a requested range of a prior NVIDIA report. The complete report is already saved at the file_path returned by validation. Prefer rg and targeted local file reads. Read-only and free of validation quota.",
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


def write_report_file(path, report):
    """Stream canonical ASCII JSON without a total-report size or finding cap."""
    digest = hashlib.sha256()
    size = 0
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb", buffering=65536) as stream:
        for text in json.JSONEncoder(
            sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True
        ).iterencode(report):
            raw = text.encode("ascii")
            stream.write(raw)
            digest.update(raw)
            size += len(raw)
        stream.write(b"\n")
        digest.update(b"\n")
        size += 1
    return {"sha256": digest.hexdigest(), "bytes": size, "total_characters": size}


def _fingerprint(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
        value.st_nlink,
    )


class ReportStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        if self.root.resolve() != self.root:
            raise ValueError("Private nonsymlink report store required")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.bindings = {}
        self.lock = threading.RLock()

    def register(self, episode, report):
        if not isinstance(report, dict):
            return {
                "available": False,
                "complete": False,
                "reason": "validator_report_unavailable",
            }
        with (
            self.lock,
            tempfile.TemporaryDirectory(dir=self.root, prefix="staging-") as temporary,
        ):
            staged = Path(temporary) / "report.json"
            receipt = write_report_file(staged, report)
            sha = receipt["sha256"]
            path = self.root / (sha + ".json")
            if path.exists() or path.is_symlink():
                with self._open(path) as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
                        raise ValueError("Stored validator report changed")
            else:
                os.replace(staged, path)
            with self._open(path) as stream:
                self.bindings[(episode, sha)] = (
                    path,
                    _fingerprint(os.fstat(stream.fileno())),
                )
        counts = {
            k: len(report.get(k, [])) if isinstance(report.get(k), list) else 0
            for k in ("issues", "soft_issues", "warnings", "infos")
        }
        return {
            "available": True,
            "report_id": sha,
            **receipt,
            "issue_counts": counts,
            "complete_report_available": True,
            "delivery": "agent_file",
            "file_path": AGENT_REPORT_ROOT + "/" + sha + ".json",
            "report": None,
            "retrieval_tool": "get_validation_report",
            "offset_unit": "Unicode_code_points",
            "search_hint": "rg -n -m 20 -M 240 --max-columns-preview -F -e <rule_id> -e <candidate_prim> <file_path>; use sed -n <start>,<end>p for nearby lines",
            "redactions": "private host paths and credential patterns only; original diagnostic wording, suggestions and candidate USD sites retained",
        }

    def _open(self, path):
        try:
            if path.resolve() != path or path.parent != self.root:
                raise ValueError("Confined regular report required")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            stream = os.fdopen(descriptor, "rb")
            value = os.fstat(stream.fileno())
            if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
                stream.close()
                raise ValueError("Stored report unavailable")
            return stream
        except OSError as exc:
            raise ValueError("Stored report unavailable") from exc

    def read(self, episode, arguments):
        if not isinstance(arguments, dict) or set(arguments) - {
            "report_id",
            "offset",
            "limit",
        }:
            raise ValueError("Invalid report read arguments")
        sha = arguments.get("report_id")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 32768)
        if (
            not isinstance(sha, str)
            or not re.fullmatch("[a-f0-9]{64}", sha)
            or type(offset) is not int
            or offset < 0
            or type(limit) is not int
            or not 1 <= limit <= 65536
        ):
            raise ValueError("Bounded report page required")
        with self.lock:
            binding = self.bindings.get((episode, sha))
            if binding is None:
                raise ValueError("Report not bound to this episode")
            path, expected = binding
            with self._open(path) as stream:
                if _fingerprint(os.fstat(stream.fileno())) != expected:
                    raise ValueError("Stored report changed")
                size = expected[2]
                if offset > size:
                    raise ValueError("Report offset outside content")
                # Canonical JSON is ASCII, so byte seeks preserve Unicode-code-point offsets.
                stream.seek(offset)
                raw = stream.read(limit)
                if _fingerprint(os.fstat(stream.fileno())) != expected:
                    raise ValueError("Stored report changed")
        end = offset + len(raw)
        return {
            "status": "completed",
            "report_id": sha,
            "sha256": sha,
            "offset": offset,
            "next_offset": end if end < size else None,
            "total_characters": size,
            "total_bytes": size,
            "text": raw.decode("ascii"),
            "complete": end == size,
            "report_truncated": False,
            "validation_calls_consumed": 0,
        }
