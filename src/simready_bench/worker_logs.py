"""Read live Pi session activity without changing an executing benchmark."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import time


READ_LIMIT = 256 * 1024
RECORD_LIMIT = 1024 * 1024
TEXT_LIMIT = 300
_SECRET = re.compile(
    r'(?i)(\b(?:[a-z0-9]+[_-])*(?:authorization|api[_-]?key|access[_-]?token|token|password|secret(?:[_-]access[_-]key)?)'
    r'\b["\x27]?(?:\s*[=:]\s*|\s+))(?:"[^"]*"|\x27[^\x27]*\x27|[^\s,;]+)'
)
_BEARER = re.compile(r'(?i)\bBearer\s+[^\s"\x27,;]+')
_TOKEN = re.compile(r'\b(?:hf_|sk-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}\b')
_BLOB = re.compile(r'[A-Za-z0-9+/=_-]{100,}')
_CONTROL = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b-\x1f\x7f]')


def safe_text(value: object, limit: int = TEXT_LIMIT) -> str:
    """Bound visible text and remove credentials, blobs and terminal controls."""
    text = str(value)
    text = _BEARER.sub('Bearer [redacted]', text)
    text = _SECRET.sub(r'\1[redacted]', text)
    text = _TOKEN.sub('[redacted]', text)
    text = _BLOB.sub('[encoded data omitted]', text)
    text = ' '.join(_CONTROL.sub('', text).split())
    text = text.encode('ascii', errors='backslashreplace').decode('ascii')
    return text if len(text) <= limit else text[:limit - 3] + '...'


def summarise_record(record: dict) -> list[str]:
    """Select visible assistant text and tool activity, never reasoning or images."""
    message = record.get('message')
    if record.get('type') != 'message' or not isinstance(message, dict):
        return []
    role = message.get('role')
    if role not in ('assistant', 'toolResult'):
        return []
    content = message.get('content', [])
    if isinstance(content, str):
        content = [{'type': 'text', 'text': content}]
    if not isinstance(content, list):
        return []
    lines = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get('type')
        if role == 'assistant' and kind == 'toolCall':
            name = safe_text(block.get('name', 'unknown'), 60)
            args = block.get('arguments', {})
            details = ''
            if isinstance(args, dict):
                details = next((safe_text(args[key]) for key in
                                ('command', 'path', 'file_path', 'query')
                                if args.get(key) is not None), '')
            lines.append(f'tool {name}: {details}' if details else f'tool {name}')
        elif kind == 'text':
            text = safe_text(block.get('text', ''))
            if not text:
                continue
            if role == 'toolResult':
                name = safe_text(message.get('toolName', 'tool'), 60)
                status = 'error' if message.get('isError') else 'result'
                lines.append(f'{status} {name}: {text}')
            else:
                lines.append('assistant: ' + text)
    return lines


def timestamp(record: dict) -> str:
    value = record.get('timestamp')
    if isinstance(value, str):
        return safe_text(value, 40)
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def request_summary(root: Path) -> str:
    """Read request status counts without logging request or response contents."""
    counts: dict[str, int] = {}
    try:
        for path in root.glob('**/inference-gateway.sqlite3'):
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1) as db:
                for raw, in db.execute('SELECT record FROM requests'):
                    record = json.loads(raw)
                    status = record.get('status', 'unknown')
                    counts[status] = counts.get(status, 0) + 1
    except (OSError, sqlite3.Error, ValueError):
        return 'request status temporarily unavailable'
    if not counts:
        return 'no model requests recorded'
    return 'model requests: ' + ', '.join(f'{safe_text(key, 30)}={count}'
                                         for key, count in sorted(counts.items()))


class SessionFollower:
    """Incrementally read complete JSONL records with bounded memory."""

    def __init__(self, path: Path, initial_lines: int = 20):
        self.path = path
        self.initial_lines = initial_lines
        self.offset = 0
        self.pending = b''
        self.discarding = False
        self.started = False
        self.identity = None

    def poll(self) -> list[str]:
        try:
            stat = self.path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if self.identity != identity or stat.st_size < self.offset:
                self.offset = 0
                self.pending = b''
                self.discarding = False
                self.started = False
                self.identity = identity
            initial = not self.started
            if initial and stat.st_size > READ_LIMIT:
                self.offset = stat.st_size - READ_LIMIT
                self.discarding = True
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                chunk = stream.read(READ_LIMIT)
            self.offset += len(chunk)
        except (FileNotFoundError, PermissionError):
            return []
        self.started = True
        if self.discarding:
            split = chunk.find(b'\n')
            if split < 0:
                return []
            chunk = chunk[split + 1:]
            self.discarding = False
        data = self.pending + chunk
        records = data.split(b'\n')
        self.pending = records.pop()
        if len(self.pending) > RECORD_LIMIT:
            self.pending = b''
            self.discarding = True
        output = deque(maxlen=self.initial_lines if initial else 100)
        for raw in records:
            if not raw or len(raw) > RECORD_LIMIT:
                continue
            try:
                record = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            for line in summarise_record(record):
                output.append(timestamp(record) + ' ' + line)
        return list(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/dev/shm/bench'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='Show recent activity and exit')
    mode.add_argument('--follow', action='store_true', help='Follow live activity (default)')
    parser.add_argument('--lines', type=int, default=12)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.lines <= 100 or not 0.2 <= args.interval <= 60:
        parser.error('--lines must be 1-100 and --interval must be 0.2-60 seconds')
    followers: dict[Path, SessionFollower] = {}
    last_activity = last_heartbeat = time.monotonic()
    print('SimReadyBench live agent log (UTC). Press Ctrl+C to close this view.', flush=True)
    try:
        while True:
            paths = sorted(args.root.glob('**/agent/pi/sessions/*.jsonl'))
            for stale in followers.keys() - set(paths):
                del followers[stale]
            for path in paths:
                if path not in followers:
                    followers[path] = SessionFollower(path, args.lines)
                label = safe_text(path.parents[3].name, 60)
                for line in followers[path].poll():
                    print(f'{line} [{label}]', flush=True)
                    last_activity = time.monotonic()
            now = time.monotonic()
            if args.once or now - last_heartbeat >= 15:
                stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
                print(f'{stamp} Watching {len(paths)} agent session(s), '
                      f'{int(now - last_activity)}s since visible agent activity, '
                      f'{request_summary(args.root)}', flush=True)
                last_heartbeat = now
            if args.once:
                return 0
            time.sleep(args.interval)
    except (KeyboardInterrupt, BrokenPipeError):
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
