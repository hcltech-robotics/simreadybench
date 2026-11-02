"""Non-scoring worker diagnostics for the offline evaluator entry point."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
import re
import resource
import sys
import time

LOG_BYTES = 65536


def _resources():
    result = {}
    for name in ('cpu.max', 'cpu.stat', 'memory.max', 'memory.events'):
        try:
            result[name] = (Path('/sys/fs/cgroup') / name).read_text()[:2048].strip()
        except OSError:
            result[name] = None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result['children_user_seconds'] = usage.ru_utime
    result['children_system_seconds'] = usage.ru_stime
    return result


def observed_process(original, output, deadline):
    output = Path(output)

    def run(command, log, **kwargs):
        started = time.monotonic()
        before = _resources()
        status = original(command, log, **kwargs)
        elapsed = time.monotonic() - started
        try:
            output.mkdir(parents=True, exist_ok=True)
            phase = Path(log).parent.name
            if phase not in ('reference', 'formal', 'compare', 'render'):
                phase = 'worker'
            source = Path(log)
            size = source.stat().st_size if source.exists() else 0
            data = b''
            if size:
                with source.open('rb') as stream:
                    stream.seek(max(0, size - LOG_BYTES))
                    data = stream.read(LOG_BYTES)
            text = data.decode('utf-8', errors='replace')
            text = re.sub(r'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}', '[REDACTED]', text)
            text = re.sub(r'(?i)(Authorization:\s*Bearer\s+)\S+', r'\1[REDACTED]', text)
            relative = 'worker-' + phase + '.log'
            (output / relative).write_bytes(text.encode()[-LOG_BYTES:])
            row = {'schema': 'simready-worker-diagnostics-v1', 'phase': phase,
                   'started_utc': (datetime.datetime.now(datetime.timezone.utc)
                                   - datetime.timedelta(seconds=elapsed)).isoformat(),
                   'elapsed_seconds': elapsed, 'timeout_seconds': kwargs.get('timeout'),
                   'remaining_final_seconds_at_start': max(0, deadline - started),
                   'worker_status': status, 'resources_before': before,
                   'resources_after': _resources(), 'worker_log': relative,
                   'worker_log_original_bytes': size, 'log_truncated': size > LOG_BYTES}
            with (output / 'worker-diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        except (OSError, ValueError, TypeError):
            # Diagnostics must not change a scientific result or worker status.
            pass
        return status

    return run


def main(argv=None):
    from . import cli
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != 'evaluate' or '--output' not in args:
        return cli.main(args)
    output = Path(args[args.index('--output') + 1])
    original_worker, original_process = cli._run_worker, cli._run_process

    def worker(mode, request, scratch, deadline, **kwargs):
        cli._run_process = observed_process(original_process, output, deadline)
        try:
            return original_worker(mode, request, scratch, deadline, **kwargs)
        finally:
            cli._run_process = original_process

    cli._run_worker = worker
    try:
        return cli.main(args)
    finally:
        cli._run_worker, cli._run_process = original_worker, original_process


if __name__ == '__main__':
    raise SystemExit(main())
