"""Attach a read-only live log session to owned benchmark workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from daytona import Daytona, SessionExecuteRequest

SESSION = 'simready-live-logs'
COMMAND = ('set -o pipefail; docker exec simready-benchmark-controller '
           'python -u /tmp/simready-worker-logs.py 2>&1 | tee /work/simready-live.log')
WRAPPER = b'#!/bin/sh\nexec tail -n 80 -F /work/simready-live.log\n'


def install(client, worker, source, previous=None):
    """Attach an observer without restarting or writing to the actor."""
    box = client.get(worker['sandbox_id'])
    if box.labels.get('project') != 'simreadybench' or box.labels.get('role') != 'benchmark-executor':
        raise ValueError('Worker ownership does not match SimReadyBench')
    if previous:
        command = box.process.get_session_command(SESSION, previous['command_id'])
        if command.exit_code is None:
            return previous
        return dict(previous, status='finished', exit_code=command.exit_code)
    sessions = box.process.list_sessions()
    if any(getattr(item, 'session_id', None) == SESSION for item in sessions):
        raise RuntimeError('Existing observer session needs reconciliation')
    box.fs.upload_file(source, '/work/simready-worker-logs.py')
    box.fs.upload_file(WRAPPER, '/usr/local/bin/simready-logs')
    response = box.process.exec(
        'chmod 755 /usr/local/bin/simready-logs && '
        'docker cp /work/simready-worker-logs.py '
        'simready-benchmark-controller:/tmp/simready-worker-logs.py', timeout=20)
    if response.exit_code:
        raise RuntimeError('Could not install read-only log viewer')
    box.process.create_session(SESSION)
    result = box.process.execute_session_command(
        SESSION, SessionExecuteRequest(command=COMMAND, run_async=True))
    return {'sandbox_id':box.id, 'session':SESSION, 'command_id':result.cmd_id,
            'installed_at':time.time(), 'status':'running',
            'dashboard_url':'https://app.daytona.io/dashboard/sandboxes?sandboxId='+box.id,
            'terminal_command':'simready-logs'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1]/'src/simready_bench/worker_logs.py')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args(argv)
    source = args.source.read_bytes()
    state = json.loads(args.state.read_text()) if args.state.exists() else {'workers':{}}
    client = Daytona()
    while True:
        all_done = True
        for root in args.runs:
            results = json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else {}
            if not results.get('planned_conditions') or results.get('completed_conditions') != results.get('planned_conditions'):
                all_done = False
            for path in sorted(root.glob('executions/*/daytona-delivery.json')):
                worker = json.loads(path.read_text())
                if worker.get('cleanup_confirmed'):
                    continue
                all_done = False
                if not worker.get('controller_started') or not worker.get('sandbox_id'):
                    continue
                identity = worker['sandbox_id']
                previous = state['workers'].get(identity)
                if previous:
                    if previous.get('status') == 'finished' or time.time()-previous.get('checked_at',0) < 60:
                        continue
                    coordinator = root/'coordinator-state.json'
                    if not coordinator.exists() or time.time()-coordinator.stat().st_mtime > 120:
                        continue
                try:
                    observed = install(client, worker, source, previous)
                    changed = not previous or observed.get('status') != previous.get('status')
                    state['workers'][identity] = dict(observed,checked_at=time.time())
                    if changed:
                        print(json.dumps(state['workers'][identity],ensure_ascii=True), flush=True)
                except Exception as error:
                    print(json.dumps({'sandbox_id':identity,'observer_error':type(error).__name__},ensure_ascii=True),flush=True)
        state['checked_at'] = time.time()
        state['status'] = 'complete' if all_done else 'watching'
        args.state.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.state.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2,ensure_ascii=True)+'\n',encoding='ascii')
        temporary.replace(args.state)
        if all_done or not args.watch:
            return 0
        time.sleep(20)


if __name__ == '__main__':
    raise SystemExit(main())
