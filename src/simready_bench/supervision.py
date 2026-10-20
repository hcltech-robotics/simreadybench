"""Restart a failed Daytona coordinator against its original saved benchmark."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

TEMPORARY_FAILURE = 75
INITIAL_DELAY_SECONDS = 1
MAXIMUM_DELAY_SECONDS = 60
_CRASH_SIGNALS = frozenset(
    value
    for name in ("SIGKILL", "SIGSEGV", "SIGBUS", "SIGABRT", "SIGILL", "SIGFPE")
    if (value := getattr(signal, name, None)) is not None
)


class SupervisorBusyError(RuntimeError):
    """Another supervisor already owns this output directory."""


def coordinator_exit_code(error, *, deploy):
    """Separate retryable observation failures from terminal benchmark outcomes."""
    if deploy != "daytona":
        return 2
    from .daytona import DaytonaObservationError
    from .engine.downloads import is_transient
    from .publishing import PublicationError
    from .recovery import RecoveryLimitReached

    if isinstance(error, RecoveryLimitReached):
        return 2
    if isinstance(error, DaytonaObservationError) or is_transient(error):
        return TEMPORARY_FAILURE
    if isinstance(error, PublicationError) and error.retryable:
        return TEMPORARY_FAILURE
    return 2


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path):
    if (
        path.resolve() != path
        or not path.is_file()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 * 1024**2
    ):
        raise ValueError("A bounded regular supervisor record is required")
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path, value):
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            json.dump(value, stream, ensure_ascii=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _child_arguments(argv):
    result, index = [], 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--supervise":
            index += 1
        elif argument == "--coordinator-restarts":
            index += 2
        elif argument.startswith("--coordinator-restarts="):
            index += 1
        else:
            result.append(argument)
            index += 1
    return result


def _invocation_arguments(arguments):
    result, index = [], 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--resume" or argument.startswith("--workers="):
            index += 1
        elif argument == "--workers":
            index += 2
        else:
            result.append(argument)
            index += 1
    return result


def _frozen_run(output, expected=None):
    path = output / "run.json"
    if not path.exists():
        if expected is not None:
            raise ValueError("The original saved benchmark is missing")
        return None
    saved = _read(path)
    if saved.get("schema") != "simready-bench-run-v1" or saved.get("deployment") != "daytona":
        raise ValueError("Supervision requires a saved Daytona benchmark")
    for name, key in (
        ("experiment.json", "experiment_sha256"),
        ("operator.json", "operator_sha256"),
    ):
        if _digest(_read(output / name)) != saved.get(key):
            raise ValueError("The saved benchmark configuration changed")
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and fingerprint != expected:
        raise ValueError("The original saved benchmark identity changed")
    return fingerprint


@contextmanager
def _manager_lock(path):
    import fcntl

    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) & 0o077
        ):
            raise ValueError("A regular supervisor lock is required")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SupervisorBusyError("Another supervisor owns this benchmark output") from error
        yield
    finally:
        os.close(descriptor)


def _retryable_returncode(code):
    return code == TEMPORARY_FAILURE or code < 0 and -code in _CRASH_SIGNALS


def supervise(argv, *, output_dir, maximum_restarts=5):
    """Run one coordinator at a time and resume only its frozen Daytona manifest."""
    if sys.platform != "linux":
        raise ValueError("--supervise requires Linux or WSL")
    if type(maximum_restarts) is not int or not 0 <= maximum_restarts <= 20:
        raise ValueError("Coordinator restarts must be an integer between 0 and 20")
    arguments = _child_arguments(list(argv))
    if not arguments or arguments[0] != "benchmark":
        raise ValueError("Supervision requires a benchmark command")
    output = Path(os.path.abspath(output_dir))
    if output.resolve() != output or not output.name:
        raise ValueError("A literal benchmark output directory is required")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_path = output.with_name("." + output.name + ".supervisor.json")
    lock_path = output.with_name("." + output.name + ".supervisor.lock")
    invocation = _digest(_invocation_arguments(arguments))

    with _manager_lock(lock_path):
        if state_path.exists():
            state = _read(state_path)
            if (
                state.get("schema") != "simready-bench-supervisor-v1"
                or state.get("invocation_sha256") != invocation
                or state.get("maximum_restarts") != maximum_restarts
            ):
                raise ValueError("The supervisor belongs to a different benchmark command")
            if (
                type(state.get("restarts")) is not int
                or not 0 <= state["restarts"] <= maximum_restarts
                or type(state.get("launches")) is not int
                or state["launches"] < 0
                or type(state.get("next_start_at")) not in (int, float)
                or not math.isfinite(state["next_start_at"])
                or state.get("status") not in {
                    "new", "starting", "running", "restart_backoff", "completed",
                    "incomplete", "failed", "restart_exhausted", "not_resumable",
                }
            ):
                raise ValueError("The durable supervisor counters are invalid")
            _frozen_run(output, state.get("saved_run_sha256"))
            if state.get("status") in ("completed", "incomplete", "failed", "restart_exhausted", "not_resumable"):
                if type(state.get("exit_code")) is not int:
                    raise ValueError("The terminal supervisor exit code is invalid")
                return state["exit_code"]
        else:
            state = {
                "schema": "simready-bench-supervisor-v1",
                "invocation_sha256": invocation,
                "maximum_restarts": maximum_restarts,
                "restarts": 0,
                "launches": 0,
                "saved_run_sha256": None,
                "status": "new",
                "next_start_at": 0,
            }

        def save(**values):
            state.update(values, updated_at=time.time())
            _save(state_path, state)

        def schedule(code):
            frozen = _frozen_run(output, state["saved_run_sha256"])
            if frozen is None:
                save(status="not_resumable", exit_code=TEMPORARY_FAILURE, last_exit_code=code)
                return False
            state["saved_run_sha256"] = frozen
            if state["restarts"] >= maximum_restarts:
                save(status="restart_exhausted", exit_code=TEMPORARY_FAILURE, last_exit_code=code)
                return False
            state["restarts"] += 1
            delay = min(MAXIMUM_DELAY_SECONDS, INITIAL_DELAY_SECONDS * 2 ** (state["restarts"] - 1))
            save(status="restart_backoff", last_exit_code=code, next_start_at=time.time() + delay)
            return True

        # The previous manager's child has a kernel parent-death signal. Any
        # restarted coordinator still has to acquire the benchmark's own lock.
        if state["status"] in ("starting", "running"):
            if not schedule(TEMPORARY_FAILURE):
                return TEMPORARY_FAILURE

        while True:
            while (remaining := state["next_start_at"] - time.time()) > 0:
                time.sleep(min(1, remaining))
            frozen = _frozen_run(output, state["saved_run_sha256"])
            if frozen is not None:
                state["saved_run_sha256"] = frozen
            command = list(arguments)
            if frozen is not None and "--resume" not in command:
                command.append("--resume")
            state["launches"] += 1
            save(status="starting", pid=os.getpid(), child_pid=None)
            child = subprocess.Popen(
                [sys.executable, "-m", "simready_bench.supervision", "--child-parent-pid", str(os.getpid()), "--", *command],
                close_fds=True,
            )
            save(status="running", child_pid=child.pid)
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            state["child_pid"] = None
            if not _retryable_returncode(code):
                code = code if code >= 0 else 128 - code
                save(status="completed" if code == 0 else "incomplete" if code == 2 else "failed", exit_code=code, last_exit_code=code)
                return code
            if not schedule(code):
                return TEMPORARY_FAILURE


def _exec_child(parent_pid, arguments):
    if sys.platform != "linux":
        raise ValueError("Coordinator supervision requires Linux or WSL")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    if os.getppid() != parent_pid:
        return TEMPORARY_FAILURE
    os.execv(sys.executable, [sys.executable, "-m", "simready_bench", *arguments])


if __name__ == "__main__":
    if len(sys.argv) < 5 or sys.argv[1] != "--child-parent-pid" or sys.argv[3] != "--":
        raise SystemExit("A supervised coordinator bootstrap is required")
    raise SystemExit(_exec_child(int(sys.argv[2]), sys.argv[4:]))
