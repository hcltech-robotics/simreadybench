"""NVIDIA validation of immutable candidate submissions."""

from __future__ import annotations
import hashlib, json, os, selectors, signal, socket, sqlite3, subprocess, tempfile, threading, time
from pathlib import Path
from simready_bench.engine.snapshots import verify_frozen_snapshot


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


class CandidateObserver:
    def __init__(
        self,
        *,
        validator_python,
        source_root,
        state_dir,
        expected_hostname,
        execution_enabled=False,
    ):
        self.python = Path(validator_python).absolute()
        self.source = Path(source_root).absolute()
        self.root = Path(state_dir).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (
            not self.python.is_file()
            or not (self.source / "simready_bench/engine/scoring.py").is_file()
        ):
            raise ValueError("Validator Python environment is unavailable")
        if (
            self.root.resolve() != self.root
            or not expected_hostname
            or type(execution_enabled) is not bool
        ):
            raise ValueError("Validator execution host must be configured")
        self.expected_hostname = expected_hostname
        self.enabled = execution_enabled
        package = Path(__file__).parent
        from simready_bench.engine.nvidia_compat import compatibility_identity

        self.identity = {
            "kind": "NVIDIA_Robotics_Prop_validator",
            "python": str(self.python),
            "compatibility_adapters": [compatibility_identity()],
            "sources": {
                name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                for name in (
                    "evaluation.py",
                    "scoring.py",
                    "validator.py",
                    "nvidia_compat.py",
                    "package_checks.py",
                    "snapshots.py",
                    "semantics.py",
                    "nvidia_profile.json",
                    "nvidia_optional_profile.json",
                )
            },
        }

    def __call__(self, snapshot, *, episode_id, deadline_at):
        if not self.enabled or socket.gethostname() != self.expected_hostname:
            raise RuntimeError("CPU observation not enabled on this host")
        if snapshot.episode_id != episode_id:
            raise ValueError("Snapshot episode mismatch")
        verify_frozen_snapshot(snapshot)
        directory = Path(tempfile.mkdtemp(dir=self.root, prefix="observe-"))
        output = directory / "result.json"
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(directory),
            "TMPDIR": str(directory),
            "PYTHONPATH": str(self.source),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        process = None
        started = time.monotonic()
        try:
            argv = [
                str(self.python),
                "-m",
                "simready_bench.engine.scoring",
                "observe",
                "--source",
                str(Path(snapshot.path) / snapshot.entrypoint),
                "--output",
                str(output),
                "--snapshot-sha256",
                snapshot.sha256,
                "--package-root",
                str(snapshot.path),
            ]
            with (directory / "worker.log").open("xb") as log:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=directory,
                    env=env,
                    start_new_session=True,
                    close_fds=True,
                )
                os.set_blocking(process.stdout.fileno(), False)
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    logged = 0
                    while selector.get_map():
                        if time.time() >= deadline_at:
                            raise TimeoutError("Observer deadline")
                        if output.is_symlink():
                            raise ValueError("Bounded observer result required")
                        for key, _ in selector.select(0.05):
                            block = os.read(key.fd, 65536)
                            if not block:
                                selector.unregister(key.fileobj)
                                continue
                            logged += len(block)
                            if logged > 1024 * 1024:
                                raise ValueError("Observer log limit")
                            log.write(block)
                    if process.wait(timeout=max(0.01, deadline_at - time.time())) != 0:
                        raise RuntimeError("Observer failed")
            if (
                output.resolve() != output
                or output.is_symlink()
                or not output.is_file()
                or output.stat().st_nlink != 1
            ):
                raise ValueError("Invalid observer output")
            value = json.loads(
                output.read_text(),
                parse_constant=lambda _: (_ for _ in ()).throw(
                    ValueError("Nonfinite JSON")
                ),
            )
            verify_frozen_snapshot(snapshot)
            return {
                "snapshot_sha256": snapshot.sha256,
                "observation": value,
                "resource_cost": {
                    "host": self.expected_hostname,
                    "cpu_worker_wall_seconds": time.monotonic() - started,
                    "cost_usd": None,
                    "cost_status": "unpriced",
                },
            }
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
                process.stdout.close()


class Evaluator:
    def __init__(self, state_dir, *, observer=None, allowed_reference_root=None):
        if observer is not None and not callable(observer):
            raise ValueError("A callable candidate validator is required")
        self.root = Path(state_dir).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.resolve() != self.root:
            raise ValueError("Evaluator directory must not contain links")
        self.allowed_reference_root = (
            Path(allowed_reference_root).absolute() if allowed_reference_root else None
        )
        self.observer = observer
        self.lock = threading.RLock()
        self.identity = digest(
            {
                "observer": getattr(observer, "identity", None),
                "reference_root": str(self.allowed_reference_root),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            }
        )
        self.db = self.root / "evaluation.sqlite3"
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS observations(identity TEXT,snapshot TEXT,record TEXT,PRIMARY KEY(identity,snapshot))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS finals(episode TEXT PRIMARY KEY,snapshot TEXT,record TEXT)"
            )
        self.db.chmod(0o600)

    def _connect(self):
        return sqlite3.connect(self.db, timeout=10)

    def evaluate(
        self,
        snapshot,
        task_config,
        episode_id,
        deadline_at,
        *,
        final=True,
        stop_reason="unknown",
    ):
        from simready_bench.engine.scoring import evaluate_snapshot

        if snapshot.episode_id != episode_id:
            raise ValueError("Snapshot belongs to another episode")
        if "validator_scoring" not in task_config:
            raise ValueError("Task requires a Robotics-Prop scoring policy")
        return evaluate_snapshot(
            self,
            snapshot,
            task_config,
            episode_id,
            deadline_at,
            final=final,
            stop_reason=stop_reason,
        )
