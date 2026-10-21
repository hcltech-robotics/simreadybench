"""Run disjoint reconstruction shards with one report publisher."""

from __future__ import annotations

import copy
import functools
import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


_held_roots = ContextVar("simready_coordinator_roots", default=())


def validate_workers(workers):
    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer")
    return workers


@contextmanager
def coordinator_lock(output_dir):
    """Give one host coordinator ownership of a benchmark output tree."""
    root = Path(os.path.abspath(output_dir))
    if root.resolve() != root or not root.is_dir():
        raise ValueError("An existing regular benchmark output directory is required")
    key = str(root)
    held = _held_roots.get()
    if key in held:
        yield
        return
    path = root / ".coordinator.lock"
    if path.is_symlink() or path.exists() and path.stat().st_nlink != 1:
        raise ValueError("The benchmark coordinator lock must be a regular file")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    token = None
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ValueError(
                    "Another benchmark coordinator owns this output"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    "Another benchmark coordinator owns this output"
                ) from error
        locked = True
        token = _held_roots.set((*held, key))
        yield
    finally:
        if token is not None:
            _held_roots.reset(token)
        if locked:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def coordinated_resume(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        validate_workers(kwargs.get("workers", 1))
        with coordinator_lock(kwargs["output_dir"]):
            return function(*args, **kwargs)

    return wrapped


def partition_manifest(manifest, workers):
    """Balance asset groups while retaining their original condition identities."""
    from simready_bench.resume import condition_plan

    validate_workers(workers)
    expected = condition_plan(manifest)
    groups = []
    for asset in manifest["assets"]:
        repetitions = list(
            manifest.get("reconstruction_selection", {}).get(
                asset["id"], range(manifest["repetitions"])
            )
        )
        if repetitions:
            groups.append([(asset["id"], repetition) for repetition in repetitions])
    count = min(workers, sum(len(group) for group in groups))
    if not count:
        raise ValueError("A parallel execution requires pending reconstructions")
    while len(groups) < count:
        index = max(range(len(groups)), key=lambda item: len(groups[item]))
        group = groups.pop(index)
        middle = len(group) // 2
        groups[index:index] = [group[:middle], group[middle:]]
    assignments = [[] for _ in range(count)]
    for group in sorted(groups, key=lambda item: -len(item)):
        target = min(range(count), key=lambda item: len(assignments[item]))
        assignments[target].extend(group)
    shards, seen = [], {}
    for assignment in assignments:
        shard = copy.deepcopy(manifest)
        selection = {asset["id"]: [] for asset in manifest["assets"]}
        for asset, repetition in assignment:
            selection[asset].append(repetition)
        shard["reconstruction_selection"] = selection
        conditions = condition_plan(shard)
        if seen.keys() & conditions.keys():
            raise ValueError("Parallel reconstruction shards cannot overlap")
        seen.update(conditions)
        shards.append(shard)
    if seen != expected:
        raise ValueError(
            "Parallel reconstruction shards must preserve the original matrix"
        )
    return shards


def _stage_shards(
    manifest,
    policy,
    output,
    workers,
    *,
    assets_root,
    skills_root,
    wheel,
    cpu_snapshot,
    runtime_images,
):
    from simready_bench.benchmark import _copy_file
    from simready_bench.resume import _digest, _tree, _write, condition_plan

    shards = partition_manifest(manifest, workers)
    executions = output / "executions"
    executions.mkdir(exist_ok=True, mode=0o700)
    children = list(executions.iterdir())
    if any(
        path.resolve() != path or not path.is_dir() or not path.name.isdecimal()
        for path in children
    ):
        raise ValueError("Saved execution directories must have numeric names")
    number = max([0, *[int(path.name) for path in children]]) + 1
    roots = {
        name: Path(policy["roots"][name])
        for name in ("assets", "skills", "input_views", "private_manifests")
    }
    roots.update(assets=Path(assets_root), skills=Path(skills_root))
    inputs = {name: _tree(path) for name, path in roots.items()}
    prepared = []
    for index, shard in enumerate(shards, number):
        batch = executions / format(index, "03d")
        batch.mkdir(mode=0o700)
        selected_policy = copy.deepcopy(policy)
        for name, members in inputs.items():
            target = batch / name
            target.mkdir(mode=0o700)
            for relative in members:
                _copy_file(roots[name] / relative, target / relative)
            if _tree(target) != members:
                raise ValueError("A staged parallel input differs from the original")
            selected_policy["roots"][name] = str(target)
        _write(batch / "experiment.json", shard)
        _write(batch / "operator.json", selected_policy)
        _write(batch / "runtime-images.json", runtime_images)
        prepared.append((batch, shard, selected_policy))
    plan = {
        "schema": "simready-bench-parallel-plan-v1",
        "experiment_sha256": _digest(manifest),
        "workers": len(prepared),
        "wheel_sha256": (
            hashlib.sha256(Path(wheel).read_bytes()).hexdigest() if wheel else None
        ),
        "snapshot": cpu_snapshot,
        "runtime_images": runtime_images,
        "shards": [
            {
                "directory": batch.relative_to(output).as_posix(),
                "experiment_sha256": _digest(shard),
                "conditions": condition_plan(shard),
            }
            for batch, shard, _ in prepared
        ],
    }
    _write(output / ("parallel-plan-" + format(number, "03d") + ".json"), plan)
    return prepared


def run_parallel_daytona(
    manifest,
    *,
    policy,
    output_dir,
    workers,
    assets_root=None,
    skills_root=None,
    work_dir=None,
    wheel=None,
    cpu_snapshot=None,
    runtime_images=None,
    on_report=None,
    max_retries=2,
    retry_delay_seconds=5,
    maximum_retry_delay_seconds=60,
):
    """Run disjoint slots and restore unadmitted work as each executor finishes."""
    from simready_bench.orchestration import coordinate_daytona
    from simready_bench.recovery import frozen_wheel
    from simready_bench.runtime_images import DEFAULT_IMAGES

    validate_workers(workers)
    output = Path(os.path.abspath(output_dir))
    images = copy.deepcopy(DEFAULT_IMAGES if runtime_images is None else runtime_images)
    with coordinator_lock(output):
        wheel = frozen_wheel(output, wheel)
        if wheel is None:
            from simready_bench.daytona import build_wheel
            from uuid import uuid4

            wheel = build_wheel(output / (".parallel-package-" + uuid4().hex))
        from simready_bench.benchmark import freeze_execution_spec
        freeze_execution_spec(output, manifest, policy, wheel=wheel, runtime_images=images, cpu_snapshot=cpu_snapshot)
        return coordinate_daytona(
            manifest,
            policy=policy,
            output_dir=output,
            workers=workers,
            wheel=wheel,
            cpu_snapshot=cpu_snapshot,
            runtime_images=images,
            work_dir=work_dir,
            on_report=on_report,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            maximum_retry_delay_seconds=maximum_retry_delay_seconds,
        )
