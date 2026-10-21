import hashlib
import json
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from simready_bench import daytona, orchestration, parallel, resume
from simready_bench.engine.inference_gateway import ProjectBudget
from simready_bench.local import ParentAuthorityLedger
from simready_bench.runtime_images import DEFAULT_IMAGES
from test_parallel import manifest
from test_resume import portable, write

pytestmark = pytest.mark.skipif(
    __import__("os").name == "nt", reason="Native allowance ownership requires Linux"
)


def setup_run(tmp_path, monkeypatch):
    selected = manifest()
    selected["assets"] = selected["assets"][:1]
    output = tmp_path / "run"
    output.mkdir()
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        (output / name).mkdir()
        roots[name] = str(output / name)
    parent = ProjectBudget(
        project="simready-bench",
        authorization_id="original",
        approved=True,
        upstream_origin="https://api.openai.com",
        model="gpt-6.1-sol",
        expires_at=time.time() + 3600,
        maximum_requests=128,
        maximum_spend_usd=10,
        pricing_status="known",
    )
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    policy = {
        "roots": roots,
        "providers": {"selected": {"kind": "openai", "project_budget": asdict(parent)}},
        "runtime": {"authority_state_dir": str(authority)},
    }
    wheel = output / "frozen.whl"
    wheel.write_bytes(b"frozen wheel")

    class Deleted:
        def get(self, identity):
            from daytona.common.errors import DaytonaNotFoundError

            raise DaytonaNotFoundError("deleted")

    monkeypatch.setattr(resume, "_daytona_client", Deleted)
    return output, selected, policy, parent, wheel


def source_failure(batch, selected, policy, parent, *, transient=True, requests=0):
    count = len(resume.condition_plan(selected)) * 32
    allocation = "allocation-" + batch.name
    ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
        parent,
        allocation_id=allocation,
        requests=count,
        maximum_spend_usd=count / 32 * 2.5,
        work_root=batch,
    )
    remote = batch / "daytona-executor"
    (remote / "authority").mkdir(parents=True, mode=0o700)
    ParentAuthorityLedger(remote / "authority")
    relay = {
        "model": parent.model,
        "generation_requests": requests,
        "input_counter_requests": 0,
        "maximum_requests": count,
        "maximum_condition_requests": 32,
        "expires_at": parent.expires_at,
        "events": [],
        "condition_generation_requests": {},
        "condition_input_counter_requests": {},
        "condition_limits": {},
    }
    write(remote / "relay-ledger.json", relay)
    status = {"status": "failed", "error": "source download failed"}
    if transient:
        status["source_transport_exhaustion"] = {
            "schema": "simready-bench-download-exhaustion-v1",
            "method": "GET",
            "attempts": 5,
            "maximum_attempts": 5,
            "transient": True,
            "host": "example.org",
            "url_sha256": "e" * 64,
            "exception_class": "ConnectionResetError",
        }
    write(remote / "controller-status.json", status)
    write(
        batch / "daytona-delivery.json",
        {
            "schema": "simready-bench-daytona-delivery-v2",
            "collection": "completed",
            "cleanup_confirmed": True,
            "sandbox_id": "deleted-" + batch.name,
            "parent_allocation": {"allocation_id": allocation},
            "planned_requests": count,
            "maximum_spend_usd": count / 32 * 2.5,
            "relay_ledger_sha256": hashlib.sha256(
                (remote / "relay-ledger.json").read_bytes()
            ).hexdigest(),
        },
    )


@pytest.mark.parametrize("transient", [False, True])
def test_only_correlated_zero_request_transient_source_failure_releases_replacement(
    tmp_path, monkeypatch, transient
):
    output, selected, policy, parent, wheel = setup_run(tmp_path, monkeypatch)
    batch, shard, _ = parallel._stage_shards(
        selected,
        policy,
        output,
        1,
        assets_root=policy["roots"]["assets"],
        skills_root=policy["roots"]["skills"],
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
    )[0]
    source_failure(batch, shard, policy, parent, transient=transient)
    calls = []

    def execute(shard, **options):
        calls.append(options["output_dir"])
        portable(
            options["output_dir"] / "attempts/final",
            [
                (key, value, "completed")
                for key, value in resume.condition_plan(shard).items()
            ],
        )

    monkeypatch.setattr(daytona, "run_daytona", execute)
    options = dict(
        policy=policy,
        output_dir=output,
        workers=1,
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
    )
    if transient:
        assert (
            orchestration.coordinate_daytona(selected, **options)[
                "completed_conditions"
            ]
            == 4
        )
        assert len(calls) == 4
    else:
        from simready_bench.recovery import RecoveryLimitReached

        with pytest.raises(RecoveryLimitReached):
            orchestration.coordinate_daytona(selected, **options)
        assert calls == []
    with ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])._db() as db:
        assert db.execute("SELECT requests,usd FROM retirements").fetchall() == [
            (0, "0")
        ]
        assert db.execute(
            "SELECT maximum_requests,maximum_usd FROM parents"
        ).fetchone() == (128, "10")


def test_retained_provider_request_prevents_zero_request_source_retry(
    tmp_path, monkeypatch
):
    output, selected, policy, parent, wheel = setup_run(tmp_path, monkeypatch)
    batch, shard, _ = parallel._stage_shards(
        selected,
        policy,
        output,
        1,
        assets_root=policy["roots"]["assets"],
        skills_root=policy["roots"]["skills"],
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
    )[0]
    source_failure(batch, shard, policy, parent, requests=1)
    with pytest.raises(ValueError, match="cannot reset an admitted provider request"):
        resume._retire(
            batch, [], policy=policy, manifest=selected, client=resume._daytona_client()
        )


def test_failed_slot_is_replaced_before_healthy_sibling_finishes(tmp_path, monkeypatch):
    output, selected, policy, parent, wheel = setup_run(tmp_path, monkeypatch)
    parallel._stage_shards(
        selected,
        policy,
        output,
        2,
        assets_root=policy["roots"]["assets"],
        skills_root=policy["roots"]["skills"],
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
    )
    replacement_started = threading.Event()
    slow_started = threading.Event()
    calls = []

    def execute(shard, **options):
        batch = options["output_dir"]
        calls.append(batch.name)
        if batch.name == "001":
            assert slow_started.wait(10)
            source_failure(batch, shard, policy, parent)
            raise ConnectionResetError("five source GET attempts exhausted")
        if batch.name == "002":
            slow_started.set()
            assert replacement_started.wait(
                10
            ), "A failed slot waited for its healthy sibling"
        else:
            replacement_started.set()
        portable(
            batch / "attempts/final",
            [
                (key, value, "completed")
                for key, value in resume.condition_plan(shard).items()
            ],
        )

    monkeypatch.setattr(daytona, "run_daytona", execute)
    result = orchestration.coordinate_daytona(
        selected,
        policy=policy,
        output_dir=output,
        workers=2,
        wheel=wheel,
        cpu_snapshot=None,
        runtime_images=DEFAULT_IMAGES,
        retry_delay_seconds=0,
    )
    assert result["completed_conditions"] == 4
    assert set(calls) == {"001", "002", "003", "004"}


def test_running_status_without_command_id_remains_observable(tmp_path, monkeypatch):
    states = iter([{"status": "running"}] * 15 + [{"status": "completed"}])
    monkeypatch.setattr(daytona, "_controller_status", lambda sandbox: next(states))
    monkeypatch.setattr(daytona.time, "sleep", lambda _: None)
    monkeypatch.setattr(daytona, "_collect_live_reports", lambda *args, **kwargs: None)
    from simready_bench import controller_runner

    monkeypatch.setattr(controller_runner, "forward_telemetry", lambda *args, **kwargs: None)

    class Files:
        def download_file(self, *args):
            raise FileNotFoundError

    sandbox = SimpleNamespace(fs=Files())
    record = {"deadline_at": time.time() + 60}
    daytona._monitor_executor(sandbox, record, tmp_path, save=lambda: None)
    assert record["terminal_confirmed"] is True
    assert record["controller_status"] == "completed"


def test_permanent_source_failure_closes_only_the_unadmitted_shard_remainder(
    tmp_path, monkeypatch
):
    from test_resume import stopped_run

    output, selected, policy, ledger = stopped_run(tmp_path, monkeypatch)
    terminal = output / "daytona-executor/controller-status.json"
    write(
        terminal,
        {
            "status": "failed",
            "source_acquisition_failure": True,
            "error": "BOM mismatch",
        },
    )
    proof = resume._retire(
        output,
        resume._reports(output),
        policy=policy,
        manifest=selected,
        client=resume._daytona_client(),
    )
    assert len(proof["admitted_conditions"]) == 3
    assert len(proof["terminal_unadmitted_conditions"]) == 97
    assert not set(proof["admitted_conditions"]) & set(
        proof["terminal_unadmitted_conditions"]
    )
    assert proof["requests"] == 96
    assert proof["maximum_spend_usd"] == "7.5"


@pytest.mark.parametrize("known", [False, True])
def test_unknown_create_acceptance_uses_only_the_exact_saved_name(
    tmp_path, monkeypatch, known
):
    output, selected, policy, parent, wheel = setup_run(tmp_path, monkeypatch)
    run_id = "a" * 32
    name = "simready-benchmark-" + run_id[:12]
    allocation = hashlib.sha256((str(output) + "\n" + run_id).encode()).hexdigest()
    ParentAuthorityLedger(policy["runtime"]["authority_state_dir"]).reserve(
        parent,
        allocation_id=allocation,
        requests=128,
        maximum_spend_usd=10,
        work_root=output,
    )
    record = {
        "phase": "creating",
        "run_id": run_id,
        "sandbox_name": name,
        "parent_allocation": {"allocation_id": allocation},
        "planned_requests": 128,
        "maximum_spend_usd": 10,
        "planned_conditions": 4,
        "controller_started": False,
    }
    write(output / "daytona-delivery.json", record)
    calls = []

    class Files:
        def list_files(self, path):
            from daytona.common.errors import DaytonaFileNotFoundError

            assert path in ("/work", "/run/simready-benchmark")
            raise DaytonaFileNotFoundError("not staged")

    sandbox = SimpleNamespace(
        id="accepted-executor",
        name=name,
        labels={"project": "simreadybench", "role": "benchmark-executor"},
        fs=Files(),
    )

    class Client:
        deleted = False

        def get(self, identity, **_):
            from daytona.common.errors import DaytonaNotFoundError

            calls.append(identity)
            assert identity in (name, sandbox.id)
            if not known or self.deleted:
                raise DaytonaNotFoundError("not found")
            return sandbox

        def create(self, *args, **kwargs):
            pytest.fail("An uncertain accepted create must never be duplicated")

        def delete(self, existing, **_):
            assert existing is sandbox
            self.deleted = True

    client = Client()
    if not known:
        with pytest.raises(daytona.DaytonaObservationError):
            daytona.reconnect_daytona(
                selected, policy=policy, output_dir=output, client=client
            )
        assert calls == [name]
        assert not (output / "retirement.json").exists()
        return
    original = ParentAuthorityLedger.retire
    monkeypatch.setattr(
        ParentAuthorityLedger,
        "retire",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("interrupted before commit")
        ),
    )
    with pytest.raises(RuntimeError, match="interrupted before commit"):
        daytona.reconnect_daytona(
            selected, policy=policy, output_dir=output, client=client
        )
    proof = (output / "retirement.json").read_bytes()
    saved = json.loads((output / "daytona-delivery.json").read_text())
    assert saved["cleanup_confirmed"] is True
    # A create RPC can already have returned before the deletion/retirement gap.
    saved["phase"] = "bootstrap"
    write(output / "daytona-delivery.json", saved)
    monkeypatch.setattr(ParentAuthorityLedger, "retire", original)
    daytona.reconnect_daytona(selected, policy=policy, output_dir=output, client=client)
    resume._retire(output, [], policy=policy, manifest=selected, client=client)
    assert (output / "retirement.json").read_bytes() == proof
    with ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])._db() as db:
        assert db.execute("SELECT requests,usd FROM retirements").fetchall() == [
            (0, "0")
        ]
        assert db.execute(
            "SELECT maximum_requests,maximum_usd FROM parents"
        ).fetchone() == (128, "10")
