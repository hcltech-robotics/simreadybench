import io
import json
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def reconnect_fixture(
    tmp_path,
    monkeypatch,
    *,
    outcomes,
    terminal="completed",
    html=True,
    download_error=False,
):
    from simready_bench import daytona, local

    output = tmp_path / "run"
    output.mkdir()
    allocation = {"allocation_id": "original-allocation", "status": "reserved"}
    delivery = {
        "status": "running",
        "phase": "controller",
        "sandbox_id": "original-executor",
        "session_name": "original-session",
        "command_id": "original-command",
        "controller_started": True,
        "cleanup_confirmed": False,
        "parent_allocation": allocation,
        "planned_requests": 32,
        "maximum_spend_usd": 2.5,
        "planned_conditions": 1,
        "deadline_at": time.time() + 60,
    }
    (output / "daytona-delivery.json").write_text(json.dumps(delivery))
    row = {
        "condition_id": "original-condition",
        "asset_id": "reference-box",
        "repetition": 0,
        "status": "completed" if terminal == "completed" else "infrastructure_error",
        "simready_reward": 80 if terminal == "completed" else None,
    }
    result = {
        "planned_conditions": 1,
        "completed_conditions": int(terminal == "completed"),
        "results": [row],
    }
    portable = {"schema": "simready-bench-portable-report-v1", **result}
    terminal_state = {"status": terminal, "completed_at": time.time()}
    files = {
        "work/output/attempts/000-reference/results.json": json.dumps(
            portable
        ).encode(),
        "work/output/attempts/000-reference/report.html": b"<h1>Attempt</h1>",
        "work/output/attempts/000-reference/generated.usda": b"#usda 1.0\n",
        "work/run-result.json": json.dumps(result).encode(),
        "work/controller-status.json": json.dumps(terminal_state).encode(),
        "work/controller.stdout.log": b"retained output\n",
        "work/controller.stderr.log": b"private-openai-key\n",
        "work/authority/parent-allowances.sqlite3": b"retained authority bytes",
    }
    if html:
        files.update(
            {
                "work/output/report.html": b"<h1>Results</h1>",
                "work/output/results.json": json.dumps(portable).encode(),
                "work/output/scores.csv": b"asset_id,simready_reward\nreference-box,80\n",
            }
        )
    archive = tmp_path / "remote.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            stream.addfile(member, io.BytesIO(raw))
    events, polls, commands, downloads = [], [], [], []
    state = {"deleted": False, "terminal": False}
    pending = iter(outcomes)

    class Process:
        def create_session(self, *args, **kwargs):
            raise AssertionError("Reconnection must reuse the saved session")

        def execute_session_command(self, *args, **kwargs):
            raise AssertionError("Reconnection must not restart the controller")

        def get_session_command(self, session, command, **kwargs):
            polls.append((session, command))
            outcome = next(pending, TimeoutError("API observation unavailable"))
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is not None:
                state["terminal"] = True
            return SimpleNamespace(exit_code=outcome)

        def exec(self, command, timeout=30, **kwargs):
            commands.append(command)
            if "cat /work/controller-status.json" in command:
                return SimpleNamespace(exit_code=0, result=json.dumps(terminal_state))
            return SimpleNamespace(exit_code=0, result="")

    class FileSystem:
        def download_file(self, source, destination=None, **kwargs):
            downloads.append(source)
            if source == "/tmp/simready-benchmark-results.tar.gz":
                if download_error:
                    raise OSError("Collection download failed")
                events.append("collected")
                Path(destination).write_bytes(archive.read_bytes())
                return None
            if source == "/work/controller-status.json":
                if terminal is None:
                    raise TimeoutError("Controller status observation unavailable")
                status = (
                    terminal_state
                    if state["terminal"] or terminal != "completed"
                    else {"status": "running"}
                )
                return json.dumps(status).encode()
            if source == "/work/output/results.json":
                return json.dumps(portable).encode()
            raise AssertionError("Unexpected remote read: " + source)

    sandbox = SimpleNamespace(
        id="original-executor", process=Process(), fs=FileSystem()
    )

    class Client:
        def create(self, *args, **kwargs):
            raise AssertionError("Reconnection must not allocate another executor")

        def get(self, identity, **kwargs):
            assert identity == "original-executor"
            events.append("get")
            if state["deleted"]:
                from daytona.common.errors import DaytonaNotFoundError

                raise DaytonaNotFoundError("deleted")
            return sandbox

        def delete(self, selected, timeout=60, wait=True):
            assert selected is sandbox and "collected" in events
            assert (
                output / "daytona-executor/authority/parent-allowances.sqlite3"
            ).is_file()
            events.append("deleted")
            state["deleted"] = True

    class NoNewAuthority:
        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "Reconnection must retain the original parent reservation"
            )

    monkeypatch.setattr(local, "ParentAuthorityLedger", NoNewAuthority)
    monkeypatch.setattr(daytona.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-key")
    monkeypatch.setenv("DAYTONA_API_KEY", "private-daytona-key")
    manifest = {
        "assets": [{"id": "reference-box"}],
        "repetitions": 1,
        "models": [{"provider_ref": "selected", "model_id": "gpt-6.1-sol"}],
        "harnesses": [{"id": "pi"}],
        "skill_sets": [{"id": "none"}],
    }
    policy = {"providers": {"selected": {"api_key_env": "OPENAI_API_KEY"}}}
    return SimpleNamespace(
        output=output,
        manifest=manifest,
        policy=policy,
        client=Client(),
        events=events,
        polls=polls,
        commands=commands,
        downloads=downloads,
        allocation=allocation,
        delivery=delivery,
        state=state,
    )


def test_reconnect_monitors_same_running_handle_then_collects_and_deletes(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import reconnect_daytona

    fixture = reconnect_fixture(tmp_path, monkeypatch, outcomes=[None, 0])
    result = reconnect_daytona(
        fixture.manifest,
        policy=fixture.policy,
        output_dir=fixture.output,
        client=fixture.client,
    )
    assert fixture.polls == [("original-session", "original-command")] * 2
    assert fixture.events.index("collected") < fixture.events.index("deleted")
    assert result["completed_conditions"] == 1
    assert result["deployment"]["cleanup_confirmed"] is True
    assert result["deployment"]["parent_allocation"] == fixture.allocation
    assert result["deployment"]["planned_requests"] == 32
    assert result["deployment"]["maximum_spend_usd"] == 2.5
    assert (fixture.output / "attempts/000-reference/generated.usda").is_file()
    assert (
        "private-openai-key"
        not in (fixture.output / "daytona-executor/controller.stderr.log").read_text()
    )
    assert not any("provider.json" in command for command in fixture.commands)
    assert not any("provider.json" in source for source in fixture.downloads)
    with tarfile.open(fixture.output / "daytona-results.tar.gz", "r:gz") as archive:
        for member in archive.getmembers():
            assert (
                member.name.startswith("work/") and "provider.json" not in member.name
            )
            if member.isfile():
                assert b"private-openai-key" not in archive.extractfile(member).read()


def test_reconnect_already_collected_and_deleted_keeps_the_original_result(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import reconnect_daytona

    fixture = reconnect_fixture(tmp_path, monkeypatch, outcomes=[0])
    first = reconnect_daytona(
        fixture.manifest,
        policy=fixture.policy,
        output_dir=fixture.output,
        client=fixture.client,
    )
    poll_count = len(fixture.polls)
    second = reconnect_daytona(
        fixture.manifest,
        policy=fixture.policy,
        output_dir=fixture.output,
        client=fixture.client,
    )
    assert second["results"] == first["results"]
    assert second["deployment"]["parent_allocation"] == fixture.allocation
    assert second["deployment"]["cleanup_confirmed"] is True
    assert len(fixture.polls) == poll_count and fixture.events.count("deleted") == 1
    saved = json.loads((fixture.output / "daytona-delivery.json").read_text())
    assert saved["status"] == "completed" and saved["collection"] == "completed"


def test_terminal_failed_controller_collects_attempts_without_root_html(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import reconnect_daytona

    fixture = reconnect_fixture(
        tmp_path,
        monkeypatch,
        outcomes=[TimeoutError("Session observation unavailable")],
        terminal="failed",
        html=False,
    )
    result = reconnect_daytona(
        fixture.manifest,
        policy=fixture.policy,
        output_dir=fixture.output,
        client=fixture.client,
    )
    assert result["deployment"]["status"] == "failed"
    assert result["deployment"]["collection"] == "completed"
    assert result["deployment"]["cleanup_confirmed"] is True
    assert not (fixture.output / "report.html").exists()
    assert (fixture.output / "attempts/000-reference/generated.usda").is_file()
    assert (
        fixture.output / "daytona-executor/authority/parent-allowances.sqlite3"
    ).read_bytes() == b"retained authority bytes"
    assert result["results"][0]["status"] == "infrastructure_error"


def test_unknown_api_state_preserves_executor_and_original_reservation(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import reconnect_daytona

    fixture = reconnect_fixture(
        tmp_path, monkeypatch, outcomes=[], terminal=None, html=False
    )
    fixture.delivery["deadline_at"] = time.time() + 0.05
    (fixture.output / "daytona-delivery.json").write_text(json.dumps(fixture.delivery))
    with pytest.raises((RuntimeError, TimeoutError)):
        reconnect_daytona(
            fixture.manifest,
            policy=fixture.policy,
            output_dir=fixture.output,
            client=fixture.client,
        )
    assert not fixture.state["deleted"] and "collected" not in fixture.events
    assert fixture.polls and set(fixture.polls) == {
        ("original-session", "original-command")
    }
    assert not any(
        "docker stop" in command or "docker kill" in command
        for command in fixture.commands
    )
    saved = json.loads((fixture.output / "daytona-delivery.json").read_text())
    assert saved["parent_allocation"] == fixture.allocation
    assert saved["cleanup_confirmed"] is False
    assert saved.get("collection") != "completed"


def test_collection_failure_keeps_terminal_executor_for_another_collection(
    tmp_path, monkeypatch
):
    from simready_bench.daytona import reconnect_daytona

    fixture = reconnect_fixture(
        tmp_path, monkeypatch, outcomes=[1], terminal="failed", download_error=True
    )
    with pytest.raises((RuntimeError, OSError)):
        reconnect_daytona(
            fixture.manifest,
            policy=fixture.policy,
            output_dir=fixture.output,
            client=fixture.client,
        )
    assert not fixture.state["deleted"]
    saved = json.loads((fixture.output / "daytona-delivery.json").read_text())
    assert saved["parent_allocation"] == fixture.allocation
    assert saved["cleanup_confirmed"] is False
    assert saved.get("collection_error")
