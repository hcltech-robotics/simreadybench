"""Fault fixtures for recovery without fresh authoring or relaxed evidence."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from simready_bench.engine import hf_pricing as hf
from simready_bench.engine.request_cost import CostDenied
from simready_bench.local import complete_trial_result
from test_budget_completion import docker_boundary, evaluate_final, native_result
from test_confirmed_submit_stop import setup
from test_hf_cost import MODEL, catalog


def test_concurrent_admissions_share_only_an_inflight_fresh_check(monkeypatch):
    async def exercise():
        requests = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def fetch(model, *, deadline_at):
            requests.append(model)
            entered.set()
            await release.wait()
            return hf.validate_catalog_route(catalog(), model)

        monkeypatch.setattr(hf, "_fetch_hf_route_once", fetch)
        first = asyncio.create_task(hf.fetch_hf_route(MODEL, deadline_at=time.time() + 30))
        await entered.wait()
        second = asyncio.create_task(hf.fetch_hf_route(MODEL, deadline_at=time.time() + 30))
        await asyncio.sleep(0)
        assert requests == [MODEL]
        release.set()
        a, b = await asyncio.gather(first, second)
        assert a is b
        await hf.fetch_hf_route(MODEL, deadline_at=time.time() + 30)
        assert requests == [MODEL, MODEL]

    asyncio.run(exercise())


def test_cancelled_catalogue_waiter_does_not_cancel_other_owned_admission(monkeypatch):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def fetch(model, *, deadline_at):
            calls.append(True)
            entered.set()
            await release.wait()
            return hf.validate_catalog_route(catalog(), model)

        monkeypatch.setattr(hf, "_fetch_hf_route_once", fetch)
        first = asyncio.create_task(hf.fetch_hf_route(MODEL, deadline_at=time.time() + 30))
        await entered.wait()
        second = asyncio.create_task(hf.fetch_hf_route(MODEL, deadline_at=time.time() + 30))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        release.set()
        assert (await second).model == MODEL
        assert calls == [True]

    asyncio.run(exercise())


def test_hf_admission_jitter_cannot_extend_the_original_deadline(monkeypatch):
    from test_hf_route_recovery import run_probe

    calls, waits, run = run_probe(monkeypatch, [CostDenied("hf_pricing_metadata_unavailable"), catalog()], deadline=105.5)
    monkeypatch.setattr(hf.random, "uniform", lambda low, high: high)
    with pytest.raises(CostDenied, match="hf_pricing_metadata_unavailable"):
        run()
    assert calls == [(MODEL, 105.5)] and waits == []


def test_failed_stop_is_resumed_without_capturing_or_stopping_twice(tmp_path, monkeypatch):
    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        calls = []

        async def verifier(episode_id):
            calls.append(episode_id)
            if len(calls) == 1:
                raise TimeoutError("temporary verifier setup failure")

        bridge.services = SimpleNamespace(revoke_episode=lambda episode: None, ensure_verifier=verifier)
        try:
            with pytest.raises(TimeoutError):
                await bridge.request_stop("episode", "submitted")
            frozen = env.final_snapshot
            assert frozen is not None and env.actor_stopped
            assert await bridge.request_stop("episode", "submitted") is True
            assert env.final_snapshot is frozen
            assert bridge.completed_actor_stop(env)
            receipts = list((controller.state_dir / "native-stop-transitions").glob("*.json"))
            assert len(receipts) == 2
            records = [json.loads(path.read_text()) for path in sorted(receipts)]
            assert [row["status"] for row in records] == ["failed", "completed"]
            assert records[1]["snapshot_id"] == frozen.snapshot_id
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def test_stop_reobserves_container_after_kill_inspection_failure(tmp_path, monkeypatch):
    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        kills, inspections = [], []
        stopped = {"Running": False, "Paused": False, "OOMKilled": False, "ExitCode": 137}

        async def compose(args, **kwargs):
            if args[0] == "kill":
                kills.append(True)
            return SimpleNamespace(return_code=0, stdout="a" * 64 if args[0] == "ps" else "")

        async def engine(*args):
            inspections.append(True)
            if len(inspections) == 2:
                raise RuntimeError("temporary state inspection loss")
            return json.dumps({"Running": True} if len(inspections) == 1 else stopped).encode()

        monkeypatch.setattr(env, "_run_docker_compose_command", compose)
        monkeypatch.setattr(env, "_engine", engine)
        try:
            with pytest.raises(RuntimeError, match="temporary state"):
                await bridge.request_stop("episode", "submitted")
            progress = list((controller.state_dir / "native-environments").glob("*.actor-stop-progress.json"))
            assert len(progress) == 1 and json.loads(progress[0].read_text())["kill_return_code"] == 0
            assert await bridge.request_stop("episode", "submitted")
            assert kills == [True]
            assert bridge.completed_actor_stop(env)
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def test_persistent_stop_failure_exhausts_three_owned_transitions(tmp_path, monkeypatch):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        calls = []

        async def fail():
            calls.append(True)
            raise RuntimeError("transport failure")

        monkeypatch.setattr(env, "stop_actor_writes", fail)
        try:
            for _ in range(3):
                with pytest.raises(RuntimeError, match="transport failure"):
                    await bridge.request_stop("episode", "submitted")
            with pytest.raises(RuntimeError, match="recovery allowance exhausted"):
                await bridge.request_stop("episode", "submitted")
            assert len(calls) == 3
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def recovery_result(tmp_path, env, evaluation, *, stop="stop", response_id="response-ok"):
    from simready_bench.local import export_submission

    session = tmp_path / "native/pi/sessions/session.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({"type": "message", "id": "final", "message": {
        "role": "assistant", "stopReason": stop, "responseId": response_id,
        "content": [{"type": "text", "text": "Saved the candidate."}],
        **({"errorMessage": "Provider failed"} if stop == "error" else {}),
    }}) + "\n")
    return {
        "trial_id": "episode", "termination_reason": "submitted",
        "actor_stopped": True, "actor_stop_audit": env._actor_stop_audit,
        "snapshot_sha256": env.final_snapshot.sha256,
        "submission": export_submission(env.final_snapshot, tmp_path / "submission"),
        "evaluation": evaluation, "official_reward": evaluation["official_reward"],
        "model_quality_evidence": True, "harbor_trace_paths": [str(session)],
        "gateway_records": [
            {"gateway_request_id": "old", "status": "cancelled", "request_sha256": "old-request"},
            {"gateway_request_id": "new", "status": "completed", "upstream_status": 200,
             "request_sha256": "new-request", "provider_response_ids": ["response-ok"]},
        ],
    }


@pytest.mark.parametrize("stop,expected", [("stop", "completed"), ("error", "infrastructure_error")])
def test_recovered_request_requires_native_actor_submission_and_evaluation(tmp_path, monkeypatch, stop, expected):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "submitted")
            result = recovery_result(tmp_path, env, evaluate_final(tmp_path, env.final_snapshot), stop=stop)
            result["actor_stop_receipt"] = vars(bridge._controlled_receipts["episode"])
            complete_trial_result(result, native_result(None), normal_stop=True)
            assert result["status"] == expected
            if expected == "completed":
                assert result["official_reward"] == 1
                assert result["request_recovery"]["resolved_request_ids"] == ["old"]
                assert result["request_recovery"]["successful_request_id"] == "new"
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize("missing", ["submission", "evaluation", "harbor_trace_paths", "actor_stop_receipt"])
def test_later_http_success_alone_never_overrides_a_historical_failure(tmp_path, monkeypatch, missing):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "submitted")
            result = recovery_result(tmp_path, env, evaluate_final(tmp_path, env.final_snapshot))
            result["actor_stop_receipt"] = vars(bridge._controlled_receipts["episode"])
            result.pop(missing)
            complete_trial_result(result, native_result(None), normal_stop=True)
            assert result["status"] == "infrastructure_error"
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def test_recovered_transport_retains_a_genuine_measured_zero(tmp_path, monkeypatch):
    from simready_bench.engine.scoring import pinned_profile

    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "submitted")
            evaluation = evaluate_final(tmp_path, env.final_snapshot,
                failed=[item["code"] for item in pinned_profile()["mandatory_requirements"]])
            assert evaluation["simready"]["score_percent"] == 0
            result = recovery_result(tmp_path, env, evaluation)
            result["actor_stop_receipt"] = vars(bridge._controlled_receipts["episode"])
            complete_trial_result(result, native_result(None), normal_stop=True)
            assert result["status"] == "completed"
            assert result["official_reward"] == 0
            assert result["evaluation"]["simready"]["score_percent"] == 0
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize("native_error,hold", [
    ('503 {"error":{"code":"hf_pricing_metadata_changed"}}', "hf_pricing_metadata_changed"),
    ("400 hf_model_declaration_route_mismatch", "hf_model_declaration_route_mismatch"),
    ("Unexpected failure mentioning hf_pricing_metadata_changed somewhere", None),
])
def test_pre_admission_native_errors_are_terminal_and_pin_codes_hold_without_guessing(tmp_path, monkeypatch, native_error, hold):
    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "submitted")
            result = recovery_result(tmp_path, env, evaluate_final(tmp_path, env.final_snapshot), stop="error")
            result["gateway_records"] = [result["gateway_records"][-1]]
            result["actor_stop_receipt"] = vars(bridge._controlled_receipts["episode"])
            session = result["harbor_trace_paths"][0]
            from pathlib import Path
            row = json.loads(Path(session).read_text())
            row["message"]["errorMessage"] = native_error
            Path(session).write_text(json.dumps(row) + "\n")
            complete_trial_result(result, native_result(None), normal_stop=True)
            assert result["status"] == "infrastructure_error"
            assert result.get("infrastructure_hold") == hold
            assert result["official_reward"] is None
            assert result["automatic_retry"] is False
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def test_verified_final_stream_stop_retains_partial_native_message_without_becoming_provider_error(tmp_path, monkeypatch):
    from pathlib import Path

    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "inference_budget_exhausted")
            result = recovery_result(tmp_path, env, evaluate_final(tmp_path, env.final_snapshot))
            result["termination_reason"] = "inference_budget_exhausted"
            result["actor_stop_receipt"] = vars(bridge._controlled_receipts["episode"])
            session = Path(result["harbor_trace_paths"][0])
            with session.open("a") as stream:
                stream.write(json.dumps({"type": "message", "id": "interrupted", "parentId": "final", "message": {
                    "role": "assistant", "stopReason": "aborted", "responseId": "response-partial",
                    "content": [{"type": "text", "text": "Partial final explanation"}]}}) + "\n")
            result["gateway_records"].append({"gateway_request_id": "partial", "status": "cancelled",
                "observed_upstream_status": 200, "observed_provider_response_ids": ["response-partial"]})
            complete_trial_result(result, native_result(None), normal_stop=True)
            assert result["status"] == "completed"
            assert result["official_reward"] == 1
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize("damage", ["missing", "malformed", "oversized", "no_assistant"])
def test_referenced_native_terminal_evidence_fails_closed_without_fresh_authoring(tmp_path, monkeypatch, damage):
    from pathlib import Path

    async def exercise():
        _, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        try:
            await bridge.request_stop("episode", "submitted")
            result = recovery_result(tmp_path, env, evaluate_final(tmp_path, env.final_snapshot))
            result["actor_stop_receipt"] = vars(bridge._controlled_receipts["episode"])
            result["gateway_records"] = [result["gateway_records"][-1]]
            session = Path(result["harbor_trace_paths"][0])
            if damage == "missing":
                session.unlink()
            elif damage == "malformed":
                session.write_text("malformed native session\n")
            elif damage == "oversized":
                with session.open("wb") as stream:
                    stream.truncate(128 * 1024 * 1024 + 1)
            else:
                session.write_text(json.dumps({"type": "session", "id": "session"}) + "\n")
            complete_trial_result(result, native_result(None), normal_stop=True)
            assert result["status"] == "infrastructure_error"
            assert result["infrastructure_hold"] == "native_terminal_evidence_unavailable"
            assert result["automatic_retry"] is False
            assert result["official_reward"] is None
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


def test_private_validation_failure_receipt_retains_sanitised_chain_and_correlation(tmp_path):
    from simready_bench.engine.harbor_services import ReferenceActorTools

    tools = object.__new__(ReferenceActorTools)
    tools.validator = object()
    tools.controller = SimpleNamespace(state_dir=tmp_path)

    def fail(job):
        try:
            raise OSError("Cannot read /home/operator/private/report.json Authorization: Bearer SECRET_EXAMPLE_VALUE")
        except OSError as error:
            raise RuntimeError("Validation transport failed") from error

    tools._adapter = lambda *args: fail
    result = tools._validation_job(binding=SimpleNamespace(episode_id="ep"),
        job=SimpleNamespace(job_id="job", episode_id="ep", snapshot=SimpleNamespace(snapshot_id="snap", sha256="a" * 64)),
        payload={}, arguments={})
    correlation = result.structured_content["failure_id"]
    receipts = list((tmp_path / "validation-failures").glob("*.json"))
    assert len(receipts) == 1
    encoded = receipts[0].read_text()
    receipt = json.loads(encoded)
    assert receipt["failure_id"] == correlation
    assert receipt["phase"] == "interactive_evaluation"
    assert [cause["type"] for cause in receipt["causes"]] == ["RuntimeError", "OSError"]
    assert "SECRET_EXAMPLE_VALUE" not in encoded and "/home/operator" not in encoded
    assert receipt["snapshot_sha256"] == "a" * 64
    assert "Validation transport failed" not in result.model_dump_json()


def test_final_validation_failure_returns_private_receipt_correlation(tmp_path):
    import hashlib
    from simready_bench.engine.harbor_services import ReferenceVerifierTransport

    def failed(*args, **kwargs):
        raise OSError("Cannot run /home/operator/private/validator token=SECRET_EXAMPLE_VALUE")

    async def exercise():
        snapshot = SimpleNamespace(snapshot_id="snap", sha256="a" * 64, valid=True)
        env = SimpleNamespace(actor_stopped=True, final_snapshot=snapshot)
        controller = SimpleNamespace(state_dir=tmp_path,
            get_episode=lambda _: SimpleNamespace(status="closed", termination_reason="submitted"))
        transport = object.__new__(ReferenceVerifierTransport)
        transport.token_hash = hashlib.sha256(b"token").digest()
        transport.episode_id, transport.lock = "ep", asyncio.Lock()
        transport.services = SimpleNamespace(lifecycle=SimpleNamespace(controller=controller, _episodes={"ep": env}),
            evaluator=SimpleNamespace(evaluate=failed), verifier_timeout_seconds=5,
            input_for_episode=lambda _: SimpleNamespace(task_config={"validator_scoring": {}}))

        async def body():
            return {}

        request = SimpleNamespace(headers={"Authorization": "Bearer token"}, method="POST",
            path="/internal/final", query_string="", content_length=2, json=body)
        response = await transport.handle(request)
        assert response.status == 503
        payload = json.loads(response.text)
        assert payload["error"] == "final_validation_failed"
        record = json.loads((tmp_path / "validation-failures" / (payload["failure_id"] + ".json")).read_text())
        assert record["phase"] == "final_evaluation"
        assert "SECRET_EXAMPLE_VALUE" not in json.dumps(record)

    asyncio.run(exercise())


def test_final_evaluation_retries_the_identical_sealed_candidate_and_preserves_deadline(tmp_path, monkeypatch):
    from simready_bench.engine.evaluation import Evaluator
    from simready_bench.engine.harbor_services import _DiagnosticObserver
    from simready_bench.engine.validation_recovery import evaluate_final_snapshot
    from simready_bench.engine.verifier import scoring_policy
    from test_validator_reward_v1 import official_report

    async def exercise():
        controller, bridge, env, _ = setup(tmp_path)
        docker_boundary(env, monkeypatch)
        calls = []

        def observe(snapshot, *, episode_id, deadline_at):
            calls.append((snapshot.snapshot_id, snapshot.sha256, episode_id, deadline_at))
            if len(calls) == 1:
                raise TimeoutError("temporary observer transport failure")
            return {"snapshot_sha256": snapshot.sha256, "observation": {
                "source_record": {"sha256": snapshot.files[0].sha256},
                "nvidia_report": official_report(), "preflight": {"package_ok": True}}}

        evaluator = Evaluator(tmp_path / "evaluator", observer=_DiagnosticObserver(observe, controller))
        try:
            await bridge.request_stop("episode", "submitted")
            before = controller.get_episode("episode")
            original_deadline = time.time() + 5
            report = await evaluate_final_snapshot(evaluator, controller, env.final_snapshot,
                {"validator_scoring": scoring_policy()}, "episode", deadline_at=original_deadline, stop_reason="voluntary_good_enough")
            assert report["simready"]["score_percent"] == 100
            assert len(calls) == 2 and calls[0] == calls[1]
            assert calls[0][3] == original_deadline
            repeated = await evaluate_final_snapshot(evaluator, controller, env.final_snapshot,
                {"validator_scoring": scoring_policy()}, "episode", deadline_at=time.time() + 50, stop_reason="voluntary_good_enough")
            assert repeated == report and len(calls) == 2
            after = controller.get_episode("episode")
            assert after.inference_calls_used == before.inference_calls_used
            assert after.deadline_at == before.deadline_at
            receipts = [json.loads(path.read_text()) for path in sorted((controller.state_dir / "final-validation-attempts").glob("*.finished.json"))]
            assert [row["status"] for row in receipts] == ["failed", "completed"]
            assert receipts[0]["failure_id"] is not None
            assert receipts[0]["snapshot_sha256"] == receipts[1]["snapshot_sha256"]
        finally:
            bridge.uninstall()

    asyncio.run(exercise())


@pytest.mark.parametrize("error", [TimeoutError("temporary transport"), ValueError("snapshot digest mismatch")])
def test_final_evaluation_retry_allowance_and_integrity_hold_are_durable(tmp_path, error):
    from simready_bench.engine.validation_recovery import evaluate_final_snapshot

    async def exercise():
        snapshot = SimpleNamespace(snapshot_id="snapshot", sha256="a" * 64)
        controller = SimpleNamespace(state_dir=tmp_path)
        calls = []

        def failed(*args, **kwargs):
            calls.append(True)
            raise error

        evaluator = SimpleNamespace(evaluate=failed)
        expected = 3 if isinstance(error, TimeoutError) else 1
        for _ in range(2):
            with pytest.raises((TimeoutError, ValueError)):
                await evaluate_final_snapshot(evaluator, controller, snapshot, {}, "episode",
                    deadline_at=time.time() + 5, stop_reason="deadline")
        assert len(calls) == expected
        receipts = list((tmp_path / "final-validation-attempts").glob("*.started.json"))
        assert len(receipts) == expected

    asyncio.run(exercise())
