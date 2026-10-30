"""Exercise the original 2.2.0 feedback ledger with synthetic outcomes, not a model.

No replacement controller is installed. These tests characterize the benchmark
service's five *validator-call* quota and same-request retry semantics.
"""

import hashlib
import importlib
import os
from pathlib import Path
import sys

import pytest

WHEEL = os.environ.get("NV100_EVAL_WHEEL")
pytestmark = pytest.mark.skipif(not WHEEL, reason="requires the unchanged NV100 2.2.0 wheel")


@pytest.fixture
def feedback():
    wheel = Path(WHEEL)
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == "caf759ff7d9b392922530567b22678da4556de366fa879ef6395299425627d52"
    sys.path.insert(0, str(wheel))
    module = importlib.import_module("simready_eval.feedback")
    assert str(wheel) in module.__file__
    try:
        yield module
    finally:
        sys.path.remove(str(wheel))


def outcome(passing):
    return {"outcome": "complete" if passing else "incomplete", "score_percent": 100 if passing else 50,
            "simready": {"all_scored_mandatory_pass": passing,
                         "checks": [{"requirement_id": "fixture-rule", "status": "pass" if passing else "fail"}]},
            "findings": [] if passing else [{"rule_id": "fixture-rule", "message": "repair this fixture"}]}


def send(service, feedback, package, request_id):
    body = feedback.candidate_archive(package)
    return service.validate(body, request_id, hashlib.sha256(body).hexdigest())


def test_failure_repair_and_early_completion_need_not_use_five(feedback, tmp_path):
    package = tmp_path / "candidate"
    package.mkdir()
    asset = package / "asset.usd"
    asset.write_text("bad synthetic candidate")
    service = feedback.FeedbackService(tmp_path / "state", evaluator=lambda p: outcome((p / "asset.usd").read_text().startswith("good")))
    try:
        status, first = send(service, feedback, package, "candidate-1")
        assert status == 200 and not first["result"]["simready"]["all_scored_mandatory_pass"]
        assert service.ledger.health()["used"] == 1
        asset.write_text("good synthetic candidate")
        status, final = send(service, feedback, package, "candidate-2")
        assert status == 200 and final["result"]["simready"]["all_scored_mandatory_pass"]
        assert service.ledger.health()["remaining"] == 3
    finally:
        service.close()


def test_passing_first_candidate_can_finish_with_four_calls_left(feedback, tmp_path):
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "asset.usd").write_text("synthetic")
    service = feedback.FeedbackService(tmp_path / "state", evaluator=lambda p: outcome(True))
    try:
        assert send(service, feedback, package, "final")[1]["result"]["simready"]["all_scored_mandatory_pass"]
        assert service.ledger.health() == {"status": "ok", "limit": 5, "used": 1, "remaining": 4}
    finally:
        service.close()


def test_exhaustion_transport_retry_changed_bytes_and_busy_are_counted_correctly(feedback, tmp_path):
    package = tmp_path / "candidate"
    package.mkdir()
    asset = package / "asset.usd"
    asset.write_text("synthetic")
    calls = []
    def evaluate(p):
        calls.append(1)
        return outcome(False)
    service = feedback.FeedbackService(tmp_path / "state", evaluator=evaluate)
    try:
        service._inflight.acquire()
        try:
            assert send(service, feedback, package, "a")[0] == 503
            assert service.ledger.health()["used"] == 0
        finally:
            service._inflight.release()
        first = send(service, feedback, package, "a")
        assert send(service, feedback, package, "a") == first  # Lost response, same bytes/ID.
        assert len(calls) == 1 and service.ledger.health()["used"] == 1
        asset.write_text("changed synthetic")
        assert send(service, feedback, package, "a")[0] == 409
        for i in range(4):
            assert send(service, feedback, package, f"repair-{i}")[0] == 200
        assert send(service, feedback, package, "sixth")[0] == 429
        assert len(calls) == 5 and service.ledger.health()["remaining"] == 0
    finally:
        service.close()


def test_admitted_validator_timeout_is_unknown_and_same_id_does_not_charge_again(feedback, tmp_path):
    package = tmp_path / "candidate"
    package.mkdir()
    (package / "asset.usd").write_text("synthetic")
    service = feedback.FeedbackService(tmp_path / "state", evaluator=lambda p: feedback.unknown("validator_timeout"))
    try:
        status, result = send(service, feedback, package, "timed-out")
        assert status == 200 and result["result"]["outcome"] == "infrastructure_error"
        assert result["result"]["score_percent"] is None
        assert send(service, feedback, package, "timed-out")[1] == result
        assert service.ledger.health()["used"] == 1
    finally:
        service.close()
