import copy
import time
from dataclasses import replace

import pytest

from simready_bench.config import resolve_experiment
from simready_bench.daytona import _matrix_complete, _plan


def test_repetition_selection_preserves_original_identity_and_seed(setup):
    root, manifest, policy = setup
    manifest["repetitions"] = 4
    full = resolve_experiment(manifest, base_dir=root, policy=policy)
    expected = full["conditions"][3]
    manifest["reconstruction_selection"] = {"cup": [3]}
    policy["limits"]["maximum_matrix_runs"] = 1
    resumed = resolve_experiment(manifest, base_dir=root, policy=policy)
    assert resumed["conditions"] == [expected]
    assert resumed["normalized_manifest"]["repetitions"] == 4


@pytest.mark.parametrize(
    "selection",
    [{"missing": [3]}, {"cup": [4]}, {"cup": [1, 1]}, {"cup": [True]}, {"cup": [2, 1]}],
)
def test_invalid_repetition_selection_is_rejected(setup, selection):
    root, manifest, policy = setup
    manifest["repetitions"] = 4
    manifest["reconstruction_selection"] = selection
    with pytest.raises(ValueError, match="selection|repetitions"):
        resolve_experiment(manifest, base_dir=root, policy=policy)


def test_retirement_keeps_original_parent_cap_and_cannot_be_replayed(tmp_path):
    from simready_bench.local import ParentAuthorityLedger
    from simready_bench.engine.inference_gateway import ProjectBudget

    state = tmp_path / "authority"
    state.mkdir(mode=0o700)
    ledger = ParentAuthorityLedger(state)
    parent = ProjectBudget(
        project="simready-bench",
        authorization_id="original",
        approved=True,
        upstream_origin="https://api.openai.com",
        model="gpt-6.1-sol",
        expires_at=time.time() + 3600,
        maximum_requests=3200,
        maximum_spend_usd=250,
        pricing_status="known",
        external_spend_cap_verified=False,
    )
    initial = dict(
        allocation_id="first",
        requests=3200,
        maximum_spend_usd=250,
        work_root=tmp_path / "first",
    )
    ledger.reserve(parent, **initial)
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve(
            parent,
            allocation_id="early",
            requests=3104,
            maximum_spend_usd=242.5,
            work_root=tmp_path / "early",
        )
    proof = dict(
        allocation_id="first", requests=96, maximum_spend_usd=7.5, proof_sha256="a" * 64
    )
    assert ledger.retire(**proof)["status"] == "retired"
    assert ledger.retire(**proof)["status"] == "retired"
    with pytest.raises(ValueError, match="cannot change"):
        ledger.retire(**{**proof, "maximum_spend_usd": 0})
    with pytest.raises(ValueError, match="retired allocation"):
        ledger.reserve(parent, **initial)
    with pytest.raises(ValueError, match="fingerprint changed"):
        ledger.reserve(
            replace(parent, maximum_spend_usd=500),
            allocation_id="reset",
            requests=3104,
            maximum_spend_usd=242.5,
            work_root=tmp_path / "reset",
        )
    ledger.reserve(
        parent,
        allocation_id="remaining",
        requests=3104,
        maximum_spend_usd=242.5,
        work_root=tmp_path / "remaining",
    )
    with pytest.raises(ValueError, match="exhausted"):
        ledger.reserve(
            parent,
            allocation_id="extra",
            requests=1,
            maximum_spend_usd=0.001,
            work_root=tmp_path / "extra",
        )


def test_daytona_resume_plans_only_original_missing_repetitions():
    from test_daytona import experiment

    manifest, policy = experiment()
    original = copy.deepcopy(manifest)
    manifest["reconstruction_selection"] = {
        asset["id"]: list(range(4)) for asset in manifest["assets"]
    }
    manifest["reconstruction_selection"]["0"] = [3]
    provider, parent, requests, spend = _plan(manifest, policy)
    assert requests == 3104 and spend == 242.5
    assert parent.maximum_spend_usd == 250 and parent.maximum_requests == 3200
    rows = [
        dict(
            asset_id=asset["id"],
            repetition=repetition,
            status="completed",
            simready_reward=80,
        )
        for asset in manifest["assets"]
        for repetition in manifest["reconstruction_selection"][asset["id"]]
    ]
    result = dict(planned_conditions=97, completed_conditions=97, results=rows)
    assert _matrix_complete(result, manifest)
    assert not _matrix_complete(result, original)
    result["results"][0]["repetition"] = 0
    assert not _matrix_complete(result, manifest)
