from copy import deepcopy
import json

import pytest

from test_publishing import publication
from simready_bench.provenance import build_attempt_provenance, build_run_spec
from simready_bench.publishing import HuggingFacePublisher, PublicationError, _settings
from simready_bench.reporting import compact_report


def test_public_settings_distinguish_effective_budgets():
    manifest = {"assets": [{"id": "a"}], "models": [{"id": "m"}],
        "harnesses": [{"id": "pi"}], "skill_sets": [{"id": "none"}],
        "repetitions": 4, "seed": 0, "scoring": {"profile": {}, "policy_id": "fixed"},
        "defaults": {"budgets": {"inference_requests": 64, "maximum_spend_usd": 8}}}
    first = _settings(manifest)
    manifest["defaults"]["budgets"]["inference_requests"] = 1
    assert first != _settings(manifest)


def test_portable_and_public_results_retain_bound_runtime_and_interventions(publication):
    manifest, report, hub, row, save = publication
    spec = build_run_spec(manifest, runtime={"wheel_sha256": "b" * 64})
    current = row(passed=0)
    receipt = build_attempt_provenance(spec, attempt_id="attempt-1", condition_id=current["condition_id"],
        runtime={"wheel_sha256": "b" * 64}, candidate={"manifest_sha256": current["candidate"]["manifest_sha256"]},
        interventions=[{"kind": "collection_retry", "phase": "collection", "attempt": 1}])
    current["execution_provenance"] = receipt
    current['request_recovery'] = {'schema': 'simready-request-recovery-v1',
                                   'disposition': 'verified_recovered',
                                   'resolved_request_ids': ['gateway-1'],
                                   'successful_request_id': 'gateway-2',
                                   'native_session_sha256': 'c' * 64}
    saved = save([current])
    raw = json.loads((report / "results.json").read_text())
    raw["run_spec"] = spec
    compact = compact_report(raw)
    assert compact["run_spec"] == spec
    assert compact["results"][0]["execution_provenance"] == receipt
    (report / "results.json").write_text(json.dumps(raw))
    publisher = HuggingFacePublisher("team/results", "canonical", manifest, api=hub)
    publisher(saved)
    public = json.loads(hub.files["runs/canonical/results.json"])
    assert public["run_spec"] == spec
    assert public["results"][0]["execution_provenance"] == receipt
    assert public['results'][0]['request_recovery'] == current['request_recovery']
    assert public["results"][0]["simready_reward"] == 0
    assert hub.files["runs/canonical/run-spec.json"]
    changed = deepcopy(raw)
    changed["results"][0]["execution_provenance"] = build_attempt_provenance(spec,
        attempt_id="attempt-2", condition_id=current["condition_id"],
        runtime={"wheel_sha256": "b" * 64}, candidate=receipt["candidate"])
    (report / "results.json").write_text(json.dumps(changed))
    with pytest.raises(PublicationError, match="already published reconstruction"):
        publisher(saved)


def test_legacy_publication_can_gain_schema_without_changing_measured_rows(publication):
    manifest, _, hub, row, save = publication
    publisher = HuggingFacePublisher("team/results", "legacy", manifest, api=hub)
    publisher(save([row(passed=0)]))
    key = "runs/legacy/results.json"
    previous = json.loads(hub.files[key])
    previous["settings"].pop("budgets", None)
    previous["settings"].pop("overrides", None)
    previous["settings"].pop("tool_protocol", None)
    previous["settings"].pop("settings_schema", None)
    previous["results"][0].pop("provenance_coverage", None)
    hub.files[key] = json.dumps(previous).encode()
    publisher(save([row(passed=0), row(repetition=1)]))
    actual = json.loads(hub.files[key])
    assert actual["results"][0]["simready_reward"] == 0
    assert actual["results"][0]["simready"] == previous["results"][0]["simready"]
    assert actual["results"][0]["provenance_coverage"]["status"] == "legacy_unbound"
    assert actual["legacy_settings_conditions"] == [previous["results"][0]["condition_id"]]


def test_later_interventions_preserve_published_attempt_identity(publication, tmp_path):
    from simready_bench import benchmark
    from simready_bench.provenance import write_run_spec

    manifest, source, hub, row, save = publication
    current = row(passed=0)
    current.pop('images')
    current.pop('traces')
    current['candidate'].update(directory='original', entrypoint='original/asset.usda')
    source_report = save([current])
    write_run_spec(tmp_path, build_run_spec(manifest))
    publisher = HuggingFacePublisher('team/results', 'canonical', manifest, api=hub)
    report = benchmark._aggregate_reports([(source, source_report)],
        output_dir=tmp_path, planned_conditions=4)
    publisher(report)
    first = json.loads((tmp_path / 'results.json').read_text())['results'][0]['execution_provenance']
    interventions = tmp_path / 'operational-interventions'
    interventions.mkdir()
    (interventions / '001.json').write_text(json.dumps({'kind': 'executor_stop', 'recorded_at': 1}))
    report = benchmark._aggregate_reports([(source, source_report)],
        output_dir=tmp_path, planned_conditions=4)
    second = json.loads((tmp_path / 'results.json').read_text())['results'][0]['execution_provenance']
    assert second == first
    publisher(report)
    public = json.loads(hub.files['runs/canonical/results.json'])
    assert public['results'][0]['execution_provenance'] == first
    assert public['operational_provenance']['entries'][0]['path'] == 'operational-interventions/001.json'
    assert len(hub.commits) == 2


def test_resuming_published_canonical_rows_does_not_add_attempt_receipts(publication):
    from simready_bench import benchmark
    from simready_bench.provenance import write_run_spec

    manifest, source, hub, row, save = publication
    current = row(passed=0)
    current.pop('images')
    current.pop('traces')
    current['candidate'].update(directory='original', entrypoint='original/asset.usda')
    report = save([current])
    publisher = HuggingFacePublisher('team/results', 'canonical', manifest, api=hub)
    publisher(report)
    original = json.loads(hub.files['runs/canonical/results.json'])['results'][0]
    assert 'execution_provenance' not in original

    write_run_spec(source, build_run_spec(manifest))
    report = benchmark._aggregate_reports([(source, report)],
        output_dir=source, planned_conditions=4)
    resumed = json.loads((source / 'results.json').read_text())['results'][0]
    assert 'execution_provenance' not in resumed
    assert resumed['simready_reward'] == current['simready_reward']
    publisher(report)
    assert json.loads(hub.files['runs/canonical/results.json'])['results'][0] == original
