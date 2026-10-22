from copy import deepcopy
import hashlib
import json
import zipfile

import pytest

from simready_bench.provenance import (
    build_attempt_provenance, build_run_spec, provenance_fields, write_run_spec,
)


def experiment():
    return {
        "assets": [{"id": "asset", "closure": [
            {"path": "asset.usda", "sha256": "a" * 64, "size_bytes": 12}]}],
        "models": [{"id": "qwen", "model_id": "Qwen/model", "provider_ref": "hf",
                    "adapter": "openai-completions", "generation": {"temperature": 0}}],
        "harnesses": [{"id": "pi", "adapter": "pi"}],
        "skill_sets": [{"id": "none", "paths": [], "closure": []}],
        "repetitions": 4, "seed": 19, "authoring_protocol": "64-turn-v1",
        "scoring": {"policy_id": "95-plus-5"},
        "defaults": {"budgets": {"inference_requests": 64, "maximum_spend_usd": 8},
                     "system_prompt": "Produce the asset", "generation": {"temperature": 0}},
    }


@pytest.fixture
def source(tmp_path):
    package = tmp_path / "src" / "simready_bench"
    package.mkdir(parents=True)
    (package / "config.py").write_text("REQUESTS = 64\n")
    (package / "prompt.txt").write_text("Immutable prompt\n")
    return tmp_path


def test_effective_budgets_content_and_runtime_are_distinct_bindings(source):
    baseline = build_run_spec(experiment(), runtime={"wheel_sha256": "b" * 64}, source_root=source)
    changed = experiment()
    changed["defaults"]["budgets"]["inference_requests"] = 1
    budget = build_run_spec(changed, runtime={"wheel_sha256": "b" * 64}, source_root=source)
    runtime = build_run_spec(experiment(), runtime={"wheel_sha256": "c" * 64}, source_root=source)
    assert baseline["protocol_sha256"] != budget["protocol_sha256"]
    assert baseline["protocol_sha256"] == runtime["protocol_sha256"]
    assert baseline["sha256"] != runtime["sha256"]
    assert baseline["protocol"]["manifest"]["defaults"]["budgets"]["inference_requests"] == 64
    assert baseline["protocol"]["contract"]["files"]["prompt.txt"] == hashlib.sha256(b"Immutable prompt\n").hexdigest()


def test_freeze_is_immutable_and_rejects_tampered_spec(source, tmp_path):
    spec = build_run_spec(experiment(), source_root=source)
    out = tmp_path / "run"
    out.mkdir()
    write_run_spec(out, spec)
    assert write_run_spec(out, spec) == out / "run-spec.json"
    altered = deepcopy(spec)
    altered["protocol"]["manifest"]["defaults"]["budgets"]["inference_requests"] = 1
    with pytest.raises(ValueError, match="digest"):
        write_run_spec(out, altered)


def test_attempt_receipts_preserve_interventions_and_bind_candidate(source):
    spec = build_run_spec(experiment(), source_root=source)
    runtime = {"wheel_sha256": "b" * 64, "runtime_images": {"fidelity": "sha256:" + "c" * 64}}
    receipt = build_attempt_provenance(spec, attempt_id="attempt-1", condition_id="condition-a",
        runtime=runtime, candidate={"snapshot_sha256": "d" * 64},
        interventions=[{"kind": "collection_retry", "phase": "collection", "attempt": 1}])
    row = {"condition_id": "condition-a", "snapshot_sha256": "d" * 64,
           "execution_provenance": receipt, "simready_reward": 0}
    public = provenance_fields(row)
    assert public["execution_provenance"]["interventions"] == receipt["interventions"]
    assert public["protocol_sha256"] == spec["protocol_sha256"]
    assert row["simready_reward"] == 0
    changed = deepcopy(row)
    changed["execution_provenance"]["runtime"]["wheel_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="digest"):
        provenance_fields(changed)
    changed = deepcopy(row)
    changed["snapshot_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="candidate"):
        provenance_fields(changed)


def test_legacy_missing_provenance_is_explicit_and_never_fabricated():
    public = provenance_fields({"condition_id": "old", "simready_reward": 0})
    assert public["provenance_coverage"]["status"] == "legacy_unbound"
    assert "protocol_sha256" not in public
    assert "execution_provenance" not in public


def test_absolute_paths_and_credentials_do_not_escape(source):
    value = experiment()
    value["assets"][0]["closure"][0]["absolute_path"] = "/private/asset.usda"
    resolved = {"normalized_manifest": value, "operator_policy": {
        "providers": {"hf": {"kind": "huggingface", "base_url": "https://router.huggingface.co/v1",
                              "api_key": "secret", "api_key_env": "HF_TOKEN"}},
        "roots": {"assets": "/private"}, "limits": {"maximum_spend_usd": 50}}}
    spec = build_run_spec(resolved, source_root=source)
    raw = json.dumps(spec)
    assert "secret" not in raw and "/private" not in raw
    assert spec["protocol"]["limits"]["maximum_spend_usd"] == 50
    assert spec["protocol"]["providers"]["hf"]["kind"] == "huggingface"


def test_missing_or_malformed_content_never_claims_complete_coverage(source):
    value = experiment()
    del value["assets"][0]["closure"]
    spec = build_run_spec(value, source_root=source)
    assert "reference_content" in spec["coverage"]["missing"]
    assert spec["coverage"]["status"] == "partial"
    value["assets"][0]["closure"] = [{"path": "a.usda", "sha256": "invented", "size_bytes": 1}]
    with pytest.raises(ValueError, match="SHA-256"):
        build_run_spec(value, source_root=source)


def test_changed_attempt_wheel_cannot_inherit_the_original_protocol_without_proof(source):
    spec = build_run_spec(experiment(), runtime={"wheel_sha256": "a" * 64}, source_root=source)
    with pytest.raises(ValueError, match="approved byte-equivalent repair"):
        build_attempt_provenance(spec, attempt_id="a", condition_id="c",
                                 runtime={"wheel_sha256": "b" * 64})


def test_protocol_identity_does_not_depend_on_private_host_paths(source):
    value = {"normalized_manifest": experiment(), "conditions": [{
        "condition_id": "c", "configuration_sha256": "a" * 64,
        "asset": {"closure": [{"path": "a.usda", "absolute_path": "/host-one/a.usda",
                                 "sha256": "b" * 64, "size_bytes": 12}]}}]}
    first = build_run_spec(value, source_root=source)
    value["conditions"][0]["configuration_sha256"] = "c" * 64
    value["conditions"][0]["asset"]["closure"][0]["absolute_path"] = "/host-two/a.usda"
    second = build_run_spec(value, source_root=source)
    assert first["protocol_sha256"] == second["protocol_sha256"]


def test_frozen_protocol_binds_the_supplied_wheel_instead_of_importing_host(tmp_path):
    from simready_bench.benchmark import freeze_execution_spec
    from test_parallel import manifest

    selected = manifest()
    selected['assets'] = selected['assets'][:1]
    selected['repetitions'] = 1
    roots = {}
    for name in ('assets', 'skills', 'input_views', 'private_manifests'):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    specs = []
    for index, code in enumerate((b'REQUESTS = 64\n', b'REQUESTS = 1\n')):
        wheel = tmp_path / ('runtime-' + str(index) + '.whl')
        with zipfile.ZipFile(wheel, 'w') as archive:
            archive.writestr('simready_bench/config.py', code)
            archive.writestr('simready_bench/engine/pi.py', b'PROMPT = "frozen"\n')
        output = tmp_path / str(index)
        output.mkdir()
        specs.append(freeze_execution_spec(output, selected, {'roots': roots}, wheel=wheel))
        assert specs[-1]['protocol']['contract']['files']['config.py'] == hashlib.sha256(code).hexdigest()
        assert specs[-1]['protocol']['components']['prompt']['files']['engine/pi.py'] == hashlib.sha256(b'PROMPT = "frozen"\n').hexdigest()
    assert specs[0]['protocol_sha256'] != specs[1]['protocol_sha256']


@pytest.mark.parametrize('size', [256 * 1024**2 + 1, 1024**3 + 1])
def test_freeze_reference_uses_the_same_closure_bound_as_resolution(tmp_path, size):
    from simready_bench.benchmark import freeze_execution_spec
    from simready_bench.config import _closure
    from test_parallel import manifest

    selected = manifest()
    selected['assets'] = [{'id': 'reference', 'usd': 'asset.usda', 'dependencies': []}]
    selected['repetitions'] = 1
    roots = {}
    for name in ('assets', 'skills', 'input_views', 'private_manifests'):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    reference = tmp_path / 'assets/asset.usda'
    # Hash a real sparse file above the general-input bound without storing
    # hundreds of MiB of fixture data in the test checkout.
    with reference.open('wb') as stream:
        stream.truncate(size)
    output = tmp_path / 'output'
    output.mkdir()
    if size > 1024**3:
        with pytest.raises(ValueError, match='asset exceeds configured closure byte bound'):
            _closure(reference.parent, ['asset.usda'], 'asset')
        with pytest.raises(ValueError, match='asset exceeds configured closure byte bound'):
            freeze_execution_spec(output, selected, {'roots': roots})
    else:
        expected = _closure(reference.parent, ['asset.usda'], 'asset')
        spec = freeze_execution_spec(output, selected, {'roots': roots})
        asset = spec['protocol']['manifest']['assets'][0]
        assert asset['closure'][0]['size_bytes'] == size
        assert asset['closure'][0]['sha256'] == expected[0]['sha256']


@pytest.mark.parametrize('kind', ['skill', 'input view'])
def test_freeze_keeps_the_resolution_bound_for_other_inputs(tmp_path, kind):
    from simready_bench.benchmark import freeze_execution_spec
    from test_parallel import manifest

    selected = manifest()
    selected['assets'] = [{'id': 'reference', 'usd': 'asset.usda', 'dependencies': [], 'input_views': []}]
    selected['repetitions'] = 1
    roots = {}
    for name in ('assets', 'skills', 'input_views', 'private_manifests'):
        path = tmp_path / name
        path.mkdir()
        roots[name] = str(path)
    (tmp_path / 'assets/asset.usda').write_text('#usda 1.0\n')
    if kind == 'skill':
        oversized = tmp_path / 'skills/SKILL.md'
        selected['skill_sets'] = [{'id': 'fixture', 'paths': ['SKILL.md']}]
    else:
        oversized = tmp_path / 'input_views/thumbnail.png'
        selected['assets'][0]['input_views'] = ['thumbnail.png']
    with oversized.open('wb') as stream:
        stream.truncate(256 * 1024**2 + 1)
    output = tmp_path / 'output'
    output.mkdir()
    with pytest.raises(ValueError, match=kind + ' exceeds configured closure byte bound'):
        freeze_execution_spec(output, selected, {'roots': roots})


@pytest.mark.parametrize('name', ('../simready_bench/config.py', 'simready_bench/../config.py',
                                 '/simready_bench/config.py', 'simready_bench\\config.py'))
def test_wheel_contract_rejects_unconfined_members(tmp_path, name):
    from simready_bench.provenance import package_contract_from_wheel

    wheel = tmp_path / 'runtime.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('simready_bench/config.py', b'GOOD = True\n')
        archive.writestr(name, b'BAD = True\n')
    with pytest.raises(ValueError, match='confined'):
        package_contract_from_wheel(wheel)


def test_wheel_contract_rejects_duplicate_members(tmp_path):
    from simready_bench.provenance import package_contract_from_wheel

    wheel = tmp_path / 'runtime.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('simready_bench/config.py', b'ONE = 1\n')
        with pytest.warns(UserWarning, match='Duplicate name'):
            archive.writestr('simready_bench/config.py', b'ONE = 2\n')
    with pytest.raises(ValueError, match='unique'):
        package_contract_from_wheel(wheel)


def test_wheel_contract_rejects_relocated_package_collisions(tmp_path):
    from simready_bench.provenance import package_contract_from_wheel

    wheel = tmp_path / 'runtime.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('simready_bench/config.py', b'ONE = 1\n')
        archive.writestr('runtime.data/purelib/simready_bench/config.py', b'ONE = 2\n')
    with pytest.raises(ValueError, match='unique installed'):
        package_contract_from_wheel(wheel)


def test_frozen_wheel_cannot_change_on_resume(tmp_path):
    from simready_bench.provenance import package_contract_from_wheel

    wheel = tmp_path / 'runtime.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('simready_bench/config.py', b'ONE = 1\n')
        archive.writestr('runtime.dist-info/METADATA', b'Requires-Dist: first\n')
    output = tmp_path / 'run'
    output.mkdir()
    original = build_run_spec(experiment(), wheel=wheel)
    write_run_spec(output, original)
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('simready_bench/config.py', b'ONE = 1\n')
        archive.writestr('runtime.dist-info/METADATA', b'Requires-Dist: second\n')
    changed = build_run_spec(experiment(), wheel=wheel)
    assert original['protocol_sha256'] != changed['protocol_sha256']
    with pytest.raises(ValueError, match='cannot change'):
        write_run_spec(output, changed)


def test_attempt_source_cannot_change_after_receipt_is_frozen(source, tmp_path):
    from simready_bench.provenance import load_attempt_provenance, write_attempt_provenance

    spec = build_run_spec(experiment(), source_root=source)
    receipt = build_attempt_provenance(spec, attempt_id='a', condition_id='c', runtime={})
    identity = {'attempt': 'executions/001', 'condition_id': 'c', 'measurement_sha256': 'a' * 64}
    write_attempt_provenance(tmp_path, source=identity, receipt=receipt)
    assert load_attempt_provenance(tmp_path, source=identity) == receipt
    changed = {**identity, 'measurement_sha256': 'b' * 64}
    with pytest.raises(ValueError, match='immutable provenance source'):
        load_attempt_provenance(tmp_path, source=changed)


def test_operational_history_cannot_remove_or_rewrite_previous_entries(source, tmp_path):
    from simready_bench.provenance import collect_operational_provenance

    spec = build_run_spec(experiment(), source_root=source)
    folder = tmp_path / 'operational-interventions'
    folder.mkdir()
    first = folder / '001.json'
    first.write_text('{"kind":"stop"}')
    original = collect_operational_provenance(tmp_path, spec)
    second = folder / '002.json'
    second.write_text('{"kind":"resume"}')
    appended = collect_operational_provenance(tmp_path, spec)
    assert appended['entries'][:1] == original['entries']
    first.write_text('{"kind":"changed"}')
    with pytest.raises(ValueError, match='cannot be removed or rewritten'):
        collect_operational_provenance(tmp_path, spec)
    first.unlink()
    with pytest.raises(ValueError, match='cannot be removed or rewritten'):
        collect_operational_provenance(tmp_path, spec)
