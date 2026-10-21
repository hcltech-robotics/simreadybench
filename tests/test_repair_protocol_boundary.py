import hashlib
import zipfile

import pytest

from simready_bench.executor_repair import approve_executor_repair, repaired_wheel


def wheel(path, content, *, compression=zipfile.ZIP_STORED):
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, raw in content.items():
            archive.writestr(name, raw)
    return path


@pytest.mark.parametrize("module, before, after", [
    ("daytona.py", b"requests = conditions * 64\n", b"requests = 1\n"),
    ("engine/inference_gateway.py", b"validate_price()\n", b"pass\n"),
    ("engine/actor_relay.py", b"check_budget()\n", b"pass\n"),
    ("engine/hf_pricing.py", b"async def fetch_hf_route():\n    return await validate()\n",
     b"async def fetch_hf_route():\n    return {'price': 0}\n"),
])
def test_no_module_allowlist_can_claim_protocol_equivalence(tmp_path, module, before, after):
    name = "simready_bench/" + module
    original = wheel(tmp_path / "original.whl", {name: before})
    changed = wheel(tmp_path / "changed.whl", {name: after})
    with pytest.raises(ValueError, match="scientific revision"):
        approve_executor_repair(tmp_path, original, changed, reason="Transport repair")


def test_explicit_scientific_revision_cannot_resume_as_equivalent(tmp_path):
    name = "simready_bench/daytona.py"
    original = wheel(tmp_path / "original.whl", {name: b"requests = 64\n"})
    changed = wheel(tmp_path / "changed.whl", {name: b"requests = 1\n"})
    receipt = approve_executor_repair(tmp_path, original, changed, reason="New protocol",
                                      scientific_revision="bounded-tools-v2")
    assert receipt["protocol_equivalent"] is False
    assert receipt["protected_files_identical"] is False
    assert receipt["requires_new_run_spec"] is True
    assert receipt["original_contract_sha256"] != receipt["replacement_contract_sha256"]
    with pytest.raises(ValueError, match="new run specification"):
        repaired_wheel(tmp_path, {hashlib.sha256(original.read_bytes()).hexdigest()})


def test_only_identical_package_contract_can_resume_equivalently(tmp_path):
    content = {"simready_bench/daytona.py": b"requests = 64\n" * 100}
    original = wheel(tmp_path / "original.whl", content)
    changed = wheel(tmp_path / "changed.whl", content, compression=zipfile.ZIP_DEFLATED)
    receipt = approve_executor_repair(tmp_path, original, changed, reason="Repackage identical files")
    assert receipt["protocol_equivalent"] is True
    assert receipt["protected_files_identical"] is True
    assert receipt["original_contract_sha256"] == receipt["replacement_contract_sha256"]
    assert repaired_wheel(tmp_path, {hashlib.sha256(original.read_bytes()).hexdigest()}) == changed


def test_dependency_metadata_is_part_of_the_immutable_contract(tmp_path):
    original = wheel(tmp_path / "original.whl", {"bench.dist-info/METADATA": b"Requires-Dist: safe==1\n"})
    changed = wheel(tmp_path / "changed.whl", {"bench.dist-info/METADATA": b"Requires-Dist: unsafe\n"})
    with pytest.raises(ValueError, match="scientific revision"):
        approve_executor_repair(tmp_path, original, changed, reason="Dependency repair")


def test_pricing_wrapper_cannot_skip_an_unchanged_validator(tmp_path):
    import inspect
    from simready_bench.engine import hf_pricing

    source = inspect.getsource(hf_pricing)
    start = source.index("async def fetch_hf_route(")
    helper = source.index("async def _fetch_hf_route_once(")
    bypass = source[:start] + (
        "async def fetch_hf_route(*args, **kwargs):\n"
        "    return {'input_cost_per_million': 0, 'output_cost_per_million': 0}\n\n"
    ) + source[helper:]
    name = "simready_bench/engine/hf_pricing.py"
    original = wheel(tmp_path / "original.whl", {name: source})
    changed = wheel(tmp_path / "changed.whl", {name: bypass})
    with pytest.raises(ValueError, match="scientific revision"):
        approve_executor_repair(tmp_path, original, changed, reason="Pricing retry wrapper")
