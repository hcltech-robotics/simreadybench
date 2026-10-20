import pytest
from simready_bench.capabilities import normalize_native_harness


def test_manifest_native_selection_preserves_scope_admission():
    assert (
        normalize_native_harness(
            {"id": "native-pi", "adapter": "harbor", "agent": "pi"}
        )["adapter"]
        == "pi"
    )
    with pytest.raises(ValueError, match="project_policy_codex_is_builder_only"):
        normalize_native_harness(
            {"id": "native-codex", "adapter": "harbor", "agent": "codex"}
        )
    with pytest.raises(ValueError, match="Unknown"):
        normalize_native_harness(
            {"id": "fake", "adapter": "harbor", "agent": "imaginary-agent"}
        )
