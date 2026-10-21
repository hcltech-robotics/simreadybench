import os

import pytest

from simready_bench.benchmark import compose_experiment, select_model


def pytest_addoption(parser):
    for tier in ("licensed", "container", "paid"):
        parser.addoption("--run-" + tier, action="store_true", default=False,
                         help="Explicitly enable the local " + tier + " acceptance tier")


def pytest_collection_modifyitems(config, items):
    for item in items:
        for tier in ("licensed", "container", "paid"):
            if item.get_closest_marker(tier) is None:
                continue
            if os.environ.get("CI") or not config.getoption("--run-" + tier):
                item.add_marker(pytest.mark.skip(
                    reason=tier + " acceptance requires explicit local opt-in"))


@pytest.fixture
def setup(tmp_path):
    roots = {}
    for name in ("assets", "input_views", "skills", "private_manifests"):
        directory = tmp_path / name
        directory.mkdir()
        roots[name] = str(directory)
    (tmp_path / "assets/cup.usda").write_text('#usda 1.0\ndef Xform "Asset" {}\n')
    skill = tmp_path / "skills/basic"
    skill.mkdir()
    (skill / "SKILL.md").write_text("Author USD using Python/PXR.")
    manifest = compose_experiment(
        {"assets": [{"id": "cup", "usd": "cup.usda"}]},
        model=select_model("gpt-6.1-sol", provider="openai")[0],
        maximum_spend_usd=0,
        repetitions=1,
        authoring_turns=32,
    )
    policy = {
        "roots": roots,
        "limits": {
            "maximum_spend_usd": 0,
            "maximum_matrix_runs": 125,
            "inference_requests": 32,
            "max_input_tokens": None,
            "max_output_tokens": None,
        },
        "providers": {
            "selected": {
                "kind": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "TEST_UNUSED_KEY",
            }
        },
    }
    return tmp_path, manifest, policy
