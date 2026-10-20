import os
import pytest


@pytest.mark.skipif(os.name == "nt", reason="Linux RAM workspace and file ownership")
def test_installed_profile_binds_only_the_current_package_and_nvidia_profile():
    from simready_bench.policy import installed_profile
    from simready_bench.engine.environment import verify_profile_code

    profile, source = installed_profile()
    assert profile["version"] == "1.0.0"
    assert all(path.startswith("src/simready_bench/") for path in profile["code_files"])
    assert "src/simready_bench/engine/verifier.py" in profile["code_files"]
    assert "src/simready_bench/engine/nvidia_profile.json" in profile["code_files"]
    assert verify_profile_code(profile)["files"] == len(profile["code_files"])
