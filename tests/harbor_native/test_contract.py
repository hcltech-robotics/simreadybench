import json
import math
import subprocess
import sys

import pytest


def known(value):
    return {"status": "measured", "score_percent": value, "coverage_fraction": 1}


def test_atomic_verifier_outputs_are_readable_by_the_host_collector(tmp_path):
    import os
    import stat
    from simready_eval.contract import write_outputs

    report = {'scores': {name: known(100) for name in ('formal', 'visual', 'kinematic')}}
    (tmp_path / 'reward.txt').write_text('stale scalar')
    previous = os.umask(0o077)
    try:
        reward = write_outputs(report, tmp_path)
    finally:
        os.umask(previous)
    for name in ('report.json', 'reward.json'):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o644
    assert json.loads((tmp_path / 'reward.json').read_text()) == reward
    assert not (tmp_path / 'reward.txt').exists()


def test_visual_requires_real_geometry_and_four_view_appearance():
    from simready_eval.contract import visual_score

    assert visual_score(known(9), known(100))["score_percent"] == 30
    assert visual_score(known(100), known(25))["score_percent"] == 50
    assert visual_score(known(0), known(100))["score_percent"] == 0


def test_missing_view_keeps_unknown_and_bounds_instead_of_shape_only():
    from simready_eval.contract import visual_score

    partial = {"status": "unknown", "score_percent": None,
               "coverage_fraction": .75, "lower_bound_percent": 60,
               "upper_bound_percent": 85}
    result = visual_score(known(100), partial)
    assert result["score_percent"] is None
    assert result["coverage_fraction"] == .875
    assert result["lower_bound_percent"] == pytest.approx(math.sqrt(6000))
    assert result["upper_bound_percent"] == pytest.approx(math.sqrt(8500))


def test_known_zero_geometry_stays_in_visual_mean_with_unavailable_appearance():
    from simready_eval.contract import component, reward_record, visual_score
    from simready_eval.metric import aggregate

    appearance = component(reason="no_visible_candidate_surface")
    visual = visual_score(known(0), appearance)
    assert visual["status"] == "measured"
    assert visual["score_percent"] == 0
    assert visual["coverage_fraction"] == .5
    assert visual["factors"]["rendered_appearance"]["status"] == "unknown"
    reward = reward_record({"scores": {"formal": known(0), "visual": visual,
                                        "kinematic": known(0)}})
    metrics = aggregate([reward])
    assert metrics["visual_known_count"] == 1
    assert metrics["visual_fidelity_mean_known"] == 0


def test_partial_zero_observation_does_not_prove_visual_zero():
    from simready_eval.contract import component, visual_score

    result = visual_score(component(0, coverage=.5), component())
    assert result["status"] == "unknown"


def test_rewards_preserve_partial_knowns_and_explicit_unknowns():
    from simready_eval.contract import reward_record
    from simready_eval.metric import aggregate

    report = {"scores": {"formal": known(80), "visual": {
        "status": "unknown", "score_percent": None, "coverage_fraction": .6,
        "lower_bound_percent": 20, "upper_bound_percent": 70},
        "kinematic": known(75)}, "candidate_invalid": False}
    reward = reward_record(report)
    assert reward["visual_known"] == 0
    assert reward["formal_known_credit"] == .8
    result = aggregate([reward, None])
    assert result["formal_fidelity_mean_known"] == 80
    assert result["formal_known_fraction"] == .5
    assert result["visual_mean_available"] == 0
    assert "visual_fidelity_mean_known" not in result
    assert result["visual_lower_bound_mean"] == 10
    assert result["visual_upper_bound_mean"] == 85
    assert result["unavailable_verifier_count"] == 1


def test_metric_does_not_treat_malformed_reward_as_candidate_failure(tmp_path):
    from simready_eval.contract import reward_record
    from simready_eval.metric import aggregate

    good = reward_record({"scores": {name: known(0) for name in
                          ("formal", "visual", "kinematic")}, "candidate_invalid": True})
    bad = {**good, "formal_known_credit": float("nan")}
    result = aggregate([good, bad, {"formal_known_credit": .8}])
    assert result["candidate_invalid_count"] == 1
    assert result["invalid_contract_count"] == 2
    assert result["formal_known_count"] == 1
    assert result["formal_unknown_count"] == 2
    assert result["formal_upper_bound_mean"] == pytest.approx(200 / 3)


def test_metric_script_is_standalone_jsonl_interface(tmp_path):
    import simready_eval.metric as metric

    script = tmp_path / "metric.py"
    script.write_bytes(__import__("pathlib").Path(metric.__file__).read_bytes())
    inputs, outputs = tmp_path / "rewards.jsonl", tmp_path / "metric.json"
    inputs.write_text("null\n")
    process = subprocess.run([sys.executable, "-I", str(script), "-i", str(inputs),
                              "-o", str(outputs)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(outputs.read_text())["formal_upper_bound_mean"] == 100
