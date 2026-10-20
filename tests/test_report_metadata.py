from copy import deepcopy
import csv

import pytest

from simready_bench.reporting import compact_report, score_partials, write_scores


@pytest.mark.parametrize("passed", [0, 45, 46])
def test_known_mandatory_bounds_equal_the_measured_score_without_mutating_inputs(
    tmp_path, passed
):
    base = 95 * passed / 46
    bonus = 5 if passed == 46 else 0
    primary = {
        "status": "scored",
        "passed_count": passed,
        "failed_count": 46 - passed,
        "unknown_count": 0,
        "applicable_count": 46,
        "score_percent": base + bonus,
        "score_lower_bound": base,
        "score_upper_bound": base + bonus,
        "completion_bonus": bonus,
    }
    value = {
        "schema": "simready-bench-portable-report-v1",
        "results": [{"simready": primary, "simready_reward": base + bonus}],
    }
    original = deepcopy(value)
    result = compact_report(value)
    row = result["results"][0]
    assert row["simready"]["score_lower_bound"] == base + bonus
    assert row["simready"]["score_upper_bound"] == base + bonus
    assert row["simready"]["score_percent"] == base + bonus
    assert row["simready"]["completion_bonus"] == bonus
    assert value == original
    write_scores(tmp_path / "scores.csv", value["results"])
    written = next(csv.DictReader((tmp_path / "scores.csv").open(newline="")))
    assert float(written["mandatory_lower_bound_percent"]) == base + bonus
    assert float(written["mandatory_upper_bound_percent"]) == base + bonus


@pytest.mark.parametrize("passed,failed,unknown", [(45, 0, 1), (33, 3, 10)])
def test_unknown_mandatory_bounds_and_bonus_math_remain_unchanged(
    passed, failed, unknown
):
    primary = {
        "score_percent": None,
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "applicable_count": 46,
        "score_lower_bound": 95 * passed / 46,
        "score_upper_bound": 95 * (passed + unknown) / 46 + (5 if not failed else 0),
        "completion_bonus": 0,
    }
    value = {
        "schema": "simready-bench-portable-report-v1",
        "results": [{"simready": primary}],
    }
    assert compact_report(value)["results"][0]["simready"] == primary
    partials = score_partials(value["results"][0])
    assert partials["mandatory_lower_bound_percent"] == primary["score_lower_bound"]
    assert partials["mandatory_upper_bound_percent"] == primary["score_upper_bound"]


@pytest.mark.parametrize("unknown", [None, False])
def test_missing_or_invalid_unknown_count_does_not_assert_complete_measurement(unknown):
    primary = {
        "score_percent": 100,
        "unknown_count": unknown,
        "score_lower_bound": 95,
        "score_upper_bound": 100,
    }
    assert score_partials({"simready": primary})["mandatory_lower_bound_percent"] == 95
