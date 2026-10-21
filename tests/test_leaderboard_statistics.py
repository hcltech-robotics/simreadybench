import copy

import pytest

from simready_bench.leaderboard_statistics import (
    optional_summary_with_ci,
    statistical_summaries,
    success_summary,
)


def measurements(values=(10, 20, 30)):
    rows, parts, usage = [], [], []
    for asset, value in enumerate(values):
        for repetition in range(4):
            rows.append({"asset_id": str(asset), "repetition": repetition})
            parts.append({"eligible": 1, "score_percent": value})
            usage.append({"total_tokens": value * 1000, "elapsed_seconds": value * 2})
    return (
        rows,
        {"formal": parts},
        {"formal": {"score": sum(values) / len(values)}},
        usage,
    )


def summary(data):
    rows, parts, summaries, usage_rows = data
    return statistical_summaries(
        rows, parts=parts, summaries=summaries, usage_rows=usage_rows
    )


def test_asset_clusters_preserve_all_repetitions_and_pair_resource_ratios():
    data = measurements()
    composites, efficiency = summary(data)
    interval = composites["formal"]["confidence_interval"]
    assert interval["asset_count"] == 3
    assert interval["lower"] == 10
    assert interval["upper"] == 30
    assert composites["formal"]["score"] == 20
    assert data[2] == {"formal": {"score": 20}}
    for name, expected in (("per_million_tokens", 1000), ("per_elapsed_minute", 30)):
        metric = efficiency["formal"][name]
        assert metric["value"] == expected
        assert metric["confidence_interval"]["lower"] == pytest.approx(expected)
        assert metric["confidence_interval"]["upper"] == pytest.approx(expected)
        assert metric["eligible_attempts"] == metric["measured_attempts"] == 12
        assert metric["eligible_assets"] == 3


def test_repeated_attempts_are_not_bootstrapped_as_independent_assets():
    composites, _ = summary(measurements((0, 100)))
    interval = composites["formal"]["confidence_interval"]
    assert interval["lower"] == 0
    assert interval["upper"] == 100
    assert interval["asset_count"] == 2


def test_row_order_does_not_change_deterministic_bootstrap():
    original = measurements()
    shuffled = copy.deepcopy(original)
    shuffled[0].reverse()
    shuffled[1]["formal"].reverse()
    shuffled[3].reverse()
    assert summary(original) == summary(shuffled)


def test_reference_inapplicable_resources_are_excluded_from_each_metric():
    data = measurements()
    for index in range(4):
        data[1]["formal"][index] = {"eligible": 0, "score_percent": None}
        data[3][index] = {}
    data[2]["formal"]["score"] = 25
    composites, efficiency = summary(data)
    assert composites["formal"]["confidence_interval"]["asset_count"] == 2
    metric = efficiency["formal"]["per_million_tokens"]
    assert metric["value"] == 1000
    assert metric["measured_attempts"] == metric["eligible_attempts"] == 8
    assert metric["total_attempts"] == 12
    assert metric["denominator_mean"] == 25000


def test_missing_resource_keeps_coverage_and_unknown_efficiency():
    data = measurements()
    data[3][0]["total_tokens"] = None
    composites, efficiency = summary(data)
    assert composites["formal"]["confidence_interval"]["lower"] == 10
    token = efficiency["formal"]["per_million_tokens"]
    assert token["measured_attempts"] == 11
    assert token["value"] is None
    assert token["confidence_interval"] is None
    assert token["denominator_mean"] is None
    assert efficiency["formal"]["per_elapsed_minute"]["value"] == 30


@pytest.mark.parametrize("score", [None, float("nan")])
def test_missing_score_never_gets_a_confidence_interval_or_efficiency(score):
    data = measurements()
    data[2]["formal"]["score"] = score
    data[1]["formal"][0]["score_percent"] = None
    composites, efficiency = summary(data)
    assert composites["formal"]["confidence_interval"] is None
    assert all(metric["value"] is None for metric in efficiency["formal"].values())


def test_single_cluster_has_no_claimed_sampling_interval():
    composites, efficiency = summary(measurements((10,)))
    assert composites["formal"]["confidence_interval"]["lower"] is None
    assert efficiency["formal"]["per_million_tokens"]["value"] == 1000
    assert (
        efficiency["formal"]["per_million_tokens"]["confidence_interval"]["upper"]
        is None
    )


def test_zero_resource_means_do_not_make_infinite_efficiency():
    data = measurements()
    for usage in data[3]:
        usage.update(total_tokens=0, elapsed_seconds=0)
    _, efficiency = summary(data)
    for metric in efficiency["formal"].values():
        assert metric["value"] is None
        assert metric["denominator_mean"] == 0
        assert metric["measured_attempts"] == 12


def test_no_eligible_attempts_remain_not_applicable():
    data = measurements()
    data[2]["formal"]["score"] = None
    for part in data[1]["formal"]:
        part.update(eligible=0, score_percent=None)
    composites, efficiency = summary(data)
    assert composites["formal"]["eligible_assets"] == 0
    assert composites["formal"]["confidence_interval"] is None
    assert efficiency["formal"]["per_million_tokens"]["eligible_attempts"] == 0


def test_optional_backfill_retains_separate_score_and_counts_with_cluster_ci():
    rows, _, _, _ = measurements((0, 100))
    for row in rows:
        row["optional_score"] = {"score_percent": 0 if row["asset_id"] == "0" else 100}
    original = {"score": 50, "passed": 204, "eligible": 408, "coverage": 100}
    enriched = optional_summary_with_ci(rows, original)
    assert {key: enriched[key] for key in original} == original
    assert original == {"score": 50, "passed": 204, "eligible": 408, "coverage": 100}
    assert enriched["confidence_interval"]["lower"] == 0
    assert enriched["confidence_interval"]["upper"] == 100
    assert enriched["confidence_interval"]["asset_count"] == 2
    rows.reverse()
    assert optional_summary_with_ci(rows, original) == enriched


def test_optional_missing_measurements_never_produce_a_ci():
    rows, _, _, _ = measurements((10, 20))
    for row in rows:
        row["evaluation"] = {"optional_score": {"score_percent": 50}}
    assert (
        optional_summary_with_ci(rows, {"score": 50})["confidence_interval"] is not None
    )
    rows[0]["evaluation"]["optional_score"]["score_percent"] = None
    assert (
        optional_summary_with_ci(rows, {"score": None})["confidence_interval"] is None
    )
    assert optional_summary_with_ci(rows, {"score": 50})["confidence_interval"] is None


def test_mandatory_pass_rates_distinguish_attempts_from_four_try_assets():
    rows, _, _, _ = measurements((10, 20))
    for row in rows:
        passed = 46 if row["asset_id"] == "0" and row["repetition"] == 0 else 40
        row["simready"] = {
            "applicable_count": 46,
            "passed_count": passed,
            "failed_count": 46 - passed,
            "unknown_count": 0,
        }
    successes = success_summary(rows)
    assert successes["mandatory_pass_per_attempt"] == {
        "percent": 12.5,
        "passed": 1,
        "measured": 8,
        "total": 8,
    }
    assert successes["mandatory_pass_within_four"] == {
        "percent": 50,
        "passed": 1,
        "measured": 2,
        "total": 2,
    }
    rows[-1]["simready"].update(passed_count=40, failed_count=0, unknown_count=6)
    successes = success_summary(rows)
    assert successes["mandatory_pass_per_attempt"]["percent"] is None
    assert successes["mandatory_pass_per_attempt"]["measured"] == 7
    assert successes["mandatory_pass_within_four"]["percent"] is None
