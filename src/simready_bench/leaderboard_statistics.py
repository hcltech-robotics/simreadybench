"""Asset-cluster uncertainty and resource efficiency for complete evaluations.

The sampling unit is the reference asset. Each bootstrap draw retains all of
an asset's reconstructions, rather than treating the four repetitions as
independent assets. Headline means retain the original attempt weighting.
Resource ratios use the same eligible attempts as their corresponding score.
"""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


RESAMPLES = 10_000
SEED = 20_261_020


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _interval(samples, *, assets):
    interval = {
        "level": 0.95,
        "lower": None,
        "upper": None,
        "method": "asset_cluster_bootstrap_percentile",
        "resamples": RESAMPLES,
        "seed": SEED,
        "asset_count": assets,
    }
    if assets >= 2 and np.isfinite(samples).all():
        lower, upper = np.quantile(samples, [0.025, 0.975], method="linear")
        interval.update(lower=float(lower), upper=float(upper))
    return interval


def statistical_summaries(rows, *, parts, summaries, usage_rows):
    """Add paired bootstrap intervals without changing any original mean.

    ``parts[name]`` is the original composite per attempt. Unknown headline
    scores remain unknown. Reference-inapplicable attempts contribute neither
    score nor resource use to that metric's efficiency denominator.
    """
    enriched, efficiency = {}, {}
    for name, summary in summaries.items():
        summary = dict(summary)
        selected = [
            index
            for index, part in enumerate(parts[name])
            if _number(part.get("eligible")) and part["eligible"] > 0
        ]
        clusters = defaultdict(list)
        for index in selected:
            clusters[rows[index]["asset_id"]].append(index)
        asset_ids = sorted(clusters)
        asset_count = len(asset_ids)
        summary["confidence_interval"] = None
        summary["eligible_assets"] = asset_count
        groups = [clusters[asset] for asset in asset_ids]
        counts = np.array([len(group) for group in groups], dtype=float)
        draws = (
            np.random.default_rng(SEED).integers(
                asset_count, size=(RESAMPLES, asset_count)
            )
            if asset_count
            else np.empty((RESAMPLES, 0), dtype=int)
        )
        draw_counts = counts[draws].sum(axis=1)
        known = _number(summary.get("score")) and bool(selected)
        score_totals = (
            np.array(
                [
                    sum(parts[name][index]["score_percent"] for index in group)
                    for group in groups
                ],
                dtype=float,
            )
            if known
            else None
        )
        score_samples = score_totals[draws].sum(axis=1) / draw_counts if known else None
        if known:
            summary["confidence_interval"] = _interval(
                score_samples, assets=asset_count
            )
        enriched[name] = summary
        efficiency[name] = {}
        for key, resource, multiplier, unit in (
            (
                "per_million_tokens",
                "total_tokens",
                1_000_000,
                "fidelity points per million tokens",
            ),
            (
                "per_elapsed_minute",
                "elapsed_seconds",
                60,
                "fidelity points per elapsed minute",
            ),
        ):
            values = [usage_rows[index].get(resource) for index in selected]
            measured = sum(_number(value) and value >= 0 for value in values)
            complete = bool(values) and measured == len(values)
            denominator_mean = sum(values) / len(values) if complete else None
            metric = {
                "value": None,
                "status": "not_applicable"
                if summary.get("status") == "not_applicable"
                else "unknown",
                "unit": unit,
                "confidence_interval": None,
                "measured_attempts": measured,
                "eligible_attempts": len(selected),
                "eligible_assets": asset_count,
                "total_attempts": len(rows),
                "denominator": "mean_" + resource + "_per_eligible_attempt",
                "denominator_mean": denominator_mean,
                "multiplier": multiplier,
            }
            if known and complete and denominator_mean > 0:
                metric["value"] = multiplier * summary["score"] / denominator_mean
                metric["status"] = "measured"
                resource_totals = np.array(
                    [
                        sum(usage_rows[index][resource] for index in group)
                        for group in groups
                    ],
                    dtype=float,
                )
                resource_samples = resource_totals[draws].sum(axis=1)
                # Both totals use the identical sampled asset clusters. A zero
                # resource draw makes a finite percentile interval undefined.
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = (
                        multiplier * score_totals[draws].sum(axis=1) / resource_samples
                    )
                metric["confidence_interval"] = _interval(ratios, assets=asset_count)
            efficiency[name][key] = metric
    return enriched, efficiency


def success_summary(rows):
    """Report strict mandatory pass rates across attempts and four-try assets."""
    groups = defaultdict(list)
    for row in rows:
        primary = (
            row.get("simready") or (row.get("evaluation") or {}).get("simready") or {}
        )
        passed, failed, unknown = (
            primary.get(key)
            for key in ("passed_count", "failed_count", "unknown_count")
        )
        complete = (
            primary.get("applicable_count") == 46
            and all(
                type(value) is int and value >= 0 for value in (passed, failed, unknown)
            )
            and passed + failed + unknown == 46
        )
        outcome = passed == 46 if complete and (unknown == 0 or failed > 0) else None
        groups[row["asset_id"]].append(outcome)
    attempts = [outcome for group in groups.values() for outcome in group]
    assets = [
        True
        if any(outcome is True for outcome in group)
        else False
        if all(outcome is False for outcome in group)
        else None
        for group in groups.values()
    ]

    def rate(values):
        measured = sum(value is not None for value in values)
        return {
            "percent": 100 * sum(value is True for value in values) / len(values)
            if measured == len(values)
            else None,
            "passed": sum(value is True for value in values),
            "measured": measured,
            "total": len(values),
        }

    return {
        "mandatory_pass_per_attempt": rate(attempts),
        "mandatory_pass_within_four": rate(assets),
    }
