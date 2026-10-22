"""Summarise complete public NV25 evaluations for the project leaderboard."""

from collections import Counter
from importlib.resources import files
import json
import math
from statistics import mean
from simready_bench.composites import (
    NAMES as COMPOSITE_NAMES,
    scores as composite_scores,
)
from simready_bench.reporting import USAGE_FIELDS, usage_columns
from simready_bench.leaderboard_statistics import (
    optional_summary_with_ci,
    statistical_summaries,
    success_summary,
)


COMPONENTS = ("shape", "rigid_parts", "joints", "materials")


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _score(value):
    return _number(value) and 0 <= value <= 100


def _total(values):
    return (
        sum(values) if all(_number(value) and value >= 0 for value in values) else None
    )


def _mean(values):
    return mean(values) if values and all(_number(value) for value in values) else None


def _profile_summary(parts, scores, *, count, optional=False):
    for part in parts:
        if part.get("applicable_count") not in (None, count):
            raise ValueError("The complete pinned profile denominator is required")
    passed = _total([part.get("passed_count") for part in parts])
    failed = _total([part.get("failed_count") for part in parts])
    unknown = _total([part.get("unknown_count") for part in parts])
    eligible = _total([part.get("applicable_count") for part in parts])
    measured = passed + failed if passed is not None and failed is not None else None
    bounds = (
        ("lower_bound_percent", "upper_bound_percent")
        if optional
        else ("score_lower_bound", "score_upper_bound")
    )
    summary = {
        "score": _mean(scores) if all(_score(score) for score in scores) else None,
        "passed": passed,
        "failed": failed,
        "unknown": unknown,
        "eligible": eligible,
        "measured": measured,
        "coverage": (
            100 * measured / eligible if measured is not None and eligible else None
        ),
        "lower_bound": _mean([part.get(bounds[0]) for part in parts]),
        "upper_bound": _mean([part.get(bounds[1]) for part in parts]),
        "measured_attempts": sum(_score(score) for score in scores),
        "unknown_attempts": sum(not _score(score) for score in scores),
        "reported_attempts": sum(bool(part) for part in parts),
    }
    if not optional:
        summary["completion_bonus"] = _mean(
            [part.get("completion_bonus") for part in parts]
        )
    return summary


def _fidelity_summary(parts):
    eligible = [
        part for part in parts if _number(part.get("eligible")) and part["eligible"] > 0
    ]
    inapplicable = sum(
        _number(part.get("eligible")) and part["eligible"] == 0 for part in parts
    )
    unknown_eligibility = len(parts) - len(eligible) - inapplicable
    scores = [part.get("score_percent") for part in eligible]
    measured_attempts = sum(_score(score) for score in scores)
    coverage = [part.get("coverage_fraction") for part in eligible]
    counts_known = not unknown_eligibility
    total = _total([part.get("eligible") for part in parts])
    credit = _total([part.get("earned") for part in eligible]) if counts_known else None
    measured = (
        _total([part.get("measured") for part in eligible]) if counts_known else None
    )
    score = (
        _mean(scores)
        if counts_known and all(_score(value) for value in scores)
        else None
    )
    return {
        "score": score,
        "status": (
            "not_applicable"
            if inapplicable == len(parts)
            else "measured"
            if score is not None
            else "unknown"
        ),
        "earned": credit,
        "eligible": total,
        "measured": measured,
        "coverage": (
            100 * mean(coverage)
            if counts_known
            and coverage
            and all(_number(value) and 0 <= value <= 1 for value in coverage)
            else 0
            if inapplicable == len(parts)
            else None
        ),
        "measurement_coverage": (
            100 * measured / total
            if measured is not None and total
            else 0
            if total == 0
            else None
        ),
        "eligible_attempts": len(eligible),
        "measured_attempts": measured_attempts,
        "not_applicable_attempts": inapplicable,
        "unknown_eligibility_attempts": unknown_eligibility,
        "unknown_attempts": len(eligible) - measured_attempts + unknown_eligibility,
        "lower_bound": _mean(
            [
                part.get("lower_bound_percent", part.get("score_percent"))
                for part in eligible
            ]
        ),
        "upper_bound": _mean(
            [
                part.get("upper_bound_percent", part.get("score_percent"))
                for part in eligible
            ]
        ),
    }


def _reference_pin(manifest, reference):
    source = manifest.get("source")
    freeze = manifest.get("benchmark_freeze")
    expected = reference["source"]
    if freeze is not None and freeze != reference["benchmark_freeze"]:
        raise ValueError("An NV25 entry must retain the packaged reference freeze")
    if source is None:
        if freeze is None:
            raise ValueError(
                "An NV25 entry requires its pinned reference source or freeze"
            )
    elif (
        not isinstance(source, dict)
        or source.get("kind") != expected["kind"]
        or source.get("revision") != expected["revision"]
        or not isinstance(source.get("packages"), list)
        or not all(isinstance(package, str) for package in source["packages"])
        or len(source["packages"]) != len(expected["packages"])
        or set(source["packages"]) != set(expected["packages"])
    ):
        raise ValueError("An NV25 entry must retain the pinned public reference source")
    assets = {asset["id"]: asset for asset in reference["assets"]}
    if any(
        "usd" in asset and asset["usd"] != assets[asset["id"]]["usd"]
        for asset in manifest["assets"]
    ):
        raise ValueError("An NV25 entry cannot replace a reference entrypoint")


def _generation(row, manifest, model):
    generation = {
        **model.get("generation", {}),
        **manifest.get("defaults", {}).get("generation", {}),
    }
    labels = {
        "asset": row["asset_id"],
        "harness": row["harness_id"],
        "model": model["id"],
        "skill_set": row["skill_set_id"],
    }
    for override in manifest.get("overrides", []):
        match = override.get("match", {})
        if set(match) - labels.keys():
            raise ValueError("A leaderboard override changes its declared factors")
        if all(value == "*" or labels[key] == value for key, value in match.items()):
            generation.update(override.get("settings", {}).get("generation", {}))
    return generation


def _output_policy(row, manifest, model, protocol):
    from simready_bench.engine.native_output import authoring_protocol, authoring_turns

    generation = _generation(row, manifest, model)
    cap = generation.get("max_output_tokens")
    limit = generation.get("output_limit")
    if protocol is None:
        if cap is not None or limit == "bounded":
            raise ValueError("A bounded leaderboard output requires its authoring protocol")
        return {}
    if authoring_protocol(
        authoring_turns(protocol), cap,
        tool_protocol='bounded-v1' if '-tools-bounded-v1' in protocol else 'legacy',
        continue_on_length=protocol.endswith('-continue-length-v1'),
    ) != protocol:
        raise ValueError("A leaderboard output cap differs from its authoring protocol")
    expected_limit = "provider-native" if cap is None else "bounded"
    if limit not in (None, expected_limit) or (
        cap is not None and limit != expected_limit
    ):
        raise ValueError("A leaderboard output policy differs from its authoring protocol")
    effective = row.get("effective_settings") or {}
    for recorded in (
        row, row.get("generation") or {}, effective.get("generation") or {}
    ):
        if (
            "max_output_tokens" in recorded
            and (
                recorded["max_output_tokens"] != cap
                or type(recorded["max_output_tokens"]) is not type(cap)
            )
            or "output_limit" in recorded and recorded["output_limit"] != expected_limit
        ):
            raise ValueError("A leaderboard row changes its effective output cap or policy")
    budgets = dict(manifest.get("defaults", {}).get("budgets", {}))
    labels = {
        "asset": row["asset_id"],
        "harness": row["harness_id"],
        "model": model["id"],
        "skill_set": row["skill_set_id"],
    }
    for override in manifest.get("overrides", []):
        if all(
            value == "*" or labels[key] == value
            for key, value in override.get("match", {}).items()
        ):
            budgets.update(override.get("settings", {}).get("budgets", {}))
    for recorded in (budgets, effective.get("budgets") or {}):
        if "max_output_tokens" in recorded and (
            recorded["max_output_tokens"] != cap
            or type(recorded["max_output_tokens"]) is not type(cap)
        ):
            raise ValueError("A leaderboard output budget differs from its authoring protocol")
    return (
        {"output_limit": expected_limit, "max_output_tokens": cap}
        if cap is not None
        else {}
    )


def _row_provenance(rows, manifest):
    from simready_bench.config import condition_identity
    from simready_bench.engine.native_output import authoring_turns

    declared = [
        manifest.get(name, []) for name in ("models", "harnesses", "skill_sets")
    ]
    if any(
        not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict)
        for items in declared
    ):
        raise ValueError(
            "A leaderboard entry requires one declared model, harness and skill set"
        )
    model, harness, skills = [items[0] for items in declared]
    if not all(
        isinstance(value, str) and value
        for value in (
            model.get("id"),
            model.get("model_id"),
            harness.get("id"),
            skills.get("id"),
        )
    ):
        raise ValueError("Explicit leaderboard factor identities are required")
    expected = (model["model_id"], harness["id"], skills["id"])
    protocol = manifest.get("authoring_protocol")
    if protocol is not None:
        if not isinstance(protocol, str):
            raise ValueError("An admitted authoring protocol is required")
        authoring_turns(protocol)
    recorded_protocols = [
        row.get("authoring_protocol")
        for row in rows
        if row.get("authoring_protocol") is not None
    ]
    if any(not isinstance(value, str) for value in recorded_protocols):
        raise ValueError("A recorded authoring protocol must be a string")
    recorded_protocols = set(recorded_protocols)
    for value in recorded_protocols:
        authoring_turns(value)
    if (
        len(recorded_protocols) > 1
        or protocol is not None
        and recorded_protocols - {protocol}
    ):
        raise ValueError("A leaderboard entry cannot mix authoring protocols")
    if (
        protocol is None
        and len(recorded_protocols) == 1
        and all(row.get("authoring_protocol") is not None for row in rows)
    ):
        protocol = next(iter(recorded_protocols))
    master_seed = manifest.get("seed")
    if master_seed is not None and type(master_seed) is not int:
        raise ValueError("An explicit leaderboard master seed must be an integer")
    identifiers, efforts = set(), []
    output_policy = {}
    for row in rows:
        actual = (
            row.get("requested_model") or (row.get("model") or {}).get("model_id"),
            row.get("harness_id"),
            row.get("skill_set_id"),
        )
        if actual != expected:
            raise ValueError(
                "A leaderboard row differs from its declared model, harness or skills"
            )
        identifier, seed = row.get("condition_id"), row.get("seed")
        if identifier is not None:
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in identifiers
            ):
                raise ValueError(
                    "A leaderboard row requires a distinct condition identity"
                )
            identifiers.add(identifier)
        if seed is not None and type(seed) is not int:
            raise ValueError("A recorded reconstruction seed must be an integer")
        if protocol is not None and master_seed is not None:
            expected_id, expected_seed = condition_identity(
                {
                    "asset": row["asset_id"],
                    "harness": harness["id"],
                    "model": model["id"],
                    "skill_set": skills["id"],
                },
                repetition=row["repetition"],
                master_seed=master_seed,
                authoring_protocol=protocol,
            )
            if (
                identifier is not None
                and identifier != expected_id
                or seed is not None
                and seed != expected_seed
            ):
                raise ValueError(
                    "A leaderboard row changes its frozen condition identity or seed"
                )
        generation = _generation(row, manifest, model)
        output_policy = _output_policy(row, manifest, model, protocol)
        declared_effort = generation.get("reasoning_effort")
        recorded = []
        effective = row.get("effective_settings") or {}
        for settings in (
            row,
            row.get("generation") or {},
            effective.get("generation") or {},
        ):
            if "reasoning_effort" in settings:
                recorded.append(settings["reasoning_effort"])
        if any(
            value is not None
            and (
                not isinstance(value, str)
                or value
                not in {
                    "none",
                    "off",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                }
            )
            for value in [declared_effort, *recorded]
        ):
            raise ValueError(
                "A leaderboard reasoning effort must retain an admitted value"
            )
        known = {value for value in recorded if value is not None}
        if len(known) > 1 or declared_effort is not None and known - {declared_effort}:
            raise ValueError("A leaderboard row changes its effective reasoning effort")
        efforts.append(next(iter(known)) if known else declared_effort)
    known = {value for value in efforts if value is not None}
    if len(known) > 1:
        raise ValueError("A leaderboard entry cannot mix effective reasoning efforts")
    effort = (
        next(iter(known))
        if known and all(value is not None for value in efforts)
        else None
    )
    return protocol, effort, output_policy


def leaderboard_entry(result, *, manifest, results_url):
    """Require the public NV25 set and retain missing component scores."""
    reference = json.loads(
        files("simready_bench")
        .joinpath("resources/benchmarks/nvidia-25.json")
        .read_text()
    )
    expected = {asset["id"] for asset in reference["assets"]}
    assets = manifest.get("assets", [])
    if len(assets) != 25 or {asset["id"] for asset in assets} != expected:
        raise ValueError("An NV25 entry requires the 25 public reference assets")
    _reference_pin(manifest, reference)
    repetitions = manifest.get("repetitions")
    if repetitions != 4:
        raise ValueError("An NV25 entry requires four reconstructions per reference")
    if manifest.get("scoring", {}).get("profile") != {
        "id": "Robotics-Prop",
        "version": "4.0.0",
    }:
        raise ValueError("The leaderboard requires the Robotics-Prop 4.0.0 profile")
    rows = result.get("results", [])
    if len(rows) != 25 * repetitions or Counter(
        row.get("asset_id") for row in rows
    ) != Counter({asset: repetitions for asset in expected}):
        raise ValueError("A result for every reference and repetition is required")
    identities = Counter((row.get("asset_id"), row.get("repetition")) for row in rows)
    if any(
        type(row.get("repetition")) is not int for row in rows
    ) or identities != Counter(
        {(asset, repetition): 1 for asset in expected for repetition in range(4)}
    ):
        raise ValueError(
            "Every reference requires four distinct reconstruction identities"
        )
    if any(row.get("status") != "completed" for row in rows):
        raise ValueError("Every reference attempt must have completed")
    models = {
        row.get("requested_model") or (row.get("model") or {}).get("model_id")
        for row in rows
    }
    harnesses = {row.get("harness_id") for row in rows}
    skill_sets = {row.get("skill_set_id") for row in rows}
    if (
        len(models) != 1
        or not next(iter(models))
        or len(harnesses) != 1
        or len(skill_sets) != 1
    ):
        raise ValueError(
            "Each leaderboard entry describes one model, harness and skill set"
        )
    protocol, reasoning, output_policy = _row_provenance(rows, manifest)
    scores = []
    primary_parts = []
    optional_parts = []
    for row in rows:
        primary = (
            row.get("simready") or (row.get("evaluation") or {}).get("simready") or {}
        )
        score = primary.get("score_percent", row.get("simready_reward"))
        if not _score(score):
            raise ValueError(
                "Every attempt requires a measured SimReady Robotics-Prop score"
            )
        scores.append(score)
        primary_parts.append(primary)
        optional_parts.append(
            row.get("optional_score")
            or primary.get("optional_score")
            or (row.get("evaluation") or {}).get("optional_score")
            or {}
        )
    fidelity = {}
    for name in COMPONENTS:
        parts = [
            (row.get("fidelity") or {}).get("components", {}).get(name, {})
            for row in rows
        ]
        fidelity[name] = _fidelity_summary(parts)
    skills = manifest.get("skill_sets", [{"paths": []}])[0].get("paths", [])
    usage_rows = [usage_columns(row) for row in rows]
    usage = {}
    for name in USAGE_FIELDS:
        values = [row[name] for row in usage_rows]
        measured = [value for value in values if value is not None]
        usage[name] = {
            "mean": _mean(values),
            "total": sum(values) if len(measured) == len(values) else None,
            "measured_mean": _mean(measured) if measured else None,
            "measured_total": sum(measured) if measured else None,
            "measured_attempts": len(measured),
            "total_attempts": len(rows),
        }
    composite_parts = {name: [] for name in COMPOSITE_NAMES}
    for row in rows:
        for name, part in composite_scores(row).items():
            if name in composite_parts:
                composite_parts[name].append(part)
    composites, efficiency = statistical_summaries(
        rows,
        parts=composite_parts,
        summaries={
            name: _fidelity_summary(parts) for name, parts in composite_parts.items()
        },
        usage_rows=usage_rows,
    )
    return {
        "reference_set": "NV25",
        "reference_revision": reference["source"]["revision"],
        "benchmark_freeze": (
            {key: manifest["benchmark_freeze"][key] for key in ("id", "version")}
            if manifest.get("benchmark_freeze") is not None
            else None
        ),
        "reference_count": 25,
        "repetitions": repetitions,
        "completed_attempts": len(rows),
        "model": next(iter(models)),
        "harness": next(iter(harnesses)),
        "reasoning_effort": reasoning,
        "authoring_protocol": protocol,
        **output_policy,
        "inference_requests_per_attempt": manifest.get("defaults", {})
        .get("budgets", {})
        .get("inference_requests"),
        "skills": skills,
        "compliance": mean(scores),
        "usage": usage,
        "mandatory": _profile_summary(primary_parts, scores, count=46),
        "optional": optional_summary_with_ci(
            rows,
            _profile_summary(
                optional_parts,
                [part.get("score_percent") for part in optional_parts],
                count=51,
                optional=True,
            ),
        ),
        "fidelity": fidelity,
        "composite_scores": composites,
        "efficiency": efficiency,
        "success": success_summary(rows),
        "profile": manifest["scoring"]["profile"],
        "results_url": results_url,
    }
