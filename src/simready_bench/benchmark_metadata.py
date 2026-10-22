"""Describe the benchmark and display its admitted result sets on the Hub."""

from html import escape
from importlib.resources import files
import json
import math
import re
from urllib.parse import urlsplit

import yaml


_START = "<!-- simreadybench-leaderboard:start -->"
_END = "<!-- simreadybench-leaderboard:end -->"
_TAGS = ("benchmark", "evaluation", "results", "3d", "robotics")


def benchmark_descriptor():
    """Return the packaged HF benchmark registration descriptor."""
    return (
        files("simready_bench").joinpath("resources/benchmarks/eval.yaml").read_bytes()
    )


def _cell(value):
    text = "unknown" if value is None else " ".join(str(value).split())
    text = escape(text).encode("ascii", "xmlcharrefreplace").decode()
    for character in "\\`*_{}[]|":
        text = text.replace(character, "&#" + str(ord(character)) + ";")
    return text


def _score(entry, name):
    composites = entry.get("composite_scores") or {}
    if not isinstance(composites, dict):
        raise ValueError("Leaderboard composite summaries must be objects")
    summary = (
        entry.get("optional") if name == "optional" else composites.get(name)
    ) or {}
    if not isinstance(summary, dict):
        raise ValueError("A leaderboard composite summary must be an object")
    score = summary.get("score")
    if score is None:
        if name == 'optional' and all(
            type(summary.get(key)) in (int, float) and math.isfinite(summary[key])
            and 0 <= summary[key] <= 100 for key in ('lower_bound', 'upper_bound', 'coverage')
        ) and summary['lower_bound'] <= summary['upper_bound'] and summary.get('measured', 0) > 0:
            return (f"{summary['lower_bound']:.2f}-{summary['upper_bound']:.2f} bounds"
                    f" ({summary['coverage']:.3f}% measured)")
        return "N/A" if summary.get("status") == "not_applicable" else "unknown"
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 100
    ):
        raise ValueError("A leaderboard composite must be a percent score or null")
    return _estimate(score, summary.get("confidence_interval"))


def _estimate(value, interval=None):
    rendered = f"{value:.2f}"
    if interval is None:
        return rendered
    if (
        isinstance(interval, dict)
        and interval.get("level") == 0.95
        and "lower" in interval
        and "upper" in interval
        and interval["lower"] is None
        and interval["upper"] is None
    ):
        return rendered
    if (
        not isinstance(interval, dict)
        or interval.get("level") != 0.95
        or any(
            type(interval.get(bound)) not in (int, float)
            or not math.isfinite(interval[bound])
            for bound in ("lower", "upper")
        )
        or interval["lower"] > interval["upper"]
    ):
        raise ValueError(
            "A leaderboard confidence interval must have ordered 95% bounds"
        )
    return rendered + f" [{interval['lower']:.2f}, {interval['upper']:.2f}]"


def _skills(entry):
    skills = entry.get("skills")
    if skills is None:
        return "unknown"
    if not isinstance(skills, list) or any(
        not isinstance(skill, str) for skill in skills
    ):
        raise ValueError("Leaderboard skills must be a list of names")
    return ", ".join(_cell(skill) for skill in skills) if skills else "none"


def _source(entry):
    url = entry.get("results_url")
    if not isinstance(url, str):
        return "unknown"
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(ord(character) < 32 for character in url)
    ):
        raise ValueError("A leaderboard readout requires an HTTP source URL")
    return '<a href="' + escape(url, quote=True).replace("|", "%7C") + '">Readout</a>'


def _order(entry):
    return tuple(
        json.dumps(entry.get(field), sort_keys=True, ensure_ascii=True)
        for field in (
            "authoring_protocol",
            "reference_revision",
            "profile",
            "model",
            "harness",
            "skills",
            "reasoning_effort",
            "results_url",
        )
    )


def _condition(entry):
    return "<br>".join(
        (
            _cell(entry.get("model")),
            _cell(entry.get("reasoning_effort"))
            + " reasoning, "
            + _cell(entry.get("harness"))
            + ", skills: "
            + _skills(entry),
            _cell(entry.get("authoring_protocol")),
        )
    )


def _usage(entry, name, scale=1, suffix=""):
    metric = (entry.get("usage") or {}).get(name) or {}
    value = metric.get("mean")
    measured, total = metric.get("measured_attempts"), metric.get("total_attempts")
    if (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
        and type(total) is int
        and total > 0
        and measured == total
    ):
        return f"{value / scale:,.2f}" + suffix
    value = metric.get("measured_mean")
    if (
        type(value) in (int, float) and math.isfinite(value) and value >= 0
        and type(measured) is int and type(total) is int and 0 < measured < total
    ):
        return f"{value / scale:,.2f}" + suffix + "*"
    return "unknown"


def _efficiency(entry, name, metric):
    summary = ((entry.get("efficiency") or {}).get(name) or {}).get(metric) or {}
    value = summary.get("value")
    if value is None:
        return (
            "N/A"
            if (entry.get("composite_scores") or {}).get(name, {}).get("status")
            == "not_applicable"
            else "unknown"
        )
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(
            "A leaderboard efficiency must be a non-negative value or null"
        )
    measured, eligible = summary.get("measured_attempts"), summary.get("eligible_attempts")
    partial = type(measured) is int and type(eligible) is int and 0 < measured < eligible
    return _estimate(value, summary.get("confidence_interval")) + ("*" if partial else "")


def _success(entry, name):
    summary = (entry.get("success") or {}).get(name) or {}
    passed, measured, total = (
        summary.get(key) for key in ("passed", "measured", "total")
    )
    percent = summary.get("percent")
    counts_valid = (
        all(type(value) is int for value in (passed, measured, total))
        and 0 <= passed <= measured <= total
        and total > 0
    )
    if percent is None:
        return f"unknown ({measured}/{total} measured)" if counts_valid else "unknown"
    if (
        type(percent) not in (int, float)
        or not math.isfinite(percent)
        or not 0 <= percent <= 100
        or not counts_valid
        or measured != total
    ):
        raise ValueError("A leaderboard success rate requires a measured percentage")
    return f"{percent:.2f}% ({passed}/{total})"


def _section(board):
    if (
        not isinstance(board, dict)
        or board.get("schema") != "simreadybench-leaderboard/v1"
        or not isinstance(board.get("entries"), list)
    ):
        raise ValueError("The existing leaderboard has an unexpected schema")
    if any(not isinstance(entry, dict) for entry in board["entries"]):
        raise ValueError("A leaderboard entry must be an object")
    lines = [_START, "## NV25 leaderboard", ""]
    if not board["entries"]:
        lines.append("No complete NV25 sets have been published.")
    else:
        lines.extend(
            [
                "Each entry contains 100 reconstructions: four attempts for each of the 25 NV25 assets. Formal, visual and kinematic fidelity are independent scores on a 0-100 scale, with no combined ranking. Optional results score the 51 optional SimReady requirements separately from the 46 mandatory requirements in formal fidelity. Values are means [95% confidence interval].",
                "",
            ]
        )
        ordered = sorted(board["entries"], key=_order)
        lines.extend([
            "| Model | Reasoning | Harness | Skills | Protocol | Formal score | Optional score | Visual score | Kinematic score | Results |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for entry in ordered:
            profile = entry.get("profile") or {}
            if not isinstance(profile, dict):
                raise ValueError("A leaderboard profile must be an object")
            cells = [
                _cell(entry.get("model")),
                _cell(entry.get("reasoning_effort")),
                _cell(entry.get("harness")),
                _skills(entry),
                _cell(entry.get("authoring_protocol")) + "<br>"
                + _cell(profile.get("id")) + " " + _cell(profile.get("version"))
                + "<br>Reference: " + _cell(entry.get("reference_revision")),
                *(
                    _score(entry, name)
                    for name in ("formal", "optional", "visual", "kinematic")
                ),
                _source(entry),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "Confidence intervals use 10,000 asset-cluster bootstrap samples, retaining the four reconstructions of each sampled asset together. N/A excludes references without eligible criteria. Unknown measurements remain unknown. [Scores, coverage and source revisions](leaderboard.json).",
                "",
                "![Formal, visual and kinematic fidelity means with 95% confidence intervals.](media/leaderboard-scores.png)",
                "",
                "### Efficiency",
                "",
                "Efficiency is the mean fidelity divided by mean resource use across eligible reconstructions with both measurements. Token efficiency uses total input and output tokens, while time efficiency uses elapsed reconstruction time, including model requests, tools and evaluation. Cost efficiency uses pricing-derived inference cost in USD, with compute cost reported separately. Each ratio uses paired score and resource measurements, with joint asset-cluster bootstrap intervals.",
                "",
                "| Configuration | Formal points/1M tokens | Visual points/1M tokens | Kinematic points/1M tokens |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for entry in ordered:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _condition(entry),
                        *(
                            _efficiency(entry, name, "per_million_tokens")
                            for name in ("formal", "visual", "kinematic")
                        ),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "![Formal, visual and kinematic fidelity points per million tokens with 95% confidence intervals.](media/leaderboard-token-efficiency.png)",
                "",
                "| Configuration | Formal points/minute | Visual points/minute | Kinematic points/minute |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for entry in ordered:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _condition(entry),
                        *(
                            _efficiency(entry, name, "per_elapsed_minute")
                            for name in ("formal", "visual", "kinematic")
                        ),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "![Formal, visual and kinematic fidelity points per elapsed minute with 95% confidence intervals.](media/leaderboard-time-efficiency.png)",
                "",
                "| Configuration | Formal points/USD inference | Visual points/USD inference | Kinematic points/USD inference |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for entry in ordered:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _condition(entry),
                        *(
                            _efficiency(entry, name, "per_inference_usd")
                            for name in ("formal", "visual", "kinematic")
                        ),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "![Formal, visual and kinematic fidelity points per USD of pricing-derived inference cost with 95% confidence intervals.](media/leaderboard-inference-cost-efficiency.png)",
                "",
                "### Resource use",
                "",
                "Means per reconstruction with measured usage. Full precision, totals and measurement coverage are retained in [leaderboard.json](leaderboard.json).",
                "",
                "| Configuration | Total tokens | Elapsed minutes | Model requests | Pricing-derived inference cost (USD) | Compute cost (USD) | Total cost (USD) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for entry in ordered:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _condition(entry),
                        _usage(entry, "total_tokens"),
                        _usage(entry, "elapsed_seconds", 60),
                        _usage(entry, "inference_calls_used"),
                        _usage(entry, "inference_cost_usd"),
                        _usage(entry, "compute_cost_usd"),
                        _usage(entry, "total_cost_usd"),
                    ]
                )
                + " |"
            )
        if any(
            type(metric.get("measured_attempts")) is int
            and type(metric.get(total_field)) is int
            and 0 < metric["measured_attempts"] < metric[total_field]
            for entry in ordered
            for metric, total_field in [
                *((metric, "total_attempts") for metric in (entry.get("usage") or {}).values()),
                *((metric, "eligible_attempts")
                  for scores in (entry.get("efficiency") or {}).values()
                  for metric in scores.values()),
            ]
        ):
            lines.extend(["", "\\* Incomplete usage coverage."])
        lines.extend(
            [
                "",
                "### Strict conformance",
                "",
                "A strict pass satisfies all 46 mandatory SimReady requirements. The four-attempt rate is the fraction of assets with at least one strict pass among their four reconstructions.",
                "",
                "| Configuration | Strict pass per reconstruction | At least one strict pass in four |",
                "| --- | ---: | ---: |",
            ]
        )
        for entry in ordered:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _condition(entry),
                        _success(entry, "mandatory_pass_per_attempt"),
                        _success(entry, "mandatory_pass_within_four"),
                    ]
                )
                + " |"
            )
    lines.extend(["", _END])
    return "\n".join(lines)


def render_benchmark_card(existing_card, board):
    """Retain card content and metadata while updating the owned leaderboard."""
    if existing_card.startswith("---\n"):
        end = existing_card.index("\n---\n", 4)
        metadata = yaml.safe_load(existing_card[4:end]) or {}
        body = existing_card[end + 5 :]
    else:
        metadata, body = {}, existing_card or "# SimReadyBench results\n"
    if not isinstance(metadata, dict):
        raise ValueError("The existing dataset card has invalid metadata")
    tags = metadata.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise ValueError("Dataset discovery tags must be a list of strings")
    metadata["tags"] = tags + [tag for tag in _TAGS if tag not in tags]
    metadata.setdefault("pretty_name", "SimReadyBench results")
    section = _section(board)
    if body.count(_START) != body.count(_END) or body.count(_START) > 1:
        raise ValueError("The managed leaderboard section has invalid markers")
    if _START in body:
        start, end = body.index(_START), body.index(_END)
        if end < start:
            raise ValueError("The managed leaderboard section has invalid markers")
        body = body[:start].rstrip() + "\n\n" + body[end + len(_END) :].lstrip()
    # The academic header and introductory paragraph precede the first manual
    # second-level section. Keep them intact and place results before the guide.
    first_section = re.search(r"(?m)^##[ \t]+", body)
    if first_section:
        at = first_section.start()
        body = body[:at].rstrip() + "\n\n" + section + "\n\n" + body[at:]
    else:
        body = body.rstrip() + "\n\n" + section + "\n"
    return (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False)
        + "---\n"
        + body
    )


def build_benchmark_metadata(existing_card, board):
    """Build card and descriptor bytes without changing the authoritative board."""
    from simready_bench.leaderboard_plots import render_leaderboard_plots

    return {
        "README.md": render_benchmark_card(existing_card, board).encode("utf-8"),
        "eval.yaml": benchmark_descriptor(),
        **render_leaderboard_plots(board),
    }
