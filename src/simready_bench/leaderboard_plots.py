"""Render score and paired resource-efficiency figures for the leaderboard."""

from __future__ import annotations

from io import BytesIO
import math


METRICS = ("formal", "visual", "kinematic")
PLOTS = (
    ("scores", "Mean fidelity", "Fidelity points (0-100)", None),
    (
        "token-efficiency",
        "Token efficiency",
        "Fidelity points per million tokens",
        "per_million_tokens",
    ),
    (
        "time-efficiency",
        "Elapsed-time efficiency",
        "Fidelity points per elapsed minute",
        "per_elapsed_minute",
    ),
    (
        "inference-cost-efficiency",
        "Pricing-derived inference cost efficiency",
        "Fidelity points per USD of pricing-derived inference cost",
        "per_inference_usd",
    ),
)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _summary(entry, name, efficiency):
    if efficiency is None:
        return (entry.get("composite_scores") or {}).get(name) or {}, "score"
    return ((entry.get("efficiency") or {}).get(name) or {}).get(
        efficiency
    ) or {}, "value"


def _label(entry):
    skills = ", ".join(entry.get("skills") or []) or "no skills"
    protocol = str(entry.get("authoring_protocol") or "unknown protocol")
    if protocol.startswith("authoring-") and protocol.endswith("-native-output-v1"):
        protocol = (
            protocol.removeprefix("authoring-").removesuffix("-native-output-v1")
            + " turns"
        )
    return (
        str(entry.get("model", "Unknown model"))
        + "\n"
        + ", ".join(
            str(value)
            for value in (
                entry.get("harness", "unknown harness"),
                entry.get("reasoning_effort") or "unknown reasoning",
                skills,
                protocol,
            )
        )
    )


def render_leaderboard_plots(board):
    """Return deterministic PNG files for complete entries, skipping empty boards.

    Each entry is a complete NV25 x4 condition. Efficiency points use its
    paired measured subset, with 95% asset-cluster percentile intervals.
    The corresponding machine-readable values remain in ``leaderboard.json``.
    """
    entries = [
        entry
        for entry in board.get("entries", [])
        if (
            entry.get("reference_set") == "NV25"
            and entry.get("reference_count") == 25
            and entry.get("repetitions") == 4
            and entry.get("completed_attempts") == 100
        )
    ]
    entries.sort(
        key=lambda entry: tuple(
            str(entry.get(key) or "")
            for key in (
                "authoring_protocol",
                "model",
                "harness",
                "reasoning_effort",
                "skills",
                "results_url",
            )
        )
    )
    if not entries:
        return {}
    # Import only for publication so normal score generation has no plotting
    # runtime requirement. The publish extra pins the renderer dependency.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = {}
    for filename, title, unit, efficiency in PLOTS:
        summaries = [
            [_summary(entry, name, efficiency) for entry in entries] for name in METRICS
        ]
        height = max(3.4, 1.8 + 0.66 * len(entries))
        figure = Figure(
            figsize=(13.0, height), dpi=160, facecolor="white", layout="constrained"
        )
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3, sharey=True)
        incomplete_usage = False
        for axis, name, group in zip(axes, METRICS, summaries):
            axis.set_title(name.capitalize(), fontsize=12, loc="left", pad=12)
            axis.set_yticks(range(len(entries)))
            axis.set_yticklabels([_label(entry) for entry in entries], fontsize=9)
            axis.set_ylim(len(entries) - 0.35, -0.65)
            axis.set_axisbelow(True)
            axis.grid(axis="x", color="#e8ebef", linewidth=0.7)
            axis.tick_params(axis="both", labelsize=9, length=0, pad=7)
            for spine in axis.spines.values():
                spine.set_visible(False)
            for y, (summary, key) in enumerate(group):
                value = summary.get(key)
                if not _number(value):
                    label = (
                        "N/A"
                        if summary.get("status") == "not_applicable"
                        else "Not measured"
                    )
                    axis.text(
                        0.03,
                        y,
                        label,
                        transform=axis.get_yaxis_transform(),
                        va="center",
                        fontsize=8,
                        color="#777777",
                    )
                    continue
                interval = summary.get("confidence_interval") or {}
                lower, upper = interval.get("lower"), interval.get("upper")
                if _number(lower) and _number(upper):
                    axis.plot([lower, upper], [y, y], color="#526d82", linewidth=1.8)
                    axis.plot(
                        [lower, lower],
                        [y - 0.065, y + 0.065],
                        color="#526d82",
                        linewidth=1.1,
                    )
                    axis.plot(
                        [upper, upper],
                        [y - 0.065, y + 0.065],
                        color="#526d82",
                        linewidth=1.1,
                    )
                axis.plot(value, y, "o", color="#173c54", markersize=5, clip_on=False)
                measured, eligible = summary.get("measured_attempts"), summary.get("eligible_attempts")
                if efficiency and type(measured) is int and type(eligible) is int and 0 < measured < eligible:
                    axis.annotate("*", (value, y), xytext=(4, 4), textcoords="offset points",
                                  fontsize=9, color="#173c54", annotation_clip=False)
                    incomplete_usage = True
            axis.set_xlim(left=0)
            if efficiency is None:
                axis.set_xlim(-2, 102)
                axis.set_xticks([0, 25, 50, 75, 100])
            elif not any(_number(summary.get(key)) for summary, key in group):
                axis.set_xlim(0, 1)
                axis.set_xticks([])
            axis.set_xlabel(unit, fontsize=9, labelpad=10)
        figure.suptitle(
            title + " with 95% confidence intervals", fontsize=14, x=0.01, ha="left"
        )
        if incomplete_usage:
            figure.supxlabel("* Incomplete usage coverage.", fontsize=8, x=0.01, ha="left")
        buffer = BytesIO()
        figure.savefig(
            buffer, format="png", dpi=160, metadata={"Software": "SimReadyBench"}
        )
        output["media/leaderboard-" + filename + ".png"] = buffer.getvalue()
        figure.clear()
    return output
