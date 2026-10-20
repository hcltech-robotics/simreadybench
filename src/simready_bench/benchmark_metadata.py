"""Describe the benchmark and display its admitted result sets on the Hub."""

from html import escape
from importlib.resources import files
import json
import math
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
    summary = composites.get(name) or {}
    if not isinstance(summary, dict):
        raise ValueError("A leaderboard composite summary must be an object")
    score = summary.get("score")
    if score is None:
        return "N/A" if summary.get("status") == "not_applicable" else "unknown"
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 100
    ):
        raise ValueError("A leaderboard composite must be a percent score or null")
    return str(score)


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
                "Each row is one complete reference set. Rows are grouped by authoring protocol, with independent fidelity scores and no combined ranking. Full values, measurement coverage, usage and source revisions are retained in [leaderboard.json](leaderboard.json).",
                "",
                "| Model | Harness | Skills | Reasoning | Protocol | Reference set | Profile | Formal (%) | Visual (%) | Kinematic (%) | Source |",
                "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for entry in sorted(board["entries"], key=_order):
            profile = entry.get("profile") or {}
            if not isinstance(profile, dict):
                raise ValueError("A leaderboard profile must be an object")
            cells = [
                _cell(entry.get("model")),
                _cell(entry.get("harness")),
                _skills(entry),
                _cell(entry.get("reasoning_effort")),
                _cell(entry.get("authoring_protocol")),
                _cell(entry.get("reference_set"))
                + " ("
                + _cell(entry.get("reference_count"))
                + " assets x "
                + _cell(entry.get("repetitions"))
                + ")",
                _cell(profile.get("id")) + " " + _cell(profile.get("version")),
                *(_score(entry, name) for name in ("formal", "visual", "kinematic")),
                _source(entry),
            ]
            lines.append("| " + " | ".join(cells) + " |")
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
        body = body[:start] + section + body[end + len(_END) :]
    else:
        body += ("\n" if body.endswith("\n") else "\n\n") + section + "\n"
    return (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False)
        + "---\n"
        + body
    )


def build_benchmark_metadata(existing_card, board):
    """Build card and descriptor bytes without changing the authoritative board."""
    return {
        "README.md": render_benchmark_card(existing_card, board).encode("utf-8"),
        "eval.yaml": benchmark_descriptor(),
    }
