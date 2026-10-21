from copy import deepcopy

import pytest
import yaml

from simready_bench.benchmark_metadata import (
    benchmark_descriptor,
    build_benchmark_metadata,
    render_benchmark_card,
)
from simready_bench.leaderboard_statistics import statistical_summaries


def entry(**changes):
    return {
        "model": "model/a",
        "harness": "pi",
        "skills": [],
        "reasoning_effort": "medium",
        "authoring_protocol": "authoring-64-native-output-v1",
        "reference_set": "NV25",
        "reference_count": 25,
        "repetitions": 4,
        "reference_revision": "pinned-reference-revision",
        "profile": {"id": "Robotics-Prop", "version": "4.0.0"},
        "composite_scores": {
            "formal": {"score": 0, "status": "measured"},
            "visual": {"score": None, "status": "unknown"},
            "kinematic": {"score": None, "status": "not_applicable"},
        },
        "results_url": "https://huggingface.co/datasets/team/results/blob/main/runs/a/report.html",
        **changes,
    }


def board(*entries):
    return {"schema": "simreadybench-leaderboard/v1", "entries": list(entries)}


def test_packaged_descriptor_declares_real_framework_and_separate_fixed_tasks():
    raw = benchmark_descriptor()
    assert raw.isascii()
    data = yaml.safe_load(raw)
    assert data["evaluation_framework"] == "harbor"
    assert data["name"] == "SimReadyBench NV25"
    assert data["tasks"] == [
        {"id": "nv25_64x4_formal_fidelity"},
        {"id": "nv25_64x4_visual_fidelity"},
        {"id": "nv25_64x4_kinematic_fidelity"},
    ]
    assert "64 authoring turns" in data["description"]
    assert "verifyToken" not in data


def test_empty_table_adds_discovery_metadata_and_keeps_card_and_configs():
    metadata = {
        "license": "cc-by-4.0",
        "tags": ["custom-tag", "3d"],
        "pretty_name": "Existing scientific title",
        "citation": "Retained citation",
        "configs": [
            {"config_name": name, "data_files": name + ".csv"}
            for name in (
                "all",
                "by_model",
                "by_harness",
                "by_skills",
                "by_reference_set",
                "by_protocol",
                "by_reasoning",
                "by_asset",
                "manual",
            )
        ],
    }
    body = "# Scientific results\n\nAttribution and methodology.\n\n![Figure](figures/protocol.png)\n"
    card = "---\n" + yaml.safe_dump(metadata) + "---\n" + body
    authoritative = board()
    original = deepcopy(authoritative)
    update = build_benchmark_metadata(card, authoritative)
    assert set(update) == {"README.md", "eval.yaml"}
    rendered = update["README.md"].decode()
    after = yaml.safe_load(rendered.split("---\n", 2)[1])
    assert after["configs"] == metadata["configs"]
    assert after["license"] == metadata["license"]
    assert after["citation"] == metadata["citation"]
    assert after["pretty_name"] == metadata["pretty_name"]
    assert after["tags"] == [
        "custom-tag",
        "3d",
        "benchmark",
        "evaluation",
        "results",
        "robotics",
    ]
    assert body + "\n<!-- simreadybench-leaderboard:start -->" in rendered
    assert "No complete NV25 sets have been published." in rendered
    assert "| Model |" not in rendered
    assert "benchmark:official" not in rendered and "rl-environment" not in rendered
    assert authoritative == original
    assert build_benchmark_metadata(rendered, authoritative) == update


def test_table_preserves_zero_unknown_inapplicability_and_full_factor_identity():
    item = entry()
    item["usage"] = {"input_tokens": {"mean": None, "total_attempts": 100}}
    authoritative = board(item)
    before = deepcopy(authoritative)
    rendered = render_benchmark_card("# Results\n", authoritative)
    assert (
        "| model/a | medium | pi | none | 0.00 | unknown | unknown | N/A |" in rendered
    )
    assert '<a href="' + item["results_url"] + '">Readout</a>' in rendered
    assert "[leaderboard.json](leaderboard.json)" in rendered
    assert "combined ranking" in rendered
    assert authoritative == before


def test_protocol_groups_are_deterministic_without_ranking_by_score():
    earlier = entry(model="model/z", authoring_protocol="authoring-32-native-output-v1")
    unknown = entry(model="model/b", reasoning_effort=None, authoring_protocol=None)
    normal = entry(model="model/a", skills=["authoring-tools"])
    a = render_benchmark_card("# Results\n", board(unknown, normal, earlier))
    b = render_benchmark_card("# Results\n", board(earlier, unknown, normal))
    assert a == b
    assert a.index("| model/z |") < a.index("| model/a |") < a.index("| model/b |")
    assert "| model/b | unknown | pi | none |" in a
    assert "Protocol: `unknown`" in a
    assert "| model/a | medium | pi | authoring-tools |" in a


def test_managed_table_replacement_preserves_surrounding_manual_content():
    original = render_benchmark_card(
        "# Research\n\nCredit and illustrations.\n", board()
    )
    original += "\n## Citation\n\nExact trailing citation.\n"
    updated = render_benchmark_card(original, board(entry()))
    assert updated.count("<!-- simreadybench-leaderboard:start -->") == 1
    assert "No complete NV25 sets" not in updated
    assert "# Research\n\nCredit and illustrations.\n\n<!--" in updated
    assert updated.endswith("\n## Citation\n\nExact trailing citation.\n")
    assert render_benchmark_card(updated, board(entry())) == updated


def test_display_values_are_inert_markdown_and_html():
    item = entry(
        model="[model](https://other.invalid) | <script>\nnext", skills=["skill|name"]
    )
    rendered = render_benchmark_card("# Results\n", board(item))
    assert "[model](https://other.invalid)" not in rendered
    assert "<script>" not in rendered
    assert "&#124;" in rendered and "&lt;script&gt; next" in rendered
    row = next(line for line in rendered.splitlines() if "&lt;script&gt;" in line)
    assert row.count("|") == 10


@pytest.mark.parametrize("score", [True, -1, 101, float("nan"), float("inf"), "0"])
def test_invalid_board_composite_is_rejected_without_coercion(score):
    item = entry()
    item["composite_scores"]["formal"]["score"] = score
    with pytest.raises(ValueError, match="percent score or null"):
        render_benchmark_card("", board(item))


@pytest.mark.parametrize(
    "source",
    ["javascript:alert(1)", "https://user:secret@example.org/trace", "relative.html"],
)
def test_unsafe_source_is_not_rendered(source):
    with pytest.raises(ValueError, match="HTTP source"):
        render_benchmark_card("", board(entry(results_url=source)))


@pytest.mark.parametrize(
    "body",
    [
        "<!-- simreadybench-leaderboard:start -->",
        "<!-- simreadybench-leaderboard:end -->",
        "<!-- simreadybench-leaderboard:end --><!-- simreadybench-leaderboard:start -->",
        "<!-- simreadybench-leaderboard:start --><!-- simreadybench-leaderboard:start --><!-- simreadybench-leaderboard:end --><!-- simreadybench-leaderboard:end -->",
    ],
)
def test_incomplete_or_ambiguous_owned_markers_fail_before_replacement(body):
    with pytest.raises(ValueError, match="invalid markers"):
        render_benchmark_card(body, board())


def test_leaderboard_moves_ahead_of_guide_without_replacing_academic_header():
    header = '<div align="center">\n  <h1>SimReadyBench</h1>\n  <img src="media/card.png">\n</div>\n\nResearch introduction.\n'
    guide = "## Reference sets\n\nExact scientific guide.\n\n## Citation\n\nExact citation.\n"
    old_section = "<!-- simreadybench-leaderboard:start -->\nOld results.\n<!-- simreadybench-leaderboard:end -->\n"
    rendered = render_benchmark_card(
        header + "\n" + guide + "\n" + old_section, board(entry())
    )
    body = rendered.split("---\n", 2)[2]
    assert body.startswith(header + "\n<!-- simreadybench-leaderboard:start -->")
    assert body.index("## NV25 leaderboard") < body.index("## Reference sets")
    assert guide in body
    assert body.count("<!-- simreadybench-leaderboard:start -->") == 1
    assert "Old results." not in body
    assert render_benchmark_card(rendered, board(entry())) == rendered


def test_intervals_efficiency_and_resource_coverage_are_readable():
    item = entry()
    interval = {"level": 0.95, "lower": 10.123, "upper": 23.456}
    item["composite_scores"]["formal"] = {
        "score": 17.12345,
        "confidence_interval": interval,
    }
    item["efficiency"] = {
        "formal": {
            "per_million_tokens": {"value": 117.12, "confidence_interval": interval},
            "per_elapsed_minute": {"value": 3.14},
            "per_inference_usd": {"value": 12.5},
        }
    }
    item["usage"] = {
        "total_tokens": {"mean": None, "measured_mean": 1234, "measured_attempts": 97, "total_attempts": 100},
        "elapsed_seconds": {
            "mean": 120,
            "measured_attempts": 100,
            "total_attempts": 100,
        },
    }
    rendered = render_benchmark_card("# Results\n", board(item))
    assert "17.12 [10.12, 23.46]" in rendered
    assert "117.12 [10.12, 23.46]" in rendered
    assert "1,234.00* | 2.00" in rendered
    assert rendered.count("\\* Incomplete usage coverage.") == 1
    assert "97/100 measured" not in rendered
    assert "10,000 asset-cluster bootstrap" in rendered
    assert "media/leaderboard-scores.png" in rendered
    assert "media/leaderboard-token-efficiency.png" in rendered
    assert "media/leaderboard-time-efficiency.png" in rendered
    assert "media/leaderboard-inference-cost-efficiency.png" in rendered
    assert "Pricing-derived inference cost (USD) | Compute cost (USD) | Total cost (USD)" in rendered
    assert "12.50" in rendered


def test_publication_bundle_includes_the_figures_referenced_by_the_card(monkeypatch):
    figures = {
        "media/leaderboard-scores.png": b"scores",
        "media/leaderboard-token-efficiency.png": b"tokens",
        "media/leaderboard-time-efficiency.png": b"time",
        "media/leaderboard-inference-cost-efficiency.png": b"cost",
    }
    authoritative = board(entry())
    original = deepcopy(authoritative)

    def plots(data):
        assert data == authoritative
        return figures

    monkeypatch.setattr(
        "simready_bench.leaderboard_plots.render_leaderboard_plots", plots
    )
    update = build_benchmark_metadata("# Results\n", authoritative)
    assert {path: update[path] for path in figures} == figures
    assert all(path in update["README.md"].decode() for path in figures)
    assert authoritative == original


@pytest.mark.parametrize(
    "interval",
    [
        {"level": 0.9, "lower": 0, "upper": 1},
        {"level": 0.95, "lower": 4, "upper": 1},
        {"level": 0.95, "lower": float("nan"), "upper": 1},
        {"level": 0.95, "lower": None, "upper": 1},
        {"level": 0.95, "lower": 1, "upper": None},
        {"level": 0.95},
    ],
)
def test_invalid_confidence_interval_is_not_displayed_as_valid(interval):
    item = entry()
    item["composite_scores"]["formal"]["confidence_interval"] = interval
    with pytest.raises(ValueError, match="95% bounds"):
        render_benchmark_card("", board(item))


@pytest.mark.parametrize("values,tokens", [((10,), (1000,)), ((25, 25), (0, 1000))])
def test_real_bootstrap_output_without_interval_still_renders_measured_values(
    values, tokens
):
    rows, parts, usage = [], [], []
    for asset, (value, token_count) in enumerate(zip(values, tokens)):
        for repetition in range(4):
            rows.append({"asset_id": str(asset), "repetition": repetition})
            parts.append({"eligible": 1, "score_percent": value})
            usage.append({"total_tokens": token_count, "elapsed_seconds": 60})
    composites, efficiency = statistical_summaries(
        rows,
        parts={"formal": parts},
        summaries={"formal": {"score": sum(values) / len(values)}},
        usage_rows=usage,
    )
    item = entry()
    item["composite_scores"]["formal"] = composites["formal"]
    item["efficiency"] = efficiency
    assert (
        efficiency["formal"]["per_million_tokens"]["confidence_interval"]["lower"]
        is None
    )
    rendered = render_benchmark_card("# Results\n", board(item))
    assert f"| {efficiency['formal']['per_million_tokens']['value']:.2f} |" in rendered
    assert "[None" not in rendered


def test_resource_tables_distinguish_protocols_and_show_eligible_coverage():
    items = [
        entry(authoring_protocol=f"authoring-{turns}-native-output-v1")
        for turns in (32, 64)
    ]
    items[0]["efficiency"] = {
        "formal": {
            "per_million_tokens": {
                "value": 123.45,
                "measured_attempts": 97,
                "eligible_attempts": 100,
            }
        }
    }
    items[1]["efficiency"] = {
        "kinematic": {
            "per_million_tokens": {
                "value": 67.89,
                "measured_attempts": 39,
                "eligible_attempts": 40,
            }
        }
    }
    rendered = render_benchmark_card("# Results\n", board(*items))
    for item in items:
        assert "<br>" + item["authoring_protocol"] + " |" in rendered
    assert "123.45*" in rendered
    assert "67.89*" in rendered
    assert "97/100 measured" not in rendered
    assert "39/40 measured" not in rendered


def test_strict_success_rates_keep_zero_and_partial_measurement_coverage():
    item = entry(
        success={
            "mandatory_pass_per_attempt": {
                "percent": 0,
                "passed": 0,
                "measured": 100,
                "total": 100,
            },
            "mandatory_pass_within_four": {
                "percent": None,
                "passed": 0,
                "measured": 24,
                "total": 25,
            },
        }
    )
    rendered = render_benchmark_card("# Results\n", board(item))
    assert "| 0.00% (0/100) | unknown (24/25 measured) |" in rendered
    assert "all 46 mandatory SimReady requirements" in rendered


def test_optional_scores_have_their_own_interval_beside_formal_fidelity():
    item = entry(
        optional={
            "score": 72.345,
            "confidence_interval": {"level": 0.95, "lower": 68.123, "upper": 77.456},
        }
    )
    original = deepcopy(item)
    rendered = render_benchmark_card("# Results\n", board(item))
    assert "| Formal (%) | Optional (%) | Visual (%) |" in rendered
    assert "| 0.00 | 72.34 [68.12, 77.46] | unknown | N/A |" in rendered
    assert (
        "51 optional SimReady requirements separately from the 46 mandatory" in rendered
    )
    assert item == original
