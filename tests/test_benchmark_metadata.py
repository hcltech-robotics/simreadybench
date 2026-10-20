from copy import deepcopy

import pytest
import yaml

from simready_bench.benchmark_metadata import (
    benchmark_descriptor,
    build_benchmark_metadata,
    render_benchmark_card,
)


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
        "| model/a | pi | none | medium | authoring-64-native-output-v1 | NV25 (25 assets x 4) | Robotics-Prop 4.0.0 | 0 | unknown | N/A |"
        in rendered
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
    assert "| model/b | pi | none | unknown | unknown |" in a
    assert "| model/a | pi | authoring-tools | medium |" in a


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
    assert row.count("|") == 12


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
