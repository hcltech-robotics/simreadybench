from io import BytesIO

from PIL import Image

from simready_bench.leaderboard_plots import render_leaderboard_plots


def board():
    return {
        "entries": [
            {
                "reference_set": "NV25",
                "reference_count": 25,
                "repetitions": 4,
                "completed_attempts": 100,
                "model": "example/model",
                "harness": "pi",
                "reasoning_effort": "high",
                "skills": [],
                "composite_scores": {
                    name: {
                        "score": 50,
                        "confidence_interval": {"lower": 40, "upper": 60},
                    }
                    for name in ("formal", "visual", "kinematic")
                },
                "efficiency": {
                    name: {
                        "per_million_tokens": {
                            "value": 1000,
                            "confidence_interval": {"lower": 800, "upper": 1300},
                        },
                        "per_elapsed_minute": {
                            "value": 10,
                            "confidence_interval": {"lower": 8, "upper": 13},
                        },
                        "per_inference_usd": {
                            "value": 100,
                            "confidence_interval": {"lower": 80, "upper": 130},
                        },
                    }
                    for name in ("formal", "visual", "kinematic")
                },
            }
        ]
    }


def test_figures_are_real_pngs_with_stable_publication_paths():
    figures = render_leaderboard_plots(board())
    assert set(figures) == {
        "media/leaderboard-scores.png",
        "media/leaderboard-token-efficiency.png",
        "media/leaderboard-time-efficiency.png",
        "media/leaderboard-inference-cost-efficiency.png",
    }
    for content in figures.values():
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            assert image.width == 2080
            assert image.height >= 544
    assert render_leaderboard_plots(board()) == figures


def test_incomplete_boards_generate_no_figures():
    assert render_leaderboard_plots({"entries": []}) == {}
    partial = board()
    partial["entries"][0]["completed_attempts"] = 99
    assert render_leaderboard_plots(partial) == {}


def test_old_entries_and_missing_metrics_do_not_invent_resource_values():
    legacy = board()
    del legacy["entries"][0]["efficiency"]
    legacy["entries"][0]["composite_scores"]["kinematic"] = {
        "score": None,
        "status": "not_applicable",
    }
    figures = render_leaderboard_plots(legacy)
    assert len(figures) == 4


def test_partial_efficiency_points_show_small_asterisks_and_one_footnote(monkeypatch):
    from matplotlib.figure import Figure

    partial = board()
    for name, measured, eligible in (("formal", 97, 100), ("visual", 97, 100), ("kinematic", 54, 55)):
        partial["entries"][0]["efficiency"][name]["per_million_tokens"].update(
            measured_attempts=measured, eligible_attempts=eligible,
        )
    captured = {}
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        captured[figure._suptitle.get_text()] = {
            "text": [text.get_text() for text in figure.texts],
            "annotations": [text.get_text() for axis in figure.axes for text in axis.texts],
            "points": [float(line.get_xdata()[0]) for axis in figure.axes
                       for line in axis.lines if line.get_marker() == "o"],
        }
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    render_leaderboard_plots(partial)
    token = captured["Token efficiency with 95% confidence intervals"]
    assert token["points"] == [1000, 1000, 1000]
    assert token["annotations"] == ["*", "*", "*"]
    assert token["text"].count("* Incomplete usage coverage.") == 1
    assert "* Incomplete usage coverage." not in captured["Mean fidelity with 95% confidence intervals"]["text"]
