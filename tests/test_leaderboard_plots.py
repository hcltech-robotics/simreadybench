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
    assert len(figures) == 3
