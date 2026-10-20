import csv
import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path

import pytest

from simready_bench.reporting import (
    USAGE_FIELDS,
    USAGE_HEADINGS,
    _statistics,
    compact_report,
    export_report,
    usage_columns,
)


def usage_condition():
    return {
        "condition_id": "usage-fixture",
        "asset_id": "box",
        "status": "completed",
        "repetition": 0,
        "inference_calls_used": 2,
        "validation_calls_used": 3,
        "elapsed_seconds": 9.5,
        "inference_cost_usd": 0.0000125,
        "compute_cost_usd": 0.0000025,
        "gateway_records": [
            {
                "status": "completed",
                "upstream_status": 200,
                "elapsed_seconds": 1.25,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 60},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            },
            {
                "status": "completed",
                "upstream_status": 200,
                "elapsed_seconds": 2.75,
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
        ],
    }


class FirstTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.rows = []
        self.cells = []
        self.parts = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        elif tag == "tr":
            self.cells = []
        elif tag in ("th", "td"):
            self.parts = []
            self.in_cell = True

    def handle_data(self, data):
        if self.in_cell:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in ("th", "td"):
            self.cells.append("".join(self.parts))
            self.in_cell = False
        elif tag == "tr":
            self.rows.append(self.cells)
        elif tag == "table":
            self.tables.append(self.rows)


def test_usage_survives_export_compaction_csv_and_main_table(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    original = usage_condition()
    before = deepcopy(original)
    exported = export_report(
        work, tmp_path / "report",
        results={"schema": "simready-bench-matrix-result-v1", "results": [original]},
    )
    report = json.loads(Path(exported["results"]).read_text())
    row = report["results"][0]
    expected = {
        "inference_calls_used": 2, "validation_calls_used": 3,
        "model_request_errors": 0, "elapsed_seconds": 9.5,
        "model_request_seconds": 4.0, "input_tokens": 150,
        "output_tokens": 30, "total_tokens": 180,
        "inference_cost_usd": 0.0000125, "compute_cost_usd": 0.0000025,
        "total_cost_usd": 0.000015,
    }
    assert usage_columns(row) == pytest.approx(expected)
    assert usage_columns(compact_report(report)["results"][0]) == pytest.approx(expected)
    assert original == before
    with Path(exported["scores"]).open(newline="") as stream:
        flat = next(csv.DictReader(stream))
    for field in USAGE_FIELDS:
        assert float(flat[field]) == pytest.approx(expected[field])
    parsed = FirstTable()
    parsed.feed(Path(exported["html"]).read_text())
    headings, values = parsed.tables[0]
    cells = dict(zip(headings, values))
    assert set(USAGE_HEADINGS) <= cells.keys()
    assert cells["Input tokens"] == "150"
    assert cells["Model requests"] == "2"
    assert cells["Elapsed time (s)"] == "9.5"
    assert float(cells["Inference cost (USD)"]) == pytest.approx(0.0000125)


def test_incomplete_records_keep_usage_unknown_instead_of_counting_partial_totals():
    row = usage_condition()
    row["gateway_records"].pop()
    _, statistics = _statistics(row)
    for field in ("input_tokens", "output_tokens", "total_tokens", "model_request_seconds", "model_request_errors"):
        assert statistics[field] is None
    assert statistics["inference_calls_used"] == 2
    assert statistics["elapsed_seconds"] == 9.5


def test_failed_request_counts_and_time_survive_without_terminal_token_usage():
    row = usage_condition()
    row["gateway_records"][1].update(status="failed", upstream_status=503)
    row["gateway_records"][1]["partial_usage"] = row["gateway_records"][1].pop("usage")
    _, statistics = _statistics(row)
    assert statistics["model_request_errors"] == 1
    assert statistics["model_request_seconds"] == 4
    assert statistics["input_tokens"] is statistics["output_tokens"] is None
    assert statistics["total_tokens"] is None


def test_observed_zero_and_unavailable_usage_remain_distinct():
    _, zero = _statistics({"inference_calls_used": 0, "gateway_records": []})
    _, unknown = _statistics({})
    for field in ("input_tokens", "output_tokens", "total_tokens", "model_request_seconds", "model_request_errors"):
        assert zero[field] == 0
        assert unknown[field] is None
    values = usage_columns({"inference_cost_usd": 0, "statistics": {"compute_cost_usd": None}})
    assert values["inference_cost_usd"] == 0
    assert values["compute_cost_usd"] is values["total_cost_usd"] is None


@pytest.mark.parametrize("value", [False, -1, float("nan"), float("inf"), "12"])
def test_invalid_usage_is_not_published_as_a_measurement(value):
    row = {field: value for field in USAGE_FIELDS}
    assert all(item is None for item in usage_columns(row).values())


def test_fractional_token_counts_and_request_counts_are_unmeasured():
    assert usage_columns({"input_tokens": 1.5, "output_tokens": 2})["total_tokens"] is None
    _, statistics = _statistics({"inference_calls_used": False, "gateway_records": []})
    assert statistics["input_tokens"] is None
