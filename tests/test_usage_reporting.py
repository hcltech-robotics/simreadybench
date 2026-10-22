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


def priced_request():
    from simready_bench.engine.hf_pricing import HF_PRICING_CHECKED
    return {
        "gateway_request_id": "priced-request",
        "model": "Qwen/Qwen3.5-397B-A17B:deepinfra",
        "provider_reported_models": ["Qwen/Qwen3.5-397B-A17B"],
        "api_path": "/v1/chat/completions",
        "status": "completed", "upstream_status": 200,
        "request_sha256": "a" * 64, "response_sha256": "b" * 64,
        "route_metadata_sha256": "c" * 64,
        "pricing_source": "https://router.huggingface.co/v1/models",
        "pricing_checked": HF_PRICING_CHECKED,
        "reserved_upper_cost_micro_usd": 200000,
        "maximum_output_tokens": 4096,
        "actual_cost_usd": None,
        "elapsed_seconds": 2.5,
        "usage": {
            "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 800},
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    }


def test_live_flat_hf_cost_uses_verified_terminal_totals_without_double_counting():
    row = {"inference_calls_used": 1, "inference_attempts_used": 1,
           "gateway_records": [priced_request()], "elapsed_seconds": 9}
    requests, measured = _statistics(row)
    assert measured["inference_cost_usd"] == pytest.approx(0.00075)
    assert measured["inference_cost_kind"] == "pricing-derived inference cost"
    assert measured["input_tokens"] == 1000
    assert measured["output_tokens"] == 100
    assert measured["model_request_seconds"] == 2.5
    assert measured["elapsed_seconds"] == 9
    assert measured["pricing_derived_cost_complete"] is True
    assert requests[0]["actual_cost_usd"] is None
    assert requests[0]["pricing_derived_inference_cost_usd"] == pytest.approx(0.00075)


@pytest.mark.parametrize("status", ["cancelled", "transport_failed"])
def test_attempt_timing_and_priced_subtotal_include_failed_and_cancelled_attempts(status):
    failed = {**priced_request(), "gateway_request_id": "failed", "status": status,
              "elapsed_seconds": 4, "error_code": "upstream_transport_error"}
    failed["partial_usage"] = failed.pop("usage")
    _, measured = _statistics({"inference_calls_used": 1, "inference_attempts_used": 2,
                              "gateway_records": [priced_request(), failed]})
    assert measured["inference_calls_used"] == 1
    assert measured["inference_attempts_used"] == 2
    assert measured["model_request_seconds"] == 6.5
    assert measured["model_request_errors"] == 1
    assert measured["inference_cost_usd"] is None
    assert measured["input_tokens"] is None
    assert measured["priced_requests"] == 1
    assert measured["measured_pricing_derived_cost_usd"] == pytest.approx(0.00075)
    assert measured["pricing_derived_cost_complete"] is False


@pytest.mark.parametrize("change", [
    {"pricing_source": "https://untrusted.example"},
    {"route_metadata_sha256": None},
    {"provider_reported_models": ["other/model"]},
    {"reserved_upper_cost_micro_usd": None},
    {"usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 5}},
])
def test_live_price_requires_verified_route_and_complete_consistent_usage(change):
    _, measured = _statistics({"inference_calls_used": 1,
        "gateway_records": [{**priced_request(), **change}]})
    assert measured["inference_cost_usd"] is None
    assert measured["priced_requests"] == 0


def test_live_pricing_preserves_prior_cost_and_complete_measurements():
    _, measured = _statistics({"inference_calls_used": 1,
        "inference_cost_usd": 3, "statistics": {"model_request_seconds": 7},
        "gateway_records": [priced_request()]})
    assert measured["inference_cost_usd"] == 3
    assert measured["model_request_seconds"] == 7
    assert "inference_cost_kind" not in measured


def test_live_cost_and_attempt_measurements_survive_native_export(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    row = {"condition_id": "native-priced", "asset_id": "box", "status": "completed",
           "inference_calls_used": 1, "inference_attempts_used": 1,
           "gateway_records": [priced_request()]}
    exported = export_report(work, tmp_path / "report", results={
        "schema": "simready-bench-matrix-result-v1", "results": [row],
    })
    recovered = json.loads(Path(exported["results"]).read_text())["results"][0]
    assert recovered["inference_cost_usd"] == pytest.approx(0.00075)
    assert recovered["inference_cost_kind"] == "pricing-derived inference cost"
    assert recovered["statistics"]["priced_requests"] == 1
    assert recovered["statistics"]["inference_attempts_used"] == 1
    assert recovered["model_requests"][0]["actual_cost_usd"] is None
