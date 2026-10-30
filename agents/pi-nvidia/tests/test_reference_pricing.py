import pytest

from pi_nvidia import accounting


MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"


@pytest.mark.parametrize("cached,expected", [(800, "0.000122"), (None, "0.00025")])
def test_baseten_reference_subtracts_cached_subset_and_never_adds_reasoning(cached, expected):
    usage = {"inputTokens": 1000, "outputTokens": 100, "thoughtTokens": 60}
    if cached is not None:
        usage["cachedReadTokens"] = cached
    receipt = {"status": "verified", "usage": usage}
    result = accounting.baseten_reference(receipt, MODEL)
    assert result["usd"] == expected
    assert result["billed"] is False and result["provider"] == "Baseten reference only"
    assert "cost" not in usage and "cost_usd" not in receipt


def test_baseten_reference_does_not_price_unknown_or_invalid_usage():
    for receipt, model in [
        ({"status": "unavailable", "usage": {}}, MODEL),
        ({"status": "verified", "usage": {"inputTokens": 1}}, MODEL),
        ({"status": "verified", "usage": {"inputTokens": 1, "outputTokens": 2, "cachedReadTokens": 3}}, MODEL),
        ({"status": "verified", "usage": {"inputTokens": 1, "outputTokens": 2}}, "unknown/model"),
    ]:
        assert accounting.baseten_reference(receipt, model) is None
