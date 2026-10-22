"""Vision model metadata and reasoning configuration."""

from copy import deepcopy


def model_configuration(provider, model, metadata=None):
    from simready_bench.engine.request_cost import native_output_facts

    adapter = "openai-completions" if provider == "huggingface" else "openai-responses"
    if metadata is None:
        facts = native_output_facts(model)
        reasoning = model in ("gpt-6.1-sol", "gpt-6-astra")
        metadata = {
            "contextWindow": facts["context_window"],
            "maxTokens": facts["output_token_upper_bound"],
            "input": ["text", "image"],
            "reasoning": reasoning,
            "compat": {
                "supportsReasoningEffort": reasoning,
                "supportsMaxOutputTokens": False,
            },
        }
    else:
        metadata = deepcopy(metadata)
    required = {"contextWindow", "maxTokens", "input", "reasoning"}
    if not isinstance(metadata, dict) or not required <= set(metadata):
        raise ValueError(
            "Model metadata requires contextWindow, maxTokens, input and reasoning"
        )
    if metadata["input"] not in (["text", "image"], ["image", "text"]):
        raise ValueError("The model must accept text and images")
    for name in ("contextWindow", "maxTokens"):
        if type(metadata[name]) is not int or not 0 < metadata[name] <= 10000000:
            raise ValueError(
                "Model context and output capacities must be positive integers"
            )
    if (
        metadata["maxTokens"] > metadata["contextWindow"]
        or type(metadata["reasoning"]) is not bool
    ):
        raise ValueError("Model output capacity must fit its context window")
    configuration = {
        "id": "selected",
        "provider_ref": "selected",
        "model_id": model,
        "adapter": adapter,
        "metadata": metadata,
        "generation": {"output_limit": "provider-native", "max_output_tokens": None},
    }
    if model in ("gpt-6.1-sol", "gpt-6-astra"):
        configuration["generation"]["reasoning_effort"] = "medium"
    return configuration


def configure_reasoning(model, effort=None):
    chosen = effort or model.get("generation", {}).get("reasoning_effort")
    if chosen is None:
        return
    if chosen not in ("off", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Choose off, low, medium, high, xhigh or max reasoning")
    metadata = model["metadata"]
    levels = metadata.get("thinkingLevelMap", {})
    if chosen == "off" and "off" in levels and levels["off"] is None:
        raise ValueError("The selected model cannot disable reasoning")
    if chosen != "off" and (
        not metadata["reasoning"]
        or metadata.get("compat", {}).get("supportsReasoningEffort") is False
    ):
        raise ValueError("The selected model does not support configurable reasoning")
    model.setdefault("generation", {})["reasoning_effort"] = chosen
