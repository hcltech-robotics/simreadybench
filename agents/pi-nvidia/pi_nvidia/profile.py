"""Explicit endpoint metadata; unspecified model capabilities are not defaults."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

THINKING = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
ENDPOINT = "https://integrate.api.nvidia.com/v1"
PI_VERSION = "1.4.0"


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    input: list[Literal["text", "image"]] = Field(min_length=1)
    contextWindow: int = Field(gt=0)
    maxTokens: int = Field(gt=0)
    reasoning: bool
    toolCalls: Literal[True]
    thinkingLevelMap: dict[str, str | None]
    compat: dict[str, Any]
    evidence: str = Field(min_length=1)
    inputLimits: dict[str, Any] | None = None
    samplingParams: dict[str, Any] | None = None
    samplingParamsByThinkingLevel: dict[str, dict[str, Any]] | None = None

    @model_validator(mode="after")
    def explicit_capabilities(self):
        if self.id.startswith("SELECT_") or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./:@-]+", self.id):
            raise ValueError("Select an exact NVIDIA vendor/model ID")
        if len(set(self.input)) != len(self.input) or "text" not in self.input:
            raise ValueError("Declare unique input modalities including text")
        if self.maxTokens > self.contextWindow:
            raise ValueError("maxTokens exceeds the declared contextWindow")
        if self.reasoning:
            if set(self.thinkingLevelMap) != set(THINKING):
                raise ValueError("Map every Pi thinking level explicitly; null means unsupported")
            if not any(v is not None for v in self.thinkingLevelMap.values()):
                raise ValueError("At least one thinking level must be supported")
            if "thinkingFormat" not in self.compat:
                raise ValueError("Declare the model-specific thinkingFormat")
        elif self.thinkingLevelMap:
            raise ValueError("Non-reasoning models use only off and an empty thinkingLevelMap")
        required = {"supportsStore", "supportsDeveloperRole", "supportsReasoningEffort",
                    "supportsUsageInStreaming", "maxTokensField"}
        if not required <= self.compat.keys():
            raise ValueError("Declare store/developer/reasoning/stream-usage/max-token compatibility")
        for key in required - {"maxTokensField"}:
            if type(self.compat[key]) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if self.compat["maxTokensField"] not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported maximum-token field")
        return self

    def supports(self, effort):
        return effort in THINKING and (effort == "off" if not self.reasoning
                                      else self.thinkingLevelMap[effort] is not None)

    def pi_model(self):
        return self.model_dump(exclude={"evidence", "toolCalls"}, exclude_none=True)
