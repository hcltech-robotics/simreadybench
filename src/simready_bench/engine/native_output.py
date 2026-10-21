"""Versioned authoring transport policy; substantive grading remains unchanged."""

import re


DEFAULT_AUTHORING_TURNS = 64
MAX_AUTHORING_TURNS = 128
REQUEST_SECONDS = 1800
AUTHORING_PROTOCOL = "authoring-64-native-output-v1"
NATIVE_OUTPUT_POLICY = "provider-native"
MAX_OUTPUT_TOKENS = 10000000
EXTENSION_FILENAME = "simready-bench-native-output-policy.ts"
EXTENSION_PATH = "/task/" + EXTENSION_FILENAME

# Pi0.85.1's native extension hook runs after its provider builds the request and
# before the existing SDK sends it. No loop/provider/catalog is reimplemented.
EXTENSION_SOURCE = """export default function (pi) {
  pi.on("before_provider_request", (event) => {
    if (!event.payload || typeof event.payload !== "object" || Array.isArray(event.payload)) {
      throw new Error("Native output policy requires an object payload");
    }
    const payload = { ...event.payload };
    delete payload.max_tokens;
    delete payload.max_completion_tokens;
    delete payload.max_output_tokens;
    return payload;
  });
}
"""


def authoring_protocol(turns=DEFAULT_AUTHORING_TURNS, max_output_tokens=None):
    if type(turns) is not int or not 1 <= turns <= MAX_AUTHORING_TURNS:
        raise ValueError("Authoring turns must be an integer between 1 and 128")
    validate_output_cap(max_output_tokens)
    output = "native-output" if max_output_tokens is None else "bounded-output-" + str(max_output_tokens)
    return "authoring-" + str(turns) + "-" + output + "-v1"


def authoring_turns(protocol):
    match = re.fullmatch(r"authoring-([1-9][0-9]*)-(?:native-output|bounded-output-([1-9][0-9]*))-v1", protocol or "")
    if match is None:
        raise ValueError("Unknown authoring protocol")
    turns = int(match[1])
    cap = int(match[2]) if match[2] is not None else None
    if authoring_protocol(turns, cap) != protocol:
        raise ValueError("Unknown authoring protocol")
    return turns


def authoring_seconds(turns=DEFAULT_AUTHORING_TURNS):
    """Allow each turn a request and tool interval plus shutdown time."""
    authoring_protocol(turns)
    return turns * (REQUEST_SECONDS + 1800) + 600


def native_output(condition):
    try:
        authoring_turns(condition.get("authoring_protocol"))
    except (ValueError, TypeError):
        return False
    return condition["authoring_protocol"].endswith("-native-output-v1") and condition.get("settings", {}).get("generation", {}).get("max_output_tokens") is None


def validate_output_cap(value):
    if value is not None and (type(value) is not int or not 16 <= value <= MAX_OUTPUT_TOKENS):
        raise ValueError("Output token cap must be an integer between 16 and 10000000")
    return value
