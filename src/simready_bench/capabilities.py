"""Inspect Harbor's installed registry; never maintain a duplicate agent catalog."""

from simready_bench.harbor_agents import normalize_harness


def validate_pi_kwargs(value):
    if not isinstance(value, dict) or set(value) - {"version", "retry_owner"}:
        raise ValueError("Pi accepts fixed version and an explicit retry owner")
    if value.get("version", "0.85.1") != "0.85.1" or value.get(
        "retry_owner", "harness"
    ) not in ("harness", "relay"):
        raise ValueError("Pi requires version=0.85.1 and retry_owner=harness or relay")
    return value


def normalize_native_harness(entry):
    if entry.get("adapter") == "harbor":
        return normalize_harness(entry)
    if entry.get("kwargs"):
        if entry.get("adapter") != "pi":
            raise ValueError("Native options require a Pi or Harbor harness")
        validate_pi_kwargs(entry["kwargs"])
    return entry
