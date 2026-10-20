"""Reference source identities for the established asset set."""

import re


def validate_freeze(value):
    required = {"id", "version", "source_files", "conformance_profile"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("Exact benchmark freeze fields required")
    if not all(
        isinstance(value[k], str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value[k])
        for k in ("id", "version")
    ):
        raise ValueError("Explicit benchmark identity required")
    if value["conformance_profile"] != {"id": "Robotics-Prop", "version": "4.0.0"}:
        raise ValueError("Unsupported frozen conformance profile")
    from simready_bench.config import _relative

    for field in ("source_files",):
        if (
            type(value[field]) is not dict
            or not value[field]
            or len(value[field]) > 10000
        ):
            raise ValueError("Bounded complete freeze file map required")
        for path, pin in value[field].items():
            _relative(path, "freeze member")
            if (
                type(pin) is not dict
                or set(pin) != {"sha256", "bytes"}
                or not isinstance(pin["sha256"], str)
                or not re.fullmatch("[0-9a-f]{64}", pin["sha256"])
                or type(pin["bytes"]) is not int
                or pin["bytes"] < 0
            ):
                raise ValueError("Explicit file size and SHA256 required")
    return value


def verify_freeze(freeze, assets):
    validate_freeze(freeze)
    for asset in assets:
        for member in asset["closure"]:
            pin = freeze["source_files"].get(member["path"])
            if pin != {"sha256": member["sha256"], "bytes": member["size_bytes"]}:
                raise ValueError("Frozen source differs: " + member["path"])
