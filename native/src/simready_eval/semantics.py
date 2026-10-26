"Read semantic labels from candidate USD assets."

from copy import deepcopy

SCHEMA = "simready-semantic-information-v1"


def observe_labels(stage):
    entries = []
    used = 0
    truncated = False
    for index, prim in enumerate(stage.Traverse()):
        if index >= 10000:
            truncated = True
            break
        for attr in prim.GetAttributes():
            name = attr.GetName()
            modern = name.startswith("semantics:labels:")
            if not modern and (
                not (name.startswith("semantics:") and name.endswith(":semanticData"))
            ):
                continue
            instance = (
                name[len("semantics:labels:") :] if modern else name.split(":")[1]
            )
            value = attr.Get()
            if value is None:
                labels = []
            elif isinstance(value, str):
                labels = [value]
            else:
                try:
                    from itertools import islice

                    bounded = list(islice(iter(value), 17))
                    truncated = truncated or len(bounded) > 16
                    labels = [str(v) for v in bounded[:16]]
                except TypeError:
                    labels = [str(value)]
            labels = [v[:128] for v in labels[:16]]
            row = {
                "prim_path": str(prim.GetPath())[:512],
                "instance": instance[:128],
                "labels": labels,
                "value_type": str(attr.GetTypeName())[:64],
                "format": "labels_api" if modern else "semantic_data_attribute",
                "time_samples": attr.GetNumTimeSamples(),
            }
            import json

            size = len(json.dumps(row, ensure_ascii=True).encode())
            if len(entries) >= 128 or used + size > 32768:
                truncated = True
                break
            entries.append(row)
            used += size
        if truncated:
            break
    return {
        "schema": SCHEMA,
        "status": "reported",
        "required": False,
        "scoring": False,
        "entries": entries,
        "truncated": truncated,
    }


def information(record=None, *, findings=()):
    """Revalidate the worker projection before actor or human presentation."""
    result = {
        "schema": SCHEMA,
        "status": "unavailable",
        "required": False,
        "scoring": False,
        "entries": [],
        "truncated": False,
        "findings": [],
    }
    if (
        isinstance(record, dict)
        and record.get("schema") == SCHEMA
        and (record.get("status") == "reported")
        and isinstance(record.get("entries"), list)
    ):
        import json

        raw = json.dumps(record, allow_nan=False).encode()
        if len(raw) <= 65536 and len(record["entries"]) <= 128:
            result.update(
                status="reported",
                entries=deepcopy(record["entries"]),
                truncated=record.get("truncated") is True,
            )
    for row in findings:
        if (
            isinstance(row, dict)
            and isinstance(row.get("check_id"), str)
            and row["check_id"].startswith("nvidia.com.nvidia.simready.SL.")
        ):
            result["findings"].append(
                {
                    k: deepcopy(row[k])
                    for k in ("check_id", "status", "message", "candidate_prim")
                    if k in row
                }
            )
    return result
