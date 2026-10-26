# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Standalone Harbor reward-JSONL metric; never silently call unknown a failure."""

import argparse
import json
import math
from pathlib import Path

DIMENSIONS = ("formal", "visual", "kinematic")
SUFFIXES = ("known_credit", "known", "measurement_coverage", "lower", "upper")
KEYS = {name + "_" + suffix for name in DIMENSIONS for suffix in SUFFIXES} | {
    "contract_version", "candidate_invalid", "evaluation_complete"}


def valid_reward(row):
    if not isinstance(row, dict) or set(row) != KEYS:
        return False
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not 0 <= value <= 1 for value in row.values()):
        return False
    if row["contract_version"] != 1 or any(row[key] not in (0, 1)
        for key in ("candidate_invalid", "evaluation_complete")):
        return False
    for name in DIMENSIONS:
        known, credit = row[name + "_known"], row[name + "_known_credit"]
        low, high = row[name + "_lower"], row[name + "_upper"]
        if known not in (0, 1) or not low <= high or (not known and credit != 0):
            return False
        if known and not math.isclose(low, credit, abs_tol=1e-12):
            return False
        if known and not math.isclose(high, credit, abs_tol=1e-12):
            return False
    return row["evaluation_complete"] == int(all(row[n + "_known"] for n in DIMENSIONS))


def aggregate(rewards):
    rows = [row if valid_reward(row) else None for row in rewards]
    total = len(rows)
    result = {"attempt_count": total,
              "unavailable_verifier_count": sum(row is None for row in rewards),
              "invalid_contract_count": sum(row is not None and not valid_reward(row) for row in rewards),
              "candidate_invalid_count": sum(row["candidate_invalid"] for row in rows if row is not None),
              "complete_evaluation_count": sum(row["evaluation_complete"] for row in rows if row is not None)}
    for name in DIMENSIONS:
        known = sum(row[name + "_known"] for row in rows if row is not None)
        result.update({name + "_known_count": known,
                       name + "_unknown_count": total - known,
                       name + "_known_fraction": known / total if total else 0,
                       name + "_mean_available": int(known > 0),
                       name + "_measurement_coverage_mean": sum(row[name + "_measurement_coverage"]
                           for row in rows if row is not None) / total if total else 0,
                       name + "_lower_bound_mean": 100 * sum(row[name + "_lower"]
                           for row in rows if row is not None) / total if total else 0,
                       name + "_upper_bound_mean": 100 * sum(row[name + "_upper"] if row is not None
                           else 1 for row in rows) / total if total else 100})
        if known:
            result[name + "_fidelity_mean_known"] = 100 * sum(row[name + "_known_credit"]
                for row in rows if row is not None) / known
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"invalid_json": 1})
    args.output.write_text(json.dumps(aggregate(rows), sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
