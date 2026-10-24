"""Public explorer data must stay attached to the measured reconstruction."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_viewer_manifest.py"
spec = importlib.util.spec_from_file_location("build_viewer_manifest", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def selected_trials(tmp_path):
    work, output = tmp_path / "work", tmp_path / "site"
    work.mkdir()
    output.mkdir()
    assets = [{"id": f"object_{number}", "label": f"Object {number}"} for number in range(5)]
    conditions, jobs = [], []
    url = "https://huggingface.co/datasets/owner/benchmark/resolve/" + "a" * 40
    for asset in assets:
        jobs.append({"asset_id": asset["id"], "condition_id": "reference",
                     "source_sha256": "1" * 64, "source_url": url + "/reference.usd",
                     "input_usd": "/private/path/that/must/not/be/published"})
    for number in range(4):
        condition = {"id": f"condition_{number}", "label": f"Model {number}",
                     "run_id": f"run_{number}", "url": url + f"/run_{number}/results.json",
                     "source_sha256": str(number + 2) * 64, "rows": [],
                     "orichalcum_pin": "b" * 40 if number == 3 else None,
                     "private_notes": "DO_NOT_PUBLISH_PRIVATE_RECIPE"}
        for asset_number, asset in enumerate(assets):
            row = {"asset_id": asset["id"], "condition_id": f"trial_{asset_number}_{number}",
                   "repetition": 0, "requested_model": f"model_{number}",
                   "harness_id": "codex" if number == 3 else "pi", "reasoning_effort": "high",
                   "generation": {"output_limit": "provider-native", "max_output_tokens": None}}
            for index, name in enumerate(builder.SCORES):
                row[f"{name}_fidelity_percent"] = asset_number + number + index / 10
                row[f"{name}_fidelity_status"] = "measured"
                row[f"{name}_fidelity_coverage"] = 1
            for index, name in enumerate(builder.USAGE):
                row[name] = 100 * asset_number + 10 * number + index
            if number == 3:
                row["inference_cost_usd"] = None
                row["total_cost_usd"] = None
            condition["rows"].append(row)
            jobs.append({"asset_id": asset["id"], "condition_id": condition["id"],
                         "source_sha256": "9" * 64,
                         "input_usd": "/private/path/that/must/not/be/published"})
        conditions.append(condition)
    conditions[0]["rows"][0]["kinematic_fidelity_percent"] = None
    conditions[0]["rows"][0]["kinematic_fidelity_status"] = "not_applicable"
    conditions[1]["rows"][0]["kinematic_fidelity_percent"] = 0
    discovery = {"assets": assets, "conditions": conditions, "selection": "First repetition uniformly.",
                 "dataset": "owner/benchmark", "dataset_revision": "a" * 40,
                 "reference_dataset": "nvidia/simready-assets", "reference_revision": "c" * 40}
    write_json(work / "data-discovery.json", discovery)
    write_json(work / "public-export-jobs.json", {"jobs": jobs})
    write_json(work / "private-export-jobs.json", [])
    return work, output, discovery


def test_all_twenty_trials_keep_exact_metric_and_usage_identity(selected_trials):
    work, output, discovery = selected_trials
    manifest = builder.build_manifest(work, output)
    assert len(manifest["objects"]) == 5
    assert sum(len(obj["variants"]) for obj in manifest["objects"]) == 20
    for obj in manifest["objects"]:
        for variant in obj["variants"]:
            condition = next(c for c in discovery["conditions"] if c["id"] == variant["id"])
            row = next(r for r in condition["rows"] if r["asset_id"] == obj["id"])
            assert variant["condition_id"] == row["condition_id"]
            assert variant["source_results_sha256"] == condition["source_sha256"]
            assert variant["usage"] == {name: row[name] for name in builder.USAGE}
            for name in builder.SCORES:
                assert variant["scores"][name] == {
                    "value": row[f"{name}_fidelity_percent"],
                    "status": row[f"{name}_fidelity_status"],
                    "coverage": row[f"{name}_fidelity_coverage"],
                }
    first = manifest["objects"][0]["variants"]
    assert first[0]["scores"]["kinematic"]["value"] is None
    assert first[0]["scores"]["kinematic"]["status"] == "not_applicable"
    assert first[1]["scores"]["kinematic"]["value"] == 0
    assert first[3]["usage"]["inference_cost_usd"] is None


def test_private_build_fields_never_enter_public_manifest(selected_trials):
    work, output, _ = selected_trials
    text = json.dumps(builder.build_manifest(work, output))
    assert "/private/" not in text
    assert "DO_NOT_PUBLISH_PRIVATE_RECIPE" not in text
    assert "input_usd" not in text
    assert "b" * 40 in text


def test_changed_trial_metric_is_detected(selected_trials):
    work, output, discovery = selected_trials
    manifest = builder.build_manifest(work, output)
    manifest["objects"][0]["variants"][0]["usage"]["elapsed_seconds"] += 1
    with pytest.raises(ValueError, match="usage differs"):
        builder.validate_metric_identity(manifest, discovery)


def test_duplicate_source_trial_is_rejected(selected_trials):
    work, output, discovery = selected_trials
    discovery["conditions"][0]["rows"].append(copy.deepcopy(discovery["conditions"][0]["rows"][0]))
    write_json(work / "data-discovery.json", discovery)
    with pytest.raises(ValueError, match="Exactly repetition zero"):
        builder.build_manifest(work, output)


def test_unavailable_geometry_keeps_scores_and_usage_without_source_error(selected_trials):
    work, output, _ = selected_trials
    write_json(work / "export-receipts.json", {
        "object_0/condition_1": {"status": "unavailable", "reason": "/private/source/path"}})
    variant = builder.build_manifest(work, output)["objects"][0]["variants"][1]
    assert variant["src"] is None
    assert variant["geometry_status"] == "unavailable"
    assert variant["scores"]["kinematic"]["value"] == 0
    assert variant["usage"]["input_tokens"] == 10
    assert "/private/" not in json.dumps(variant)


def test_available_geometry_hash_size_and_source_are_verified(selected_trials):
    work, output, _ = selected_trials
    target = output / "object_0" / "condition_0.glb"
    target.parent.mkdir()
    payload = b"known geometry projection"
    target.write_bytes(payload)
    receipt = {"status": "available", "source_sha256": "9" * 64,
               "glb_sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
               "triangles": 1, "bounds_m": [[0, 0, 0], [1, 1, 1]]}
    write_json(work / "export-receipts.json", {"object_0/condition_0": receipt})
    variant = builder.build_manifest(work, output)["objects"][0]["variants"][0]
    assert variant["sha256"] == receipt["glb_sha256"]
    target.write_bytes(b"tampered geometry")
    with pytest.raises(ValueError, match="differs from its export receipt"):
        builder.build_manifest(work, output)
