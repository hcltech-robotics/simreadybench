#!/usr/bin/env python3
"""Build the public NV25 explorer manifest from allowlisted measured data.

The work-data directory is private build input. Only named public fields and
hashes enter the manifest; source paths and authoring content never do.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SCORES = ("formal", "visual", "kinematic")
USAGE = (
    "input_tokens", "output_tokens", "total_tokens", "elapsed_seconds",
    "model_request_seconds", "inference_cost_usd", "compute_cost_usd",
    "total_cost_usd",
)
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
SHA256 = re.compile(r"[a-f0-9]{64}\Z")
PIN = re.compile(r"[a-f0-9]{40}\Z")
PUBLIC_PREFIX = "assets/explorer/"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measurement(value, *, maximum=None):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Measurements must be finite nonnegative numbers or null")
    if maximum is not None and value > maximum:
        raise ValueError("Measurement exceeds its declared range")
    return value


def source_hash(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("A source SHA-256 is required")
    return value


def public_url(value):
    if not isinstance(value, str) or not value.startswith("https://huggingface.co/datasets/"):
        raise ValueError("A public Hugging Face dataset URL is required")
    return value


def load_jobs(work_data):
    jobs = {}
    for name in ("public-export-jobs.json", "private-export-jobs.json"):
        data = read_json(work_data / name)
        for job in data.get("jobs", []) if isinstance(data, dict) else data:
            key = (job["asset_id"], job["condition_id"])
            if key in jobs:
                raise ValueError("Duplicate geometry job identity")
            jobs[key] = job
    return jobs


def geometry(asset_id, condition_id, job, receipts, output_dir):
    if not SLUG.fullmatch(asset_id) or not SLUG.fullmatch(condition_id):
        raise ValueError("Invalid public object or condition identifier")
    relative = f"{asset_id}/{condition_id}.glb"
    source = source_hash(job["source_sha256"])
    receipt = receipts.get(f"{asset_id}/{condition_id}")
    result = {"src": PUBLIC_PREFIX + relative, "source_sha256": source,
              "geometry_status": "pending", "sha256": None}
    if receipt is None:
        return result
    if receipt["status"] == "unavailable":
        result.update(src=None, geometry_status="unavailable",
                      geometry_note="A faithful geometry preview is unavailable for this reconstruction.")
        return result
    if receipt["status"] != "available" or receipt["source_sha256"] != source:
        raise ValueError("Geometry receipt differs from the selected source asset")
    path = output_dir / relative
    actual = digest(path)
    if actual != receipt["glb_sha256"] or path.stat().st_size != receipt["bytes"]:
        raise ValueError("Geometry file differs from its export receipt")
    result.update(geometry_status="available", sha256=actual,
                  bytes=receipt["bytes"], triangles=receipt["triangles"],
                  bounds_m=receipt["bounds_m"])
    return result


def build_manifest(work_data, output_dir):
    work_data, output_dir = Path(work_data), Path(output_dir)
    discovery = read_json(work_data / "data-discovery.json")
    jobs = load_jobs(work_data)
    receipt_path = work_data / "export-receipts.json"
    receipts = read_json(receipt_path) if receipt_path.exists() else {}
    objects = []
    for asset in discovery["assets"]:
        asset_id = asset["id"]
        reference_job = jobs[(asset_id, "reference")]
        reference = geometry(asset_id, "reference", reference_job, receipts, output_dir)
        reference["source_url"] = public_url(reference_job["source_url"])
        variants = []
        for condition in discovery["conditions"]:
            matches = [row for row in condition["rows"] if row["asset_id"] == asset_id]
            if len(matches) != 1 or matches[0]["repetition"] != 0:
                raise ValueError("Exactly repetition zero is required for each selected condition")
            row = matches[0]
            generation = row.get("generation") or condition.get("generation") or {}
            variant = {
                "id": condition["id"], "label": condition["label"],
                "model": row["requested_model"], "harness": row["harness_id"],
                "reasoning": row.get("reasoning_effort") or generation.get("reasoning_effort"),
                "repetition": row["repetition"], "condition_id": row["condition_id"],
                "run_id": condition["run_id"], "source_url": public_url(condition["url"]),
                "source_results_sha256": source_hash(condition["source_sha256"]),
                "evaluator_version": condition.get("evaluator_version"),
                "authoring_protocol": row.get("authoring_protocol") or condition.get("authoring_protocol"),
                "output_limit": generation.get("output_limit"),
                "max_output_tokens": generation.get("max_output_tokens"),
                "inference_cost_kind": row.get("inference_cost_kind"),
                "scores": {
                    name: {"value": measurement(row.get(f"{name}_fidelity_percent"), maximum=100),
                           "status": row.get(f"{name}_fidelity_status"),
                           "coverage": measurement(row.get(f"{name}_fidelity_coverage"), maximum=1)}
                    for name in SCORES
                },
                "usage": {name: measurement(row.get(name)) for name in USAGE},
            }
            pin = condition.get("orichalcum_pin")
            if pin is not None:
                if not PIN.fullmatch(pin):
                    raise ValueError("Orichalcum attribution must be a source pin")
                variant["orichalcum_pin"] = pin
            variant.update(geometry(asset_id, condition["id"], jobs[(asset_id, condition["id"])],
                                    receipts, output_dir))
            variants.append(variant)
        objects.append({"id": asset_id, "label": asset["label"],
                        "reference": reference, "variants": variants})
    result = {
        "schema_version": 1,
        "selection_policy": discovery["selection"],
        "dataset": discovery["dataset"], "dataset_revision": discovery["dataset_revision"],
        "reference_dataset": discovery["reference_dataset"],
        "reference_revision": discovery["reference_revision"],
        "reference_license": {
            "label": "NVIDIA SimReady Assets · CC BY 4.0",
            "url": "https://huggingface.co/datasets/nvidia/simready-assets/blob/"
                   + discovery["reference_revision"] + "/"
                   + discovery["assets"][0]["id"] + "/.metadata/LICENSE.txt",
            "local_url": PUBLIC_PREFIX + "NVIDIA-ASSET-LICENSE.txt",
        },
        "projection_note": "Static geometry previews use neutral shading. Textures, materials, physics and joint motion are not reproduced. Scores are the original measured benchmark results.",
        "usage_note": "Cost means pricing-derived inference cost where recorded; compute and total cost remain unavailable. Tokens include recorded input and output usage. Time is elapsed reconstruction time. Null means unavailable; it is never replaced by zero.",
        "objects": objects,
    }
    validate_metric_identity(result, discovery)
    return result


def validate_metric_identity(manifest, discovery):
    expected = {(row["asset_id"], condition["id"]): (condition, row)
                for condition in discovery["conditions"] for row in condition["rows"]}
    seen = set()
    for obj in manifest["objects"]:
        for variant in obj["variants"]:
            key = (obj["id"], variant["id"])
            if key in seen or key not in expected:
                raise ValueError("Unexpected or duplicate measured trial identity")
            seen.add(key)
            condition, row = expected[key]
            for field in ("condition_id", "repetition"):
                if variant[field] != row[field]:
                    raise ValueError("Viewer trial identity differs from source")
            if variant["run_id"] != condition["run_id"]:
                raise ValueError("Viewer run identity differs from source")
            for name in SCORES:
                score = variant["scores"][name]
                for target, suffix in (("value", "percent"), ("status", "status"), ("coverage", "coverage")):
                    if score[target] != row.get(f"{name}_fidelity_{suffix}"):
                        raise ValueError("Viewer score differs from its measured trial")
            if any(variant["usage"][name] != row.get(name) for name in USAGE):
                raise ValueError("Viewer usage differs from its measured trial")
    if seen != set(expected):
        raise ValueError("A selected measured trial is absent from the viewer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_data", type=Path)
    parser.add_argument("output_dir", type=Path, help="The site/assets/explorer directory")
    parser.add_argument("--check", action="store_true", help="Check the existing manifest without writing")
    parser.add_argument("--require-geometry", action="store_true", help="Require every export to have succeeded or explicitly failed")
    args = parser.parse_args()
    manifest = build_manifest(args.work_data, args.output_dir)
    if args.require_geometry:
        projections = [p for obj in manifest["objects"] for p in [obj["reference"], *obj["variants"]]]
        if any(p["geometry_status"] == "pending" for p in projections):
            raise ValueError("Geometry exports are still pending")
    destination = args.output_dir / "manifest.json"
    if args.check:
        if read_json(destination) != manifest:
            raise ValueError("Published manifest differs from the pinned source snapshot or geometry receipts")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    count = sum(len(obj["variants"]) for obj in manifest["objects"])
    print(f"Verified {len(manifest['objects'])} objects and {count} exact measured trial identities.")


if __name__ == "__main__":
    main()
