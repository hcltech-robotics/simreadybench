#!/usr/bin/env python3
"""Rebuild the public Pareto snapshot from a pinned HF results revision.

Only explicit public identifiers, numeric measurements and evidence digests are
exported. Raw trajectories, prompts, tools and private settings never are.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.parse
import urllib.request


REPO = "hcltech-robotics/simreadybench"
BASE = f"https://huggingface.co/datasets/{REPO}"
SCORES = ("formal", "visual", "kinematic")
METRICS = {
    "cost_usd_per_asset": "inference_cost_usd",
    "time_seconds_per_asset": "model_request_seconds",
    "tokens_per_asset": "total_tokens",
    "elapsed_seconds_per_asset": "elapsed_seconds",
}
KINDS = {
    "inference_cost_usd": "pricing-derived inference cost",
    "model_request_seconds": "sum of recorded inference request durations",
    "total_tokens": "input and output tokens, including cached input once",
    "elapsed_seconds": "elapsed reconstruction time, including tool execution",
}


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def complete_sum(values):
    return math.fsum(values) if values and all(number(v) for v in values) else None


def mean_reconstruction(rows, value):
    """Follow leaderboard attempt weighting; the unit is one reconstructed asset."""
    values = [value(row) for row in rows if number(value(row))]
    return math.fsum(values) / len(values) if values else None


def score_value(row, name):
    value = row.get(f"{name}_fidelity_percent")
    if value is not None and (not number(value) or value > 100):
        raise ValueError("Invalid measured fidelity score")
    return value


def is_eligible(row, name):
    eligible = row.get(f"{name}_fidelity_eligible")
    if number(eligible):
        return eligible > 0
    return row.get(f"{name}_fidelity_status") != "not_applicable" and score_value(row, name) is not None


def score_summary(rows, name):
    eligible = [r for r in rows if is_eligible(r, name)]
    measured = [r for r in eligible if score_value(r, name) is not None]
    complete = bool(eligible) and len(measured) == len(eligible)
    return {
        "value": mean_reconstruction(measured, lambda r: score_value(r, name)),
        "status": "measured" if complete else "partial" if measured else "unknown" if eligible else "not_applicable",
        "complete": complete,
        "lower": mean_reconstruction(eligible, lambda r: r.get(f"{name}_fidelity_lower_bound_percent")),
        "upper": mean_reconstruction(eligible, lambda r: r.get(f"{name}_fidelity_upper_bound_percent")),
        "coverage": len(measured) / len(eligible) if eligible else None,
        "measured_attempts": len(measured), "eligible_attempts": len(eligible),
        "measured_assets": len({r["asset_id"] for r in measured}),
        "eligible_assets": len({r["asset_id"] for r in eligible}),
    }


def model_family(model):
    """A model family, independent of provider, version, harness or private inputs."""
    name = model.rsplit("/", 1)[-1].split(":", 1)[0]
    for pattern, family in (
        (r"^gpt-", "GPT"), (r"^claude-", "Claude"),
        (r"^glm-", "GLM"), (r"nemotron", "Nemotron"),
        (r"^deepseek-", "DeepSeek"), (r"^minimax-", "MiniMax"),
        (r"^qwen", "Qwen"), (r"llama", "Llama"), (r"^gemma", "Gemma"),
    ):
        if re.search(pattern, name, re.IGNORECASE):
            return family
    return name


def add_cumulative_scores(runs):
    """Equal-weight mean of three z-scores, with a fixed snapshot population."""
    components = {}
    for name in SCORES:
        values = [run["scores"][name]["value"] for run in runs
                  if number(run["scores"][name].get("value"))]
        mean = math.fsum(values) / len(values) if values else None
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values) if values else None
        components[name] = {"mean": mean, "standard_deviation": math.sqrt(variance) if variance is not None else None,
                            "configuration_count": len(values)}
    for run in runs:
        z_scores = {}
        for name, normaliser in components.items():
            value = run["scores"][name].get("value")
            sigma = normaliser["standard_deviation"]
            z_scores[name] = ((value - normaliser["mean"]) / sigma if sigma else 0.0) if number(value) else None
        measured = all(value is not None for value in z_scores.values())
        complete = measured and all(run["scores"][name].get("complete", False) for name in SCORES)
        run["scores"]["cumulative"] = {
            "value": math.fsum(z_scores.values()) / len(SCORES) if measured else None,
            "status": "measured" if complete else "partial" if measured else "unknown",
            "complete": complete, "components": z_scores, "unit": "mean z-score",
        }
    return {"method": "mean_z_score", "formula": "mean((subscore - population mean) / population standard deviation)",
            "basis": "available subscore means across all configurations in this leaderboard snapshot",
            "ddof": 0, "components": components,
            "missing_components": "All three measured subscore means are required; missing observations are not imputed.",
            "zero_variance": "A constant component contributes zero for every configuration.",
            "axis_invariant": True}


def metric_summary(rows, field):
    measured = [r for r in rows if number(r.get(field))]
    paired = {}
    for name in SCORES:
        eligible = [r for r in rows if is_eligible(r, name)]
        selected = [r for r in eligible if number(r.get(field)) and score_value(r, name) is not None]
        paired[name] = {
            "value": mean_reconstruction(selected, lambda r: score_value(r, name)),
            "x_value": mean_reconstruction(selected, lambda r: r[field]),
            "complete": bool(eligible) and len(selected) == len(eligible),
            "measured_attempts": len(selected), "eligible_attempts": len(eligible),
            "measured_assets": len({r["asset_id"] for r in selected}),
            "eligible_assets": len({r["asset_id"] for r in eligible}),
            "cohort_sha256": canonical_hash(sorted(r["condition_id"] for r in selected)),
            "reference_cohort_sha256": canonical_hash(sorted([r["asset_id"], r["repetition"]] for r in selected)),
        }
    sources = Counter(r.get("_usage_source", {}).get(field, "published per-attempt measurement") for r in measured)
    return {
        "value": mean_reconstruction(measured, lambda r: r[field]),
        "complete": bool(rows) and len(measured) == len(rows),
        "measured_attempts": len(measured), "total_attempts": len(rows),
        "measured_assets": len({r["asset_id"] for r in measured}),
        "total_assets": len({r["asset_id"] for r in rows}),
        "kind": KINDS[field], "source_description": "; ".join(f"{n} attempts: {s}" for s, n in sorted(sources.items())) or "Unavailable in retained evidence",
        "scores": paired,
    }


def reconcile(row, recovered, source):
    row = dict(row)
    row["_usage_source"] = dict(row.get("_usage_source", {}))
    for field, value in recovered.items():
        if value is None:
            continue
        if not number(value):
            raise ValueError("Invalid reconstructed usage")
        recorded = row.get(field)
        if recorded is not None and not math.isclose(recorded, value, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"Reconstructed {field} differs from the published measurement")
        row[field] = value
        row["_usage_source"][field] = source
    return row


def trajectory_usage(trajectory):
    """Pi trajectories contain per-request metrics; error zeros are placeholders."""
    calls = []
    for step in trajectory.get("steps", []):
        m = step.get("metrics") or {}
        if m.get("prompt_tokens") is None and m.get("completion_tokens") is None:
            continue
        if (step.get("extra") or {}).get("stop_reason") == "error" and all(m.get(k) == 0 for k in ("prompt_tokens", "completion_tokens", "cached_tokens")):
            continue
        extra = m.get("extra") or {}
        calls.append({"input_tokens": m.get("prompt_tokens"), "output_tokens": m.get("completion_tokens"),
                      "cached_tokens": m.get("cached_tokens"), "reasoning_tokens": extra.get("reasoning_tokens"),
                      "cache_write_tokens": extra.get("cache_write_tokens")})
    final = trajectory.get("final_metrics") or {}
    for key, target in (("input_tokens", "total_prompt_tokens"), ("output_tokens", "total_completion_tokens")):
        actual = complete_sum([r[key] for r in calls])
        if actual is not None and final.get(target) is not None and actual != final[target]:
            raise ValueError("Trajectory final metrics differ from request metrics")
    return calls


def usage_key(value):
    return tuple(value.get(k) for k in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens"))


def reconstruct_public(row, trajectory=None):
    """Recompute usage from retained request proofs, checking saved trajectories."""
    proof = row.get("usage_reconstruction") or {}
    requests = [dict(r) for r in proof.get("requests") or []]
    source = "retained provider request proofs"
    recovered_cost_authorised = False
    if requests:
        identities = [r.get("gateway_request_id") for r in requests]
        if any(not v for v in identities) or len(set(identities)) != len(identities):
            raise ValueError("Provider proof contains missing or duplicated request identities")
        if trajectory is not None:
            calls = trajectory_usage(trajectory)
            observed = [r for r in requests if number(r.get("input_tokens")) and number(r.get("output_tokens"))]
            actual, expected = Counter(map(usage_key, calls)), Counter(map(usage_key, observed))
            if actual != expected:
                missing = [r for r in requests if not number(r.get("input_tokens")) or not number(r.get("output_tokens"))]
                extra = list((actual - expected).elements())
                recovery = row.get("request_recovery") or {}
                # The explicit recovery receipt binds one missing request to
                # the one additional terminal usage event in this trajectory.
                recoverable = (not expected - actual and len(extra) == len(missing) == 1
                               and recovery.get("disposition") == "verified_recovered"
                               and missing[0]["gateway_request_id"] in recovery.get("resolved_request_ids", []))
                if not recoverable:
                    raise ValueError("Trajectory does not match the provider request proof: " + row["condition_id"])
                call = next(c for c in calls if usage_key(c) == extra[0])
                missing[0].update(call)
                tariff = proof.get("tariff") or {}
                rates = tariff.get("usd_per_million_tokens") or {}
                if tariff.get("billing_basis") == "input_output" and tariff.get("route_verified") and tariff.get("service_tier_verified"):
                    rates = {k: float(rates[k]) for k in ("input", "output")}
                    if not all(number(v) for v in rates.values()):
                        raise ValueError("Invalid retained tariff")
                    missing[0]["pricing_derived_inference_cost_usd"] = (call["input_tokens"] * rates["input"] + call["output_tokens"] * rates["output"]) / 1_000_000
                    recovered_cost_authorised = True
            source = "SHA-256-verified trajectory matched to retained provider request proofs"
        complete = len(requests) == proof.get("recorded_calls")
        inp = complete_sum([r.get("input_tokens") for r in requests]) if complete else None
        out = complete_sum([r.get("output_tokens") for r in requests]) if complete else None
        cost = complete_sum([r.get("pricing_derived_inference_cost_usd") for r in requests]) if complete else None
        row = reconcile(row, {"total_tokens": inp + out if inp is not None and out is not None else None,
                              "inference_cost_usd": cost if proof.get("pricing_derived_cost_complete") or recovered_cost_authorised else None}, source)
    native = row.get("native_anthropic_usage")
    if native:
        total = complete_sum([native.get(k) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")])
        row = reconcile(row, {"total_tokens": total}, "retained native Anthropic usage; cache reads and writes counted once")
    if row.get("input_tokens") is not None and row.get("output_tokens") is not None:
        row = reconcile(row, {"total_tokens": row["input_tokens"] + row["output_tokens"]},
                        row.get("_usage_source", {}).get("total_tokens", "published provider input and output usage"))
    return row


def private_receipt(path, run_id, public_results=None):
    """Reduce private Codex evidence to numeric, prompt-free receipts in memory."""
    path = Path(path)
    data = path.read_bytes()
    document = json.loads(data)
    public_bytes = Path(public_results).read_bytes() if public_results else None
    public = json.loads(public_bytes) if public_bytes else None
    public_rows = {r["condition_id"]: r for r in public["results"]} if public else {}
    evaluator_versions = {(row.get("fidelity") or {}).get("schema") for row in document["results"]}
    source_evaluator_version = next(iter(evaluator_versions)) if len(evaluator_versions) == 1 else None
    if source_evaluator_version is not None and not re.fullmatch(r"source-usd-fidelity-v[0-9]+", source_evaluator_version):
        source_evaluator_version = None
    # A published rescore can preserve the original authoring resource records.
    evaluator_version = (public or {}).get("evaluator_version") or source_evaluator_version
    output = []
    for row in document["results"]:
        if public:
            selected = public_rows.get(row["condition_id"])
            if selected is None or any(row.get(k) != selected.get(k) for k in
                                       ("asset_id", "repetition", "seed", "requested_model", "harness_id", "elapsed_seconds")):
                raise ValueError("Private source does not match the public selected reconstruction")
            if any(score_value(row, name) != score_value(selected, name) for name in SCORES):
                rescore = public.get("evaluation_rescore") or {}
                left = (row.get("candidate") or {}).get("manifest_sha256")
                right = (selected.get("candidate") or {}).get("manifest_sha256")
                if not (left and left == right and rescore.get("model_calls_added") == 0
                        and rescore.get("original_authoring_outputs_and_accounting_preserved") is True):
                    raise ValueError("Private source has different evaluated scores without identical preserved authoring output")
        records = row.get("model_requests") or []
        unique = {v["gateway_request_id"]: v for v in records}
        if len(unique) != len(records):
            raise ValueError("Duplicated private provider records")
        native = []
        digests = []
        for trace in row.get("traces", []):
            if not trace["path"].endswith(".jsonl"):
                continue
            target = (path.parent / "evidence" / trace["path"]).resolve()
            if not target.is_relative_to(path.parent.resolve()):
                raise ValueError("Trace outside run evidence")
            raw = target.read_bytes()
            if sha256(raw) != trace["sha256"]:
                raise ValueError("Private native trace digest mismatch")
            digests.append(trace["sha256"])
            for line in raw.splitlines():
                event = json.loads(line)
                if event.get("type") == "token_usage_record":
                    native.append((event.get("payload") or {}).get("usage") or {})
        observed = [r["usage"] for r in records if r.get("usage")]
        native_key = lambda u: (u.get("input_tokens"), u.get("output_tokens"), u.get("cached_input_tokens"))
        provider_key = lambda u: (u.get("input_tokens"), u.get("output_tokens"), (u.get("input_tokens_details") or {}).get("cached_tokens"))
        matched = bool(native) and Counter(map(native_key, native)) == Counter(map(provider_key, observed))
        recovery = row.get("request_recovery") or {}
        resolved = recovery.get("resolved_request_ids", []) if recovery.get("disposition") == "verified_recovered" else []
        def failed_record(identity):
            record = unique.get(identity)
            return record is not None and (record.get("status") != "completed"
                                           or type(record.get("upstream_status")) is int and record["upstream_status"] >= 400)
        recovered_failures = (len(set(resolved)) if len(set(resolved)) == len(resolved)
                              and all(failed_record(v) for v in resolved) else 0)
        expected_requests = row.get("inference_calls_used")
        if type(expected_requests) is int:
            expected_requests += recovered_failures
        count_complete = bool(records) and len(records) == expected_requests
        complete = count_complete and len(observed) == len(records)
        tokens = complete_sum([u.get("total_tokens") for u in observed]) if complete and matched else None
        seconds = complete_sum([r.get("elapsed_seconds") for r in records]) if count_complete else None
        output.append({"condition_id": row["condition_id"], "asset_id": row["asset_id"], "repetition": row["repetition"],
                       "total_tokens": tokens, "model_request_seconds": seconds,
                       "recorded_requests": len(records), "native_usage_records": len(native),
                       "verified_recovered_request_count": recovered_failures,
                       "native_usage_matched": matched, "provider_usage_complete": complete,
                       "cancelled_requests_without_terminal_usage": sum(r.get("status") == "cancelled" and not r.get("usage") for r in records),
                       "native_trace_sha256": sorted(digests)})
    return {"schema_version": 1, "run_id": run_id, "source_results_sha256": sha256(data),
            "public_results_sha256": sha256(public_bytes) if public_bytes else None,
            "evaluator_version": evaluator_version, "source_evaluator_version": source_evaluator_version,
            "attempts": output}


def get_bytes(url, cache):
    key = cache / sha256(url.encode())
    if key.exists():
        return key.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "simreadybench-pareto/1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        data = response.read()
    key.write_bytes(data)
    return data


def run_identity(entry):
    path = urllib.parse.urlparse(entry["results_url"]).path
    match = re.search(r"/runs/([A-Za-z0-9._-]+)/", path)
    if not match:
        raise ValueError("Leaderboard result has no valid run identity")
    return match[1]


def build_run(entry, document, *, run_id, source_url, results_sha256, trajectory_loader, supplement=None):
    rows = document["results"]
    if not document.get("benchmark_complete") or len(rows) != entry.get("completed_attempts"):
        raise ValueError("Only completed published runs belong in the comparison")
    identities = [r["condition_id"] for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("Duplicate selected reconstruction")
    supplements = {r["condition_id"]: r for r in (supplement or {}).get("attempts", [])}
    if supplement:
        if supplement.get("run_id") != run_id or len(supplements) != len(supplement["attempts"]) or set(supplements) != set(identities):
            raise ValueError("Supplement does not contain this run's exact selected attempts")
        private_hash = (document.get("private_evidence") or {}).get("source_report_sha256")
        bound = (private_hash is not None and supplement.get("source_results_sha256") == private_hash
                 or supplement.get("public_results_sha256") == results_sha256)
        if not bound:
            raise ValueError("Supplement is not bound to the published source snapshot")
    trace_count = 0
    revised = []
    for row in rows:
        trace = next((t for t in row.get("traces", []) if t.get("kind") == "agent_trajectory"), None)
        trajectory = trajectory_loader(trace) if trace is not None else None
        trace_count += trajectory is not None
        row = reconstruct_public(row, trajectory)
        evidence = supplements.get(row["condition_id"])
        if evidence:
            if any(row[k] != evidence[k] for k in ("asset_id", "repetition")):
                raise ValueError("Private receipt refers to a different selected reconstruction")
            row = reconcile(row, {"model_request_seconds": evidence.get("model_request_seconds")},
                            "complete retained provider request duration records")
            row = reconcile(row, {"total_tokens": evidence.get("total_tokens")},
                            "retained native Codex token_usage_record events matched to provider requests")
        revised.append(row)
    pin = next((s.split("@", 1)[1] for s in entry.get("skills", []) if re.fullmatch(r"Orichalcum@[a-f0-9]{40}", s)), None)
    protocol = entry.get("authoring_protocol")
    evaluator_version = entry.get("evaluator_version") or document.get("evaluator_version")
    retained_evaluator = (supplement or {}).get("evaluator_version")
    # Supplements are already bound above to the exact public source snapshot.
    if retained_evaluator and evaluator_version and retained_evaluator != evaluator_version:
        raise ValueError("Private evaluator version differs from the published evaluator")
    evaluator_version = evaluator_version or retained_evaluator
    scores = {name: score_summary(revised, name) for name in SCORES}
    for name, score in scores.items():
        published = (entry.get("composite_scores", {}).get(name) or {}).get("score")
        if score["value"] is not None and published is not None and not math.isclose(score["value"], published, rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("Reconstructed score does not match the leaderboard: " + name)
    label = f"{entry['model']} · {entry['harness']}"
    if entry.get("reasoning_effort"):
        label += " · " + entry["reasoning_effort"]
    if pin:
        label += " · Orichalcum …" + pin[-8:]
    return {
        "id": run_id, "label": label, "model": entry["model"], "model_family": model_family(entry["model"]), "harness": entry["harness"],
        "reasoning": entry.get("reasoning_effort"), "orichalcum_pin": pin,
        "protocol_id": protocol, "protocol": "64 turns · native output limit" if protocol == "authoring-64-native-output-v1" else "64 turns · 4,096-token output limit" if "4096" in (protocol or "") else protocol,
        "frontier_group": canonical_hash([evaluator_version or run_id, entry.get("reference_revision") or run_id,
                                           entry.get("profile") or run_id]),
        "scores": scores,
        "metrics": {key: metric_summary(revised, field) for key, field in METRICS.items()},
        "source_url": source_url, "source_results_sha256": results_sha256,
        "evaluator_version": evaluator_version, "reference_revision": entry.get("reference_revision"),
        "completed_attempts": len(rows), "reference_count": len({r["asset_id"] for r in rows}),
        "provenance": {"verified_public_trajectories": trace_count,
                       "private_native_usage_attempts": sum(r.get("native_usage_matched", False) for r in supplements.values()),
                       "private_source_results_sha256": (supplement or {}).get("source_results_sha256"),
                       "missing_usage_is_zero": False},
    }


def build_snapshot(revision, cache, supplements):
    leaderboard_url = f"{BASE}/resolve/{revision}/leaderboard.json"
    raw = get_bytes(leaderboard_url, cache)
    leaderboard = json.loads(raw)
    entries = [e for e in leaderboard["entries"] if e.get("reference_count") == 25 and e.get("repetitions") == 4 and e.get("completed_attempts") == 100]
    def build(entry):
        run = run_identity(entry)
        base = f"{BASE}/resolve/{revision}/runs/{run}"
        results = get_bytes(base + "/results.json", cache)
        def trajectory_loader(trace):
            relative = trace["path"]
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("Invalid public trace path")
            raw_trace = get_bytes(base + "/evidence/" + urllib.parse.quote(relative, safe="/"), cache)
            if sha256(raw_trace) != trace["sha256"]:
                raise ValueError("Public trajectory digest mismatch")
            return json.loads(raw_trace)
        return build_run(entry, json.loads(results), run_id=run, source_url=base + "/results.json",
                         results_sha256=sha256(results), trajectory_loader=trajectory_loader,
                         supplement=supplements.get(run))
    with ThreadPoolExecutor(max_workers=5) as pool:
        runs = list(pool.map(build, entries))
    normalisation = add_cumulative_scores(runs)
    return {"schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_revision": revision, "source_url": leaderboard_url, "source_sha256": sha256(raw),
            "methodology": "Plot each configuration's available mean score against its available mean resource measurement, following the leaderboard's attempt weighting. Coverage may differ between axes and is reported for each mean. Per asset means one reconstruction, not the sum of four attempts. Kinematic means use eligible measured attempts. Cumulative score is the equal-weight mean of the three z-scores, normalised by population means and standard deviations across all leaderboard configurations in this snapshot; the normalisation is fixed across resource axes. Inference time sums request durations; elapsed reconstruction time is separate. Missing observations remain null and are never zero-filled. Frontiers include partial coverage and compare configurations sharing the reference revision, scoring profile and evaluator. Authoring protocols and output limits remain configuration attributes.",
            "normalisation": normalisation,
            "runs": runs}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("site/assets/pareto.json"))
    p.add_argument("--cache", type=Path, default=Path(".cache/pareto"))
    p.add_argument("--revision")
    p.add_argument("--evidence", type=Path)
    p.add_argument("--private-results", type=Path)
    p.add_argument("--public-results", type=Path, help="Pinned public results to bind a private receipt to")
    p.add_argument("--run-id")
    args = p.parse_args()
    if args.private_results:
        if not args.run_id:
            p.error("--private-results requires --run-id")
        result = private_receipt(args.private_results, args.run_id, args.public_results)
    else:
        args.cache.mkdir(parents=True, exist_ok=True)
        revision = args.revision
        if not revision:
            with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=60) as response:
                revision = json.load(response)["sha"]
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise ValueError("A pinned HF revision is required")
        evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
        receipts = evidence.get("runs", [evidence]) if evidence else []
        supplements = {e["run_id"]: e for e in receipts}
        result = build_snapshot(revision, args.cache, supplements)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {len(result.get('runs', result.get('attempts', [])))} records")


if __name__ == "__main__":
    main()
