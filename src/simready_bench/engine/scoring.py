"NVIDIA Robotics-Prop 4.0.0 requirement scoring."

from copy import deepcopy
from pathlib import Path
import hashlib, json, math, re
from simready_bench.engine.verifier import (
    WIRE_SCHEMA,
    POLICY_ID,
    digest,
    pinned_profile,
    validate_policy,
    validate_wire,
)

SCHEMA = "simready-validator-score-v1"


def pinned_optional_profile():
    return json.loads(
        Path(__file__).with_name("nvidia_optional_profile.json").read_text()
    )


def _recorded_optional_requirements(nvidia, roster):
    execution = nvidia["rule_execution"]
    if "optional_requirements" in execution:
        return execution["optional_requirements"]
    from simready_bench.engine.validator import requirement_outcomes

    buckets = execution.get("buckets")
    if not isinstance(buckets, list) or len(buckets) > 100:
        raise ValueError("Recorded SDK runtime buckets are required")
    for bucket in buckets:
        if not isinstance(bucket, dict) or any(
            not isinstance(bucket.get(key), list)
            for key in ("declared_requirement_codes", "enabled_rules", "method_calls")
        ):
            raise ValueError("Invalid recorded SDK runtime bucket")
        if any(
            not isinstance(value, str)
            for key in ("declared_requirement_codes", "enabled_rules")
            for value in bucket[key]
        ):
            raise ValueError("Invalid recorded SDK rule identities")
        if any(
            not isinstance(call, dict)
            or not isinstance(call.get("rule"), str)
            or not isinstance(call.get("method"), str)
            or type(call.get("count")) is not int
            or call["count"] < 0
            for call in bucket["method_calls"]
        ):
            raise ValueError("Invalid recorded SDK check calls")
    issues = [
        issue
        for key in ("issues", "soft_issues", "warnings", "infos")
        for issue in nvidia.get(key, [])
    ]
    if any(not isinstance(issue, dict) for issue in issues):
        raise ValueError("Invalid recorded NVIDIA findings")
    issues = [
        {
            "rule_id": issue.get("rule_id"),
            "rule": issue.get("rule"),
            "severity": issue.get("severity"),
        }
        for issue in issues
    ]
    return requirement_outcomes(roster, buckets, issues)


def score_optional_validation(nvidia, *, candidate_invalid=False):
    """Score optional requirement IDs independently from the mandatory grade."""
    mandatory = pinned_profile()
    profile = pinned_optional_profile()
    roster = profile["optional_requirements"]
    statuses = {row["code"]: "unknown" for row in roster}
    reason = None
    if candidate_invalid:
        reason = "invalid_candidate_package"
    elif not isinstance(nvidia, dict):
        reason = "validator_unavailable"
    else:
        inventory = nvidia.get("inventory") or {}
        execution = nvidia.get("rule_execution") or {}
        identity_fields = (
            "profile_id",
            "profile_version",
            "profile_manifest_sha256",
            "tier_manifest_hashes",
            "versions",
        )
        if profile["mandatory_inventory_sha256"] != digest(mandatory) or any(
            inventory.get(key) != mandatory[key] for key in identity_fields
        ):
            reason = "pinned_inventory_mismatch"
        elif "optional_requirements" in inventory and any(
            inventory.get(key) != profile[key]
            for key in (
                "optional_requirements",
                "optional_features",
                "optional_requirement_count",
            )
        ):
            reason = "optional_inventory_mismatch"
        elif (
            nvidia.get("engine_dispatch_completed") is not True
            or execution.get("completed") is not True
        ):
            reason = "validator_execution_incomplete"
        elif any(
            (execution.get("sdk_source_sha256") or {}).get(key) != value
            for key, value in mandatory["SDK_dispatch_source_sha256"].items()
        ):
            reason = "validator_implementation_identity_mismatch"
        elif any(
            not isinstance(error, dict)
            or error.get("code") != "validator_rule_execution_error"
            for error in nvidia.get("infrastructure_errors", [])
        ):
            reason = "validator_infrastructure_error"
        else:
            try:
                rows = _recorded_optional_requirements(nvidia, roster)
                if not isinstance(rows, list) or len(rows) > len(roster):
                    raise ValueError("Invalid optional requirement execution")
                indexed = {}
                for row in rows:
                    if (
                        not isinstance(row, dict)
                        or row.get("code") not in statuses
                        or row["code"] in indexed
                    ):
                        raise ValueError("Invalid optional requirement execution")
                    indexed[row["code"]] = row
                for requirement in roster:
                    code = requirement["code"]
                    row = indexed.get(code, {})
                    module = requirement["rule"].rsplit(".", 1)[0]
                    if (execution.get("sdk_source_sha256") or {}).get(
                        module
                    ) != profile["rule_module_sha256"][module]:
                        reason = "validator_implementation_identity_mismatch"
                        continue
                    calls = row.get("method_calls")
                    if (
                        any(
                            row.get(key) != requirement[key]
                            for key in ("version", "rule")
                        )
                        or row.get("rule_enabled") is not True
                        or row.get("rule_invoked") is not True
                        or not isinstance(calls, dict)
                        or any(
                            not isinstance(key, str)
                            or type(value) is not int
                            or value < 0
                            for key, value in calls.items()
                        )
                        or not any(value > 0 for value in calls.values())
                    ):
                        continue
                    disposition = row.get("disposition")
                    if disposition == "failed":
                        statuses[code] = "fail"
                    elif disposition in ("no_failure_reported", "warning_reported"):
                        statuses[code] = "pass"
            except (ValueError, KeyError, TypeError):
                statuses = {code: "unknown" for code in statuses}
                reason = "invalid_optional_requirement_execution"
    count = len(roster)
    passed = sum(status == "pass" for status in statuses.values())
    failed = sum(status == "fail" for status in statuses.values())
    unknown = count - passed - failed
    return {
        "schema": "simready-validator-optional-score-v1",
        "profile": {"id": profile["profile_id"], "version": profile["profile_version"]},
        "profile_inventory_sha256": digest(profile),
        "scope": "all_optional_profile",
        "status": "invalid_candidate"
        if candidate_invalid
        else "unknown"
        if unknown
        else "scored",
        "score_percent": None if unknown else 100 * passed / count,
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "measured_count": passed + failed,
        "applicable_count": count,
        "profile_optional_count": count,
        "coverage_fraction": (passed + failed) / count,
        "lower_bound_percent": 100 * passed / count,
        "upper_bound_percent": 100 * (passed + unknown) / count,
        "unknown_reason": (reason or "optional_requirement_not_assessed")
        if unknown
        else None,
        "checks": [
            {
                "requirement_id": row["code"],
                "requirement_version": row["version"],
                "category": category(row["code"]),
                "status": statuses[row["code"]],
                "mandatory": False,
            }
            for row in roster
        ],
    }


def validate_optional_score(score):
    """Validate the optional denominator, outcomes and partial measurement bounds."""
    profile = pinned_optional_profile()
    roster = profile["optional_requirements"]
    rows = score.get("checks") if isinstance(score, dict) else None
    if (
        not isinstance(rows, list)
        or len(rows) != len(roster)
        or any(not isinstance(row, dict) for row in rows)
        or any(
            row.get("requirement_id") != requirement["code"]
            or row.get("requirement_version") != requirement["version"]
            or row.get("mandatory") is not False
            or row.get("status") not in ("pass", "fail", "unknown")
            for row, requirement in zip(rows, roster)
        )
    ):
        raise ValueError("Exact pinned optional requirement outcomes are required")
    count = len(roster)
    passed = sum(row["status"] == "pass" for row in rows)
    failed = sum(row["status"] == "fail" for row in rows)
    unknown = count - passed - failed
    counts = {
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "measured_count": passed + failed,
        "applicable_count": count,
        "profile_optional_count": count,
    }
    if any(
        type(score.get(key)) is not int or score[key] != value
        for key, value in counts.items()
    ):
        raise ValueError("Optional requirement accounting is inconsistent")
    expected_status = "unknown" if unknown else "scored"
    if score.get("status") == "invalid_candidate" and unknown == count:
        expected_status = "invalid_candidate"
    if (
        score.get("schema") != "simready-validator-optional-score-v1"
        or score.get("profile")
        != {"id": profile["profile_id"], "version": profile["profile_version"]}
        or score.get("profile_inventory_sha256") != digest(profile)
        or score.get("scope") != "all_optional_profile"
        or score.get("status") != expected_status
    ):
        raise ValueError("Pinned optional scoring scope is required")
    expected = {
        "score_percent": None if unknown else 100 * passed / count,
        "coverage_fraction": (passed + failed) / count,
        "lower_bound_percent": 100 * passed / count,
        "upper_bound_percent": 100 * (passed + unknown) / count,
    }
    for key, value in expected.items():
        actual = score.get(key)
        if value is None:
            valid = actual is None
        else:
            valid = (
                type(actual) in (int, float)
                and math.isfinite(actual)
                and math.isclose(actual, value, abs_tol=1e-10)
            )
        if not valid:
            raise ValueError("Optional score or measurement bounds are inconsistent")
    return score


def category(code):
    suffix = code.split(".", 3)[-1]
    if suffix.startswith(("RB.", "RB.COL.")):
        return "rigid_body_collision"
    if suffix.startswith("VG."):
        return "geometry_mesh"
    if suffix.startswith("VM."):
        return "materials_textures"
    if suffix.startswith("HI."):
        return "hierarchy"
    if suffix.startswith(("NP.", "AA.")):
        return "asset_paths_and_naming"
    if suffix.startswith("UN."):
        return "units"
    if suffix.startswith("SL."):
        return "semantics"
    if suffix.startswith("JT."):
        return "joints_articulations"
    if suffix.startswith(("NVM.", "PMT.")):
        return "nonvisual_materials"
    if suffix.startswith(("PHYSX.", "NEWTON.", "MUJOCO.", "RV.")):
        return "runtime_physics"
    if suffix.startswith("DISP."):
        return "display_colour"
    if suffix.startswith("GSP."):
        return "grasp_physics"
    if suffix.startswith("ISA."):
        return "runtime_composition"
    return "thumbnail_metadata"


def score_validation(nvidia, policy, *, candidate_invalid=False):
    """Pure projection of genuine SDK requirement outcomes; no candidate mutation."""
    policy = validate_policy(policy)
    p = pinned_profile()
    roster = p["mandatory_requirements"]
    active = set(policy["applicable_requirement_ids"])
    statuses = {r["code"]: "unknown" for r in roster}
    reason = None
    official = None
    if candidate_invalid:
        statuses = {c: "invalid" for c in statuses}
        reason = "invalid_candidate_package"
    elif not isinstance(nvidia, dict):
        reason = "validator_unavailable"
    else:
        inv = nvidia.get("inventory") or {}
        execution = nvidia.get("rule_execution") or {}
        fields = (
            "profile_id",
            "profile_version",
            "mandatory_features",
            "mandatory_requirements",
            "mandatory_requirement_count",
            "profile_manifest_sha256",
            "tier_manifest_hashes",
            "versions",
        )
        if any((inv.get(k) != p[k] for k in fields)):
            reason = "pinned_inventory_mismatch"
        elif (
            nvidia.get("engine_dispatch_completed") is not True
            or execution.get("completed") is not True
        ):
            reason = "validator_execution_incomplete"
        elif any(
            (
                (execution.get("sdk_source_sha256") or {}).get(k) != v
                for k, v in {
                    **p["SDK_dispatch_source_sha256"],
                    **p["rule_module_sha256"],
                }.items()
            )
        ):
            reason = "validator_implementation_identity_mismatch"
        elif any(
            (
                not isinstance(e, dict)
                or e.get("code") != "validator_rule_execution_error"
                for e in nvidia.get("infrastructure_errors", [])
            )
        ):
            reason = "validator_infrastructure_error"
        else:
            raw = execution.get("requirements", [])
            indexed = {}
            if not isinstance(raw, list) or len(raw) > 10000:
                reason = "invalid_requirement_execution"
            else:
                for r in raw:
                    if (
                        not isinstance(r, dict)
                        or r.get("code") not in statuses
                        or r["code"] in indexed
                    ):
                        reason = "invalid_requirement_execution"
                        break
                    indexed[r["code"]] = r
                if reason is None:
                    for requirement in roster:
                        c = requirement["code"]
                        r = indexed.get(c, {})
                        if (
                            any(
                                (
                                    r.get(k) != requirement[k]
                                    for k in ("version", "rule")
                                )
                            )
                            or r.get("rule_invoked") is not True
                        ):
                            continue
                        calls = r.get("method_calls")
                        if (
                            r.get("rule_enabled") is not True
                            or not isinstance(calls, dict)
                            or (
                                not any(
                                    (type(v) is int and v > 0 for v in calls.values())
                                )
                            )
                        ):
                            continue
                        disposition = r.get("disposition")
                        if disposition == "failed":
                            statuses[c] = "fail"
                        elif disposition in ("no_failure_reported", "warning_reported"):
                            statuses[c] = "pass"
                    official = (
                        nvidia.get("passed") is True
                        if not nvidia.get("infrastructure_errors")
                        and all((v in ("pass", "fail") for v in statuses.values()))
                        else None
                    )
                    if official is True and any(
                        (v != "pass" for v in statuses.values())
                    ):
                        statuses = {c: "unknown" for c in statuses}
                        official = None
                        reason = "inconsistent_official_result"
    checked = [statuses[c] for c in policy["applicable_requirement_ids"]]
    count = len(checked)
    passed = checked.count("pass")
    failed = checked.count("fail") + checked.count("invalid")
    unknown = checked.count("unknown")
    all_pass = not unknown and passed == count
    base = 95 * passed / count
    bonus = None if unknown else 5 if all_pass else 0
    score = None if unknown else base + bonus
    checks = []
    categories = {}
    for r in roster:
        c = r["code"]
        status = statuses[c]
        scoring = c in active
        group = category(c)
        row = {
            "requirement_id": c,
            "requirement_version": r["version"],
            "category": group,
            "status": status,
            "mandatory": True,
            "scoring": scoring,
            "weight": 1 if scoring else 0,
            "base_credit_percent": 95 / count
            if scoring and status == "pass"
            else None
            if scoring and status == "unknown"
            else 0,
            "status_basis": "pinned_NVIDIA_rule_execution_requirement_outcome",
            "corrective_action": None,
        }
        checks.append(row)
        if scoring:
            summary = categories.setdefault(
                group, {"applicable": 0, "passed": 0, "failed": 0, "unknown": 0}
            )
            summary["applicable"] += 1
            summary["passed"] += status == "pass"
            summary["failed"] += status in ("fail", "invalid")
            summary["unknown"] += status == "unknown"
    return {
        "schema": SCHEMA,
        "policy_id": POLICY_ID,
        "policy_sha256": digest(policy),
        "scope": policy["scope"],
        "status": "invalid_candidate"
        if candidate_invalid
        else "unknown"
        if unknown
        else "scored",
        "score_percent": score,
        "score_fraction": None if score is None else score / 100,
        "base_credit_percent": None if unknown else base,
        "completion_bonus": bonus,
        "score_lower_bound": base,
        "score_upper_bound": 95 * (passed + unknown) / count + (5 if not failed else 0),
        "passed_count": passed,
        "failed_count": failed,
        "unknown_count": unknown,
        "applicable_count": count,
        "profile_mandatory_count": len(roster),
        "profile_coverage_fraction": count / len(roster),
        "all_scored_mandatory_pass": all_pass,
        "official_conformance": {
            "profile": policy["profile"],
            "passed": official,
            "filtered": False,
        },
        "unknown_reason": reason if unknown else None,
        "checks": checks,
        "categories": categories,
        "optional_requirements": "report_only",
        "optional_score": score_optional_validation(
            nvidia, candidate_invalid=candidate_invalid
        ),
    }


def public_report(report, *, candidate_path=None):
    """Keep original NVIDIA fields/messages/locations; redact only host/private data.

    Candidate USD prim paths are preserved. This is a candidate-only report; no
    reference report or target is ever supplied to this boundary.
    """
    if not isinstance(report, dict):
        return None
    root = str(Path(candidate_path).absolute()) if candidate_path else None
    asset = report.get("asset_path")
    asset_parent = str(Path(asset).parent) if isinstance(asset, str) else None
    host = re.compile(
        "/(?:home|Users|tmp|dev/shm|var|private|proc|run|etc)(?:/[^\\s\\\"\\'<>),;\\]]+)*"
    )
    secret = re.compile(
        "(?i)(?:bearer\\s+|(?:api[_-]?key|secret|password)\\s*[:=]\\s*)[^\\s\\\"\\'<>]+"
    )
    sites = set()

    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("prim_path"), str) and value[
                "prim_path"
            ].startswith("/"):
                sites.add(value["prim_path"])
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(report)
    credential_keys = {
        "api_key",
        "apikey",
        "api-key",
        "password",
        "secret",
        "client_secret",
        "authorization",
        "proxy-authorization",
        "x-api-key",
    }

    def project(value, key=None):
        if isinstance(key, str) and key.casefold() in credential_keys:
            return "<redacted secret>"
        if key == "profile_manifest_path":
            return "<installed pinned profile>"
        if key == "prim_path" and isinstance(value, str):
            return value
        if isinstance(value, dict):
            return {k: project(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [project(v, key) for v in value]
        if isinstance(value, str):
            if root:
                value = value.replace(root, "/output")
            if asset_parent:
                value = value.replace(asset_parent, "/output")
            protected = {}
            for index, site in enumerate(sorted(sites, key=len, reverse=True)):
                marker = "__CANDIDATE_USD_SITE_" + str(index) + "__"
                if site in value:
                    value = re.sub(
                        re.escape(site) + "(?![A-Za-z0-9_/.:])",
                        lambda match: marker,
                        value,
                    )
                    protected[marker] = site
            value = secret.sub(
                "<redacted secret>", host.sub("<private host path>", value)
            )
            for marker, site in protected.items():
                value = value.replace(marker, site)
            return value
        return value

    return project(deepcopy(report))


def verify_validator_installation(inventory):
    import importlib, inspect
    from simready_bench.engine.validator import _sdk_source_identity

    p = pinned_profile()
    for key in (
        "profile_id",
        "profile_version",
        "mandatory_features",
        "mandatory_requirements",
        "mandatory_requirement_count",
        "profile_manifest_sha256",
        "tier_manifest_hashes",
        "versions",
    ):
        if inventory.get(key) != p[key]:
            raise RuntimeError("Pinned NVIDIA validator inventory changed")
    if _sdk_source_identity() != p["SDK_dispatch_source_sha256"]:
        raise RuntimeError("Pinned NVIDIA dispatch implementation changed")
    observed = {}
    for module, expected in p["rule_module_sha256"].items():
        path = Path(inspect.getfile(importlib.import_module(module)))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[module] = actual
        if actual != expected:
            raise RuntimeError("Pinned NVIDIA requirement implementation changed")
    return {
        "profile_inventory_sha256": digest(p),
        "rule_module_sha256": observed,
        "SDK_dispatch_source_sha256": p["SDK_dispatch_source_sha256"],
    }


def observe_validator_candidate(source, *, package_root, snapshot_sha256):
    """Safe dependency scan, then real NVIDIA dispatch with no extra physics gate."""
    import tempfile, shutil
    from simready_bench.engine.package_checks import inspect_package
    from simready_bench.engine.validator import validate_asset

    source = Path(source).absolute()
    package = Path(package_root).absolute()
    if (
        source.resolve() != source
        or package.resolve() != package
        or (not source.is_relative_to(package))
    ):
        raise ValueError("Confined candidate source required")
    pre = inspect_package(package, entrypoint=source.relative_to(package).as_posix())
    nvidia = None
    implementation = None
    if pre.get("package_ok") is True:
        with tempfile.TemporaryDirectory(prefix="simready-validator-") as temporary:
            target = Path(temporary) / "asset/package"
            shutil.copytree(package, target)
            originals = {
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package.rglob("*")
                if p.is_file()
            }
            copied = {
                str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in target.rglob("*")
                if p.is_file()
            }
            if originals != copied:
                raise RuntimeError("Candidate validation copy changed")
            nvidia = validate_asset(target / source.relative_to(package))
            if nvidia.get("inventory") is not None:
                implementation = verify_validator_installation(nvidia["inventory"])
            if originals != {
                str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package.rglob("*")
                if p.is_file()
            }:
                raise RuntimeError("Candidate changed during validation")
    labels = None
    if pre.get("package_ok") is True:
        try:
            from pxr import Usd
            from simready_bench.engine.semantics import observe_labels

            stage = Usd.Stage.Open(str(source))
            labels = observe_labels(stage) if stage else None
        except Exception:
            pass
    return {
        "schema": "simready-validator-observation-v1",
        "snapshot_sha256": snapshot_sha256,
        "source_record": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "preflight": pre,
        "nvidia_report": nvidia,
        "semantic_labels": labels,
        "validator_implementation_identity": implementation,
    }


def evaluate_snapshot(
    evaluator,
    snapshot,
    task_config,
    episode_id,
    deadline_at,
    *,
    final=True,
    stop_reason="unknown",
):
    """New scorer uses only actual candidate validator evidence, independent of GPU."""
    import time
    from simready_bench.engine.snapshots import verify_frozen_snapshot
    from simready_bench.engine.semantics import information
    from simready_bench.engine.verifier import PROTOCOL_VERSION

    policy = validate_policy(task_config["validator_scoring"])
    if "validator_reference" in task_config:
        from simready_bench.engine.environment import validate_task_reference_binding

        validate_task_reference_binding(
            task_config, allowed_reference_root=evaluator.allowed_reference_root
        )
    binding = digest(
        {"policy": policy, "config": task_config, "evaluator": evaluator.identity}
    )
    invalid_codes = {
        "missing_entrypoint",
        "symlink_not_allowed",
        "hardlink_not_allowed",
        "special_file_not_allowed",
        "package_too_large",
        "package_entry_limit_exceeded",
        "package_too_deep",
        "unsafe_path",
    }
    invalid = snapshot.error_code in invalid_codes
    observation = {}
    cache_hit = False
    native = getattr(evaluator, "_validator_native_invalid", {}).get(
        (snapshot.episode_id, snapshot.snapshot_id, snapshot.sha256)
    )
    if native and native["proof"].get("proven_invalid") is True:
        invalid = True
    with evaluator.lock:
        with evaluator._connect() as db:
            old = (
                db.execute(
                    "SELECT snapshot,record FROM finals WHERE episode=?", (episode_id,)
                ).fetchone()
                if final
                else None
            )
            if old:
                r = json.loads(old[1])
                if (
                    old[0] != (snapshot.sha256 or snapshot.snapshot_id)
                    or r["evaluation_binding"] != binding
                ):
                    raise ValueError("Final validator snapshot or policy changed")
                return r
            cached = (
                db.execute(
                    "SELECT record FROM observations WHERE identity=? AND snapshot=?",
                    (evaluator.identity, snapshot.sha256),
                ).fetchone()
                if snapshot.valid
                else None
            )
        if snapshot.valid:
            verify_frozen_snapshot(snapshot)
            try:
                if cached:
                    observation = json.loads(cached[0])
                    cache_hit = True
                elif evaluator.observer is not None and time.time() < deadline_at:
                    result = evaluator.observer(
                        snapshot, episode_id=episode_id, deadline_at=deadline_at
                    )
                    if result.get("snapshot_sha256") != snapshot.sha256:
                        raise ValueError("Snapshot observation mismatch")
                    observation = result["observation"]
                    verify_frozen_snapshot(snapshot)
                    expected = next(
                        (
                            f.sha256
                            for f in snapshot.files
                            if f.relative_path == snapshot.entrypoint
                        )
                    )
                    if observation.get("source_record", {}).get("sha256") != expected:
                        raise ValueError("Validator source binding mismatch")
                    with evaluator._connect() as db:
                        db.execute(
                            "INSERT OR IGNORE INTO observations VALUES(?,?,?)",
                            (
                                evaluator.identity,
                                snapshot.sha256,
                                json.dumps(
                                    observation, sort_keys=True, allow_nan=False
                                ),
                            ),
                        )
            except Exception:
                observation = {}
        raw = observation.get("nvidia_report")
        pre = observation.get("preflight", {})
        if pre.get("package_ok") is False and (not pre.get("infrastructure_errors")):
            invalid = True
        if (
            isinstance(raw, dict)
            and raw.get("outcome") == "invalid_submission"
            and (not raw.get("infrastructure_errors"))
        ):
            invalid = True
        score = score_validation(raw, policy, candidate_invalid=invalid)
        official = score["official_conformance"]["passed"]
        known = score["status"] != "unknown"
        info = information(observation.get("semantic_labels"))
        info.update(
            schema="simready-NVIDIA-mandatory-semantic-observations-v1",
            required=True,
            scoring=True,
            scoring_source="six mandatory NVIDIA SL requirement IDs; no bespoke label metric",
        )
        safe = public_report(raw, candidate_path=snapshot.path)
        for check in score["checks"]:
            check["findings"] = [
                deepcopy(i)
                for key in ("issues", "soft_issues", "warnings", "infos")
                for i in (safe or {}).get(key, [])
                if isinstance(i, dict) and i.get("rule_id") == check["requirement_id"]
            ]
        feedback = {
            "schema_version": "simready-validator-feedback-v1",
            "snapshot_sha256": snapshot.sha256,
            "scoring": True,
            "reward": score["score_percent"],
            "simready": score,
            "optional_score": score["optional_score"],
            "nvidia_validator_report": safe,
            "informational_semantics": info,
        }
        components = [
            {
                "component_id": r["requirement_id"],
                "status": r["status"],
                "score": 1
                if r["status"] == "pass"
                else 0
                if r["status"] in ("fail", "invalid")
                else None,
                "weight": r["weight"],
                "credit": r["base_credit_percent"],
            }
            for r in score["checks"]
            if r["scoring"]
        ]
        report = {
            "schema_version": "simready-evaluation-validator-v1",
            "protocol": PROTOCOL_VERSION,
            "episode_id": episode_id,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_sha256": snapshot.sha256,
            "evaluation_binding": binding,
            "outcome": "invalid_submission"
            if invalid
            else "infrastructure_error"
            if not known
            else "complete"
            if score["all_scored_mandatory_pass"]
            else "incomplete",
            "reward": score["score_percent"],
            "simready_reward": score["score_percent"],
            "diagnostic_reward": None,
            "official_reward": 1
            if official is True
            else 0
            if official is False
            else None,
            "reward_reporting": {
                "schema": "simready-validator-reporting-v1",
                "harbor_key": "simready_reward",
                "scalar": "95 * passed / fixed_applicable + 5_if_all_pass",
                "scale": [0, 100],
                "official": False,
                "scope": score["scope"],
            },
            "simready": score,
            "optional_score": score["optional_score"],
            "nvidia_validator_report": safe,
            "validator_implementation_identity": observation.get(
                "validator_implementation_identity"
            ),
            "semantic_labels": info,
            "feedback": feedback,
            "components": components,
            "coverage": {
                "measured": score["passed_count"] + score["failed_count"],
                "total": score["applicable_count"],
                "fraction": 1 - score["unknown_count"] / score["applicable_count"],
                "profile_fraction": score["profile_coverage_fraction"],
            },
            "completion": {
                "all_mandatory_pass": score["all_scored_mandatory_pass"],
                "bonus_eligible": score["all_scored_mandatory_pass"],
                "configured_bonus": 5,
                "earned_bonus": score["completion_bonus"],
                "maximum_final_scalar": 100,
            },
            "acceptance": {
                "accepted": score["all_scored_mandatory_pass"],
                "scope": score["scope"],
                "official_conformance_passed": official,
            },
            "readiness": {
                "official_ready": official is True,
                "blockers": []
                if official is True
                else ["official_conformance_not_confirmed"],
            },
            "absolute_quality": {
                "component_sum": score["base_credit_percent"],
                "max_component_sum": 95,
                "lower_bound": score["score_lower_bound"],
                "upper_bound": score["score_upper_bound"],
            },
            "observation_cache_hit": cache_hit,
        }
        if final and known:
            with evaluator._connect() as db:
                db.execute(
                    "INSERT INTO finals VALUES(?,?,?)",
                    (
                        episode_id,
                        snapshot.sha256 or snapshot.snapshot_id,
                        json.dumps(report, sort_keys=True, allow_nan=False),
                    ),
                )
        return report


def main():
    import argparse, resource

    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=["observe"])
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--snapshot-sha256", required=True)
    p.add_argument("--package-root", required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    result = observe_validator_candidate(
        a.source, package_root=a.package_root, snapshot_sha256=a.snapshot_sha256
    )
    Path(a.output).write_text(
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()


def build_wire(report, *, snapshot):
    if (
        report.get("schema_version") != "simready-evaluation-validator-v1"
        or report.get("snapshot_sha256") != snapshot.sha256
        or report.get("episode_id") != snapshot.episode_id
        or (report.get("snapshot_id") != snapshot.snapshot_id)
    ):
        raise ValueError("Authoritative validator report and snapshot required")
    score = deepcopy(report["simready"])
    for r in score["checks"]:
        r.pop("findings", None)
    wire = {
        "protocol": report["protocol"],
        "schema_version": WIRE_SCHEMA,
        "episode_id": snapshot.episode_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": snapshot.sha256,
        "outcome": report["outcome"],
        "reward": report["reward"],
        "simready": score,
        "acceptance": deepcopy(report["acceptance"]),
        "readiness": deepcopy(report["readiness"]),
        "errors": []
        if score["status"] != "unknown"
        else ["validator_execution_unknown"],
    }
    return validate_wire(wire)


def record_native_invalid_snapshot(
    evaluator, *, controller, environment, snapshot, task_config, task_input=None
):
    """Bind invalid submissions to the stopped actor and its immutable snapshot."""
    from dataclasses import asdict
    import sqlite3
    from simready_bench.engine.contracts import Snapshot
    from simready_bench.engine.harbor import _trusted_environment
    from simready_bench.engine.snapshots import (
        verify_native_invalid_evidence,
        PACKAGE_INVALID_CODES,
    )
    from simready_bench.engine.environment import verify_profile_code

    if snapshot.error_code not in PACKAGE_INVALID_CODES:
        return None
    if (
        type(snapshot) is not Snapshot
        or not _trusted_environment(environment)
        or environment.bridge.controller is not controller
        or (environment.final_snapshot != snapshot)
        or (environment.actor_stopped is not True)
        or (environment.actor_start_attempted is not True)
    ):
        raise ValueError("Exact stopped native final snapshot required")
    verify_profile_code(task_config["shared_environment_profile"])
    if snapshot.path and snapshot.sha256:
        from dataclasses import replace
        from simready_bench.engine.snapshots import verify_frozen_snapshot

        verify_frozen_snapshot(replace(snapshot, error_code=None))
    with sqlite3.connect(
        "file:" + str(controller.db_path) + "?mode=ro", uri=True
    ) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM episodes WHERE episode_id=?", (snapshot.episode_id,)
        ).fetchone()
        saved = db.execute(
            "SELECT record FROM snapshots WHERE snapshot_id=?", (snapshot.snapshot_id,)
        ).fetchone()
    if (
        row is None
        or saved is None
        or digest(json.loads(saved["record"])) != digest(asdict(snapshot))
    ):
        raise ValueError("Original controller snapshot proof differs")
    audit = environment._actor_stop_audit
    if not isinstance(audit, dict) or set(audit) != {
        "container_id",
        "was_running",
        "kill_return_code",
        "state",
    }:
        raise ValueError("Actual native stop audit required")
    lifecycle_pin = hashlib.sha256(
        Path(__file__).with_name("harbor.py").read_bytes()
    ).hexdigest()
    stop = {
        "status": "confirmed_native_actor_stopped",
        "producer": "BenchmarkDockerEnvironment.stop_actor_writes",
        "producer_source_sha256": lifecycle_pin,
        "episode_id": snapshot.episode_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": snapshot.sha256,
        "snapshot_error_code": snapshot.error_code,
        "actor_stopped": True,
        "actor_start_attempted": True,
        **deepcopy(audit),
    }
    native_task_id = task_config["task_id"]
    if task_input is not None:
        from simready_bench.engine.harbor_services import ReferenceTaskInput
        from harbor.models.task.task import Task

        if type(task_input) is not ReferenceTaskInput or digest(
            task_input.task_config
        ) != digest(task_config):
            raise ValueError("Exact canonical/native task input required")
        if (
            task_input.task_path / "environment"
            != Path(environment.environment_dir).absolute()
        ):
            raise ValueError("Native environment and staged task input differ")
        ReferenceTaskInput(
            task_input.task_path,
            task_input.thumbnail,
            task_input.public_text,
            task_input.task_config,
            task_input.public_dir,
        )
        native_task_id = Task(task_input.task_path).name
    proof = verify_native_invalid_evidence(
        dict(row),
        json.loads(json.dumps(asdict(snapshot))),
        stop,
        expected_episode_id=snapshot.episode_id,
        expected_snapshot_id=snapshot.snapshot_id,
        expected_task_id=native_task_id,
        expected_native_lifecycle_sha256=lifecycle_pin,
    )
    binding = {
        "episode_id": snapshot.episode_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_sha256": snapshot.sha256,
        "canonical_task_id": task_config["task_id"],
        "native_task_id": native_task_id,
        "proof": proof,
        "stop_evidence_sha256": digest(stop),
    }
    root = evaluator.root / "native-invalid-proofs"
    root.mkdir(mode=448, exist_ok=True)
    path = root / (
        digest({"episode": snapshot.episode_id, "snapshot": snapshot.snapshot_id})
        + ".json"
    )
    raw = json.dumps(binding, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != raw:
            raise ValueError("Existing native invalid proof changed")
    else:
        with path.open("x") as f:
            f.write(raw)
        path.chmod(384)
    if not hasattr(evaluator, "_validator_native_invalid"):
        evaluator._validator_native_invalid = {}
    evaluator._validator_native_invalid[
        snapshot.episode_id, snapshot.snapshot_id, snapshot.sha256
    ] = binding
    return proof
