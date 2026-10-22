"""Real NVIDIA validation and a strict actor-facing factual feedback boundary.

Run this module only in the isolated validator environment. Raw reports, logs,
exception text and upstream suggestions are evaluator-owned. Only
factual_findings(report) may cross the actor boundary.
"""

from __future__ import annotations
import collections
import inspect
import sys
import hashlib
import importlib
import importlib.metadata
import json
import logging
from pathlib import Path
import re
import threading
import tomllib
import traceback

PROFILE_ID = "Robotics-Prop"
PROFILE_VERSION = "4.0.0"
_LOCK = threading.RLock()
_RULE_ID = re.compile("[A-Za-z][A-Za-z0-9_-]*(?:\\.[A-Za-z0-9_-]+)+\\Z")
_USD_PATH = re.compile(
    "/(?:[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*)?(?:\\.[A-Za-z_][A-Za-z0-9_:]*)?\\Z"
)
_FACTS = {
    "com.nvidia.simready.RB.007": "Required mass specification does not satisfy the static rule.",
    "com.nvidia.simready.SL.001": "Required semantic labeling does not satisfy the static rule.",
    "com.nvidia.simready.SL.002": "Deprecated semantic labels do not satisfy the modern-label requirement.",
    "com.nvidia.simready.SL.003": "Semantic schema structure does not satisfy the static rule.",
    "com.nvidia.simready.SL.QCODE.001": "Semantic classification does not satisfy the Q-code rule.",
    "com.nvidia.simready.SR.002": "Thumbnail evidence does not satisfy the static rule.",
    "com.nvidia.simready.SR.004": "Required asset metadata does not satisfy the static rule.",
    "com.nvidia.simready.UN.006": "The asset does not satisfy the required Z-up rule.",
    "com.nvidia.simready.UN.007": "The asset does not satisfy the required metre-unit rule.",
    "com.nvidia.simready.NP.008": "Asset dependencies do not satisfy the local-path existence rule.",
    "com.nvidia.usd.VG.014": "Mesh topology does not satisfy the static rule.",
}
_PUBLIC_FACTS = {
    "external_dependency": "An authored asset dependency is outside the submitted package.",
    "missing_dependency": "An authored local asset dependency is missing.",
    "executable_dependency": "The package contains an executable dependency prohibited by the public contract.",
    "package_limit_exceeded": "The package exceeds a published byte or entry limit.",
    "expanded_package_limit_exceeded": "The expanded package exceeds a published byte or entry limit.",
    "nonregular_or_linked_file": "The package contains a nonregular or linked file.",
    "usd_composition_error": "OpenUSD reported a composition error.",
    "usd_validation_error": "An installed general OpenUSD validation rule failed.",
    "explicit_positive_mass_required": "The effective rigid body has no authored finite positive mass.",
    "collision_geometry_required": "No enabled collision representation is present.",
    "collider_outside_effective_body": "An enabled collider is outside the effective rigid body.",
    "metres_required": "Stage metersPerUnit differs from one.",
    "z_up_required": "The stage upAxis differs from Z.",
    "default_prim_required": "The stage has no valid defaultPrim.",
    "nominal_dimensions_exceeded": "A measured posed extent exceeds its published tolerance.",
    "origin_frame_mismatch": "The posed geometry does not satisfy the published origin convention.",
    "triangle_limit_exceeded": "The measured mesh triangle count exceeds the published cap.",
    "texture_count_exceeded": "The package texture count exceeds the published cap.",
    "texture_resolution_exceeded": "A texture dimension exceeds the published cap.",
    "invalid_texture": "A submitted texture could not be decoded.",
    "invalid_mesh_topology": "A visible mesh has inconsistent face topology.",
    "finite_nonempty_geometry_required": "No finite nonempty visible geometry was measured.",
    "nonfinite_geometry": "A visible geometry coordinate is nonfinite.",
    "nonfinite_transform": "A visible geometry transform is nonfinite.",
    "invalid_physics_material": "A physics-material coefficient violates the published finite ranges.",
}
_FACTS.update(
    {"simready.public." + code: message for code, message in _PUBLIC_FACTS.items()}
)


def map_finding(raw: dict) -> dict:
    """Map one real serialized issue; never inspect arbitrary upstream prose."""
    identity = raw.get("rule_id")
    if (
        not isinstance(identity, str)
        or len(identity) > 160
        or (not _RULE_ID.fullmatch(identity))
    ):
        identity = "unmapped"
    path = raw.get("prim_path")
    if not isinstance(path, str) or len(path) > 1024 or (not _USD_PATH.fullmatch(path)):
        path = None
    severity = raw.get("severity")
    severity = {
        "failure": "error",
        "error": "error",
        "warning": "warning",
        "info": "info",
        "success": "info",
    }.get(severity, "error")
    return {
        "rule_id": identity,
        "prim_path": path,
        "severity": severity,
        "finding": _FACTS.get(
            identity, "The validator reported a diagnostic for this rule."
        ),
    }


def factual_findings(report: dict) -> list[dict]:
    """Only this whitelist enters the actor; the input report remains private."""
    return [
        map_finding(issue)
        for name in ("issues", "soft_issues")
        for issue in report.get(name, [])
        if isinstance(issue, dict)
    ]


def _dependencies():
    plugin = {"physx_resource_plugins": False, "physx_native_bindings": False}
    try:
        importlib.import_module("physx_usd_schemas")
        plugin["physx_resource_plugins"] = True
    except ModuleNotFoundError as exc:
        if exc.name != "physx_usd_schemas":
            raise
    importlib.import_module("PIL.Image")
    importlib.import_module("MaterialX")
    from pxr import Usd

    try:
        importlib.import_module("pxr.PhysxSchema")
        plugin["physx_native_bindings"] = True
    except ImportError:
        pass
    if not Usd.SchemaRegistry.GetTypeFromSchemaTypeName("SemanticsLabelsAPI"):
        raise RuntimeError("SemanticsLabelsAPI schema plugin is unavailable")
    import simready.validate as api

    return (api, plugin)


def _inventory(api, plugin):
    from usd_validation_nvidia import ProfileRegistry, RequirementsRegistry
    from simready.validate.api import _copy_capability_with_dependencies

    api.initialize([], [], [])
    profile = ProfileRegistry().find(PROFILE_ID, PROFILE_VERSION)
    if profile is None:
        raise RuntimeError("Pinned Robotics-Prop 4.0.0 profile did not register")
    distribution = importlib.metadata.distribution("simready-foundation-tier-core")
    root = Path(distribution.locate_file("simready/foundation/tier_core"))
    manifest = root / "profiles/robotics_prop.toml"
    parsed = tomllib.loads(manifest.read_text())
    features, optional_features = [], []
    for item in parsed[PROFILE_ID][PROFILE_VERSION]["features"]:
        for fid, spec in item.items():
            if fid != "optional":
                selected = (
                    optional_features if item.get("optional", False) else features
                )
                selected.append({"id": fid, "version": str(spec["version"])})
    required = set(profile.mandatory_requirements)
    rows, optional_rows = {}, {}
    for capability in profile.capabilities:
        full, _ = _copy_capability_with_dependencies(capability)
        for requirement in full.requirements:
            rule = RequirementsRegistry().get_validator(requirement)
            key = (requirement.code, str(requirement.version))
            selected = rows if requirement.code in required else optional_rows
            selected[key] = {
                "code": key[0],
                "version": key[1],
                "rule": None if rule is None else rule.__module__ + "." + rule.__name__,
            }
    missing = sorted(required - {key[0] for key in rows})
    missing += [r["code"] for r in rows.values() if r["rule"] is None]
    manifests = {}
    for folder in ("profiles", "features", "capabilities"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in (".json", ".toml", ".yaml", ".yml"):
                manifests[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    payload = {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "mandatory_features": sorted(features, key=lambda r: r["id"]),
        "mandatory_requirements": [rows[k] for k in sorted(rows)],
        "mandatory_requirement_count": len(required),
        "optional_features": sorted(optional_features, key=lambda r: r["id"]),
        "optional_requirements": [optional_rows[k] for k in sorted(optional_rows)],
        "optional_requirement_count": len(optional_rows),
        "optional_missing_rule_implementations": sorted(
            row["code"] for row in optional_rows.values() if row["rule"] is None
        ),
        "missing_rule_implementations": missing,
        "profile_manifest_path": str(manifest),
        "profile_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tier_manifest_hashes": manifests,
        "schema_plugins": plugin,
        "versions": {
            n: importlib.metadata.version(n)
            for n in (
                "simready-foundation-tier-core",
                "simready-validate",
                "usd-validation-nvidia",
                "usd-profiles-nvidia",
            )
        },
    }
    payload["effective_inventory_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def derive_inventory() -> dict:
    """Resolve the installed pinned profile; requires the validator environment."""
    with _LOCK:
        api, plugin = _dependencies()
        try:
            return _inventory(api, plugin)
        finally:
            api.destroy()


def _serialize_issue(issue) -> dict:
    requirement = getattr(issue, "requirement", None)
    rule = getattr(issue, "rule", None)
    at = getattr(issue, "at", None)
    path = getattr(at, "path", None)
    return {
        "rule_id": getattr(requirement, "code", None),
        "requirement_version": str(getattr(requirement, "version", "")),
        "severity": str(
            getattr(getattr(issue, "severity", None), "name", "error")
        ).lower(),
        "prim_path": str(path) if path is not None else None,
        "rule": rule.__module__ + "." + rule.__name__ if rule is not None else None,
        "message": getattr(issue, "message", None),
        "location": str(at) if at is not None else None,
        "asset": str(getattr(issue, "asset", None)),
        "suggestions": [
            {
                "message": getattr(s, "message", None),
                "location": str(getattr(s, "at", None)),
                "callable": repr(getattr(s, "callable", None)),
            }
            for s in getattr(issue, "suggestions", ())
        ],
    }


_SDK_SOURCE_PINS = {
    "simready.validate.api": "f3185014fa445f28db2630af24ee7e78dd7e35a24423557b22146c40a9800f36",
    "usd_validation_nvidia._engine": "150c9ae2b37243f8b87b20128cf779e61e0ff99c3b5daacdef709604fc6cee52",
    "usd_validation_nvidia._compliance_checker": "8255df20e50330f9e5edbc4c0a3a5c35f9f8c70d2ae2deaff83d95a718738115",
    "usd_validation_nvidia._compliance_runners": "9f66e8212fa9a5adadde1ce4f8f28640e62d8495fd3596a2c2cdee18803bce8f",
    "usd_validation_nvidia._base_rule_checker": "b46e2cacd61b7d6b10d4a49a6392b0927e323d9735df695dc9f820de636c3a23",
}
_CHECK_METHODS = (
    "CheckStage",
    "CheckLayer",
    "CheckPrim",
    "CheckZipFile",
    "CheckDiagnostics",
    "CheckUnresolvedPaths",
    "CheckDependencies",
    "CheckFormatDependency",
)
_EXECUTION_SCOPE = (
    "overridden_rule_method_invocation_not_internal_requirement_branch_coverage"
)


def _empty_execution():
    return {
        "schema": "nvidia-rule-execution-v1",
        "scope": _EXECUTION_SCOPE,
        "completed": False,
        "requirements": [],
        "buckets": [],
        "sdk_source_sha256": {},
    }


def _sdk_source_identity():
    expected_versions = {
        "simready-foundation-tier-core": "2026.8.0",
        "simready-validate": "2026.8.0",
        "usd-validation-nvidia": "1.22.0",
        "usd-profiles-nvidia": "1.22.0",
    }
    for name, version in expected_versions.items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Pinned validator distribution changed: " + name)
    identity = {}
    for module_name, expected in _SDK_SOURCE_PINS.items():
        module = importlib.import_module(module_name)
        actual = hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                "Pinned validator dispatch source changed: " + module_name
            )
        identity[module_name] = actual
    return identity


def _qualified_rule(rule):
    return rule.__module__ + "." + rule.__name__


def requirement_outcomes(requirements, buckets, issues):
    """Project recorded SDK check calls and findings to requirement outcomes."""
    rows = []
    for requirement in requirements:
        code, rule = requirement["code"], requirement["rule"]
        relevant = [
            bucket for bucket in buckets if code in bucket["declared_requirement_codes"]
        ]
        methods = collections.Counter()
        for bucket in relevant:
            for call in bucket["method_calls"]:
                if call["rule"] == rule:
                    methods[call["method"]] += call["count"]
        related = [issue for issue in issues if issue["rule_id"] == code]
        unscoped = [
            issue
            for issue in issues
            if issue["rule_id"] is None and issue["rule"] == rule
        ]
        if any(issue["severity"] == "error" for issue in unscoped):
            disposition = "infrastructure_error"
        elif any(issue["severity"] in ("error", "failure") for issue in related):
            disposition = "failed"
        elif any(issue["severity"] == "failure" for issue in unscoped):
            disposition = "unscoped_failure"
        elif not methods:
            disposition = "not_invoked"
        elif any(issue["severity"] == "warning" for issue in related):
            disposition = "warning_reported"
        else:
            disposition = "no_failure_reported"
        rows.append(
            {
                **requirement,
                "rule_enabled": any(
                    rule in bucket["enabled_rules"] for bucket in relevant
                ),
                "rule_invoked": bool(methods),
                "method_calls": dict(sorted(methods.items())),
                "disposition": disposition,
                "uninvoked_reason": None
                if methods
                else "no_observed_overridden_check_method",
            }
        )
    return rows


def _run_profile(stage, inventory):
    """Invoke the real SDK engine and observe original implemented check calls.

    Timings alone are insufficient: the SDK times inherited no-op diagnostic and
    cache-reset events too. Python call profiling observes only actual overridden
    Check methods. It never replaces a rule, changes its result, or counts a rule
    constructor/ResetCaches as a check. A shared rule invocation is not proof that
    every requirement-specific internal branch was applicable.
    """
    import simready.validate.api as sdk
    from usd_validation_nvidia import BaseRuleChecker, ProfileRegistry, ValidationEngine
    from simready_bench.engine.nvidia_compat import (
        compatibility_identity,
        material_surface_connections,
    )

    source_identity = _sdk_source_identity()
    profile = ProfileRegistry().find(PROFILE_ID, PROFILE_VERSION)
    buckets, runtime_sets = sdk._bucket_requirements_by_runtime(profile)
    variant_config = sdk._runtime_variant_config(stage, runtime_sets)
    bucket_reports, all_issues, typed_issues = ([], [], [])
    try:
        for bucket in [None] + sorted((b for b in buckets if b is not None)):
            requirements = buckets.get(bucket)
            if not requirements:
                continue
            sdk._compose_variant_selection(stage, bucket, variant_config)
            engine = ValidationEngine(init_rules=False, variants=False, processes=0)
            engine.enable_capability(
                sdk._make_bucket_capability(requirements, bucket or "neutral")
            )
            classes = tuple(engine.rules)
            implemented = {}
            for rule in classes:
                module_name = rule.__module__
                rule_source = Path(inspect.getfile(rule))
                source_identity[module_name] = hashlib.sha256(
                    rule_source.read_bytes()
                ).hexdigest()
                for name in _CHECK_METHODS:
                    method = getattr(rule, name, None)
                    if method is not None and method is not getattr(
                        BaseRuleChecker, name, None
                    ):
                        code = getattr(method, "__code__", None)
                        if code is not None:
                            implemented[_qualified_rule(rule), code] = name
            calls = collections.Counter()
            previous = sys.getprofile()
            if previous is not None and (not callable(previous)):
                raise RuntimeError(
                    "Existing non-callable profiler prevents rule-call evidence"
                )

            def observe(frame, event, argument):
                if previous is not None:
                    previous(frame, event, argument)
                if event == "call":
                    instance = frame.f_locals.get("self")
                    if instance is not None:
                        identity = _qualified_rule(type(instance))
                        name = implemented.get((identity, frame.f_code))
                        if name:
                            calls[identity, name] += 1

            sys.setprofile(observe)
            try:
                with material_surface_connections():
                    result = engine.validate(stage)
            finally:
                sys.setprofile(previous)
            if result is None:
                raise RuntimeError("NVIDIA validation engine returned no result")
            actual_issues = list(result.issues())
            typed_issues.extend(actual_issues)
            issues = [_serialize_issue(issue) for issue in actual_issues]
            all_issues.extend(issues)
            bucket_reports.append(
                {
                    "bucket": bucket or "neutral",
                    "declared_requirement_codes": sorted(
                        (r.code for r in requirements)
                    ),
                    "enabled_rules": sorted(
                        (_qualified_rule(rule) for rule in classes)
                    ),
                    "method_calls": [
                        {"rule": rule, "method": method, "count": count}
                        for (rule, method), count in sorted(calls.items())
                    ],
                    "rule_times": [
                        {"rule": _qualified_rule(rule), "seconds": seconds}
                        for rule, seconds in engine.stats.get_rule_times()
                    ],
                }
            )
    finally:
        sdk._clear_variant_selection(stage, variant_config)
    required_codes = {r["code"] for r in inventory["mandatory_requirements"]}
    hard = [
        i
        for i in all_issues
        if i["severity"] in ("error", "failure") and i["rule_id"] in required_codes
    ]
    soft = [
        i
        for i in all_issues
        if i["severity"] in ("error", "failure") and i["rule_id"] not in required_codes
    ]
    rows = requirement_outcomes(
        inventory["mandatory_requirements"], bucket_reports, all_issues
    )
    optional_rows = requirement_outcomes(
        inventory.get("optional_requirements", []), bucket_reports, all_issues
    )
    execution = {
        "schema": "nvidia-rule-execution-v1",
        "scope": _EXECUTION_SCOPE,
        "completed": True,
        "sdk_source_sha256": source_identity,
        "compatibility_adapters": [compatibility_identity()],
        "buckets": bucket_reports,
        "requirements": rows,
        "optional_requirements": optional_rows,
    }
    typed_failures = [
        i for i in typed_issues if str(i.severity.name).lower() in ("error", "failure")
    ]
    typed_hard = [
        i
        for i in typed_failures
        if getattr(i.requirement, "code", None) in required_codes
    ]
    typed_soft = [
        i
        for i in typed_failures
        if getattr(i.requirement, "code", None) not in required_codes
    ]
    summary = sdk._build_features_validation_summary(
        typed_hard,
        typed_soft,
        PROFILE_ID,
        PROFILE_VERSION,
        include_failed_requirements=True,
    )
    return (hard, soft, all_issues, execution, summary)


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(
            {
                "logger": record.name,
                "level": record.levelname,
                "message": self.format(record),
            }
        )


def validate_asset(asset_path, report_path=None) -> dict:
    """Validate actual submitted bytes; never repair or stamp candidate metadata.

    A private JSON report is written when report_path is supplied. Launch in a
    dedicated process with the installed NVIDIA rule registry.
    """
    asset = Path(asset_path)
    report = {
        "profile_id": PROFILE_ID,
        "profile_version": PROFILE_VERSION,
        "asset_path": str(asset),
        "outcome": "infrastructure_error",
        "passed": False,
        "inventory": None,
        "issues": [],
        "soft_issues": [],
        "features_summary": {},
        "infrastructure_errors": [],
        "logs": [],
        "warnings": [],
        "infos": [],
        "rule_execution": _empty_execution(),
        "engine_dispatch_completed": False,
    }
    capture = _Capture()
    root = logging.getLogger()
    previous_level = root.level
    api = None
    with _LOCK:
        root.addHandler(capture)
        root.setLevel(min(previous_level, logging.INFO))
        try:
            api, plugin = _dependencies()
            inventory = _inventory(api, plugin)
            report["inventory"] = inventory
            if inventory["missing_rule_implementations"]:
                raise RuntimeError("Mandatory rule implementations are unavailable")
            if inventory["mandatory_requirement_count"] != 46:
                raise RuntimeError(
                    "Installed pinned profile requirement inventory changed"
                )
            if not asset.is_file():
                report["outcome"] = "invalid_submission"
            else:
                from pxr import Usd

                report["asset_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
                try:
                    stage = Usd.Stage.Open(str(asset))
                except Exception:
                    report["open_error"] = traceback.format_exc()
                    stage = None
                if stage is None:
                    report["outcome"] = "invalid_submission"
                else:
                    if not plugin["physx_native_bindings"]:
                        merges = [
                            str(p.GetPath())
                            for p in stage.Traverse()
                            if any(
                                (
                                    s.split(":")[0] == "PhysxMeshMergeCollisionAPI"
                                    for s in (
                                        p.GetMetadata(
                                            "apiSchemas"
                                        ).GetAddedOrExplicitItems()
                                        if p.GetMetadata("apiSchemas")
                                        else []
                                    )
                                )
                            )
                        ]
                        if merges:
                            report["infrastructure_errors"].append(
                                {
                                    "code": "physx_merge_checker_bindings_unavailable",
                                    "prim_paths": merges,
                                }
                            )
                    empty_candidate = not any(stage.Traverse())
                    hard, soft, all_issues, execution, summary = _run_profile(
                        stage, inventory
                    )
                    report["features_summary"] = summary
                    report["engine_dispatch_completed"] = True
                    report["rule_execution"] = execution
                    report["issues"], report["soft_issues"] = (hard, soft)
                    report["warnings"] = [
                        i for i in all_issues if i["severity"] == "warning"
                    ]
                    report["infos"] = [i for i in all_issues if i["severity"] == "info"]
                    for issue in all_issues:
                        # The pinned SDK attaches a requirement to asset errors.
                        # Its execution-exception handlers emit unscoped errors.
                        if issue["severity"] == "error" and issue["rule_id"] is None:
                            report["infrastructure_errors"].append(
                                {
                                    "code": "validator_rule_execution_error",
                                    "issue": issue,
                                }
                            )
                    uninvoked = [
                        r["code"]
                        for r in execution["requirements"]
                        if not r["rule_invoked"]
                    ]
                    if uninvoked and (not empty_candidate):
                        report["infrastructure_errors"].append(
                            {
                                "code": "mandatory_rule_not_invoked",
                                "requirements": uninvoked,
                            }
                        )
                    mandatory_rules = {
                        r["rule"] for r in inventory["mandatory_requirements"]
                    }
                    unscoped_failure = any(
                        (
                            i["rule_id"] is None
                            and i["severity"] == "failure"
                            and (i["rule"] in mandatory_rules)
                            for i in soft
                        )
                    )
                    report["passed"] = (
                        bool(summary)
                        and api.features_summary_passed(summary)
                        and (not hard)
                        and (not unscoped_failure)
                        and (not uninvoked)
                    )
                    report["outcome"] = "success" if report["passed"] else "failure"
                    if empty_candidate:
                        report["candidate_invalid_reason"] = "empty_composed_stage"
                        report["outcome"] = "invalid_submission"
                        report["passed"] = False
        except Exception:
            report["infrastructure_errors"].append(
                {"code": "validator_adapter_error", "traceback": traceback.format_exc()}
            )
        finally:
            if api is not None:
                api.destroy()
            root.removeHandler(capture)
            root.setLevel(previous_level)
            report["logs"] = capture.records
            for record in capture.records:
                if record["level"] in ("ERROR", "CRITICAL") and (
                    "Rule not found" in record["message"]
                    and report["inventory"]
                    and any(
                        (
                            r["code"] in record["message"]
                            for r in report["inventory"]["mandatory_requirements"]
                        )
                    )
                    or "Tier registration failed" in record["message"]
                ):
                    report["infrastructure_errors"].append(
                        {"code": "validator_registration_or_rule_gap", "log": record}
                    )
            if report["infrastructure_errors"]:
                report["outcome"] = "infrastructure_error"
                report["passed"] = False
            if report_path is not None:
                destination = Path(report_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
                )
    return report
