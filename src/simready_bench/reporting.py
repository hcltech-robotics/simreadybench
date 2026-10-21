"""Portable benchmark measurements, captures and Harbor traces."""

from __future__ import annotations
import csv, hashlib, html, json, math
from collections import Counter
from pathlib import Path
from simready_bench.provenance import provenance_fields, validate_run_spec
from simready_bench.composites import (
    APPEARANCE_FIELDS,
    KINEMATIC_FIELDS,
    NAMES as COMPOSITE_NAMES,
    columns as composite_columns,
    scores as composite_scores,
)


def read_results(work_dir):
    root = Path(work_dir).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError("Existing nonsymlink run directory required")
    path = next(
        (
            root / name
            for name in ("results.json", "progress.json")
            if (root / name).is_file()
        ),
        None,
    )
    if path is None:
        raise ValueError("No completed matrix results or partial progress found")
    if path.resolve() != path or path.is_symlink() or path.stat().st_nlink != 1:
        raise ValueError("Confined regular results required")
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "simready-bench-matrix-result-v1"
        or type(value.get("results")) is not list
    ):
        raise ValueError("Matrix result schema required")
    return value


def _copy_evidence(source, root, target, *, sha256=None, maximum=32 * 1024**2):
    source = Path(source).absolute()
    if (
        source.resolve() != source
        or not source.is_relative_to(root)
        or (not source.is_file())
        or (source.stat().st_nlink != 1)
    ):
        raise ValueError("Evidence must be a regular file inside the run")
    if source.stat().st_size > maximum:
        raise ValueError("Evidence exceeds export size bound")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError("Diagnostic image hash changed")
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError("Evidence destination collision")
    else:
        with target.open("xb") as stream:
            stream.write(raw)
        target.chmod(384)
    return {"path": target.name, "sha256": digest, "bytes": len(raw)}


def _display(value):
    if value is None:
        return "Not measured"
    if type(value) is float:
        return format(value, ".2f").rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join((str(v) for v in value)) if value else "none"
    return str(value)


def _table(headers, rows):
    esc = lambda value: html.escape(_display(value))
    return (
        '<div class="table"><table><thead><tr>'
        + "".join(("<th>" + esc(h) + "</th>" for h in headers))
        + "</tr></thead><tbody>"
        + "".join(
            (
                "<tr>" + "".join(("<td>" + esc(v) + "</td>" for v in row)) + "</tr>"
                for row in rows
            )
        )
        + "</tbody></table></div>"
    )


MAX_GENERATED_REPORT_BYTES = 128 * 1024**2
MAX_REQUEST_RECORD_BYTES = 128 * 1024

USAGE_FIELDS = (
    "inference_calls_used",
    "validation_calls_used",
    "model_request_errors",
    "elapsed_seconds",
    "model_request_seconds",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "inference_cost_usd",
    "compute_cost_usd",
    "total_cost_usd",
)
USAGE_HEADINGS = (
    "Model requests",
    "Validation calls",
    "Request errors",
    "Elapsed time (s)",
    "Model request time (s)",
    "Input tokens",
    "Output tokens",
    "Total tokens",
    "Inference cost (USD)",
    "Compute cost (USD)",
    "Total cost (USD)",
)
_USAGE_COUNTS = frozenset(USAGE_FIELDS[:3] + USAGE_FIELDS[5:8])


def _usage_number(field, value):
    expected = (int,) if field in _USAGE_COUNTS else (int, float)
    return value if type(value) in expected and math.isfinite(value) and value >= 0 else None


def usage_columns(condition):
    """Normalise per-reconstruction usage, retaining unavailable measurements."""
    statistics = condition.get("statistics")
    statistics = statistics if isinstance(statistics, dict) else {}
    values = {
        field: _usage_number(
            field,
            condition[field] if condition.get(field) is not None else statistics.get(field),
        )
        for field in USAGE_FIELDS
    }
    for target, parts in (
        ("total_tokens", ("input_tokens", "output_tokens")),
        ("total_cost_usd", ("inference_cost_usd", "compute_cost_usd")),
    ):
        if values[target] is None and all(values[part] is not None for part in parts):
            values[target] = _usage_number(target, sum(values[part] for part in parts))
    return values


def _usage_cell(field, value):
    if value is not None and field.endswith("_usd"):
        return format(value, ".6g")
    return value


def _compact_requests(records):
    if not isinstance(records, list):
        return []
    fields = (
        "gateway_request_id",
        "model",
        "status",
        "api_path",
        "elapsed_seconds",
        "upstream_status",
        "observed_upstream_status",
        "error_code",
        "cost_status",
        "cost_usd",
        "actual_cost_usd",
    )
    usage_fields = (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens_details",
        "output_tokens_details",
        "prompt_tokens_details",
        "completion_tokens_details",
    )
    compact = []
    for source in records:
        if not isinstance(source, dict):
            continue
        row = {
            key: source[key]
            for key in fields
            if key in source
            and (
                source[key] is None
                or type(source[key]) in (int, float, bool)
                or isinstance(source[key], str)
                and len(source[key]) <= 4096
            )
        }
        for name in ("usage", "partial_usage"):
            if not isinstance(source.get(name), dict):
                continue
            usage = {}
            for key, value in source[name].items():
                if key not in usage_fields:
                    continue
                if type(value) is int:
                    usage[key] = value
                elif isinstance(value, dict):
                    usage[key] = {
                        detail: count
                        for detail, count in value.items()
                        if detail
                        in ("cached_tokens", "reasoning_tokens", "audio_tokens")
                        and type(count) is int
                    }
            row[name] = usage
        compact.append(row)
    if (
        len(json.dumps(compact, separators=(",", ":")).encode())
        > MAX_REQUEST_RECORD_BYTES
    ):
        return []
    return compact


def _statistics(condition):
    requests = _compact_requests(condition.get("gateway_records", []))

    def total(field, values):
        return (
            sum(values)
            if all(_usage_number(field, value) is not None for value in values)
            else None
        )

    count = _usage_number("inference_calls_used", condition.get("inference_calls_used"))
    complete = count is not None and len(requests) == count
    tokens = (
        lambda first, second: total(
            first,
            [
                record.get("usage", {}).get(first, record.get("usage", {}).get(second))
                for record in requests
            ]
        )
        if complete
        else None
    )
    summary = {
        "input_tokens": tokens("input_tokens", "prompt_tokens"),
        "output_tokens": tokens("output_tokens", "completion_tokens"),
        "model_request_seconds": total(
            "model_request_seconds",
            [record.get("elapsed_seconds") for record in requests]
        )
        if complete
        else None,
        "model_request_errors": sum(
            bool(record.get("error_code"))
            or (
                type(record.get("upstream_status")) is int
                and record["upstream_status"] >= 400
            )
            or record.get("status") != "completed"
            for record in requests
        )
        if complete
        else None,
        "compute_cost_usd": condition.get("compute_cost_usd"),
    }
    retained = condition.get("statistics")
    retained = dict(retained) if isinstance(retained, dict) else {}
    for field, value in summary.items():
        if retained.get(field) is None:
            retained[field] = value
    return requests, {**retained, **usage_columns({**condition, "statistics": retained})}


def compact_report(value):
    """Keep final measurements and exported artefacts from a portable report."""
    if (
        not isinstance(value, dict)
        or value.get("schema") != "simready-bench-portable-report-v1"
        or not isinstance(value.get("results"), list)
        or len(value["results"]) > 10000
    ):
        raise ValueError("A portable benchmark report is required")
    summary_fields = (
        "schema",
        "configuration_sha256",
        "planned_conditions",
        "attempted_conditions",
        "completed_conditions",
        "blocked_conditions",
        "status",
        "partial",
        "blocker",
        "failure_stage",
        "error_type",
        "error_code",
        "error_message",
        "scoring_notice",
        "run_spec",
        "operational_provenance",
        "legacy_settings_conditions",
    )
    fields = set(SCORE_FIELDS) | {
        "model_id",
        "trial_id",
        "seed",
        "snapshot_sha256",
        "blocker",
        "failure_stage",
        "error_type",
        "error_code",
        "error_message",
        "failure_scope",
        "automatic_retry",
        "infrastructure_hold",
        "infrastructure_hold_evidence",
        "request_recovery",
        "termination_reason",
        "inference_calls_used",
        "validation_calls_used",
        "elapsed_seconds",
        "inference_cost_usd",
        "inference_cost_kind",
        "usage_reconstruction",
        "scoring_policy_id",
        "official_conformance_passed",
        "fidelity_status",
        "fidelity_complete_geometry_comparison",
        "fidelity_complete_reconstruction_success",
        "candidate",
        "nvidia_validator_report_file",
        "images",
        "traces",
        "export_failures",
        "fidelity",
        "report_path",
        "actor_stop_receipt",
        "actor_stop_audit",
        "optional_score",
        "execution_provenance",
        "protocol_sha256",
        "run_spec_sha256",
        "provenance_coverage",
    }
    fields.update(
        "fidelity_" + component + "_eligible" for component in FIDELITY_COMPONENTS
    )
    score_fields = {
        "schema",
        "policy_id",
        "policy_sha256",
        "scope",
        "status",
        "unknown_reason",
        "score_percent",
        "score_fraction",
        "score_lower_bound",
        "score_upper_bound",
        "base_credit_percent",
        "completion_bonus",
        "all_scored_mandatory_pass",
        "passed_count",
        "failed_count",
        "unknown_count",
        "applicable_count",
        "profile_coverage_fraction",
        "profile_mandatory_count",
        "categories",
        "official_conformance",
        "optional_score",
    }
    check_fields = {
        "requirement_id",
        "requirement_version",
        "category",
        "mandatory",
        "scoring",
        "status",
        "status_basis",
        "weight",
        "base_credit_percent",
    }
    conditions = []
    for source in value["results"]:
        if not isinstance(source, dict):
            raise ValueError("Portable report attempts must be objects")
        row = {key: source[key] for key in fields if key in source}
        row.update(provenance_fields(source, run_spec=value.get("run_spec")))
        primary = source.get("simready")
        if isinstance(primary, dict):
            primary = mandatory_report_metadata(primary)
            row["simready"] = {
                key: primary[key] for key in score_fields if key in primary
            }
            if isinstance(primary.get("checks"), list):
                row["simready"]["checks"] = [
                    {key: item[key] for key in check_fields if key in item}
                    for item in primary["checks"]
                    if isinstance(item, dict)
                ]
        else:
            row["simready"] = None
        row.update(score_partials(row))
        row["composite_scores"] = composite_scores(row)
        requests = _compact_requests(source.get("model_requests", []))
        if requests:
            row["model_requests"] = requests
        statistics = source.get("statistics")
        if isinstance(statistics, dict):
            row["statistics"] = {
                key: statistics[key]
                for key in USAGE_FIELDS
                if key in statistics
            }
        row.update(usage_columns(row))
        illustrations = source.get("illustrations")
        if isinstance(illustrations, dict):
            row["illustrations"] = {
                key: illustrations[key]
                for key in (
                    "status",
                    "renderer",
                    "renderer_plugin",
                    "rig",
                    "rig_sha256",
                    "scoring",
                )
                if key in illustrations
            }
        conditions.append(row)
    if "run_spec" in value:
        validate_run_spec(value["run_spec"])
    if "operational_provenance" in value:
        from simready_bench.provenance import validate_operational_provenance
        history = validate_operational_provenance(value['operational_provenance'])
        if history['run_spec_sha256'] != value.get('run_spec', {}).get('sha256'):
            raise ValueError('Operational history requires its unchanged run specification')
    return {
        **{key: value[key] for key in summary_fields if key in value},
        "results": conditions,
    }


def read_generated_report(path):
    """Read bounded generated output independently of the input manifest limit."""
    source = Path(path).absolute()
    if (
        source.resolve() != source
        or not source.is_file()
        or source.stat().st_nlink != 1
        or source.stat().st_size > MAX_GENERATED_REPORT_BYTES
    ):
        raise ValueError(
            "A regular generated report no larger than 128 MiB is required"
        )
    with source.open("rb") as stream:
        raw = stream.read(MAX_GENERATED_REPORT_BYTES + 1)
    if len(raw) > MAX_GENERATED_REPORT_BYTES:
        raise ValueError("Generated report exceeds 128 MiB")
    return compact_report(json.loads(raw))


def write_scores(path, conditions, *, mode="x"):
    with Path(path).open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for condition in conditions:
            values = {**condition, **score_partials(condition), **usage_columns(condition)}
            row = {key: values.get(key) for key in SCORE_FIELDS}
            row["repetition"] = _human_repetition(condition.get("repetition"))
            writer.writerow(row)


FIDELITY_COMPONENTS = ("shape", "rigid_parts", "joints", "materials")
SCORE_FIELDS = (
    (
        "condition_id",
        "asset_id",
        "requested_model",
        "harness_id",
        "skill_set_id",
        "repetition",
        "status",
        "formal_fidelity_percent",
        "visual_fidelity_percent",
        "kinematic_fidelity_percent",
        "simready_reward",
        "mandatory_passed",
        "mandatory_total",
        "mandatory_failed",
        "mandatory_unknown",
        "mandatory_lower_bound_percent",
        "mandatory_upper_bound_percent",
        "completion_bonus",
        "optional_score_percent",
        "optional_passed",
        "optional_failed",
        "optional_unknown",
        "optional_total",
        "optional_measured",
        "optional_coverage",
        "optional_lower_bound_percent",
        "optional_upper_bound_percent",
        "fidelity_shape_percent",
        "fidelity_rigid_parts_percent",
        "fidelity_joints_percent",
        "fidelity_materials_percent",
        "fidelity_shape_coverage",
        "fidelity_rigid_parts_coverage",
        "fidelity_joints_coverage",
        "fidelity_materials_coverage",
        "fidelity_reference_geometry_status",
    )
    + tuple(
        name + "_fidelity_" + suffix
        for name in COMPOSITE_NAMES
        for suffix in (
            "status",
            "earned",
            "eligible",
            "measured",
            "coverage",
            "lower_bound_percent",
            "upper_bound_percent",
        )
    )
    + APPEARANCE_FIELDS
    + KINEMATIC_FIELDS
    + tuple(
        "fidelity_" + component + "_" + suffix
        for component in FIDELITY_COMPONENTS
        for suffix in ("earned", "eligible", "measured", "status")
    )
    + (
        "fidelity_shape_precision",
        "fidelity_shape_recall",
        "fidelity_shape_samples_each_direction",
        "fidelity_rigid_parts_geometry_percent",
        "fidelity_rigid_parts_structure_percent",
        "fidelity_rigid_parts_structure_earned",
        "fidelity_rigid_parts_structure_eligible",
        "fidelity_joints_reference_count",
        "fidelity_joints_candidate_count",
    )
    + USAGE_FIELDS
)
SCORING_NOTICE = (
    "Formal fidelity measures SimReady Robotics-Prop requirements. Visual fidelity "
    "combines shape and appearance materials. Kinematic fidelity measures joint "
    "recovery and motion correspondence. Each score is reported independently."
)


def mandatory_report_metadata(primary):
    """Tighten fully measured bounds without changing the measured score."""
    result = dict(primary)
    score = primary.get("score_percent")
    if (
        type(primary.get("unknown_count")) is int
        and primary["unknown_count"] == 0
        and type(score) in (int, float)
        and math.isfinite(score)
        and 0 <= score <= 100
    ):
        result.update(score_lower_bound=score, score_upper_bound=score)
    return result


def score_partials(condition):
    """Expose measured credit, eligibility and coverage without rounding the data."""
    primary = mandatory_report_metadata(condition.get("simready") or {})
    optional = condition.get("optional_score") or primary.get("optional_score") or {}
    result = {
        **composite_columns(condition),
        "mandatory_failed": primary.get("failed_count"),
        "mandatory_unknown": primary.get("unknown_count"),
        "mandatory_lower_bound_percent": primary.get("score_lower_bound"),
        "mandatory_upper_bound_percent": primary.get("score_upper_bound"),
    }
    for field, key in (
        ("score_percent", "score_percent"),
        ("passed", "passed_count"),
        ("failed", "failed_count"),
        ("unknown", "unknown_count"),
        ("total", "applicable_count"),
        ("measured", "measured_count"),
        ("coverage", "coverage_fraction"),
        ("lower_bound_percent", "lower_bound_percent"),
        ("upper_bound_percent", "upper_bound_percent"),
    ):
        if key in optional:
            result["optional_" + field] = optional[key]
    components = (condition.get("fidelity") or {}).get("components", {})
    for name in FIDELITY_COMPONENTS:
        component = components.get(name, {})
        prefix = "fidelity_" + name + "_"
        for key in ("earned", "eligible", "measured", "status"):
            if key in component:
                result[prefix + key] = component[key]
        for suffix, key in (
            ("percent", "score_percent"),
            ("coverage", "coverage_fraction"),
        ):
            if key in component:
                result[prefix + suffix] = component[key]
    shape = components.get("shape", {})
    for key in ("precision", "recall", "samples_each_direction"):
        if key in shape:
            result["fidelity_shape_" + key] = shape[key]
    rigid = components.get("rigid_parts", {})
    structure = rigid.get("structure", {})
    for target, source, key in (
        ("geometry_percent", rigid, "geometry_score_percent"),
        ("structure_percent", structure, "score_percent"),
        ("structure_earned", structure, "earned"),
        ("structure_eligible", structure, "eligible"),
    ):
        if key in source:
            result["fidelity_rigid_parts_" + target] = source[key]
    joints = components.get("joints", {})
    for field in ("reference", "candidate"):
        if field + "_joint_count" in joints:
            result["fidelity_joints_" + field + "_count"] = joints[
                field + "_joint_count"
            ]
    return result


def _fraction(earned, eligible):
    if type(earned) not in (int, float) or type(eligible) not in (int, float):
        return None
    return format(earned, ".6g") + "/" + format(eligible, ".6g")


def _score_cell(score, *, earned=None, eligible=None, measured=None):
    if eligible == 0:
        return "N/A"
    fraction = _fraction(earned, eligible)
    value = _display(score) + "%" if type(score) in (int, float) else "Not measured"
    if fraction:
        value += " (" + fraction + ")"
    if (
        type(measured) in (int, float)
        and type(eligible) in (int, float)
        and measured < eligible
    ):
        value += ", " + _fraction(measured, eligible) + " measured"
    return value


def _profile_cell(condition, *, optional=False):
    primary = condition.get("simready") or {}
    value = (
        (condition.get("optional_score") or primary.get("optional_score") or {})
        if optional
        else primary
    )
    score = value.get("score_percent") if optional else condition.get("simready_reward")
    total = value.get("applicable_count")
    passed = value.get("passed_count")
    cell = _score_cell(score, earned=passed, eligible=total)
    if value.get("unknown_count"):
        cell += ", " + str(value["unknown_count"]) + " unmeasured"
        lower = value.get("lower_bound_percent" if optional else "score_lower_bound")
        upper = value.get("upper_bound_percent" if optional else "score_upper_bound")
        if type(lower) in (int, float) and type(upper) in (int, float):
            cell += ", bounds " + _display(lower) + "%-" + _display(upper) + "%"
    return cell


def _fidelity_cell(condition, component):
    value = (condition.get("fidelity") or {}).get("components", {}).get(component, {})
    return _score_cell(
        _fidelity_score(condition, component),
        earned=value.get("earned"),
        eligible=value.get("eligible"),
        measured=value.get("measured"),
    )


def _human_repetition(value):
    return value + 1 if type(value) is int and value >= 0 else None


def _fidelity_score(condition, component):
    part = condition.get("fidelity", {}).get("components", {}).get(component, {})
    score = condition.get(
        "fidelity_" + component + "_percent", part.get("score_percent")
    )
    return "Not applicable" if score is None and part.get("eligible") == 0 else score


def _failure_details(value):
    if value.get("status") == "completed":
        return ""
    fields = [
        ("Reason", "blocker"),
        ("Stage", "failure_stage"),
        ("Error", "error_code"),
        ("Error type", "error_type"),
        ("Details", "error_message"),
    ]
    facts = [(label, value[key]) for label, key in fields if value.get(key) is not None]
    return (
        (
            '<details class="failure"><summary>Failure details</summary>'
            + _table(["Field", "Details"], facts)
            + "</details>"
        )
        if facts
        else ""
    )


def _capture_label(image):
    view = str(image.get("view") or "Capture")
    if "/" in view:
        role, angle = (part.strip() for part in view.split("/", 1))
        role = {"reference": "Reference", "candidate": "Reconstruction"}.get(role, role)
        return role + ": " + angle
    return view[0].upper() + view[1:]


def render_html(summary):
    """Render exported measurements and artefact links for benchmark readers."""
    esc = lambda value: html.escape(_display(value))
    conditions = summary.get("results", [])
    headings = [
        "Asset",
        "Model",
        "Harness",
        "Skills",
        "Repetition",
        "Status",
        "Formal fidelity (%)",
        "Visual fidelity (%)",
        "Kinematic fidelity (%)",
        "SimReady Robotics-Prop optional",
        "Shape (%)",
        "Rigid parts (%)",
        "Joints (%)",
        "Materials (%)",
        *USAGE_HEADINGS,
    ]
    matrix_rows = []
    sections = []
    for index, condition in enumerate(conditions):
        asset = condition.get("asset_id")
        model = condition.get("requested_model") or condition.get("model_id")
        repetition = _human_repetition(condition.get("repetition"))
        optional_score = _profile_cell(condition, optional=True)
        composites = composite_scores(condition)
        headline = [
            _score_cell(
                composites[name].get("score_percent"),
                earned=composites[name].get("earned"),
                eligible=composites[name].get("eligible"),
                measured=composites[name].get("measured"),
            )
            for name in COMPOSITE_NAMES
        ]
        fidelity_scores = [
            _fidelity_cell(condition, component) for component in FIDELITY_COMPONENTS
        ]
        usage = usage_columns(condition)
        usage_values = [_usage_cell(field, usage[field]) for field in USAGE_FIELDS]
        matrix_rows.append(
            [
                asset,
                model,
                condition.get("harness_id"),
                condition.get("skill_set_id"),
                repetition,
                condition.get("status"),
                *headline,
                optional_score,
                *fidelity_scores,
                *usage_values,
            ]
        )
        heading = esc(asset) + " / " + esc(model)
        if repetition is not None:
            heading += " / repetition " + str(repetition)
        identity = _table(
            ["Model", "Harness", "Skills", "Repetition", "Status"],
            [
                [
                    model,
                    condition.get("harness_id"),
                    condition.get("skill_set_id"),
                    repetition,
                    condition.get("status"),
                ]
            ],
        )
        scores = _table(
            [
                "Formal fidelity (%)",
                "Visual fidelity (%)",
                "Kinematic fidelity (%)",
                "SimReady Robotics-Prop optional",
                "Shape (%)",
                "Rigid parts (%)",
                "Joints (%)",
                "Materials (%)",
            ],
            [[*headline, optional_score, *fidelity_scores]],
        )
        scores += "<h3>Requests, timing and costs</h3>" + _table(
            USAGE_HEADINGS, [usage_values]
        )
        primary = condition.get("simready") or {}
        if (
            primary.get("passed_count") is not None
            and primary.get("applicable_count") is not None
        ):
            scores += (
                "<p>"
                + esc(primary["passed_count"])
                + " of "
                + esc(primary["applicable_count"])
                + " mandatory requirements passed.</p>"
            )
        fidelity = condition.get("fidelity") or {}
        if fidelity:
            measurements = []
            for component in FIDELITY_COMPONENTS:
                part = fidelity.get("components", {}).get(component, {})
                coverage = part.get("coverage_fraction")
                coverage = coverage * 100 if type(coverage) in (int, float) else None
                measurements.append(
                    [
                        component.replace("_", " ").capitalize(),
                        _fidelity_score(condition, component),
                        _fraction(part.get("earned"), part.get("eligible")),
                        part.get("measured"),
                        part.get("eligible"),
                        coverage,
                    ]
                )
            scores += "<h3>Direct USD reconstruction fidelity</h3>" + _table(
                [
                    "Component",
                    "Fidelity (%)",
                    "Credit",
                    "Measured",
                    "Eligible",
                    "Reference coverage (%)",
                ],
                measurements,
            )
            components = fidelity.get("components", {})
            appearance = composites["visual"]["factors"]["appearance"]
            if appearance["eligible"] or appearance["status"] == "unknown":
                scores += "<h3>Appearance materials</h3>" + _table(
                    ["Score (%)", "Earned/eligible", "Status"],
                    [
                        [
                            appearance["score_percent"],
                            _fraction(appearance["earned"], appearance["eligible"]),
                            appearance["status"],
                        ]
                    ],
                )
            kinematic = composites["kinematic"]
            if kinematic.get("groups"):
                scores += "<h3>Kinematic correspondence</h3>" + _table(
                    [
                        "Factor",
                        "Score (%)",
                        "Earned/eligible",
                        "Measured/eligible",
                        "Coverage (%)",
                    ],
                    [
                        [
                            name.replace("_", " ").capitalize(),
                            part.get("score_percent"),
                            _fraction(part.get("earned"), part.get("eligible")),
                            _fraction(part.get("measured"), part.get("eligible")),
                            100 * part.get("coverage_fraction", 0),
                        ]
                        for name, part in kinematic["groups"].items()
                    ],
                )
                scores += _table(
                    [
                        "Reference joint",
                        "Candidate joint",
                        "Score (%)",
                        "Motion RMS (mm)",
                        "Status",
                    ],
                    [
                        [
                            joint.get("reference_path", joint.get("reference_joint")),
                            joint.get("candidate_path", joint.get("candidate_joint")),
                            joint.get("score_percent"),
                            joint["motion_rms_m"] * 1000
                            if joint.get("motion_rms_m") is not None
                            else None,
                            joint.get("reason", joint.get("status")),
                        ]
                        for joint in kinematic.get("joints", [])
                    ],
                )
            shape = components.get("shape", {})
            if shape.get("precision") is not None:
                scores += "<h3>Surface correspondence</h3>" + _table(
                    [
                        "Precision (%)",
                        "Recall (%)",
                        "Samples per direction",
                        "Distance band (mm)",
                        "RMS distance (mm)",
                        "95th-percentile distance (mm)",
                    ],
                    [
                        [
                            shape["precision"] * 100,
                            shape["recall"] * 100
                            if shape.get("recall") is not None
                            else None,
                            shape.get("samples_each_direction"),
                            shape["band_m"] * 1000
                            if shape.get("band_m") is not None
                            else None,
                            shape["rms_m"] * 1000
                            if shape.get("rms_m") is not None
                            else None,
                            shape["p95_m"] * 1000
                            if shape.get("p95_m") is not None
                            else None,
                        ]
                    ],
                )
            rigid = components.get("rigid_parts", {})
            if rigid.get("parts"):
                scores += "<h3>Rigid-part correspondence</h3>" + _table(
                    ["Reference body", "Candidate body", "Surface correspondence (%)"],
                    [
                        [
                            part.get("reference_body"),
                            part.get("candidate_body"),
                            part.get("score_percent"),
                        ]
                        for part in rigid["parts"]
                    ],
                )
            joints = components.get("joints", {})
            if joints.get("reference_joint_count") is not None:
                scores += "<h3>Joint correspondence</h3>" + _table(
                    ["Reference joints", "Candidate joints", "Passed/eligible checks"],
                    [
                        [
                            joints.get("reference_joint_count"),
                            joints.get("candidate_joint_count"),
                            _fraction(joints.get("earned"), joints.get("eligible")),
                        ]
                    ],
                )
            for title, checks in (
                (
                    "Rigid-structure checks",
                    rigid.get("structure", {}).get("checks", []),
                ),
                ("Joint checks", joints.get("checks", [])),
            ):
                if checks:
                    eligible = Counter(check.get("feature") for check in checks)
                    measured = Counter(
                        check.get("feature")
                        for check in checks
                        if type(check.get("passed")) is bool
                    )
                    passed = Counter(
                        check.get("feature")
                        for check in checks
                        if check.get("passed") is True
                    )
                    scores += (
                        "<h3>"
                        + title
                        + "</h3>"
                        + _table(
                            ["Feature", "Passed/eligible", "Measured/eligible"],
                            [
                                [
                                    feature,
                                    _fraction(passed[feature], total),
                                    _fraction(measured[feature], total),
                                ]
                                for feature, total in eligible.items()
                            ],
                        )
                    )
            materials = components.get("materials", {}).get("features", {})
            if materials:
                scores += "<h3>Material correspondence</h3>" + _table(
                    ["Property", "Credit/eligible", "Correspondence (%)"],
                    [
                        [
                            feature,
                            _fraction(part.get("earned"), part.get("eligible")),
                            100 * part["earned"] / part["eligible"]
                            if part.get("eligible")
                            else "N/A",
                        ]
                        for feature, part in materials.items()
                    ],
                )
        links = []
        candidate = condition.get("candidate") or {}
        if candidate.get("status") == "available":
            links += [
                '<a href="' + esc(candidate["entrypoint"]) + '">Generated USD</a>',
                '<a href="'
                + esc(candidate["manifest"])
                + '">Asset package manifest</a>',
            ]
        validator = condition.get("nvidia_validator_report_file")
        if validator:
            links.append(
                '<a href="'
                + esc(validator["path"])
                + '">NVIDIA validation findings</a>'
            )
        if condition.get("report_path"):
            links.append(
                '<a href="' + esc(condition["report_path"]) + '">Attempt report</a>'
            )
        for number, trace in enumerate(condition.get("traces", []), 1):
            label = (
                "Harbor trajectory"
                if trace.get("kind") == "agent_trajectory"
                else "Harbor log"
            )
            links.append(
                '<a href="evidence/'
                + esc(trace["path"])
                + '">'
                + label
                + " "
                + str(number)
                + "</a>"
            )
        artefacts = (
            '<p class="artefacts">' + " | ".join(links) + "</p>" if links else ""
        )
        images = condition.get("images", [])
        captures = ""
        if images:
            figures = []

            # Keep each view's reference and reconstruction adjacent.
            def image_order(image):
                view = str(image.get("view") or "")
                role, _, angle = view.partition("/")
                return (angle.strip() or view, 0 if role.strip() == "reference" else 1)

            for image in sorted(images, key=image_order):
                path = "evidence/" + esc(image["path"])
                label = _capture_label(image)
                context = [condition.get("harness_id"), condition.get("skill_set_id")]
                context = [str(value) for value in context if value is not None]
                if repetition is not None:
                    context.append("repetition " + str(repetition))
                caption = label + (" | " + " | ".join(context) if context else "")
                figures.append(
                    '<figure><a href="'
                    + path
                    + '"><img loading="lazy" src="'
                    + path
                    + '" alt="'
                    + esc(asset)
                    + ": "
                    + esc(label)
                    + '"></a><figcaption>'
                    + esc(caption)
                    + "</figcaption></figure>"
                )
            captures = (
                '<h3>Matched captures</h3><div class="images">'
                + "".join(figures)
                + "</div>"
            )
        sections.append(
            '<section id="attempt-'
            + str(index + 1)
            + '"><h2>'
            + heading
            + "</h2>"
            + identity
            + scores
            + _failure_details(condition)
            + artefacts
            + captures
            + "</section>"
        )
    completed = summary.get("completed_conditions")
    if completed is None:
        completed = sum(
            condition.get("status") == "completed" for condition in conditions
        )
    planned = summary.get("planned_conditions")
    if planned is None:
        planned = len(conditions)
    progress = (
        "<p>"
        + esc(completed)
        + " of "
        + esc(planned)
        + " reconstruction attempts completed.</p>"
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; img-src 'self'; style-src 'unsafe-inline'\">"
        "<title>SimReadyBench results</title><style>"
        "body{font:16px Georgia,serif;line-height:1.6;max-width:1200px;margin:3rem auto;padding:0 1.5rem;background:#fff;color:#202020}"
        "a{color:#254e75}h1,h2,h3{line-height:1.25}summary{cursor:pointer}details{margin:1rem 0}"
        ".images{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1.5rem}figure{margin:0}img{width:100%;height:auto}"
        "figcaption{font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #ddd;padding:8px}"
        ".table{overflow-x:auto;margin:1rem 0}section{border-top:1px solid #ddd;padding:1.5rem 0}.artefacts{font-size:14px}"
        "</style><h1>SimReadyBench results</h1><p>"
        + esc(summary.get("scoring_notice", SCORING_NOTICE))
        + "</p>"
        + progress
        + '<p><a href="scores.csv">Scores CSV</a> | <a href="results.json">Detailed results JSON</a></p>'
        + _failure_details(summary)
        + "<h2>Results by asset</h2>"
        + _table(headings, matrix_rows)
        + "".join(sections)
        + "</html>"
    )


def export_report(work_dir, output_dir, *, results=None):
    root = Path(work_dir).absolute()
    out = Path(output_dir).absolute()
    if root.resolve() != root or (not root.is_dir() and results is None):
        raise ValueError(
            "Existing nonsymlink run directory or explicit blocked results required"
        )
    if (
        out.exists()
        or out.is_symlink()
        or out.parent.resolve() != out.parent
        or not out.parent.is_dir()
    ):
        raise ValueError(
            "Fresh report directory below existing nonsymlink parent required"
        )
    if out == root or out.is_relative_to(root) or root.is_relative_to(out):
        raise ValueError("Durable report must be separate from RAM run tree")
    result = read_results(root) if results is None else results
    out.mkdir(mode=0o700)
    evidence = out / "evidence"
    evidence.mkdir(mode=0o700)
    conditions = []
    for index, condition in enumerate(result.get("results", [])):
        evaluation = condition.get("evaluation") or {}
        primary = evaluation.get("simready") or {}
        row = {
            key: condition.get(key)
            for key in (
                "condition_id",
                "asset_id",
                "harness_id",
                "model_id",
                "skill_set_id",
                "repetition",
                "status",
            )
        }
        row["requested_model"] = (
            condition.get("requested_model")
            or (condition.get("model") or {}).get("model_id")
            or condition.get("model_id")
        )
        row.update(
            simready_reward=primary.get("score_percent"),
            mandatory_passed=primary.get("passed_count"),
            mandatory_total=primary.get("applicable_count"),
            completion_bonus=primary.get("completion_bonus"),
            scoring_policy_id=primary.get("policy_id"),
            official_conformance_passed=primary.get("official_conformance", {}).get(
                "passed"
            ),
        )
        row.update(composite_columns(condition))
        optional = evaluation.get("optional_score") or primary.get("optional_score")
        if isinstance(optional, dict):
            row["optional_score"] = optional
        fidelity = condition.get("fidelity") or {}
        geometry = fidelity.get("reference_geometry_status")
        if geometry is None:
            coverage = (
                fidelity.get("components", {}).get("shape", {}).get("coverage_fraction")
            )
            geometry = (
                "partial"
                if type(coverage) in (int, float) and coverage < 1
                else "unknown"
            )
        row.update(
            fidelity_status=fidelity.get("status", "unavailable"),
            fidelity_reference_geometry_status=geometry,
            fidelity_complete_geometry_comparison=fidelity.get(
                "complete_geometry_comparison"
            ),
            fidelity_complete_reconstruction_success=fidelity.get(
                "complete_reconstruction_success"
            ),
        )
        for component in FIDELITY_COMPONENTS:
            measured = fidelity.get("components", {}).get(component, {})
            row["fidelity_" + component + "_percent"] = measured.get("score_percent")
            row["fidelity_" + component + "_eligible"] = measured.get("eligible")
            row["fidelity_" + component + "_coverage"] = measured.get(
                "coverage_fraction"
            )
        images, traces, failures = [], [], []
        captures = []
        illustration = condition.get("illustrations") or {}
        if illustration.get("images"):
            captures = [
                *captures,
                {"event": "Matched captures", "images": illustration["images"]},
            ]
        for event in captures:
            for image in event.get("images", []):
                try:
                    identity = (
                        str(event.get("event"))
                        + image["sha256"]
                        + str(image.get("view"))
                    )
                    name = (
                        "image-"
                        + str(index)
                        + "-"
                        + hashlib.sha256(identity.encode()).hexdigest()[:20]
                        + ".png"
                    )
                    receipt = _copy_evidence(
                        image["file"], root, evidence / name, sha256=image["sha256"]
                    )
                    images.append(
                        {
                            **receipt,
                            "view": image.get("view"),
                            "event": event.get("event"),
                            "turn": event.get("turn"),
                        }
                    )
                except (ValueError, OSError, KeyError) as exc:
                    failures.append({"kind": "image", "error": str(exc)})
        for number, path in enumerate(condition.get("harbor_trace_paths", [])):
            try:
                name = "trace-" + str(index) + "-" + str(number) + Path(path).suffix
                traces.append(
                    {
                        **_copy_evidence(path, root, evidence / name),
                        "kind": "agent_trajectory"
                        if Path(path).name == "trajectory.json"
                        else "agent_log",
                    }
                )
            except (ValueError, OSError) as exc:
                failures.append({"kind": "trace", "error": str(exc)})
        model_requests, statistics = _statistics(condition)
        candidate = {"status": "unavailable", "reason": "no_frozen_authored_submission"}
        if condition.get("submission") is not None:
            from simready_bench.candidate_export import export_candidate

            try:
                candidate = export_candidate(
                    condition["submission"],
                    work_root=root,
                    report_root=out,
                    condition_id=condition.get("condition_id"),
                    expected_snapshot_sha256=condition.get("snapshot_sha256"),
                )
            except (ValueError, OSError, KeyError, TypeError):
                candidate = {
                    "status": "unavailable",
                    "reason": "frozen_submission_missing_unsafe_changed_or_over_export_limit",
                }
                failures.append({"kind": "candidate", "error": candidate["reason"]})
        validator_file = None
        if isinstance(evaluation.get("nvidia_validator_report"), dict):
            report_path = evidence / ("nvidia-validator-" + str(index) + ".json")
            from simready_bench.engine.validation_reports import write_report_file

            receipt = write_report_file(
                report_path, evaluation["nvidia_validator_report"]
            )
            validator_file = {
                "path": "evidence/" + report_path.name,
                **receipt,
                "complete": True,
                "findings_truncated": False,
            }
        details = {
            key: condition.get(key)
            for key in (
                "trial_id",
                "seed",
                "snapshot_sha256",
                "blocker",
                "failure_stage",
                "error_type",
                "error_code",
                "error_message",
                "failure_scope",
                "automatic_retry",
                "infrastructure_hold",
                "infrastructure_hold_evidence",
                "request_recovery",
                "termination_reason",
                "inference_calls_used",
                "validation_calls_used",
                "elapsed_seconds",
                "inference_cost_usd",
                "inference_cost_kind",
                "usage_reconstruction",
                "actor_stop_receipt",
                "actor_stop_audit",
                "optional_score",
                "execution_provenance",
                "protocol_sha256",
                "run_spec_sha256",
                "provenance_coverage",
            )
        }
        conditions.append(
            {
                **details,
                **row,
                "simready": primary or None,
                "nvidia_validator_report_file": validator_file,
                "model_requests": model_requests,
                "statistics": statistics,
                "fidelity": fidelity,
                "composite_scores": composite_scores(condition),
                "illustrations": {
                    key: value for key, value in illustration.items() if key != "images"
                },
                "candidate": candidate,
                "images": images,
                "traces": traces,
                "export_failures": failures,
            }
        )
    summary = {
        key: value
        for key, value in result.items()
        if key not in ("results", "allowance_ledger")
    }
    summary.update(
        schema="simready-bench-portable-report-v1",
        results=conditions,
        scoring_notice=SCORING_NOTICE,
    )
    summary = compact_report(summary)
    (out / "results.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (out / "results.json").chmod(0o600)
    write_scores(out / "scores.csv", summary["results"])
    (out / "scores.csv").chmod(0o600)
    (out / "report.html").write_text(render_html(summary))
    (out / "report.html").chmod(0o600)
    return {
        "directory": str(out),
        "html": str(out / "report.html"),
        "scores": str(out / "scores.csv"),
        "results": str(out / "results.json"),
        "conditions": len(conditions),
        "provider_calls": 0,
    }
