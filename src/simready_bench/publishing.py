"""Publish completed reconstructions and their scientific readouts to Hugging Face."""

from copy import deepcopy
import csv
from functools import lru_cache
import hashlib
from importlib.resources import files
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

import yaml

from .benchmark_metadata import build_benchmark_metadata
from .config import AUTHORING_PROTOCOL, condition_identity
from .leaderboard import leaderboard_entry
from .provenance import provenance_fields, validate_operational_provenance, validate_run_spec
from .reporting import (
    SCORE_FIELDS,
    mandatory_report_metadata,
    read_generated_report,
    render_html,
    USAGE_FIELDS,
    usage_columns,
    write_scores,
)


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
_PROFILE = {"id": "Robotics-Prop", "version": "4.0.0"}
_MAX_JSON = 128 * 1024**2
_VIEWER_SCHEMA = "simreadybench-viewer-index/v3"
_CSV_VIEWER_SCHEMA = "simreadybench-viewer-index/v2"
_LEGACY_VIEWER_SCHEMA = "simreadybench-viewer-index/v1"
_VIEWER_AXES = (
    "by_model", "by_harness", "by_skills", "by_reference_set",
    "by_protocol", "by_reasoning", "by_asset",
)
_SPLIT = re.compile(r"[a-z0-9][a-z0-9_]{0,95}\Z")
_VIEWER_IMAGES = ("reference_image", "candidate_image")
_VIEWER_CAPTURE_FIELDS = (*_VIEWER_IMAGES, "capture_view")
_VIEWER_PROTOCOL_FIELDS = (
    "run_id", "authoring_protocol", "reasoning_effort", "output_limit", "max_output_tokens",
)


class ViewerRebuildRequired(ValueError):
    """Retained source tables are needed to update the shared viewer schema."""


def _viewer_columns(columns):
    names = set(columns) | set(USAGE_FIELDS)
    return ([name for name in _VIEWER_CAPTURE_FIELDS if name in names]
            + [name for name in SCORE_FIELDS if name in names]
            + sorted(names - set(SCORE_FIELDS) - set(_VIEWER_CAPTURE_FIELDS)))


def _viewer_features(columns):
    labels = {"condition_id", "asset_id", "requested_model", "harness_id", "skill_set_id", "status"}
    return [{"name": name, "dtype": (
        "image" if name in _VIEWER_IMAGES else
        "int64" if name in ("repetition", "max_output_tokens") else
        "float64" if name in SCORE_FIELDS and name not in labels and not name.endswith("_status") else
        "string"
    )} for name in columns]


def _viewer_schema_digest(columns):
    return hashlib.sha256(json.dumps(_viewer_features(columns), sort_keys=True).encode()).hexdigest()


def _viewer_parquet(records, columns):
    """Store image bytes and explicitly typed nullable measurements together."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    features = _viewer_features(columns)
    types = {"int64": pa.int64(), "float64": pa.float64(), "string": pa.string(),
             "image": pa.struct([("bytes", pa.binary()), ("path", pa.string())])}
    metadata = {feature["name"]: (
        {"_type": "Image"} if feature["dtype"] == "image" else
        {"_type": "Value", "dtype": feature["dtype"]}
    ) for feature in features}
    schema = pa.schema([(feature["name"], types[feature["dtype"]]) for feature in features],
                       metadata={b"huggingface": json.dumps({"info": {"features": metadata}}).encode()})
    converted = []
    for record in records:
        current = {}
        for feature in features:
            name, dtype = feature["name"], feature["dtype"]
            value = record.get(name, "")
            if dtype in ("int64", "float64"):
                value = None if value == "" else int(value) if dtype == "int64" else float(value)
                if value is not None and not math.isfinite(value):
                    raise ValueError("Viewer measurements must be finite or missing")
            current[name] = value
        converted.append(current)
    stream = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(converted, schema=schema), stream, compression="zstd",
                   row_group_size=100, write_page_index=True)
    return stream.getvalue().to_pybytes()


class PublicationError(RuntimeError):
    """An upload can be retried without repeating the reconstruction."""

    def __init__(self, message, *, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def _json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > _MAX_JSON:
        raise ValueError(
            "A regular readout JSON file no larger than 128 MiB is required"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _relative(value):
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ValueError("A relative artefact path is required")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != value
    ):
        raise ValueError("An artefact path must stay inside its readout")
    return path


def _source(root, value):
    path = root.joinpath(*_relative(value).parts)
    if path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("An artefact must be a confined regular file")
    return path


def _copy(root, item, target, *, path=None, size_key="bytes", maximum=32 * 1024**2):
    source = _source(root, path or item["path"])
    expected = item.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch("[a-f0-9]{64}", expected):
        raise ValueError("Every artefact requires its recorded SHA-256")
    size = source.stat().st_size
    if size > maximum or item.get(size_key) != size:
        raise ValueError(
            "An artefact differs from its recorded size or exceeds its bound"
        )
    with source.open("rb") as stream:
        before = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != expected:
        raise ValueError("An artefact differs from its recorded SHA-256")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024**2)
    with target.open("rb") as stream:
        after = hashlib.file_digest(stream, "sha256").hexdigest()
    if after != expected:
        raise ValueError("An artefact changed during publication")
    return {"sha256": after, "bytes": size}


def _settings(manifest):
    defaults = manifest.get("defaults", {})
    return {
        "settings_schema": "simready-bench-public-settings-v2",
        "reference_assets": [asset["id"] for asset in manifest["assets"]],
        "reference_revision": manifest.get("source", {}).get("revision"),
        "repetitions": manifest["repetitions"],
        "master_seed": manifest["seed"],
        "models": [
            {
                key: model[key]
                for key in ("id", "model_id", "adapter", "generation")
                if key in model
            }
            for model in manifest["models"]
        ],
        "harnesses": [
            {key: harness[key] for key in ("id", "agent", "adapter") if key in harness}
            for harness in manifest["harnesses"]
        ],
        "skill_sets": [
            {
                "id": skill["id"],
                "skills": [
                    PurePosixPath(path.replace("\\", "/")).name
                    for path in skill.get("paths", [])
                ],
            }
            for skill in manifest["skill_sets"]
        ],
        "authoring_protocol": manifest.get("authoring_protocol"),
        "scoring_profile": manifest["scoring"]["profile"],
        "scoring_policy_id": manifest["scoring"]["policy_id"],
        "generation": defaults.get("generation", {}),
        "system_prompt": defaults.get("system_prompt", ""),
        "budgets": deepcopy(defaults.get("budgets", {})),
        "overrides": deepcopy(manifest.get("overrides", [])),
        "tool_protocol": manifest.get("tool_protocol"),
        "fidelity": {
            key: value
            for key, value in defaults.get("fidelity", {}).items()
            if key not in ("seconds", "enabled")
        },
        "captures": {
            key: value
            for key, value in defaults.get("illustrations", {}).items()
            if key != "seconds"
        },
    }


def _compatible_settings(previous, current):
    """Retain every published setting while exposing older coverage explicitly."""
    if not isinstance(previous, dict):
        return False
    if previous.get("settings_schema") == "simready-bench-public-settings-v2":
        return previous == current
    return all(key in current and value == current[key] for key, value in previous.items())


def _validate_comparison_rig(rig):
    required = {
        "schema",
        "basis",
        "center",
        "span",
        "projection",
        "aperture",
        "lighting",
        "views",
        "width",
    }

    def vector(value, count):
        return (
            isinstance(value, list)
            and len(value) == count
            and all(
                type(item) in (int, float) and math.isfinite(item) for item in value
            )
        )

    if (
        not isinstance(rig, dict)
        or not required <= rig.keys()
        or rig.keys() - required - {"candidate_alignment"}
        or rig["schema"] != "simready-illustration-rig-v1"
        or rig["basis"] != "reference bounds in metres/Z-up"
        or rig["projection"] != "orthographic"
        or rig["lighting"] != "Storm fixed camera light, source lights disabled"
        or not vector(rig["center"], 3)
        or not vector([rig["span"], rig["aperture"]], 2)
        or rig["span"] <= 0
        or not math.isclose(rig["aperture"], rig["span"] * 1.65, rel_tol=1e-12)
        or type(rig["width"]) is not int
        or not 128 <= rig["width"] <= 1024
        or not isinstance(rig["views"], dict)
        or not rig["views"]
        or rig["views"].keys() - {"isometric", "front", "side", "top"}
    ):
        raise ValueError("A complete canonical Storm comparison rig is required")
    for camera in rig["views"].values():
        if (
            not isinstance(camera, dict)
            or set(camera) != {"camera_to_world", "clipping_range"}
            or not isinstance(camera["camera_to_world"], list)
            or len(camera["camera_to_world"]) != 4
            or not all(vector(row, 4) for row in camera["camera_to_world"])
            or not vector(camera["clipping_range"], 2)
            or not 0 < camera["clipping_range"][0] < camera["clipping_range"][1]
        ):
            raise ValueError(
                "Finite matched camera transforms and clipping are required"
            )


def _comparison_metadata(row):
    """Validate the renderer's comparison rig against its measured alignment."""
    illustrations = row.get("illustrations") or {}
    if not isinstance(illustrations, dict):
        raise ValueError("Comparison metadata must be an object")
    keys = {"rig", "rig_sha256"}
    present = keys & illustrations.keys()
    if not present:
        return None
    if present != keys or not isinstance(illustrations["rig"], dict):
        raise ValueError("The comparison rig and its SHA-256 must be supplied together")
    rig = illustrations["rig"]
    _validate_comparison_rig(rig)
    digest = hashlib.sha256(
        json.dumps(rig, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if digest != illustrations["rig_sha256"]:
        raise ValueError("The comparison rig differs from its recorded SHA-256")
    if rig.get("candidate_alignment") != (row.get("fidelity") or {}).get("alignment"):
        raise ValueError("Comparison and measured reconstruction alignments differ")
    return {key: deepcopy(illustrations[key]) for key in keys}


def _enrich_previous_metadata(previous, current):
    """Permit a previously omitted rig without replacing published evidence."""
    current = {row["condition_id"]: row for row in current}
    result = []
    for original in previous:
        row = deepcopy(original)
        prior_rig = _comparison_metadata(row)
        next_rig = _comparison_metadata(current[row["condition_id"]])
        if prior_rig is None and next_rig is not None:
            row["illustrations"] = next_rig
        if "reasoning_effort" not in row and "reasoning_effort" in current[row["condition_id"]]:
            row["reasoning_effort"] = current[row["condition_id"]]["reasoning_effort"]
        for field in USAGE_FIELDS:
            if row.get(field) is None and current[row["condition_id"]].get(field) is not None:
                row[field] = current[row["condition_id"]][field]
        result.append(row)
    return result


def _signature(rows):
    """Compare scientific values and file hashes independently of local paths."""
    values = []
    for row in rows:
        primary = mandatory_report_metadata(row.get("simready") or {})
        values.append(
            {
                **{
                    key: row.get(key)
                    for key in (*SCORE_FIELDS, "seed", "simready", "fidelity")
                },
                "simready": primary,
                "mandatory_lower_bound_percent": primary.get("score_lower_bound"),
                "mandatory_upper_bound_percent": primary.get("score_upper_bound"),
                "illustrations": _comparison_metadata(row),
                "reasoning_effort": row.get("reasoning_effort"),
                "candidate": sorted(
                    (item["relative_path"], item["sha256"], item["size_bytes"])
                    for item in row["candidate"].get("files", [])
                ),
                "candidate_status": row["candidate"]["status"],
                "candidate_manifest": row["candidate"].get("manifest_sha256"),
                "images": sorted(
                    (item["view"], item["sha256"]) for item in row["images"]
                ),
                "traces": sorted(item["sha256"] for item in row["traces"]),
                "nvidia_findings": (row.get("nvidia_validator_report_file") or {}).get(
                    "sha256"
                ),
                "execution_provenance": provenance_fields(row).get("execution_provenance"),
            }
        )
    return json.dumps(
        sorted(values, key=lambda value: value["condition_id"]),
        sort_keys=True,
        allow_nan=False,
    )


def _authoring_signature(rows):
    """Bind every published authoring value while allowing evaluation repair."""
    values = json.loads(_signature(rows))
    for row in values:
        for key in list(row):
            if key in {"fidelity", "illustrations", "images"} or key.startswith(
                ("fidelity_", "visual_", "kinematic_")
            ):
                del row[key]
    return json.dumps(values, sort_keys=True, allow_nan=False)


def _evaluation_recovery(previous, current, request, revision):
    if (
        not isinstance(request, dict)
        or request.get("schema") != "simready-evaluation-recovery/v1"
        or request.get("previous_revision") != revision
        or not re.fullmatch(r"[a-f0-9]{40}", str(request.get("source_commit", "")))
        or not re.fullmatch(r"[a-f0-9]{64}", str(request.get("source_tree_sha256", "")))
        or not isinstance(request.get("reason"), str)
        or not 1 <= len(request["reason"]) <= 500
        or set(request.get("condition_ids", [])) != {row["condition_id"] for row in previous}
    ):
        raise ValueError("Explicit evaluation recovery requires its pinned revision, source and conditions")
    if _authoring_signature(previous) != _authoring_signature(current):
        raise ValueError("Evaluation recovery cannot change authoring, formal scores, candidates, traces or usage")
    receipt = {key: deepcopy(request[key]) for key in (
        "schema", "previous_revision", "source_commit", "source_tree_sha256", "reason", "condition_ids",
    )}
    if 'stages' in request:
        stages = request['stages']
        if not isinstance(stages, list) or not 1 <= len(stages) <= 8 or any(
            not isinstance(stage, dict)
            or set(stage) != {'operation', 'source_commit', 'source_tree_sha256'}
            or stage['operation'] not in {'cpu_fidelity', 'captures'}
            or not re.fullmatch(r'[a-f0-9]{40}', str(stage['source_commit']))
            or not re.fullmatch(r'[a-f0-9]{64}', str(stage['source_tree_sha256']))
            for stage in stages
        ):
            raise ValueError('Evaluation recovery stages require exact evaluator source identities')
        receipt['stages'] = deepcopy(stages)
    receipt.update(
        previous_measurements_sha256=hashlib.sha256(_signature(previous).encode()).hexdigest(),
        replacement_measurements_sha256=hashlib.sha256(_signature(current).encode()).hexdigest(),
        authoring_sha256=hashlib.sha256(_authoring_signature(current).encode()).hexdigest(),
        authoring_unchanged=True,
    )
    return receipt


def _stage_candidate(candidate, source, stage, prefix):
    if candidate.get("status") == "unavailable":
        return {"status": "unavailable", "reason": "no_exportable_generated_usd"}
    if candidate.get("status") != "available":
        raise ValueError("A generated package must declare its availability")
    package = PurePosixPath("assets") / prefix / "candidate"
    files, recorded = [], {}
    for item in candidate["files"]:
        relative = _relative(item["relative_path"])
        if relative.as_posix() in recorded:
            raise ValueError("A candidate package repeats a file")
        target = package / relative
        copied = _copy(
            source,
            item,
            stage.joinpath(*target.parts),
            size_key="size_bytes",
            maximum=256 * 1024**2,
        )
        recorded[relative.as_posix()] = (copied["sha256"], copied["bytes"])
        files.append(
            {
                "path": target.as_posix(),
                "relative_path": relative.as_posix(),
                "sha256": copied["sha256"],
                "size_bytes": copied["bytes"],
            }
        )
    if sum(item[1] for item in recorded.values()) > 256 * 1024**2:
        raise ValueError("The generated package exceeds 256 MiB")
    original_manifest = _source(source, candidate["manifest"])
    manifest = _json(original_manifest)
    entrypoint = _relative(candidate["source_entrypoint"]).as_posix()
    if (
        not recorded
        or entrypoint not in recorded
        or manifest.get("entrypoint") != entrypoint
    ):
        raise ValueError("The generated USD is absent from its package manifest")
    if recorded != {
        item["relative_path"]: (item["sha256"], item["size_bytes"])
        for item in manifest["files"]
    }:
        raise ValueError("The generated package differs from its manifest")
    manifest_target = package / "manifest.json"
    _copy(
        source,
        {
            "path": candidate["manifest"],
            "sha256": candidate["manifest_sha256"],
            "bytes": original_manifest.stat().st_size,
        },
        stage.joinpath(*manifest_target.parts),
    )
    return {
        "status": "available",
        "schema": "simready-bench-portable-candidate-v1",
        "directory": package.as_posix(),
        "entrypoint": (package / entrypoint).as_posix(),
        "manifest": manifest_target.as_posix(),
        "manifest_sha256": candidate["manifest_sha256"],
        "source_entrypoint": entrypoint,
        "files": files,
    }


def _stage_row(row, source, stage):
    prefix = PurePosixPath(row["asset_id"]) / (
        "reconstruction-" + str(row["repetition"] + 1)
    )
    public = {
        key: deepcopy(row[key])
        for key in (
            *SCORE_FIELDS,
            "seed",
            "scoring_policy_id",
            "official_conformance_passed",
            "inference_cost_kind",
            "usage_reconstruction",
            "request_recovery",
        )
        if key in row
    }
    public.update(usage_columns(row))
    public.update(provenance_fields(row))
    primary_fields = (
        "schema",
        "policy_id",
        "policy_sha256",
        "status",
        "score_percent",
        "score_fraction",
        "score_lower_bound",
        "score_upper_bound",
        "passed_count",
        "failed_count",
        "unknown_count",
        "applicable_count",
        "profile_mandatory_count",
        "profile_coverage_fraction",
        "completion_bonus",
        "categories",
        "checks",
        "official_conformance",
        "optional_score",
    )
    public["simready"] = {
        key: deepcopy(row["simready"][key])
        for key in primary_fields
        if key in row["simready"]
    }
    optional = row.get("optional_score") or row["simready"].get("optional_score")
    if isinstance(optional, dict):
        public["optional_score"] = deepcopy(optional)
    public["fidelity"] = {
        key: deepcopy(row.get("fidelity", {})[key])
        for key in (
            "schema",
            "status",
            "complete_geometry_comparison",
            "alignment",
            "components",
            "candidate_invalid",
        )
        if key in row.get("fidelity", {})
    }
    from simready_bench.composites import scores as composite_scores

    public["composite_scores"] = composite_scores(row)
    comparison = _comparison_metadata(row)
    if comparison is not None:
        public["illustrations"] = comparison
    public["candidate"] = _stage_candidate(row["candidate"], source, stage, prefix)
    public["images"], public["traces"] = [], []
    for kind in ("images", "traces"):
        for index, item in enumerate(row.get(kind, []), 1):
            if kind == "traces" and item.get("kind") != "agent_trajectory":
                continue
            if kind == "images":
                view = str(item.get("view") or "")
                if not (
                    "matched" in view.lower()
                    or view.startswith(("reference /", "candidate /"))
                ):
                    continue
                suffix = PurePosixPath(item["path"]).suffix.lower()
                if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
                    raise ValueError("Unsupported capture format")
                name = re.sub("[^a-z0-9]+", "-", view.lower()).strip("-") or "capture"
                name = name + "-" + str(index) + suffix
                metadata = {"view": view}
            else:
                name, metadata = (
                    "harbor-trajectory-" + str(index) + ".json",
                    {"kind": "agent_trajectory"},
                )
            relative = prefix / name
            copied = _copy(
                source,
                item,
                stage / "evidence" / Path(*relative.parts),
                path="evidence/" + item["path"],
            )
            public[kind].append({**metadata, "path": relative.as_posix(), **copied})
    validator = row.get("nvidia_validator_report_file")
    if validator is None and row["simready"].get("status") == "invalid_candidate":
        public["nvidia_validator_report_file"] = None
        return public
    if (
        not isinstance(validator, dict)
        or validator.get("complete") is not True
        or validator.get("findings_truncated") is not False
    ):
        raise ValueError("The full NVIDIA findings file is required")
    target = PurePosixPath("evidence") / prefix / "nvidia-validator.json"
    copied = _copy(source, validator, stage.joinpath(*target.parts))
    public["nvidia_validator_report_file"] = {
        "path": target.as_posix(),
        "complete": True,
        "findings_truncated": False,
        **copied,
    }
    return public


def _viewer_index(value):
    index = deepcopy(value) if value is not None else {
        "schema": _VIEWER_SCHEMA, "runs": {}, "splits": {axis: {} for axis in _VIEWER_AXES},
        "columns": _viewer_columns([]), "schema_sha256": _viewer_schema_digest(_viewer_columns([])),
    }
    legacy = isinstance(index, dict) and index.get("schema") == _LEGACY_VIEWER_SCHEMA
    extension = ".parquet" if isinstance(index, dict) and index.get("schema") == _VIEWER_SCHEMA else ".csv"
    keys = {"schema", "runs", "splits"} | (set() if legacy else {"columns", "schema_sha256"})
    if (
        not isinstance(index, dict) or index.get("schema") not in (_VIEWER_SCHEMA, _CSV_VIEWER_SCHEMA, _LEGACY_VIEWER_SCHEMA)
        or set(index) != keys
        or not isinstance(index.get("runs"), dict)
        or not isinstance(index.get("splits"), dict)
        or set(index["splits"]) != set(_VIEWER_AXES)
    ):
        raise ValueError("The dataset viewer index has an unexpected schema")
    if not legacy:
        columns = index["columns"]
        if (
            not isinstance(columns, list) or not all(isinstance(name, str) and name for name in columns)
            or len(columns) != len(set(columns)) or columns != _viewer_columns(columns)
            or index["schema_sha256"] != _viewer_schema_digest(columns)
            or index["runs"] and "condition_id" not in columns
        ):
            raise ValueError("The dataset viewer columns or schema fingerprint changed")
    for registry in index["splits"].values():
        if not isinstance(registry, dict):
            raise ValueError("Viewer split names must be distinct and immutable")
        for identity, split in registry.items():
            if not isinstance(identity, str) or not isinstance(split, str) or not _SPLIT.fullmatch(split):
                raise ValueError("Invalid dataset viewer split identity")
            if json.dumps(json.loads(identity), ensure_ascii=True, sort_keys=True) != identity:
                raise ValueError("A viewer group requires its canonical identity")
        if len(set(registry.values())) != len(registry):
            raise ValueError("Viewer split names must be distinct and immutable")
    for run_id, run in index["runs"].items():
        if not isinstance(run_id, str) or not _NAME.fullmatch(run_id) or not isinstance(run, dict):
            raise ValueError("Invalid dataset viewer run identity")
        if (
            set(run) != ({"scores", "groups"} if legacy else {"scores", "viewer_scores", "groups"})
            or run.get("scores") != "runs/" + run_id + "/scores.csv"
            or not legacy and run.get("viewer_scores") != "viewer/" + run_id + "/scores" + extension
            or not isinstance(run.get("groups"), dict)
            or set(run["groups"]) != set(_VIEWER_AXES)
        ):
            raise ValueError("Viewer sources must retain their original run paths")
        for axis, groups in run["groups"].items():
            if not isinstance(groups, list):
                raise ValueError("Viewer group membership must be a list")
            for group in groups:
                if (
                    not isinstance(group, dict)
                    or set(group) != {"split", "path"}
                    or group.get("split") not in index["splits"][axis].values()
                    or group.get("path") not in (
                        run["scores"] if legacy else run["viewer_scores"],
                        "viewer/" + run_id + "/" + axis + "/" + str(group.get("split")) + extension,
                    )
                ):
                    raise ValueError("A viewer group must use its owned score source")
    return index


def _viewer_split(index, axis, value):
    identity = json.dumps(value, ensure_ascii=True, sort_keys=True)
    registry = index["splits"][axis]
    if identity in registry:
        return registry[identity]
    label = "unknown" if value is None else value
    if not isinstance(label, str):
        raise ValueError("Viewer factor values must be strings or null")
    base = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:70] or "value"
    split = base
    if split in registry.values():
        split += "__" + hashlib.sha256(identity.encode()).hexdigest()[:16]
    if split in registry.values():
        raise ValueError("A dataset viewer split identity collides")
    registry[identity] = split
    return split


def _reference_set(settings):
    if settings.get("reference_set") is not None:
        return settings["reference_set"]
    assets, revision = settings.get("reference_assets"), settings.get("reference_revision")
    if not isinstance(assets, list) or not assets or not all(isinstance(asset, str) for asset in assets):
        return None
    reference = json.loads(files("simready_bench").joinpath("resources/benchmarks/nvidia-25.json").read_text())
    if (
        len(assets) == len(set(assets))
        and set(assets) == {asset["id"] for asset in reference["assets"]}
        and revision == reference.get("source", {}).get("revision")
        and revision is not None
    ):
        return "nv25"
    identity = json.dumps({"assets": sorted(assets), "revision": revision}, sort_keys=True).encode()
    return "custom_" + str(len(assets)) + "_assets_" + hashlib.sha256(identity).hexdigest()[:12]


def _reasoning(row, settings):
    effective = row.get("effective_settings")
    if isinstance(effective, dict) and isinstance(effective.get("generation"), dict):
        return effective["generation"].get("reasoning_effort")
    if "reasoning_effort" in row:
        return row["reasoning_effort"]
    if isinstance(row.get("generation"), dict) and "reasoning_effort" in row["generation"]:
        return row["generation"]["reasoning_effort"]
    return None


def _manifest_generation(row, manifest):
    model = next(model for model in manifest["models"] if model["model_id"] == row["requested_model"])
    generation = {**model.get("generation", {}), **manifest.get("defaults", {}).get("generation", {})}
    labels = {"asset": row["asset_id"], "harness": row["harness_id"],
              "model": model["id"], "skill_set": row["skill_set_id"]}
    for override in manifest.get("overrides", []):
        if all(value == "*" or labels[key] == value for key, value in override["match"].items()):
            generation.update(override["settings"].get("generation", {}))
    return generation


def _viewer_protocol(row, settings, run_id):
    protocol = row.get("authoring_protocol", settings.get("authoring_protocol"))
    effective = row.get("effective_settings") or {}
    generation = effective.get("generation", row.get("generation", {}))
    policy, cap = generation.get("output_limit"), generation.get("max_output_tokens")
    bounded = re.fullmatch(r"authoring-[1-9][0-9]*-bounded-output-([1-9][0-9]*)-v1(?:-tools-bounded-v1)?(?:-continue-length-v1)?", protocol or "")
    if bounded:
        expected = int(bounded[1])
        if policy not in (None, "bounded") or cap not in (None, expected):
            raise ValueError("Viewer output settings differ from the authoring protocol")
        policy, cap = "bounded", expected
    elif re.fullmatch(r"authoring-[1-9][0-9]*-native-output-v1(?:-tools-bounded-v1)?(?:-continue-length-v1)?", protocol or ""):
        if policy not in (None, "provider-native") or cap is not None:
            raise ValueError("Viewer output settings differ from the authoring protocol")
        policy = "provider-native"
    return {
        "run_id": run_id, "authoring_protocol": protocol or "",
        "reasoning_effort": _reasoning(row, settings) or "",
        "output_limit": policy or "", "max_output_tokens": cap if cap is not None else "",
    }


@lru_cache(maxsize=2)
def _missing_viewer_capture(role):
    """Give missing images an explicit display value in the Hub viewer."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (384, 384), "#f1f3f5")
    draw = ImageDraw.Draw(image)
    draw.text((192, 182), role.capitalize() + " capture", fill="#495057", anchor="mm")
    draw.text((192, 202), "Unavailable", fill="#495057", anchor="mm")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _viewer_captures(row, run_id, image_reader):
    """Embed a shared camera from the published comparison captures."""
    captures = {role: {} for role in ("reference", "candidate")}
    for item in row.get("images", []):
        view = str(item.get("view", "")).lower().split(" / ")
        if len(view) == 2 and view[0] in captures and view[1] in ("isometric", "front", "side", "top"):
            # Duplicate labels would make the selected pair ambiguous.
            if view[1] in captures[view[0]]:
                raise ValueError("Comparison captures require distinct role and camera labels")
            captures[view[0]][view[1]] = item
    available = set(captures["reference"]) & set(captures["candidate"])
    if not available:
        available = set(captures["reference"]) | set(captures["candidate"])
    camera = next((view for view in ("isometric", "front", "side", "top") if view in available), "")
    # The Hub renders null Image values as "Not supported with pagination yet".
    # Keep missing evidence explicit without changing the source measurements.
    result = {role + "_image": {
        "bytes": _missing_viewer_capture(role), "path": role + "-capture-unavailable.png",
    } for role in captures}
    result["capture_view"] = camera
    if camera and not callable(image_reader):
        raise ValueError("An image reader is required to embed viewer captures")
    for role in captures:
        item = captures[role].get(camera)
        if item:
            path = _relative(item["path"])
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                raise ValueError("Unsupported viewer capture format")
            raw = image_reader("runs/" + run_id + "/evidence/" + path.as_posix())
            if not isinstance(raw, bytes) or not raw or len(raw) > 32 * 1024**2:
                raise ValueError("Viewer captures require non-empty image bytes no larger than 32 MiB")
            if ("bytes" in item and item["bytes"] != len(raw)
                or "sha256" in item and item["sha256"] != hashlib.sha256(raw).hexdigest()):
                raise ValueError("A viewer capture differs from its published evidence")
            from PIL import Image

            try:
                with Image.open(io.BytesIO(raw)) as image:
                    expected = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP"}
                    if image.format != expected[path.suffix.lower()]:
                        raise ValueError("A viewer capture format differs from its file extension")
                    image.verify()
            except OSError as error:
                raise ValueError("A viewer capture must contain a valid image") from error
            result[role + "_image"] = {"bytes": raw, "path": path.name}
    return result


def build_viewer_update(run_id, readout, score_csv, *, index=None, repo_id=None, image_reader=None):
    """Normalise one run to the shared viewer schema without changing its source.

    Schema expansion with retained runs requires ``build_viewer_dataset`` so
    every affected viewer file can be replaced in the same Hub commit.
    """
    if not isinstance(run_id, str) or not _NAME.fullmatch(run_id):
        raise ValueError("Use a simple viewer run identifier")
    if not isinstance(readout, dict):
        raise ValueError("A public readout object is required")
    index = _viewer_index(index)
    if isinstance(score_csv, bytes):
        score_csv = score_csv.decode("utf-8-sig")
    if not isinstance(score_csv, str) or len(score_csv.encode("utf-8")) > _MAX_JSON:
        raise ValueError("A bounded UTF-8 score table is required")
    reader = csv.DictReader(io.StringIO(score_csv, newline=""))
    columns = reader.fieldnames
    if not columns or len(columns) != len(set(columns)) or "condition_id" not in columns:
        raise ValueError("A score table requires distinct columns and condition identities")
    if set(columns) & set(_VIEWER_CAPTURE_FIELDS):
        raise ValueError("Capture columns are derived from the published comparison evidence")
    canonical = _viewer_columns([*index.get("columns", []), *columns,
                                 *_VIEWER_CAPTURE_FIELDS, *_VIEWER_PROTOCOL_FIELDS])
    if set(index["runs"]) - {run_id} and (
        index["schema"] != _VIEWER_SCHEMA or canonical != index["columns"]
    ):
        raise ViewerRebuildRequired("A shared viewer schema update requires all retained source tables")
    index.update(schema=_VIEWER_SCHEMA, columns=canonical,
                 schema_sha256=_viewer_schema_digest(canonical))
    records = list(reader)
    source_rows = readout.get("results", [])
    if (
        not isinstance(source_rows, list) or len(records) != len(source_rows)
        or not all(isinstance(row, dict) and isinstance(row.get("condition_id"), str) for row in source_rows)
    ):
        raise ValueError("Viewer score rows must match their public readout")
    rows = {row["condition_id"]: row for row in source_rows}
    if len(rows) != len(source_rows) or len({row.get("condition_id") for row in records}) != len(records):
        raise ValueError("Viewer rows require unique condition identities")
    settings = readout.get("settings") or {}
    if not isinstance(settings, dict):
        raise ValueError("Public viewer settings must be an object")
    reference = _reference_set(settings)
    grouped = {axis: {} for axis in _VIEWER_AXES}
    for record in records:
        row = rows.get(record["condition_id"])
        if row is None or None in record or any(value is None for value in record.values()):
            raise ValueError("Viewer score rows must match their public readout")
        for field in ("requested_model", "harness_id", "skill_set_id", "asset_id"):
            if field in record and record[field] != str(row.get(field) or ""):
                raise ValueError("A viewer score row changes its public factor identity")
        protocol = _viewer_protocol(row, settings, run_id)
        for field, value in protocol.items():
            if field in record and record[field] != str(value):
                raise ValueError("A viewer score row changes its public protocol identity")
        record.update(protocol)
        record.update(_viewer_captures(row, run_id, image_reader))
        factors = {
            "by_model": row.get("requested_model"),
            "by_harness": row.get("harness_id"),
            "by_skills": row.get("skill_set_id"),
            "by_reference_set": row.get("reference_set", reference),
            "by_protocol": row.get("authoring_protocol", settings.get("authoring_protocol")),
            "by_reasoning": _reasoning(row, settings),
            "by_asset": row.get("asset_id"),
        }
        for axis, value in factors.items():
            identity = json.dumps(value, ensure_ascii=True, sort_keys=True)
            grouped[axis].setdefault(identity, []).append(record)
    score_path = "runs/" + run_id + "/scores.csv"
    viewer_path = "viewer/" + run_id + "/scores.parquet"
    groups, shards = {}, {}

    def encoded(selected):
        return _viewer_parquet(selected, canonical)

    shards[viewer_path] = encoded(records)
    for axis, values in grouped.items():
        groups[axis] = []
        for identity, records in sorted(values.items()):
            split = _viewer_split(index, axis, json.loads(identity))
            path = viewer_path
            if len(values) > 1:
                path = "viewer/" + run_id + "/" + axis + "/" + split + ".parquet"
                shards[path] = encoded(records)
            groups[axis].append({"split": split, "path": path})
    index["runs"][run_id] = {"scores": score_path, "viewer_scores": viewer_path, "groups": groups}
    return index, shards


def build_viewer_dataset(sources, *, index=None, repo_id=None, image_reader=None):
    """Build schema-compatible viewer files from all retained raw run sources.

    ``sources`` maps run identifiers to ``(public_readout, score_csv)`` pairs.
    The returned files are confined to ``viewer/``. Original run artefacts and
    measurements are read only, including during index migration. ``image_reader``
    returns the bytes of a capture from its repository-relative evidence path.
    """
    previous = _viewer_index(index)
    if not isinstance(sources, dict) or not set(previous["runs"]) <= sources.keys():
        raise ViewerRebuildRequired("All retained viewer runs require their original source tables")
    columns = list(previous.get("columns", []))
    for _, raw in sources.values():
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig")
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > _MAX_JSON:
            raise ValueError("A bounded UTF-8 score table is required")
        fields = csv.DictReader(io.StringIO(raw, newline="")).fieldnames
        if not fields or len(fields) != len(set(fields)) or "condition_id" not in fields:
            raise ValueError("A score table requires distinct columns and condition identities")
        columns.extend(fields)
    columns = _viewer_columns([*columns, *_VIEWER_CAPTURE_FIELDS, *_VIEWER_PROTOCOL_FIELDS])
    updated = {
        "schema": _VIEWER_SCHEMA, "runs": {}, "splits": deepcopy(previous["splits"]),
        "columns": columns, "schema_sha256": _viewer_schema_digest(columns),
    }
    shards = {}
    for run_id, (readout, raw) in sorted(sources.items()):
        updated, current = build_viewer_update(run_id, readout, raw, index=updated,
                                             repo_id=repo_id, image_reader=image_reader)
        shards.update(current)
    return updated, shards


def _data_paths(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [path for item in value for path in _data_paths(item)]
    if isinstance(value, dict):
        return _data_paths(value.get("path", []))
    return []


def render_viewer_card(existing, index):
    """Keep the card body and manual configs while updating native HF subsets."""
    index = _viewer_index(index)
    if index["schema"] != _VIEWER_SCHEMA:
        raise ViewerRebuildRequired("Legacy viewer sources require schema-normalised files")
    if existing.startswith("---\n"):
        end = existing.index("\n---\n", 4)
        metadata, body = yaml.safe_load(existing[4:end]) or {}, existing[end + 5:]
    else:
        metadata, body = {}, existing or "# SimReadyBench results\n"
    if not isinstance(metadata, dict) or not isinstance(metadata.get("configs", []), list):
        raise ValueError("The existing dataset card has invalid configurations")
    managed = {"all", *_VIEWER_AXES}
    configs = []
    for original in metadata.get("configs", []):
        if not isinstance(original, dict):
            raise ValueError("A dataset viewer configuration must be an object")
        name = original.get("config_name")
        if name in index["runs"] and _data_paths(original.get("data_files")) == [index["runs"][name]["scores"]]:
            continue
        if name in managed:
            if metadata.get("simreadybench_viewer") not in (1, 2, 3):
                raise ValueError("A managed viewer name conflicts with an existing manual configuration")
            continue
        config = deepcopy(original)
        config.pop("default", None)
        configs.append(config)
    generated = [{
        "config_name": "all",
        "data_files": [{"split": "test", "path": sorted(run["viewer_scores"] for run in index["runs"].values())}],
        "features": _viewer_features(index["columns"]),
    }]
    for axis in _VIEWER_AXES:
        groups = {}
        for run in index["runs"].values():
            for group in run["groups"][axis]:
                groups.setdefault(group["split"], set()).add(group["path"])
        if groups:
            generated.append({"config_name": axis, "features": _viewer_features(index["columns"]), "data_files": [
                {"split": split, "path": sorted(paths)} for split, paths in sorted(groups.items())
            ]})
    default = next((config for config in generated if config["config_name"] == "by_model"), generated[0])
    default["default"] = True
    metadata.update(configs=generated + configs, simreadybench_viewer=3)
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False) + "---\n" + body


class HuggingFacePublisher:
    """Publish one model, harness and skill configuration as results complete."""

    def __init__(self, repo_id, run_id, manifest, *, api=None):
        if (
            not isinstance(repo_id, str)
            or len(repo_id.split("/")) != 2
            or not all(_NAME.fullmatch(part) for part in repo_id.split("/"))
        ):
            raise ValueError("Use a Hugging Face dataset ID in owner/name form")
        if not isinstance(run_id, str) or not _NAME.fullmatch(run_id):
            raise ValueError("Use a simple run identifier")
        self.repo_id, self.run_id = repo_id, run_id
        self.manifest = deepcopy(manifest)
        self._api, self._private = api, None
        if (
            not manifest.get("assets")
            or any(
                len(manifest.get(key, [])) != 1
                for key in ("models", "harnesses", "skill_sets")
            )
            or type(manifest.get("repetitions")) is not int
            or not 1 <= manifest["repetitions"] <= 10000
            or type(manifest.get("seed")) is not int
            or manifest.get("scoring", {}).get("profile") != _PROFILE
        ):
            raise ValueError(
                "A complete experiment with one model, harness and skill set is required"
            )
        ids = [asset["id"] for asset in manifest["assets"]]
        if len(set(ids)) != len(ids) or any(
            not _NAME.fullmatch(asset) for asset in ids
        ):
            raise ValueError("Reference assets require distinct simple identifiers")
        if len(ids) * manifest["repetitions"] > 10000:
            raise ValueError(
                "A published experiment can contain at most 10000 reconstructions"
            )
        self.expected = {}
        for asset, harness, model, skill in itertools.product(
            *(manifest[key] for key in ("assets", "harnesses", "models", "skill_sets"))
        ):
            for repetition in range(manifest["repetitions"]):
                labels = {
                    "asset": asset["id"],
                    "harness": harness["id"],
                    "model": model["id"],
                    "skill_set": skill["id"],
                }
                identifier, seed = condition_identity(
                    labels, repetition=repetition, master_seed=manifest["seed"],
                    authoring_protocol=manifest.get("authoring_protocol", AUTHORING_PROTOCOL),
                )
                self.expected[identifier] = (
                    asset["id"],
                    harness["id"],
                    model["model_id"],
                    skill["id"],
                    repetition,
                    seed,
                )
        self.settings = _settings(self.manifest)
        json.dumps(self.settings, allow_nan=False)

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi()
        return self._api

    def check_access(self):
        """Check Hub authentication and prepare the destination before inference."""
        import importlib

        for dependency in ("huggingface_hub", "matplotlib", "pyarrow.parquet"):
            try:
                importlib.import_module(dependency)
            except (ImportError, OSError) as error:
                raise PublicationError(
                    "Publication dependencies are unavailable. Install simready-bench[publish] before starting a run",
                    retryable=False,
                ) from error
        from huggingface_hub.errors import RepositoryNotFoundError

        identity = self.api.whoami()
        if identity.get("auth", {}).get("accessToken", {}).get("role") == "read":
            raise PublicationError(
                "Hugging Face authentication requires write access", retryable=False
            )
        try:
            info = self.api.repo_info(self.repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            self.api.create_repo(
                self.repo_id, repo_type="dataset", private=True, exist_ok=True
            )
            info = self.api.repo_info(self.repo_id, repo_type="dataset")
        self._private = info.private
        return {"repo_id": self.repo_id, "private": info.private, "revision": info.sha}

    def _download(self, filename, revision):
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(
                self.api.hf_hub_download(
                    self.repo_id, filename, repo_type="dataset", revision=revision
                )
            ).resolve(strict=True)
        except EntryNotFoundError:
            return None

    def _rows(self, value):
        rows, identities = [], set()
        for row in value["results"]:
            if row.get("status") != "completed":
                continue
            identifier = row.get("condition_id")
            actual = tuple(
                row.get(key)
                for key in (
                    "asset_id",
                    "harness_id",
                    "requested_model",
                    "skill_set_id",
                    "repetition",
                    "seed",
                )
            )
            if (
                identifier not in self.expected
                or actual != self.expected[identifier]
                or identifier in identities
            ):
                raise ValueError(
                    "A completed reconstruction differs from the original experiment identity"
                )
            identities.add(identifier)
            score, primary = row.get("simready_reward"), row.get("simready") or {}
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not 0 <= score <= 100
                or primary.get("score_percent") != score
                or primary.get("status") not in ("scored", "invalid_candidate")
                or primary.get("official_conformance", {}).get("profile") != _PROFILE
                or primary.get("policy_id") != self.manifest["scoring"]["policy_id"]
                or primary.get("unknown_count") != 0
                or primary.get("profile_coverage_fraction") != 1
            ):
                raise ValueError(
                    "Every completed reconstruction requires its measured 4.0.0 profile score"
                )
            passed = primary.get("passed_count")
            if (
                type(passed) is not int
                or not 0 <= passed <= 46
                or primary.get("applicable_count") != 46
                or primary.get("profile_mandatory_count") != 46
                or primary.get("failed_count") != 46 - passed
                or not math.isclose(
                    score, 95 * passed / 46 + (5 if passed == 46 else 0), abs_tol=1e-9
                )
            ):
                raise ValueError(
                    "The score differs from the complete NVIDIA 95 plus five definition"
                )
            if primary["status"] == "invalid_candidate" and (
                score != 0 or passed != 0 or primary.get("completion_bonus") != 0
            ):
                raise ValueError("An invalid submission must receive zero credit")
            rows.append(row)
        return sorted(rows, key=lambda row: self.expected[row["condition_id"]])

    def _viewer_update(self, existing, public, score_csv, revision, stage):
        def image_reader(name):
            target = stage.joinpath(*_relative(name).parts)
            if target.is_file():
                return target.read_bytes()
            retained = self._download(name, revision)
            if retained is None:
                raise ValueError("A retained viewer capture requires its published evidence")
            return retained.read_bytes()

        retained = self._download("viewer-index.json", revision)
        index = _viewer_index(_json(retained) if retained else None)
        retained_ids = set(index["runs"])
        legacy_ids = set()
        if existing.startswith("---\n"):
            end = existing.index("\n---\n", 4)
            metadata = yaml.safe_load(existing[4:end]) or {}
            for config in metadata.get("configs", []):
                run_id = config.get("config_name")
                if (
                    isinstance(run_id, str) and _NAME.fullmatch(run_id)
                    and run_id not in index["runs"] and run_id != self.run_id
                    and _data_paths(config.get("data_files")) == ["runs/" + run_id + "/scores.csv"]
                ):
                    legacy_ids.add(run_id)
        if index["schema"] == _VIEWER_SCHEMA and not legacy_ids:
            try:
                return build_viewer_update(self.run_id, public, score_csv, index=index,
                                           repo_id=self.repo_id, image_reader=image_reader)
            except ViewerRebuildRequired:
                pass
        sources = {self.run_id: (public, score_csv)}
        for run_id in sorted((retained_ids | legacy_ids) - {self.run_id}):
            readout = self._download("runs/" + run_id + "/results.json", revision)
            scores = self._download("runs/" + run_id + "/scores.csv", revision)
            if readout is None or scores is None:
                raise ValueError("A retained viewer run requires its original score table and readout")
            sources[run_id] = (_json(readout), scores.read_bytes())
        return build_viewer_dataset(sources, index=index, repo_id=self.repo_id, image_reader=image_reader)

    def __call__(self, report, *, final=False, evaluation_recovery=None):
        source = Path(
            report["directory"] if isinstance(report, dict) else report
        ).absolute()
        try:
            value = read_generated_report(source / "results.json")
            rows = self._rows(value)
            if not rows:
                return {
                    "status": "empty",
                    "completed_conditions": 0,
                    "planned_conditions": len(self.expected),
                }
            from huggingface_hub import CommitOperationAdd

            info = self.api.repo_info(self.repo_id, repo_type="dataset")
            if self._private is not None and info.private != self._private:
                raise PublicationError(
                    "The dataset visibility changed during evaluation", retryable=False
                )
            path = "runs/" + self.run_id
            old_readout = self._download(path + "/results.json", info.sha)
            with tempfile.TemporaryDirectory(
                prefix=".hf-readout-", dir=source
            ) as temporary:
                stage = Path(temporary)
                run_stage = stage / "runs" / self.run_id
                run_stage.mkdir(parents=True)
                public = {
                    "schema": "simready-bench-portable-report-v1",
                    "planned_conditions": len(self.expected),
                    "completed_conditions": len(rows),
                    "partial": len(rows) < len(self.expected),
                    "settings": self.settings,
                    "results": [_stage_row(row, source, run_stage) for row in rows],
                }
                if self.manifest.get('defaults', {}).get('fidelity', {}).get('enabled') is True:
                    from simready_bench.completion import evaluation_completion, completion_summary
                    capture_settings = self.manifest.get('defaults', {}).get('illustrations', {})
                    for row in public['results']:
                        row['evaluation_completion'] = evaluation_completion(
                            row, captures_enabled=capture_settings.get('renderer', 'none') != 'none',
                            views=capture_settings.get('views', ['isometric', 'front', 'side']),
                        )
                    public.update(completion_summary(public['results'], len(self.expected)))
                    public['partial'] = not public['benchmark_complete']
                spec = value.get("run_spec")
                if spec is not None:
                    public["run_spec"] = validate_run_spec(spec)
                    _write_json(run_stage / "run-spec.json", public["run_spec"])
                history = value.get('operational_provenance')
                if history is not None:
                    public['operational_provenance'] = validate_operational_provenance(history)
                    if spec is None or history['run_spec_sha256'] != spec['sha256']:
                        raise ValueError('Operational history requires its unchanged run specification')
                    _write_json(run_stage / 'run-provenance.json', history)
                for row in public["results"]:
                    if row.get("execution_provenance") and (
                            spec is None or row["run_spec_sha256"] != spec["sha256"]
                            or row["protocol_sha256"] != spec["protocol_sha256"]):
                        raise ValueError("Every bound attempt requires its unchanged run specification")
                    if row.get("execution_provenance"):
                        provenance_fields(row, run_spec=spec)
                for row in public["results"]:
                    generation = _manifest_generation(row, self.manifest)
                    row["reasoning_effort"] = generation.get("reasoning_effort")
                    row["generation"] = generation
                    row["authoring_protocol"] = self.manifest.get("authoring_protocol")
                if old_readout:
                    old = _json(old_readout)
                    old_rows = self._rows(old)
                    old_ids = {row["condition_id"] for row in old_rows}
                    if not _compatible_settings(old.get("settings"), self.settings) or not old_ids <= {
                        row["condition_id"] for row in rows
                    }:
                        raise ValueError(
                            "Publication would change settings or omit completed reconstructions"
                        )
                    if old.get("run_spec") is not None and old["run_spec"] != spec:
                        raise ValueError("Publication would change the frozen run specification")
                    if old.get('operational_provenance') is not None:
                        previous = validate_operational_provenance(old['operational_provenance'])
                        if (history is None or history['entries'][:len(previous['entries'])] != previous['entries']):
                            raise ValueError('Publication would remove or rewrite operational provenance history')
                    legacy = set(old.get("legacy_settings_conditions", []))
                    if old.get("settings", {}).get("settings_schema") != "simready-bench-public-settings-v2":
                        legacy.update(old_ids)
                    if legacy:
                        public["legacy_settings_conditions"] = sorted(legacy)
                    previous = _enrich_previous_metadata(old_rows, public["results"])
                    current = [row for row in public["results"] if row["condition_id"] in old_ids]
                    recoveries = deepcopy(old.get("evaluation_recoveries", []))
                    if _signature(previous) != _signature(current):
                        if evaluation_recovery is None:
                            raise ValueError("An already published reconstruction has changed")
                        receipt = _evaluation_recovery(previous, current, evaluation_recovery, info.sha)
                        recoveries.append(receipt)
                        identifier = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
                        recovery_path = run_stage / "evaluation-recoveries" / (identifier + ".json")
                        recovery_path.parent.mkdir(exist_ok=True)
                        _write_json(recovery_path, receipt)
                    if recoveries:
                        public["evaluation_recoveries"] = recoveries
                _write_json(run_stage / "results.json", public)
                write_scores(run_stage / "scores.csv", public["results"])
                (run_stage / "report.html").write_text(
                    render_html(public), encoding="utf-8"
                )
                card = self._download("README.md", info.sha)
                existing = card.read_text(encoding="utf-8") if card else ""
                index, shards = self._viewer_update(existing, public, (run_stage / "scores.csv").read_bytes(), info.sha, stage)
                for name, raw in shards.items():
                    target = stage.joinpath(*_relative(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                _write_json(stage / "viewer-index.json", index)
                board_file = self._download("leaderboard.json", info.sha)
                board = _json(board_file) if board_file else {"schema": "simreadybench-leaderboard/v1", "entries": []}
                if board.get("schema") != "simreadybench-leaderboard/v1" or not isinstance(board.get("entries"), list):
                    raise ValueError("The existing leaderboard has an unexpected schema")
                url = "https://huggingface.co/datasets/" + self.repo_id + "/blob/main/" + path + "/report.html"
                try:
                    entry = leaderboard_entry(public, manifest=self.manifest, results_url=url)
                except ValueError:
                    entry = None
                board["entries"] = [item for item in board["entries"] if item.get("results_url") != url]
                if entry:
                    board["entries"].append(entry)
                _write_json(stage / "leaderboard.json", board)
                for name, raw in build_benchmark_metadata(render_viewer_card(existing, index), board).items():
                    target = stage.joinpath(*_relative(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                latest = self.api.repo_info(self.repo_id, repo_type="dataset")
                if latest.private != info.private:
                    raise PublicationError("The dataset visibility changed during publication", retryable=False)
                committed = self.api.create_commit(
                    repo_id=self.repo_id, repo_type="dataset", parent_commit=info.sha,
                    commit_message="Publish reconstruction readouts and benchmark leaderboard",
                    operations=[CommitOperationAdd(path_in_repo=target.relative_to(stage).as_posix(), path_or_fileobj=target)
                                for target in sorted(stage.rglob("*")) if target.is_file()],
                )
            return {
                "status": "published",
                "repo_id": self.repo_id,
                "run_id": self.run_id,
                "completed_conditions": len(rows),
                "planned_conditions": len(self.expected),
                "run_commit": committed.oid,
                "commit": committed.oid,
                "leaderboard_complete": entry is not None,
                "final": bool(final),
            }
        except PublicationError:
            raise
        except ValueError as error:
            raise PublicationError(str(error), retryable=False) from None
        except Exception as error:
            raise PublicationError(
                "Hugging Face publication failed (" + type(error).__name__ + ")"
            ) from None
