"""Content-bound scientific specifications and per-attempt execution receipts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import zipfile


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE = {"absolute_path", "usd_path", "api_key", "token", "password", "secret",
            "authorization", "original_path", "replacement_path"}
PROVENANCE_FIELDS = ("execution_provenance", "protocol_sha256", "run_spec_sha256",
                     "provenance_coverage")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def content_sha256(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _digest(value, label="content"):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(label + " requires a lowercase SHA-256 digest")
    return value


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if key.lower() not in _PRIVATE and not (
                    key == "path" and isinstance(item, str)
                    and (PurePosixPath(item).is_absolute() or PureWindowsPath(item).is_absolute()))}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return deepcopy(value)


def _closure(files):
    if not isinstance(files, list):
        raise ValueError("A content closure must be a list")
    names = set()
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("A content closure requires named files")
        name = item["path"].replace("\\", "/")
        if name.startswith("/") or ":" in name or ".." in name.split("/") or name in names:
            raise ValueError("Content paths must be unique and relative")
        names.add(name)
        _digest(item.get("sha256"))
        if type(item.get("size_bytes")) is not int or item["size_bytes"] < 0:
            raise ValueError("Content closures require nonnegative byte sizes")


def package_contract(source_root=None):
    """Bind every executable module and bundled scientific resource."""
    root = Path(source_root) if source_root is not None else Path(__file__).parent
    for candidate in (root / "src" / "simready_bench", root / "simready_bench", root):
        if candidate.is_dir() and (candidate.name == "simready_bench"):
            root = candidate
            break
    if not root.is_dir() or root.name != "simready_bench":
        raise ValueError("A package source directory is required for the protocol contract")
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ValueError("The protocol contract cannot contain symbolic links")
        if path.is_file():
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError("The protocol contract cannot be empty")
    return {"schema": "simready-bench-package-contract-v1", "files": files,
            "sha256": content_sha256(files)}


def _content_map(files, label):
    if not isinstance(files, dict) or not files:
        raise ValueError(label + " requires a nonempty file map")
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or '\\' in name or ':' in name
                or name.startswith('/') or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError(label + " requires confined relative paths")
        _digest(digest, label)


def package_contract_from_wheel(path):
    """Bind package components to the exact archive dispatched to executors."""
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("The executor wheel must remain a regular file")
    archive_files, names = {}, set()
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name = member.filename.rstrip('/') if member.is_dir() else member.filename
                if name in names:
                    raise ValueError("The executor wheel requires unique confined files")
                _content_map({name: '0' * 64}, "The executor wheel")
                names.add(name)
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("The executor wheel cannot contain symbolic links")
                if not member.is_dir():
                    archive_files[name] = hashlib.sha256(archive.read(member)).hexdigest()
    except zipfile.BadZipFile as error:
        raise ValueError("The executor wheel must be a valid wheel archive") from error
    files = _wheel_package_files(archive_files)
    _content_map(files, "The executor package contract")
    return {"schema": "simready-bench-package-contract-v1", "files": files,
            "sha256": content_sha256(files), "archive_files": archive_files,
            "archive_sha256": content_sha256(archive_files)}


def _wheel_package_files(archive_files):
    installed = {}
    for name, digest in archive_files.items():
        parts = name.split('/')
        if parts[0].endswith('.data') and len(parts) >= 3 and parts[1] in ('purelib', 'platlib'):
            name = '/'.join(parts[2:])
        if name in installed:
            raise ValueError("The executor wheel requires unique installed package files")
        installed[name] = digest
    return {name.removeprefix('simready_bench/'): digest
            for name, digest in installed.items() if name.startswith('simready_bench/')}


def _runtime(value):
    if not isinstance(value, dict):
        raise ValueError("An explicit runtime receipt must be an object")
    value = _public(value)
    for key, item in value.items():
        if key.endswith("sha256") and item is not None:
            _digest(item, "Runtime " + key)
        if key in ("container_image_digest", "image_digest") and item is not None:
            if not isinstance(item, str):
                raise ValueError("Container identities require a SHA-256 digest")
            _digest(item.removeprefix("sha256:"), "Container")
    images = value.get("runtime_images", {})
    if not isinstance(images, dict):
        raise ValueError("Runtime images must be named explicitly")
    for image in images.values():
        if not isinstance(image, str):
            raise ValueError("Runtime image identities must be strings")
        if "sha256:" in image:
            _digest(image.rsplit("sha256:", 1)[1], "Container")
    return value


def _components(manifest, contract):
    reference_files = {}
    freeze = manifest.get("benchmark_freeze")
    if freeze is not None:
        from .manifest import validate_freeze

        validate_freeze(freeze)
        reference_files = {name: {"sha256": item["sha256"], "size_bytes": item["bytes"]}
                           for name, item in freeze["source_files"].items()}
    references = {}
    frozen_digest = content_sha256(reference_files) if reference_files else None
    for asset in manifest.get("assets", []):
        if asset.get("closure"):
            references[asset["id"]] = {"basis": "resolved_closure", "sha256": content_sha256(asset["closure"])}
        elif asset.get("usd") in reference_files:
            references[asset["id"]] = {"basis": "frozen_file_map",
                                       "sha256": frozen_digest}
    skills = {item["id"]: content_sha256(item.get("closure", []))
              for item in manifest.get("skill_sets", [])
              if "closure" in item or not item.get("paths")}
    files = contract["files"]
    groups = {
        "prompt": ("engine/pi.py", "engine/task_contract.py", "asset_inputs.py", "harbor_agents.py"),
        "tool": ("engine/tools.py", "engine/actor_mcp.py", "engine/harbor_services.py", "engine/pi_runtime.py"),
        "evaluator": ("engine/scoring.py", "engine/validator.py", "engine/evaluation.py", "engine/nvidia_profile.json",
                      "engine/nvidia_optional_profile.json", "composites.py", "fidelity.py", "fidelity_worker.py"),
    }
    result = {"references": references, "skills": skills,
              "system_prompt_sha256": hashlib.sha256(
                  manifest.get("defaults", {}).get("system_prompt", "").encode("utf-8")).hexdigest()}
    for name, members in groups.items():
        selected = {member: files[member] for member in members if member in files}
        if selected:
            result[name] = {"files": selected, "sha256": content_sha256(selected)}
    return result


def build_run_spec(experiment, *, runtime=None, source_root=None, scientific_revision=None,
                   wheel=None):
    """Bind the full effective experiment without inventing absent content identities.

    A resolved experiment includes effective conditions, provider facts and limits.
    An older manifest remains representable with explicit partial coverage.
    """
    manifest = _public(experiment.get("normalized_manifest", experiment))
    missing = []
    assets = manifest.get("assets", [])
    for asset in assets:
        for field in ("closure", "input_view_closure"):
            if field in asset:
                _closure(asset[field])
    skills = manifest.get("skill_sets", [])
    if not skills or any(item.get("paths") and "closure" not in item for item in skills):
        missing.append("skill_content")
    for skill in skills:
        if "closure" in skill:
            _closure(skill["closure"])
        if "sha256" in skill:
            _digest(skill["sha256"], "Skill closure")
    policy = experiment.get("operator_policy", {})
    providers = {name: _public({key: provider[key] for key in (
        "kind", "base_url", "model_spec", "project_budget", "hf_router_relay") if key in provider})
        for name, provider in policy.get("providers", {}).items()}
    # The older configuration digest also binds absolute staging paths. Rebind
    # the complete public condition content so host paths do not alter science.
    conditions = [{key: item for key, item in condition.items() if key != "configuration_sha256"}
                  for condition in _public(experiment.get("conditions", []))]
    if not conditions:
        missing.append("effective_conditions")
    if not providers:
        missing.append("provider_registry")
    if not manifest.get("defaults", {}).get("budgets"):
        missing.append("budgets")
    if wheel is not None and source_root is not None:
        raise ValueError("Select the executed wheel or the local package source")
    contract = package_contract_from_wheel(wheel) if wheel is not None else package_contract(source_root)
    components = _components(manifest, contract)
    if not assets or len(components["references"]) != len(assets):
        missing.append("reference_content")
    missing.extend(name + "_content" for name in ("prompt", "tool", "evaluator") if name not in components)
    protocol = {"manifest": manifest, "conditions": conditions,
                "providers": providers, "limits": _public(policy.get("limits", {})),
                "aggregate_budget": _public(experiment.get("aggregate_budget", {})),
                "contract": contract, "components": components}
    if scientific_revision is not None:
        if not isinstance(scientific_revision, str) or not scientific_revision.strip():
            raise ValueError("A scientific revision must be explicit and nonempty")
        protocol["scientific_revision"] = scientific_revision
    effective_runtime = _runtime(runtime or {})
    if wheel is not None:
        wheel_hash = hashlib.sha256(Path(wheel).read_bytes()).hexdigest()
        if effective_runtime.get('wheel_sha256', wheel_hash) != wheel_hash:
            raise ValueError("The runtime wheel differs from its package contract")
        effective_runtime['wheel_sha256'] = wheel_hash
    if "wheel_sha256" not in effective_runtime:
        missing.append("runtime_wheel")
    images = effective_runtime.get("runtime_images", {})
    pinned = effective_runtime.get("container_image_digest") or images and all(
        "sha256:" in image for name, image in images.items() if name != "snapshot")
    if not pinned:
        missing.append("runtime_container_digests")
    result = {"schema": "simready-bench-run-spec-v1", "protocol": protocol,
              "protocol_sha256": content_sha256(protocol), "runtime": effective_runtime,
              "runtime_sha256": content_sha256(effective_runtime),
              "coverage": {"status": "partial" if missing else "complete", "missing": sorted(missing)}}
    result["sha256"] = content_sha256(result)
    return result


def validate_run_spec(value):
    if not isinstance(value, dict) or value.get("schema") != "simready-bench-run-spec-v1":
        raise ValueError("A content-bound run specification is required")
    for field, bound in (("protocol_sha256", value.get("protocol")),
                         ("runtime_sha256", value.get("runtime")),
                         ("sha256", {key: item for key, item in value.items() if key != "sha256"})):
        if _digest(value.get(field), field) != content_sha256(bound):
            raise ValueError("The run specification differs from its recorded digest")
    _runtime(value["runtime"])
    contract = value["protocol"].get("contract", {})
    files = contract.get("files")
    if not isinstance(files, dict) or not files or contract.get("sha256") != content_sha256(files):
        raise ValueError("The package contract differs from its recorded digest")
    for digest in files.values():
        _digest(digest, "Package file")
    _content_map(files, "The package contract")
    if 'archive_files' in contract:
        _content_map(contract['archive_files'], "The wheel archive contract")
        if contract.get('archive_sha256') != content_sha256(contract['archive_files']):
            raise ValueError("The wheel archive contract differs from its recorded digest")
        package_files = _wheel_package_files(contract['archive_files'])
        if package_files != files:
            raise ValueError("The package contract differs from its wheel archive")
    manifest = value["protocol"]["manifest"]
    if value["protocol"].get("components") != _components(manifest, contract):
        raise ValueError("Protocol components differ from their bound content")
    for asset in manifest.get("assets", []):
        for field in ("closure", "input_view_closure"):
            if field in asset:
                _closure(asset[field])
    for skill in manifest.get("skill_sets", []):
        if "closure" in skill:
            _closure(skill["closure"])
    coverage = value.get("coverage", {})
    if (coverage.get("status") not in ("partial", "complete")
            or not isinstance(coverage.get("missing"), list)
            or coverage["missing"] != sorted(set(coverage["missing"]))
            or (coverage["status"] == "complete") != (not coverage["missing"])):
        raise ValueError("Explicit provenance coverage must match missing evidence")
    return deepcopy(value)


def write_run_spec(output, spec):
    """Create a frozen specification once, then require exact identity on resume."""
    from .recovery import _save

    spec = validate_run_spec(spec)
    path = Path(output) / "run-spec.json"
    if path.exists():
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError("The run specification must remain a regular file")
        previous = validate_run_spec(json.loads(path.read_text(encoding="utf-8")))
        if previous != spec:
            raise ValueError("The frozen run specification cannot change")
    else:
        _save(path, spec)
    return path


def _regular_record(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError("Provenance receipts must remain regular confined files")
    return json.loads(path.read_text(encoding='utf-8'))


def _attempt_receipt_path(output, source):
    if not isinstance(source, dict) or not isinstance(source.get('attempt'), str):
        raise ValueError("An immutable attempt source identity is required")
    if not isinstance(source.get('condition_id'), str) or not source['condition_id']:
        raise ValueError("A canonical condition identity is required")
    _digest(source.get('measurement_sha256'), "Source measurement")
    identity = {key: source[key] for key in ('attempt', 'condition_id')}
    return Path(output) / 'provenance' / 'attempts' / (content_sha256(identity) + '.json')


def load_attempt_provenance(output, *, source):
    """Reuse the receipt originally bound to a completed source measurement."""
    path = _attempt_receipt_path(output, source)
    if not path.exists() and not path.is_symlink():
        return None
    record = _regular_record(path)
    if (record.get('schema') != 'simready-bench-attempt-provenance-record-v1'
            or record.get('source') != source
            or record.get('sha256') != content_sha256({key: item for key, item in record.items() if key != 'sha256'})):
        raise ValueError("A completed attempt differs from its immutable provenance source")
    receipt = record.get('receipt')
    provenance_fields({'condition_id': source['condition_id'], 'execution_provenance': receipt})
    return receipt


def write_attempt_provenance(output, *, source, receipt):
    """Freeze one receipt without incorporating subsequent run interventions."""
    from .recovery import _save

    previous = load_attempt_provenance(output, source=source)
    if previous is not None:
        if previous != receipt:
            raise ValueError("The completed attempt provenance cannot change")
        return previous
    provenance_fields({'condition_id': source['condition_id'], 'execution_provenance': receipt})
    path = _attempt_receipt_path(output, source)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent.absolute():
        raise ValueError("Attempt provenance requires a confined record directory")
    record = {'schema': 'simready-bench-attempt-provenance-record-v1',
              'source': deepcopy(source), 'receipt': deepcopy(receipt)}
    record['sha256'] = content_sha256(record)
    _save(path, record)
    return deepcopy(receipt)


def validate_operational_provenance(value):
    if (not isinstance(value, dict) or value.get('schema') != 'simready-bench-operational-provenance-v1'
            or value.get('sha256') != content_sha256({key: item for key, item in value.items() if key != 'sha256'})
            or not isinstance(value.get('entries'), list)):
        raise ValueError("A content-bound operational provenance history is required")
    _digest(value.get('run_spec_sha256'), "Operational run specification")
    seen = set()
    for item in value['entries']:
        if not isinstance(item, dict) or item.get('kind') not in ('repair', 'intervention'):
            raise ValueError("Operational provenance requires named history entries")
        name = item.get('path')
        _content_map({name: item.get('sha256')}, "Operational history")
        if name in seen:
            raise ValueError("Operational history paths must remain unique")
        seen.add(name)
    return deepcopy(value)


def collect_operational_provenance(output, run_spec):
    """Append run-level observations while preserving every earlier identity."""
    from .recovery import _save

    output = Path(output)
    spec = validate_run_spec(run_spec)
    paths = [output / 'executor-repair.json',
             *sorted((output / 'executor-repairs').glob('*.json')),
             *sorted((output / 'operational-interventions').glob('*.json'))]
    current = {}
    for path in paths:
        if path.exists() or path.is_symlink():
            _regular_record(path)
            name = path.relative_to(output).as_posix()
            current[name] = {'kind': 'intervention' if name.startswith('operational-interventions/') else 'repair',
                             'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    path = output / 'run-provenance.json'
    entries = []
    if path.exists() or path.is_symlink():
        previous = validate_operational_provenance(_regular_record(path))
        if previous['run_spec_sha256'] != spec['sha256']:
            raise ValueError("Operational history belongs to a different frozen run")
        entries = previous['entries']
        if any(current.get(item['path']) != item for item in entries):
            raise ValueError("Operational provenance history cannot be removed or rewritten")
    seen = {item['path'] for item in entries}
    entries.extend(item for name, item in sorted(current.items()) if name not in seen)
    result = {'schema': 'simready-bench-operational-provenance-v1',
              'run_spec_sha256': spec['sha256'], 'entries': entries}
    result['sha256'] = content_sha256(result)
    if not path.exists() or result != previous:
        _save(path, result)
    return result


def build_attempt_provenance(run_spec, *, attempt_id, condition_id, runtime,
                             candidate=None, repairs=(), interventions=()):
    spec = validate_run_spec(run_spec)
    if not all(isinstance(value, str) and value for value in (attempt_id, condition_id)):
        raise ValueError("An attempt and canonical condition identity are required")
    bound_candidate = _public(candidate or {})
    for key in ("snapshot_sha256", "manifest_sha256"):
        if key in bound_candidate and bound_candidate[key] is not None:
            _digest(bound_candidate[key], "Candidate " + key)
    if "files" in bound_candidate:
        _closure([{**item, "path": item.get("relative_path", item.get("path"))}
                  for item in bound_candidate["files"]])
    runtime = _runtime(runtime)
    repairs = _public(list(repairs))
    _compatible_runtime(spec, runtime, repairs)
    missing = list(spec["coverage"]["missing"])
    if "wheel_sha256" not in runtime:
        missing.append("attempt_runtime_wheel")
    if not bound_candidate:
        missing.append("candidate_content")
    receipt = {"schema": "simready-bench-attempt-provenance-v1", "attempt_id": attempt_id,
               "condition_id": condition_id, "run_spec_sha256": spec["sha256"],
               "protocol_sha256": spec["protocol_sha256"], "runtime": runtime,
               "runtime_sha256": content_sha256(runtime), "candidate": bound_candidate,
               "repairs": repairs, "interventions": _public(list(interventions)),
               "coverage": {"status": "partial" if missing else "complete",
                            "missing": sorted(set(missing))}}
    receipt["sha256"] = content_sha256(receipt)
    return receipt


def _compatible_runtime(spec, runtime, repairs):
    original = spec["runtime"].get("wheel_sha256")
    replacement = runtime.get("wheel_sha256")
    allowed = {original} if original else set()
    for repair in repairs:
        before, after = repair.get("original_contract"), repair.get("replacement_contract")
        if (repair.get("schema") != "simready-bench-executor-repair-v2"
                or repair.get("protocol_equivalent") is not True
                or repair.get("protected_files_identical") is not True
                or repair.get("requires_new_run_spec") is not False
                or not before or before != after
                or before.get("sha256") != content_sha256(before.get("files"))
                or repair.get("original_contract_sha256") != before["sha256"]
                or repair.get("replacement_contract_sha256") != after["sha256"]
                or repair.get("original_sha256") not in allowed):
            raise ValueError("The attempt requires an approved byte-equivalent repair or a new run specification")
        allowed.add(_digest(repair.get("replacement_sha256"), "Repair wheel"))
    if original and replacement and replacement not in allowed:
        raise ValueError("The attempt requires an approved byte-equivalent repair or a new run specification")
    for key in ("runtime_images", "container_image_digest"):
        if key in runtime and key in spec["runtime"] and runtime[key] != spec["runtime"][key]:
            raise ValueError("Changed runtime containers require a new run specification")


def provenance_fields(row, *, run_spec=None):
    """Validate receipts before exporting them without changing measured values."""
    receipt = row.get("execution_provenance")
    if receipt is None:
        if row.get("protocol_sha256") or row.get("run_spec_sha256"):
            raise ValueError("Public provenance digests require an attempt receipt")
        return {"provenance_coverage": {"status": "legacy_unbound",
                                        "missing": ["run_spec", "attempt_receipt"]}}
    if not isinstance(receipt, dict) or receipt.get("schema") != "simready-bench-attempt-provenance-v1":
        raise ValueError("A bound attempt provenance receipt is required")
    unsigned = {key: item for key, item in receipt.items() if key != "sha256"}
    if _digest(receipt.get("sha256")) != content_sha256(unsigned):
        raise ValueError("Attempt provenance differs from its recorded digest")
    _runtime(receipt.get("runtime"))
    if receipt.get("runtime_sha256") != content_sha256(receipt["runtime"]):
        raise ValueError("The attempt runtime differs from its recorded digest")
    if receipt.get("condition_id") != row.get("condition_id"):
        raise ValueError("Provenance belongs to a different canonical condition")
    for field in ("protocol_sha256", "run_spec_sha256"):
        _digest(receipt.get(field), field)
        if row.get(field) is not None and row[field] != receipt[field]:
            raise ValueError("The public provenance digest differs from its receipt")
    if run_spec is not None:
        spec = validate_run_spec(run_spec)
        if receipt["run_spec_sha256"] != spec["sha256"] or receipt["protocol_sha256"] != spec["protocol_sha256"]:
            raise ValueError("The attempt differs from its bound run specification")
        _compatible_runtime(spec, receipt["runtime"], receipt.get("repairs", []))
    candidate = receipt.get("candidate", {})
    snapshot = row.get("snapshot_sha256")
    if snapshot and candidate.get("snapshot_sha256") and snapshot != candidate["snapshot_sha256"]:
        raise ValueError("The candidate differs from its bound snapshot")
    exported = row.get("candidate") or {}
    if exported.get("manifest_sha256") and candidate.get("manifest_sha256") and (
            exported["manifest_sha256"] != candidate["manifest_sha256"]):
        raise ValueError("The candidate differs from its bound package")
    return {"execution_provenance": deepcopy(receipt),
            "protocol_sha256": receipt["protocol_sha256"],
            "run_spec_sha256": receipt["run_spec_sha256"],
            "provenance_coverage": deepcopy(receipt["coverage"])}
