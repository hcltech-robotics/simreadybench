"""Strict experiment manifests and deterministic, content-bound condition plans."""

from __future__ import annotations
import copy
import hashlib
import itertools
import json
import math
import os
import re
import stat
import time
from decimal import Decimal
from pathlib import Path, PurePosixPath
from simready_bench.engine.native_output import (
    AUTHORING_PROTOCOL,
    NATIVE_OUTPUT_POLICY,
    authoring_seconds,
    authoring_turns,
)

SCHEMA_VERSION = "simready-bench/v1"
_ID = re.compile("[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_HASH = re.compile("[0-9a-f]{64}")
ROOTS = ("assets", "input_views", "skills", "private_manifests")
BUDGETS = {
    "agent_seconds": authoring_seconds(),
    "final_seconds": 600,
    "inference_requests": 16,
    "validation_calls": 5,
    "preview_requests": 12,
    "views_per_preview": 4,
    "preview_seconds": 90,
    "validation_seconds": 120,
    "maximum_spend_usd": 0,
    "max_output_tokens": 4096,
    "max_input_tokens": 64000,
}
LIMITS = {
    **BUDGETS,
    "inference_requests": 128,
    "maximum_matrix_runs": 125,
    "maximum_condition_spend_usd": 0,
}
SETTING_KEYS = {
    "system_prompt",
    "budgets",
    "profile",
    "generation",
    "illustrations",
    "fidelity",
    "acceptance_preset",
}


def _canonical(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("Finite JSON data required") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _object(value, keys, label):
    if type(value) is not dict or any((type(k) is not str for k in value)):
        raise ValueError(label + " must be an object")
    unknown = set(value) - set(keys)
    if unknown:
        raise ValueError(label + " unknown keys: " + ", ".join(sorted(unknown)))
    return value


def _id(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError(label + " must be a safe explicit identifier")
    return value


def _number(value, label, minimum=0, maximum=1000000000.0, integer=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < minimum
        or (value > maximum)
        or (isinstance(value, float) and (not math.isfinite(value)))
        or (integer and type(value) is not int)
    ):
        raise ValueError(label + " outside its finite bound")
    return value


def _merge(left, right):
    result = copy.deepcopy(left)
    for k, v in right.items():
        result[k] = (
            _merge(result[k], v)
            if type(v) is dict and type(result.get(k)) is dict
            else copy.deepcopy(v)
        )
    return result


def _relative(value, label):
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any((c in value for c in ("\\", ":", "$", "~", "\x00", "\n", "\r")))
    ):
        raise ValueError(label + " must be a literal relative path")
    p = PurePosixPath(value)
    if p.is_absolute() or any((x in ("", ".", "..") for x in value.split("/"))):
        raise ValueError(label + " must stay under its trusted root")
    return p


def _path(root, value, label, directory=False):
    relative = _relative(value, label)
    p = root
    for part in relative.parts:
        p = p / part
        if p.is_symlink():
            raise ValueError(label + " symlinks are forbidden")
    try:
        st = p.stat()
    except OSError as exc:
        raise ValueError(label + " file is unavailable") from exc
    p.resolve().relative_to(root)
    if directory:
        if not stat.S_ISDIR(st.st_mode):
            raise ValueError(label + " must be a directory")
    elif not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        raise ValueError(label + " requires a regular non-hardlinked file")
    return p


def _closure(root, paths, label, allow_directories=False):
    files = {}
    count = 0
    total = 0
    deadline = time.monotonic() + (30 if label == "asset" else 5)
    maximum_bytes = 1024**3 if label == "asset" else 256 * 1024 * 1024

    def add(p, depth=0):
        nonlocal count, total
        if depth > 64 or time.monotonic() > deadline:
            raise ValueError(label + " closure exceeds depth/time bound")
        count += 1
        if count > 10000:
            raise ValueError(label + " exceeds 10000 entries")
        st = p.lstat()
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(label + " symlinks are forbidden")
        p.resolve().relative_to(root)
        if stat.S_ISDIR(st.st_mode):
            if not allow_directories:
                raise ValueError(label + " directory not permitted")
            with os.scandir(p) as children:
                for child in children:
                    add(Path(child.path), depth + 1)
        elif stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
            total += st.st_size
            if total > maximum_bytes:
                raise ValueError(label + " exceeds configured closure byte bound")
            before = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or (opened.st_dev, opened.st_ino) != (st.st_dev, st.st_ino)
                ):
                    raise ValueError(label + " changed while opening")
                digest = hashlib.sha256()
                while data := stream.read(1024 * 1024):
                    if time.monotonic() > deadline:
                        raise ValueError(label + " closure exceeds time bound")
                    digest.update(data)
                after = os.fstat(stream.fileno())
            if before != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ValueError(label + " changed while hashing")
            rel = str(p.relative_to(root))
            files[rel] = {
                "path": rel,
                "absolute_path": str(p),
                "sha256": digest.hexdigest(),
                "size_bytes": st.st_size,
            }
        else:
            raise ValueError(label + " special or hardlinked files are forbidden")

    for value in paths:
        rel = _relative(value, label)
        candidate = root / rel
        for parent in [candidate, *candidate.parents]:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(label + " symlinks are forbidden")
        if not candidate.exists():
            raise ValueError(label + " file is unavailable")
        add(candidate)
    return [files[k] for k in sorted(files)]


def _closure_digest(files):
    return _digest(
        [{k: v for k, v in row.items() if k != "absolute_path"} for row in files]
    )


def _json_object(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("Duplicate JSON object key: " + k)
        result[k] = v
    return result


def validate_policy(value=None, *, base_dir):
    """Validate trusted operator settings. Credential ENV references are names, never values."""
    base = Path(base_dir).absolute()
    p = _object(
        copy.deepcopy({} if value is None else value),
        {"roots", "limits", "providers", "runtime"},
        "operator policy",
    )
    roots = _object(p.get("roots", {}), ROOTS, "policy roots")
    normalized_roots = {}
    for name in ROOTS:
        r = Path(roots.get(name, base / name))
        if not r.is_absolute():
            raise ValueError("Trusted roots must be absolute")
        if r.is_symlink() or r.resolve() != r or (not r.is_dir()):
            raise ValueError("Trusted roots must be existing non-symlink directories")
        normalized_roots[name] = str(r)
    paths = [Path(p) for p in normalized_roots.values()]
    if any(
        (
            a == b or a.is_relative_to(b) or b.is_relative_to(a)
            for i, a in enumerate(paths)
            for b in paths[i + 1 :]
        )
    ):
        raise ValueError("Trusted roots must be separate and disjoint")
    limits = _object(p.get("limits", {}), LIMITS, "policy limits")
    limits = {**LIMITS, **limits}
    if "maximum_condition_spend_usd" not in p.get("limits", {}):
        limits["maximum_condition_spend_usd"] = limits["maximum_spend_usd"]
    for k, v in limits.items():
        if k in ("max_output_tokens", "max_input_tokens") and v is None:
            continue
        _number(
            v,
            "policy." + k,
            maximum=10000 if k == "maximum_matrix_runs" else 1000000000.0,
            integer=k not in ("maximum_spend_usd", "maximum_condition_spend_usd"),
        )
    if limits["maximum_matrix_runs"] < 1:
        raise ValueError("maximum_matrix_runs must be positive")
    for k, cap in (
        ("max_output_tokens", 10000000),
        ("max_input_tokens", 262144),
        ("views_per_preview", 4),
    ):
        if limits[k] is not None and limits[k] > cap:
            raise ValueError("Immutable transport bound exceeded: " + k)
    providers = p.get("providers", {})
    if type(providers) is not dict:
        raise ValueError("Provider registry must be an object")
    budget_keys = {
        "project",
        "authorization_id",
        "approved",
        "upstream_origin",
        "model",
        "expires_at",
        "maximum_requests",
        "maximum_spend_usd",
        "pricing_status",
        "external_spend_cap_verified",
    }
    for key, provider in providers.items():
        _id(key, "provider ref")
        _object(
            provider,
            {
                "kind",
                "base_url",
                "api_key_env",
                "project_id_env",
                "project_budget",
                "model_spec",
                "hf_router_relay",
            },
            "provider",
        )
        if provider.get("kind") not in ("openai", "huggingface"):
            raise ValueError("Unsupported operator provider kind")
        for field in ("api_key_env", "project_id_env"):
            if field in provider and (
                not isinstance(provider[field], str)
                or not re.fullmatch("[A-Z_][A-Z0-9_]{0,127}", provider[field])
            ):
                raise ValueError("Credential reference must be an ENV name")
        if "base_url" in provider:
            from urllib.parse import urlsplit

            url = urlsplit(provider["base_url"])
            if (
                url.scheme not in ("https", "http")
                or not url.hostname
                or url.username
                or url.password
                or url.query
                or url.fragment
                or (
                    url.scheme == "http"
                    and url.hostname not in ("127.0.0.1", "localhost", "::1")
                )
            ):
                raise ValueError(
                    "Trusted provider endpoint must use HTTPS or fixed loopback HTTP"
                )
        if "project_budget" in provider:
            _object(provider["project_budget"], budget_keys, "provider budget")
        if "model_spec" in provider:
            from simready_bench.engine.custom_model import validate_model_spec

            provider["model_spec"] = validate_model_spec(
                provider["model_spec"], base_url=provider.get("base_url")
            )
        if "hf_router_relay" in provider:
            from simready_bench.engine.custom_model import validate_hf_router_relay

            provider["hf_router_relay"] = validate_hf_router_relay(
                provider["hf_router_relay"],
                base_url=provider.get("base_url"),
                model=provider.get("project_budget", {}).get("model"),
            )
    runtime = _object(
        p.get("runtime", {}),
        {
            "repository_root",
            "source_root",
            "validator_python",
            "expected_hostname",
            "allowed_reference_root",
            "shared_profile",
            "execution_enabled",
            "authority_state_dir",
            "container_resources",
            "illustration_image",
            "fidelity_image",
            "executor_stall_seconds",
        },
        "runtime",
    )
    if "executor_stall_seconds" in runtime:
        _number(runtime["executor_stall_seconds"], "executor stall seconds", 1800, 86400)
    if "fidelity_image" in runtime and (
        not isinstance(runtime["fidelity_image"], str)
        or not re.fullmatch("sha256:[0-9a-f]{64}", runtime["fidelity_image"])
    ):
        raise ValueError("Immutable CPU fidelity image required")
    if "illustration_image" in runtime and (
        not isinstance(runtime["illustration_image"], str)
        or not re.fullmatch("sha256:[0-9a-f]{64}", runtime["illustration_image"])
    ):
        raise ValueError("Immutable CPU illustration image required")
    if "container_resources" in runtime:
        resources = _object(
            runtime["container_resources"], {"cpus", "memory_mb"}, "container resources"
        )
        if set(resources) != {"cpus", "memory_mb"}:
            raise ValueError("Explicit CPU and memory container bounds required")
        _number(resources["cpus"], "container cpus", 1, 64, True)
        _number(resources["memory_mb"], "container memory MiB", 128, 1048576, True)
    for k in ("execution_enabled",):
        if k in runtime and type(runtime[k]) is not bool:
            raise ValueError("runtime switches must be boolean")
    if "authority_state_dir" in runtime:
        text = runtime["authority_state_dir"]
        if not isinstance(text, str) or any(
            (c in text for c in ("$", "~", "\x00", "\n", "\r"))
        ):
            raise ValueError("Literal authority directory required")
        authority = Path(text)
        if (
            not authority.is_absolute()
            or authority.resolve() != authority
            or (not authority.is_dir())
        ):
            raise ValueError(
                "Existing absolute non-symlink authority directory required"
            )
        st = authority.stat()
        if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) & 63:
            raise ValueError("Authority directory must be operator-owned and private")
        excluded = [Path(r) for r in normalized_roots.values()]
        if runtime.get("allowed_reference_root"):
            excluded.append(Path(runtime["allowed_reference_root"]).absolute())
        if any(
            (
                authority == r
                or authority.is_relative_to(r)
                or r.is_relative_to(authority)
                for r in excluded
            )
        ):
            raise ValueError(
                "Authority directory must be outside asset/input/skill/private-reference roots"
            )
        runtime["authority_state_dir"] = str(authority)
    result = {
        "roots": normalized_roots,
        "limits": limits,
        "providers": providers,
        "runtime": runtime,
    }
    _canonical(result)
    return result


def _generation(g):
    _object(g, {"max_output_tokens", "reasoning_effort", "output_limit"}, "generation")
    if "output_limit" in g and g["output_limit"] not in (NATIVE_OUTPUT_POLICY, "bounded"):
        raise ValueError("Unsupported output limit policy")
    if "max_output_tokens" in g:
        if g["max_output_tokens"] is None:
            if g.get("output_limit") != NATIVE_OUTPUT_POLICY:
                raise ValueError(
                    "Null output bound requires provider-native output policy"
                )
        else:
            if g.get("output_limit") == NATIVE_OUTPUT_POLICY:
                raise ValueError(
                    "Native output protocol cannot carry a benchmark token cap"
                )
            _number(g["max_output_tokens"], "output tokens", 16, 10000000, True)
    if g.get("output_limit") == "bounded" and g.get("max_output_tokens") is None:
        raise ValueError("Bounded output requires an explicit output token cap")
    if "reasoning_effort" in g and g["reasoning_effort"] not in (
        "none",
        "off",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        raise ValueError("Unsupported reasoning effort")
    for k, cap in (("temperature", 2), ("top_p", 1)):
        if k in g:
            _number(g[k], k, 0, cap)


def _settings(s, limits, *, native=False):
    _object(s, SETTING_KEYS, "settings")
    if "system_prompt" in s and (
        not isinstance(s["system_prompt"], str)
        or len(s["system_prompt"].encode()) > 65536
        or "\x00" in s["system_prompt"]
    ):
        raise ValueError("System prompt must be bounded literal text")
    if "profile" in s:
        _object(s["profile"], {"id"}, "profile")
        if s["profile"] not in ({"id": "NVIDIA-Robotics-Prop-4.0.0"},):
            raise ValueError("Unknown pinned benchmark profile")
    for k, v in _object(s.get("budgets", {}), BUDGETS, "budgets").items():
        if k in ("max_output_tokens", "max_input_tokens") and v is None:
            if not native or limits[k] is not None:
                raise ValueError(
                    "Native output requires explicit trusted null output limit and versioned generation policy"
                )
            continue
        maximum = (
            limits["maximum_condition_spend_usd"]
            if k == "maximum_spend_usd"
            else limits[k]
        )
        if maximum is None:
            maximum = (
                (10000000 if native else 262144) if k == "max_input_tokens" else 10000000
            )
        _number(
            v,
            k,
            minimum=0 if k == "maximum_spend_usd" else 1,
            maximum=maximum,
            integer=k != "maximum_spend_usd",
        )
    if s.get("acceptance_preset") == "calibrated-official":
        raise ValueError("The calibrated-official preset requires independent evidence")
    if "acceptance_preset" in s and s["acceptance_preset"] not in (
        "nvidia-mandatory-95-5-v1",
    ):
        raise ValueError("Unknown versioned acceptance preset")
    if "fidelity" in s:
        from simready_bench.fidelity import settings

        settings(s["fidelity"])
    if "illustrations" in s:
        from simready_bench.captures import validate_settings

        validate_settings(s["illustrations"])
    _generation(s.get("generation", {}))
    _canonical(s)
    return copy.deepcopy(s)


def _entries(value, label):
    if not isinstance(value, list) or not value or len(value) > 1000:
        raise ValueError(label + " requires 1..1000 entries")
    return value


def validate_manifest(value, *, base_dir, policy=None):
    """Return an independent normalized manifest; no execution, providers or environment reads."""
    _canonical(value)
    m = _object(
        copy.deepcopy(value),
        {
            "schema_version",
            "assets",
            "private_asset_manifest",
            "harnesses",
            "models",
            "skill_sets",
            "defaults",
            "overrides",
            "seed",
            "repetitions",
            "reconstruction_selection",
            "source",
            "authoring_protocol",
            "tool_protocol",
            "benchmark_freeze",
            "scoring",
        },
        "manifest",
    )
    if "scoring" in m:
        from simready_bench.engine.scoring import (
            validate_policy as validate_scoring_policy,
        )

        validate_scoring_policy(m["scoring"])
    if "benchmark_freeze" in m:
        from simready_bench.manifest import validate_freeze

        validate_freeze(m["benchmark_freeze"])
    protocol = m.get("authoring_protocol", AUTHORING_PROTOCOL)
    turns = authoring_turns(protocol)
    tool_protocol = m.get('tool_protocol', 'legacy')
    continue_on_length = protocol.endswith('-continue-length-v1')
    tool_identity = protocol.removesuffix('-continue-length-v1')
    if tool_protocol not in ('legacy', 'bounded-v1') or tool_identity.endswith('-tools-bounded-v1') != (tool_protocol == 'bounded-v1'):
        raise ValueError('Authoring and tool protocol must match')
    m["authoring_protocol"] = protocol
    if "source" in m:
        from simready_bench.sources import validate_source

        m["source"] = validate_source(m["source"], base_dir=base_dir)
    if m.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    op = validate_policy(policy, base_dir=base_dir)
    roots = {k: Path(v) for k, v in op["roots"].items()}
    limits = op["limits"]
    if ("assets" in m) == ("private_asset_manifest" in m):
        raise ValueError("Choose assets or private_asset_manifest")
    if "private_asset_manifest" in m:
        p = _path(
            roots["private_manifests"], m["private_asset_manifest"], "private manifest"
        )
        if p.stat().st_size > 1024 * 1024:
            raise ValueError("Private manifest exceeds 1MiB")
        loaded = _object(
            json.loads(p.read_text(), object_pairs_hook=_json_object),
            {"assets"},
            "private asset manifest",
        )
        m["assets"] = loaded["assets"]
    if isinstance(m["assets"], str):
        m["assets"] = [m["assets"]]
    m.setdefault("skill_sets", [{"id": "none", "paths": []}])
    m.setdefault("defaults", {})
    m.setdefault("overrides", [])
    m.setdefault("seed", 0)
    m.setdefault("repetitions", 4)
    if "scoring" in m:
        m["defaults"].setdefault("profile", {"id": "NVIDIA-Robotics-Prop-4.0.0"})
        m["defaults"].setdefault("acceptance_preset", "nvidia-mandatory-95-5-v1")
    cap = m["defaults"].get("generation", {}).get("max_output_tokens")
    if limits["max_input_tokens"] is not None or (
        limits["max_output_tokens"] is not None and cap is None
    ):
        raise ValueError(
            "Provider-native input requires max_input_tokens=null, and uncapped output requires max_output_tokens=null"
        )
    m["defaults"].setdefault("budgets", {}).setdefault("inference_requests", turns)
    m["defaults"]["budgets"].setdefault("max_output_tokens", cap)
    m["defaults"]["budgets"].setdefault("max_input_tokens", None)
    m["defaults"].setdefault("generation", {}).setdefault(
        "output_limit", NATIVE_OUTPUT_POLICY if cap is None else "bounded"
    )
    m["defaults"]["generation"].setdefault("max_output_tokens", None)
    _number(m["seed"], "seed", 0, 2**63 - 1, True)
    _number(m["repetitions"], "repetitions", 1, 1000, True)
    for label in ("assets", "harnesses", "models", "skill_sets"):
        _entries(m.get(label), label)
    count = (
        math.prod((len(m[k]) for k in ("assets", "harnesses", "models", "skill_sets")))
        * m["repetitions"]
    )
    for label in ("assets", "harnesses", "models", "skill_sets"):
        normalized = []
        for entry in m[label]:
            if isinstance(entry, str):
                if label == "assets":
                    entry = {"id": Path(entry).stem, "usd": entry}
                elif label == "harnesses":
                    entry = {"id": entry, "adapter": entry}
                else:
                    raise ValueError("Explicit " + label + " entry required")
            keys = {
                "assets": {
                    "id",
                    "usd",
                    "dependencies",
                    "input_views",
                    "source_view",
                    "nominal_dimensions_m",
                },
                "harnesses": {
                    "id",
                    "adapter",
                    "agent",
                    "native_agent",
                    "import_path",
                    "kwargs",
                },
                "models": {
                    "id",
                    "adapter",
                    "provider_ref",
                    "model_id",
                    "metadata",
                    "generation",
                },
                "skill_sets": {"id", "paths", "sha256"},
            }[label]
            _object(entry, keys, label)
            _id(entry.get("id"), label + " id")
            if label == "assets":
                entry.setdefault("dependencies", [])
                entry.setdefault("input_views", [])
                if not isinstance(entry["dependencies"], list) or not isinstance(
                    entry["input_views"], list
                ):
                    raise ValueError("Asset dependencies/views must be arrays")
                if len(entry["dependencies"]) > 10000 or len(entry["input_views"]) > 32:
                    raise ValueError("Asset closure exceeds limits")
                _relative(entry.get("usd"), "USD")
                if Path(entry["usd"]).suffix.lower() not in (".usd", ".usda", ".usdc"):
                    raise ValueError(
                        "Only plain USD/USDA/USDC assets are supported; USDZ is unavailable"
                    )
                if "nominal_dimensions_m" in entry:
                    if (
                        not isinstance(entry["nominal_dimensions_m"], list)
                        or len(entry["nominal_dimensions_m"]) != 3
                    ):
                        raise ValueError("Nominal dimensions require XYZ")
                    for n in entry["nominal_dimensions_m"]:
                        _number(n, "dimension", 1e-09, 1000000.0)
                entry["closure"] = _closure(
                    roots["assets"], [entry["usd"], *entry["dependencies"]], "asset"
                )
                entry["usd_path"] = str(roots["assets"] / entry["usd"])
                entry["input_view_closure"] = _closure(
                    roots["input_views"], entry["input_views"], "input view"
                )
                if "source_view" in entry:
                    _relative(entry["source_view"], "source PNG")
                    if Path(entry["source_view"]).suffix.lower() != ".png" or entry[
                        "source_view"
                    ] not in {r["path"] for r in entry["closure"]}:
                        raise ValueError(
                            "Source PNG must be part of the frozen asset closure"
                        )
            elif label == "harnesses":
                from simready_bench.capabilities import normalize_native_harness

                entry = normalize_native_harness(entry)
                if tool_protocol == 'bounded-v1' and entry.get('adapter') != 'pi':
                    raise ValueError('Bounded tools require the Pi harness')
                if continue_on_length and entry.get('adapter') != 'pi':
                    raise ValueError('Continuation after length requires the Pi harness')
                if entry.get("adapter") not in ("pi", "harbor"):
                    raise ValueError("Use pi or an installed native Harbor harness")
            elif label == "models":
                if entry.get("adapter") not in (
                    "openai-responses",
                    "openai-completions",
                ):
                    raise ValueError("Unsupported model transport")
                ref = _id(entry.get("provider_ref"), "provider ref")
                if ref not in op["providers"]:
                    raise ValueError("Provider ref missing from trusted registry")
                expected_kind = {
                    "openai-responses": "openai",
                    "openai-completions": "huggingface",
                }[entry["adapter"]]
                declared_compatible = (
                    op["providers"][ref].get("model_spec") is not None
                    and op["providers"][ref]["kind"] == "openai"
                    and (
                        entry["adapter"]
                        == op["providers"][ref]["model_spec"]["api_style"]
                    )
                )
                if (
                    op["providers"][ref]["kind"] != expected_kind
                    and (not declared_compatible)
                ):
                    raise ValueError(
                        "Model adapter incompatible with trusted provider kind"
                    )
                if (
                    not isinstance(entry.get("model_id"), str)
                    or not entry["model_id"]
                    or len(entry["model_id"]) > 256
                    or any((ord(c) < 32 for c in entry["model_id"]))
                ):
                    raise ValueError("Explicit fixed model ID required")
                metadata = entry.setdefault("metadata", {})
                _object(
                    metadata,
                    {
                        "contextWindow",
                        "maxTokens",
                        "input",
                        "reasoning",
                        "compat",
                        "cost_status",
                    },
                    "model metadata",
                )
                if not {"contextWindow", "maxTokens", "input", "reasoning"} <= set(
                    metadata
                ):
                    raise ValueError(
                        "Explicit text/image/context/output/reasoning model metadata required"
                    )
                if metadata.get("cost_status", "unknown") != "unknown":
                    raise ValueError(
                        "Manifest model metadata cannot declare provider pricing"
                    )
                metadata["cost_status"] = "unknown"
                if (
                    not isinstance(metadata["input"], list)
                    or any((type(x) is not str for x in metadata["input"]))
                    or set(metadata["input"]) != {"text", "image"}
                    or (len(metadata["input"]) != 2)
                ):
                    raise ValueError(
                        "Pi vision model must explicitly support text and image"
                    )
                for name in ("contextWindow", "maxTokens"):
                    _number(
                        metadata[name],
                        "model " + name,
                        1,
                        10000000,
                        True,
                    )
                if metadata["maxTokens"] > metadata["contextWindow"]:
                    raise ValueError("Model maxTokens exceeds contextWindow")
                if "reasoning" in metadata and type(metadata["reasoning"]) is not bool:
                    raise ValueError("reasoning metadata must be boolean")
                compat = _object(
                    metadata.get("compat", {}),
                    {
                        "supportsReasoningEffort",
                        "supportsMaxOutputTokens",
                        "maxTokensField",
                    },
                    "model compatibility",
                )
                for k in ("supportsReasoningEffort", "supportsMaxOutputTokens"):
                    if k in compat and type(compat[k]) is not bool:
                        raise ValueError("Compatibility switch must be boolean")
                if "maxTokensField" in compat and compat["maxTokensField"] not in (
                    "max_tokens",
                    "max_completion_tokens",
                ):
                    raise ValueError("Unsupported maxTokensField")
                if (
                    entry["adapter"] == "openai-responses"
                    and "maxTokensField" in compat
                ):
                    raise ValueError("Chat token field incompatible with Responses")
                entry.setdefault("generation", {})
                _generation(entry["generation"])
            else:
                entry.setdefault("paths", [])
                if not isinstance(entry["paths"], list) or len(entry["paths"]) > 1000:
                    raise ValueError("Skill paths must be bounded array")
                entry["closure"] = _closure(
                    roots["skills"], entry["paths"], "skill", allow_directories=True
                )
                digest = _closure_digest(entry["closure"])
                if "sha256" in entry and entry["sha256"] != digest:
                    raise ValueError("Skill closure hash mismatch")
                entry["sha256"] = digest
            normalized.append(entry)
        if len({e["id"] for e in normalized}) != len(normalized):
            raise ValueError("Duplicate " + label + " IDs")
        m[label] = sorted(normalized, key=lambda e: e["id"])
    if "reconstruction_selection" in m:
        selection = m["reconstruction_selection"]
        asset_ids = {asset["id"] for asset in m["assets"]}
        if not isinstance(selection, dict) or set(selection) != asset_ids:
            raise ValueError("Reconstruction selection must name every reference asset")
        for repetitions in selection.values():
            if (
                not isinstance(repetitions, list)
                or any(
                    type(value) is not int or not 0 <= value < m["repetitions"]
                    for value in repetitions
                )
                or repetitions != sorted(set(repetitions))
            ):
                raise ValueError(
                    "Selected repetitions must be unique ordered repetition indices"
                )
        count = sum(map(len, selection.values())) * math.prod(
            len(m[k]) for k in ("harnesses", "models", "skill_sets")
        )
    if not count or count > limits["maximum_matrix_runs"]:
        raise ValueError("Experiment matrix exceeds operator maximum_matrix_runs")
    m["defaults"] = _settings(m["defaults"], limits, native=True)
    if not isinstance(m["overrides"], list) or len(m["overrides"]) > 1000:
        raise ValueError("Overrides require bounded ordered array")
    dimensions = {
        "asset": "assets",
        "harness": "harnesses",
        "model": "models",
        "skill_set": "skill_sets",
    }
    for override in m["overrides"]:
        _object(override, {"match", "settings"}, "override")
        match = _object(override.get("match", {}), dimensions, "override match")
        for k, v in match.items():
            if v != "*" and v not in {e["id"] for e in m[dimensions[k]]}:
                raise ValueError("Override selects an unknown " + k)
        override["settings"] = _settings(
            override.get("settings", {}), limits, native=True
        )
    settings = [m["defaults"], *(o["settings"] for o in m["overrides"])]
    for s in settings:
        b = s.get("budgets", {})
        if b.get("max_input_tokens") is not None:
            raise ValueError(
                "Provider-native input cannot carry a benchmark input token cap"
            )
        if b.get("inference_requests", turns) > turns:
            raise ValueError(
                "Authoring protocol permits at most " + str(turns) + " inference requests"
            )
    if "benchmark_freeze" in m:
        from simready_bench.manifest import verify_freeze

        verify_freeze(m["benchmark_freeze"], m["assets"])
    m.pop("private_asset_manifest", None)
    return m


def condition_identity(ids, *, repetition, master_seed, authoring_protocol=AUTHORING_PROTOCOL):
    """Identify an original matrix repetition independently of execution hosts."""
    identity = {
        **ids,
        "repetition": repetition,
        "master_seed": master_seed,
        "authoring_protocol": authoring_protocol,
    }
    seed = int(_digest(identity)[:16], 16) % 2**31
    return "condition-" + _digest({**identity, "seed": seed}), seed


def resolve_experiment(value, *, base_dir, policy=None):
    m = validate_manifest(value, base_dir=base_dir, policy=policy)
    op = validate_policy(policy, base_dir=base_dir)
    defaults = {
        "system_prompt": "",
        "budgets": {**BUDGETS, "maximum_spend_usd": 0},
        "profile": {"id": "NVIDIA-Robotics-Prop-4.0.0"},
        "acceptance_preset": "nvidia-mandatory-95-5-v1",
    }
    for k in defaults["budgets"]:
        upper = (
            op["limits"]["maximum_condition_spend_usd"]
            if k == "maximum_spend_usd"
            else op["limits"][k]
        )
        if upper is not None:
            defaults["budgets"][k] = min(defaults["budgets"][k], upper)
    turns = authoring_turns(m["authoring_protocol"])
    defaults["budgets"].update(
        inference_requests=min(turns, op["limits"]["inference_requests"]),
        max_output_tokens=None,
        max_input_tokens=None,
    )
    defaults["generation"] = {
        "output_limit": NATIVE_OUTPUT_POLICY,
        "max_output_tokens": None,
    }
    conditions = []
    for asset, harness, model, skill in itertools.product(
        *(m[k] for k in ("assets", "harnesses", "models", "skill_sets"))
    ):
        ids = {
            "asset": asset["id"],
            "harness": harness["id"],
            "model": model["id"],
            "skill_set": skill["id"],
        }
        settings = _merge(defaults, {"generation": model.get("generation", {})})
        settings = _merge(settings, m["defaults"])
        for override in m["overrides"]:
            if all((v == "*" or ids[k] == v for k, v in override["match"].items())):
                settings = _merge(settings, override["settings"])
        _settings(settings, op["limits"], native=True)
        if harness["adapter"] not in ("pi", "harbor"):
            raise ValueError("Native output requires an admitted native harness")
        from simready_bench.engine.request_cost import native_output_facts

        declared = op["providers"][model["provider_ref"]].get("model_spec")
        if declared is not None:
            from simready_bench.engine.custom_model import (
                validate_model_spec,
                facts as declared_facts,
            )

            validate_model_spec(
                declared, model=model["model_id"], api_style=model["adapter"]
            )
            facts = declared_facts(declared)
        else:
            facts = native_output_facts(model["model_id"])
        facts["protocol"] = m["authoring_protocol"]
        model = copy.deepcopy(model)
        model["native_output_facts"] = facts
        cap = settings["generation"].get("max_output_tokens")
        from simready_bench.engine.native_output import authoring_protocol

        if m["authoring_protocol"] != authoring_protocol(turns, cap, tool_protocol=m.get('tool_protocol', 'legacy'), continue_on_length=m['authoring_protocol'].endswith('-continue-length-v1')):
            raise ValueError("Authoring protocol and output token cap must match")
        if settings["budgets"]["max_output_tokens"] != cap:
            raise ValueError("Generation and budget output token caps must match")
        if cap is not None and cap > facts["output_token_upper_bound"]:
            raise ValueError("Output token cap exceeds the declared provider maximum")
        model["metadata"]["maxTokens"] = cap or facts["output_token_upper_bound"]
        model["metadata"]["contextWindow"] = facts["context_window"]
        settings["budgets"]["max_input_tokens"] = facts["maximum_input_tokens"]
        model["metadata"].setdefault("compat", {})["supportsMaxOutputTokens"] = cap is not None
        if cap is not None and model["adapter"] == "openai-completions":
            connection = op["providers"][model["provider_ref"]]
            model["metadata"]["compat"]["maxTokensField"] = (
                "max_tokens"
                if connection["kind"] == "huggingface" or connection.get("hf_router_relay")
                else "max_completion_tokens"
            )
        effort = settings["generation"].get("reasoning_effort")
        if effort not in (None, "none", "off") and (
            not model.get("metadata", {}).get("reasoning")
            or model.get("metadata", {})
            .get("compat", {})
            .get("supportsReasoningEffort")
            is False
        ):
            raise ValueError(
                "Requested reasoning is unsupported by configured model/transport"
            )
        for repetition in range(m["repetitions"]):
            if repetition not in m.get("reconstruction_selection", {}).get(
                asset["id"], range(m["repetitions"])
            ):
                continue
            identifier, seed = condition_identity(
                ids, repetition=repetition, master_seed=m["seed"],
                authoring_protocol=m["authoring_protocol"],
            )
            c = {
                "condition_id": identifier,
                "asset_id": asset["id"],
                "harness_id": harness["id"],
                "model_id": model["id"],
                "skill_set_id": skill["id"],
                "asset": copy.deepcopy(asset),
                "harness": copy.deepcopy(harness),
                "model": copy.deepcopy(model),
                "skill_set": copy.deepcopy(skill),
                "settings": copy.deepcopy(settings),
                "repetition": repetition,
                "seed": seed,
            }
            c["authoring_protocol"] = m["authoring_protocol"]
            if 'tool_protocol' in m:
                c['tool_protocol'] = m['tool_protocol']
            c["configuration_sha256"] = _digest(c)
            conditions.append(c)
    allocated = sum(
        (
            Decimal(str(c["settings"]["budgets"]["maximum_spend_usd"]))
            for c in conditions
        ),
        Decimal(0),
    )
    if allocated > Decimal(str(op["limits"]["maximum_spend_usd"])):
        raise ValueError("aggregate condition spend exceeds operator maximum_spend_usd")
    result = {
        "schema_version": SCHEMA_VERSION,
        "normalized_manifest": m,
        "operator_policy": op,
        "policy_sha256": _digest(op),
        "conditions": conditions,
        "aggregate_budget": {
            "maximum_spend_usd": op["limits"]["maximum_spend_usd"],
            "allocated_spend_usd": float(allocated),
            "maximum_matrix_runs": op["limits"]["maximum_matrix_runs"],
        },
    }
    result["configuration_sha256"] = _digest(result)
    return result
