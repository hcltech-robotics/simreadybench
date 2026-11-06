"""Explicit profile selections and portable, experiment-frozen inventories.

The historical Robotics-Prop policy remains byte-for-byte version 1. Other
policies carry the installed inventory so coordinators need no NVIDIA imports;
the validator worker verifies that inventory against its actual installation.
"""

from copy import deepcopy
import re

DEFAULT_PROFILE = {"id": "Robotics-Prop", "version": "4.0.0"}
LEGACY_SELECTION = {"id": "NVIDIA-Robotics-Prop-4.0.0"}


def selection(value=None):
    if value is None or value == LEGACY_SELECTION:
        return dict(DEFAULT_PROFILE)
    if (
        not isinstance(value, dict)
        or set(value) != {"id", "version"}
        or not isinstance(value["id"], str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,99}", value["id"])
        or not isinstance(value["version"], str)
        or not re.fullmatch(r"\d+\.\d+\.\d+", value["version"])
    ):
        raise ValueError("Explicit NVIDIA profile id and exact version required")
    return dict(value)


def inventories(policy=None):
    if policy and policy.get("schema") == "simready-validator-scoring-policy-v2":
        mandatory = deepcopy(policy["profile_inventory"])
        optional = deepcopy(policy["optional_inventory"])
        validate_inventories(mandatory, optional)
        return mandatory, optional
    from simready_bench.engine.verifier import pinned_profile
    from simready_bench.engine.scoring import pinned_optional_profile
    return pinned_profile(), pinned_optional_profile()


def validate_inventories(mandatory, optional):
    from simready_bench.engine.verifier import digest
    identity = selection({
        "id": mandatory["profile_id"], "version": mandatory["profile_version"]
    })
    if identity == DEFAULT_PROFILE:
        raise ValueError("The default profile must use its frozen v1 inventory")
    if any(optional.get(key) != mandatory.get(key)
           for key in ("profile_id", "profile_version")):
        raise ValueError("Mandatory and optional profile identities differ")
    codes = set()
    for inventory, kind in ((mandatory, "mandatory"), (optional, "optional")):
        roster = inventory[kind + "_requirements"]
        if (
            not isinstance(roster, list)
            or len(roster) > 10000
            or type(inventory[kind + "_requirement_count"]) is not int
            or inventory[kind + "_requirement_count"] != len(roster)
            or (kind == "mandatory" and not roster)
        ):
            raise ValueError("A bounded nonempty mandatory inventory is required")
        for row in roster:
            if (
                not isinstance(row, dict)
                or set(row) != {"code", "version", "rule"}
                or not all(isinstance(row[k], str) and row[k] for k in row)
                or row["code"] in codes
                or "." not in row["rule"]
            ):
                raise ValueError("Unique requirement IDs with installed rules required")
            codes.add(row["code"])
            if row["rule"].rsplit(".", 1)[0] not in inventory["rule_module_sha256"]:
                raise ValueError("Requirement implementation hash missing")
        for hashes in (
            inventory["rule_module_sha256"],
            mandatory["SDK_dispatch_source_sha256"],
            mandatory["tier_manifest_hashes"],
        ):
            if not isinstance(hashes, dict) or any(
                not isinstance(v, str) or not re.fullmatch("[a-f0-9]{64}", v)
                for v in hashes.values()
            ):
                raise ValueError("SHA256 implementation and manifest identities required")
    if optional["mandatory_inventory_sha256"] != digest(mandatory):
        raise ValueError("Optional inventory is not bound to mandatory inventory")


def installed_policy(profile):
    """Freeze only a successfully resolved installed NVIDIA profile and its rules."""
    from simready_bench.engine.validator import derive_inventory

    return _policy_from_inventory(derive_inventory(profile=selection(profile)))


def _policy_from_inventory(inventory):
    from simready_bench.engine.validator import _sdk_source_identity
    from simready_bench.engine.verifier import scoring_policy, digest
    import hashlib
    import importlib
    import inspect
    from pathlib import Path

    profile = selection({
        "id": inventory["profile_id"], "version": inventory["profile_version"]
    })
    if profile == DEFAULT_PROFILE:
        from simready_bench.engine.scoring import verify_validator_installation
        verify_validator_installation(inventory)
        return scoring_policy()
    if inventory["missing_rule_implementations"] or inventory["optional_missing_rule_implementations"]:
        raise ValueError("Selected profile has unavailable rule implementations")
    mandatory = {
        k: deepcopy(v) for k, v in inventory.items()
        if k not in ("profile_manifest_path", "effective_inventory_sha256")
        and not k.startswith("optional_")
    }
    mandatory["SDK_dispatch_source_sha256"] = _sdk_source_identity()

    def hashes(rows):
        result = {}
        for name in sorted({r["rule"].rsplit(".", 1)[0] for r in rows}):
            path = Path(inspect.getfile(importlib.import_module(name)))
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    mandatory["rule_module_sha256"] = hashes(mandatory["mandatory_requirements"])
    optional = {k: deepcopy(v) for k, v in inventory.items() if k.startswith("optional_")}
    optional.update(
        profile_id=profile["id"], profile_version=profile["version"],
        mandatory_inventory_sha256=digest(mandatory),
        rule_module_sha256=hashes(optional["optional_requirements"]),
    )
    validate_inventories(mandatory, optional)
    return scoring_policy(inventory=mandatory, optional_inventory=optional)


def installed_profiles():
    """Enumerate registered profiles, including explicit unsupported reasons."""
    from simready_bench.engine.validator import _LOCK, _dependencies, _inventory
    from usd_validation_nvidia import ProfileRegistry
    result = []
    with _LOCK:
        api, plugin = _dependencies()
        try:
            api.initialize([], [], [])
            for registered in sorted(ProfileRegistry().profiles, key=lambda p: (p.id, str(p.version))):
                identity = {"id": registered.id, "version": str(registered.version)}
                row = {**identity, "supported": False}
                try:
                    inventory = _inventory(api, plugin, profile=identity)
                    row.update(mandatory_count=inventory["mandatory_requirement_count"],
                               optional_count=inventory["optional_requirement_count"])
                    _policy_from_inventory(inventory)
                    row["supported"] = True
                except (ValueError, RuntimeError, KeyError) as exc:
                    row["reason"] = str(exc)
                result.append(row)
        finally:
            api.destroy()
    return result
