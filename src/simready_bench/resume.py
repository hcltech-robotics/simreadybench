"""Continue an interrupted benchmark within its original allowance."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import sqlite3
import stat
import time
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from simready_bench.parallel import coordinated_resume, validate_workers


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _read(path, maximum=16 * 1024**2):
    path = Path(path).absolute()
    if (
        path.resolve() != path
        or not path.is_file()
        or path.stat().st_nlink != 1
        or path.stat().st_size > maximum
    ):
        raise ValueError("A regular saved benchmark file is required")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    Path(path).chmod(0o600)


def _tree(path):
    path = Path(path).absolute()
    if path.resolve() != path or not path.exists():
        raise ValueError("Saved benchmark inputs must be regular files and directories")
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    result = {}
    for item in files:
        if item.resolve() != item or not (item.is_file() or item.is_dir()):
            raise ValueError("Saved benchmark inputs cannot contain links")
        if item.is_file():
            if item.stat().st_nlink != 1:
                raise ValueError("Saved benchmark inputs cannot contain linked files")
            digest = hashlib.sha256()
            with item.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[
                item.name if path.is_file() else item.relative_to(path).as_posix()
            ] = digest.hexdigest()
    return result


def _reference(reference):
    from simready_bench.benchmark import reference_manifest

    manifest, _, _ = reference_manifest(reference)
    result = {"manifest_sha256": _digest(manifest)}
    if str(reference) not in ("NV25", "nvidia-25"):
        source = Path(reference).absolute()
        result.update(
            path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest()
        )
    return result


def save_run_configuration(
    output_dir,
    *,
    reference,
    deploy,
    manifest,
    policy,
    runtime_images,
    cpu_snapshot=None,
    skills=None,
):
    """Save the configuration and staged input bindings of a new benchmark."""
    from simready_bench.runtime_images import CPU_SNAPSHOT

    output = Path(os.path.abspath(output_dir))
    value = {
        "schema": "simready-bench-run-v1",
        "deployment": deploy,
        "reference": _reference(reference),
        "experiment_sha256": _digest(manifest),
        "operator_sha256": _digest(policy),
        "runtime_images": runtime_images,
        "snapshot": (
            (cpu_snapshot or runtime_images.get("snapshot") or CPU_SNAPSHOT)
            if deploy == "daytona"
            else None
        ),
        "inputs": {
            name: _tree(output / name)
            for name in ("assets", "skills", "input_views", "private_manifests")
        },
        "skill_inputs": [_tree(path) for path in skills or []],
    }
    _write(output / "run.json", value)
    return value


def condition_plan(manifest):
    from simready_bench.config import AUTHORING_PROTOCOL, condition_identity

    result = {}
    for asset, harness, model, skills in itertools.product(
        *(manifest[name] for name in ("assets", "harnesses", "models", "skill_sets"))
    ):
        ids = {
            "asset": asset["id"],
            "harness": harness["id"],
            "model": model["id"],
            "skill_set": skills["id"],
        }
        for repetition in manifest.get("reconstruction_selection", {}).get(
            asset["id"], range(manifest["repetitions"])
        ):
            identifier, seed = condition_identity(
                ids,
                repetition=repetition,
                master_seed=manifest["seed"],
                authoring_protocol=manifest.get(
                    "authoring_protocol", AUTHORING_PROTOCOL
                ),
            )
            result[identifier] = {
                "asset_id": asset["id"],
                "repetition": repetition,
                "seed": seed,
                "harness_id": harness["id"],
                "model_id": model["id"],
                "skill_set_id": skills["id"],
            }
    return result


def _reports(batch):
    from simready_bench.reporting import read_generated_report

    roots = (batch / "attempts", batch / "daytona-executor/output/attempts")
    attempts = next((sorted(root.iterdir()) for root in roots if root.is_dir()), [])
    if (
        not attempts
        and (batch / "results.json").is_file()
        and not any(batch.glob("parallel-plan-*.json"))
    ):
        attempts = [batch]
    result = []
    for attempt in attempts:
        path = attempt / "results.json"
        if not path.is_file():
            from simready_bench.recovery import salvage_draft

            recovered = salvage_draft(batch, attempt)
            if recovered is None:
                continue
            attempt, path = recovered, recovered / "results.json"
        report = read_generated_report(path)
        result.append(
            (
                attempt,
                {
                    "results": str(path),
                    "html": str(attempt / "report.html"),
                    "scores": str(attempt / "scores.csv"),
                },
                report,
            )
        )
    return result


def _batches(output):
    result = [output]
    executions = output / "executions"
    if executions.is_dir():
        for child in sorted(executions.iterdir()):
            if (
                child.resolve() != child
                or not child.is_dir()
                or not child.name.isdecimal()
            ):
                raise ValueError("Saved execution directories must have numeric names")
            result.append(child)
    return result


def _file_receipt(root, receipt):
    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
        return False
    path = root / receipt["path"]
    if path.resolve() != path or not path.is_relative_to(root) or not path.is_file():
        return False
    return (
        not receipt.get("sha256")
        or hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    )


def _canonical_invalid_score(row):
    """Recognise the pinned measured zero for an attempted invalid package."""
    from simready_bench.engine.scoring import score_validation
    from simready_bench.engine.verifier import scoring_policy
    from simready_bench.reporting import compact_report

    if (
        not isinstance(row.get("trial_id"), str)
        or not row["trial_id"]
        or type(row.get("inference_calls_used")) is not int
        or row["inference_calls_used"] <= 0
    ):
        return False
    primary = row.get("simready")
    expected = compact_report(
        {
            "schema": "simready-bench-portable-report-v1",
            "results": [
                {
                    "simready": score_validation(
                        None, scoring_policy(), candidate_invalid=True
                    )
                }
            ],
        }
    )["results"][0]["simready"]
    return (
        row.get("simready_reward") == 0
        and primary == expected
        and all(
            type(primary.get(key)) is int
            for key in (
                "passed_count",
                "failed_count",
                "unknown_count",
                "applicable_count",
                "profile_mandatory_count",
                "completion_bonus",
            )
        )
        and all(
            type(primary.get(key)) in (int, float)
            for key in (
                "score_percent",
                "score_fraction",
                "score_lower_bound",
                "score_upper_bound",
                "base_credit_percent",
                "profile_coverage_fraction",
            )
        )
    )


def _invalid_without_submission(row):
    return (
        row.get("candidate")
        == {"status": "unavailable", "reason": "no_frozen_authored_submission"}
        and row.get("snapshot_sha256") is None
        and row.get("nvidia_validator_report_file") is None
        and _canonical_invalid_score(row)
    )


def _completed(attempt, row):
    score = row.get("simready_reward")
    if (
        row.get("status") != "completed"
        or type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 100
    ):
        return False
    for kind in ("images", "traces"):
        receipts = row.get(kind, [])
        if not isinstance(receipts, list) or any(
            not _file_receipt(attempt / "evidence", receipt) for receipt in receipts
        ):
            return False
    if _invalid_without_submission(row):
        return True
    candidate = row.get("candidate") or {}
    if candidate.get("status") != "available" or candidate.get(
        "snapshot_sha256"
    ) != row.get("snapshot_sha256"):
        return False
    validator = row.get("nvidia_validator_report_file")
    if validator is None:
        members = candidate.get("files")
        if (
            not _canonical_invalid_score(row)
            or not isinstance(members, list)
            or not members
            or any(not isinstance(member, dict) for member in members)
        ):
            return False
        hashes = [
            row.get("snapshot_sha256"), candidate.get("manifest_sha256"),
            *[member.get("sha256") for member in members],
        ]
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            return False
    elif not _file_receipt(attempt, validator):
        return False
    for key in ("entrypoint", "manifest"):
        if not isinstance(candidate.get(key), str) or not _file_receipt(
            attempt, {"path": candidate[key]}
        ):
            return False
    if not _file_receipt(
        attempt,
        {"path": candidate["manifest"], "sha256": candidate.get("manifest_sha256")},
    ):
        return False
    frozen = _read(attempt / candidate["manifest"], maximum=4 * 1024**2)
    if frozen.get("snapshot_sha256") != row.get("snapshot_sha256"):
        return False
    return all(_file_receipt(attempt, member) for member in candidate.get("files", []))


def _unrun(row):
    candidate = row.get("candidate") or {}
    return (
        row.get("status") == "blocked"
        and row.get("trial_id") is None
        and row.get("inference_calls_used") in (None, 0)
        and not row.get("model_requests")
        and row.get("validation_calls_used") in (None, 0)
        and row.get("simready") is None
        and row.get("snapshot_sha256") is None
        and candidate.get("status") != "available"
        and not candidate.get("files")
        and not candidate.get("entrypoint")
    )


def _daytona_client():
    from daytona import Daytona

    return Daytona()


def _executor_terminal_proof(batch, delivery):
    expected = delivery.get("executor_terminal_proof_sha256")
    if expected is None:
        return None
    path = batch / "executor-terminal-proof.json"
    health = delivery.get("health_policy") or {}
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise ValueError("The trusted executor terminal proof is missing") from None
    if (
        path.resolve() != path
        or path.parent != batch
        or not stat.S_ISREG(details.st_mode)
        or details.st_nlink != 1
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) & 0o077
        or details.st_size > 16 * 1024
        or not isinstance(expected, str)
        or len(expected) != 64
        or health.get("schema") != "simready-bench-executor-health-v1"
        or delivery.get("terminal_confirmed") is not True
    ):
        raise ValueError("A private bound executor terminal proof is required")
    with path.open("rb") as stream:
        raw = stream.read(16 * 1024 + 1)
    if len(raw) > 16 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("The executor terminal proof hash changed")
    proof = json.loads(raw)
    if (
        not isinstance(proof, dict)
        or set(proof) != {
            "schema", "sandbox_id", "run_id", "observed_at", "controller_exit",
            "reason", "all_containers_stopped",
        }
        or proof.get("schema") != "simready-bench-executor-terminal-v1"
        or not isinstance(proof.get("sandbox_id"), str)
        or not proof["sandbox_id"]
        or proof["sandbox_id"] != delivery.get("sandbox_id")
        or not isinstance(proof.get("run_id"), str)
        or not proof["run_id"]
        or proof["run_id"] != delivery.get("run_id")
        or type(proof.get("observed_at")) not in (int, float)
        or not math.isfinite(proof["observed_at"])
        or not 0 < proof["observed_at"] <= time.time() + 60
        or type(proof.get("controller_exit")) is not int
        or proof["controller_exit"] != delivery.get("controller_exit")
        or proof.get("reason") not in {
            "executor_stalled", "executor_deadline_exceeded", "controller_exited",
            "inference_transport_circuit_open",
        }
        or proof["reason"] != delivery.get("termination_reason")
        or proof.get("all_containers_stopped") is not True
    ):
        raise ValueError("The executor terminal proof differs from its delivery")
    return proof


def _retire(batch, reports, *, policy, manifest, client):
    from daytona.common.errors import DaytonaNotFoundError
    from simready_bench.local import ParentAuthorityLedger
    from simready_bench.engine.inference_gateway import ProjectBudget

    delivery_path = batch / "daytona-delivery.json"
    delivery = _read(delivery_path)
    missing = delivery.get("missing_executor_proof") or {}
    health = delivery.get("health_policy") or {}
    lost = (
        delivery.get("collection") == "lost"
        and delivery.get("terminal_confirmed") is True
        and delivery.get("cleanup_confirmed") is True
        and health.get("schema") == "simready-bench-executor-health-v1"
        and type(health.get("orphan_stop_minutes")) is int
        and health["orphan_stop_minutes"] > 0
        and missing.get("schema") == "simready-bench-daytona-not-found-v1"
        and missing.get("sandbox_id") == delivery.get("sandbox_id")
        and type(missing.get("observed_at")) in (int, float)
        and math.isfinite(missing["observed_at"])
        and missing["observed_at"] > 0
    )
    cold = (
        delivery.get("phase") == "cold_bootstrap_retired"
        and delivery.get("collection") == "not_started"
        and delivery.get("controller_started") is False
        and delivery.get("terminal_confirmed") is True
        and delivery.get("cleanup_confirmed") is True
        and (batch / "retirement.json").is_file()
    )
    if (delivery.get("collection") != "completed" and not cold and not lost) or not delivery.get(
        "sandbox_id"
    ):
        raise ValueError("Collect the previous executor before resuming")
    terminal_path = batch / "daytona-executor/controller-status.json"
    terminal = (
        _read(terminal_path) if terminal_path.is_file() else {}
    ) if not cold and not lost else {"status": "failed"}
    external_terminal = None
    if terminal.get("status") not in ("completed", "failed", "incomplete"):
        external_terminal = _executor_terminal_proof(batch, delivery)
        if external_terminal is None:
            raise ValueError("The previous executor has not reached a terminal state")
    try:
        client.get(delivery["sandbox_id"])
    except DaytonaNotFoundError:
        pass
    else:
        raise ValueError("The previous Daytona executor still exists")
    allocation = delivery["parent_allocation"]["allocation_id"]
    authority = ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    parent = policy["providers"][manifest["models"][0]["provider_ref"]][
        "project_budget"
    ]
    with authority._db() as db:
        held = db.execute(
            "SELECT parent,requests,usd,work_root FROM allocations WHERE id=?",
            (allocation,),
        ).fetchone()
        fingerprint = db.execute(
            "SELECT fingerprint FROM parents WHERE id=?", (parent["authorization_id"],)
        ).fetchone()
    if (
        not held
        or held[0] != parent["authorization_id"]
        or held[3] != str(batch)
        or held[1] != delivery["planned_requests"]
        or Decimal(held[2]) != Decimal(str(delivery["maximum_spend_usd"]))
        or not fingerprint
        or fingerprint[0] != _digest(asdict(ProjectBudget(**parent)))
    ):
        raise ValueError(
            "The saved parent allocation differs from the original allowance"
        )
    proof_path = batch / "retirement.json"
    if proof_path.exists():
        proof = _read(proof_path)
        with authority._db() as db:
            settled = db.execute(
                "SELECT requests,usd,proof_sha256 FROM retirements WHERE allocation=?",
                (allocation,),
            ).fetchone()
        if (
            not settled
            or settled[0] != proof.get("requests")
            or Decimal(settled[1]) != Decimal(str(proof.get("maximum_spend_usd")))
            or settled[2] != _digest(proof)
            or proof.get("sandbox_id") != delivery["sandbox_id"]
            or proof.get("executor_deleted") is not True
        ):
            raise ValueError("A saved retirement differs from its immutable authority")
        return proof
    if lost:
        admitted = sorted(condition_plan(_read(batch / "experiment.json")))
        proof = {
            "sandbox_id": delivery["sandbox_id"],
            "executor_deleted": True,
            "delivery_sha256": hashlib.sha256(delivery_path.read_bytes()).hexdigest(),
            "terminal_sha256": None,
            "authority_sha256": None,
            "reports": {
                str(Path(report["results"]).relative_to(batch)): hashlib.sha256(
                    Path(report["results"]).read_bytes()
                ).hexdigest()
                for _, report, _ in reports
            },
            "requests": held[1],
            "maximum_spend_usd": held[2],
            "admitted_conditions": admitted,
            "unreported_admitted_conditions": admitted,
            "missing_executor_proof": missing,
            "usage_status": "unavailable_full_original_ceiling_charged",
        }
        authority.retire(
            allocation_id=allocation,
            requests=held[1],
            maximum_spend_usd=held[2],
            proof_sha256=_digest(proof),
        )
        _write(proof_path, proof)
        return proof
    ledger_path = batch / "daytona-executor/authority/parent-allowances.sqlite3"
    if (
        ledger_path.resolve() != ledger_path
        or not ledger_path.is_file()
        or ledger_path.stat().st_nlink != 1
    ):
        raise ValueError("The collected executor authority ledger is required")
    with sqlite3.connect(ledger_path.as_uri() + "?mode=ro", uri=True) as db:
        parents = db.execute(
            "SELECT id,maximum_requests,maximum_usd,expires FROM parents"
        ).fetchall()
        issued = db.execute(
            "SELECT id,parent,requests,usd,work_root FROM allocations"
        ).fetchall()
    expected_parent = "cloud-" + allocation
    if any(
        item[0] != expected_parent
        or item[1] != held[1]
        or Decimal(item[2]) != Decimal(held[2])
        or item[3] != parent["expires_at"]
        for item in parents
    ):
        raise ValueError(
            "The collected executor allowance differs from its original parent"
        )
    indexed, issued_conditions = {}, {}
    for _, _, report in reports:
        configuration = report.get("configuration_sha256")
        if not isinstance(configuration, str) or len(configuration) != 64:
            raise ValueError(
                "Collected condition reports must retain their configuration identity"
            )
        for row in report["results"]:
            for entry in issued:
                identity = _digest(
                    {
                        "work_root": entry[4],
                        "configuration": configuration,
                        "condition": row["condition_id"],
                    }
                )
                if identity == entry[0]:
                    if identity in indexed:
                        raise ValueError(
                            "An issued condition appears in more than one report"
                        )
                    indexed[identity] = row
                    issued_conditions[identity] = row["condition_id"]
    plans = {}
    for directory in (
        batch / "attempts",
        batch / "daytona-executor/output/attempts",
    ):
        for path in sorted(directory.glob("*/admission-plan.json")):
            plan = _read(path)
            if (
                plan.get("schema") != "simready-bench-admission-plan-v1"
                or not isinstance(plan.get("configuration_sha256"), str)
                or len(plan["configuration_sha256"]) != 64
                or not isinstance(plan.get("work_dir"), str)
                or not isinstance(plan.get("conditions"), list)
                or len(plan["conditions"]) != len(set(plan["conditions"]))
                or not set(plan["conditions"]) <= condition_plan(manifest).keys()
            ):
                raise ValueError(
                    "A collected admission plan changes its original matrix"
                )
            plans[str(path.relative_to(batch))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for condition in plan["conditions"]:
                identity = _digest(
                    {
                        "work_root": plan["work_dir"],
                        "configuration": plan["configuration_sha256"],
                        "condition": condition,
                    }
                )
                if (
                    identity in issued_conditions
                    and issued_conditions[identity] != condition
                ):
                    raise ValueError(
                        "A child allocation changes its admitted condition"
                    )
                issued_conditions[identity] = condition
    charged_requests, charged_usd = 0, Decimal(0)
    admitted, unreported = set(), set()
    for identifier, scope, requests, usd, _ in issued:
        if scope != expected_parent or identifier not in issued_conditions:
            raise ValueError(
                "A collected executor allocation has no matching condition result"
            )
        if requests != manifest["defaults"]["budgets"]["inference_requests"] or Decimal(
            usd
        ) != Decimal(str(manifest["defaults"]["budgets"]["maximum_spend_usd"])):
            raise ValueError(
                "An issued condition ceiling differs from the original benchmark"
            )
        if identifier not in indexed or not _unrun(indexed[identifier]):
            charged_requests += requests
            charged_usd += Decimal(usd)
            admitted.add(issued_conditions[identifier])
            if identifier not in indexed:
                unreported.add(issued_conditions[identifier])
    if sum(entry[2] for entry in issued) > held[1] or sum(
        (Decimal(entry[3]) for entry in issued), Decimal(0)
    ) > Decimal(held[2]):
        raise ValueError("Collected condition allocations exceed their parent")
    if external_terminal is not None and not issued:
        charged_requests, charged_usd = held[1], Decimal(held[2])
        admitted = set(condition_plan(_read(batch / "experiment.json")))
        unreported = set(admitted)
    proof = {
        "sandbox_id": delivery["sandbox_id"],
        "executor_deleted": True,
        "delivery_sha256": hashlib.sha256(delivery_path.read_bytes()).hexdigest(),
        "terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "authority_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "reports": {
            str(Path(report["results"]).relative_to(batch)): hashlib.sha256(
                Path(report["results"]).read_bytes()
            ).hexdigest()
            for _, report, _ in reports
        },
        "requests": charged_requests,
        "maximum_spend_usd": str(charged_usd),
        "admitted_conditions": sorted(admitted),
        "unreported_admitted_conditions": sorted(unreported),
        "admission_plans": plans,
    }
    if external_terminal is not None:
        proof["executor_terminal_proof"] = external_terminal
        proof["executor_terminal_proof_sha256"] = delivery[
            "executor_terminal_proof_sha256"
        ]
    transport = terminal.get("source_transport_exhaustion")
    retryable = (
        isinstance(transport, dict)
        and transport.get("schema") == "simready-bench-download-exhaustion-v1"
        and transport.get("method") == "GET"
        and transport.get("attempts") == 5
        and transport.get("maximum_attempts") == 5
        and transport.get("transient") is True
        and isinstance(transport.get("url_sha256"), str)
        and len(transport["url_sha256"]) == 64
        and isinstance(transport.get("exception_class"), str)
    )
    if not issued and terminal.get("status") != "completed":
        if external_terminal is not None:
            pass
        elif retryable:
            relay_path = batch / "daytona-executor/relay-ledger.json"
            if (
                relay_path.resolve() != relay_path
                or not relay_path.is_file()
                or relay_path.stat().st_nlink != 1
                or relay_path.stat().st_size > 32 * 1024**2
                or hashlib.sha256(relay_path.read_bytes()).hexdigest()
                != delivery.get("relay_ledger_sha256")
            ):
                raise ValueError(
                    "A source retry requires its unchanged retained relay ledger"
                )
            relay = _read(relay_path)
            if (
                relay.get("model") != parent["model"]
                or relay.get("maximum_requests") != held[1]
                or relay.get("maximum_condition_requests")
                != manifest["defaults"]["budgets"]["inference_requests"]
                or relay.get("expires_at") != parent["expires_at"]
                or relay.get("generation_requests") != 0
                or relay.get("input_counter_requests") != 0
                or relay.get("events") != []
                or relay.get("condition_generation_requests") != {}
                or relay.get("condition_input_counter_requests") != {}
                or relay.get("condition_limits") != {}
            ):
                raise ValueError(
                    "A source retry cannot reset an admitted provider request"
                )
            proof["source_transport_exhaustion"] = transport
            proof["relay_ledger_sha256"] = delivery["relay_ledger_sha256"]
        else:
            proof["terminal_unadmitted_conditions"] = sorted(
                condition_plan(_read(batch / "experiment.json"))
            )
    if terminal.get("source_acquisition_failure") is True and not retryable:
        proof["terminal_unadmitted_conditions"] = sorted(
            set(condition_plan(_read(batch / "experiment.json"))) - admitted
        )
    authority.retire(
        allocation_id=allocation,
        requests=charged_requests,
        maximum_spend_usd=charged_usd,
        proof_sha256=_digest(proof),
    )
    if proof_path.exists():
        if _read(proof_path) != proof:
            raise ValueError("A settled executor proof cannot change")
    else:
        _write(proof_path, proof)
    return proof


def _remaining(policy, manifest):
    from simready_bench.engine.inference_gateway import ProjectBudget
    from simready_bench.local import ParentAuthorityLedger

    provider = policy["providers"][manifest["models"][0]["provider_ref"]]
    parent = ProjectBudget(**provider["project_budget"])
    authority = ParentAuthorityLedger(policy["runtime"]["authority_state_dir"])
    with authority._db() as db:
        fingerprint = db.execute(
            "SELECT fingerprint FROM parents WHERE id=?", (parent.authorization_id,)
        ).fetchone()
        rows = db.execute(
            "SELECT COALESCE(r.requests,a.requests),COALESCE(r.usd,a.usd) FROM allocations a LEFT JOIN retirements r ON r.allocation=a.id WHERE a.parent=?",
            (parent.authorization_id,),
        ).fetchall()
    if fingerprint and fingerprint[0] != _digest(asdict(parent)):
        raise ValueError("The original project allowance cannot change when resuming")
    return parent.maximum_requests - sum(row[0] for row in rows), Decimal(
        str(parent.maximum_spend_usd)
    ) - sum((Decimal(row[1]) for row in rows), Decimal(0))


def _condition_rejections(output):
    return {
        row["condition_id"]
        for batch in _batches(Path(output))
        for _, _, report in _reports(batch)
        for row in report["results"]
        if row.get("status") == "condition_error"
    }


@coordinated_resume
def resume_benchmark(
    reference,
    *,
    model,
    harness,
    deploy,
    output_dir,
    skills=None,
    policy=None,
    maximum_spend_usd=None,
    repetitions=4,
    provider=None,
    api_key_env=None,
    model_metadata=None,
    assets_root=None,
    work_dir=None,
    wheel=None,
    cpu_snapshot=None,
    reasoning=None,
    runtime_images=None,
    hf_repo=None,
    hf_run=None,
    workers=1,
    dispatch_pending=True,
    max_retries=2,
    retry_delay_seconds=5,
    maximum_retry_delay_seconds=60,
    _initial_failure=False,
    authoring_turns=None,
):
    """Collect existing executors and retry incomplete work within saved limits."""
    options = dict(locals())
    for key in ("_initial_failure",):
        options.pop(key)
    import time
    from simready_bench.daytona import DaytonaObservationError
    from simready_bench.recovery import (
        RecoveryBudgetExhausted,
        RecoveryLimitReached,
        attempted_conditions,
        record_state,
        recovery_state,
        validate_recovery,
    )

    if type(dispatch_pending) is not bool:
        raise ValueError("Dispatch pending must be a boolean")
    validate_recovery(max_retries, retry_delay_seconds, maximum_retry_delay_seconds)
    if not dispatch_pending:
        return _resume_once(**options)
    output = Path(os.path.abspath(output_dir))
    state_path, state = recovery_state(
        output,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        maximum_retry_delay_seconds=maximum_retry_delay_seconds,
    )
    if _initial_failure and state["status"] == "ready":
        state["retry_waves"] = 1
        state["stalled_waves"] = 1
        record_state(
            state_path, state, "backoff", not_before=time.time() + retry_delay_seconds
        )
    if (
        state.get("status") == "retry_exhausted"
        and state["stalled_waves"] > max_retries
    ):
        raise RecoveryLimitReached(
            "Incomplete reconstructions exhausted their automatic retries"
        )
    expected = condition_plan(_read(output / "experiment.json"))
    maximum_waves = (max_retries + 1) * (1 + len(expected) * (max_retries + 1))
    while True:
        attempts, completed = attempted_conditions(output)
        rejected = _condition_rejections(output)
        state["condition_attempts"] = attempts
        eligible = {
            identity
            for identity in expected
            if identity not in completed | rejected
            and attempts.get(identity, 0) <= max_retries
        }
        exhausted = set(expected) - completed - eligible
        if state.get("not_before") is not None or (
            eligible and any(attempts.get(identity, 0) for identity in eligible)
        ):
            if state.get("not_before") is None:
                delay = min(
                    maximum_retry_delay_seconds,
                    retry_delay_seconds
                    * 2
                    ** min(
                        max(
                            (attempts.get(identity, 0) - 1 for identity in eligible),
                            default=0,
                        ),
                        10,
                    ),
                )
                record_state(
                    state_path, state, "backoff", not_before=time.time() + delay
                )
            delay = max(0, state["not_before"] - time.time())
            if delay:
                time.sleep(delay)
            state["not_before"] = None
        record_state(
            state_path, state, "collecting", exhausted_conditions=sorted(exhausted)
        )
        state["dispatch_waves"] += 1
        result, execution_error = None, None
        try:
            result = _resume_once(**options, _allowed_pending=eligible)
        except RecoveryBudgetExhausted:
            record_state(state_path, state, "budget_exhausted")
            raise
        except RecoveryLimitReached:
            record_state(state_path, state, "terminal_incomplete", not_before=None)
            raise
        except DaytonaObservationError:
            record_state(state_path, state, "observation_pending")
            raise
        except (ValueError, FileNotFoundError):
            record_state(state_path, state, "requires_attention")
            raise
        except Exception as error:
            execution_error = error
            result = getattr(error, "result", None)
        if execution_error is None and result.get("completed_conditions") == len(
            expected
        ):
            record_state(state_path, state, "completed", not_before=None)
            return result
        next_attempts, next_completed = attempted_conditions(output)
        rejected = _condition_rejections(output)
        state["condition_attempts"] = next_attempts
        progressed = any(
            next_attempts.get(identity, 0) > attempts.get(identity, 0)
            for identity in eligible
            if identity not in next_completed or attempts.get(identity, 0)
        )
        state["stalled_waves"] = 0 if progressed else state["stalled_waves"] + 1
        remaining = {
            identity
            for identity in expected
            if identity not in next_completed | rejected
            and next_attempts.get(identity, 0) <= max_retries
        }
        if (
            not max_retries
            or state["stalled_waves"] > max_retries
            or state["dispatch_waves"] >= maximum_waves
            or not remaining
            and len(next_completed) != len(expected)
        ):
            record_state(state_path, state, "retry_exhausted", not_before=None)
            if execution_error is not None:
                raise execution_error
            error = RecoveryLimitReached(
                "Incomplete reconstructions exhausted their automatic retries"
            )
            error.result = result
            raise error
        state["retry_waves"] += 1
        delay = min(
            maximum_retry_delay_seconds,
            retry_delay_seconds
            * 2
            ** min(
                max(
                    [
                        0,
                        state["stalled_waves"] - 1,
                        *(next_attempts.get(identity, 0) - 1 for identity in remaining),
                    ]
                ),
                10,
            ),
        )
        record_state(state_path, state, "backoff", not_before=time.time() + delay)


def _resume_once(
    reference,
    *,
    model,
    harness,
    deploy,
    output_dir,
    skills=None,
    policy=None,
    maximum_spend_usd=None,
    repetitions=4,
    provider=None,
    api_key_env=None,
    model_metadata=None,
    assets_root=None,
    work_dir=None,
    wheel=None,
    cpu_snapshot=None,
    reasoning=None,
    runtime_images=None,
    hf_repo=None,
    hf_run=None,
    workers=1,
    dispatch_pending=True,
    _allowed_pending=None,
    authoring_turns=None,
    max_retries=2,
    retry_delay_seconds=5,
    maximum_retry_delay_seconds=60,
):
    from simready_bench.benchmark import (
        _aggregate_reports,
        _copy_file,
        execute_local_reported,
        publication_callback,
    )
    from simready_bench.runtime_images import load_images

    validate_workers(workers)
    if deploy != "daytona" and workers != 1:
        raise ValueError("Parallel workers require Daytona deployment")
    output = Path(os.path.abspath(output_dir))
    original = _read(output / "experiment.json")
    if authoring_turns is not None:
        from simready_bench.engine.native_output import (
            authoring_turns as protocol_turns,
        )

        if authoring_turns != protocol_turns(original["authoring_protocol"]):
            raise ValueError("Resume must retain its original authoring turn limit")
    operator = _read(output / "operator.json")
    saved = _read(output / "run.json")
    images = load_images(output / "runtime-images.json")
    if (
        saved.get("schema") != "simready-bench-run-v1"
        or saved["experiment_sha256"] != _digest(original)
        or saved["operator_sha256"] != _digest(operator)
        or saved["runtime_images"] != images
        or saved["deployment"] != deploy
        or saved["reference"] != _reference(reference)
    ):
        raise ValueError("Resume requires the original benchmark configuration")
    selected = original["models"][0]
    connection = operator["providers"][selected["provider_ref"]]
    if (
        len(original["models"]) != 1
        or model != selected["model_id"]
        or len(original["harnesses"]) != 1
        or harness != original["harnesses"][0]["id"]
        or repetitions != original["repetitions"]
        or provider is not None
        and provider != connection["kind"]
        or api_key_env is not None
        and api_key_env != connection.get("api_key_env")
        or reasoning is not None
        and reasoning != selected.get("generation", {}).get("reasoning_effort")
        or maximum_spend_usd is not None
        and Decimal(str(maximum_spend_usd))
        != Decimal(str(operator["limits"]["maximum_spend_usd"]))
        or cpu_snapshot is not None
        and cpu_snapshot != saved["snapshot"]
        or runtime_images is not None
        and load_images(runtime_images) != images
    ):
        raise ValueError(
            "Model, harness, repetitions, reasoning, deployment and allowance must remain unchanged"
        )
    if policy is not None and _read(policy) != operator:
        raise ValueError("Resume uses the original saved operator policy")
    if model_metadata is not None and _read(model_metadata) != selected.get("metadata"):
        raise ValueError("Resume uses the original model metadata")
    if assets_root is not None and Path(assets_root).absolute() != Path(
        operator["roots"]["assets"]
    ):
        raise ValueError("Resume uses the original staged asset inputs")
    if skills is not None and [_tree(path) for path in skills] != saved["skill_inputs"]:
        raise ValueError("The requested skills differ from the original benchmark")
    if saved["inputs"] != {name: _tree(output / name) for name in saved["inputs"]}:
        raise ValueError("The staged benchmark input files have changed")
    publisher = publication_callback(output, original, hf_repo=hf_repo, hf_run=hf_run)
    if deploy == "daytona":
        from simready_bench.orchestration import coordinate_daytona
        from simready_bench.recovery import frozen_wheel

        return coordinate_daytona(
            original,
            policy=operator,
            output_dir=output,
            workers=workers,
            wheel=frozen_wheel(output, wheel) if dispatch_pending else wheel,
            cpu_snapshot=saved["snapshot"],
            runtime_images=images,
            work_dir=work_dir,
            on_report=publisher,
            dispatch_pending=dispatch_pending,
            allowed_pending=_allowed_pending,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds,
            maximum_retry_delay_seconds=maximum_retry_delay_seconds,
        )

    import threading

    expected = condition_plan(original)
    reports, sources, complete, rejected = [], {}, set(), set()
    publication_lock = threading.RLock()
    published_reports = {}

    def publish_unlocked(report, *, final=False):
        from simready_bench.reporting import read_generated_report

        if not isinstance(report, dict) or not all(
            isinstance(report.get(key), (str, os.PathLike))
            for key in ("directory", "results")
        ):
            raise ValueError(
                "A reconstruction report requires its directory and results file"
            )
        root = Path(report["directory"]).absolute()
        result_path = Path(report["results"]).absolute()
        if (
            root.resolve() != root
            or not root.is_dir()
            or not root.is_relative_to(output)
            or result_path.resolve() != result_path
            or not result_path.is_relative_to(root)
            or not result_path.is_file()
            or result_path.stat().st_nlink != 1
        ):
            raise ValueError(
                "A reconstruction report must stay within its owned output"
            )
        owner = max(
            (batch for batch in _batches(output) if root.is_relative_to(batch)),
            key=lambda batch: len(batch.parts),
        )
        owned = condition_plan(
            original if owner == output else _read(owner / "experiment.json")
        )
        for row in read_generated_report(result_path)["results"]:
            identity = row.get("condition_id")
            if identity not in owned or any(
                row.get(key) is not None and row[key] != value
                for key, value in expected[identity].items()
            ):
                raise ValueError(
                    "A published reconstruction changes its owned identity"
                )
            if row.get("status") == "completed" and not _completed(root, row):
                raise ValueError(
                    "A completed reconstruction requires its frozen artefacts"
                )
        proposed = {
            **published_reports,
            str(Path(report["results"]).absolute()): (
                root,
                dict(report),
            ),
        }
        chosen, completed = {}, set()
        for path, (_, entry) in proposed.items():
            for row in read_generated_report(entry["results"])["results"]:
                identity = row["condition_id"]
                if identity not in expected:
                    raise ValueError(
                        "A published reconstruction differs from the original benchmark"
                    )
                if any(
                    row.get(key) is not None and row[key] != value
                    for key, value in expected[identity].items()
                ):
                    raise ValueError(
                        "A published reconstruction changes its scientific identity"
                    )
                if identity not in completed or row.get("status") == "completed":
                    chosen[identity] = path
                if row.get("status") == "completed":
                    completed.add(identity)
        published_reports.clear()
        published_reports.update(proposed)
        combined = _aggregate_reports(
            list(proposed.values()),
            output_dir=output,
            planned_conditions=len(expected),
            condition_sources=chosen,
        )
        return publisher(combined, final=final) if publisher is not None else combined

    def publish(report, *, final=False):
        with publication_lock:
            return publish_unlocked(report, final=final)

    publication_options = {"on_report": publish} if publisher is not None else {}
    batches = _batches(output)
    for batch in batches:
        batch_manifest = original
        if batch != output:
            batch_manifest = _read(batch / "experiment.json")
            baseline, actual = copy.deepcopy(original), copy.deepcopy(batch_manifest)
            baseline.pop("reconstruction_selection", None)
            actual.pop("reconstruction_selection", None)
            if baseline != actual:
                raise ValueError(
                    "A saved execution differs from the original benchmark"
                )
        batch_reports = _reports(batch)
        for attempt, report, portable in batch_reports:
            published_reports[str(Path(report["results"]).absolute())] = (
                attempt,
                report,
            )
            for row in portable["results"]:
                identifier = row.get("condition_id")
                if identifier not in expected or any(
                    row.get(key) is not None and row[key] != value
                    for key, value in expected[identifier].items()
                ):
                    raise ValueError(
                        "A collected condition differs from the original benchmark identity"
                    )
                valid = _completed(attempt, row)
                if valid and identifier in complete:
                    raise ValueError("A reconstruction has completed more than once")
                if identifier not in sources or valid or identifier not in complete:
                    sources[identifier] = str(Path(report["results"]).absolute())
                if valid:
                    complete.add(identifier)
                if row.get("status") == "condition_error":
                    rejected.add(identifier)
            reports.append((attempt, report))
    selection = {asset["id"]: [] for asset in original["assets"]}
    for identifier, condition in expected.items():
        if identifier not in complete | rejected and (
            _allowed_pending is None or identifier in _allowed_pending
        ):
            selection[condition["asset_id"]].append(condition["repetition"])
    remaining_count = sum(len(repetitions) for repetitions in selection.values())
    report = _aggregate_reports(
        reports,
        output_dir=output,
        planned_conditions=len(expected),
        condition_sources=sources,
    )
    if not remaining_count or not dispatch_pending:
        from simready_bench.reporting import read_generated_report

        if publisher is not None:
            publisher(report, final=len(complete) == len(expected))
        return {
            **read_generated_report(report["results"]),
            "report": report,
            "dispatch_pending": dispatch_pending,
        }
    if connection.get("api_key_env") and not os.environ.get(connection["api_key_env"]):
        raise ValueError("Set " + connection["api_key_env"] + " before resuming")
    requests, dollars = _remaining(operator, original)
    budgets = original["defaults"]["budgets"]
    if (
        remaining_count * budgets["inference_requests"] > requests
        or Decimal(str(budgets["maximum_spend_usd"])) * remaining_count > dollars
    ):
        from simready_bench.recovery import RecoveryBudgetExhausted

        error = RecoveryBudgetExhausted(
            "The remaining reconstructions exceed the original unspent request/spend ceilings"
        )
        from simready_bench.reporting import read_generated_report

        error.result = {**read_generated_report(report["results"]), "report": report}
        raise error
    executions = output / "executions"
    executions.mkdir(exist_ok=True, mode=0o700)
    number = max([0, *[int(path.name) for path in executions.iterdir()]]) + 1
    batch = executions / format(number, "03d")
    batch.mkdir(mode=0o700)
    resumed = copy.deepcopy(original)
    resumed["reconstruction_selection"] = selection
    resumed_policy = copy.deepcopy(operator)
    for name in saved["inputs"]:
        target = batch / name
        target.mkdir(mode=0o700)
        for relative in saved["inputs"][name]:
            _copy_file(output / name / relative, target / relative)
        resumed_policy["roots"][name] = str(target)
    _write(batch / "experiment.json", resumed)
    _write(batch / "operator.json", resumed_policy)
    _write(batch / "runtime-images.json", images)
    error = None
    try:
        execute_local_reported(
            resumed,
            policy=resumed_policy,
            output_dir=batch,
            work_dir=work_dir,
            **publication_options,
        )
    except Exception as caught:
        error = caught
    for attempt, source, portable in _reports(batch):
        reports.append((attempt, source))
        for row in portable["results"]:
            identifier = row["condition_id"]
            if identifier not in complete:
                sources[identifier] = str(Path(source["results"]).absolute())
    report = _aggregate_reports(
        reports,
        output_dir=output,
        planned_conditions=len(expected),
        condition_sources=sources,
    )
    from simready_bench.reporting import read_generated_report

    result = {**read_generated_report(report["results"]), "report": report}
    if error is not None:
        error.result = result
        raise error
    return result
