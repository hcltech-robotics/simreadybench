"""Immutable records shared by benchmark services; no harness-specific behaviour."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SnapshotFile:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    episode_id: str
    sha256: str | None
    path: str | None
    files: tuple[SnapshotFile, ...] = ()
    size_bytes: int = 0
    entrypoint: str = "asset.usd"
    error_code: str | None = None

    @property
    def valid(self) -> bool:
        return (
            self.error_code is None
            and self.sha256 is not None
            and self.path is not None
        )


@dataclass(frozen=True)
class Episode:
    episode_id: str
    task_id: str
    trial_id: str
    runner: str
    actor_root: str
    status: Literal["registered", "active", "closed"]
    deadline_seconds: float
    started_at: float | None
    deadline_at: float | None
    termination_reason: str | None
    validation_limit: int
    preview_limit: int
    inference_limit: int
    validation_calls_used: int = 0
    preview_calls_used: int = 0
    inference_calls_used: int = 0


@dataclass(frozen=True)
class ValidationJob:
    job_id: str
    episode_id: str
    request_id: str
    kind: Literal["validation", "preview", "inference"]
    sequence: int
    snapshot: Snapshot | None
    status: str = "reserved"
    admitted_at: float = 0.0
    deadline_at: float = 0.0
    result_json: str | None = None


@dataclass(frozen=True)
class Finding:
    rule_id: str
    prim_path: str | None
    severity: str
    finding: str
    observed_json: str = "{}"


@dataclass(frozen=True)
class ValidationReport:
    request_id: str
    snapshot_sha256: str | None
    validation_calls_used: int
    validation_calls_remaining: int
    findings: tuple[Finding, ...]
    status: str


@dataclass(frozen=True)
class RenderReport:
    request_id: str
    snapshot_sha256: str | None
    image_paths: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class EvaluationReport:
    snapshot_sha256: str | None
    outcome: str
    reward: int | None
    gates_json: str
    termination_reason: str | None = None
