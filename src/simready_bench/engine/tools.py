"""Episode-bound thumbnail, validation and submission tools."""

from __future__ import annotations
import base64
from dataclasses import dataclass
import hashlib
import hmac
import io
import json
from pathlib import Path
import threading
import time
from mcp.types import CallToolResult, ImageContent, TextContent, Tool
from PIL import Image
from simready_bench.engine.controller import AdmissionError

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _result(data, images=(), error=False):
    return CallToolResult(
        content=[
            TextContent(type="text", text=json.dumps(data, sort_keys=True)),
            *images,
        ],
        structured_content=data,
        is_error=error,
    )


def _error(status):
    return _result({"status": status}, error=True)


def _png(data, expected_size=None):
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the configured transport size")
    with Image.open(io.BytesIO(data)) as im:
        if im.format != "PNG":
            raise ValueError("PNG required")
        if expected_size is not None and im.size != expected_size:
            raise ValueError("Unexpected rendered resolution")
        if im.width <= 0 or im.height <= 0 or im.width * im.height > 4096 * 4096:
            raise ValueError("Invalid image dimensions")
        im.load()
        size = im.size
    return (
        ImageContent(
            type="image",
            data=base64.b64encode(data).decode("ascii"),
            mime_type="image/png",
        ),
        size,
    )


@dataclass(frozen=True)
class _Binding:
    episode_id: str
    token_digest: bytes
    thumbnail: bytes
    public_text: str
    submission_lock: threading.Lock
    validation_lock: threading.Lock
    public_contract: dict


def _is_production_adapter(adapter, kind, controller):
    from simready_bench.engine.harbor import NativeActorStopAdapter

    return (
        kind == "stop_actor"
        and type(adapter) is NativeActorStopAdapter
        and (adapter.controller is controller)
    )


class ActorTools:
    def __init__(self, controller, *, validator=None, stop_actor=None):
        if stop_actor is not None and (
            not _is_production_adapter(stop_actor, "stop_actor", controller)
        ):
            raise ValueError("A native actor stop adapter is required")
        self.controller = controller
        self.validator = validator
        self.stop_actor = stop_actor
        self._bindings = {}
        self._guard = threading.Lock()
        self._submissions = {}
        self._reports = controller.state_dir / "tool-reports"
        self._reports.mkdir(exist_ok=True, mode=448)

    def _adapter(self, adapter, kind):
        if adapter is not None and (
            not _is_production_adapter(adapter, kind, self.controller)
        ):
            raise RuntimeError("Unrecognised actor service adapter")
        return adapter

    def bind_episode(
        self, episode_id, token, thumbnail, public_text, *, public_contract=None
    ):
        """Trusted registration only; no actor operation can change a binding."""
        self.controller.get_episode(episode_id)
        if not isinstance(token, str) or len(token) < 16 or len(token) > 512:
            raise ValueError("An episode-scoped token is required")
        if not isinstance(public_text, str):
            raise ValueError("Public task text is required")
        if public_contract is not None and (not isinstance(public_contract, dict)):
            raise ValueError("Public contract must be a mapping")
        public_contract = json.loads(json.dumps(public_contract or {}, allow_nan=False))
        data = Path(thumbnail).read_bytes()
        _png(data)
        digest = hashlib.sha256(token.encode()).digest()
        with self._guard:
            if episode_id in self._bindings or any(
                (
                    hmac.compare_digest(digest, b.token_digest)
                    for b in self._bindings.values()
                )
            ):
                raise ValueError("Episode binding or token already registered")
            self._bindings[episode_id] = _Binding(
                episode_id,
                digest,
                data,
                public_text,
                threading.Lock(),
                threading.Lock(),
                public_contract,
            )

    def _binding(self, token):
        if not isinstance(token, str) or len(token) > 512:
            return None
        digest = hashlib.sha256(token.encode()).digest()
        with self._guard:
            return next(
                (
                    b
                    for b in self._bindings.values()
                    if hmac.compare_digest(digest, b.token_digest)
                ),
                None,
            )

    def _private(self, identifier, raw):
        destination = self._reports / (identifier + ".json")
        destination.write_text(
            json.dumps(raw, sort_keys=True, indent=2, default=str) + "\n"
        )
        destination.chmod(384)

    def describe(self, token):
        binding = self._binding(token)
        if binding is None:
            return _error("unauthorized")
        definitions = tool_definitions()
        return _result(
            {
                "tools": [
                    t.model_dump(mode="json", by_alias=True, exclude_none=True)
                    for t in definitions
                ]
            }
        )

    def invoke(self, token, operation, arguments):
        binding = self._binding(token)
        if binding is None:
            return _error("unauthorized")
        required = {
            "get_task_thumbnail": set(),
            "validate_candidate": {"request_id"},
            "submit_asset": {"request_id"},
        }
        if operation not in required:
            return _error("unknown_operation")
        if not isinstance(arguments, dict):
            return _error("invalid_arguments")
        if set(arguments) != required[operation]:
            return _error("invalid_arguments")
        if "request_id" in arguments:
            rid = arguments["request_id"]
            if not isinstance(rid, str) or not rid or len(rid) > 256 or ("\x00" in rid):
                return _error("invalid_arguments")
        try:
            episode = self.controller.get_episode(binding.episode_id)
            is_submission_retry = (
                operation == "submit_asset" and binding.episode_id in self._submissions
            )
            if not is_submission_retry:
                if episode.status == "closed":
                    return _error("episode_closed")
                if episode.status != "active" and operation != "get_task_thumbnail":
                    return _error("episode_not_started")
                if (
                    episode.deadline_at is not None
                    and time.time() >= episode.deadline_at
                ):
                    return _error("deadline_exceeded")
            if operation == "get_task_thumbnail":
                episode = self.controller.get_episode(binding.episode_id)
                if episode.status == "closed":
                    return _error("episode_closed")
                image, size = _png(binding.thumbnail)
                return _result(
                    {
                        "status": "completed",
                        "task": binding.public_text,
                        "thumbnail": {
                            "sha256": hashlib.sha256(binding.thumbnail).hexdigest(),
                            "width": size[0],
                            "height": size[1],
                        },
                    },
                    [image],
                )
            if operation == "submit_asset":
                return self._submit(binding, arguments["request_id"])
            with binding.validation_lock:
                job = self.controller.reserve_validation(
                    binding.episode_id, arguments["request_id"]
                )
                if job.result_json is not None:
                    return CallToolResult.model_validate_json(job.result_json)
                if job.status != "reserved":
                    return _error(
                        "job_in_progress"
                        if job.status == "running"
                        else "episode_closed"
                    )
                self.controller.mark_job_running(job.job_id)
                result = self._job(job, arguments)
                self.controller.complete_job(
                    job.job_id, result.model_dump(mode="json", by_alias=True)
                )
                return result
        except AdmissionError as exc:
            return _error(exc.code)
        except Exception:
            return _error("infrastructure_error")

    def _job(self, job, arguments):
        episode = self.controller.get_episode(job.episode_id)
        payload = {"request_id": job.request_id, "snapshot_sha256": job.snapshot.sha256}
        payload.update(
            validation_calls_used=episode.validation_calls_used,
            validation_calls_remaining=max(
                0, episode.validation_limit - episode.validation_calls_used
            ),
        )
        if time.time() >= job.deadline_at:
            return _result({**payload, "status": "validation_timeout"}, error=True)
        return self._validation_job(
            binding=self._bindings[job.episode_id],
            job=job,
            payload=payload,
            arguments=arguments,
        )

    def _validation_job(self, *, binding, job, payload, arguments):
        raise RuntimeError("The NVIDIA validation service has not been configured")

    def _submit(self, binding, request_id):
        with binding.submission_lock:
            if binding.episode_id in self._submissions:
                return self._submissions[binding.episode_id]
            stop_actor = self._adapter(self.stop_actor, "stop_actor")
            if stop_actor is None:
                return _error("infrastructure_error")
            self.controller.close_episode(binding.episode_id, "submitted")
            if stop_actor(binding.episode_id, "submitted") is not True:
                return _error("infrastructure_error")
            snapshot = self.controller.finalize_submission(
                binding.episode_id, reason="submitted", request_id=request_id
            )
            result = _result(
                {
                    "status": "submission_frozen",
                    "request_id": request_id,
                    "snapshot_sha256": snapshot.sha256,
                }
            )
            self._submissions[binding.episode_id] = result
            return result


def tool_definitions():
    request = {"type": "string", "minLength": 1, "maxLength": 256}

    def schema(properties):
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    return [
        Tool(
            name="get_task_thumbnail",
            description="Read the supplied task thumbnail and public brief.",
            input_schema=schema({}),
        ),
        Tool(
            name="validate_candidate",
            description="Validate a frozen candidate and return NVIDIA findings. Each new request consumes one validation call. Repeating a request ID returns its existing result.",
            input_schema=schema({"request_id": request}),
        ),
        Tool(
            name="submit_asset",
            description="End authoring and freeze the submitted asset. Returns the submission hash.",
            input_schema=schema({"request_id": request}),
        ),
    ]
