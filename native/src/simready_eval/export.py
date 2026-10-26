"""Generate clean native Harbor packages including pinned public task inputs."""
from __future__ import annotations

import argparse
import ast
from email.parser import Parser
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import os
import zipfile

from .fetch import REPO_ID, fetch_assets, validate_manifest
from .dependencies import runtime_lock
from .limits import FEEDBACK_SCRATCH_BYTES

DATASET_NAME = "hcltech-robotics/simreadybench"
RELEASE_VERSIONS = {"simready-harbor-release-v1": ("v1", "1.0.2"),
                    "simready-harbor-release-v2": ("v2", "2.0.0")}

BASE_IMAGE = "python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"
SOURCE_COLLECTION = {"repo_id": "nvidia/simready-assets", "revision": "f4860aa0c75607adb50a78bf96e77510bffc040f"}
BLENDER_PACKAGE_VERSION = "4.3.2+dfsg-2"
RUNTIME_ENV = "ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PXR_WORK_THREAD_LIMIT=2"


def _safe_path(value):
    if (not isinstance(value, str) or not value or value.startswith("/") or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError("expected a canonical safe relative path")
    return value


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _task_name(task_id):
    return task_id if task_id.startswith(("nv100_", "nv100-")) else "nv100-" + task_id


def release_version(release):
    """Return the schema-bound asset prefix version and Harbor package version."""
    schema = release.get("schema") if isinstance(release, dict) else None
    if not isinstance(schema, str) or schema not in RELEASE_VERSIONS:
        raise ValueError("unsupported release schema version")
    return RELEASE_VERSIONS[schema]


def _validate_release(release, fixture):
    if not isinstance(release, dict) or set(release) != {"schema", "repo_id", "revision", "source_commit", "source_collection", "tasks"}:
        raise ValueError("release must contain exactly the release schema fields")
    asset_version, _ = release_version(release)
    if release["repo_id"] != REPO_ID:
        raise ValueError("unsupported release repo_id")
    for field in ("revision", "source_commit"):
        if not isinstance(release[field], str) or not re.fullmatch(r"[0-9a-f]{40}", release[field]):
            raise ValueError(f"{field} must be a full immutable 40-character revision")
    if release["source_collection"] != SOURCE_COLLECTION:
        raise ValueError("source_collection must be the approved pinned NVIDIA collection")
    tasks = release["tasks"]
    if not isinstance(tasks, list) or not tasks or len(tasks) > 100 or (not fixture and len(tasks) != 100):
        raise ValueError("NV100 releases require exactly 100 tasks; fixtures permit 1 to 100")
    seen, names = set(), set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {"id", "object_name", "nominal_dimensions_m", "actor", "verifier", "reference", "entrypoint", "views", "budgets"}:
            raise ValueError("task must contain exactly the supported task fields")
        task_id = task["id"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", task_id) or task_id in seen:
            raise ValueError("task ids must be unique safe components")
        seen.add(task_id)
        object_name = task["object_name"]
        if (not isinstance(object_name, str) or not 1 <= len(object_name) <= 160
                or object_name != object_name.strip() or not any(char.isalpha() for char in object_name)
                or any(ord(char) < 32 or ord(char) == 127 or char in "<>[]`\\" for char in object_name)):
            raise ValueError("object_name must be a concise readable single-line label without markup")
        dimensions = task["nominal_dimensions_m"]
        if (not isinstance(dimensions, list) or len(dimensions) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not 1e-9 <= value <= 1e6 for value in dimensions)):
            raise ValueError("nominal_dimensions_m requires three positive finite canonical metre XYZ values")
        name = _task_name(task_id)
        if name in names:
            raise ValueError("task ids must produce unique Hub names")
        names.add(name)
        for role in ("actor", "verifier"):
            manifest = validate_manifest(task[role])
            if (manifest["repo_id"] != release["repo_id"] or manifest["revision"] != release["revision"]
                    or manifest["task_id"] != task_id or manifest["role"] != role):
                raise ValueError("role manifest must match release repo, revision, task and role")
            # Preserve the release layout verbatim. Renaming an allowed public
            # observation into a reference destination would break its contract.
            prefix = f"benchmark/nv100/{asset_version}/tasks/{task_id}/"
            for file in manifest["files"]:
                if not file["path"].startswith(prefix):
                    raise ValueError("role path version must match the release schema prefix")
                expected = file["path"][len(prefix):]
                if role == "actor":
                    expected = expected.removeprefix("public/")
                if file["destination"] != expected:
                    raise ValueError("role destination must preserve the canonical release layout")
        reference = _safe_path(task["reference"])
        if not reference.startswith("reference/") or Path(reference).suffix.lower() not in (".usd", ".usda", ".usdc"):
            raise ValueError("reference must name a reference USD entrypoint")
        entrypoint = _safe_path(task["entrypoint"])
        if entrypoint != "asset.usd":
            raise ValueError("candidate entrypoint must be exactly asset.usd")
        views = task["views"]
        if (not isinstance(views, list) or len(views) != 4
                or any(not isinstance(view, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", view) for view in views)
                or len(set(views)) != 4):
            raise ValueError("views require four unique safe view identifiers")
        images = {f"comparison/{view}{suffix}.png" for view in views for suffix in ("", "-mask")}
        if len(images) != 8:
            raise ValueError("view RGB and mask paths must be distinct")
        required = {reference, "reference/eligibility.json", "comparison/cameras.json"} | images
        declared = {file["destination"] for file in task["verifier"]["files"]}
        if not required <= declared:
            raise ValueError("verifier manifest is missing reference, eligibility, cameras, RGB images or masks")
        budgets = task["budgets"]
        if (not isinstance(budgets, dict) or set(budgets) != {"agent_seconds", "final_seconds"}
                or any(type(value) not in (int, float) or not 0 < value < float("inf") for value in budgets.values())):
            raise ValueError("budgets require positive finite agent_seconds and final_seconds")
    # JSON roundtrip severs mutable caller references and rejects exotic types.
    return json.loads(_json(release))


def _forbidden_module(module):
    return any(part in {"simready_bench", "harbor", "private"} or part.startswith("private_") for part in module.split("."))


def _check_imports(source, name):
    try:
        tree = ast.parse(source, filename=name)
    except (SyntaxError, UnicodeError) as error:
        raise ValueError("wheel contains invalid Python source") from error
    dynamic_importers = {"__import__", "import_module"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("importlib", "builtins"):
            dynamic_importers.update(alias.asname or alias.name for alias in node.names
                                     if alias.name in ("import_module", "__import__"))
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.func, ast.Name) and node.func.id in dynamic_importers:
                modules = [node.args[0].value]
            elif isinstance(node.func, ast.Attribute) and node.func.attr in ("import_module", "__import__"):
                modules = [node.args[0].value]
        if any(isinstance(module, str) and _forbidden_module(module) for module in modules):
            raise ValueError("wheel contains a legacy, Harbor or private import")


def _wheel_inputs(wheel, expected_version=None):
    wheel = Path(wheel)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", wheel.name):
        raise ValueError("wheel must have a safe wheel filename")
    payload = wheel.read_bytes()
    if len(payload) > 256 * 1024 * 1024:
        raise ValueError("native wheel exceeds export size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if len(members) > 1000 or sum(member.file_size for member in members) > 256 * 1024 * 1024:
                raise ValueError("native wheel has excessive contents")
            names, distributions = set(), set()
            for member in members:
                name = _safe_path(member.filename.rstrip("/"))
                if name in names or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("wheel contains duplicate paths or symlinks")
                names.add(name)
                path = PurePosixPath(name)
                parts = path.parts
                if parts[0] == "simready_eval":
                    if any(_forbidden_module(part.removesuffix(".py")) for part in parts[1:]):
                        raise ValueError("wheel contains a private or legacy package path")
                    if not member.is_dir() and not (len(parts) == 2 and path.suffix == ".py" or len(parts) == 3 and parts[1] == "profiles" and path.suffix == ".json"):
                        raise ValueError("wheel contains unexpected evaluator payload")
                    if path.suffix == ".py" and not member.is_dir():
                        _check_imports(archive.read(member), name)
                elif re.fullmatch(r"[A-Za-z0-9_.+-]+\.dist-info", parts[0]):
                    distributions.add(parts[0])
                    if not member.is_dir() and not (len(parts) == 2 and parts[1] in {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt", "LICENSE", "LICENSE.txt", "NOTICE", "direct_url.json"}
                                                        or len(parts) >= 3 and parts[1] == "licenses"):
                        raise ValueError("unexpected distribution metadata")
                    if path.name == "METADATA":
                        metadata = archive.read(member).decode("utf-8")
                        if expected_version is not None:
                            headers = Parser().parsestr(metadata)
                            if headers.get("Name", "").lower().replace("_", "-") != "simreadybench-eval" or headers.get("Version") != expected_version:
                                raise ValueError(f"release requires the simreadybench-eval {expected_version} wheel")
                        if re.search(r"^Requires-Dist:\s*(?:harbor|simready[-_]bench)(?:[\s;(<=>[]|$)", metadata, re.MULTILINE | re.IGNORECASE):
                            raise ValueError("native wheel must not depend on Harbor or the old package")
                else:
                    raise ValueError("wheel may contain only simready_eval and distribution metadata")
            if len(distributions) != 1 or not {"simready_eval/__init__.py", "simready_eval/metric.py"} <= names:
                raise ValueError("wheel must contain one native distribution and its standalone metric")
            dist = next(iter(distributions))
            if not {dist + "/" + name for name in ("METADATA", "WHEEL", "RECORD")} <= names:
                raise ValueError("wheel is missing required distribution metadata")
            metric = archive.read("simready_eval/metric.py")
    except (zipfile.BadZipFile, KeyError) as error:
        raise ValueError("invalid native wheel") from error
    return wheel.name, payload, metric


def _task_toml(task, fixture, package_version="1.0.2"):
    return f'''schema_version = "1.4"
artifacts = [{{source = "/output/submission", destination = "candidate"}}]

[task]
name = "hcltech-robotics/{_task_name(task['id'])}"
version = "{package_version}"
description = {json.dumps("Reconstruct this " + task["object_name"] + " from one frozen observation")}

[metadata]
fixture = {str(fixture).lower()}
task_id = {json.dumps(task['id'])}
object_name = {json.dumps(task['object_name'])}
nominal_dimensions_m = {json.dumps(task['nominal_dimensions_m'])}

[environment]
network_mode = "public"
build_timeout_sec = 1200
cpus = 4
memory_mb = 8192
storage_mb = 8192

[agent]
timeout_sec = {task['budgets']['agent_seconds']}
user = 1000
network_mode = "allowlist"
allowed_hosts = []

[verifier]
environment_mode = "separate"
network_mode = "no-network"
timeout_sec = {task['budgets']['final_seconds']}

[verifier.environment]
network_mode = "no-network"
build_timeout_sec = 1200
cpus = 4
memory_mb = 16384
storage_mb = 8192
'''


def _actor_docker(wheel):
    return f'''FROM {BASE_IMAGE}
{RUNTIME_ENV}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git && rm -rf /var/lib/apt/lists/*
COPY {wheel} /opt/wheels/{wheel}
COPY actor.requirements.txt /opt/locks/actor.requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/locks/actor.requirements.txt && python -m pip install --no-cache-dir --no-deps '/opt/wheels/{wheel}'
COPY actor-assets.json /opt/setup/actor-assets.json
COPY actor-runtime.json /opt/setup/actor-runtime.json
RUN mkdir -p /opt/setup/frozen-public && chmod 700 /opt/setup/frozen-public
COPY observation.png /opt/setup/frozen-public/observation.png
COPY source-public-instruction.txt /opt/setup/frozen-public/instruction.md
RUN python -m simready_eval.fetch --manifest /opt/setup/actor-assets.json --destination /opt/setup/frozen-public --verify-only && rm -rf /opt/setup/frozen-public
COPY observation.png instruction.md /task/
RUN python -m simready_eval.fetch --manifest /opt/setup/actor-runtime.json --destination /task --verify-only && chmod -R a+rX /task && chmod -R a-w /task
RUN useradd --create-home --uid 1000 agent && mkdir -p /output/submission && chown -R 1000:1000 /output /home/agent
ENV HOME=/home/agent
WORKDIR /output
USER 1000
CMD ["sh", "-c", "sleep infinity"]
'''


def _verifier_docker(wheel):
    return f'''FROM {BASE_IMAGE}
{RUNTIME_ENV}
ARG BLENDER_PACKAGE_VERSION={BLENDER_PACKAGE_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends blender=$BLENDER_PACKAGE_VERSION && blender --background --version | head -n 1 | grep -F 'Blender 4.3.2' && rm -rf /var/lib/apt/lists/*
COPY {wheel} /opt/wheels/{wheel}
COPY verifier.requirements.txt /opt/locks/verifier.requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/locks/verifier.requirements.txt && python -m pip install --no-cache-dir --no-deps '/opt/wheels/{wheel}'
COPY verifier-assets.json /tests/verifier-assets.json
COPY evaluator.json /tests/evaluator.json
COPY test.sh /tests/test.sh
RUN python -m simready_eval.fetch --manifest /tests/verifier-assets.json --destination /fixtures --cache /tmp/verifier-cache && chmod -R a+rX /fixtures /tests && chmod -R a-w /fixtures /tests && chmod a+x /tests/test.sh && rm -rf /tmp/verifier-cache
WORKDIR /output
CMD ["sh", "-c", "sleep infinity"]
'''


def _feedback_docker(wheel):
    return f'''FROM {BASE_IMAGE}
{RUNTIME_ENV}
COPY {wheel} /opt/wheels/{wheel}
COPY feedback.requirements.txt /opt/locks/feedback.requirements.txt
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/locks/feedback.requirements.txt && python -m pip install --no-cache-dir --no-deps '/opt/wheels/{wheel}'
RUN mkdir -p /state && chmod 700 /state
USER 0
ENTRYPOINT ["python", "-m", "simready_eval.feedback", "serve", "--state", "/state", "--host", "0.0.0.0", "--port", "8080"]
'''


def _compose():
    return {"services": {
        "main": {"depends_on": {"feedback": {"condition": "service_healthy"}}},
        "feedback": {"build": {"context": "${CONTEXT_DIR}", "dockerfile": "Dockerfile.feedback"},
                     "user": "0", "cap_drop": ["ALL"], "cap_add": ["SETUID", "SETGID", "CHOWN"],
                     "security_opt": ["no-new-privileges:true"], "read_only": True,
                     "volumes": ["feedback-ledger:/state"], "tmpfs": [f"/tmp:size={FEEDBACK_SCRATCH_BYTES},mode=1777"],
                     "cpus": 2, "mem_limit": "8g",
                     "healthcheck": {"test": ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2).read()"],
                                     "interval": "2s", "timeout": "3s", "retries": 30}}},
            "volumes": {"feedback-ledger": {}}}


LICENSE = '''MIT License

Copyright (c) 2026 Chris von Csefalvay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


def _task_files(task, wheel_name, wheel_payload, fixture, package_version="1.0.2"):
    evaluator = {"schema": "simready-evaluator-config-v1", "task_id": task["id"],
                 "entrypoint": task["entrypoint"], "reference": task["reference"],
                 "cameras": "comparison/cameras.json", "eligibility": "reference/eligibility.json",
                 "fidelity_settings": {"enabled": True}, "final_seconds": task["budgets"]["final_seconds"]}
    image = next(file for file in task["actor"]["files"] if file["destination"] == "observation.png")
    image_url = f"https://huggingface.co/datasets/{task['actor']['repo_id']}/resolve/{task['actor']['revision']}/{image['path']}"
    instruction = f'''Reconstruct this {task['object_name']} from the single frozen observation as a self-contained USD package.

Nominal dimensions in canonical metres XYZ: {json.dumps(task['nominal_dimensions_m'])}
Stable task ID: `{task['id']}`

![Frozen task observation]({image_url})

[Download the frozen observation]({image_url}) · [Packaged image](environment/observation.png) · [Public instructions](environment/instruction.md)

The packaged observation is /task/observation.png.
Read /task/instruction.md for the task-specific public instructions.
Save the package under /output/submission, with its entrypoint at {task['entrypoint']}.
Do not use Blender for authoring; Blender is used only by the verifier renderer.

Up to five formal feedback calls are available. Use a fresh request ID for each
changed candidate; retry the same request ID after a transport interruption:

    python -m simready_eval.feedback validate_candidate --package /output/submission --request-id attempt-1 --url http://127.0.0.1:8080

HTTP 503 feedback_busy is retryable after Retry-After with the same request ID;
the busy request has not been admitted and does not consume a feedback call.

Final evaluation independently measures formal, visual, and kinematic fidelity.
'''
    return {"instruction.md": instruction, "task.toml": _task_toml(task, fixture, package_version),
            "environment/instruction.md": instruction.replace("(environment/observation.png)", "(observation.png)")
                                                     .replace(" · [Public instructions](environment/instruction.md)", ""),
            "environment/Dockerfile": _actor_docker(wheel_name),
            "environment/Dockerfile.feedback": _feedback_docker(wheel_name),
            "environment/docker-compose.yaml": _json(_compose()),
            "environment/actor-assets.json": _json(task["actor"]),
            "environment/actor.requirements.txt": runtime_lock("actor"),
            "environment/feedback.requirements.txt": runtime_lock("feedback"),
            "environment/" + wheel_name: wheel_payload,
            "tests/Dockerfile": _verifier_docker(wheel_name),
            "tests/verifier-assets.json": _json(task["verifier"]),
            "tests/verifier.requirements.txt": runtime_lock("verifier"),
            "tests/evaluator.json": _json(evaluator),
            "tests/" + wheel_name: wheel_payload,
            "tests/test.sh": "#!/bin/sh\nset -eu\nexec python -m simready_eval.cli evaluate --submission /output/submission --fixtures /fixtures --config /tests/evaluator.json --output /logs/verifier\n"}



def release_readme(release, *, fixture=False):
    """Return public benchmark usage, score definitions, provenance and index."""
    count = len(release["tasks"])
    title = "SimReadyBench NV100" + (" (synthetic fixture)" if fixture else "")
    text = f'''# {title}

SimReadyBench measures reconstruction of articulated USD assets from one image
and nominal dimensions. This set contains {count} tasks: {count} public observation images
and {4 * count} comparison images. Each agent receives one observation. The four
comparison views and reference USD are used by the verifier.

## Install and run

Use Linux x86_64 with Docker, Compose and Harbor 0.24.0:

```sh
python -m pip install 'harbor==0.24.0'
harbor run --dataset {DATASET_NAME} --agent YOUR_AGENT --model YOUR_MODEL
```

Use the dataset content digest to reproduce a specific release. Configure the
chosen provider's endpoints in `agents[].extra_allowed_hosts`. Public inputs
are included in each task package. Verifier references are fetched from pinned
HF files during setup; final evaluation runs offline.

## Submission and feedback

Write a self-contained USD package to `/output/submission`, with `asset.usd` as
its entrypoint. The package limit is 1 GiB of expanded files and 10000 files.
The task's nominal dimensions use canonical metres in XYZ order. Blender is
reserved for the verifier renderer and must not be used for agent authoring.

Up to five candidate-only formal feedback calls are available:

```sh
python -m simready_eval.feedback validate_candidate --package /output/submission --request-id attempt-1 --url http://127.0.0.1:8080
```

Use a new request ID for each changed candidate; retry a transport interruption
with the same ID. Final verification independently scores the submitted files.

## Scores

Formal, visual and kinematic fidelity are separate scores on a 0–100 scale;
Harbor rewards normalize each to 0–1. There is no combined overall score.

- **Formal:** `95 * passed / 46`, plus 5 when all 46 mandatory requirements pass.
  The 51 optional requirements are diagnostic and do not change this score.
- **Visual:** `sqrt(G * R)`. `G` measures full 3D surface agreement in both
  directions after translation/yaw alignment, without scale fitting. `R` is the
  mean foreground SSIM across the four fixed comparison views, using the union
  of reference and candidate foreground masks. Cameras and lighting are fixed.
  Material-property correspondence and its coverage are reported as diagnostics.
- **Kinematic:** recovery of moving and fixed joint constraints, including motion,
  limits and endpoints over 17 deterministic forward-kinematic states. Applicable
  moving/fixed recovery groups combine by geometric mean.

Known scores, coverage and unavailable dimensions are reported separately. The
dataset metric averages each dimension over its own known results and reports
coverage and score bounds; infrastructure-unknown dimensions are not model zeros.

## Data and licences

The canonical dataset is [{release['repo_id']}](https://huggingface.co/datasets/{release['repo_id']}/tree/{release['revision']})
at immutable revision `{release['revision']}`. Task manifests pin each file's
size and SHA-256. NVIDIA source: `nvidia/simready-assets@{SOURCE_COLLECTION['revision']}`.

Evaluator code is MIT licensed; see [LICENSE](LICENSE). Assets and rendered
references retain their asset-specific upstream licences and provenance in the
pinned dataset. Dependency licences are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Frozen task observations

'''
    rows = [f"- [{task['object_name']}]({_task_name(task['id'])}/instruction.md) (`{task['id']}`): "
            f"[preview/download](https://huggingface.co/datasets/{release['repo_id']}/resolve/{release['revision']}/"
            + next(file["path"] for file in task["actor"]["files"] if file["destination"] == "observation.png") + ")"
            for task in sorted(release["tasks"], key=lambda task: task["id"])]
    return text + "\n".join(rows) + "\n"


def _publisher_digest(files):
    # Harbor 0.24 Packager.compute_content_hash: sorted portable path, NUL,
    # per-file SHA256, newline. No Harbor runtime import is needed by exporter.
    digest = hashlib.sha256()
    for path, payload in sorted(files.items()):
        digest.update(f"{path}\0{_hash(payload)}\n".encode())
    return "sha256:" + digest.hexdigest()


def _fresh_output(output):
    output = Path(os.path.abspath(os.fspath(output)))
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a fresh nonexistent directory")
    for parent in output.parents:
        git = parent / ".git"
        if parent.is_symlink() or git.is_file() or (git / "HEAD").is_file():
            raise ValueError("output must be outside any Git checkout and symlink path")
    return output


def export_release(release, output, wheel, *, fixture=False, public_assets=None):
    """Export only supplied release tasks; return sorted file hashes and sizes.

    The two declared actor inputs are acquired with their frozen size/hash checks.
    The raw frozen instruction is retained as source-public-instruction.txt;
    named public/runtime wrappers have a separate local verification identity.
    By default they are fetched anonymously at the immutable HF revision. An
    optional local ``public_assets`` root mirrors the manifest's HF paths and
    avoids network access; only the exact actor file paths are read. No reference
    or comparison file is acquired or embedded. No publication occurs.
    ``fixture=True`` permits a
    smaller synthetic roster, identified in its title and machine metadata.
    ``inventory.json`` inventories every other file, avoiding a self-hash cycle.
    """
    if type(fixture) is not bool:
        raise ValueError("fixture must be a boolean")
    release = _validate_release(release, fixture)
    _, package_version = release_version(release)
    output = _fresh_output(output)
    wheel_name, wheel_payload, metric = _wheel_inputs(wheel, expected_version=package_version)
    files, task_refs = {}, []
    with tempfile.TemporaryDirectory(prefix="simready-export-public-") as temporary:
        staging = Path(temporary)
        local_root = Path(public_assets) if public_assets is not None else None
        downloader = None if local_root is None else lambda **kwargs: local_root / kwargs["filename"]
        for task in sorted(release["tasks"], key=lambda task: task["id"]):
            public = staging / task["id"]
            fetch_assets(task["actor"], public, cache_dir=staging / "cache", downloader=downloader,
                         timeout_seconds=300)
            task_files = {path: payload.encode() if isinstance(payload, str) else payload
                          for path, payload in _task_files(task, wheel_name, wheel_payload, fixture, package_version).items()}
            for file in task["actor"]["files"]:
                destination = "source-public-instruction.txt" if file["destination"] == "instruction.md" else file["destination"]
                task_files["environment/" + destination] = (public / file["destination"]).read_bytes()
            original_brief = task_files["environment/source-public-instruction.txt"]
            for path in ("instruction.md", "environment/instruction.md"):
                task_files[path] += b"\n## Frozen public instructions\n\n" + original_brief
            runtime = {
                "schema": "simready-actor-runtime-v1", "task_id": task["id"], "instruction_origin": "derived-public-wrapper",
                "frozen_actor_manifest_sha256": _hash(json.dumps(task["actor"], sort_keys=True, separators=(",", ":")).encode()),
                "files": [{"destination": name, "size_bytes": len(task_files["environment/" + name]),
                           "sha256": _hash(task_files["environment/" + name])}
                          for name in ("observation.png", "instruction.md")],
            }
            task_files["environment/actor-runtime.json"] = _json(runtime).encode()
            name = _task_name(task["id"])
            files.update({name + "/" + path: payload for path, payload in task_files.items()})
            task_refs.append({"name": "hcltech-robotics/" + name, "digest": _publisher_digest(task_files)})
    release["export"] = {"fixture": fixture}
    files.update({"metric.py": metric, "release.json": _json(release).encode(), "LICENSE": LICENSE.encode(),
                  "README.md": release_readme(release, fixture=fixture).encode(),
                  "THIRD_PARTY_NOTICES.md": ("# Third-party notices\n\nEvaluator code is MIT licensed. Downloaded NVIDIA assets retain their upstream license and provenance; only the frozen public observation PNGs and instructions are bundled; verifier USD, dependencies and comparison images remain external. "
                    "The source collection is nvidia/simready-assets at " + SOURCE_COLLECTION["revision"] + ". "
                    "Runtime dependencies retain their own licenses, including Blender (GPL-3.0-or-later), OpenUSD (Apache-2.0), Pillow (HPND), NumPy (BSD-3-Clause), and HF Hub/Xet (Apache-2.0). "
                    "Per-asset source and licence provenance accompany the pinned dataset.\n").encode()})
    dataset = f'schema_version = "1.0"\n\n[dataset]\nname = {json.dumps(DATASET_NAME)}\nversion = "{package_version}"\ndescription = "SimReadyBench NV100' + (" fixture" if fixture else "") + '"\n'
    for task in task_refs:
        dataset += '\n[[tasks]]\nname = ' + json.dumps(task["name"]) + '\ndigest = ' + json.dumps(task["digest"]) + '\n'
    for name in ("metric.py", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "release.json"):
        dataset += '\n[[files]]\npath = ' + json.dumps(name) + '\ndigest = "sha256:' + _hash(files[name]) + '"\n'
    files["dataset.toml"] = dataset.encode()
    inventory = [{"path": name, "size_bytes": len(payload), "sha256": _hash(payload)} for name, payload in sorted(files.items())]
    files["inventory.json"] = _json(inventory).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".simready-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "release"
        staged.mkdir()
        for name, payload in files.items():
            path = staged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o755 if name.endswith("/test.sh") else 0o644)
        _fresh_output(output)
        staged.rename(output)
    return inventory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--public-assets", type=Path, help="Optional local mirror of pinned HF paths; checksums are still enforced")
    args = parser.parse_args(argv)
    try:
        inventory = export_release(json.loads(args.release.read_text()), args.output, args.wheel, fixture=args.fixture, public_assets=args.public_assets)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"native export failed: {error}\n")
    print(_json(inventory), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
