"""Native Harbor exports contain only generated files and pinned setup inputs."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import zipfile

import pytest


REPO = "hcltech-robotics/simreadybench"
REVISION = "1" * 40
VIEWS = ["isometric", "front", "side", "top"]
BASE = "python:3.12-slim-trixie@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704"


def exporter():
    assert importlib.util.find_spec("simready_eval.export") is not None, "native exporter is missing"
    return importlib.import_module("simready_eval.export")


def role_manifest(task_id, role, destinations):
    prefix = f"benchmark/nv100/v1/tasks/{task_id}/"
    return {"schema": "simready-role-assets-v1", "repo_id": REPO, "revision": REVISION,
            "task_id": task_id, "role": role, "files": [
                {"path": prefix + ("public/" + name if role == "actor" else name),
                 "destination": name, "size_bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}
                for name in destinations]}


def release_input(count=1):
    tasks = []
    for index in range(count):
        task_id = f"fixture-{index:03}"
        tasks.append({"id": task_id, "object_name": "articulated workshop clamp", "nominal_dimensions_m": [0.2, 0.1, 0.3], "actor": role_manifest(task_id, "actor", ["observation.png", "instruction.md"]),
                      "verifier": role_manifest(task_id, "verifier", ["reference/asset.usd", "reference/eligibility.json", "comparison/cameras.json",
                          *[f"comparison/{view}{suffix}.png" for view in VIEWS for suffix in ("", "-mask")]]),
                      "reference": "reference/asset.usd", "entrypoint": "asset.usd", "views": list(VIEWS),
                      "budgets": {"agent_seconds": 231000, "final_seconds": 600}})
    return {"schema": "simready-harbor-release-v1", "repo_id": REPO, "revision": REVISION,
            "source_commit": "2" * 40, "source_collection": {"repo_id": "nvidia/simready-assets", "revision": "f4860aa0c75607adb50a78bf96e77510bffc040f"}, "tasks": tasks}


@pytest.fixture
def wheel(tmp_path):
    path = tmp_path / "simreadybench_eval-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("simready_eval/__init__.py", '__version__ = "1.0.0"\n')
        archive.writestr("simready_eval/cli.py", "def main(): pass\n")
        archive.writestr("simready_eval/metric.py", Path("native/src/simready_eval/metric.py").read_bytes())
        archive.writestr("simreadybench_eval-1.0.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: simreadybench-eval\nVersion: 1.0.0\n")
        archive.writestr("simreadybench_eval-1.0.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: local-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr("simreadybench_eval-1.0.0.dist-info/RECORD", "")
    return path


@pytest.fixture(autouse=True)
def synthetic_public_download(tmp_path, monkeypatch):
    source = tmp_path / "public-download"
    source.write_bytes(b"x")
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return source
    monkeypatch.setattr("simready_eval.fetch._hub_download", download)
    return calls


def _run_actor_public_build(environment, root):
    import shlex
    import shutil
    frozen, actual = root / "frozen-public", root / "task"
    for line in (environment / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY "):
            words = shlex.split(line)[1:]
            sources, target = words[:-1], words[-1]
            if target.startswith("/opt/setup/frozen-public/"):
                destination = frozen / Path(target).name
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(environment / sources[0], destination)
            elif target == "/task/":
                actual.mkdir(mode=0o700)
                for source in sources:
                    shutil.copyfile(environment / source, actual / source)
        elif line.startswith("RUN python -m simready_eval.fetch"):
            command = line.removeprefix("RUN ")
            for source, target in (("/opt/setup/actor-assets.json", environment / "actor-assets.json"),
                                   ("/opt/setup/actor-runtime.json", environment / "actor-runtime.json"),
                                   ("/opt/setup/frozen-public", frozen), ("/task", actual)):
                command = command.replace(source, shlex.quote(str(target)))
            subprocess.run(["sh", "-c", command], check=True, text=True, capture_output=True)
    return actual


def test_export_bundles_only_frozen_actor_inputs_and_exposes_preview(tmp_path, wheel, synthetic_public_download):
    release = release_input()
    root = tmp_path / "release"
    inventory = exporter().export_release(release, root, wheel, fixture=True)
    task = root / "nv100-fixture-000"
    for file in release["tasks"][0]["actor"]["files"]:
        name = "source-public-instruction.txt" if file["destination"] == "instruction.md" else file["destination"]
        path = task / "environment" / name
        assert path.is_file(), "published task folder must include its public input bytes"
        assert path.read_bytes() == b"x"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file["sha256"]
    assert {call["filename"] for call in synthetic_public_download} == {
        file["path"] for file in release["tasks"][0]["actor"]["files"]}
    assert all(call["revision"] == REVISION and call["token"] is False for call in synthetic_public_download)
    image = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/benchmark/nv100/v1/tasks/fixture-000/public/observation.png"
    assert f"![" in (task / "instruction.md").read_text()
    assert image in (task / "instruction.md").read_text()
    assert "environment/observation.png" in (task / "instruction.md").read_text()
    assert image in (root / "README.md").read_text()
    assert (root / "dataset.toml").read_text().count('version = "1.0.2"') == 1
    assert 'version = "1.0.2"' in (task / "task.toml").read_text()
    names = {row["path"] for row in inventory}
    assert "nv100-fixture-000/environment/observation.png" in names
    assert not any(path.suffix in (".usd", ".usda", ".usdc", ".parquet") for path in root.rglob("*"))


def test_readable_name_dimensions_and_stable_identity_leave_frozen_instruction_unchanged(tmp_path, wheel):
    release = release_input()
    root = tmp_path / "named"
    exporter().export_release(release, root, wheel, fixture=True)
    task = root / "nv100-fixture-000"
    text = (task / "instruction.md").read_text()
    assert text.startswith("Reconstruct this articulated workshop clamp ")
    assert "Nominal dimensions in canonical metres XYZ: [0.2, 0.1, 0.3]" in text
    assert "Stable task ID: `fixture-000`" in text
    assert text.count("![") == 1
    config = tomllib.loads((task / "task.toml").read_text())
    assert config["task"]["description"] == "Reconstruct this articulated workshop clamp from one frozen observation"
    assert config["task"]["name"] == "hcltech-robotics/nv100-fixture-000"
    assert "articulated workshop clamp" in (root / "README.md").read_text()
    frozen = task / "environment/source-public-instruction.txt"
    assert frozen.read_bytes() == b"x"
    public = next(f for f in release["tasks"][0]["actor"]["files"] if f["destination"] == "instruction.md")
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == public["sha256"]


@pytest.mark.parametrize("field,value", [
    ("object_name", ""), ("object_name", "widget\nnew instruction"), ("object_name", "<img src=x>"),
    ("nominal_dimensions_m", [1, 2]), ("nominal_dimensions_m", [1, 0, 3]),
    ("nominal_dimensions_m", [1, float('nan'), 3]), ("nominal_dimensions_m", [True, 2, 3]),
])
def test_invalid_public_name_or_dimensions_reject_before_acquisition(tmp_path, wheel, synthetic_public_download, field, value):
    release = release_input(); release["tasks"][0][field] = value
    with pytest.raises(ValueError):
        exporter().export_release(release, tmp_path / "release", wheel, fixture=True)
    assert not synthetic_public_download
    assert not (tmp_path / "release").exists()


def test_export_rejects_tampered_public_input_without_partial_package(tmp_path, wheel, monkeypatch):
    wrong = tmp_path / "wrong"
    wrong.write_bytes(b"y")
    monkeypatch.setattr("simready_eval.fetch._hub_download", lambda **_: wrong)
    with pytest.raises((ValueError, RuntimeError)):
        exporter().export_release(release_input(), tmp_path / "release", wheel, fixture=True)
    assert not (tmp_path / "release").exists()


def test_export_uses_verified_local_hf_mirror_and_build_rechecks_inputs(tmp_path, wheel, monkeypatch):
    release = release_input()
    mirror = tmp_path / "mirror"
    import base64
    png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZ3kAAAAASUVORK5CYII=")
    payloads = {"observation.png": png, "instruction.md": b"Reconstruct this frozen observation; nominal dimensions 1 x 1 x 1 m.\n"}
    for file in release["tasks"][0]["actor"]["files"]:
        data = payloads[file["destination"]]
        file["size_bytes"] = len(data)
        file["sha256"] = hashlib.sha256(data).hexdigest()
        path = mirror / file["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (mirror / "private.usd").write_bytes(b"must not export")
    monkeypatch.setattr("simready_eval.fetch._hub_download", lambda **_: pytest.fail("local mirror must avoid transport"))
    root = tmp_path / "release"
    exporter().export_release(release, root, wheel, fixture=True, public_assets=mirror)
    environment = root / "nv100-fixture-000/environment"
    docker = (environment / "Dockerfile").read_text()
    assert "COPY observation.png instruction.md /task/" in docker
    assert "--verify-only" in docker
    assert "--cache /tmp/actor-cache" not in docker
    # Execute the generated COPY/verification steps without a container or HF.
    actual = _run_actor_public_build(environment, tmp_path / "build")
    assert {path.name for path in actual.iterdir()} == {"observation.png", "instruction.md"}
    assert (actual / "observation.png").read_bytes() == png
    assert (actual / "instruction.md").read_text().startswith("Reconstruct this articulated workshop clamp ")
    assert payloads["instruction.md"] in (actual / "instruction.md").read_bytes()
    assert not (tmp_path / "build/frozen-public").exists()
    assert not (root / "private.usd").exists()
    original = (environment / "source-public-instruction.txt").read_bytes()
    changed = bytes([original[0] ^ 1]) + original[1:]
    (environment / "source-public-instruction.txt").write_bytes(changed)
    with pytest.raises(subprocess.CalledProcessError) as failed:
        _run_actor_public_build(environment, tmp_path / "bad-build")
    assert "instruction.md" in failed.value.stderr
    assert hashlib.sha256(original).hexdigest() in failed.value.stderr
    assert hashlib.sha256(changed).hexdigest() in failed.value.stderr
    assert not (tmp_path / "bad-build/task").exists(), "the frozen brief must be checked before runtime installation"


def test_fixture_export_parses_with_standard_harbor_and_hashes_all_generated_files(tmp_path, wheel):
    from harbor.models.task.task import Task
    from harbor.models.dataset.manifest import DatasetManifest
    from harbor.publisher.packager import Packager
    output = tmp_path / "export"
    inventory = exporter().export_release(release_input(), output, wheel, fixture=True)
    task = Task(output / "nv100-fixture-000")
    config = task.config
    assert config.schema_version == "1.4"
    assert config.task.name == "hcltech-robotics/nv100-fixture-000"
    assert config.agent.timeout_sec == 231000
    assert config.verifier.timeout_sec == 600
    assert config.verifier.environment_mode.value == "separate"
    assert config.environment.network_mode.value == "public"
    assert config.agent.network_mode.value == "allowlist"
    assert config.agent.allowed_hosts == []
    assert config.environment.docker_image is None
    assert config.verifier.environment.network_mode.value == "no-network"
    assert config.verifier.network_mode.value == "no-network"
    assert [(item.source, item.destination) for item in config.artifacts] == [("/output/submission", "candidate")]
    dataset = DatasetManifest.from_toml_file(output / "dataset.toml")
    assert dataset.schema_version == "1.0"
    assert tomllib.loads((output / "dataset.toml").read_text())["dataset"]["name"] == "hcltech-robotics/simreadybench"
    expected_digest, _ = Packager.compute_content_hash(task.task_dir)
    assert dataset.tasks[0].digest == "sha256:" + expected_digest
    assert dataset.tasks[0].name == config.task.name
    assert json.loads((output / "release.json").read_text())["export"]["fixture"] is True
    assert json.loads((output / "release.json").read_text())["export"] == {"fixture": True}
    assert json.loads((output / "inventory.json").read_text()) == inventory
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    assert actual == {entry["path"] for entry in inventory} | {"inventory.json"}
    for entry in inventory:
        data = (output / entry["path"]).read_bytes()
        assert entry["size_bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert all(not path.is_symlink() for path in output.rglob("*"))


def test_frozen_nv100_task_ids_keep_their_declared_hub_name(tmp_path, wheel):
    from harbor.models.task.task import Task
    task_id = 'nv100_e006be66e4cc28f9'
    release = release_input()
    task = release['tasks'][0]
    task['id'] = task_id
    task['actor'] = role_manifest(task_id, 'actor', ['observation.png', 'instruction.md'])
    task['verifier'] = role_manifest(task_id, 'verifier', [
        'reference/asset.usd', 'reference/eligibility.json', 'comparison/cameras.json',
        *[f'comparison/{view}{suffix}.png' for view in VIEWS for suffix in ('', '-mask')]])
    root = tmp_path / 'export'
    exporter().export_release(release, root, wheel, fixture=True)
    exported = Task(root / task_id)
    assert exported.config.task.name == 'hcltech-robotics/' + task_id
    manifest = tomllib.loads((root / 'dataset.toml').read_text())
    assert manifest['tasks'][0]['name'] == exported.config.task.name


def test_normalized_hub_name_collisions_are_rejected_before_writing(tmp_path, wheel):
    release = release_input(2)
    second = release['tasks'][1]
    task_id = 'nv100-fixture-000'
    second['id'] = task_id
    for role in ('actor', 'verifier'):
        second[role] = role_manifest(task_id, role,
            [row['destination'] for row in second[role]['files']])
    root = tmp_path / 'export'
    with pytest.raises(ValueError, match='unique Hub names'):
        exporter().export_release(release, root, wheel, fixture=True)
    assert not root.exists()


def test_build_contexts_have_role_scoped_fetch_and_no_target_data_in_feedback(tmp_path, wheel):
    root = tmp_path / "export"
    exporter().export_release(release_input(), root, wheel, fixture=True)
    task = root / "nv100-fixture-000"
    environment, tests = task / "environment", task / "tests"
    for recipe in (environment / 'Dockerfile', environment / 'Dockerfile.feedback', tests / 'Dockerfile'):
        text = recipe.read_text()
        assert 'OPENBLAS_NUM_THREADS=1' in text
        assert 'OMP_NUM_THREADS=1' in text
        assert 'PXR_WORK_THREAD_LIMIT=2' in text
    actor = json.loads((environment / "actor-assets.json").read_text())
    verifier = json.loads((tests / "verifier-assets.json").read_text())
    assert actor["role"] == "actor"
    assert verifier["role"] == "verifier"
    assert not (environment / "verifier-assets.json").exists()
    for directory in (environment, tests):
        assert (directory / wheel.name).read_bytes() == wheel.read_bytes()
    actor_docker = (environment / "Dockerfile").read_text()
    verifier_docker = (tests / "Dockerfile").read_text()
    feedback_docker = (environment / "Dockerfile.feedback").read_text()
    assert actor_docker.startswith("FROM " + BASE + "\n")
    assert verifier_docker.startswith("FROM " + BASE + "\n")
    assert "simready_eval.fetch" in actor_docker and "--destination /task --verify-only" in actor_docker
    assert "simready_eval.fetch" in verifier_docker and "--destination /fixtures" in verifier_docker
    assert "COPY test.sh /tests/test.sh" in verifier_docker
    assert "USER 1000" in actor_docker
    assert "USER 0" in feedback_docker
    assert "simready_eval.fetch" not in feedback_docker
    assert "actor-assets" not in feedback_docker and "verifier-assets" not in feedback_docker
    assert "COPY ." not in "\n".join([actor_docker, verifier_docker, feedback_docker])
    assert "simready_eval.cli evaluate" in (tests / "test.sh").read_text()
    assert "--submission /output/submission --fixtures /fixtures --config /tests/evaluator.json --output /logs/verifier" in (tests / "test.sh").read_text()
    evaluator = json.loads((tests / "evaluator.json").read_text())
    assert evaluator == {"schema": "simready-evaluator-config-v1", "task_id": "fixture-000", "entrypoint": "asset.usd", "reference": "reference/asset.usd", "cameras": "comparison/cameras.json", "eligibility": "reference/eligibility.json", "fidelity_settings": {"enabled": True}, "final_seconds": 600}
    compose = json.loads((environment / "docker-compose.yaml").read_text())
    assert "networks" not in compose
    assert set(compose["services"]) == {"main", "feedback"}
    assert all("networks" not in value and "network_mode" not in value for value in compose["services"].values())
    feedback = compose["services"]["feedback"]
    assert feedback["cap_drop"] == ["ALL"]
    assert set(feedback["cap_add"]) == {"SETUID", "SETGID", "CHOWN"}
    assert feedback["user"] == "0"
    assert "ports" not in feedback
    assert feedback["volumes"] == ["feedback-ledger:/state"]
    assert compose["volumes"] == {"feedback-ledger": {}}
    assert compose["services"]["main"]["depends_on"]["feedback"]["condition"] == "service_healthy"
    assert "/health" in " ".join(feedback["healthcheck"]["test"])
    assert "127.0.0.1:8080" in (task / "instruction.md").read_text()
    assert [path.relative_to(root).as_posix() for path in root.rglob("*.png")] == ["nv100-fixture-000/environment/observation.png"]
    assert not any(path.suffix in (".usd", ".usda", ".usdc", ".parquet") for path in root.rglob("*"))


def test_export_is_deterministic_for_identical_inputs(tmp_path, wheel):
    first = exporter().export_release(release_input(2), tmp_path / "first", wheel, fixture=True)
    second = exporter().export_release(release_input(2), tmp_path / "second", wheel, fixture=True)
    assert first == second


def test_release_requires_exact_100_tasks(tmp_path, wheel):
    with pytest.raises(ValueError, match="100"):
        exporter().export_release(release_input(), tmp_path / "export", wheel)
    assert not (tmp_path / "export").exists()


def test_release_100_tasks_uses_the_supplied_roster_only(tmp_path, wheel):
    output = tmp_path / "export"
    exporter().export_release(release_input(100), output, wheel)
    dataset = tomllib.loads((output / "dataset.toml").read_text())
    assert len(dataset["tasks"]) == 100
    assert {item["name"] for item in dataset["tasks"]} == {f"hcltech-robotics/nv100-fixture-{index:03}" for index in range(100)}


@pytest.mark.parametrize("mutation", ["other_repo", "mutable_revision", "task_mismatch", "role_mismatch", "missing_mask", "missing_camera", "missing_eligibility", "missing_reference", "unsafe_reference", "unsafe_entrypoint", "duplicate_view", "unsafe_view", "wrong_source", "wrong_source_revision", "duplicate_task", "bad_budget", "unknown_task_field"])
def test_invalid_release_fails_before_creating_output(tmp_path, wheel, mutation):
    release = release_input()
    task = release["tasks"][0]
    if mutation == "other_repo": release["repo_id"] = "wrong/repo"
    elif mutation == "mutable_revision": release["revision"] = "main"
    elif mutation == "task_mismatch": task["actor"]["task_id"] = "other"
    elif mutation == "role_mismatch": task["actor"]["role"] = "verifier"
    elif mutation.startswith("missing_"):
        suffix = {"missing_mask": "isometric-mask.png", "missing_camera": "cameras.json", "missing_eligibility": "eligibility.json", "missing_reference": "asset.usd"}[mutation]
        task["verifier"]["files"] = [item for item in task["verifier"]["files"] if not item["destination"].endswith(suffix)]
    elif mutation == "unsafe_reference": task["reference"] = "../secret.usd"
    elif mutation == "unsafe_entrypoint": task["entrypoint"] = "../asset.usd"
    elif mutation == "duplicate_view": task["views"][1] = task["views"][0]
    elif mutation == "unsafe_view": task["views"][0] = "../front"
    elif mutation == "wrong_source": release["source_collection"]["repo_id"] = "another/collection"
    elif mutation == "wrong_source_revision": release["source_collection"]["revision"] = "3" * 40
    elif mutation == "duplicate_task": release["tasks"].append(copy.deepcopy(task))
    elif mutation == "bad_budget": task["budgets"]["agent_seconds"] = -1
    else: task["private_prompt"] = "must never export"
    with pytest.raises(ValueError):
        exporter().export_release(release, tmp_path / "export", wheel, fixture=True)
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("member,content", [("simready_bench/private.py", ""), ("results/rows.parquet", "data"), ("simready_eval/legacy.py", "import simready_bench.engine\n"), ("simready_eval/adapter.py", "from harbor.models import Task\n"), ("simready_eval/private/source.py", ""), ("../escape", ""), ("simready_eval/unsafe.py", "__import__('simready_bench.engine')\n")])
def test_wheel_rejects_legacy_private_and_traversal_payload(tmp_path, wheel, member, content):
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr(member, content)
    with pytest.raises(ValueError):
        exporter().export_release(release_input(), tmp_path / "export", wheel, fixture=True)
    assert not (tmp_path / "export").exists()


def test_output_must_be_fresh_and_outside_any_git_checkout(tmp_path, wheel):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError):
        exporter().export_release(release_input(), existing, wheel, fixture=True)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: /arbitrary/worktree\n")
    with pytest.raises(ValueError):
        exporter().export_release(release_input(), checkout / "export", wheel, fixture=True)
    assert not (checkout / "export").exists()


def test_export_cli_produces_reviewable_fixture_inventory(tmp_path, wheel):
    manifest = tmp_path / "release-input.json"
    manifest.write_text(json.dumps(release_input()))
    public = tmp_path / "public"
    for file in release_input()["tasks"][0]["actor"]["files"]:
        source = public / file["path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x")
    process = subprocess.run([sys.executable, "-m", "simready_eval.export", "--release", str(manifest), "--output", str(tmp_path / "export"), "--wheel", str(wheel), "--fixture", "--public-assets", str(public)], text=True, capture_output=True)
    assert process.returncode == 0, process.stderr
    assert isinstance(json.loads(process.stdout), list)
    assert (tmp_path / "export/inventory.json").is_file()


def test_aliased_dynamic_private_import_cannot_enter_public_wheel(tmp_path, wheel):
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("simready_eval/unsafe_alias.py", "from importlib import import_module as load\nload('simready_bench.engine')\n")
    with pytest.raises(ValueError):
        exporter().export_release(release_input(), tmp_path / "export", wheel, fixture=True)


def test_view_rgb_and_mask_paths_cannot_collide(tmp_path, wheel):
    release = release_input()
    task = release["tasks"][0]
    task["views"] = ["front", "front-mask", "side", "top"]
    destinations = ["reference/asset.usd", "reference/eligibility.json", "comparison/cameras.json",
                    *sorted({f"comparison/{view}{suffix}.png" for view in task["views"] for suffix in ("", "-mask")})]
    task["verifier"] = role_manifest(task["id"], "verifier", destinations)
    with pytest.raises(ValueError):
        exporter().export_release(release, tmp_path / "export", wheel, fixture=True)


def test_blender_is_verifier_only_and_explicitly_forbidden_for_authoring(tmp_path, wheel):
    output = tmp_path / "export"
    exporter().export_release(release_input(), output, wheel, fixture=True)
    task = output / "nv100-fixture-000"
    actor = (task / "environment/Dockerfile").read_text()
    feedback = (task / "environment/Dockerfile.feedback").read_text()
    verifier = (task / "tests/Dockerfile").read_text()
    assert "blender" not in actor.lower()
    assert "blender" not in feedback.lower()
    assert "apt-get install -y --no-install-recommends blender=$BLENDER_PACKAGE_VERSION" in verifier
    assert "Do not use Blender for authoring; Blender is used only by the verifier renderer." in (task / "instruction.md").read_text()


@pytest.mark.parametrize("role,relative", [("actor", "observation.png"), ("verifier", "reference/asset.usd")])
def test_setup_handoff_makes_fetched_tree_world_readable_and_immutable(tmp_path, wheel, role, relative):
    import shlex
    import stat
    from simready_eval.fetch import fetch_assets

    release = release_input()
    output = tmp_path / "export"
    exporter().export_release(release, output, wheel, fixture=True)
    source = tmp_path / "source"
    source.write_bytes(b"x")
    staged = tmp_path / "staged"
    fetch_assets(release["tasks"][0][role], staged, cache_dir=tmp_path / "cache", downloader=lambda **_: source)
    assert stat.S_IMODE(staged.stat().st_mode) == 0o700
    assert stat.S_IMODE((staged / relative).stat().st_mode) == 0o600
    directory = "environment" if role == "actor" else "tests"
    dockerfile = (output / "nv100-fixture-000" / directory / "Dockerfile").read_text()
    target = "/task" if role == "actor" else "/fixtures"
    fetch_line = next(line for line in dockerfile.splitlines() if line.startswith("RUN python -m simready_eval.fetch") and "--destination " + target + " " in line)
    # Execute the generated handoff commands against real fetched files. Cache
    # permissions remain private; only the final immutable tree becomes public.
    target = "/task" if role == "actor" else "/fixtures"
    commands = [command.replace(target, shlex.quote(str(staged))) for command in fetch_line.split(" && ")
                if command.startswith("chmod ") and target in command]
    if role == "verifier":
        commands = [command.replace(" /tests", "") for command in commands]
    subprocess.run(["sh", "-c", " && ".join(commands)], check=True)
    assert stat.S_IMODE(staged.stat().st_mode) == 0o555
    assert stat.S_IMODE((staged / relative).stat().st_mode) == 0o444
    assert all(not path.stat().st_mode & 0o222 for path in staged.rglob("*"))
    cached = next((tmp_path / "cache").rglob(Path(relative).name))
    assert stat.S_IMODE(cached.stat().st_mode) == 0o600


@pytest.mark.parametrize("entrypoint", ["model.usda", "nested/asset.usd"])
def test_export_requires_canonical_candidate_asset_entrypoint(tmp_path, wheel, entrypoint):
    release = release_input()
    release["tasks"][0]["entrypoint"] = entrypoint
    with pytest.raises(ValueError, match="asset.usd"):
        exporter().export_release(release, tmp_path / "export", wheel, fixture=True)


def test_reference_entrypoint_can_remain_nested(tmp_path, wheel):
    release = release_input()
    task = release["tasks"][0]
    task["reference"] = "reference/nested/source.usda"
    file = next(file for file in task["verifier"]["files"] if file["destination"] == "reference/asset.usd")
    file["path"] = file["path"].removesuffix("asset.usd") + "nested/source.usda"
    file["destination"] = task["reference"]
    exporter().export_release(release, tmp_path / "export", wheel, fixture=True)
    config = json.loads((tmp_path / "export/nv100-fixture-000/tests/evaluator.json").read_text())
    assert config["reference"] == "reference/nested/source.usda"
    assert config["entrypoint"] == "asset.usd"


def test_exported_role_locks_bound_transitive_runtime_resolution(tmp_path, wheel):
    import re
    output = tmp_path / "export"
    exporter().export_release(release_input(), output, wheel, fixture=True)
    task = output / "nv100-fixture-000"
    closures = {}
    for role, directory, docker_name in [("actor", "environment", "Dockerfile"), ("feedback", "environment", "Dockerfile.feedback"), ("verifier", "tests", "Dockerfile")]:
        lock = task / directory / (role + ".requirements.txt")
        assert lock.is_file(), f"{role} transitive dependency lock is missing"
        lines = [line for line in lock.read_text().splitlines() if line and not line.startswith("#")]
        assert lines and all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+ --hash=sha256:[a-f0-9]{64}", line) for line in lines)
        closures[role] = {line.split("==", 1)[0].lower().replace("_", "-") for line in lines}
        docker = (task / directory / docker_name).read_text()
        install = next(line for line in docker.splitlines() if line.startswith("RUN python -m pip install"))
        assert "--require-hashes --no-deps" in install
        assert f"-r /opt/locks/{role}.requirements.txt" in install
        assert "&& python -m pip install --no-cache-dir --no-deps" in install
        assert "simreadybench-eval" not in closures[role]
    assert len(closures["actor"]) == 18
    assert len(closures["feedback"]) == 13
    assert closures["verifier"] == closures["actor"] | closures["feedback"]
    assert "huggingface-hub" in closures["actor"]
    assert "simready-validate" not in closures["actor"]
    assert "simready-validate" in closures["feedback"]
    assert "huggingface-hub" not in closures["feedback"]


def test_named_public_and_runtime_instructions_keep_raw_frozen_brief_distinct(tmp_path, wheel):
    release = release_input()
    source_actor = copy.deepcopy(release["tasks"][0]["actor"])
    output = tmp_path / "export"
    exporter().export_release(release, output, wheel, fixture=True)
    task = output / "nv100-fixture-000"
    environment = task / "environment"
    for path in (task / "instruction.md", environment / "instruction.md"):
        text = path.read_text()
        assert text.startswith("Reconstruct this articulated workshop clamp ")
        assert "[0.2, 0.1, 0.3]" in text
        assert text.count("![") == 1
    assert (environment / "source-public-instruction.txt").read_bytes() == b"x"
    assert json.loads((environment / "actor-assets.json").read_text()) == source_actor
    runtime = json.loads((environment / "actor-runtime.json").read_text())
    assert runtime["schema"] == "simready-actor-runtime-v1"
    assert runtime["instruction_origin"] == "derived-public-wrapper"
    assert runtime["frozen_actor_manifest_sha256"] == hashlib.sha256(json.dumps(source_actor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert {file["destination"] for file in runtime["files"]} == {"observation.png", "instruction.md"}
    for file in runtime["files"]:
        data = (environment / file["destination"]).read_bytes()
        assert file["size_bytes"] == len(data)
        assert file["sha256"] == hashlib.sha256(data).hexdigest()
    assert [path.name for path in environment.glob("*.png")] == ["observation.png"]
    docker = (environment / "Dockerfile").read_text()
    assert "COPY source-public-instruction.txt /opt/setup/frozen-public/instruction.md" in docker
    frozen_check = "--manifest /opt/setup/actor-assets.json --destination /opt/setup/frozen-public --verify-only"
    runtime_install = "COPY observation.png instruction.md /task/"
    runtime_check = "--manifest /opt/setup/actor-runtime.json --destination /task --verify-only"
    assert docker.index(frozen_check) < docker.index(runtime_install) < docker.index(runtime_check)


def test_named_wrappers_retain_original_public_constraints_verbatim(tmp_path, wheel):
    release = release_input()
    original = (b"Frozen public protocol.\r\nSubmit asset.usd and all dependencies.\r\n"
                b"Maximum 1 GiB and 10000 files/directories.\r\n"
                b"Do not use Blender for authoring.\r\nUse at most five feedback calls.\r\n")
    mirror = tmp_path / "public"
    for file in release["tasks"][0]["actor"]["files"]:
        data = original if file["destination"] == "instruction.md" else b"x"
        file["size_bytes"], file["sha256"] = len(data), hashlib.sha256(data).hexdigest()
        source = mirror / file["path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(data)
    root = tmp_path / "release"
    exporter().export_release(release, root, wheel, fixture=True, public_assets=mirror)
    task = root / "nv100-fixture-000"
    for path in (task / "instruction.md", task / "environment/instruction.md"):
        wrapper = path.read_bytes()
        assert wrapper.startswith(b"Reconstruct this articulated workshop clamp ")
        assert b"[0.2, 0.1, 0.3]" in wrapper
        assert original in wrapper, "the named brief must preserve the original public constraints byte-for-byte"
        assert wrapper.count(original) == 1
    assert (task / "environment/source-public-instruction.txt").read_bytes() == original
