"""Opt-in real Docker checks, using independently installed evaluator images."""

import json
import io
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import uuid

import pytest

pytestmark = pytest.mark.container


def docker(*args, check=True, timeout=120):
    env = {**os.environ, "DOCKER_CONFIG": "/tmp/simready-native-docker"}
    process = subprocess.run(["docker", *map(str, args)], capture_output=True,
                             text=True, timeout=timeout, env=env)
    if check:
        assert process.returncode == 0, process.stdout + process.stderr
    return process


def copy_into(container, source, destination):
    """Load declared files through exec into a writable tmpfs on read-only roots."""
    source = Path(source)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        if source.is_dir():
            for child in sorted(source.iterdir()):
                archive.add(child, arcname=child.name)
        else:
            archive.add(source, arcname=source.name)
    result = subprocess.run(["docker", "exec", "-i", "--user", "0", container,
                             "tar", "--no-same-owner", "--no-same-permissions", "-xf", "-", "-C", destination],
                            input=stream.getvalue(), capture_output=True,
                            env={**os.environ, "DOCKER_CONFIG": "/tmp/simready-native-docker"})
    assert result.returncode == 0, result.stderr.decode()
    docker("exec", "--user", "0", container, "chmod", "-R", "a+rX", destination)


def test_feedback_real_root_worker_quota_restart_and_actor_tools(tmp_path):
    from runtime_fixtures import articulated_box

    feedback = os.environ.get("SIMREADY_FEEDBACK_IMAGE", "simready-native-feedback:local")
    actor = os.environ.get("SIMREADY_ACTOR_IMAGE", "simready-native-actor:local")
    for image in (feedback, actor):
        docker("image", "inspect", image)
    package = tmp_path / "submission"
    articulated_box(package / "asset.usd")
    name, volume = "native-feedback-" + uuid.uuid4().hex[:10], "native-ledger-" + uuid.uuid4().hex[:10]
    try:
        docker("run", "-d", "--name", name, "--network=none", "--read-only",
               "--cap-drop=ALL", "--cap-add=CHOWN", "--cap-add=SETUID", "--cap-add=SETGID",
               "--security-opt=no-new-privileges", "--tmpfs", "/tmp:rw,mode=1777",
               "-v", volume + ":/state", feedback,
               "python", "-m", "simready_eval.feedback", "serve", "--state", "/state")
        docker("exec", name, "mkdir", "/tmp/submission")
        copy_into(name, package, "/tmp/submission")
        for _ in range(30):
            ready = docker("exec", name, "python", "-c",
                           "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/health').read().decode())",
                           check=False)
            if ready.returncode == 0:
                break
            time.sleep(.1)
        assert ready.returncode == 0, docker("logs", name).stdout
        command = ("exec", "--user", "1000:1000", name, "python", "-m", "simready_eval.feedback", "validate_candidate",
                   "--package", "/tmp/submission", "--url", "http://127.0.0.1:8080", "--request-id")
        results = []
        for number in range(5):
            response = docker(*command, "call-" + str(number))
            body = json.loads(response.stdout)
            assert body["result"]["simready"]["applicable_count"] == 46
            assert body["result"]["simready"]["score_percent"] is not None, body
            results.append(body)
        assert docker(*command, "sixth", check=False).returncode != 0
        docker("exec", "--user", "65534:65534", name, "sh", "-c",
               "test ! -r /state/feedback.sqlite3 && test ! -w /state")
        docker("restart", name)
        docker("exec", name, "mkdir", "/tmp/submission")
        copy_into(name, package, "/tmp/submission")
        for _ in range(30):
            cached = docker(*command, "call-0", check=False)
            if cached.returncode == 0:
                break
            time.sleep(.1)
        assert cached.returncode == 0, cached.stderr + docker("logs", name).stderr
        assert json.loads(cached.stdout) == results[0]
        assert docker(*command, "seventh", check=False).returncode != 0
        docker("run", "--rm", "--network=none", actor, "python", "-c",
               "import importlib.util,shutil; assert shutil.which('blender') is None; assert importlib.util.find_spec('simready') is None")
    finally:
        docker("rm", "-f", name, check=False)
        docker("volume", "rm", volume, check=False)


def test_standard_compose_actor_uses_candidate_only_feedback(tmp_path):
    """Exercise the exported service definition; full Harbor needs NFT FIB."""
    from PIL import Image
    from runtime_fixtures import articulated_box
    from simready_eval.export import _compose

    actor = tmp_path / "actor"
    actor.mkdir()
    articulated_box(actor / "asset.usd")
    Image.new("RGB", (16, 16), (100, 40, 10)).save(actor / "observation.png")
    (actor / "instruction.md").write_text("Original neutral integration fixture.\n")
    (actor / "Dockerfile").write_text(
        "FROM simready-native-actor:local\nUSER 0\n"
        "COPY observation.png instruction.md /task/\n"
        "COPY asset.usd /output/submission/asset.usd\n"
        "RUN chmod -R a+rX /task /output/submission\nUSER 1000\n")
    (tmp_path / "Dockerfile.feedback").write_text(
        "FROM simready-native-feedback:local\n"
        'ENTRYPOINT ["python","-m","simready_eval.feedback","serve","--state","/state","--host","0.0.0.0","--port","8080"]\n')
    compose = _compose()
    compose["services"]["main"].update(build={"context": str(actor)},
        network_mode="service:namespace")
    compose["services"]["feedback"].update(
        build={"context": str(tmp_path), "dockerfile": "Dockerfile.feedback"},
        network_mode="service:namespace", depends_on={"namespace": {"condition": "service_started"}})
    # Match Harbor's shared namespace topology while leaving its unavailable
    # dynamic network enforcement unclaimed. The anchor has no external network.
    compose["services"]["namespace"] = {
        "image": "alpine:3.23.4@sha256:5b10f432ef3da1b8d4c7eb6c487f2f5a8f096bc91145e68878dd4a5019afde11",
        "network_mode": "none", "command": ["sleep", "infinity"]}
    manifest = tmp_path / "compose.json"
    manifest.write_text(json.dumps(compose))
    command = ("compose", "-f", str(manifest), "-p", "native-compose-" + uuid.uuid4().hex[:8])
    try:
        docker(*command, "up", "--build", "--wait", "--wait-timeout", "90", timeout=150)
        docker(*command, "exec", "-T", "main", "sh", "-c",
               "test -r /task/observation.png && test ! -e /fixtures && test ! -e /tests")
        response = docker(*command, "exec", "-T", "main", "python", "-m",
            "simready_eval.feedback", "validate_candidate", "--package", "/output/submission",
            "--request-id", "compose-1", "--url", "http://127.0.0.1:8080")
        body = json.loads(response.stdout)
        assert body["result"]["simready"]["score_percent"] is not None, body
        assert body["result"]["simready"]["applicable_count"] == 46
    finally:
        docker(*command, "down", "--volumes", "--remove-orphans", check=False)


def test_fresh_offline_verifier_grades_declared_articulated_submission(tmp_path):
    from runtime_fixtures import articulated_box, cameras
    from simready_eval import correspondence, render
    from simready_eval.usd import observe

    image = os.environ.get("SIMREADY_VERIFIER_IMAGE", "simready-native-verifier:local")
    docker("image", "inspect", image)
    fixtures = tmp_path / "fixtures"
    ref = articulated_box(fixtures / "reference/asset.usd")
    comparison = fixtures / "comparison"
    comparison.mkdir()
    rig = cameras()
    (comparison / "cameras.json").write_text(json.dumps(rig))
    rendered = render.render_asset(ref, rig, tmp_path / "reference-renders")
    assert rendered["status"] == "measured", rendered
    for row in rendered["images"]:
        for field, suffix in (("file", ".png"), ("mask_file", "-mask.png")):
            shutil.copyfile(row[field], comparison / (row["camera_id"] + suffix))
    settings = {"enabled": True, "samples": 128}
    profile = correspondence.reference_profile(observe(ref), settings)
    import hashlib
    files = [{"path": path.relative_to(fixtures).as_posix(), "size_bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(fixtures.rglob("*")) if path.is_file()]
    (fixtures / "reference/eligibility.json").write_text(json.dumps({
        "schema": "simready-reference-eligibility-v1", "files": files,
        "profile": profile, "profile_sha256": correspondence.digest(profile),
        "render_recipe_sha256": rendered["recipe_sha256"],
        "render_camera_sha256": rendered["camera_sha256"],
        "render_bridge_code_sha256": rendered["bridge_code_sha256"],
        "renderer_binary_sha256": rendered["renderer_binary_sha256"]}))
    config = tmp_path / "evaluator.json"
    config.write_text(json.dumps({"schema": "simready-evaluator-config-v1", "task_id": "neutral-hinge",
        "entrypoint": "asset.usd", "reference": "reference/asset.usd",
        "cameras": "comparison/cameras.json", "eligibility": "reference/eligibility.json",
        "fidelity_settings": settings, "final_seconds": 600}))
    submission = tmp_path / "submission"
    submission.mkdir()
    shutil.copyfile(ref, submission / "asset.usd")
    name = "native-verifier-" + uuid.uuid4().hex[:10]
    try:
        docker("run", "-d", "--name", name, "--network=none", "--read-only", "--user", "1000:1000",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--tmpfs", "/tmp:rw,mode=1777",
               "--tmpfs", "/logs:rw,mode=1777", "--tmpfs", "/fixtures:rw,mode=1777",
               "--tmpfs", "/tests:rw,mode=1777", "--tmpfs", "/output/submission:rw,mode=1777",
               image, "sleep", "infinity")
        copy_into(name, fixtures, "/fixtures")
        copy_into(name, config, "/tests")
        copy_into(name, submission, "/output/submission")
        evaluated = docker("exec", name, "python", "-m", "simready_eval.cli", "evaluate",
               "--submission", "/output/submission", "--fixtures", "/fixtures", "--config", "/tests/evaluator.json",
               "--output", "/logs/verifier", timeout=180)
        (tmp_path / "evaluator-stdout.txt").write_text(evaluated.stdout)
        (tmp_path / "evaluator-stderr.txt").write_text(evaluated.stderr)
        present = docker("exec", name, "ls", "-laR", "/logs", check=False)
        assert docker("exec", name, "test", "-r", "/logs/verifier/report.json", check=False).returncode == 0, evaluated.stdout + evaluated.stderr + present.stdout + present.stderr
        saved = tmp_path / "verifier"
        saved.mkdir()
        for filename in ("report.json", "reward.json"):
            (saved / filename).write_text(docker("exec", name, "cat", "/logs/verifier/" + filename).stdout)
        report = json.loads((saved / "report.json").read_text())
        reward = json.loads((saved / "reward.json").read_text())
        assert report["scores"]["formal"]["score_percent"] is not None, report
        for key in ("geometry", "appearance", "visual", "kinematic"):
            assert report["scores"][key]["score_percent"] == pytest.approx(100), report
        assert reward["evaluation_complete"] == 1
        assert reward["kinematic_known_credit"] == 1
        assert report["candidate"]["files"]
    finally:
        docker("rm", "-f", name, check=False)
