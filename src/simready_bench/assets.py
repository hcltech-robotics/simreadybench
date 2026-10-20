"Generate Harbor reconstruction tasks from reference USD packages."

from pathlib import Path
import hashlib, json, subprocess
from simready_bench.preparation import confined_work_dir, _reference_input, write_json

CONTRACT = """# Reconstruction contract

Reconstruct the supplied object at /output/asset.usd using Python/PXR. Include
its local USD and texture dependencies and place authoring scripts in /tmp.
Read the supplied dimensions, image and scoring-policy.json.

The evaluation reports three independent scores:

- Formal fidelity: NVIDIA SimReady-Foundation Robotics-Prop 4.0.0. Each of its
  46 mandatory requirements counts once, with 95 * passed / 46 points and five
  further points when all pass. The 51 optional requirements are reported separately.
- Visual fidelity: reference-scale shape, colours, textures and surface appearance.
- Kinematic fidelity: the object's connected rigid parts, joint types, hinge or
  sliding axes, attachment locations and allowed ranges of motion.

Identify the object's parts and their mechanical relationships before authoring.
Keep parts that move relative to one another as distinct rigid bodies and author
the corresponding UsdPhysics joints. Specify each joint's connected bodies,
local positions, local orientations, axis and appropriate motion limits. For
example, folding eyewear needs hinged temples rather than one rigid mesh.
Use fixed joints for genuine fixed attachments and keep genuinely rigid objects
rigid. Base this structure on the supplied observations and the object's function.

Check that joint frames meet at each attachment, that movable parts can travel
through their intended range and that the bodies and joints have the appropriate
enabled and kinematic states. Preserve the observed shape and materials while
constructing this mechanical structure.

Use validate_candidate to inspect the NVIDIA findings and repair the candidate.
The final submission freezes the authored files for evaluation.
"""
TOOLS = '# Authoring tools\n\nget_task_thumbnail returns the reference image. validate_candidate accepts {"request_id":"check-1"} and returns the current score, remaining validation calls and nvidia_validator_report.file_path. Search that JSON file with rg for a requirement ID or candidate prim path. get_validation_report reads a selected range of the same report.\n\nUse a new request ID after each repair. submit_asset accepts {"request_id":"final"} and stops authoring. The model has {inference_requests} inference requests, with provider-native output limits. Captures appear in the final report.\n'


def _metadata(source, package, runtime, output):
    code = "import json,resource,sys\nresource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))\nfrom pathlib import Path\nfrom simready_bench.reference import check_reference\nfrom pxr import Usd,UsdGeom\np=json.load(sys.stdin);result=check_reference(p['source'],package_root=p['package'])\nif not result['accepted']:raise RuntimeError('Source does not pass the pinned NVIDIA profile')\ns=Usd.Stage.Open(p['source']);box=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False).ComputeWorldBound(s.GetDefaultPrim()).ComputeAlignedRange()\nresult['visible_bounds_m']=[list(box.GetMin()),list(box.GetMax())]\nPath(p['output']).write_text(json.dumps(result,allow_nan=False))\n"
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": runtime["source_root"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    run = subprocess.run(
        [runtime["validator_python"], "-B", "-c", code],
        input=json.dumps(
            {"source": str(source), "package": str(package), "output": str(output)}
        ),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if run.returncode:
        raise RuntimeError(
            "Bounded NVIDIA reference conformance/visible-bounds read failed; private source bytes preserved"
        )
    if (
        output.resolve() != output
        or not output.is_file()
        or output.stat().st_nlink != 1
    ):
        raise RuntimeError("Regular actual source receipt unavailable")
    return json.loads(output.read_text())


def validate_reference_binding(cfg, *, allowed_reference_root, actual_source_root=None):
    from simready_bench.engine.environment import (
        validate_shared_profile,
        verify_profile_code,
        validate_dependency_manifest,
        _package_files,
    )
    from simready_bench.engine.scoring import validate_policy

    validate_policy(cfg["validator_scoring"])
    profile = validate_shared_profile(cfg["shared_environment_profile"])
    code = verify_profile_code(profile, actual_source_root=actual_source_root)
    binding = cfg["validator_reference"]
    root = Path(allowed_reference_root).absolute()
    package = root / binding["package"]
    if (
        root.resolve() != root
        or package.resolve() != package
        or package.parent != root
        or (not package.is_dir())
    ):
        raise ValueError("Confined private reference package required")
    manifest = validate_dependency_manifest(binding["dependency_manifest"])
    if (
        manifest["package"] != package.name
        or _package_files(package) != manifest["files"]
    ):
        raise ValueError("Reference source closure changed")
    receipt = binding["conformance_receipt"]
    path = Path(receipt["path"])
    if (
        path.resolve() != path
        or path.is_symlink()
        or (not path.is_file())
        or (path.stat().st_nlink != 1)
        or (hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"])
    ):
        raise ValueError("Original source conformance receipt changed")
    report = json.loads(path.read_bytes())
    if (
        report.get("accepted") is not True
        or report.get("source_sha256") != manifest["source_sha256"]
    ):
        raise ValueError("Unfiltered source NVIDIA conformance binding differs")
    return {
        "schema": "simready-validator-reference-binding-v1",
        "status": "verified",
        "asset_id": binding["asset_id"],
        "source_sha256": manifest["source_sha256"],
        "closure_sha256": manifest["closure_sha256"],
        "profile_sha256": profile["profile_sha256"],
        "code_identity": code,
        "visibility": "trusted_verifier_only",
    }


def public_instruction(dimensions):
    return (
        "Reconstruct the supplied object as /output/asset.usd.\nNominal dimensions in canonical metres XYZ: "
        + json.dumps(dimensions)
        + "\nRead /task/contract.md, /task/tool-docs.md and /task/scoring-policy.json. Identify the rigid parts, joints and their motion before authoring, then reconstruct appearance and mechanical structure. Use Python/PXR. Do not use Blender. Validate and inspect the NVIDIA findings before submitting.\n"
    )


def prepare_validator_assets(resolved, *, work_dir, runtime):
    from simready_bench.engine.scoring import validate_policy
    from simready_bench.engine.environment import (
        validate_shared_profile,
        verify_profile_code,
    )
    from simready_bench.engine.native_output import EXTENSION_FILENAME, EXTENSION_SOURCE
    from simready_bench.engine.verifier import PROTOCOL_VERSION
    from harbor.models.task.config import TaskConfig
    from PIL import Image
    import numpy as np

    policy = validate_policy(resolved["normalized_manifest"]["scoring"])
    profile = validate_shared_profile(runtime["shared_profile"])
    verify_profile_code(profile)
    root = confined_work_dir(work_dir, fresh=True, ram_only=True)
    root.mkdir(parents=True, mode=448)
    assets = {c["asset_id"]: c["asset"] for c in resolved["conditions"]}
    refs = {}
    for aid, asset in assets.items():
        ar = root / "references" / aid
        ar.mkdir(parents=True, mode=448)
        package = Path(runtime["allowed_reference_root"]) / (
            "bv_"
            + resolved["configuration_sha256"][:12]
            + "_"
            + hashlib.sha256(aid.encode()).hexdigest()[:12]
        )
        raw = _reference_input(asset, ar, package_root=package)
        metadata = _metadata(
            Path(raw["usd"]), package, runtime, ar / "source-conformance.json"
        )
        bounds = np.asarray(metadata["visible_bounds_m"], float)
        dimensions = (bounds[1] - bounds[0]).tolist()
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or any((d <= 0 for d in dimensions))
        ):
            raise RuntimeError("Actual visible source dimensions unavailable")
        image = (
            Path(raw["input_png"])
            if raw["input_png"]
            else package / asset["source_view"]
            if asset.get("source_view")
            else None
        )
        if image is None:
            raise ValueError(
                "Supply a source_view PNG or input_views for the reference asset"
            )
        if (
            image.is_symlink()
            or not image.is_file()
            or image.stat().st_size > 10 * 1024**2
        ):
            raise ValueError("Bounded original input PNG required")
        with Image.open(image) as im:
            if im.format != "PNG":
                raise ValueError("Actual PNG input required")
            im.load()
        rig = None
        ref = {
            "asset_id": aid,
            "package": package.name,
            "dependency_manifest": metadata["dependency_manifest"],
            "conformance_receipt": {
                "path": str(ar / "source-conformance.json"),
                "sha256": hashlib.sha256(
                    (ar / "source-conformance.json").read_bytes()
                ).hexdigest(),
            },
        }
        refs[aid] = (ref, image, asset.get("nominal_dimensions_m", dimensions), rig)
    prepared = []
    for condition in resolved["conditions"]:
        aid = condition["asset_id"]
        ref, image, dimensions, rig = refs[aid]
        settings = condition["settings"]
        tid = "asset_" + hashlib.sha256(aid.encode()).hexdigest()[:16]
        base = root / "conditions" / condition["condition_id"]
        task = base / "tasks" / tid
        public = task / "environment/public"
        public.mkdir(parents=True, mode=448)
        (task / "tests").mkdir(mode=448)
        (base / "private").mkdir(mode=448)
        limits = {
            k: settings["budgets"][k]
            for k in (
                "agent_seconds",
                "final_seconds",
                "inference_requests",
                "validation_calls",
                "preview_requests",
                "views_per_preview",
                "preview_seconds",
                "validation_seconds",
            )
        }
        published = {
            "validator_scoring": policy,
            "limits": limits,
            "model_call_budget": settings["budgets"]["inference_requests"],
            "benchmark_output_token_cap": None,
            "semantic_labels": {
                "required": True,
                "scoring": True,
                "criterion": "mandatory NVIDIA SL IDs only",
            },
            "reward_reporting": {
                "harbor_key": "simready_reward",
                "scale": [0, 100],
                "formula": "95 * passed / fixed_applicable + 5_if_all_pass",
            },
            "composite_scores": {
                "formal": "SimReady Robotics-Prop mandatory requirements",
                "visual": "geometric mean of shape and appearance material correspondence",
                "kinematic": "joint recovery and allowed relative motion correspondence",
            },
        }
        instruction = public_instruction(dimensions)
        contents = {
            "instruction.md": instruction.encode(),
            "thumbnail.png": image.read_bytes(),
            "contract.md": CONTRACT.encode(),
            "tool-docs.md": TOOLS.replace(
                "{inference_requests}", str(limits["inference_requests"]),
            ).encode(),
            "scoring-policy.json": (
                json.dumps(published, sort_keys=True, indent=2) + "\n"
            ).encode(),
            EXTENSION_FILENAME: EXTENSION_SOURCE.encode(),
        }
        for name, data in contents.items():
            (public / name).write_bytes(data)
            (public / name).chmod(292)
        (task / "instruction.md").write_bytes(contents["instruction.md"])
        (task / "environment/Dockerfile").write_text(
            "FROM "
            + profile["actor_image"]
            + "\nUSER 1000:1000\nWORKDIR /output\nENTRYPOINT []\n"
        )
        (task / "tests/Dockerfile").write_text(
            "FROM " + profile["verifier_image"] + "\n"
        )
        (task / "tests/test.sh").write_text(
            "#!/bin/sh\nset -eu\nexec python3 -m simready_bench.engine.verifier --socket /run/simready-verifier/service.sock --output-dir /logs/verifier\n"
        )
        (task / "tests/test.sh").chmod(365)
        resources = {
            "cpus": 4,
            "memory_mb": 16384,
            "storage_mb": 8192,
            "gpus": 0,
            "network_mode": "no-network",
            "workdir": "/output",
        }
        tc = TaskConfig.model_validate(
            {
                "schema_version": "1.4",
                "metadata": {
                    "protocol": PROTOCOL_VERSION,
                    "task_version": "reference-reconstruction-three-fidelities-v1",
                },
                "agent": {
                    "user": 1000,
                    "timeout_sec": limits["agent_seconds"],
                    "network_mode": "no-network",
                },
                "environment": {**resources, "docker_image": profile["actor_image"]},
                "verifier": {
                    "user": 1000,
                    "timeout_sec": limits["final_seconds"],
                    "environment_mode": "separate",
                    "environment": {
                        **resources,
                        "docker_image": profile["verifier_image"],
                    },
                },
                "artifacts": [{"source": "/output", "destination": "candidate"}],
            }
        )
        (task / "task.toml").write_text(tc.model_dump_toml())
        cfg = {
            "protocol": PROTOCOL_VERSION,
            "task_id": tid,
            "nominal_dimensions_m": dimensions,
            "validator_scoring": policy,
            "validator_reference": ref,
            "shared_environment_profile": profile,
            "reference_sha256": ref["dependency_manifest"]["closure_sha256"],
            "limits": limits,
            "public_file_hashes": {
                n: hashlib.sha256(b).hexdigest() for n, b in contents.items()
            },
        }
        validate_reference_binding(
            cfg, allowed_reference_root=runtime["allowed_reference_root"]
        )
        config = base / "private" / (tid + ".json")
        write_json(config, cfg)
        prepared.append(
            {
                "condition_id": condition["condition_id"],
                "asset_id": aid,
                "task_path": str(task),
                "private_config": str(config),
                "task_id": tid,
                "actor_image": profile["actor_image"],
                "verifier_image": profile["verifier_image"],
            }
        )
    result = {
        "schema": "simready-bench-prepared-v1",
        "configuration_sha256": resolved["configuration_sha256"],
        "policy_sha256": resolved["policy_sha256"],
        "profile_sha256": profile["profile_sha256"],
        "conditions": prepared,
        "provider_calls": 0,
        "official_success_claim": False,
    }
    result["files"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    write_json(root / "prepared.json", result)
    return result
