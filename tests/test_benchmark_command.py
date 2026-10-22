import json
from pathlib import Path

import pytest

from simready_bench import benchmark as workflow
from simready_bench.cli import main


def selected_model():
    return workflow.select_model("gpt-6.1-sol", provider="openai")[0]


def test_reference_set_defaults_to_four_attempts_for_each_of_25_assets():
    template, _, _ = workflow.reference_manifest("nvidia-25")
    result = workflow.compose_experiment(
        template, model=selected_model(), maximum_spend_usd=250
    )
    assert len(result["assets"]) * result["repetitions"] == 100
    assert result["source"] == template["source"]
    assert {asset["id"] for asset in result["assets"]} == set(
        result["source"]["packages"]
    )
    assert result["assets"] == template["assets"]
    assert result["benchmark_freeze"] == template["benchmark_freeze"]
    assert result["scoring"] == template["scoring"]
    assert result["defaults"]["budgets"]["maximum_spend_usd"] == 2.5
    assert result["defaults"]["illustrations"]["renderer"] == "storm-llvmpipe"
    assert result["models"][0]["model_id"] == "gpt-6.1-sol"
    assert result["harnesses"] == [{"id": "pi", "adapter": "harbor", "agent": "pi"}]
    assert result["skill_sets"] == [{"id": "none", "paths": []}]
    single = workflow.compose_experiment(
        template, model=selected_model(), repetitions=1, maximum_spend_usd=250
    )
    assert len(single["assets"]) * single["repetitions"] == 25


def test_json_asset_list_preserves_composition_and_texture_package(tmp_path):
    package = tmp_path / "cup"
    package.mkdir()
    (package / "asset.usda").write_text(
        '#usda 1.0\ndef Xform "Asset" (references = @part.usda@) {}\n'
    )
    (package / "part.usda").write_text(
        '#usda 1.0\ndef Cube "Part" {\n    asset texture = @texture.png@\n}\n'
    )
    (package / "texture.png").write_bytes(b"fixture texture")
    listing = tmp_path / "assets.json"
    listing.write_text(json.dumps(["cup/asset.usda"]))
    manifest, base, package_inputs = workflow.reference_manifest(listing)
    output = tmp_path / "results"
    output.mkdir()
    assets = output / "assets"
    assets.mkdir()
    workflow.stage_assets(
        manifest,
        base_dir=base,
        destination=assets,
        output_dir=output,
        package_inputs=package_inputs,
    )
    entry = manifest["assets"][0]
    staged = assets / entry["usd"]
    assert staged.read_text() == (package / "asset.usda").read_text()
    assert (staged.parent / "part.usda").read_text() == (
        package / "part.usda"
    ).read_text()
    assert (staged.parent / "texture.png").read_bytes() == b"fixture texture"
    assert len(entry["dependencies"]) == 2


def test_explicit_asset_manifest_and_skills_keep_declared_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "cup.usda").write_text("#usda 1.0\n")
    (source / "thumbnail.png").write_bytes(b"source pixels")
    (source / "unrelated.txt").write_text("unrelated")
    output = tmp_path / "results"
    output.mkdir()
    manifest = {
        "assets": [{"id": "cup", "usd": "cup.usda", "source_view": "thumbnail.png"}]
    }
    workflow.stage_assets(
        manifest, base_dir=source, destination=output / "assets", output_dir=output
    )
    assert manifest["assets"][0]["dependencies"] == ["thumbnail.png"]
    assert not (output / "assets/unrelated.txt").exists()
    skill = tmp_path / "usd-authoring"
    skill.mkdir()
    (skill / "SKILL.md").write_text("Author a USD asset using Python/PXR.")
    supplied = workflow.stage_skills([skill], output / "skills")
    assert len(supplied) == 1
    staged = output / "skills" / supplied[0]["paths"][0]
    assert (staged / "SKILL.md").read_text() == (skill / "SKILL.md").read_text()


@pytest.mark.parametrize("deploy", ["local", "daytona"])
def test_cli_dispatches_whole_reference_set_to_selected_executor(
    tmp_path, monkeypatch, capsys, deploy
):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-key-kept-out-of-files")
    from simready_bench import sources
    monkeypatch.setattr(sources, "import_source", lambda *args, **kwargs: None)
    output = tmp_path / deploy
    calls = []

    def policy(**kwargs):
        return {
            "roots": {
                name: str(output / name)
                for name in ("assets", "skills", "input_views", "private_manifests")
            },
            "providers": {"selected": kwargs["provider"]},
            "limits": {},
            "runtime": {},
        }

    def execute(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return {
            "schema": "simready-bench-matrix-result-v1",
            "planned_conditions": 25,
            "completed_conditions": 25,
            "results": [
                {"asset_id": asset["id"], "status": "completed"}
                for asset in manifest["assets"]
            ],
        }

    monkeypatch.setattr(workflow, "prepare_policy", policy)
    if deploy == "daytona":
        from simready_bench import parallel

        monkeypatch.setattr(parallel, "run_parallel_daytona", execute)
    else:
        monkeypatch.setattr(workflow, "execute_local_sequential", execute)
    assert (
        main(
            [
                "benchmark",
                "NV25",
                "--model",
                "gpt-6.1-sol",
                "--harness",
                "pi",
                "--deploy",
                deploy,
                "--output-dir",
                str(output),
                "--max-spend-usd",
                "250",
                "--repetitions",
                "1",
            ]
        )
        == 0
    )
    assert "25 of 25 reconstruction attempts completed" in capsys.readouterr().out
    assert len(calls) == 1 and len(calls[0][0]["assets"]) == 25
    assert calls[0][0]["repetitions"] == 1
    provider = calls[0][1]["policy"]["providers"]["selected"]
    assert provider["project_budget"]["maximum_requests"] == (4800 if deploy == "daytona" else 1600)
    assert provider["project_budget"]["maximum_spend_usd"] == 250
    assert provider["api_key_env"] == "OPENAI_API_KEY"
    assert (
        "fixture-key-kept-out-of-files" not in (output / "experiment.json").read_text()
    )


def test_missing_credentials_does_not_create_output_or_allocate_work(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        workflow.benchmark(
            model="gpt-6.1-sol", output_dir=output, maximum_spend_usd=250
        )
    assert not output.exists()


def test_existing_policy_preserves_persistent_budget_and_runtime(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    provider = workflow._new_provider(
        "openai", "gpt-6.1-sol", api_key_env=None, maximum_spend_usd=250, conditions=25
    )
    existing = {
        "providers": {"old": provider},
        "roots": {},
        "limits": {},
        "runtime": {
            "authority_state_dir": "/persistent/authority",
            "source_root": "/installed/source",
            "illustration_image": "sha256:" + "a" * 64,
        },
    }
    policy = workflow.prepare_policy(
        output_dir=output,
        provider=provider,
        maximum_spend_usd=250,
        conditions=25,
        existing=existing,
    )
    assert policy["runtime"]["authority_state_dir"] == "/persistent/authority"
    assert policy["runtime"]["source_root"] == "/installed/source"
    from simready_bench.runtime_images import CPU_IMAGE

    assert policy["runtime"]["illustration_image"] == CPU_IMAGE
    assert (
        policy["providers"]["selected"]["project_budget"]["authorization_id"]
        == provider["project_budget"]["authorization_id"]
    )
    assert policy["limits"]["maximum_matrix_runs"] == 25
    assert existing["roots"] == {}


def test_report_exports_root_artifacts_with_captured_image(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    output = tmp_path / "results"
    output.mkdir()
    image = work / "candidate.png"
    image.write_bytes(b"rendered image fixture")
    import hashlib

    result = {
        "schema": "simready-bench-matrix-result-v1",
        "results": [
            {
                "condition_id": "c",
                "asset_id": "cup",
                "status": "completed",
                "illustrations": {
                    "status": "rendered",
                    "images": [
                        {
                            "file": str(image),
                            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                            "view": "candidate / front",
                        }
                    ],
                },
            }
        ],
    }
    report = workflow.publish_report(work, output, results=result)
    assert Path(report["html"]) == output / "report.html"
    assert (output / "results.json").is_file() and (output / "scores.csv").is_file()
    rows = json.loads((output / "results.json").read_text())["results"]
    assert rows[0]["images"]
    assert not (output / ".report-export").exists()


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="The local executor uses Linux filesystem APIs",
)
def test_local_execution_resolves_assets_runs_harbor_adapter_and_exports_captures(
    tmp_path, monkeypatch
):
    roots = {}
    output = tmp_path / "results"
    output.mkdir()
    for name in ("assets", "skills", "input_views", "private_manifests"):
        directory = output / name
        directory.mkdir()
        roots[name] = str(directory)
    (output / "assets/cup.usda").write_text("#usda 1.0\n")
    (output / "assets/thumbnail.png").write_bytes(b"input pixels")
    manifest = workflow.compose_experiment(
        {
            "assets": [
                {
                    "id": "cup",
                    "usd": "cup.usda",
                    "dependencies": ["thumbnail.png"],
                    "source_view": "thumbnail.png",
                }
            ]
        },
        model=selected_model(),
        maximum_spend_usd=10,
        repetitions=1,
    )
    provider = workflow._new_provider(
        "openai", "gpt-6.1-sol", api_key_env=None, maximum_spend_usd=10, conditions=1
    )
    policy = {
        "roots": roots,
        "providers": {"selected": provider},
        "runtime": {},
        "limits": {
            "maximum_spend_usd": 10,
            "maximum_condition_spend_usd": 10,
            "maximum_matrix_runs": 1,
            "inference_requests": 64,
            "max_input_tokens": None,
            "max_output_tokens": None,
        },
    }
    calls = []

    def run(resolved, *, work_dir, policy):
        calls.append(resolved)
        work_dir.mkdir()
        image = work_dir / "candidate.png"
        image.write_bytes(b"captured candidate pixels")
        import hashlib

        return {
            "schema": "simready-bench-matrix-result-v1",
            "results": [
                {
                    "condition_id": "cup-condition",
                    "asset_id": "cup",
                    "status": "completed",
                    "illustrations": {
                        "status": "rendered",
                        "images": [
                            {
                                "file": str(image),
                                "sha256": hashlib.sha256(
                                    image.read_bytes()
                                ).hexdigest(),
                                "view": "candidate / front",
                            }
                        ],
                    },
                }
            ],
        }

    from simready_bench import local as runner

    monkeypatch.setattr(runner, "run", run)
    result = workflow.execute_local(
        manifest, policy=policy, output_dir=output, work_dir=tmp_path / "work"
    )
    assert len(calls) == 1 and calls[0]["conditions"][0]["asset"]["closure"]
    assert calls[0]["conditions"][0]["model"]["model_id"] == "gpt-6.1-sol"
    assert result["results"][0]["status"] == "completed"
    portable = json.loads((output / "results.json").read_text())
    assert portable["results"][0]["images"]
    assert (output / "report.html").is_file()


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="The local executor uses Linux filesystem APIs",
)
def test_sequential_execution_keeps_repetitions_budget_identities_and_exported_captures(
    tmp_path, monkeypatch
):
    import copy
    import hashlib
    import shutil
    import sqlite3
    from simready_bench import local as runner, sources as product_source
    from simready_bench.config import resolve_experiment
    from simready_bench.engine.inference_gateway import ProjectBudget
    from simready_bench.engine.acquisition import REVISION

    output = tmp_path / "results"
    output.mkdir()
    roots = {}
    for name in ("assets", "skills", "input_views", "private_manifests"):
        directory = output / name
        directory.mkdir()
        roots[name] = str(directory)
    references = tmp_path / "references"
    references.mkdir()
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    assets = [
        {
            "id": name,
            "usd": "imported/" + name + "/asset.usda",
            "dependencies": ["imported/" + name + "/thumbnail.png"],
            "source_view": "imported/" + name + "/thumbnail.png",
        }
        for name in ("asset-a", "asset-b")
    ]
    manifest = workflow.compose_experiment(
        {
            "assets": assets,
            "source": {
                "kind": "nvidia-hf-packages",
                "revision": REVISION,
                "packages": [a["id"] for a in assets],
                "destination": "imported",
            },
        },
        model=selected_model(),
        maximum_spend_usd=20,
    )
    provider = workflow._new_provider(
        "openai", "gpt-6.1-sol", api_key_env=None, maximum_spend_usd=20, conditions=8
    )
    original_provider = copy.deepcopy(provider)
    policy = {
        "roots": roots,
        "providers": {"selected": provider},
        "runtime": {
            "allowed_reference_root": str(references),
            "authority_state_dir": str(authority),
        },
        "limits": {
            "maximum_spend_usd": 20,
            "maximum_condition_spend_usd": 2.5,
            "maximum_matrix_runs": 8,
            "inference_requests": 64,
            "max_input_tokens": None,
            "max_output_tokens": None,
        },
    }
    acquired = []

    def acquire(source, *, base_dir, assets_root):
        assert len(source["packages"]) == 1
        name = source["packages"][0]
        imported = Path(assets_root) / "imported"
        assert not imported.exists() or not list(imported.iterdir())
        package = imported / name
        package.mkdir(parents=True)
        (package / "asset.usda").write_text("#usda 1.0\n")
        (package / "thumbnail.png").write_bytes(b"input pixels")
        acquired.append(name)

    # Compare split execution with the same fully resolved matrix identities.
    for asset in assets:
        path = output / "assets" / asset["usd"]
        path.parent.mkdir(parents=True)
        path.write_text("#usda 1.0\n")
        (path.parent / "thumbnail.png").write_bytes(b"input pixels")
    expected = {
        c["condition_id"]
        for c in resolve_experiment(manifest, base_dir=output, policy=policy)[
            "conditions"
        ]
    }
    shutil.rmtree(output / "assets/imported")
    monkeypatch.setattr(product_source, "import_source", acquire)

    def run(resolved, *, work_dir, policy):
        assert len(resolved["conditions"]) == 4
        assert policy["providers"]["selected"] == original_provider
        assert not list(references.iterdir())
        work_dir.mkdir(mode=0o700)
        package = references / ("bv_" + str(len(acquired)))
        package.mkdir()
        (package / "private-source.usda").write_text("private reference fixture")
        ledger = runner.ParentAuthorityLedger(authority)
        rows = []
        for condition in resolved["conditions"]:
            ledger.reserve(
                ProjectBudget(**provider["project_budget"]),
                allocation_id=condition["condition_id"],
                requests=32,
                maximum_spend_usd=condition["settings"]["budgets"]["maximum_spend_usd"],
                work_root=work_dir,
            )
            image = work_dir / (str(condition["repetition"]) + ".png")
            image.write_bytes(b"captured candidate pixels")
            rows.append(
                {
                    "condition_id": condition["condition_id"],
                    "asset_id": condition["asset_id"],
                    "status": "completed",
                    "illustrations": {
                        "status": "rendered",
                        "images": [
                            {
                                "file": str(image),
                                "sha256": hashlib.sha256(
                                    image.read_bytes()
                                ).hexdigest(),
                                "view": "candidate / front",
                            }
                        ],
                    },
                }
            )
        return {"schema": "simready-bench-matrix-result-v1", "results": rows}

    monkeypatch.setattr(runner, "run", run)
    started_assets = []
    result = workflow.execute_local_sequential(
        manifest,
        policy=policy,
        output_dir=output,
        work_dir=tmp_path / "work",
        on_asset_start=lambda asset, index, count: started_assets.append(
            (asset["id"], index, count)
        ),
    )
    assert result["planned_conditions"] == result["completed_conditions"] == 8
    assert {row["condition_id"] for row in result["results"]} == expected
    assert acquired == ["asset-a", "asset-b"]
    assert started_assets == [("asset-a", 1, 2), ("asset-b", 2, 2)]
    assert not list(references.iterdir()) and not list(
        (output / "assets/imported").iterdir()
    )
    portable = json.loads((output / "results.json").read_text())
    assert len(portable["results"]) == 8 and not portable["partial"]
    assert all(
        (output / "evidence" / row["images"][0]["path"]).is_file()
        for row in portable["results"]
    )
    assert all((output / row["report_path"]).is_file() for row in portable["results"])
    assert not (tmp_path / "work/a0").exists() and not (tmp_path / "work/a1").exists()
    with sqlite3.connect(authority / "parent-allowances.sqlite3") as database:
        allocations = database.execute(
            "SELECT requests,usd FROM allocations"
        ).fetchall()
    assert len(allocations) == 8 and sum(row[0] for row in allocations) == 256
    assert sum(float(row[1]) for row in allocations) == 20
