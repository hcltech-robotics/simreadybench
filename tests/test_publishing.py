from copy import deepcopy
import csv
import hashlib
from importlib.resources import files
import json
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from simready_bench.config import condition_identity
from simready_bench.publishing import (
    HuggingFacePublisher, PublicationError, ViewerRebuildRequired,
    build_viewer_dataset, build_viewer_update, render_viewer_card,
)
from simready_bench.reporting import USAGE_FIELDS


def _viewer_table(raw):
    return pq.read_table(pa.BufferReader(raw))


def _viewer_rows(raw):
    return _viewer_table(raw).to_pylist()


def _png(colour="red"):
    stream = io.BytesIO()
    Image.new("RGB", (8, 6), colour).save(stream, format="PNG")
    return stream.getvalue()


def _legacy_viewer_index(index, version):
    legacy = deepcopy(index)
    legacy["schema"] = "simreadybench-viewer-index/v" + str(version)
    for run in legacy["runs"].values():
        for groups in run["groups"].values():
            for group in groups:
                group["path"] = group["path"].removesuffix(".parquet") + ".csv"
                if version == 1 and group["path"] == run["viewer_scores"].removesuffix(".parquet") + ".csv":
                    group["path"] = run["scores"]
        run["viewer_scores"] = run["viewer_scores"].removesuffix(".parquet") + ".csv"
        if version == 1:
            del run["viewer_scores"]
    if version == 1:
        del legacy["columns"], legacy["schema_sha256"]
    return legacy


class Hub:
    def __init__(self, root):
        self.root = root
        self.files = {
            "README.md": b"---\nlicense: mit\nauthors: [Research group]\nconfigs:\n- config_name: other\n  default: true\n  data_files: other.csv\n---\n# Benchmark results\n\nShared methodology and reference illustrations.\n",
        }
        self.uploads, self.commits = [], []
        self.revision = 0
        self.fail_upload = False
        self.role = "write"

    def whoami(self):
        return {"auth": {"accessToken": {"role": self.role}}}

    def repo_info(self, *args, **kwargs):
        return SimpleNamespace(private=True, sha="head-" + str(self.revision))

    def hf_hub_download(self, repo, filename, **kwargs):
        from huggingface_hub.errors import EntryNotFoundError

        if filename not in self.files:
            raise EntryNotFoundError("The fixture file does not exist")
        path = self.root / "hub-cache" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[filename])
        return str(path)

    def upload_folder(self, **kwargs):
        if self.fail_upload:
            raise TimeoutError("Private request details must not escape")
        self.uploads.append(kwargs)
        assert kwargs["parent_commit"] == "head-" + str(self.revision)
        for path in Path(kwargs["folder_path"]).rglob("*"):
            if path.is_file():
                self.files[
                    kwargs["path_in_repo"]
                    + "/"
                    + path.relative_to(kwargs["folder_path"]).as_posix()
                ] = path.read_bytes()
        self.revision += 1
        return SimpleNamespace(oid="head-" + str(self.revision))

    def create_commit(self, **kwargs):
        if self.fail_upload:
            raise TimeoutError("Private request details must not escape")
        self.commits.append(kwargs)
        assert kwargs["parent_commit"] == "head-" + str(self.revision)
        for operation in kwargs["operations"]:
            source = operation.path_or_fileobj
            self.files[operation.path_in_repo] = source if isinstance(source, bytes) else Path(source).read_bytes()
        self.revision += 1
        return SimpleNamespace(oid="head-" + str(self.revision))


@pytest.fixture
def publication(tmp_path):
    manifest = {
        "assets": [{"id": "reference-a"}, {"id": "reference-b"}],
        "models": [
            {
                "id": "selected",
                "model_id": "another-model",
                "generation": {"reasoning_effort": "high"},
            }
        ],
        "harnesses": [{"id": "pi", "agent": "pi"}],
        "skill_sets": [{"id": "none", "paths": []}],
        "repetitions": 2,
        "seed": 19,
        "scoring": {
            "profile": {"id": "Robotics-Prop", "version": "4.0.0"},
            "policy_id": "NVIDIA-Robotics-Prop-4.0.0-mandatory-95-plus-5-v1",
        },
        "defaults": {"generation": {"reasoning_effort": "high"}},
    }
    report = tmp_path / "report"
    report.mkdir()
    hub = Hub(tmp_path)

    def fixture(path, raw):
        target = report / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return {
            "path": path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }

    usd = fixture("original/asset.usda", b'#usda 1.0\ndef Xform "Asset" {}\n')
    package = fixture(
        "original/manifest.json",
        json.dumps(
            {
                "entrypoint": "asset.usda",
                "files": [
                    {
                        "relative_path": "asset.usda",
                        "sha256": usd["sha256"],
                        "size_bytes": usd["bytes"],
                    }
                ],
            }
        ).encode(),
    )
    image = fixture("evidence/matched.png", b"matched-capture-fixture")
    trajectory = fixture("evidence/trajectory.json", b'{"messages":[]}')
    validator = fixture("original/nvidia.json", b'{"requirements":[]}')

    def row(asset="reference-a", repetition=0, *, passed=46):
        identifier, seed = condition_identity(
            {"asset": asset, "harness": "pi", "model": "selected", "skill_set": "none"},
            repetition=repetition,
            master_seed=manifest["seed"],
        )
        score = 95 * passed / 46 + (5 if passed == 46 else 0)
        return {
            "condition_id": identifier,
            "asset_id": asset,
            "repetition": repetition,
            "seed": seed,
            "requested_model": "another-model",
            "harness_id": "pi",
            "skill_set_id": "none",
            "status": "completed",
            "simready_reward": score,
            "scoring_policy_id": manifest["scoring"]["policy_id"],
            "simready": {
                "status": "scored",
                "score_percent": score,
                "policy_id": manifest["scoring"]["policy_id"],
                "unknown_count": 0,
                "profile_coverage_fraction": 1,
                "passed_count": passed,
                "failed_count": 46 - passed,
                "applicable_count": 46,
                "profile_mandatory_count": 46,
                "official_conformance": {
                    "profile": {"id": "Robotics-Prop", "version": "4.0.0"}
                },
            },
            "fidelity": {
                "status": "measured",
                "components": {
                    "shape": {
                        "score_percent": 0,
                        "eligible": 1,
                        "coverage_fraction": 1,
                    },
                    "joints": {
                        "score_percent": None,
                        "eligible": 0,
                        "coverage_fraction": 0,
                    },
                },
            },
            "candidate": {
                "status": "available",
                "source_entrypoint": "asset.usda",
                "manifest": package["path"],
                "manifest_sha256": package["sha256"],
                "files": [
                    {
                        "path": usd["path"],
                        "relative_path": "asset.usda",
                        "sha256": usd["sha256"],
                        "size_bytes": usd["bytes"],
                    }
                ],
            },
            "images": [
                {
                    **image,
                    "path": "matched.png",
                    "view": "matched reference / reconstruction contact sheet",
                }
            ],
            "traces": [
                {**trajectory, "path": "trajectory.json", "kind": "agent_trajectory"},
                {"path": "process.log", "kind": "agent_log"},
            ],
            "nvidia_validator_report_file": {
                **validator,
                "complete": True,
                "findings_truncated": False,
            },
            "model_requests": [{"cost_usd": 3}],
            "statistics": {"input_tokens": 100},
            "actor_stop_receipt": {"bridge_id": "private"},
            "actor_stop_audit": {"container_id": "private"},
            "elapsed_seconds": 999,
            "inference_cost_usd": 3,
            "error_message": "Private process details",
        }

    def save(rows):
        (report / "results.json").write_text(
            json.dumps(
                {
                    "schema": "simready-bench-portable-report-v1",
                    "planned_conditions": 2,
                    "partial": True,
                    "results": rows,
                }
            )
        )
        return {"directory": str(report), "results": str(report / "results.json")}

    return manifest, report, hub, row, save


def test_partial_publication_keeps_only_scientific_completed_readouts(publication):
    manifest, report, hub, row, save = publication
    valid = row(passed=0)
    blocked = row(repetition=1)
    blocked["status"] = "blocked"
    publisher = HuggingFacePublisher(
        "team/results", "model-evaluation", manifest, api=hub
    )
    receipt = publisher(save([valid, blocked]))
    public = json.loads(hub.files["runs/model-evaluation/results.json"])
    assert receipt["completed_conditions"] == 1
    assert public["planned_conditions"] == 4 and public["partial"]
    assert public["results"][0]["simready_reward"] == 0
    assert (
        public["results"][0]["fidelity"]["components"]["joints"]["score_percent"]
        is None
    )
    assert (
        not {
            "model_requests",
            "statistics",
            "actor_stop_receipt",
            "actor_stop_audit",
            "error_message",
        }
        & public["results"][0].keys()
    )
    assert public["results"][0]["elapsed_seconds"] == 999
    assert public["results"][0]["input_tokens"] == 100
    assert public["results"][0]["inference_cost_usd"] == 3
    assert public["results"][0]["total_cost_usd"] is None
    assert len(public["results"][0]["traces"]) == 1
    assert not any(
        "process.log" in path or "operator" in path or "authority" in path
        for path in hub.files
    )
    assert json.loads(hub.files["leaderboard.json"])["entries"] == []
    assert (report / "original/asset.usda").is_file()


def test_publication_retains_reconstructed_usage_provenance(publication):
    manifest, _, hub, row, save = publication
    value = row()
    value["inference_cost_kind"] = "pricing-derived inference cost"
    value["usage_reconstruction"] = {
        "schema": "simreadybench-usage-reconstruction/v1",
        "canonical_run_id": "model-evaluation",
        "gateway_records_sha256": "a" * 64,
        "pricing_derived_cost_complete": True,
    }
    HuggingFacePublisher("team/results", "model-evaluation", manifest, api=hub)(save([value]))
    actual = json.loads(hub.files["runs/model-evaluation/results.json"])["results"][0]
    assert actual["inference_cost_kind"] == value["inference_cost_kind"]
    assert actual["usage_reconstruction"] == value["usage_reconstruction"]


def test_usage_columns_reach_published_json_csv_and_main_table(publication):
    import csv
    import io

    manifest, _, hub, row, save = publication
    valid = row()
    valid.update(inference_calls_used=4, validation_calls_used=2)
    valid["statistics"].update(
        output_tokens=50, model_request_seconds=2.5, model_request_errors=1,
        compute_cost_usd=0.5,
    )
    publisher = HuggingFacePublisher("team/results", "usage", manifest, api=hub)
    publisher(save([valid]))
    public = json.loads(hub.files["runs/usage/results.json"])["results"][0]
    assert public["inference_calls_used"] == 4
    assert public["validation_calls_used"] == 2
    assert public["model_request_errors"] == 1
    assert public["input_tokens"] == 100
    assert public["output_tokens"] == 50
    assert public["total_tokens"] == 150
    assert public["elapsed_seconds"] == 999
    assert public["model_request_seconds"] == 2.5
    assert public["inference_cost_usd"] == 3
    assert public["compute_cost_usd"] == 0.5
    assert public["total_cost_usd"] == 3.5
    flat = next(csv.DictReader(io.StringIO(hub.files["runs/usage/scores.csv"].decode())))
    for field in USAGE_FIELDS:
        assert float(flat[field]) == public[field]
    assert b"Input tokens" in hub.files["runs/usage/report.html"]
    assert b"Total cost (USD)" in hub.files["runs/usage/report.html"]
    assert "model_requests" not in public


def test_old_readout_can_gain_usage_metadata_without_changing_existing_scores(publication):
    manifest, _, hub, row, save = publication
    publisher = HuggingFacePublisher("team/results", "usage", manifest, api=hub)
    publisher(save([row()]))
    key = "runs/usage/results.json"
    old = json.loads(hub.files[key])
    for field in USAGE_FIELDS:
        old["results"][0].pop(field, None)
    hub.files[key] = json.dumps(old).encode()
    publisher(save([row()]))
    restored = json.loads(hub.files[key])["results"][0]
    assert restored["input_tokens"] == 100
    assert restored["elapsed_seconds"] == 999
    assert restored["simready"] == old["results"][0]["simready"]
    changed = row()
    changed["statistics"]["input_tokens"] = 101
    with pytest.raises(PublicationError, match="already published reconstruction has changed"):
        publisher(save([changed]))


def test_root_card_retains_methodology_metadata_and_other_viewer_configs(publication):
    manifest, _, hub, row, save = publication
    original = hub.files["README.md"].decode().split("\n---\n", 1)[1]
    publisher = HuggingFacePublisher("team/results", "new-model", manifest, api=hub)
    publisher(save([row()]))
    card = hub.files["README.md"].decode()
    metadata, body = card[4:].split("---\n", 1)
    data = yaml.safe_load(metadata)
    assert body.startswith(original + "\n<!-- simreadybench-leaderboard:start -->")
    assert body.count("<!-- simreadybench-leaderboard:start -->") == 1
    assert "No complete NV25 sets have been published." in body
    assert data["license"] == "mit" and data["authors"] == ["Research group"]
    assert data["configs"][-1] == {
        "config_name": "other",
        "data_files": "other.csv",
    }
    assert data["configs"][0]["config_name"] == "all" and data["configs"][0]["default"]
    assert data["configs"][0]["data_files"] == [{"split": "test", "path": ["viewer/new-model/scores.parquet"]}]
    assert {item["name"]: item["dtype"] for item in data["configs"][0]["features"]}["input_tokens"] == "float64"
    assert data["configs"][1]["data_files"] == [
        {"split": "another_model", "path": ["viewer/new-model/scores.parquet"]}
    ]
    assert "another-model" not in body and "completed" not in body
    publisher(save([row(), row(repetition=1)]))
    assert (
        len(
            yaml.safe_load(hub.files["README.md"].decode().split("---\n", 2)[1])[
                "configs"
            ]
        )
        == 9
    )


@pytest.mark.parametrize("damage", ["hash", "seed", "duplicate", "score", "profile"])
def test_invalid_measurements_or_artefacts_never_upload(publication, damage):
    manifest, report, hub, row, save = publication
    rows = [row()]
    if damage == "hash":
        (report / "original/asset.usda").write_bytes(b"changed")
    if damage == "seed":
        rows[0]["seed"] += 1
    if damage == "duplicate":
        rows.append(deepcopy(rows[0]))
    if damage == "score":
        rows[0]["simready_reward"] = 50
    if damage == "profile":
        rows[0]["simready"]["official_conformance"]["profile"]["version"] = "5.0.0"
    publisher = HuggingFacePublisher("team/results", "evaluation", manifest, api=hub)
    with pytest.raises(PublicationError) as failure:
        publisher(save(rows))
    assert not failure.value.retryable
    assert hub.commits == []


def test_incremental_updates_preserve_successes_and_refuse_missing_or_changed_results(
    publication,
):
    manifest, _, hub, row, save = publication
    publisher = HuggingFacePublisher("team/results", "evaluation", manifest, api=hub)
    publisher(save([row()]))
    publisher(save([row(), row(repetition=1)]))
    assert len(json.loads(hub.files["runs/evaluation/results.json"])["results"]) == 2
    uploads = len(hub.commits)
    with pytest.raises(PublicationError, match="omit completed"):
        publisher(save([row()]))
    assert len(hub.commits) == uploads
    changed = deepcopy(manifest)
    changed["defaults"]["generation"]["reasoning_effort"] = "low"
    with pytest.raises(PublicationError, match="change settings"):
        HuggingFacePublisher("team/results", "evaluation", changed, api=hub)(
            save([row(), row(repetition=1)])
        )
    assert len(hub.commits) == uploads


def test_upload_failure_is_retryable_without_mutating_retained_results(publication):
    manifest, report, hub, row, save = publication
    callback = save([row()])
    original = (report / "results.json").read_bytes()
    publisher = HuggingFacePublisher("team/results", "evaluation", manifest, api=hub)
    hub.fail_upload = True
    with pytest.raises(PublicationError, match="TimeoutError") as failure:
        publisher(callback)
    assert failure.value.retryable
    assert "Private request" not in str(failure.value)
    assert (report / "results.json").read_bytes() == original
    hub.fail_upload = False
    assert publisher(callback, final=True)["status"] == "published"


def test_invalid_submission_is_published_and_does_not_block_later_reconstructions(
    publication,
):
    from simready_bench.engine.scoring import score_validation
    from simready_bench.engine.verifier import scoring_policy

    manifest, _, hub, row, save = publication
    failed = row(passed=0)
    failed["simready"] = score_validation(
        None, scoring_policy(), candidate_invalid=True
    )
    failed["candidate"] = {
        "status": "unavailable",
        "reason": "no_frozen_authored_submission",
    }
    failed["images"] = []
    failed["traces"] = []
    failed["nvidia_validator_report_file"] = None
    failed["fidelity"] = {"status": "unknown", "components": {}}
    publisher = HuggingFacePublisher(
        "team/results", "failed-submission", manifest, api=hub
    )
    assert publisher(save([failed]))["completed_conditions"] == 1
    assert publisher(save([failed, row(repetition=1)]))["completed_conditions"] == 2
    public = json.loads(hub.files["runs/failed-submission/results.json"])["results"]
    invalid = next(item for item in public if item["repetition"] == 0)
    assert invalid["simready_reward"] == 0
    assert invalid["simready"]["status"] == "invalid_candidate"
    assert invalid["candidate"]["status"] == "unavailable"
    assert invalid["images"] == [] and invalid["traces"] == []
    assert invalid["nvidia_validator_report_file"] is None
    assert invalid["optional_score"]["unknown_count"] == 51


def test_scored_submission_without_available_captures_still_publishes(publication):
    manifest, _, hub, row, save = publication
    measured = row()
    measured["images"] = []
    publisher = HuggingFacePublisher(
        "team/results", "capture-unavailable", manifest, api=hub
    )
    assert publisher(save([measured]))["completed_conditions"] == 1
    public = json.loads(hub.files["runs/capture-unavailable/results.json"])["results"][
        0
    ]
    assert public["simready_reward"] == 100
    assert public["images"] == []
    assert public["candidate"]["status"] == "available"
    assert public["traces"] and public["nvidia_validator_report_file"]


def test_access_check_rejects_read_only_authentication(publication):
    manifest, _, hub, _, _ = publication
    publisher = HuggingFacePublisher("team/results", "evaluation", manifest, api=hub)
    assert publisher.check_access()["private"]
    hub.role = "read"
    with pytest.raises(PublicationError, match="write access"):
        publisher.check_access()


def test_optional_score_and_fidelity_partial_numerators_are_preserved(
    publication, tmp_path
):
    manifest, _, hub, row, save = publication
    source = row()
    optional = {
        "schema": "simready-validator-optional-score-v1",
        "profile": {"id": "Robotics-Prop", "version": "4.0.0"},
        "status": "unknown",
        "score_percent": None,
        "passed_count": 10,
        "failed_count": 5,
        "unknown_count": 36,
        "measured_count": 15,
        "applicable_count": 51,
        "profile_optional_count": 51,
        "coverage_fraction": 15 / 51,
        "lower_bound_percent": 1000 / 51,
        "upper_bound_percent": 4600 / 51,
        "checks": [
            {
                "requirement_id": "optional-requirement",
                "status": "unknown",
                "mandatory": False,
            }
        ],
    }
    source["simready"]["optional_score"] = deepcopy(optional)
    source["optional_score"] = deepcopy(optional)
    component = {
        "score_percent": 30,
        "earned": 3,
        "eligible": 10,
        "measured": 4,
        "coverage_fraction": 0.4,
        "status": "partial",
    }
    source["fidelity"]["components"]["materials"] = component
    alignment = {"scale": 1, "yaw_degrees": 88.1, "translation_m": [0.01, 0.02, 0.03]}
    source["fidelity"]["alignment"] = alignment
    publisher = HuggingFacePublisher(
        "team/results", "partial-measurements", manifest, api=hub
    )
    publisher(save([source]))
    public = json.loads(hub.files["runs/partial-measurements/results.json"])["results"][
        0
    ]
    assert public["optional_score"] == optional
    assert public["simready"]["optional_score"] == optional
    assert public["fidelity"]["components"]["materials"] == component
    assert public["fidelity"]["alignment"] == alignment


def test_canonical_leaderboard_is_added_only_after_all_25_by_four_complete(publication):
    manifest, _, hub, row, save = publication
    reference = json.loads(
        files("simready_bench")
        .joinpath("resources/benchmarks/nvidia-25.json")
        .read_text()
    )
    manifest["assets"] = [{"id": asset["id"]} for asset in reference["assets"]]
    manifest["source"] = deepcopy(reference["source"])
    manifest["authoring_protocol"] = reference["authoring_protocol"]
    manifest["repetitions"] = 4
    publisher = HuggingFacePublisher(
        "team/results", "full-reference-set", manifest, api=hub
    )
    rows = [
        row(asset["id"], repetition)
        for asset in manifest["assets"]
        for repetition in range(4)
    ]
    publisher(save(rows[:-1]))
    assert json.loads(hub.files["leaderboard.json"])["entries"] == []
    publisher(save(rows))
    entry = json.loads(hub.files["leaderboard.json"])["entries"][0]
    assert entry["completed_attempts"] == 100
    assert entry["model"] == "another-model"
    card = hub.files["README.md"].decode()
    assert "| another-model | high | pi | none |" in card
    assert "authoring-64-native-output-v1" in card
    original = deepcopy(hub.files)
    publisher(save(rows), final=True)
    assert hub.files == original
    assert len(json.loads(hub.files["leaderboard.json"])["entries"]) == 1


def viewer_fixture():
    rows = [
        {"condition_id": "condition-" + str(index), "requested_model": model,
         "harness_id": harness, "skill_set_id": skills, "asset_id": asset,
         "formal_fidelity_percent": "0", "visual_fidelity_percent": "",
         "kinematic_fidelity_percent": "13.50"}
        for index, (model, harness, skills, asset) in enumerate((
            ("model/a", "pi", "none", "asset-a"),
            ("model-b", "opencode", "physics", "asset-b"),
            ("model/a", "opencode", "none", "asset-b"),
            ("model-b", "pi", "physics", "asset-a"),
        ))
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    readout = {"results": [{**row, "reasoning_effort": "medium" if row["requested_model"] == "model/a" else "none"} for row in rows], "settings": {
        "models": [{"model_id": "model/a", "generation": {"reasoning_effort": "medium"}},
                   {"model_id": "model-b", "generation": {"reasoning_effort": "none"}}],
        "reference_assets": ["asset-a", "asset-b"], "reference_revision": "pinned-revision",
        "authoring_protocol": "authoring-64-native-output-v1",
    }}
    return readout, stream.getvalue(), rows


def test_mixed_viewer_factors_route_each_row_without_changing_csv_values():
    readout, raw, rows = viewer_fixture()
    before = deepcopy(readout)
    index, shards = build_viewer_update("matrix", readout, raw)
    expected = [{**row, "formal_fidelity_percent": 0.0, "visual_fidelity_percent": None,
                 "kinematic_fidelity_percent": 13.5} for row in rows]
    assert readout == before
    card = render_viewer_card("# Scientific methodology\n", index)
    configs = yaml.safe_load(card.split("---\n", 2)[1])["configs"]
    assert [item["config_name"] for item in configs] == [
        "all", "by_model", "by_harness", "by_skills", "by_reference_set",
        "by_protocol", "by_reasoning", "by_asset",
    ]
    assert [item["config_name"] for item in configs if item.get("default")] == ["all"]
    for axis, column in (("by_model", "requested_model"), ("by_harness", "harness_id"),
                         ("by_skills", "skill_set_id"), ("by_asset", "asset_id")):
        groups = index["runs"]["matrix"]["groups"][axis]
        assert len(groups) == 2
        collected = []
        for group in groups:
            selected = _viewer_rows(shards[group["path"]])
            assert len({row[column] for row in selected}) == 1
            selected_source = [{key: row[key] for key in rows[0]} for row in selected]
            assert all(row in expected for row in selected_source)
            assert all(row["formal_fidelity_percent"] == 0 and row["visual_fidelity_percent"] is None for row in selected)
            collected.extend(selected_source)
        assert sorted(collected, key=lambda row: row["condition_id"]) == expected
    for axis in ("by_reference_set", "by_protocol"):
        assert index["runs"]["matrix"]["groups"][axis][0]["path"] == "viewer/matrix/scores.parquet"


def test_viewer_distinguishes_identical_conditions_under_different_output_protocols():
    native, raw, _ = viewer_fixture()
    capped = deepcopy(native)
    capped["settings"]["authoring_protocol"] = "authoring-64-bounded-output-8192-v1"
    for row in capped["results"]:
        row["generation"] = {"output_limit": "bounded", "max_output_tokens": 8192,
                             "reasoning_effort": row["reasoning_effort"]}
    before = deepcopy((native, capped))
    index, shards = build_viewer_dataset({"native": (native, raw), "capped": (capped, raw)})
    old = _viewer_rows(shards["viewer/native/scores.parquet"])
    new = _viewer_rows(shards["viewer/capped/scores.parquet"])
    assert old[0]["condition_id"] == new[0]["condition_id"]
    assert old[0]["run_id"] == "native" and new[0]["run_id"] == "capped"
    assert old[0]["output_limit"] == "provider-native" and old[0]["max_output_tokens"] is None
    assert new[0]["output_limit"] == "bounded" and new[0]["max_output_tokens"] == 8192
    assert new[0]["reasoning_effort"] == "medium"
    assert new[0]["authoring_protocol"] == "authoring-64-bounded-output-8192-v1"
    assert len(index["splits"]["by_protocol"]) == 2
    assert (native, capped) == before
    assert old[0]["formal_fidelity_percent"] == new[0]["formal_fidelity_percent"] == 0
    capped["results"][0]["generation"]["max_output_tokens"] = 4096
    with pytest.raises(ValueError, match="output settings"):
        build_viewer_update("capped", capped, raw)


def test_viewer_protocol_expansion_rebuilds_retained_shards():
    from simready_bench.publishing import _viewer_columns, _viewer_schema_digest

    readout, raw, _ = viewer_fixture()
    index, _ = build_viewer_update("retained", readout, raw)
    fields = {"run_id", "authoring_protocol", "reasoning_effort", "output_limit", "max_output_tokens"}
    index["columns"] = _viewer_columns([field for field in index["columns"] if field not in fields])
    index["schema_sha256"] = _viewer_schema_digest(index["columns"])
    with pytest.raises(ViewerRebuildRequired):
        build_viewer_update("current", readout, raw, index=index)
    rebuilt, shards = build_viewer_dataset({"retained": (readout, raw), "current": (readout, raw)}, index=index)
    for run_id in rebuilt["runs"]:
        rows = _viewer_rows(shards["viewer/" + run_id + "/scores.parquet"])
        assert rows[0]["run_id"] == run_id
        assert rows[0]["authoring_protocol"] == "authoring-64-native-output-v1"


def test_viewer_split_collisions_never_rename_existing_groups():
    readout, raw, _ = viewer_fixture()
    readout["results"] = readout["results"][:1]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(readout["results"][0]))
    writer.writeheader()
    writer.writerows(readout["results"])
    raw = stream.getvalue()
    index, _ = build_viewer_update("first", readout, raw)
    first = deepcopy(index)
    changed = deepcopy(readout)
    changed["results"][0]["requested_model"] = "model-a"
    index, _ = build_viewer_update("second", changed, raw.replace("model/a", "model-a"), index=index)
    assert first["splits"]["by_model"][json.dumps("model/a")] == "model_a"
    assert index["splits"]["by_model"][json.dumps("model/a")] == "model_a"
    assert index["splits"]["by_model"][json.dumps("model-a")].startswith("model_a__")
    assert first["runs"].keys() == {"first"}
    repeated, _ = build_viewer_update("second", changed, raw.replace("model/a", "model-a"), index=index)
    assert repeated == index


def test_viewer_images_select_a_shared_camera_and_keep_missing_captures_null():
    readout, raw, _ = viewer_fixture()
    first = readout["results"][0]
    first["images"] = [
        {"view": "reference / isometric", "path": "asset-a/reconstruction-1/reference-iso.png"},
        {"view": "reference / front", "path": "asset-a/reconstruction-1/reference-front.png"},
        {"view": "candidate / front", "path": "asset-a/reconstruction-1/candidate-front.png"},
        {"view": "matched reference / reconstruction contact sheet", "path": "sheet.png"},
    ]
    before = deepcopy(readout)
    captures = {
        "runs/paired/evidence/asset-a/reconstruction-1/reference-front.png": _png("red"),
        "runs/paired/evidence/asset-a/reconstruction-1/candidate-front.png": _png("blue"),
    }
    index, shards = build_viewer_dataset({"paired": (readout, raw)}, image_reader=captures.__getitem__)
    records = _viewer_rows(shards["viewer/paired/scores.parquet"])
    assert records[0]["capture_view"] == "front"
    assert records[0]["reference_image"] == {"bytes": captures[next(iter(captures))], "path": "reference-front.png"}
    assert records[0]["candidate_image"] == {"bytes": captures[list(captures)[1]], "path": "candidate-front.png"}
    assert all(record["reference_image"] is None and record["candidate_image"] is None
               and record["capture_view"] == "" for record in records[1:])
    assert readout == before
    config = yaml.safe_load(render_viewer_card("# Card\n", index).split("---\n", 2)[1])["configs"][0]
    assert config["features"][:3] == [
        {"name": "reference_image", "dtype": "image"},
        {"name": "candidate_image", "dtype": "image"},
        {"name": "capture_view", "dtype": "string"},
    ]
    assert all(_viewer_table(content).column_names[:3] == ["reference_image", "candidate_image", "capture_view"] for content in shards.values())
    for content in shards.values():
        features = json.loads(_viewer_table(content).schema.metadata[b"huggingface"])["info"]["features"]
        assert features["reference_image"] == features["candidate_image"] == {"_type": "Image"}
        for record in _viewer_rows(content):
            for role in ("reference", "candidate"):
                image = record[role + "_image"]
                if image:
                    assert ":" not in image["path"] and "/" not in image["path"]
                    with Image.open(io.BytesIO(image["bytes"])) as decoded:
                        assert decoded.size == (8, 6)
                        assert decoded.getpixel((0, 0)) == ((255, 0, 0) if role == "reference" else (0, 0, 255))
    with pytest.raises(ValueError, match="image reader"):
        build_viewer_update("paired", readout, raw)
    first["images"][1]["path"] = "../outside.png"
    with pytest.raises(ValueError, match="inside its readout"):
        build_viewer_update("paired", readout, raw, image_reader=captures.__getitem__)


@pytest.mark.parametrize("failure", ["empty", "invalid", "changed", "wrong-format"])
def test_viewer_refuses_unreadable_or_changed_capture_bytes(failure):
    readout, raw, _ = viewer_fixture()
    capture = _png()
    metadata = {"view": "reference / isometric", "path": "reference.png",
                "bytes": len(capture), "sha256": hashlib.sha256(capture).hexdigest()}
    readout["results"][0]["images"] = [metadata]
    if failure == "empty":
        capture = b""
    elif failure == "invalid":
        capture = b"unreadable"
        metadata.update(bytes=len(capture), sha256=hashlib.sha256(capture).hexdigest())
    elif failure == "changed":
        capture = _png("blue")
    else:
        metadata["path"] = "reference.jpg"
    with pytest.raises(ValueError, match="capture|image"):
        build_viewer_update("images", readout, raw, image_reader=lambda _: capture)


def test_publication_migrates_v2_captures_using_the_pinned_revision(publication):
    manifest, report, hub, row, save = publication
    capture = _png()
    (report / "evidence/matched.png").write_bytes(capture)
    measured = row()
    measured["images"] = [{"path": "matched.png", "view": role + " / isometric",
                           "bytes": len(capture), "sha256": hashlib.sha256(capture).hexdigest()}
                          for role in ("reference", "candidate")]
    HuggingFacePublisher("team/results", "retained", manifest, api=hub)(save([measured]))
    index = json.loads(hub.files["viewer-index.json"])
    hub.files["viewer-index.json"] = json.dumps(_legacy_viewer_index(index, 2)).encode()
    hub.files["README.md"] = hub.files["README.md"].replace(b"simreadybench_viewer: 3", b"simreadybench_viewer: 2")
    original = {path: raw for path, raw in hub.files.items() if path.startswith("runs/retained/")}
    requested = []
    download = hub.hf_hub_download

    def pinned(repo, filename, **options):
        requested.append((filename, options["revision"]))
        return download(repo, filename, **options)

    hub.hf_hub_download = pinned
    HuggingFacePublisher("team/results", "current", manifest, api=hub)(save([measured]))
    assert all(hub.files[path] == value for path, value in original.items())
    assert not any(operation.path_in_repo.startswith("runs/retained/") for operation in hub.commits[-1]["operations"])
    retained_reads = [path for path, revision in requested if path.startswith("runs/retained/evidence/")]
    assert len(retained_reads) == 2 and all(revision == "head-1" for _, revision in requested)
    assert not any(path.startswith("runs/current/evidence/") for path, _ in requested)
    for run in ("retained", "current"):
        embedded = _viewer_rows(hub.files["viewer/" + run + "/scores.parquet"])[0]
        assert embedded["reference_image"]["bytes"] == embedded["candidate_image"]["bytes"] == capture
        assert embedded["capture_view"] == "isometric"
    assert json.loads(hub.files["viewer-index.json"])["splits"] == index["splits"]


def test_reference_and_unknown_reasoning_do_not_guess_scientific_identity():
    readout, raw, _ = viewer_fixture()
    reference = json.loads(files("simready_bench").joinpath("resources/benchmarks/nvidia-25.json").read_text())
    readout["settings"].update(reference_assets=[asset["id"] for asset in reference["assets"]],
                               reference_revision=reference["source"]["revision"])
    readout["settings"]["models"].append({"model_id": "model/a", "generation": {"reasoning_effort": "high"}})
    for row in readout["results"]:
        if row["requested_model"] == "model/a":
            del row["reasoning_effort"]
    index, _ = build_viewer_update("partial-nv25", readout, raw)
    assert index["runs"]["partial-nv25"]["groups"]["by_reference_set"] == [
        {"split": "nv25", "path": "viewer/partial-nv25/scores.parquet"},
    ]
    assert json.dumps(None) in index["splits"]["by_reasoning"]
    assert json.dumps("none") in index["splits"]["by_reasoning"]
    readout["settings"]["reference_revision"] = "another-revision"
    changed, _ = build_viewer_update("custom", readout, raw)
    assert changed["runs"]["custom"]["groups"]["by_reference_set"][0]["split"].startswith("custom_25_assets_")


def test_viewer_refuses_mismatched_rows_and_manual_config_name_collisions():
    readout, raw, _ = viewer_fixture()
    with pytest.raises(ValueError, match="factor identity"):
        build_viewer_update("matrix", readout, raw.replace("model/a", "wrong-model"))
    index, _ = build_viewer_update("matrix", readout, raw)
    manual = "---\nconfigs:\n- config_name: all\n  data_files: manual.csv\n---\n# Preserve me\n"
    with pytest.raises(ValueError, match="manual configuration"):
        render_viewer_card(manual, index)
    index["runs"]["matrix"]["groups"]["by_asset"][0]["path"] = "../private.csv"
    with pytest.raises(ValueError, match="owned score source"):
        render_viewer_card("", index)


def test_publication_is_one_atomic_commit_with_grouped_viewer(publication):
    manifest, _, hub, row, save = publication
    receipt = HuggingFacePublisher("team/results", "atomic", manifest, api=hub)(save([row()]))
    assert hub.uploads == [] and len(hub.commits) == 1
    paths = {operation.path_in_repo for operation in hub.commits[0]["operations"]}
    assert {"README.md", "eval.yaml", "leaderboard.json", "viewer-index.json", "runs/atomic/results.json", "runs/atomic/scores.csv"} <= paths
    assert receipt["run_commit"] == receipt["commit"] == "head-1"


def test_legacy_run_configs_migrate_without_rewriting_prior_results(publication):
    manifest, _, hub, row, save = publication
    HuggingFacePublisher("team/results", "legacy", manifest, api=hub)(save([row()]))
    retained = {path: raw for path, raw in hub.files.items() if path.startswith("runs/legacy/")}
    del hub.files["viewer-index.json"]
    hub.files["README.md"] = b"---\nlicense: mit\nconfigs:\n- config_name: legacy\n  default: true\n  data_files:\n  - split: test\n    path: runs/legacy/scores.csv\n- config_name: manual\n  data_files: manual.csv\n---\n# Retained scientific card\n"
    HuggingFacePublisher("team/results", "next", manifest, api=hub)(save([row()]))
    card = hub.files["README.md"].decode()
    configs = yaml.safe_load(card.split("---\n", 2)[1])["configs"]
    assert "---\n# Retained scientific card\n\n<!-- simreadybench-leaderboard:start -->" in card
    assert configs[0]["data_files"][0]["path"] == ["viewer/legacy/scores.parquet", "viewer/next/scores.parquet"]
    assert configs[-1] == {"config_name": "manual", "data_files": "manual.csv"}
    assert "legacy" not in {config["config_name"] for config in configs}
    assert all(hub.files[path] == raw for path, raw in retained.items())
    assert not any(operation.path_in_repo.startswith("runs/legacy/") for operation in hub.commits[-1]["operations"])


def _score_csv(rows, fields=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def test_viewer_unifies_legacy_and_current_headers_with_explicit_nullable_features():
    readout, old_csv, old_rows = viewer_fixture()
    new_rows = [{**row, "input_tokens": "123", "total_cost_usd": "", "extra": "0017"} for row in old_rows]
    new_csv = _score_csv(new_rows)
    before = old_csv.encode(), new_csv
    index, shards = build_viewer_dataset({"old": (readout, old_csv), "new": (readout, new_csv)})
    assert index["schema"] == "simreadybench-viewer-index/v3"
    assert (old_csv.encode(), new_csv) == before
    assert all(path.startswith("viewer/") for path in shards)
    assert all(_viewer_table(raw).column_names == index["columns"] for raw in shards.values())
    legacy = _viewer_rows(shards["viewer/old/scores.parquet"])
    current = _viewer_rows(shards["viewer/new/scores.parquet"])
    assert all(row["input_tokens"] is None and row["total_cost_usd"] is None and row["extra"] == "" for row in legacy)
    assert all(row["input_tokens"] == 123 and row["extra"] == "0017" for row in current)
    configs = yaml.safe_load(render_viewer_card("# Retained\n", index).split("---\n", 2)[1])["configs"]
    for config in configs:
        features = {item["name"]: item["dtype"] for item in config["features"]}
        assert all(features[field] == "float64" for field in USAGE_FIELDS)
        assert features["extra"] == "string"
        for split in config["data_files"]:
            assert all(path in shards and path.startswith("viewer/") for path in split["path"])


def test_viewer_schema_expansion_rebuilds_retained_shards_and_keeps_split_names():
    readout, raw, rows = viewer_fixture()
    first, original = build_viewer_dataset({"old": (readout, raw)})
    changed = _score_csv([{**row, "future_measurement": "1.2500"} for row in rows])
    with pytest.raises(ViewerRebuildRequired, match="all retained source tables"):
        build_viewer_update("new", readout, changed, index=first)
    with pytest.raises(ViewerRebuildRequired, match="All retained viewer runs"):
        build_viewer_dataset({"new": (readout, changed)}, index=first)
    updated, shards = build_viewer_dataset({"old": (readout, raw), "new": (readout, changed)}, index=first)
    assert updated["splits"] == first["splits"]
    assert updated["schema_sha256"] != first["schema_sha256"]
    assert {path for path in original} <= shards.keys()
    assert all(row["future_measurement"] == "" for row in _viewer_rows(shards["viewer/old/scores.parquet"]))
    assert all(row["future_measurement"] == "1.2500" for row in _viewer_rows(shards["viewer/new/scores.parquet"]))
    repeated, files_again = build_viewer_update("new", readout, changed, index=updated)
    assert repeated == updated
    assert files_again == {path: content for path, content in shards.items() if path.startswith("viewer/new/")}


def test_v1_viewer_migration_preserves_registry_and_card_body():
    readout, raw, _ = viewer_fixture()
    existing, _ = build_viewer_update("old", readout, raw)
    legacy = _legacy_viewer_index(existing, 1)
    updated, _ = build_viewer_dataset({"old": (readout, raw)}, index=legacy)
    assert updated == existing
    body = '<div align="center">Academic header and social image</div>\n\n# Methodology\n'
    card = "---\nlicense: mit\nsimreadybench_viewer: 1\nconfigs:\n- config_name: all\n  data_files: runs/old/scores.csv\n- config_name: manual\n  data_files: manual.csv\n---\n" + body
    rendered = render_viewer_card(card, updated)
    assert rendered.split("---\n", 2)[2] == body
    metadata = yaml.safe_load(rendered.split("---\n", 2)[1])
    assert metadata["license"] == "mit" and metadata["simreadybench_viewer"] == 3
    assert metadata["configs"][-1] == {"config_name": "manual", "data_files": "manual.csv"}
    for invalid in ("viewer/other/scores.parquet", "../private.csv"):
        tampered = deepcopy(updated)
        tampered["runs"]["old"]["viewer_scores"] = invalid
        with pytest.raises(ValueError, match="original run paths"):
            render_viewer_card(card, tampered)
    tampered = deepcopy(updated)
    tampered["schema_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="schema fingerprint"):
        render_viewer_card(card, tampered)


def test_publication_migrates_v1_mixed_headers_without_changing_raw_legacy_files(publication):
    manifest, _, hub, row, save = publication
    HuggingFacePublisher("team/results", "legacy", manifest, api=hub)(save([row()]))
    old_rows = list(csv.DictReader(io.StringIO(hub.files["runs/legacy/scores.csv"].decode())))
    stripped = [{key: value for key, value in item.items() if key not in USAGE_FIELDS} for item in old_rows]
    hub.files["runs/legacy/scores.csv"] = _score_csv(stripped)
    index = json.loads(hub.files["viewer-index.json"])
    legacy = _legacy_viewer_index(index, 1)
    hub.files["viewer-index.json"] = json.dumps(legacy).encode()
    retained = {path: raw for path, raw in hub.files.items() if path.startswith("runs/legacy/")}
    current = row()
    current["inference_calls_used"] = 4
    HuggingFacePublisher("team/results", "current", manifest, api=hub)(save([current]))
    assert all(hub.files[path] == raw for path, raw in retained.items())
    assert not any(item.path_in_repo.startswith("runs/legacy/") for item in hub.commits[-1]["operations"])
    assert _viewer_table(hub.files["viewer/legacy/scores.parquet"]).schema == _viewer_table(hub.files["viewer/current/scores.parquet"]).schema
    legacy_view = _viewer_rows(hub.files["viewer/legacy/scores.parquet"])[0]
    assert all(legacy_view[field] is None for field in USAGE_FIELDS)
    paths = []
    original_download = hub.hf_hub_download

    def downloaded(repo, filename, **options):
        paths.append(filename)
        return original_download(repo, filename, **options)

    hub.hf_hub_download = downloaded
    HuggingFacePublisher("team/results", "current", manifest, api=hub)(save([current]))
    assert not any(path.startswith("runs/legacy/") for path in paths)


def test_publication_schema_growth_rebuilds_old_viewer_files_in_one_commit(publication, monkeypatch):
    from simready_bench import publishing

    manifest, _, hub, row, save = publication
    HuggingFacePublisher("team/results", "legacy", manifest, api=hub)(save([row()]))
    retained = {path: raw for path, raw in hub.files.items() if path.startswith("runs/legacy/")}
    previous = json.loads(hub.files["viewer-index.json"])
    original_write = publishing.write_scores

    def expanded(path, rows):
        original_write(path, rows)
        records = list(csv.DictReader(io.StringIO(Path(path).read_text())))
        Path(path).write_bytes(_score_csv([{**item, "future_measurement": "007"} for item in records]))

    monkeypatch.setattr(publishing, "write_scores", expanded)
    HuggingFacePublisher("team/results", "next", manifest, api=hub)(save([row()]))
    assert all(hub.files[path] == raw for path, raw in retained.items())
    paths = {operation.path_in_repo for operation in hub.commits[-1]["operations"]}
    assert {"viewer/legacy/scores.parquet", "viewer/next/scores.parquet", "viewer-index.json", "README.md"} <= paths
    assert not any(path.startswith("runs/legacy/") for path in paths)
    current = json.loads(hub.files["viewer-index.json"])
    assert current["schema_sha256"] != previous["schema_sha256"]
    assert current["splits"] == previous["splits"]
    assert _viewer_rows(hub.files["viewer/legacy/scores.parquet"])[0]["future_measurement"] == ""
    assert _viewer_rows(hub.files["viewer/next/scores.parquet"])[0]["future_measurement"] == "007"


def test_privacy_change_prevents_any_atomic_publication(publication, monkeypatch):
    manifest, _, hub, row, save = publication
    calls = []

    def info(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(private=len(calls) == 1, sha="head-0")

    monkeypatch.setattr(hub, "repo_info", info)
    before = deepcopy(hub.files)
    with pytest.raises(PublicationError, match="visibility changed") as failure:
        HuggingFacePublisher("team/results", "private", manifest, api=hub)(save([row()]))
    assert not failure.value.retryable
    assert hub.commits == [] and hub.files == before


@pytest.mark.parametrize("turns", [12, 32, 128])
def test_publication_preserves_nondefault_authoring_protocol(publication, turns):
    manifest, _, hub, row, save = publication
    manifest["authoring_protocol"] = "authoring-" + str(turns) + "-native-output-v1"
    measured = row()
    measured["condition_id"], measured["seed"] = condition_identity(
        {"asset": measured["asset_id"], "harness": "pi", "model": "selected", "skill_set": "none"},
        repetition=0, master_seed=manifest["seed"], authoring_protocol=manifest["authoring_protocol"],
    )
    publisher = HuggingFacePublisher("team/results", "turns-" + str(turns), manifest, api=hub)
    assert publisher(save([measured]))["completed_conditions"] == 1
    index = json.loads(hub.files["viewer-index.json"])
    assert index["runs"]["turns-" + str(turns)]["groups"]["by_protocol"][0]["split"] == "authoring_" + str(turns) + "_native_output_v1"


def test_effective_reasoning_uses_defaults_then_matching_overrides(publication):
    manifest, _, hub, row, save = publication
    manifest["models"][0]["generation"]["reasoning_effort"] = "medium"
    manifest["defaults"]["generation"]["reasoning_effort"] = "low"
    manifest["overrides"] = [{"match": {"asset": "reference-a"}, "settings": {"generation": {"reasoning_effort": "high"}}}]
    publisher = HuggingFacePublisher("team/results", "effective", manifest, api=hub)
    publisher(save([row(), row(asset="reference-b")]))
    public = json.loads(hub.files["runs/effective/results.json"])
    assert {item["asset_id"]: item["reasoning_effort"] for item in public["results"]} == {"reference-a": "high", "reference-b": "low"}
    assert "reasoning_effort" not in next(csv.reader(io.StringIO(hub.files["runs/effective/scores.csv"].decode())))
    # A legacy missing provenance field may be enriched, but explicit values stay immutable.
    for item in public["results"]:
        del item["reasoning_effort"]
    hub.files["runs/effective/results.json"] = json.dumps(public).encode()
    publisher(save([row(), row(asset="reference-b")]))
    changed = deepcopy(manifest)
    changed["overrides"][0]["settings"]["generation"]["reasoning_effort"] = "medium"
    before = deepcopy(hub.files)
    with pytest.raises(PublicationError, match="reconstruction has changed|change settings"):
        HuggingFacePublisher("team/results", "effective", changed, api=hub)(save([row(), row(asset="reference-b")]))
    assert hub.files == before


@pytest.mark.parametrize("fault", ["conflict", "lost-response"])
def test_complete_set_publication_replays_keep_prior_and_concurrent_entries(
    publication, monkeypatch, fault,
):
    manifest, report, hub, row, save = publication
    reference = json.loads(files("simready_bench").joinpath("resources/benchmarks/nvidia-25.json").read_text())
    manifest.update(
        assets=[{"id": asset["id"]} for asset in reference["assets"]],
        source=deepcopy(reference["source"]),
        authoring_protocol=reference["authoring_protocol"],
        repetitions=4,
    )
    retained = {
        "model": "earlier-model", "harness": "opencode", "skills": [],
        "reasoning_effort": None, "authoring_protocol": "authoring-32-native-output-v1",
        "results_url": "https://huggingface.co/datasets/team/results/blob/main/runs/earlier/report.html",
        "composite_scores": {"formal": {"score": 0, "status": "measured"}},
        "usage": {"input_tokens": {"mean": None, "measured_attempts": 0}},
    }
    hub.files["leaderboard.json"] = json.dumps({"schema": "simreadybench-leaderboard/v1", "entries": [retained]}).encode()
    rows = [row(asset["id"], repetition) for asset in manifest["assets"] for repetition in range(4)]
    callback = save(rows)
    original_source = (report / "results.json").read_bytes()
    original_commit = hub.create_commit
    attempted = []
    first_commit = {}

    def commit(**kwargs):
        attempted.append(kwargs["parent_commit"])
        if len(attempted) == 1:
            if fault == "conflict":
                concurrent = deepcopy(retained)
                concurrent.update(model="concurrent-model", results_url=retained["results_url"].replace("earlier", "concurrent"))
                hub.files["leaderboard.json"] = json.dumps({"schema": "simreadybench-leaderboard/v1", "entries": [retained, concurrent]}).encode()
                hub.files["README.md"] += b"\nExternal scientific note.\n"
                hub.revision += 1
                raise RuntimeError("Parent commit changed")
            original_commit(**kwargs)
            first_commit.update(hub.files)
            raise TimeoutError("Commit accepted before response was lost")
        return original_commit(**kwargs)

    monkeypatch.setattr(hub, "create_commit", commit)
    publisher = HuggingFacePublisher("team/results", "full", manifest, api=hub)
    with pytest.raises(PublicationError) as failure:
        publisher(callback)
    assert failure.value.retryable
    assert publisher(callback, final=True)["leaderboard_complete"]
    entries = json.loads(hub.files["leaderboard.json"])["entries"]
    assert entries[0] == retained
    assert sum(item["model"] == "another-model" for item in entries) == 1
    assert attempted == ["head-0", "head-1"]
    card = hub.files["README.md"].decode()
    assert card.count("<!-- simreadybench-leaderboard:start -->") == 1
    assert "| earlier-model | unknown | opencode | none |" in card
    assert "| another-model | high | pi | none |" in card
    assert "authoring-32-native-output-v1" in card and "authoring-64-native-output-v1" in card
    if fault == "conflict":
        assert len(entries) == 3
        assert "External scientific note." in card
        assert "| concurrent-model |" in card
    else:
        assert len(entries) == 2
        assert hub.files == first_commit
    assert (report / "results.json").read_bytes() == original_source
    paths = {operation.path_in_repo for operation in hub.commits[-1]["operations"]}
    assert {"leaderboard.json", "README.md", "eval.yaml", "runs/full/results.json"} <= paths
