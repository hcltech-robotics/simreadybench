import copy
from importlib.resources import files
import json
from pathlib import Path
import shutil
import subprocess
import pytest
from simready_bench.leaderboard import leaderboard_entry


def run():
    manifest = json.loads(
        files("simready_bench")
        .joinpath("resources/benchmarks/nvidia-25.json")
        .read_text()
    )
    manifest["repetitions"] = 4
    manifest["defaults"]["generation"]["reasoning_effort"] = "medium"
    rows = []
    for asset in manifest["assets"]:
        for repetition in range(4):
            rows.append(
                {
                    "asset_id": asset["id"],
                    "repetition": repetition,
                    "status": "completed",
                    "requested_model": "gpt-6.1-sol",
                    "harness_id": "pi",
                    "skill_set_id": "none",
                    "simready": {
                        "score_percent": 95 * 40 / 46,
                        "passed_count": 40,
                        "failed_count": 6,
                        "unknown_count": 0,
                        "applicable_count": 46,
                        "completion_bonus": 0,
                        "score_lower_bound": 95 * 40 / 46,
                        "score_upper_bound": 95 * 40 / 46,
                        "optional_score": {
                            "score_percent": 100 * 33 / 51,
                            "passed_count": 33,
                            "failed_count": 18,
                            "unknown_count": 0,
                            "applicable_count": 51,
                            "measured_count": 51,
                            "coverage_fraction": 1,
                            "lower_bound_percent": 100 * 33 / 51,
                            "upper_bound_percent": 100 * 33 / 51,
                        },
                    },
                    "fidelity": {
                        "components": {
                            "shape": {
                                "score_percent": 50,
                                "earned": 0.5,
                                "eligible": 1,
                                "measured": 1,
                                "coverage_fraction": 1,
                            },
                            **{
                                component: {
                                    "status": "not_applicable",
                                    "score_percent": None,
                                    "earned": 0,
                                    "eligible": 0,
                                    "measured": 0,
                                    "coverage_fraction": 0,
                                }
                                for component in ("rigid_parts", "joints", "materials")
                            },
                        }
                    },
                }
            )
    return manifest, {"results": rows}


def test_complete_set_aggregates_performance_and_preserves_inapplicable_components():
    manifest, result = run()
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["reference_set"] == "NV25"
    assert entry["completed_attempts"] == 100
    assert entry["reasoning_effort"] == "medium"
    assert entry["compliance"] == 95 * 40 / 46
    assert entry["mandatory"]["passed"] == 4000
    assert entry["mandatory"]["eligible"] == 4600
    assert entry["mandatory"]["measured"] == 4600
    assert entry["mandatory"]["completion_bonus"] == 0
    assert entry["optional"]["score"] == 100 * 33 / 51
    assert entry["optional"]["passed"] == 3300
    assert entry["optional"]["eligible"] == 5100
    assert entry["optional"]["coverage"] == 100
    assert entry["fidelity"]["shape"]["score"] == 50
    assert entry["fidelity"]["shape"]["coverage"] == 100
    assert entry["fidelity"]["joints"]["score"] is None
    assert entry["fidelity"]["joints"]["coverage"] == 0
    assert entry["fidelity"]["joints"]["status"] == "not_applicable"
    assert entry["fidelity"]["joints"]["not_applicable_attempts"] == 100
    assert entry["usage"]["input_tokens"] == {
        "mean": None,
        "measured_attempts": 0,
        "total_attempts": 100,
    }


def add_usage(row, index):
    row.update(
        inference_calls_used=index,
        validation_calls_used=index + 1,
        elapsed_seconds=10 * index,
        inference_cost_usd=index / 1000,
        compute_cost_usd=index / 2000,
        statistics={
            "input_tokens": 100 * index,
            "output_tokens": 10 * index,
            "model_request_seconds": 2 * index,
            "model_request_errors": index % 2,
            "compute_cost_usd": index / 2000,
        },
    )


def test_usage_means_cover_every_attempt_and_keep_units_separate():
    manifest, result = run()
    for index, row in enumerate(result["results"]):
        add_usage(row, index)
    usage = leaderboard_entry(
        result,
        manifest=manifest,
        results_url="https://example.org/results",
    )["usage"]
    expected = {
        "inference_calls_used": 49.5,
        "validation_calls_used": 50.5,
        "model_request_errors": 0.5,
        "elapsed_seconds": 495,
        "model_request_seconds": 99,
        "input_tokens": 4950,
        "output_tokens": 495,
        "total_tokens": 5445,
        "inference_cost_usd": 0.0495,
        "compute_cost_usd": 0.02475,
        "total_cost_usd": 0.07425,
    }
    assert set(usage) == set(expected)
    for name, value in expected.items():
        assert usage[name]["mean"] == pytest.approx(value)
        assert usage[name]["measured_attempts"] == 100
        assert usage[name]["total_attempts"] == 100


def test_partial_usage_retains_coverage_without_averaging_known_attempts():
    manifest, result = run()
    for index, row in enumerate(result["results"]):
        add_usage(row, index)
    del result["results"][0]["inference_cost_usd"]
    del result["results"][0]["statistics"]["output_tokens"]
    usage = leaderboard_entry(
        result,
        manifest=manifest,
        results_url="https://example.org/results",
    )["usage"]
    for name in (
        "inference_cost_usd",
        "total_cost_usd",
        "output_tokens",
        "total_tokens",
    ):
        assert usage[name] == {
            "mean": None,
            "measured_attempts": 99,
            "total_attempts": 100,
        }
    assert usage["compute_cost_usd"]["mean"] == pytest.approx(0.02475)
    assert usage["input_tokens"]["mean"] == 4950


def test_measured_zero_usage_remains_zero():
    manifest, result = run()
    for row in result["results"]:
        add_usage(row, 0)
        row["validation_calls_used"] = 0
    usage = leaderboard_entry(
        result,
        manifest=manifest,
        results_url="https://example.org/results",
    )["usage"]
    for metric in usage.values():
        assert metric == {"mean": 0, "measured_attempts": 100, "total_attempts": 100}


def test_completed_invalid_candidates_are_measured_zero_results():
    manifest, result = run()
    for row in result["results"]:
        row["simready"].update(
            status="invalid_candidate",
            score_percent=0,
            passed_count=0,
            failed_count=46,
            unknown_count=0,
            score_lower_bound=0,
            score_upper_bound=0,
        )
        row["simready"]["optional_score"].update(
            status="invalid_candidate",
            score_percent=None,
            passed_count=0,
            failed_count=0,
            unknown_count=51,
            measured_count=0,
            coverage_fraction=0,
            lower_bound_percent=0,
            upper_bound_percent=100,
        )
        row["fidelity"]["components"]["shape"].update(score_percent=0, earned=0)
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["reference_set"] == "NV25"
    assert entry["completed_attempts"] == 100
    assert entry["compliance"] == 0
    assert entry["mandatory"]["score"] == 0
    assert entry["mandatory"]["failed"] == 4600
    assert entry["mandatory"]["measured"] == 4600
    assert entry["mandatory"]["unknown_attempts"] == 0
    assert entry["optional"]["score"] is None
    assert entry["optional"]["unknown"] == 5100
    assert entry["optional"]["coverage"] == 0


def test_fidelity_mean_retains_credit_and_distinguishes_reference_coverage():
    manifest, result = run()
    for index, row in enumerate(result["results"]):
        row["fidelity"]["components"]["materials"] = {
            "score_percent": 100 if index < 50 else 0,
            "earned": 1 if index < 50 else 0,
            "eligible": 1 if index < 50 else 9,
            "measured": 1 if index < 50 else 6,
            "coverage_fraction": 0.25 if index < 50 else 0.75,
        }
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    material = entry["fidelity"]["materials"]
    assert material["score"] == 50
    assert material["earned"] == 50
    assert material["eligible"] == 500
    assert material["measured"] == 350
    assert material["measurement_coverage"] == 70
    assert material["coverage"] == 50
    assert material["measured_attempts"] == 100


def test_unknown_optional_attempt_preserves_bounds_without_averaging_it_away():
    manifest, result = run()
    optional = result["results"][0]["simready"]["optional_score"]
    optional.update(
        score_percent=None,
        passed_count=3,
        failed_count=7,
        unknown_count=41,
        measured_count=10,
        coverage_fraction=10 / 51,
        lower_bound_percent=100 * 3 / 51,
        upper_bound_percent=100 * 44 / 51,
    )
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["compliance"] == 95 * 40 / 46
    assert entry["optional"]["score"] is None
    assert entry["optional"]["passed"] == 3270
    assert entry["optional"]["unknown"] == 41
    assert entry["optional"]["measured"] == 5059
    assert entry["optional"]["coverage"] == 100 * 5059 / 5100
    assert entry["optional"]["lower_bound"] == pytest.approx(100 * 3270 / 5100)
    assert entry["optional"]["upper_bound"] == pytest.approx(100 * 3311 / 5100)
    assert entry["optional"]["unknown_attempts"] == 1


def test_missing_fidelity_eligibility_is_unknown_rather_than_inapplicable():
    manifest, result = run()
    del result["results"][0]["fidelity"]["components"]["shape"]
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    shape = entry["fidelity"]["shape"]
    assert shape["score"] is None
    assert shape["coverage"] is None
    assert shape["eligible"] is None
    assert shape["earned"] is None
    assert shape["status"] == "unknown"
    assert shape["unknown_eligibility_attempts"] == 1
    assert shape["not_applicable_attempts"] == 0


def test_missing_optional_results_do_not_claim_measured_coverage():
    manifest, result = run()
    for row in result["results"]:
        del row["simready"]["optional_score"]
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["optional"]["score"] is None
    assert entry["optional"]["coverage"] is None
    assert entry["optional"]["passed"] is None
    assert entry["optional"]["unknown_attempts"] == 100
    assert entry["optional"]["reported_attempts"] == 0


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js is required for the site renderer"
)
def test_site_renders_zero_scores_unknown_eligibility_and_not_applicable_results():
    manifest, result = run()
    for row in result["results"]:
        row["fidelity"]["components"]["shape"].update(score_percent=0, earned=0)
        del row["fidelity"]["components"]["materials"]
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const inputEntry = JSON.parse(fs.readFileSync(0, 'utf8'));
function element(tag) {
  return { tag, textContent: '', children: [], hidden: false,
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; } };
}
const nodes = new Map();
const context = vm.createContext({ inputEntry, URL,
  document: { querySelectorAll: () => [], createElement: element,
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, element('div')); return nodes.get(id); } },
  fetch: () => new Promise(() => {}) });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
assert.equal(vm.runInContext('isCompleteEntry(inputEntry)', context), true);
assert.equal(inputEntry.reference_set, 'NV25');
vm.runInContext('leaderboardEntries = [inputEntry]; renderLeaderboard();', context);
const cells = nodes.get('leaderboard-body').children[0].children;
const text = (node) => [node.textContent, ...node.children.map(text)].join(' ');
assert.equal(cells.length, 12);
assert.match(text(cells[4]), /4,000\/4,600 credit/);
assert.match(text(cells[5]), /^0\.00.*0\/100 credit.*100\/100 measured/);
assert.match(text(cells[6]), /^Not measured.*Reference coverage not measured.*100 with eligibility not measured/);
assert.match(text(vm.runInContext('metricCell(inputEntry.fidelity.joints)', context)), /^N\/A.*No eligible reference criteria/);
assert.match(text(vm.runInContext('metricCell({score:50,confidence_interval:{level:0.95,lower:40,upper:60}})', context)), /95% CI \[40\.00, 60\.00\]/);
for (const index of [7, 8, 9, 10]) assert.match(text(cells[index]), /^Not measured/);
delete inputEntry.usage;
assert.equal(vm.runInContext('isCompleteEntry(inputEntry)', context), true);
vm.runInContext('renderLeaderboard();', context);
const legacyCells = nodes.get('leaderboard-body').children[0].children;
for (const index of [7, 8, 9, 10]) assert.match(text(legacyCells[index]), /^Not measured/);
inputEntry.completed_attempts = 99;
assert.equal(vm.runInContext('isCompleteEntry(inputEntry)', context), false);
"""
    subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(Path(__file__).parents[1] / "site" / "app.js"),
        ],
        input=json.dumps(entry),
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.mark.skipif(
    shutil.which("node") is None, reason="Node.js is required for the site renderer"
)
def test_site_usage_cells_show_units_zero_and_incomplete_measurements():
    manifest, result = run()
    for row in result["results"]:
        add_usage(row, 0)
        row["elapsed_seconds"] = 120
        row["statistics"]["model_request_seconds"] = 30
        row["statistics"]["input_tokens"] = 1000
        row["statistics"]["output_tokens"] = 200
        row["inference_cost_usd"] = 0.025
    del result["results"][0]["compute_cost_usd"]
    del result["results"][0]["statistics"]["compute_cost_usd"]
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const inputEntry = JSON.parse(fs.readFileSync(0, 'utf8'));
function element(tag) {
  return { tag, textContent: '', children: [], hidden: false,
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; } };
}
const nodes = new Map();
const context = vm.createContext({ inputEntry, URL,
  document: { createElement: element,
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, element('div')); return nodes.get(id); } },
  fetch: () => new Promise(() => {}) });
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
vm.runInContext('leaderboardEntries = [inputEntry]; renderLeaderboard();', context);
const cells = nodes.get('leaderboard-body').children[0].children;
const text = (node) => [node.textContent, ...node.children.map(text)].join(' ');
assert.equal(cells.length, 12);
assert.match(text(cells[7]), /^0 model requests.*Validation calls: 1.*Model request errors: 0/);
assert.match(text(cells[8]), /^2 min elapsed time.*Model request time: 30 s/);
assert.match(text(cells[9]), /^1,200 tokens.*Input tokens: 1,000.*Output tokens: 200/);
assert.match(text(cells[10]), /^Not measured.*total cost: 99\/100 reconstructions measured.*Inference: USD 0\.025.*Compute: Not measured.*Compute: 99\/100 reconstructions measured/);
assert.match(text(cells[11]), /View/);
inputEntry.usage.inference_calls_used = { mean: 12, measured_attempts: 99, total_attempts: 100 };
vm.runInContext('renderLeaderboard();', context);
assert.match(text(nodes.get('leaderboard-body').children[0].children[7]), /^Not measured.*99\/100 reconstructions measured/);
assert.equal(vm.runInContext('usageCost(0)', context), 'USD 0.00');
assert.equal(vm.runInContext('usageCost(0.00001)', context), '<USD 0.0001');
"""
    subprocess.run(
        [
            shutil.which("node"),
            "-e",
            script,
            str(Path(__file__).parents[1] / "site" / "app.js"),
        ],
        input=json.dumps(entry),
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "duplicate_repeat",
        "blocked",
        "unknown",
        "model",
        "single_repeat",
        "profile",
        "optional_denominator",
    ],
)
def test_partial_or_incomparable_results_cannot_enter_leaderboard(failure):
    manifest, result = run()
    rows = result["results"]
    if failure == "missing":
        rows.pop()
    if failure == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    if failure == "duplicate_repeat":
        rows[1]["repetition"] = rows[0]["repetition"]
    if failure == "blocked":
        rows[-1]["status"] = "blocked"
    if failure == "unknown":
        rows[-1]["simready"]["score_percent"] = None
    if failure == "model":
        rows[-1]["requested_model"] = "another-model"
    if failure == "single_repeat":
        manifest["repetitions"] = 1
    if failure == "profile":
        manifest["scoring"]["profile"]["version"] = "3.2.0"
    if failure == "optional_denominator":
        rows[0]["simready"]["optional_score"]["applicable_count"] = 50
    with pytest.raises(ValueError):
        leaderboard_entry(
            result, manifest=manifest, results_url="https://example.org/results"
        )


@pytest.mark.parametrize(
    "damage", ["revision", "kind", "packages", "entrypoint", "freeze", "unbound"]
)
def test_same_asset_names_cannot_replace_the_pinned_nv25_reference(damage):
    manifest, result = run()
    if damage == "revision":
        manifest["source"]["revision"] = "a" * 40
    elif damage == "kind":
        manifest["source"]["kind"] = "local-archive"
    elif damage == "packages":
        manifest["source"]["packages"][-1] = manifest["source"]["packages"][0]
    elif damage == "entrypoint":
        manifest["assets"][0]["usd"] = "different.usda"
    elif damage == "freeze":
        member = next(iter(manifest["benchmark_freeze"]["source_files"].values()))
        member["sha256"] = "0" * 64
    else:
        del manifest["source"]
        del manifest["benchmark_freeze"]
    with pytest.raises(ValueError, match="NV25"):
        leaderboard_entry(
            result, manifest=manifest, results_url="https://example.org/results"
        )


def test_legacy_optional_provenance_can_be_absent_without_inventing_values():
    manifest, result = run()
    del manifest["source"]
    del manifest["authoring_protocol"]
    manifest["defaults"]["generation"].pop("reasoning_effort")
    manifest["models"][0]["generation"].pop("reasoning_effort", None)
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["authoring_protocol"] is None
    assert entry["reasoning_effort"] is None
    assert entry["benchmark_freeze"] == {"id": "nvidia-25", "version": "1"}
    assert entry["reference_revision"] == "f4860aa0c75607adb50a78bf96e77510bffc040f"


def test_pinned_source_without_optional_freeze_remains_admissible():
    manifest, result = run()
    del manifest["benchmark_freeze"]
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["benchmark_freeze"] is None
    assert entry["completed_attempts"] == 100


@pytest.mark.parametrize(
    "field,value",
    [
        ("requested_model", "another-model"),
        ("harness_id", "opencode"),
        ("skill_set_id", "physics"),
        ("harness_id", None),
        ("skill_set_id", None),
    ],
)
def test_uniform_rows_must_still_agree_with_the_declared_factors(field, value):
    manifest, result = run()
    for row in result["results"]:
        row[field] = value
    with pytest.raises(ValueError, match="declared"):
        leaderboard_entry(
            result, manifest=manifest, results_url="https://example.org/results"
        )


@pytest.mark.parametrize(
    "damage",
    [
        "mixed-protocol",
        "invalid-protocol",
        "changed-id",
        "duplicate-id",
        "changed-seed",
    ],
)
def test_recorded_protocol_and_condition_identities_cannot_drift(damage):
    from simready_bench.config import condition_identity

    manifest, result = run()
    for row in result["results"]:
        row["condition_id"], row["seed"] = condition_identity(
            {
                "asset": row["asset_id"],
                "harness": manifest["harnesses"][0]["id"],
                "model": manifest["models"][0]["id"],
                "skill_set": manifest["skill_sets"][0]["id"],
            },
            repetition=row["repetition"],
            master_seed=manifest["seed"],
            authoring_protocol=manifest["authoring_protocol"],
        )
    if damage == "mixed-protocol":
        result["results"][0]["authoring_protocol"] = "authoring-32-native-output-v1"
    elif damage == "invalid-protocol":
        manifest["authoring_protocol"] = "bounded"
    elif damage == "changed-id":
        result["results"][0]["condition_id"] = "changed"
    elif damage == "duplicate-id":
        result["results"][1]["condition_id"] = result["results"][0]["condition_id"]
    else:
        result["results"][0]["seed"] += 1
    with pytest.raises(ValueError):
        leaderboard_entry(
            result, manifest=manifest, results_url="https://example.org/results"
        )


def test_effective_reasoning_retains_executor_precedence_and_explicit_uniformity():
    manifest, result = run()
    manifest["models"][0]["generation"]["reasoning_effort"] = "high"
    manifest["defaults"]["generation"]["reasoning_effort"] = "low"
    manifest["overrides"] = [
        {
            "match": {"asset": "*"},
            "settings": {"generation": {"reasoning_effort": "medium"}},
        },
    ]
    for row in result["results"]:
        row["reasoning_effort"] = "medium"
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["reasoning_effort"] == "medium"
    result["results"][0]["reasoning_effort"] = "high"
    with pytest.raises(ValueError, match="effective reasoning"):
        leaderboard_entry(
            result, manifest=manifest, results_url="https://example.org/results"
        )


def test_asset_specific_reasoning_cannot_be_labelled_as_one_default_effort():
    manifest, result = run()
    manifest["overrides"] = [
        {
            "match": {"asset": manifest["assets"][0]["id"]},
            "settings": {"generation": {"reasoning_effort": "low"}},
        },
    ]
    with pytest.raises(ValueError, match="mix effective reasoning"):
        leaderboard_entry(
            result, manifest=manifest, results_url="https://example.org/results"
        )


def test_partial_reasoning_provenance_stays_unknown_when_no_effort_is_declared():
    manifest, result = run()
    manifest["defaults"]["generation"].pop("reasoning_effort")
    manifest["models"][0]["generation"].pop("reasoning_effort", None)
    result["results"][0]["reasoning_effort"] = "medium"
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["reasoning_effort"] is None


@pytest.mark.parametrize("turns", [12, 32, 128])
def test_admitted_nondefault_protocol_keeps_its_actual_frozen_identities(turns):
    from simready_bench.config import condition_identity

    manifest, result = run()
    manifest["authoring_protocol"] = "authoring-" + str(turns) + "-native-output-v1"
    manifest["defaults"]["budgets"]["inference_requests"] = turns
    for row in result["results"]:
        row["condition_id"], row["seed"] = condition_identity(
            {
                "asset": row["asset_id"],
                "harness": manifest["harnesses"][0]["id"],
                "model": manifest["models"][0]["id"],
                "skill_set": manifest["skill_sets"][0]["id"],
            },
            repetition=row["repetition"],
            master_seed=manifest["seed"],
            authoring_protocol=manifest["authoring_protocol"],
        )
    entry = leaderboard_entry(
        result, manifest=manifest, results_url="https://example.org/results"
    )
    assert entry["authoring_protocol"] == manifest["authoring_protocol"]
    assert entry["inference_requests_per_attempt"] == turns
