"""Numerical and provenance boundaries for the public Pareto projection."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("pareto_data", Path(__file__).parents[1] / "scripts/build_pareto_data.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def row(asset="a", repetition=0, score=50, usage=10):
    return {"condition_id": f"{asset}-{repetition}", "asset_id": asset, "repetition": repetition,
            "formal_fidelity_percent": score, "formal_fidelity_eligible": 46,
            "visual_fidelity_percent": score, "visual_fidelity_eligible": 2,
            "kinematic_fidelity_percent": score, "kinematic_fidelity_eligible": 1,
            "total_tokens": usage}


class ParetoDataTests(unittest.TestCase):
    def test_one_reconstruction_not_four_attempt_sum(self):
        rows = [row(repetition=n, usage=10) for n in range(4)]
        self.assertEqual(p.metric_summary(rows, "total_tokens")["value"], 10)

    def test_attempt_weighting_matches_leaderboard_after_missing_repetitions(self):
        rows = [row(repetition=n, usage=10) for n in range(4)] + [row("b", usage=100)]
        self.assertEqual(p.metric_summary(rows, "total_tokens")["value"], 28)

    def test_partial_usage_has_matching_score_cohort(self):
        metric = p.metric_summary([row(score=80, usage=10), row(repetition=1, score=20, usage=None)], "total_tokens")
        self.assertEqual(metric["scores"]["formal"]["value"], 80)
        self.assertEqual(metric["scores"]["formal"]["x_value"], 10)
        self.assertFalse(metric["complete"])
        self.assertFalse(metric["scores"]["formal"]["complete"])

    def test_kinematic_excludes_ineligible_resource_rows(self):
        a, b = row("a", usage=10), row("b", usage=100)
        b.update(kinematic_fidelity_percent=None, kinematic_fidelity_eligible=0)
        metric = p.metric_summary([a, b], "total_tokens")
        self.assertEqual(metric["value"], 55)
        self.assertEqual(metric["scores"]["kinematic"]["x_value"], 10)

    def test_missing_is_not_zero(self):
        metric = p.metric_summary([row(usage=None)], "total_tokens")
        self.assertIsNone(metric["value"])
        self.assertIsNone(metric["scores"]["visual"]["value"])
        self.assertEqual(metric["measured_attempts"], 0)

    def test_cost_upper_bound_retains_published_value_and_qualification(self):
        r = row()
        r.update(inference_cost_usd=0.125, inference_cost_kind="provider_usage_priced_upper_bound",
                 usage_reconstruction={"method": "unique-gateway-request-provider-usage-times-pinned-rates-v1",
                                       "invoice_verified": False,
                                       "cost_kind": "estimated inference cost upper bound"})
        entry = {"model": "m", "harness": "pi", "completed_attempts": 1}
        exported = p.build_run(entry, {"results": [r], "benchmark_complete": True}, run_id="run",
                               source_url="https://example.org", results_sha256="abc", trajectory_loader=lambda _: None)
        cost = exported["metrics"]["cost_usd_per_asset"]
        self.assertEqual(cost["value"], 0.125)
        self.assertEqual(cost["kind"], "estimated inference cost upper bound")
        self.assertIn("1 attempts: estimated inference cost upper bound", cost["source_description"])
        self.assertIn("provider usage times pinned rates", cost["source_description"])
        self.assertIn("invoice not verified", cost["source_description"])
        self.assertEqual(cost["scores"]["formal"]["x_value"], 0.125)
        self.assertEqual(exported["metrics"]["tokens_per_asset"]["kind"], p.KINDS["total_tokens"])
        self.assertNotIn("upper bound", exported["metrics"]["tokens_per_asset"]["source_description"])

    def test_cost_upper_bound_preserves_specific_source_evidence(self):
        r = row()
        r.update(inference_cost_usd=0.5, inference_cost_kind="provider_usage_priced_upper_bound",
                 _usage_source={"inference_cost_usd": "SHA-256-verified trajectory matched to retained provider request proofs"})
        cost = p.metric_summary([r], "inference_cost_usd")
        self.assertIn(r["_usage_source"]["inference_cost_usd"], cost["source_description"])
        self.assertIn("estimated inference cost upper bound", cost["source_description"])
        self.assertNotIn("invoice not verified", cost["source_description"])

    def test_cost_upper_bound_mixed_and_missing_rows_keep_measured_counts(self):
        upper = dict(row(), inference_cost_usd=0, inference_cost_kind="provider_usage_priced_upper_bound")
        other = dict(row(repetition=1), inference_cost_usd=2)
        missing = dict(row(repetition=2), inference_cost_usd=None, inference_cost_kind="provider_usage_priced_upper_bound")
        cost = p.metric_summary([upper, other, missing], "inference_cost_usd")
        self.assertEqual(cost["value"], 1)
        self.assertEqual(cost["measured_attempts"], 2)
        self.assertEqual(cost["total_attempts"], 3)
        self.assertFalse(cost["complete"])
        self.assertEqual(cost["kind"], "inference cost including estimated upper bounds")
        self.assertEqual(cost["source_description"], "1 attempts: estimated inference cost upper bound; 1 attempts: published per-attempt measurement")

    def test_missing_upper_bound_cost_does_not_qualify_other_measured_cost(self):
        missing = dict(row(), inference_cost_usd=None, inference_cost_kind="provider_usage_priced_upper_bound")
        known = dict(row(repetition=1), inference_cost_usd=0.25)
        cost = p.metric_summary([missing, known], "inference_cost_usd")
        self.assertEqual(cost["value"], 0.25)
        self.assertEqual(cost["kind"], p.KINDS["inference_cost_usd"])
        self.assertNotIn("upper bound", cost["source_description"])
        absent = p.metric_summary([missing], "inference_cost_usd")
        self.assertIsNone(absent["value"])
        self.assertEqual(absent["source_description"], "Unavailable in retained evidence")
        self.assertEqual(absent["measured_attempts"], 0)

    def test_cost_qualification_requires_explicit_upper_bound_kind(self):
        for kind in (None, "pricing-derived inference cost"):
            with self.subTest(kind=kind):
                r = dict(row(), inference_cost_usd=0.125, inference_cost_kind=kind,
                         usage_reconstruction={"method": "unique-gateway-request-provider-usage-times-pinned-rates-v1",
                                               "invoice_verified": False})
                cost = p.metric_summary([r], "inference_cost_usd")
                self.assertEqual(cost["kind"], p.KINDS["inference_cost_usd"])
                self.assertEqual(cost["source_description"], "1 attempts: published per-attempt measurement")

    def test_partial_scores_use_available_mean_and_retain_coverage(self):
        score = p.score_summary([row(score=80), row(repetition=1, score=None)], "formal")
        self.assertEqual(score["value"], 80)
        self.assertEqual(score["status"], "partial")
        self.assertEqual(score["coverage"], 0.5)
        self.assertFalse(score["complete"])
        self.assertIsNone(p.score_summary([row(score=None)], "formal")["value"])

    def test_family_is_independent_of_version_provider_and_harness(self):
        self.assertEqual(p.model_family("gpt-6.1-sol"), p.model_family("openai/gpt-6.2:baseten"))
        self.assertEqual(p.model_family("zai-org/GLM-5.3-Flash:baseten"), "GLM")
        self.assertEqual(p.model_family("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4:baseten"), "Nemotron")
        self.assertEqual(p.model_family("deepseek-ai/DeepSeek-V4.1-Flash-1017:baseten"), "DeepSeek")
        self.assertEqual(p.model_family("claude-sonnet-5-5"), "Claude")

    @staticmethod
    def score_runs(values):
        return [{"scores": {name: {"value": value, "complete": True}
                            for name, value in zip(p.SCORES, row)}} for row in values]

    def test_cumulative_uses_population_stdev_and_equal_component_weights(self):
        runs = self.score_runs([(10, 20, 0), (20, 40, 50), (30, 60, 100)])
        normalisation = p.add_cumulative_scores(runs)
        self.assertEqual(normalisation["ddof"], 0)
        self.assertEqual(normalisation["components"]["formal"]["configuration_count"], 3)
        self.assertAlmostEqual(normalisation["components"]["formal"]["standard_deviation"], (200 / 3) ** 0.5)
        self.assertAlmostEqual(runs[0]["scores"]["cumulative"]["value"], -(3 / 2) ** 0.5)
        self.assertEqual(runs[1]["scores"]["cumulative"]["value"], 0)
        self.assertAlmostEqual(runs[2]["scores"]["cumulative"]["value"], (3 / 2) ** 0.5)
        self.assertEqual(runs[0]["scores"]["formal"]["value"], 10)

    def test_cumulative_accepts_partial_means_but_requires_all_three_components(self):
        runs = self.score_runs([(10, 20, 0), (20, 40, None)])
        runs[0]["scores"]["formal"]["complete"] = False
        p.add_cumulative_scores(runs)
        self.assertAlmostEqual(runs[0]["scores"]["cumulative"]["value"], -2 / 3)
        self.assertEqual(runs[0]["scores"]["cumulative"]["status"], "partial")
        self.assertFalse(runs[0]["scores"]["cumulative"]["complete"])
        self.assertIsNone(runs[1]["scores"]["cumulative"]["value"])

    def test_zero_variance_component_contributes_zero(self):
        runs = self.score_runs([(50, 10, 20), (50, 30, 40)])
        normalisation = p.add_cumulative_scores(runs)
        self.assertEqual(normalisation["components"]["formal"]["standard_deviation"], 0)
        self.assertEqual(runs[0]["scores"]["cumulative"]["components"]["formal"], 0)
        self.assertAlmostEqual(runs[1]["scores"]["cumulative"]["value"], 2 / 3)

    def test_cumulative_is_invariant_to_positive_affine_component_scale(self):
        a = self.score_runs([(10, 15, 30), (40, 30, 20), (20, 60, 10)])
        b = self.score_runs([(25, 15, 30), (85, 30, 20), (45, 60, 10)])
        p.add_cumulative_scores(a)
        p.add_cumulative_scores(b)
        for first, second in zip(a, b):
            self.assertAlmostEqual(first["scores"]["cumulative"]["value"], second["scores"]["cumulative"]["value"])

    def test_cache_tokens_are_counted_once(self):
        r = row(usage=None)
        r["native_anthropic_usage"] = dict(input_tokens=3, cache_creation_input_tokens=20, cache_read_input_tokens=40, output_tokens=7)
        self.assertEqual(p.reconstruct_public(r)["total_tokens"], 70)

    def test_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            p.reconcile(row(usage=1), {"total_tokens": 2}, "fixture")

    def test_final_trajectory_total_must_match_steps(self):
        trajectory = {"steps": [{"metrics": {"prompt_tokens": 10, "completion_tokens": 5}}],
                      "final_metrics": {"total_prompt_tokens": 11}}
        with self.assertRaisesRegex(ValueError, "final metrics"):
            p.trajectory_usage(trajectory)

    def test_explicit_recovery_restores_missing_terminal_usage(self):
        r = row(usage=None)
        r["usage_reconstruction"] = {"recorded_calls": 1, "requests": [{"gateway_request_id": "a"}],
                                     "tariff": {"billing_basis": "input_output", "route_verified": True, "service_tier_verified": True,
                                                "usd_per_million_tokens": {"input": "1", "output": "2"}}}
        r["request_recovery"] = {"disposition": "verified_recovered", "resolved_request_ids": ["a"]}
        trajectory = {"steps": [{"metrics": {"prompt_tokens": 10, "completion_tokens": 5}}]}
        result = p.reconstruct_public(r, trajectory)
        self.assertEqual(result["total_tokens"], 15)
        self.assertAlmostEqual(result["inference_cost_usd"], 20 / 1_000_000)
        del r["request_recovery"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            p.reconstruct_public(r, trajectory)

    def test_private_duration_needs_all_request_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            r = row()
            r.update(inference_calls_used=2, model_requests=[{"gateway_request_id": "one", "elapsed_seconds": 5}])
            path.write_text(json.dumps({"results": [r]}))
            self.assertIsNone(p.private_receipt(path, "run")["attempts"][0]["model_request_seconds"])

    def test_recovered_failures_are_included_in_inference_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            r = row()
            r.update(inference_calls_used=1, model_requests=[
                {"gateway_request_id": "one", "elapsed_seconds": 5, "status": "completed"},
                {"gateway_request_id": "two", "elapsed_seconds": 7, "status": "completed", "upstream_status": 429}],
                request_recovery={"disposition": "verified_recovered", "resolved_request_ids": ["two"]})
            path.write_text(json.dumps({"results": [r]}))
            self.assertEqual(p.private_receipt(path, "run")["attempts"][0]["model_request_seconds"], 12)
            r["request_recovery"]["resolved_request_ids"] = ["one"]
            path.write_text(json.dumps({"results": [r]}))
            self.assertIsNone(p.private_receipt(path, "run")["attempts"][0]["model_request_seconds"])

    def test_unverified_pricing_is_not_promoted_to_cost(self):
        r = row(usage=None)
        r["usage_reconstruction"] = {"recorded_calls": 1, "pricing_derived_cost_complete": False,
                                     "requests": [{"gateway_request_id": "one", "input_tokens": 10, "output_tokens": 5,
                                                   "pricing_derived_inference_cost_usd": 0.5}]}
        result = p.reconstruct_public(r)
        self.assertEqual(result["total_tokens"], 15)
        self.assertNotIn("inference_cost_usd", result)

    def test_supplement_requires_published_hash_and_exact_selection(self):
        r = row()
        entry = {"model": "m", "harness": "h", "completed_attempts": 1}
        document = {"results": [r], "benchmark_complete": True, "private_evidence": {"source_report_sha256": "right"}}
        kwargs = dict(run_id="run", source_url="https://example.org", results_sha256="public", trajectory_loader=lambda _: None)
        receipt = {"run_id": "run", "source_results_sha256": "wrong", "attempts": [r]}
        with self.assertRaisesRegex(ValueError, "not bound"):
            p.build_run(entry, document, supplement=receipt, **kwargs)
        receipt["source_results_sha256"] = "right"
        receipt["attempts"] = [r, r]
        with self.assertRaisesRegex(ValueError, "exact selected attempts"):
            p.build_run(entry, document, supplement=receipt, **kwargs)

    def test_export_omits_private_material(self):
        r = row()
        r.update(prompt="SECRET", tool_calls=["SECRET"], skills=["SECRET"])
        entry = {"model": "m", "harness": "h", "completed_attempts": 1, "skills": ["SECRET", "Orichalcum@" + "a" * 40]}
        data = p.build_run(entry, {"results": [r], "benchmark_complete": True}, run_id="run", source_url="https://example.org",
                           results_sha256="abc", trajectory_loader=lambda _: None)
        self.assertNotIn("SECRET", json.dumps(data))
        self.assertIn("Orichalcum …aaaaaaaa", data["label"])
        self.assertNotIn("a" * 40, data["label"])

    def test_export_rejects_score_different_from_official_leaderboard(self):
        entry = {"model": "m", "harness": "h", "completed_attempts": 1,
                 "composite_scores": {"kinematic": {"score": 90}}}
        with self.assertRaisesRegex(ValueError, "does not match the leaderboard"):
            p.build_run(entry, {"results": [row(score=50)], "benchmark_complete": True}, run_id="run",
                        source_url="https://example.org", results_sha256="abc", trajectory_loader=lambda _: None)

    def test_frontier_compares_protocols_but_preserves_evaluator_boundary(self):
        base = {"model": "m", "harness": "pi", "completed_attempts": 1,
                "reference_revision": "reference", "profile": {"id": "Robotics-Prop", "version": "4.0.0"},
                "evaluator_version": "source-usd-fidelity-v7"}
        doc = {"results": [row()], "benchmark_complete": True}
        def make(entry, identity):
            return p.build_run(entry, doc, run_id=identity, source_url="https://example.org", results_sha256="abc", trajectory_loader=lambda _: None)
        a = make(dict(base, authoring_protocol="authoring-64-native-output-v1"), "a")
        b = make(dict(base, authoring_protocol="authoring-64-bounded-output-4096-v1"), "b")
        self.assertEqual(a["frontier_group"], b["frontier_group"])
        c = make(dict(base, evaluator_version="source-usd-fidelity-v4", harness="codex"), "c")
        self.assertNotEqual(a["frontier_group"], c["frontier_group"])

    def test_bound_private_receipt_resolves_missing_evaluator(self):
        r = row()
        doc = {"results": [r], "benchmark_complete": True, "private_evidence": {"source_report_sha256": "private"}}
        receipt = {"run_id": "run", "source_results_sha256": "private", "evaluator_version": "source-usd-fidelity-v4", "attempts": [r]}
        result = p.build_run({"model": "m", "harness": "codex", "completed_attempts": 1}, doc,
                             run_id="run", source_url="https://example.org", results_sha256="public", trajectory_loader=lambda _: None, supplement=receipt)
        self.assertEqual(result["evaluator_version"], "source-usd-fidelity-v4")


if __name__ == "__main__":
    unittest.main()


def test_pareto_snapshot_fetches_prefixed_results_without_changing_pinned_entry_url(tmp_path, monkeypatch):
    revision = 'a' * 40
    original_url = f'{p.BASE}/blob/{revision}/runs/earlier/report.html'
    entry = {'model': 'm', 'harness': 'pi', 'reference_count': 25, 'repetitions': 4,
             'completed_attempts': 100, 'results_url': original_url}
    board_url = f'{p.BASE}/resolve/{revision}/results/leaderboard.json'
    run_url = f'{p.BASE}/resolve/{revision}/results/runs/earlier/results.json'
    responses = {board_url: json.dumps({'entries': [entry]}).encode(),
                 run_url: json.dumps({'results': [row(str(asset), repetition) for asset in range(25) for repetition in range(4)],
                                      'benchmark_complete': True}).encode()}
    requested = []
    def local_bytes(url, cache):
        requested.append(url)
        return responses[url]
    monkeypatch.setattr(p, 'get_bytes', local_bytes)
    snapshot = p.build_snapshot(revision, tmp_path, {})
    assert requested == [board_url, run_url]
    assert snapshot['source_url'] == board_url
    assert snapshot['runs'][0]['source_url'] == run_url
    assert entry['results_url'] == original_url
