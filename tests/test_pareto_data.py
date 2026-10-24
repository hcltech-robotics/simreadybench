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
