from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lateness_ledger.canonical import canonical_json
from lateness_ledger.engine import WindowEngine, run_engine
from lateness_ledger.spec import load_bundle
from tests.helpers import mutate_sample, sample_bundle


class SpyEngine(WindowEngine):
    def __init__(self, bundle) -> None:
        super().__init__(bundle)
        self.propagation_checks: list[dict[str, object]] = []

    def _propagate_displacement(self, **kwargs):
        before = {
            window_id: canonical_json(payload)
            for window_id, payload in self.results.items()
        }
        displacement = super()._propagate_displacement(**kwargs)
        closure = {step.window_id for step in displacement.affected}
        after = {
            window_id: canonical_json(payload)
            for window_id, payload in self.results.items()
        }
        outside_changed = sorted(
            window_id
            for window_id in before
            if after.get(window_id) != before[window_id] and window_id not in closure
        )
        self.propagation_checks.append(
            {
                "root": displacement.root_window_id,
                "closure": sorted(closure),
                "outside_changed": outside_changed,
                "order": displacement.stable_topology_order,
            }
        )
        return displacement


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="engine-test-")
        cls.results = {
            strategy: run_engine(sample_bundle(strategy))
            for strategy in ("source_max", "partition_min", "fixed_step")
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_same_event_stream_finalizes_same_window_key_set(self) -> None:
        ids = {
            strategy: {window["window_id"] for window in result.windows}
            for strategy, result in self.results.items()
        }
        self.assertEqual(ids["source_max"], ids["partition_min"])
        self.assertEqual(ids["partition_min"], ids["fixed_step"])
        self.assertEqual(len(ids["source_max"]), 36)
        for result in self.results.values():
            orders = [int(window["finalization_order"]) for window in result.windows]
            self.assertEqual(orders, list(range(1, len(orders) + 1)))
            self.assertTrue(result.finalization_flush)

    def test_repeated_run_has_byte_stable_conclusions(self) -> None:
        first = self.results["source_max"]
        second = run_engine(sample_bundle("source_max"))
        self.assertEqual(canonical_json(first.windows), canonical_json(second.windows))
        self.assertEqual(
            canonical_json([point.to_db_dict() for point in first.watermarks]),
            canonical_json([point.to_db_dict() for point in second.watermarks]),
        )

    def test_source_declaration_order_does_not_change_results(self) -> None:
        directory = Path(self.temp.name)

        def reverse_sources(doc):
            doc["sources"] = list(reversed(doc["sources"]))

        sources, rules, budgets = mutate_sample(
            directory, source_mutator=reverse_sources
        )
        permuted = load_bundle(sources, rules, budgets)
        baseline = sample_bundle("source_max")
        self.assertEqual(permuted.spec_digest, baseline.spec_digest)
        result = run_engine(permuted)
        self.assertEqual(
            canonical_json(result.windows),
            canonical_json(self.results["source_max"].windows),
        )

    def test_late_budget_exact_boundary_law(self) -> None:
        found_exhausted = False
        for _strategy, result in self.results.items():
            for window in result.windows:
                late = window["late_decisions"]
                initial_count = int(late["budget_initial"]["event_count_limit"])
                initial_abs = int(late["budget_initial"]["absolute_measure_limit_fixed"])
                accepted = late["accepted"]
                discarded = late["discarded"]
                self.assertEqual(
                    [int(item["event_seq"]) for item in accepted],
                    sorted(int(item["event_seq"]) for item in accepted),
                )
                self.assertEqual(
                    [int(item["event_seq"]) for item in discarded],
                    sorted(int(item["event_seq"]) for item in discarded),
                )
                self.assertLessEqual(len(accepted), initial_count)
                self.assertLessEqual(
                    sum(int(item["absolute_measure_fixed"]) for item in accepted),
                    initial_abs,
                )
                for item in accepted:
                    self.assertEqual(item["decision"], "accepted")
                    self.assertGreaterEqual(item["remaining_event_count"], 0)
                    self.assertGreaterEqual(item["remaining_absolute_measure_fixed"], 0)
                for item in discarded:
                    self.assertEqual(item["decision"], "discarded")
                    self.assertGreater(item["rejected_cumulative_absolute_fixed"], 0)
                remaining_count = int(late["budget_remaining"]["event_count"])
                remaining_abs = int(late["budget_remaining"]["absolute_measure_fixed"])
                if remaining_count == 0:
                    exhausted = any(
                        item["remaining_event_count"] == 0 for item in accepted
                    ) or initial_count == 0
                    if discarded:
                        self.assertTrue(
                            any(
                                item["rule"] == "discarded_after_event_count_budget"
                                for item in discarded
                            )
                        )
                    found_exhausted = found_exhausted or exhausted
                if remaining_abs == 0 and initial_abs > 0:
                    found_exhausted = True
        self.assertTrue(found_exhausted)

    def test_error_envelope_has_three_segments_and_dominant_threshold(self) -> None:
        for result in self.results.values():
            for window in result.windows:
                errors = window["errors"]
                segments = errors["segments_fixed"]
                self.assertEqual(
                    set(segments),
                    {
                        "quantization_truncation",
                        "late_admission",
                        "boundary_attribution",
                    },
                )
                self.assertEqual(
                    errors["aggregation_error_bound_fixed"],
                    sum(segments.values()),
                )
                self.assertEqual(errors["dominant_threshold_ratio"], "1/2")
                self.assertIn(
                    errors["dominant_segment"],
                    {
                        "none",
                        "quantization_truncation",
                        "late_admission",
                        "boundary_attribution",
                    },
                )
                self.assertEqual(window["fixed_contract"]["truncation_direction"], "toward_zero")
                self.assertIsInstance(window["displacement_trajectory"], list)
        changed_windows = [
            window
            for result in self.results.values()
            for window in result.windows
            if window["displacement_trajectory"]
        ]
        self.assertGreater(len(changed_windows), 0)

    def test_only_dependency_closure_is_recomputed(self) -> None:
        engine = SpyEngine(sample_bundle("source_max"))
        result = engine.run()
        self.assertGreater(len(engine.propagation_checks), 0)
        self.assertTrue(
            all(not check["outside_changed"] for check in engine.propagation_checks)
        )
        for check in engine.propagation_checks:
            closure = set(check["closure"])
            ordered = check["order"]
            self.assertEqual(
                ordered,
                sorted(
                    ordered,
                    key=lambda window_id: (
                        engine.plan_by_id[window_id].start_ms,
                        engine.plan_by_id[window_id].rule_depth,
                        engine.plan_by_id[window_id].rule.rule_id,
                        window_id,
                    ),
                ),
            )
            for window_id in closure:
                self.assertTrue(window_id.startswith(("base|", "derived|", "tail|")))
        self.assertGreater(result.recalculations, 0)
        for displacement in result.displacement_runs:
            self.assertGreaterEqual(len(displacement.stable_topology_order), 1)
            for step in displacement.affected:
                self.assertEqual(step.path[0], displacement.root_window_id)
                self.assertEqual(step.path[-1], step.window_id)
                self.assertTrue(step.basis)

    def test_scheduler_limits_are_recorded(self) -> None:
        for result in self.results.values():
            stats = result.scheduler_stats
            self.assertGreaterEqual(stats.tasks, 36)
            self.assertLessEqual(stats.max_global_active, 4)
            self.assertLessEqual(stats.max_partition_active, 1)
            self.assertGreaterEqual(stats.total_wait_ms, 0)
            self.assertEqual(stats.total_timeout_attempts, 0)


if __name__ == "__main__":
    unittest.main()
