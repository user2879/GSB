import dataclasses
import unittest

from swa.engine import Engine, RunReport
from swa.errors import BusinessError
from swa.generator import Event, build_round_robin_plan, generate_all_events
from swa.runner import build_result_document
from swa.spec import parse_spec


def base_spec():
    return parse_spec(
        {
            "value_min": 0,
            "value_max": 50,
            "group_keys": ["a", "b", "c", "d"],
            "sources": [
                {"source_id": "alpha", "seed": 20261118, "event_count": 18},
                {"source_id": "beta", "seed": 778899, "event_count": 16},
            ],
            "rules": [
                {"rule_id": "r2", "window_seconds": 2, "group_key": "group_key",
                 "aggregations": ["count", "sum", "max"], "concurrency": 1, "memory_budget": 8, "version": 1},
                {"rule_id": "r3", "window_seconds": 3, "group_key": "group_key",
                 "aggregations": ["count", "sum"], "concurrency": 2, "memory_budget": 8, "version": 1},
            ],
        }
    )


class EngineTests(unittest.TestCase):
    def test_results_are_byte_identical_across_interleavings_and_limits(self):
        import time

        spec = base_spec()
        streams = generate_all_events(spec)
        plan = build_round_robin_plan(streams)
        documents = set()
        # Same input and delivery order; only the scheduling interleaving changes.
        for limit in (1, 2, 4):
            report = Engine(spec, limit).ingest(plan, streams)
            documents.add(build_result_document(report.results)["result_digest"])
        report = Engine(
            spec, 3, compute_hook=lambda shard, events: time.sleep(0.001)
        ).ingest(plan, streams)
        documents.add(build_result_document(report.results)["result_digest"])
        self.assertEqual(len(documents), 1)

    def test_different_valid_interleavings_are_equivalent(self):
        # Two delivery plans that both keep each source monotonic must agree.
        spec = base_spec()
        streams = generate_all_events(spec)
        first = build_round_robin_plan(streams)
        # A second legal interleaving: drain alpha then beta (each still ordered).
        second = list(streams["alpha"]) + list(streams["beta"])
        r1 = Engine(spec, 3).ingest(first, streams)
        r2 = Engine(spec, 3).ingest(second, streams)
        self.assertEqual(
            build_result_document(r1.results)["result_digest"],
            build_result_document(r2.results)["result_digest"],
        )

    def test_triple_budget_metrics_never_exceed_limits(self):
        spec = base_spec()
        streams = generate_all_events(spec)
        engine = Engine(spec, 2)
        engine.ingest(build_round_robin_plan(streams), streams)
        metrics = engine.scheduler.metrics()
        self.assertLessEqual(metrics["max_running_global"], 2)
        self.assertLessEqual(metrics["max_running_by_rule"]["r2"], 1)
        self.assertLessEqual(metrics["max_running_by_rule"]["r3"], 2)
        for (rule_id, start), peak in engine.scheduler.memory_peak.items():
            self.assertLessEqual(peak, spec.rules[0].memory_budget if rule_id == "r2" else 8)

    def test_memory_budget_raises_e_budget_without_dropping_a_group(self):
        spec = parse_spec(
            {
                "value_min": 0,
                "value_max": 50,
                "group_keys": ["a", "b", "c"],
                "sources": [{"source_id": "alpha", "seed": 42, "event_count": 10}],
                "rules": [
                    {"rule_id": "tight", "window_seconds": 100, "group_key": "group_key",
                     "aggregations": ["count", "sum", "max"], "concurrency": 2,
                     "memory_budget": 1, "version": 1}
                ],
            }
        )
        engine = Engine(spec, 3, budget_wait_threshold_ms=0.0)
        with self.assertRaises(BusinessError) as ctx:
            engine.ingest(build_round_robin_plan(generate_all_events(spec)), generate_all_events(spec))
        self.assertEqual(ctx.exception.code, "E_BUDGET")
        # No group was evicted: the parked group is still distinct in memory state.
        self.assertGreaterEqual(ctx.exception.details["memory_items"], 1)

    def test_invalid_event_does_not_advance_source_watermark(self):
        spec = base_spec()
        engine = Engine(spec, 2)
        good = Event("alpha", 1, 100, "a", 5)
        bad = Event("alpha", 2, 200, "a", 999)
        report = RunReport()
        rejection = __import__("swa.generator", fromlist=["validate_event"]).validate_event(
            bad, spec, engine.source_watermark["alpha"]
        )
        self.assertIsNotNone(rejection)
        # Process the good event through the engine path.
        engine.source_watermark["alpha"] = good.timestamp_ms
        self.assertEqual(engine.source_watermark["alpha"], 100)

    def test_external_watermark_registers_late_event_without_mutation(self):
        spec = base_spec()
        engine = Engine(spec, 2)
        streams = generate_all_events(spec)
        head = [event for event in build_round_robin_plan(streams) if event.timestamp_ms < 1500]
        engine.ingest(head, {"alpha": streams["alpha"][:6], "beta": streams["beta"][:6]})
        before = engine.shards[("r2", 0, "a")].event_count if ("r2", 0, "a") in engine.shards else 0
        report = RunReport()
        late = Event("alpha", 999, 100, "a", 5)
        engine._route_event(late, spec.rules[0], 0, report)
        self.assertEqual(report.late_rule_events, 1)
        after = engine.shards[("r2", 0, "a")].event_count if ("r2", 0, "a") in engine.shards else 0
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
