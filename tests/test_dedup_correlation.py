import unittest

from alert_pipeline.correlation import correlate_all
from alert_pipeline.dedup import Deduplicator, aggregate_alerts
from alert_pipeline.models import NormalizedAlert
from alert_pipeline.rules import RootCauseRule, RuleSpec
from alert_pipeline.topology import DependencyGraph


def rules(window=1000):
    return RuleSpec(
        declared_version=1, schema_version=3, default_silence_ms=window,
        window_version=7, rule_version=3,
        silence_by_fingerprint={}, silence_by_service={},
        rules=(RootCauseRule("r1", "*", True, False, 3),))


def alert(line, ts, service="svc-a", fp="fp_0123456789ab",
          version="v2", window=1000):
    return NormalizedAlert(
        alert_id=f"al_{line}", source_version=version, timestamp_ms=ts,
        service_id=service, severity="major", fingerprint=fp,
        silence_window_ms=window, labels=(), root_cause_refs=(),
        strategy_id="strategy_default", filled=(), line=line)


class DedupTest(unittest.TestCase):
    def test_closed_interval_boundary_and_integer_suppression(self):
        # window 1000 anchored at 0: ts=1000 suppressed (closed), ts=1001 opens.
        alerts = [alert(1, 0), alert(2, 500), alert(3, 1000),
                  alert(4, 1001), alert(5, 2001)]
        aggs = aggregate_alerts(alerts, rules())
        self.assertEqual(len(aggs), 2)
        first, second = aggs
        self.assertEqual((first.first_ts_ms, first.last_ts_ms,
                          first.suppressed_count), (0, 1000, 2))
        self.assertEqual(first.member_lines, (1, 2, 3))
        self.assertEqual(first.window_version, 7)
        self.assertEqual((second.first_ts_ms, second.suppressed_count),
                         (1001, 1))
        self.assertEqual(second.member_lines, (4, 5))

    def test_grouping_independent_of_traversal_order(self):
        alerts = [alert(1, 0, "svc-a", "fp_000000000001"),
                  alert(2, 10, "svc-b", "fp_000000000002"),
                  alert(3, 20, "svc-a", "fp_000000000001"),
                  alert(4, 30, "svc-b", "fp_000000000002")]
        forward = aggregate_alerts(list(alerts), rules())
        shuffled = aggregate_alerts(list(reversed(alerts)), rules())
        self.assertEqual([a.aggregate_id for a in forward],
                         [a.aggregate_id for a in shuffled])
        self.assertEqual(
            sorted(a.suppressed_count for a in forward),
            sorted(a.suppressed_count for a in shuffled))

    def test_incremental_state_round_trip_matches_batch(self):
        alerts = [alert(i, i * 400) for i in range(1, 9)]
        batch_result = sorted(aggregate_alerts(alerts, rules()),
                              key=lambda a: a.aggregate_id)

        # First half: capture sealed windows and the still-open snapshot.
        first = Deduplicator(rules())
        first_sealed: list = []
        for item in alerts[:4]:
            first_sealed.extend(first.feed(item))
        state = first.open_state()
        self.assertEqual(len(state), 1)

        # Recovered process restores the snapshot and continues the stream.
        restored = Deduplicator(rules())
        restored.restore_state(state)
        for item in alerts[4:]:
            restored.feed(item)
        # finalize() returns every aggregate sealed by the restored instance,
        # including the window that was open at the snapshot point.
        resumed = restored.finalize()

        # Full result = windows sealed before the snapshot plus the complete
        # sealed set produced by the recovered instance.
        full_ids = sorted(
            [a.aggregate_id for a in first_sealed] +
            [a.aggregate_id for a in resumed])
        self.assertEqual(len(full_ids), len(set(full_ids)))
        self.assertEqual(full_ids,
                         [a.aggregate_id for a in batch_result])


class CorrelationTest(unittest.TestCase):
    def _graph(self):
        return DependencyGraph({"services": [
            {"service_id": "gateway", "depends_on": ["auth", "order"]},
            {"service_id": "order", "depends_on": ["payment"]},
            {"service_id": "payment", "depends_on": []},
            {"service_id": "auth", "depends_on": []},
        ]})

    def test_upstream_candidates_sorted_by_layer_then_topo(self):
        graph = self._graph()
        spec = RuleSpec(
            declared_version=1, schema_version=3, default_silence_ms=0,
            window_version=1, rule_version=5, silence_by_fingerprint={},
            silence_by_service={},
            rules=(RootCauseRule("r", "*", True, False, 3),))
        agg = aggregate_alerts([alert(1, 0, "gateway")], spec)[0]
        corr = correlate_all([agg], graph, spec)[0]
        self.assertEqual(corr.rule_version, 5)
        self.assertEqual(corr.candidates[0].service_id, "gateway")
        self.assertEqual(corr.candidates[0].direction, "self")
        layer1 = [c.service_id for c in corr.candidates if c.layer == 1]
        self.assertEqual(layer1, sorted(layer1))
        paths = {c.service_id: c.path for c in corr.candidates}
        self.assertEqual(paths["payment"], ("gateway", "order", "payment"))

    def test_insertion_order_does_not_change_candidate_set(self):
        spec = rules()
        agg = aggregate_alerts([alert(1, 0, "gateway")], spec)[0]
        g1 = self._graph()
        permuted = DependencyGraph({"services": list(reversed([
            {"service_id": "gateway", "depends_on": ["auth", "order"]},
            {"service_id": "order", "depends_on": ["payment"]},
            {"service_id": "payment", "depends_on": []},
            {"service_id": "auth", "depends_on": []},
        ]))})
        c1 = correlate_all([agg], g1, spec)[0]
        c2 = correlate_all([agg], permuted, spec)[0]
        self.assertEqual([c.to_record() for c in c1.candidates],
                         [c.to_record() for c in c2.candidates])


if __name__ == "__main__":
    unittest.main()
