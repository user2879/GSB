import unittest

from arcore.model import Alert, Topology
from arcore.rootcause import derive_root_causes
from arcore.stream import CLOSED, OPEN
from arcore.suppression import suppress_closed

TOPO = Topology.build([("svc-a", "svc-b"), ("svc-b", "svc-c")])


def alert(alert_id, ts, service, fingerprint, status=OPEN):
    return Alert(alert_id=alert_id, source_id="s", timestamp_ms=ts,
                 service_id=service, level="P1", fingerprint=fingerprint,
                 labels=(), status=status, seq=0)


class SuppressionRootCauseTest(unittest.TestCase):
    def test_closed_alerts_suppressed_within_silence_window(self):
        alerts = [
            alert("c1", 0, "svc-a", "fp-1", CLOSED),
            alert("c2", 100, "svc-a", "fp-1", CLOSED),
            alert("c3", 20_000, "svc-a", "fp-1", CLOSED),
        ]
        result = suppress_closed(alerts, silence_window_ms=5000)
        self.assertEqual(result.suppressed_count, 1)
        self.assertEqual(result.suppressed_alert_ids, ["c2"])
        self.assertEqual(result.participating_alert_ids,
                         ["c1", "c2", "c3"])

    def test_suppression_order_independent(self):
        alerts = [
            alert("c1", 0, "svc-a", "fp-1", CLOSED),
            alert("c2", 100, "svc-a", "fp-1", CLOSED),
            alert("o1", 20, "svc-b", "fp-2", OPEN),
        ]
        first = suppress_closed(alerts, 5000).to_dict()
        second = suppress_closed(list(reversed(alerts)), 5000).to_dict()
        self.assertEqual(first, second)

    def test_root_cause_layering_and_stable_ranking(self):
        alerts = [
            alert("a1", 0, "svc-a", "fp-a"),
            alert("a2", 10, "svc-a", "fp-a2"),
            alert("a3", 12, "svc-a", "fp-a3"),
            alert("b1", 20, "svc-b", "fp-b"),
            alert("c1", 30, "svc-c", "fp-c"),
        ]
        result = derive_root_causes(alerts, TOPO, (100, 50, 20), [])
        candidates = result.candidates
        by_service = {c.service_id: c for c in candidates}
        # Dependency direction a -> b -> c: c is the root-cause layer 0.
        self.assertEqual(by_service["svc-c"].layer, 0)
        self.assertEqual(by_service["svc-c"].layer_weight, 100)
        self.assertEqual(by_service["svc-a"].layer, 2)
        self.assertEqual(by_service["svc-a"].layer_weight, 20)
        self.assertEqual(candidates[0].service_id, "svc-c")
        # svc-a has two alerts and the heaviest root-side weight.
        scores = {c.service_id: c.score for c in candidates}
        self.assertEqual(scores["svc-a"], 60)
        self.assertEqual(scores["svc-c"], 100)
        self.assertEqual(by_service["svc-a"].derivation_path,
                         ["svc-a", "svc-b", "svc-c"])

    def test_root_cause_arrival_order_independent(self):
        alerts = [alert(f"a{i}", i, "svc-a", f"fp-{i}") for i in range(5)]
        first = derive_root_causes(alerts, TOPO, (10, 5), []).to_dict()
        second = derive_root_causes(
            list(reversed(alerts)), TOPO, (10, 5), []).to_dict()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
