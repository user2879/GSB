import unittest

from arcore.correlation import (
    approximate_correlate,
    compare_paths,
    exact_correlate,
    select_path,
)
from arcore.errors import DivergeError
from arcore.model import Alert, Topology
from arcore.spec import Parameters


def alert(alert_id, ts, service, fingerprint, status="OPEN"):
    return Alert(alert_id=alert_id, source_id="s", timestamp_ms=ts,
                 service_id=service, level="P1", fingerprint=fingerprint,
                 labels=(), status=status, seq=0)


TOPO = Topology.build([("svc-a", "svc-b"), ("svc-b", "svc-c")])
PARAMS = Parameters(
    silence_window_ms=10, correlation_window_ms=100,
    global_concurrency=2, service_concurrency=1, lease_ttl_ms=100,
    exact_alert_threshold=4, fingerprint_cardinality_threshold=4,
    layer_weights=(10, 5), fixed_point_scale=10000, saturation_max=1000000)


class CorrelationTest(unittest.TestCase):
    def setUp(self):
        self.alerts = [
            alert("a1", 0, "svc-a", "fp-x"),
            alert("a2", 50, "svc-b", "fp-x"),
            alert("a3", 60, "svc-a", "fp-y"),
            alert("a4", 300, "svc-c", "fp-z"),
        ]

    def test_exact_path_uses_time_and_topology(self):
        pairs = exact_correlate(self.alerts, TOPO, 100)
        keys = {tuple(p["pair"]) for p in pairs}
        self.assertIn(("a1", "a2"), keys)
        self.assertIn(("a1", "a3"), keys)  # same service is always related
        self.assertNotIn(("a2", "a4"), keys)  # outside time window

    def test_approximate_path_is_subset_of_exact(self):
        exact = exact_correlate(self.alerts, TOPO, 100)
        approximate = approximate_correlate(self.alerts, TOPO, 100)
        exact_keys = {tuple(p["pair"]) for p in exact}
        approx_keys = {tuple(p["pair"]) for p in approximate}
        self.assertTrue(approx_keys.issubset(exact_keys))
        self.assertIn(("a1", "a2"), approx_keys)  # shared fingerprint survives

    def test_compare_paths_reports_bounds_without_divergence(self):
        exact = exact_correlate(self.alerts, TOPO, 100)
        approximate = approximate_correlate(self.alerts, TOPO, 100)
        bounds = compare_paths(self.alerts, TOPO, 100, exact, approximate)
        self.assertEqual(bounds.structural_false_positive_upper_bound, 0)
        self.assertGreaterEqual(bounds.structural_miss_upper_bound,
                                bounds.measured_misses)

    def test_divergence_on_false_positive(self):
        exact = exact_correlate(self.alerts, TOPO, 100)
        fake = [{"pair": ["a4", "ghost"], "reason": "fake"}]
        with self.assertRaises(DivergeError) as ctx:
            compare_paths(self.alerts, TOPO, 100, exact, fake)
        self.assertEqual(ctx.exception.code, "E_DIVERGE")

    def test_path_selection_records_boundary_values(self):
        small = select_path(self.alerts, PARAMS)
        self.assertEqual(small.selected_path, "exact")
        self.assertIn("no scale rule breached", small.reason)
        big = self.alerts + [
            alert(f"b{i}", 1000 + i, "svc-c", f"fp-{i}") for i in range(5)]
        chosen = select_path(big, PARAMS)
        self.assertEqual(chosen.selected_path, "approximate")
        self.assertTrue(chosen.boundary["alert_threshold_breach"])

    def test_results_independent_of_input_order(self):
        reversed_alerts = list(reversed(self.alerts))
        self.assertEqual(
            exact_correlate(self.alerts, TOPO, 100),
            exact_correlate(reversed_alerts, TOPO, 100))
        self.assertEqual(
            approximate_correlate(self.alerts, TOPO, 100),
            approximate_correlate(reversed_alerts, TOPO, 100))


if __name__ == "__main__":
    unittest.main()
