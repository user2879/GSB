from __future__ import annotations

import unittest

from cluster_diag.latency import (
    CLOCK_BASIS,
    QUANTIZATION_ERROR_UPPER_BOUND_MS_X1000,
    TRUTH_SOURCE,
    aggregate_latency_events,
)
from cluster_diag.ledger import LatencyEvent


def _event(event_id: int, elapsed_us: int, exceeded: bool = False) -> LatencyEvent:
    return LatencyEvent(
        id=event_id,
        session_id="s",
        kind="ui_query",
        cluster_id="edge-a",
        revision=2,
        bound_root_revision=2,
        started_ns=event_id * 10,
        ended_ns=event_id * 10 + elapsed_us * 1000,
        elapsed_us=elapsed_us,
        elapsed_ms_x1000=elapsed_us,
        budget_ms_x1000=10_000,
        exceeded=exceeded,
        clock_basis=CLOCK_BASIS,
        truth_source=TRUTH_SOURCE,
        details={},
        error_code=None,
    )


class LatencyAggregationTests(unittest.TestCase):
    def test_fixed_point_mean_p95_max_are_integer(self) -> None:
        events = [_event(index + 1, value) for index, value in enumerate(range(1, 101))]
        stats = aggregate_latency_events(events)
        self.assertEqual(stats["clock_basis"], "time.monotonic_ns")
        self.assertEqual(stats["truth_source"], "sqlite:latency_events")
        self.assertEqual(stats["quantization_error_upper_bound_ms_x1000"], QUANTIZATION_ERROR_UPPER_BOUND_MS_X1000)
        group = stats["groups"][0]
        self.assertEqual(group["mean_ms_x1000"], 51)
        self.assertEqual(group["p95_ms_x1000"], 95)
        self.assertEqual(group["max_ms_x1000"], 100)
        for value in (group["mean_ms_x1000"], group["p95_ms_x1000"], group["max_ms_x1000"]):
            self.assertIsInstance(value, int)

    def test_p95_uses_nearest_rank_without_float_interpolation(self) -> None:
        stats = aggregate_latency_events([_event(1, 10), _event(2, 30), _event(3, 20)])
        group = stats["groups"][0]
        self.assertEqual(group["p95_ms_x1000"], 30)
        self.assertEqual(group["mean_ms_x1000"], 20)


if __name__ == "__main__":
    unittest.main()
