from __future__ import annotations

import unittest

from mixedprec_diag.latency import LatencyRecorder


class LatencyTests(unittest.TestCase):
    def test_fixed_point_aggregates_and_budget_violation(self) -> None:
        recorder = LatencyRecorder(0.010)
        recorder.record("view", None, 0, 2_000)
        recorder.record("batch", "B1", 10_000, 18_000)
        recorder.record("view", None, 20_000, 40_000)
        summary = recorder.summary()
        overall = summary["aggregates"]["__all__"]
        batch = summary["aggregates"]["B1"]
        self.assertEqual(overall["mean_us"], 10)
        self.assertEqual(overall["p95_us"], 20)
        self.assertEqual(overall["max_us"], 20)
        self.assertEqual(batch["count"], 1)
        self.assertEqual(summary["codes"], ["E_LATENCY"])
        self.assertEqual(summary["exceeded_query_ids"], [3])
        self.assertEqual(summary["fixed_point"]["raw_quantization_error_upper_ms"], 0.0005)
        self.assertEqual(summary["clock_basis"], "time.perf_counter_ns (monotonic)")
        self.assertEqual(summary["truth_source"], "ui_latency.queries")

    def test_measure_context_records_monotonic_clock_truth(self) -> None:
        times = iter([100, 5_600])
        recorder = LatencyRecorder(1.0, clock_ns=lambda: next(times))
        with recorder.measure("query_view", None):
            pass
        record = recorder.records()[0]
        self.assertEqual(record.start_mono_ns, 100)
        self.assertEqual(record.end_mono_ns, 5_600)
        self.assertEqual(record.duration_us, 6)
        self.assertFalse(record.exceeded_budget)

    def test_empty_records_are_serializable(self) -> None:
        summary = LatencyRecorder(16.0).summary()
        self.assertEqual(summary["query_count"], 0)
        self.assertEqual(summary["aggregates"]["__all__"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
