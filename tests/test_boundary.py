import os
import tempfile
import unittest

from bench.mw.config import PartitionRule, SessionSpec, canonical_dumps
from bench.mw.errors import StructuredLogger
from bench.mw.latency import compute_stats, percentile
from bench.mw.session import Session

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BoundaryTests(unittest.TestCase):
    def _tmp_session(self, spec):
        self.tmp = tempfile.TemporaryDirectory()
        logger = StructuredLogger(os.path.join(self.tmp.name, "l.ndjson"))
        session = Session(spec, os.path.join(self.tmp.name, "l.db"), logger)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(session.close)
        return session

    def test_zero_state_views_match_between_paths(self):
        spec = SessionSpec(
            out_of_order_ms=10, latency_budget_ms=5, scale=1,
            rules=(PartitionRule("p0", ("sum", "count")),
                   PartitionRule("p1", ("sum", "count"))), raw={})
        session = self._tmp_session(spec)
        self.assertEqual(session.cache_view_bytes(),
                         session.authoritative_view_bytes())
        result = session.verify_views()
        self.assertEqual(result["partitions"], 2)

    def test_empty_latency_stats(self):
        stats = compute_stats([], 3).as_dict()
        self.assertEqual(stats["samples"], 0)
        self.assertEqual(stats["mean_fixed"], 0)
        self.assertEqual(stats["error_bound_decimal"], "0")

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95), 10)
        self.assertEqual(percentile([3, 1, 2], 95), 3)

    def test_int64_sum_saturation_flag(self):
        spec = SessionSpec(
            out_of_order_ms=1, latency_budget_ms=1, scale=6,
            rules=(PartitionRule("p0", ("sum", "count")),), raw={})
        from bench.mw.config import Event, INT64_MAX
        session = self._tmp_session(spec)
        huge = INT64_MAX
        events = [Event("p0", i, i * 10, huge, i + 1) for i in range(3)]
        session.ingest_events(events)
        session.drain()
        view = session.ledger.cache_views(1, spec)[0]
        self.assertTrue(view["sum_saturated"])
        self.assertEqual(view["sum_fixed"], INT64_MAX)

    def test_canonical_serialization_is_byte_stable(self):
        payload = {"b": 1, "a": [3, 2, 1], "c": {"z": 1, "y": 2}}
        first = canonical_dumps(payload)
        again = canonical_dumps({"c": {"y": 2, "z": 1}, "a": [3, 2, 1],
                                 "b": 1})
        self.assertEqual(first, again)
        self.assertNotIn(" ", first)


if __name__ == "__main__":
    unittest.main()
