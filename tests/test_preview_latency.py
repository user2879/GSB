import os
import tempfile
import unittest

from workbench.engine import ScalingEngine
from workbench.errors import LatencyError, PreviewError
from workbench.latency import CLOCK_BASIS, LatencyRecorder, fixed_point_stats
from workbench.ledger import Ledger
from workbench.logging_utils import JsonlLogger
from workbench.preview import PreviewCache
from workbench.testing_utils import spec_from_payload, tmp_declaration
from workbench.sample_data import build_cluster


class LatencyTest(unittest.TestCase):
    def test_fixed_point_stats_order_independent(self):
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        one = fixed_point_stats(values, 1000)
        two = fixed_point_stats(list(reversed(values)), 1000)
        self.assertEqual(one, two)
        self.assertEqual(one.mean_ms_fp, sum(values) * 1000 // len(values))
        self.assertEqual(one.max_ms, 9)
        self.assertEqual(one.quantization_error_bound_ms_fp, 1)
        self.assertIn("monotonic", CLOCK_BASIS)

    def test_budget_violation_keeps_sample_and_logs(self):
        log_path = os.path.join(tempfile.mkdtemp(), "latency.jsonl")
        logger = JsonlLogger(log_path)
        ticks = iter([0, 10, 10, 80])
        recorder = LatencyRecorder(50, 1000, logger, strict=True,
                                   clock=lambda: next(ticks))
        recorder.commit(0, 10, "preview")
        with self.assertRaises(LatencyError):
            recorder.commit(10, 80, "preview")
        self.assertEqual(len(recorder.over_budget_samples()), 1)
        self.assertTrue(recorder.over_budget_samples()[0].over_budget)
        with open(log_path, encoding="utf-8") as handle:
            self.assertIn("E_LATENCY", handle.read())


class PreviewTest(unittest.TestCase):
    def _engine(self, base):
        logger = JsonlLogger(os.path.join(base, "logs.jsonl"))
        ledger = Ledger(os.path.join(base, "ledger.db"))
        recorder = LatencyRecorder(50, 1000, logger)
        preview = PreviewCache()
        return ledger, ScalingEngine(ledger, logger, recorder, preview), preview

    def test_preview_matches_authority_and_never_rewrites_ledger(self):
        base = tempfile.mkdtemp()
        ledger, engine, preview = self._engine(base)
        spec = spec_from_payload(build_cluster())
        version = engine.register_declaration(spec)
        engine.run_pipeline(spec, version)
        before = ledger.load_decisions(version)
        preview.verify_against_authority(before)

        target = before[2].allocations[0]
        preview.inject_fault(version, 2, target.service_id, "allocated",
                             target.allocated + 1)
        with self.assertRaises(PreviewError) as caught:
            preview.verify_against_authority(ledger.load_decisions(version))
        self.assertEqual(caught.exception.code, "E_PREVIEW")
        self.assertEqual(caught.exception.details["minute"], 2)
        self.assertEqual(caught.exception.details["field"], "allocated")
        after = ledger.load_decisions(version)
        self.assertEqual(
            [d.digest for d in before], [d.digest for d in after]
        )
        ledger.close()

    def test_preview_query_is_single_cache_read(self):
        preview = PreviewCache()
        spec = spec_from_payload(build_cluster())
        ledger_path = os.path.join(tempfile.mkdtemp(), "x.db")
        logger = JsonlLogger(os.path.join(tempfile.mkdtemp(), "x.jsonl"))
        ledger = Ledger(ledger_path)
        engine = ScalingEngine(ledger, logger, preview=preview)
        version = engine.register_declaration(spec)
        engine.run_pipeline(spec, version)
        entry = preview.query(version, 4)
        self.assertEqual(entry.minute, 4)
        ledger.close()


if __name__ == "__main__":
    unittest.main()
