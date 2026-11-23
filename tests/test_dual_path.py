"""Dual-path consistency: incremental cache vs authoritative recompute."""

import os
import tempfile
import unittest

from tests.helpers import make_spec_files
from workbench.engine import Engine
from workbench.ledger import Ledger
from workbench.replay import (authoritative_recompute,
                              compare_cache_with_authoritative)
from workbench.spec import load_spec
from workbench.upstream_mgr import UpstreamManager


class DualPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="wbdual-")
        make_spec_files(self.tmp)
        self.spec = load_spec(self.tmp)
        self.ledger = Ledger(os.path.join(self.tmp, "ledger.db"), scale=3,
                             services_digest=self.spec.services_digest,
                             keys_digest=self.spec.keys_digest)
        self.engine = Engine(self.spec, self.ledger, global_limit=3,
                             per_service_quota=1, upstream_timeout_s=1.0)
        self.mgr = UpstreamManager(self.spec)
        self.mgr.start()
        self.engine.run_all(batch_size=3)

    def tearDown(self):
        self.mgr.stop_all()
        self.ledger.close()

    def test_cache_matches_authoritative_recompute(self):
        view = self.engine.quick_view()
        result = authoritative_recompute(self.spec, self.ledger)
        self.assertTrue(result.consistent, result.divergence)
        dual = compare_cache_with_authoritative(view, result)
        self.assertIsNone(dual, dual)

    def test_token_trajectory_identical_across_processing_orders(self):
        tmp2 = tempfile.mkdtemp(prefix="wbdual2-")
        make_spec_files(tmp2)
        spec2 = load_spec(tmp2)
        ledger2 = Ledger(os.path.join(tmp2, "ledger.db"), scale=3)
        engine2 = Engine(spec2, ledger2, global_limit=1,
                         per_service_quota=1, upstream_timeout_s=1.0)
        engine2.run_all(batch_size=2)
        rows1 = [(r["seq"], r["token_before"], r["token_refilled"],
                  r["token_after"]) for r in self.ledger.done_requests()]
        rows2 = [(r["seq"], r["token_before"], r["token_refilled"],
                  r["token_after"]) for r in ledger2.done_requests()]
        self.assertEqual(rows1, rows2)
        ledger2.close()

    def test_first_divergence_located(self):
        view = self.engine.quick_view()
        for item in view["conclusions"]:
            if item["seq"] == 2:
                item["token_after"] += 1
        result = authoritative_recompute(self.spec, self.ledger)
        dual = compare_cache_with_authoritative(view, result)
        self.assertIsNotNone(dual)
        self.assertEqual(dual.seq, 2)
        self.assertEqual(dual.field, "token_after")

    def test_ledger_tamper_detected_by_recompute(self):
        self.ledger.conn.execute(
            "UPDATE requests SET decision_code='E_RATE' WHERE seq=1")
        self.ledger.conn.commit()
        result = authoritative_recompute(self.spec, self.ledger)
        self.assertFalse(result.consistent)
        self.assertEqual(result.divergence.seq, 1)
        self.assertEqual(result.divergence.field, "decision_code")

    def test_observations_recorded(self):
        self.engine.quick_view()
        authoritative_recompute(self.spec, self.ledger)
        gui = self.ledger.observations("gui_query")
        rec = self.ledger.observations("recompute")
        self.assertGreaterEqual(len(gui), 1)
        self.assertGreaterEqual(len(rec), 1)
        scopes = {r["scope"] for r in rec}
        self.assertIn("overall", scopes)


if __name__ == "__main__":
    unittest.main()
