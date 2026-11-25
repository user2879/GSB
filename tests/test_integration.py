"""End-to-end tests using real loopback subprocess backends."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from grayroute.backend_proc import BackendPool, is_port_free
from grayroute.batching import plan_batches
from grayroute.engine import GatewayEngine
from grayroute.ledger import Ledger
from grayroute.sharding import attribute_all
from grayroute.snapshot import build_snapshot, restore_or_replace
from grayroute.spec import load_spec
from grayroute.viewmodel import DashboardModel

ROOT = os.path.dirname(os.path.dirname(__file__))


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_spec(os.path.join(ROOT, "config"),
                             os.path.join(ROOT, "data", "requests.ndjson"))
        cls.hits = attribute_all(cls.spec)
        cls.batches = plan_batches(cls.spec, cls.hits)

    def test_ports_are_loopback_and_free_check(self):
        for backend in self.spec.backends.values():
            self.assertEqual(backend.host, "127.0.0.1")
            self.assertIn(type(is_port_free(backend.host, backend.port)),
                          (bool,))

    def test_full_pipeline_with_subprocess_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = os.path.join(tmp, "ledger.sqlite")
            invalid = {("worker-gray-b", 2)}
            with Ledger(ledger_path) as ledger, \
                    BackendPool(self.spec.backends) as pool:
                engine = GatewayEngine(self.spec, ledger, pool,
                                       max_global=3, per_worker_quota=1)
                result = engine.run(self.batches, self.hits, invalid)
                self.assertEqual(
                    sorted(result.committed_batches),
                    sorted(b.batch_id for b in self.batches))
                self.assertEqual(
                    result.invalidated_requests,
                    {"req-007", "req-008", "req-009"})
                snapshot = build_snapshot(
                    rule_set_version=1, worker_set_version=1,
                    param_version_seq=2,
                    ledger_summary=result.ledger_summary,
                    cursor=ledger.cursor() or {})
                snap_path = os.path.join(tmp, "snap.json")
                _, action = restore_or_replace(snap_path, snapshot)
                self.assertEqual(action, "replaced")
                model = DashboardModel(self.spec, self.hits, self.batches,
                                       result, invalid)
                consistency = model.consistency_check(ledger)
                self.assertTrue(consistency["byte_identical"])
                self.assertTrue(consistency["within_budget"])

            # reopen ledger: committed rows survive, format version present
            with Ledger(ledger_path) as reopened:
                self.assertEqual(reopened.format_version(), 1)
                self.assertEqual(
                    len(reopened.committed_batch_ids()), len(self.batches))
                cursor = reopened.cursor()
                self.assertEqual(cursor["committed_count"],
                                 len(self.batches))


if __name__ == "__main__":
    unittest.main()
