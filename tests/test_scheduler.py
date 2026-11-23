"""Scheduler quota / trace tests."""

import tempfile
import time
import unittest

from tests.helpers import make_spec_files
from workbench.engine import Engine
from workbench.ledger import Ledger
from workbench.scheduler import Scheduler
from workbench.spec import load_spec
from workbench.upstream_mgr import UpstreamManager


class SchedulerQuotaTest(unittest.TestCase):
    def test_dual_quota_respected_and_traced(self):
        tmp = tempfile.mkdtemp(prefix="wbsched-")
        make_spec_files(tmp)
        spec = load_spec(tmp)
        ledger = Ledger(tmp + "/ledger.db", scale=3,
                        services_digest=spec.services_digest,
                        keys_digest=spec.keys_digest)
        engine = Engine(spec, ledger, global_limit=3, per_service_quota=1,
                        upstream_timeout_s=1.0)
        with UpstreamManager(spec) as mgr:
            mgr.start()
            rows = engine.run_all(batch_size=8)
        self.assertLessEqual(engine.scheduler.global_gauge.peak, 3)
        for sid, gauge in engine.scheduler.service_gauges.items():
            self.assertLessEqual(gauge.peak, 1, sid)
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertGreaterEqual(row["wait_end_ms"], row["wait_start_ms"])
            self.assertGreaterEqual(row["decide_end_ms"],
                                    row["decide_start_ms"])
            if row["decision_code"] == "OK":
                self.assertGreaterEqual(row["fwd_end_ms"],
                                        row["fwd_start_ms"])
            else:
                self.assertEqual(row["fwd_start_ms"], -1)
        ledger.close()

    def test_quota_never_bypassed_under_load(self):
        quotas = {"svc-x": 1, "svc-y": 2}
        scheduler = Scheduler(global_limit=2, service_quotas=quotas)

        class Req:
            def __init__(self, seq, sid):
                self.seq = seq
                self.service_id = sid

        reqs = ([Req(i, "svc-x") for i in range(1, 9)] +
                [Req(i + 100, "svc-y") for i in range(1, 9)])

        def decide(req):
            time.sleep(0.01)
            return type("D", (), {"passed": False})()

        scheduler.run(reqs, decide, lambda req: None)
        self.assertLessEqual(scheduler.global_gauge.peak, 2)
        self.assertLessEqual(scheduler.service_gauges["svc-x"].peak, 1)
        self.assertLessEqual(scheduler.service_gauges["svc-y"].peak, 2)


if __name__ == "__main__":
    unittest.main()
