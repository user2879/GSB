import unittest

from batch_receipt_ledger.clock import FakeClock
from batch_receipt_ledger.errors import LeaseError
from batch_receipt_ledger.leases import LeaseManager
from batch_receipt_ledger.processes import ManagedWorker
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class LeaseRenewE2ETest(unittest.TestCase):
    def test_worker_optimistic_version_roundtrip(self):
        raw = make_spec(workers=1, lease_ms=500)
        spec = parse_spec(raw)
        clock = FakeClock(1000)
        manager = LeaseManager(spec, clock)
        with ManagedWorker(spec.workers[0], raw) as worker:
            lease, _ = manager.acquire("b1", "worker-a")
            with worker.client(2000, 2000) as client:
                manager.grant_on_worker(lease, client)
                renewed = manager.renew(
                    lease.lease_id, 1, 500, worker_client=client)
                self.assertEqual(renewed.version, 2)
                with self.assertRaises(LeaseError) as ctx:
                    manager.renew(
                        lease.lease_id, 1, 500, worker_client=client)
                self.assertEqual(ctx.exception.code, "E_LEASE")

    def test_expired_lease_rejects_renew_with_real_clock(self):
        import time
        from batch_receipt_ledger.clock import MonotonicClock

        raw = make_spec(workers=1, lease_ms=30)
        spec = parse_spec(raw)
        manager = LeaseManager(spec, MonotonicClock())
        lease, _ = manager.acquire("b1", "worker-a")
        time.sleep(0.06)
        with self.assertRaises(LeaseError) as ctx:
            manager.renew(lease.lease_id, 1, 100)
        self.assertEqual(ctx.exception.code, "E_LEASE")


if __name__ == "__main__":
    unittest.main()
