import time
import unittest

from batch_receipt_ledger.clock import FakeClock
from batch_receipt_ledger.errors import LeaseError
from batch_receipt_ledger.leases import (
    LeaseManager, STATE_RELEASED_DONE, STATE_RELEASED_EXPIRED)
from batch_receipt_ledger.spec import parse_spec

from tests._support import make_spec


class _FakeWorkerClient:
    def __init__(self, fail_version=None):
        self.fail_version = fail_version
        self.calls = []

    def request(self, message):
        self.calls.append(message)
        if message["type"] == "lease_renew" and \
                message["expected_version"] == self.fail_version:
            return {"type": "error", "code": "E_LEASE",
                    "message": "worker version mismatch"}
        return {"type": "lease_renewed",
                "lease_id": message["lease_id"],
                "lease_version": message["new_version"]}


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock(1000)
        raw = make_spec(lease_ms=100)
        self.spec = parse_spec(raw)
        self.manager = LeaseManager(self.spec, self.clock)

    def test_acquire_respects_worker_quota(self):
        lease1, _ = self.manager.acquire("b1", "worker-a")
        lease2, _ = self.manager.acquire("b2", "worker-a")
        self.assertEqual(self.manager.active_count(), 2)
        lease3, waited = self.manager.acquire("b3", "worker-a", wait_timeout_ms=50)
        self.assertIsNone(lease3)
        self.assertGreaterEqual(waited, 50)
        self.manager.release(lease1.lease_id)
        lease4, _ = self.manager.acquire("b4", "worker-a", wait_timeout_ms=5)
        self.assertIsNotNone(lease4)

    def test_real_monotonic_expiry_recycles_quota_once(self):
        real_spec = parse_spec(make_spec(lease_ms=30))
        manager = LeaseManager(real_spec, __import__(
            "batch_receipt_ledger.clock", fromlist=["MonotonicClock"]).MonotonicClock())
        lease, _ = manager.acquire("b1", "worker-a")
        time.sleep(0.08)
        manager.start_reaper(interval_ms=5)
        time.sleep(0.05)
        refreshed = manager.get(lease.lease_id)
        self.assertEqual(refreshed.state, STATE_RELEASED_EXPIRED)
        before = list(refreshed.history)
        manager.release(lease.lease_id, STATE_RELEASED_DONE)
        again = manager.get(lease.lease_id)
        self.assertEqual(again.state, STATE_RELEASED_EXPIRED)
        self.assertEqual(again.history, before)
        self.assertEqual(manager.active_count(), 0)
        manager.stop_reaper()

    def test_renew_optimistic_version_check(self):
        lease, _ = self.manager.acquire("b1", "worker-a")
        client = _FakeWorkerClient()
        renewed = self.manager.renew(
            lease.lease_id, lease.version, 200, worker_client=client)
        self.assertEqual(renewed.version, 2)
        self.assertEqual(renewed.expires_at, 1200)

        with self.assertRaises(LeaseError) as ctx:
            self.manager.renew(lease.lease_id, 99, 200, worker_client=client)
        self.assertEqual(ctx.exception.code, "E_LEASE")

    def test_worker_reject_keeps_lease_version(self):
        lease, _ = self.manager.acquire("b1", "worker-a")
        client = _FakeWorkerClient(fail_version=lease.version)
        with self.assertRaises(LeaseError):
            self.manager.renew(lease.lease_id, lease.version, 200,
                               worker_client=client)
        self.assertEqual(self.manager.get(lease.lease_id).version, 1)

    def test_expired_lease_cannot_renew(self):
        lease, _ = self.manager.acquire("b1", "worker-a")
        self.clock.advance(101)
        with self.assertRaises(LeaseError) as ctx:
            self.manager.renew(lease.lease_id, 1, 100)
        self.assertEqual(ctx.exception.code, "E_LEASE")
        self.assertEqual(
            self.manager.get(lease.lease_id).state, STATE_RELEASED_EXPIRED)

    def test_double_release_is_idempotent(self):
        lease, _ = self.manager.acquire("b1", "worker-a")
        self.manager.release(lease.lease_id, STATE_RELEASED_DONE)
        self.manager.release(lease.lease_id, STATE_RELEASED_DONE)
        self.assertEqual(
            self.manager.get(lease.lease_id).state, STATE_RELEASED_DONE)
        self.assertEqual(self.manager.active_count(), 0)


if __name__ == "__main__":
    unittest.main()
