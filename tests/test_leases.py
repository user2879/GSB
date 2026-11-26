from __future__ import annotations

import time
import unittest

from gsb.errors import DiagnosticError
from gsb.leases import LeaseManager


class LeaseTests(unittest.TestCase):
    def test_version_mismatch_does_not_extend_expiry(self) -> None:
        now = [1000]
        calls = []

        def remote_renew(shard_id, lease_id, version, ttl_ms):
            calls.append((shard_id, lease_id, version, ttl_ms))
            return {"ok": True, "lease_version": version + 1}

        manager = LeaseManager(clock=lambda: now[0], remote_renew=remote_renew)
        lease = manager.issue("a", 100)
        expires = lease.expires_at_ms
        with self.assertRaises(DiagnosticError) as caught:
            manager.renew(lease.lease_id, 2)
        self.assertEqual(caught.exception.code, "E_LEASE")
        self.assertEqual(manager.get(lease.lease_id).expires_at_ms, expires)
        self.assertEqual(calls, [])
        now[0] = 1040
        renewed = manager.renew(lease.lease_id, 1)
        self.assertEqual(renewed.lease_version, 2)
        self.assertEqual(renewed.renew_count, 1)
        self.assertEqual(renewed.expires_at_ms, 1140)
        self.assertEqual(calls, [("a", lease.lease_id, 1, 100)])

    def test_real_monotonic_expiry_reclaims_quota(self) -> None:
        manager = LeaseManager()
        lease = manager.issue("a", 20)
        self.assertEqual(manager.active_count("a"), 1)
        time.sleep(0.03)
        expired = manager.prune()
        self.assertEqual(manager.active_count("a"), 0)
        self.assertEqual(expired[0].lease_id, lease.lease_id)
        self.assertEqual(expired[0].state, "expired")

    def test_release_and_cancel(self) -> None:
        manager = LeaseManager()
        lease = manager.issue("a", 1000)
        released = manager.cancel(lease.lease_id)
        self.assertEqual(released.state, "released")
        self.assertEqual(manager.active_count("a"), 0)


if __name__ == "__main__":
    unittest.main()

