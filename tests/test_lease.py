from __future__ import annotations

import time
import unittest

from mixedmem.errors import DiagnosticError
from mixedmem.lease import LeasePool
from mixedmem.util import Clock


class FakeClock(Clock):
    def __init__(self) -> None:
        self.now = 1000

    def monotonic_ms(self) -> int:
        return self.now

    def advance(self, milliseconds: int) -> None:
        self.now += milliseconds


class LeaseTests(unittest.TestCase):
    def test_exclusive_budget_and_timeout(self) -> None:
        clock = FakeClock()
        pool = LeasePool(block_size_bytes=8, memory_budget_bytes=40, clock=clock)
        first = pool.acquire("a", "l1", 3, ttl_ms=100, plan_version=1)
        with self.assertRaises(DiagnosticError) as context:
            pool.acquire("b", "l2", 3, ttl_ms=100, plan_version=1, timeout_ms=0)
        self.assertEqual(context.exception.code, "E_BUDGET")
        self.assertEqual(pool.used_blocks(), 3)
        self.assertTrue(pool.verify_invariants()["ok"])
        pool.release(first.lease_id, batch_id="a")
        self.assertEqual(pool.used_blocks(), 0)

    def test_early_release_is_idempotent(self) -> None:
        clock = FakeClock()
        pool = LeasePool(block_size_bytes=8, memory_budget_bytes=32, clock=clock)
        lease = pool.acquire("a", "l1", 2, ttl_ms=100, plan_version=1)
        first = pool.release(lease.lease_id, batch_id="a", layer_id="l1")
        second = pool.release(lease.lease_id, batch_id="a", layer_id="l1")
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "released")
        self.assertEqual(pool.used_blocks(), 0)

    def test_ownership_mismatch_is_e_lease(self) -> None:
        pool = LeasePool(block_size_bytes=8, memory_budget_bytes=32)
        lease = pool.acquire("a", "l1", 2, ttl_ms=100, plan_version=1)
        with self.assertRaises(DiagnosticError) as context:
            pool.release(lease.lease_id, batch_id="wrong", layer_id="l1")
        self.assertEqual(context.exception.code, "E_LEASE")
        self.assertEqual(pool.used_blocks(), 2)

    def test_expiry_releases_and_reorders_blocks(self) -> None:
        clock = FakeClock()
        pool = LeasePool(block_size_bytes=8, memory_budget_bytes=48, clock=clock)
        first = pool.acquire("a", "l1", 3, ttl_ms=10, plan_version=1)
        clock.advance(10)
        expired = pool.expire_due()
        self.assertEqual([lease.lease_id for lease in expired], [first.lease_id])
        self.assertEqual(first.status, "expired")
        second = pool.acquire("b", "l2", 3, ttl_ms=10, plan_version=1)
        self.assertEqual(first.block_range, second.block_range)
        self.assertTrue(pool.verify_invariants()["ok"])

    def test_real_monotonic_clock_expiry(self) -> None:
        pool = LeasePool(block_size_bytes=8, memory_budget_bytes=16)
        lease = pool.acquire("a", "l1", 1, ttl_ms=1, plan_version=1)
        time.sleep(0.015)
        expired = pool.expire_due()
        self.assertEqual([item.lease_id for item in expired], [lease.lease_id])
        self.assertEqual(lease.status, "expired")
        with self.assertRaises(DiagnosticError) as context:
            pool.assert_active(lease.lease_id, batch_id="a")
        self.assertEqual(context.exception.code, "E_LEASE")

    def test_revoke_plan_keeps_history(self) -> None:
        pool = LeasePool(block_size_bytes=8, memory_budget_bytes=32)
        lease = pool.acquire("a", "l1", 2, ttl_ms=100, plan_version=3)
        revoked = pool.revoke_plan(3, reason="E_REVOKE:test")
        self.assertEqual(revoked[0]["status"], "revoked")
        self.assertEqual(pool.used_blocks(), 0)
        history = pool.all_leases()
        self.assertEqual(history[0]["status"], "revoked")
        self.assertNotIn(lease.lease_id, [item["lease_id"] for item in pool.active_leases()])


if __name__ == "__main__":
    unittest.main()
