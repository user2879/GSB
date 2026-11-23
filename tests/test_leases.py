from __future__ import annotations

import threading
import time
import unittest

from eventbus_diag.errors import LeaseError
from eventbus_diag.leases import LeaseManager


class LeaseTests(unittest.TestCase):
    def make_manager(self, **overrides) -> LeaseManager:
        limits = {"a": 2, "b": 2, "c": 2}
        rates = {"orders": 100, "alerts": 100}
        limits.update(overrides.pop("subscriber_limits", {}))
        rates.update(overrides.pop("channel_rate_limits", {}))
        return LeaseManager(
            global_limit=overrides.pop("global_limit", 3),
            subscriber_limits=limits,
            channel_rate_limits=rates,
            **overrides,
        )

    def test_global_and_subscriber_limits_never_oversubscribe(self) -> None:
        manager = self.make_manager(global_limit=2, subscriber_limits={"a": 1})
        first = manager.acquire(
            batch_id="b1", subscriber_id="a", channel_id="orders", ttl_ms=2000
        )
        second = manager.acquire(
            batch_id="b2", subscriber_id="b", channel_id="orders", ttl_ms=2000
        )
        observed = {"max_global": 0, "max_a": 0, "third": None}
        started = threading.Event()

        def waiter() -> None:
            started.set()
            observed["third"] = manager.acquire(
                batch_id="b3", subscriber_id="a", channel_id="orders", ttl_ms=2000
            )

        thread = threading.Thread(target=waiter)
        thread.start()
        started.wait(1.0)
        for _ in range(20):
            observed["max_global"] = max(observed["max_global"], manager.active_count)
            observed["max_a"] = max(observed["max_a"], manager.active_for_subscriber("a"))
            time.sleep(0.005)
        self.assertLessEqual(observed["max_global"], 2)
        self.assertLessEqual(observed["max_a"], 1)
        manager.release(second.lease.lease_id, second.lease.version)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(observed["third"])
        manager.release(first.lease.lease_id, first.lease.version)
        manager.release(observed["third"].lease.lease_id, observed["third"].lease.version)
        self.assertEqual(manager.active_count, 0)

    def test_sliding_window_rate_limit_blocks_third_dispatch(self) -> None:
        manager = self.make_manager(
            global_limit=5, channel_rate_limits={"orders": 2}
        )
        first = manager.acquire(
            batch_id="r1", subscriber_id="a", channel_id="orders", ttl_ms=2000
        )
        second = manager.acquire(
            batch_id="r2", subscriber_id="b", channel_id="orders", ttl_ms=2000
        )
        manager.release(first.lease.lease_id, first.lease.version)
        manager.release(second.lease.lease_id, second.lease.version)
        started = time.monotonic()
        third = manager.acquire(
            batch_id="r3", subscriber_id="c", channel_id="orders", ttl_ms=2000
        )
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.9)
        self.assertEqual(third.wait.reason, "channel_rate_limit")
        manager.release(third.lease.lease_id, third.lease.version)

    def test_renewal_requires_holder_and_version(self) -> None:
        manager = self.make_manager()
        acquisition = manager.acquire(
            batch_id="b1", subscriber_id="a", channel_id="orders", ttl_ms=1000
        )
        lease = acquisition.lease
        original_expiry = lease.expires_mono_ms
        with self.assertRaises(LeaseError) as raised:
            manager.renew(
                lease.lease_id,
                subscriber_id="a",
                expected_version=lease.version + 1,
                extension_ms=2000,
            )
        self.assertEqual(raised.exception.code, "E_LEASE")
        current = manager.snapshot()[0]
        self.assertEqual(current.expires_mono_ms, original_expiry)
        with self.assertRaises(LeaseError):
            manager.renew(
                lease.lease_id,
                subscriber_id="b",
                expected_version=lease.version,
                extension_ms=2000,
            )
        renewed = manager.renew(
            lease.lease_id,
            subscriber_id="a",
            expected_version=lease.version,
            extension_ms=2000,
        )
        self.assertEqual(renewed.version, 2)
        self.assertGreater(renewed.expires_mono_ms, original_expiry)
        self.assertTrue(manager.validate(lease.lease_id, renewed.version))
        manager.release(renewed.lease_id, renewed.version)

    def test_expiry_uses_monotonic_clock_and_reports_batch(self) -> None:
        manager = self.make_manager()
        acquisition = manager.acquire(
            batch_id="expiring-batch", subscriber_id="a", channel_id="orders", ttl_ms=20
        )
        time.sleep(0.04)
        before = time.monotonic() * 1000.0
        expired = manager.expire()
        after = time.monotonic() * 1000.0
        self.assertEqual(len(expired), 1)
        self.assertEqual(expired[0].batch_id, "expiring-batch")
        self.assertGreaterEqual(expired[0].revoked_mono_ms, before)
        self.assertLessEqual(expired[0].revoked_mono_ms, after)
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(manager.active_for_subscriber("a"), 0)
        self.assertFalse(manager.validate(acquisition.lease.lease_id))

    def test_revoke_all_releases_quota(self) -> None:
        manager = self.make_manager()
        manager.acquire(batch_id="b1", subscriber_id="a", channel_id="orders", ttl_ms=1000)
        manager.acquire(batch_id="b2", subscriber_id="b", channel_id="orders", ttl_ms=1000)
        revoked = manager.revoke_all("version_replaced")
        self.assertEqual(sorted(item.batch_id for item in revoked), ["b1", "b2"])
        self.assertEqual(manager.active_count, 0)
        self.assertEqual(manager.limits_snapshot()["global_active"], 0)


if __name__ == "__main__":
    unittest.main()
