from __future__ import annotations

import time
import threading
import unittest

from feature_backfill.cache import LRUBlockCache
from feature_backfill.clock import MonotonicClock
from feature_backfill.compute import FeatureComputer, entity_lease_version
from feature_backfill.data_loader import rows_by_entity
from feature_backfill.errors import AppError, E_CONFIG_INVALID, E_LEASE_EXPIRED
from feature_backfill.leases import LeaseGuard, LeaseManager, shard_for_entity

from tests.helpers import load_fixture


class LeaseTests(unittest.TestCase):
    def make_manager(self, *, ttl_ms: int = 80) -> LeaseManager:
        return LeaseManager(
            clock=MonotonicClock("monotonic_ns"),
            ttl_ms=ttl_ms,
            renew_interval_ms=max(1, ttl_ms // 4),
            shard_count=4,
            wait_timeout_ms=3000,
        )

    def test_conflict_waits_for_real_expiry_then_reclaims(self) -> None:
        manager = self.make_manager(ttl_ms=80)
        first = manager.acquire("E001", owner="worker-A", version="v1")
        outcome: dict[str, object] = {}
        started = manager.clock.now_ns()

        def contender() -> None:
            outcome["lease"] = manager.acquire("E001", owner="worker-B", version="v1")
            outcome["finished"] = manager.clock.now_ns()

        thread = threading.Thread(target=contender, name="contender")
        thread.start()
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())
        second = outcome["lease"]
        self.assertEqual(second.owner, "worker-B")
        self.assertNotEqual(first.token, second.token)
        conflict = next(event for event in manager.events if event["event"] == "lease_conflict")
        reclaim = next(event for event in manager.events if event["event"] == "lease_reclaimed")
        self.assertEqual(conflict["code"], "E_LEASE_CONFLICT")
        self.assertLess(conflict["clock_reading_ns"], first.expires_ns)
        self.assertEqual(reclaim["previous_owner"], "worker-A")
        self.assertEqual(reclaim["basis"], "monotonic_deadline_reached")
        self.assertGreaterEqual(reclaim["reclaimed_at_ns"], first.expires_ns)
        self.assertGreaterEqual(int(outcome["finished"]), first.expires_ns)
        self.assertGreater(int(outcome["finished"]) - started, int(80 * 1_000_000 * 0.5))

    def test_guard_renews_before_expiry(self) -> None:
        manager = self.make_manager(ttl_ms=120)
        guard = LeaseGuard(
            manager,
            entity_id="E001",
            owner="worker-A",
            version="v1",
        )
        guard.__enter__()
        assert guard.lease is not None
        original_expiry = guard.lease.expires_ns
        while manager.clock.now_ns() <= original_expiry + 10_000_000:
            time.sleep(0.002)
        guard.validate()
        renewals = [event for event in manager.events if event["event"] == "lease_renewed"]
        self.assertGreaterEqual(len(renewals), 1)
        self.assertTrue(all(event["clock_reading_ns"] < event["new_expires_ns"] for event in renewals))
        guard.__exit__(None, None, None)

    def test_renewal_after_expiry_is_rejected_and_released(self) -> None:
        manager = self.make_manager(ttl_ms=25)
        lease = manager.acquire("E001", owner="worker-A", version="v1")
        while manager.clock.now_ns() < lease.expires_ns + 1_000_000:
            time.sleep(0.001)
        with self.assertRaises(AppError) as context:
            manager.renew("E001", lease.token)
        self.assertEqual(context.exception.code, E_LEASE_EXPIRED)
        self.assertFalse(manager.snapshot()["leases"])

    def test_zero_injected_ttl_is_config_invalid(self) -> None:
        manager = self.make_manager()
        with self.assertRaises(AppError) as context:
            manager.acquire("E001", owner="worker-A", version="v1", ttl_ms=0)
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

    def test_shard_mapping_is_stable(self) -> None:
        values = [shard_for_entity("E001", 4) for _ in range(5)]
        self.assertEqual(len(set(values)), 1)
        self.assertTrue(0 <= values[0] < 4)


class CacheTests(unittest.TestCase):
    def test_zero_cache_budget_is_rejected(self) -> None:
        with self.assertRaises(AppError) as context:
            LRUBlockCache(0)
        self.assertEqual(context.exception.code, E_CONFIG_INVALID)

    def test_evicted_blocks_are_recomputed_to_identical_values(self) -> None:
        loaded = load_fixture()
        grouped = rows_by_entity(loaded["rows"])
        versions = {
            entity: entity_lease_version(entity, rows, loaded["features"])
            for entity, rows in grouped.items()
        }
        rows = grouped["E001"]
        cache_large = LRUBlockCache(512)
        large = FeatureComputer(
            rows_by_entity={"E001": rows},
            features=loaded["features"],
            config=loaded["config"],
            cache=cache_large,
            lease_versions={"E001": versions["E001"]},
        ).compute_entity("E001")
        cache_tiny = LRUBlockCache(1)
        tiny = FeatureComputer(
            rows_by_entity={"E001": rows},
            features=loaded["features"],
            config=loaded["config"],
            cache=cache_tiny,
            lease_versions={"E001": versions["E001"]},
        ).compute_entity("E001")
        self.assertEqual(
            [(point.feature, point.row_id, point.value) for point in large.points],
            [(point.feature, point.row_id, point.value) for point in tiny.points],
        )
        recomputes = [event for event in tiny.cache_events if event["kind"] == "recompute"]
        recompute_counts: dict[str, int] = {}
        for event in recomputes:
            recompute_counts[event["key"]] = recompute_counts.get(event["key"], 0) + 1
        self.assertTrue(any(count > 1 for count in recompute_counts.values()))
        self.assertGreater(cache_tiny.stats()["evictions"], 0)


if __name__ == "__main__":
    unittest.main()
