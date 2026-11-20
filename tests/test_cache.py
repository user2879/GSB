from __future__ import annotations

import unittest

from edgecache.cache import EdgeCache
from edgecache.errors import EdgeError
from edgecache.spec import load_spec

from tests.helpers import SAMPLE_V1


class FakeClock:
    def __init__(self, value: int = 1_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class EdgeCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec(SAMPLE_V1)

    def _cache(self, *, capacity: int = 2, rule: str = "stable_fifo", clock: FakeClock | None = None, node: str = "edge-a"):
        return EdgeCache(
            node_id=node,
            capacity=capacity,
            eviction_rule=rule,
            policy=self.spec.policy,
            ledger_version=1,
            clock=clock or FakeClock(),
        )

    def test_hit_decrements_ttl_and_expired_entry_is_not_returned(self) -> None:
        clock = FakeClock(1_000)
        cache = self._cache(clock=clock)
        cache.put(
            ledger_version=1,
            path="/data/root",
            etag="root-v1",
            content_version=self.spec.content_version,
            ttl_ms=100,
            content_size=9,
        )
        clock.value = 1_050
        hit = cache.get(ledger_version=1, path="/data/root", etag="root-v1")
        self.assertIsNotNone(hit)
        self.assertEqual(50, hit.remaining_ttl_ms)
        clock.value = 1_100
        self.assertIsNone(cache.get(ledger_version=1, path="/data/root", etag="root-v1"))
        self.assertEqual([], cache.entry_state(ledger_version=1))
        self.assertEqual("expired", cache.events(ledger_version=1)[-1]["reason"])

    def test_entry_version_increments_on_rewrite(self) -> None:
        clock = FakeClock(1_000)
        cache = self._cache(clock=clock)
        first = cache.put(
            ledger_version=1,
            path="/data/root",
            etag="root-v1",
            content_version=self.spec.content_version,
            ttl_ms=100,
            content_size=9,
        )
        clock.value = 1_001
        second = cache.put(
            ledger_version=1,
            path="/data/root",
            etag="root-v1",
            content_version=self.spec.content_version,
            ttl_ms=100,
            content_size=9,
        )
        self.assertEqual(1, first.entry_version)
        self.assertEqual(2, second.entry_version)
        self.assertEqual(1, len(cache.entry_state(ledger_version=1)))

    def test_access_order_does_not_change_eviction_set(self) -> None:
        for rule in ("stable_fifo", "smallest_key", "nearest_expiry"):
            survivors: list[list[str]] = []
            for access_order in (("a", "b"), ("b", "a")):
                clock = FakeClock(1_000)
                cache = self._cache(capacity=2, rule=rule, clock=clock)
                records = {
                    "a": ("/api/config", "config-v1", 100),
                    "b": ("/data/root", "root-v1", 200),
                    "c": ("/page/home", "page-v1", 300),
                }
                for index, name in enumerate(("a", "b"), start=1):
                    path, etag, ttl = records[name]
                    clock.value = 1_000 + index
                    cache.put(
                        ledger_version=1,
                        path=path,
                        etag=etag,
                        content_version=self.spec.content_version,
                        ttl_ms=ttl,
                        content_size=1,
                    )
                clock.value = 1_100
                for name in access_order:
                    path, etag, _ = records[name]
                    cache.get(ledger_version=1, path=path, etag=etag)
                path, etag, ttl = records["c"]
                clock.value = 1_101
                cache.put(
                    ledger_version=1,
                    path=path,
                    etag=etag,
                    content_version=self.spec.content_version,
                    ttl_ms=ttl,
                    content_size=1,
                )
                survivors.append(sorted(entry["path"] for entry in cache.entry_state(ledger_version=1)))
            self.assertEqual(survivors[0], survivors[1], rule)
            self.assertEqual(["/data/root", "/page/home"], survivors[0], rule)
            self.assertEqual(1, len(cache.events(ledger_version=1)))

    def test_stale_ledger_is_rejected(self) -> None:
        cache = self._cache()
        with self.assertRaises(EdgeError) as context:
            cache.get(ledger_version=0, path="/data/root", etag="root-v1")
        self.assertEqual("E_STALE", context.exception.code)

    def test_capacity_eviction_event_has_rule_and_victim(self) -> None:
        clock = FakeClock(1_000)
        cache = self._cache(capacity=1, rule="stable_fifo", clock=clock)
        for index, (path, etag, ttl) in enumerate(
            [("/api/config", "config-v1", 100), ("/data/root", "root-v1", 100)], start=1
        ):
            clock.value += 1
            cache.put(
                ledger_version=1,
                path=path,
                etag=etag,
                content_version=self.spec.content_version,
                ttl_ms=ttl,
                content_size=1,
            )
        events = cache.events(ledger_version=1)
        self.assertEqual(1, len(events))
        self.assertEqual("capacity", events[0]["reason"])
        self.assertEqual("stable_fifo", events[0]["rule"])
        self.assertEqual("/api/config", events[0]["path"])


if __name__ == "__main__":
    unittest.main()
