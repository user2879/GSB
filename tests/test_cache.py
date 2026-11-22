import unittest

from dns_workbench.app import DeterministicClock
from dns_workbench.cache import TtlCache
from dns_workbench.records import CacheEntry, canonical_digest
from dns_workbench.wire import TYPE_A


def make_entry(name: str, ttl: int = 60, version: str = "v1") -> CacheEntry:
    entry = CacheEntry(name=name, qtype=TYPE_A, value=f"A 10.0.0.{len(name)}", ttl=ttl,
                       source_version=version, content_digest="")
    entry.content_digest = canonical_digest(entry.normalized_payload())
    return entry


class TtlCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = DeterministicClock()
        self.cache = TtlCache(capacity=4, negative_ttl_cap=300, clock=self.clock)

    def test_ttl_decreases_with_monotonic_ms(self) -> None:
        self.cache.put(make_entry("a.test", ttl=10))
        hit = self.cache.get(("a.test", TYPE_A))
        self.assertEqual(hit.remaining_ttl_s, 10)
        self.clock.advance_ms(3_500)
        hit = self.cache.get(("a.test", TYPE_A))
        self.assertEqual(hit.remaining_ttl_s, 7)  # 向上取整
        self.assertEqual(hit.hits, 2)

    def test_expiry_reaps_and_no_hit(self) -> None:
        self.cache.put(make_entry("a.test", ttl=1))
        self.clock.advance_ms(1_000)
        expired = self.cache.reap_expired()
        self.assertEqual(expired, [("a.test", TYPE_A)])
        self.assertIsNone(self.cache.get(("a.test", TYPE_A)))

    def test_negative_ttl_cap(self) -> None:
        self.assertEqual(self.cache.negative_ttl(60), 60)
        self.assertEqual(self.cache.negative_ttl(9999), 300)

    def test_lru_eviction_is_stable(self) -> None:
        for index in range(6):
            self.cache.put(make_entry(f"lru{index}.test"))
            self.clock.advance_ms(1)
        self.assertEqual(len(self.cache), 4)
        self.assertEqual([ev.reason for ev in self.cache.evictions], ["lru", "lru"])
        self.assertEqual([ev.key[0] for ev in self.cache.evictions], ["lru0.test", "lru1.test"])

    def test_capacity_evicts_expired_first(self) -> None:
        self.cache.put(make_entry("short.test", ttl=1))
        self.clock.advance_ms(1_000)
        for index in range(4):
            self.cache.put(make_entry(f"keep{index}.test"))
        self.assertNotIn(("short.test", TYPE_A), self.cache)
        self.assertEqual(len(self.cache), 4)


if __name__ == "__main__":
    unittest.main()
