import os
import unittest

from hotblock_diag.cache import (PersistentCache, bitmap_cache_key,
                                 block_pair_cache_key, hot_table_cache_key)
from tests._harness import temp_workspace


class CacheKeyTests(unittest.TestCase):
    def test_keys_are_deterministic_and_input_sensitive(self):
        key_a = block_pair_cache_key("q1", "k1", "q8", 32, "saturate")
        key_b = block_pair_cache_key("q1", "k1", "q8", 32, "saturate")
        key_c = block_pair_cache_key("q2", "k1", "q8", 32, "saturate")
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)
        self.assertEqual(len(key_a), 16)

        bm1 = bitmap_cache_key("q1", "pre1", "q8", 0)
        bm2 = bitmap_cache_key("q1", "pre1", "q8", 1)
        self.assertNotEqual(bm1, bm2)

        ht1 = hot_table_cache_key(["a", "b"], 4, 3, "q8", 0)
        ht2 = hot_table_cache_key(["a", "b"], 5, 3, "q8", 0)
        self.assertNotEqual(ht1, ht2)

    def test_persistent_cache_hit_miss_and_graph_invalidation(self):
        cache = PersistentCache(os.path.join(temp_workspace(), "cache"))
        cache.put("blockpair", "bp-1", {"v": 1}, refs=["qblock:0:abc"])
        cache.put("bitmap", "bm-1", {"v": 2}, refs=["bp-1"])
        cache.put("hottable", "ht-1", {"v": 3}, refs=["bm-1"])

        self.assertIsNotNone(cache.get("blockpair", "bp-1"))
        removed = cache.invalidate_descendants(["qblock:0:abc"], "edit")
        self.assertEqual(set(removed), {"bp-1", "bm-1", "ht-1"})
        self.assertIsNone(cache.get("blockpair", "bp-1"))
        self.assertEqual(cache.family_keys("blockpair"), [])

    def test_invalidation_reaches_hot_table_descendants(self):
        cache = PersistentCache(os.path.join(temp_workspace(), "cache"))
        cache.put("blockpair", "bp-1", {}, refs=["qblock:0:a"])
        cache.put("bitmap", "bm-1", {}, refs=["bp-1"])
        cache.put("hottable", "ht-1", {}, refs=["bm-1"])
        removed = cache.invalidate_descendants(["qblock:0:a"], "edit")
        self.assertIn("ht-1", removed)
        self.assertEqual(cache.family_keys("hottable"), [])

    def test_unrelated_artifacts_survive_invalidation(self):
        cache = PersistentCache(os.path.join(temp_workspace(), "cache"))
        cache.put("blockpair", "bp-a", {}, refs=["qblock:0:a"])
        cache.put("blockpair", "bp-b", {}, refs=["qblock:1:b"])
        cache.invalidate_descendants(["qblock:0:a"], "edit")
        self.assertIsNone(cache.get("blockpair", "bp-a"))
        self.assertIsNotNone(cache.get("blockpair", "bp-b"))


if __name__ == "__main__":
    unittest.main()
