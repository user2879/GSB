from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from renderlab.cache import LatentBlockCache
from renderlab.config import load_workbench
from renderlab.strategy import select_strategy
from tests.helpers import write_case


class StrategyTests(unittest.TestCase):
    def test_strategy_repeats_and_changes_with_scale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = load_workbench(write_case(Path(temporary) / "case"))
        render = result.config.render
        first = select_strategy(render, prompt_count=3, seed_count=2, requested_batch_size=2)
        second = select_strategy(render, prompt_count=3, seed_count=2, requested_batch_size=2)
        self.assertEqual(first.to_dict(), second.to_dict())
        large = select_strategy(render, prompt_count=5000, seed_count=2, requested_batch_size=2)
        self.assertNotEqual(first.selected["policy_id"], large.selected["policy_id"])
        self.assertTrue(large.selected["enabled_tiers"])
        self.assertLessEqual(large.selected["denoise_steps"], render.denoise_steps)
        self.assertEqual(large.version, "strategy-table-v1")


class LRUCacheTests(unittest.TestCase):
    def test_lru_eviction_lexical_tie_and_recompute_match(self) -> None:
        cache: LatentBlockCache[str] = LatentBlockCache(2)
        cache.put("b", "B", access_seq=5, digest=lambda value: value)
        cache.put("a", "A", access_seq=5, digest=lambda value: value)
        cache.put("c", "C", access_seq=6, digest=lambda value: value)
        # Both older entries share access_seq=5; lexical key "a" is evicted.
        self.assertEqual(cache.peek("a"), None)
        self.assertEqual(cache.peek("b"), "B")
        value = cache.get("a", recompute=lambda: "A", digest=lambda item: item)
        self.assertEqual(value, "A")
        self.assertEqual(cache.recomputes, 1)
        self.assertEqual(cache.recompute_matches, 1)
        self.assertEqual(cache.recompute_mismatches, 0)
        self.assertGreaterEqual(cache.evictions, 1)

    def test_recompute_mismatch_is_observable(self) -> None:
        cache: LatentBlockCache[str] = LatentBlockCache(1)
        cache.put("a", "A", digest=lambda value: value)
        cache.put("b", "B", digest=lambda value: value)
        cache.get("a", recompute=lambda: "WRONG", digest=lambda value: value)
        self.assertEqual(cache.recompute_mismatches, 1)


if __name__ == "__main__":
    unittest.main()
