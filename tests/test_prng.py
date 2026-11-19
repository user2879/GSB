"""Determinism tests for the self-contained PRNG."""

from __future__ import annotations

import unittest

from level_solver.prng import DeterministicRNG, derive_seed


class PrngTests(unittest.TestCase):
    def test_streams_are_reproducible(self):
        first = DeterministicRNG(12345)
        second = DeterministicRNG(12345)
        self.assertEqual([first.next64() for _ in range(16)], [second.next64() for _ in range(16)])

    def test_substreams_independent_and_stable(self):
        a = DeterministicRNG(7)
        b = DeterministicRNG(7)
        child_a = a.substream(3)
        child_b = b.substream(3)
        self.assertEqual(
            [child_a.randrange(1000) for _ in range(12)],
            [child_b.randrange(1000) for _ in range(12)],
        )
        # Deriving a child must not advance the parent stream.
        self.assertEqual(a.next64(), b.next64())

    def test_shuffle_is_deterministic(self):
        a = DeterministicRNG(99)
        b = DeterministicRNG(99)
        self.assertEqual(a.shuffled(range(20)), b.shuffled(range(20)))

    def test_derive_seed_material_rules(self):
        self.assertEqual(derive_seed(42), derive_seed(42))
        self.assertEqual(derive_seed(["x", 1]), derive_seed(["x", 1]))
        self.assertNotEqual(derive_seed(1), derive_seed(2))

    def test_randrange_bounds(self):
        rng = DeterministicRNG(5)
        for _ in range(100):
            value = rng.randrange(4)
            self.assertIn(value, range(4))


if __name__ == "__main__":
    unittest.main()
