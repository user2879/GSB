from __future__ import annotations

import unittest

from mixedprec_diag.memory import BlockPool, PoolOOM, replay_events


class MemoryTests(unittest.TestCase):
    def test_contiguous_and_single_block_allocation_no_overlap(self) -> None:
        pool = BlockPool(10, 1024)
        first = pool.allocate(
            owner_id="A",
            batch_id="B1",
            purpose="arena",
            request_bytes=3 * 1024,
            strategy="per_batch",
            contiguous=True,
        )
        second = pool.allocate_single_block(
            owner_id="B",
            batch_id="B2",
            purpose="single",
            request_bytes=1024,
        )
        self.assertEqual(first.blocks, (0, 1, 2))
        self.assertFalse(set(first.blocks) & set(second.blocks))
        self.assertEqual(pool.used_blocks, 4)
        self.assertTrue(pool.active_allocations())

    def test_oom_records_required_and_free_blocks(self) -> None:
        pool = BlockPool(5, 1000)
        pool.allocate(
            owner_id="A",
            batch_id="B1",
            purpose="arena",
            request_bytes=3000,
            strategy="per_batch",
            contiguous=True,
        )
        with self.assertRaises(PoolOOM) as raised:
            pool.allocate(
                owner_id="B",
                batch_id="B2",
                purpose="arena",
                request_bytes=3000,
                strategy="per_batch",
                contiguous=True,
            )
        self.assertEqual(raised.exception.required_blocks, 3)
        self.assertEqual(raised.exception.free_blocks, 2)
        self.assertEqual(pool.events[-1]["action"], "oom")
        self.assertEqual(pool.events[-1]["requested_blocks"], 3)
        self.assertEqual(pool.used_blocks, 3)

    def test_fragmentation_is_derived_from_free_extents(self) -> None:
        pool = BlockPool(10, 1024)
        for owner in ("A", "B", "C"):
            pool.allocate(
                owner_id=owner,
                batch_id=owner,
                purpose="arena",
                request_bytes=2 * 1024,
                strategy="per_batch",
                contiguous=True,
            )
        pool.release("B")
        snapshot = pool.snapshot()
        self.assertEqual(snapshot.free_blocks, 6)
        self.assertEqual(snapshot.free_extents, [(2, 4), (6, 10)])
        self.assertEqual(snapshot.largest_free_extent, 4)
        self.assertEqual(snapshot.fragmentation_blocks, 2)

    def test_noncontiguous_release_does_not_free_other_blocks(self) -> None:
        pool = BlockPool(8, 1024)
        pool.allocate(owner_id="A", batch_id="B1", purpose="arena", request_bytes=2048, strategy="per_batch", contiguous=True)
        pool.allocate(owner_id="B", batch_id="B2", purpose="arena", request_bytes=2048, strategy="per_batch", contiguous=True)
        pool.release("A")
        mixed = pool.allocate(
            owner_id="C",
            batch_id="B3",
            purpose="mixed",
            request_bytes=5 * 1024,
            strategy="block_pool",
            contiguous=False,
        )
        self.assertEqual(mixed.blocks, (0, 4, 5, 6, 7))
        pool.release("B")
        pool.release("C")
        self.assertEqual(pool.free_blocks, 8)
        self.assertEqual(pool.snapshot().free_extents, [(0, 8)])

    def test_event_log_replays_to_identical_state(self) -> None:
        pool = BlockPool(12, 512)
        pool.allocate(owner_id="weights", batch_id=None, purpose="model", request_bytes=1024, strategy="persistent", contiguous=True)
        pool.allocate(owner_id="A", batch_id="B1", purpose="input", request_bytes=1536, strategy="block_pool", contiguous=False)
        pool.allocate(owner_id="A2", batch_id="B1", purpose="output", request_bytes=2048, strategy="block_pool", contiguous=False)
        pool.release("A")
        try:
            pool.allocate(owner_id="huge", batch_id="B2", purpose="huge", request_bytes=12 * 512, strategy="block_pool", contiguous=False)
        except PoolOOM:
            pass
        pool.release_batch("B1")
        pool.release("weights")
        replay = replay_events(pool.events, 12)
        self.assertTrue(replay["verified"], replay["errors"])
        self.assertEqual(replay["final_used_blocks"], [])
        self.assertEqual(replay["final_free_blocks"], 12)
        self.assertEqual(replay["peak_used_blocks"], pool.peak_used_blocks)


if __name__ == "__main__":
    unittest.main()
