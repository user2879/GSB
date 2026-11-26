import os
import random
import unittest

from leasebatch.plan import build_plan
from leasebatch.spec import load_spec_dir


SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


class DeterministicPlanTest(unittest.TestCase):
    def setUp(self):
        self.spec = load_spec_dir(SAMPLE_DIR)

    def test_plan_is_byte_stable_across_rebuilds(self):
        first = build_plan(self.spec, 2, 42)
        second = build_plan(self.spec, 2, 42)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(first.signature, second.signature)

    def test_plan_ignores_dict_and_dispatch_order(self):
        baseline = build_plan(self.spec, 2, 42)
        for seed in (1, 7, 99):
            random.Random(seed).shuffle(self.spec.requests)
            shuffled = build_plan(self.spec, 2, 42)
            self.assertEqual(shuffled.canonical_bytes(),
                             baseline.canonical_bytes())

    def test_batch_ids_derive_from_worker_index_arrivals(self):
        plan = build_plan(self.spec, 2, 42)
        ids = [batch.batch_id for batch in plan.batches]
        self.assertEqual(len(ids), len(set(ids)))
        worker_one = [b for b in plan.batches if b.worker_id == "w1"]
        self.assertEqual([b.batch_index for b in worker_one], [0, 1])
        # Arrivals inside a batch are pinned into the id derivation.
        self.assertEqual(worker_one[0].arrivals, (1, 2))

    def test_batch_size_change_changes_partition(self):
        small = build_plan(self.spec, 2, 42)
        large = build_plan(self.spec, 3, 42)
        self.assertNotEqual(small.signature, large.signature)
        self.assertEqual(len(large.batches), 3)


if __name__ == "__main__":
    unittest.main()
