from __future__ import annotations

import unittest

from speaker_diag.invalidation import compute_invalidation, recompute_invalid_subset
from speaker_diag.pipeline import normalize_local_results
from speaker_diag.spec import load_workload
from speaker_diag.synthetic import synthesize


class InvalidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v1 = load_workload("samples/workload.json")
        cls.v2 = load_workload("samples/workload_v2.json")
        cls.v3 = load_workload("samples/workload_v3.json")
        cls.b1 = synthesize(cls.v1)
        cls.b2 = synthesize(cls.v2)
        cls.b3 = synthesize(cls.v3)
        cls._raw, cls.n1 = normalize_local_results(cls.v1, cls.b1)
        cls._raw2, cls.n2 = normalize_local_results(cls.v2, cls.b2)

    def test_worker_version_direct_and_indirect_boundary(self) -> None:
        invalidation = compute_invalidation(self.v1, self.v2)
        self.assertEqual(
            [item["batch_id"] for item in invalidation["direct_batches"]],
            ["B001", "B003", "B005", "B007", "B009"],
        )
        self.assertEqual(
            [item["batch_id"] for item in invalidation["indirect_batches"]],
            ["B004", "B010"],
        )
        self.assertEqual(len(invalidation["invalid_windows"]), 24)
        self.assertEqual(invalidation["unchanged_batches"], ["B002", "B006", "B008"])
        self.assertEqual(invalidation["stable_order"], "worker_id,batch_order")

    def test_selective_recompute_matches_full_and_keeps_unchanged_bytes(self) -> None:
        invalidation = compute_invalidation(self.v1, self.v2)
        recomputed = recompute_invalid_subset(self.v1, self.v2, self.b2, self.n1, invalidation)
        self.assertTrue(recomputed["unchanged_check"]["all_byte_identical"])
        for batch_id in self.n1:
            self.assertEqual(
                recomputed["receipts"][batch_id].attribution_digest,
                self.n2[batch_id].attribution_digest,
            )

    def test_quantization_and_batch_size_change_invalidates_all(self) -> None:
        invalidation = compute_invalidation(self.v2, self.v3)
        self.assertEqual(len(invalidation["invalid_batches"]), len(self.v3.requests))
        self.assertEqual(len(invalidation["invalid_windows"]), len(self.v3.window_ids))
        self.assertEqual(invalidation["unchanged_batches"], [])


if __name__ == "__main__":
    unittest.main()

