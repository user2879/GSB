import unittest

from batch_receipt_ledger.inference import (
    batch_summary, build_input_vector, compute_request_digests)


class InferenceTest(unittest.TestCase):
    def test_deterministic_outputs(self):
        batch = [("req-1", 3), ("req-2", 7)]
        first = compute_request_digests(batch, 42, 1, 16, 8)
        second = compute_request_digests(batch, 42, 1, 16, 8)
        self.assertEqual(first, second)

    def test_param_version_changes_digest(self):
        batch = [("req-1", 3)]
        v1 = compute_request_digests(batch, 42, 1, 16, 8)
        v2 = compute_request_digests(batch, 42, 2, 16, 8)
        self.assertNotEqual(v1["req-1"]["digest"], v2["req-1"]["digest"])

    def test_seed_changes_input_vector(self):
        a = build_input_vector(1, "req-1", 3, 16)
        b = build_input_vector(2, "req-1", 3, 16)
        self.assertFalse((a == b).all())

    def test_batch_summary_stable_order(self):
        digests = compute_request_digests(
            [("a", 1), ("b", 2)], 42, 1, 16, 8)
        reordered = {key: digests[key] for key in reversed(list(digests))}
        self.assertEqual(batch_summary(digests), batch_summary(reordered))


if __name__ == "__main__":
    unittest.main()
