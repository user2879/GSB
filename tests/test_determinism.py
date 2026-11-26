import unittest

from shard_diag.determinism import (
    aggregate_batch,
    build_summary,
    combine_partials,
    generate_samples,
    samples_raw_bytes,
    split_batches,
)
from shard_diag.spec import NodeSpec, ShardSpec


class DeterminismTests(unittest.TestCase):
    def setUp(self):
        self.node = NodeSpec("n", "127.0.0.1", 40001, 2, 2, 12345, 1)
        self.shard = ShardSpec("shard-A", (), 137, 2, 5)

    def test_sample_bytes_stable(self):
        first = generate_samples(self.node.seed, self.shard.shard_id, self.shard.sample_rows)
        second = generate_samples(self.node.seed, self.shard.shard_id, self.shard.sample_rows)
        self.assertEqual(samples_raw_bytes(first), samples_raw_bytes(second))
        self.assertEqual(len(samples_raw_bytes(first)), 137 * 16)

    def test_summary_independent_of_batch_split(self):
        samples = generate_samples(self.node.seed, self.shard.shard_id, self.shard.sample_rows)
        digests = set()
        byte_keys = set()
        for batch_size in (1, 3, 7, 32, 137):
            partial = combine_partials(
                aggregate_batch(batch, 1_000_000)
                for batch in split_batches(samples, batch_size)
            )
            summary = build_summary(
                node_id=self.node.node_id,
                shard_id=self.shard.shard_id,
                sample_rows=self.shard.sample_rows,
                quota_version=3,
                partial=partial,
                fixed_point_scale=1_000_000,
            )
            digests.add(summary["aggregate_digest"])
            byte_keys.add(str(summary))
        self.assertEqual(len(digests), 1)
        self.assertEqual(len(byte_keys), 1)

    def test_different_shard_changes_digest(self):
        samples_a = generate_samples(1, "a", 32)
        samples_b = generate_samples(1, "b", 32)
        self.assertNotEqual(samples_raw_bytes(samples_a), samples_raw_bytes(samples_b))


if __name__ == "__main__":
    unittest.main()
