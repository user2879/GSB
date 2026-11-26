import unittest

from shardmesh.errors import ShardMeshError
from shardmesh.verification import assert_projections_match


class DivergenceTests(unittest.TestCase):
    def test_divergence_locates_shard_node_and_field(self):
        incremental = {
            "s1": {
                "shard_id": "s1",
                "node_id": "n7",
                "outcome": {"summary": {"checksum": "abc", "total": 42}},
            }
        }
        full = {
            "s1": {
                "shard_id": "s1",
                "node_id": "n7",
                "outcome": {"summary": {"checksum": "abc", "total": 43}},
            }
        }
        with self.assertRaises(ShardMeshError) as ctx:
            assert_projections_match(incremental, full)
        self.assertEqual(ctx.exception.code, "E_DIVERGE")
        self.assertEqual(ctx.exception.details["shard_id"], "s1")
        self.assertEqual(ctx.exception.details["node_id"], "n7")
        self.assertIn("total", ctx.exception.field_name)


if __name__ == "__main__":
    unittest.main()
