import unittest

import numpy as np

from tactical_pathbench.encoding import canonical_encoding
from tactical_pathbench.generate import generate_sample
from tactical_pathbench.grid import RegionPartition, TacticalGrid
from tactical_pathbench.errors import MapShapeError, TerrainRangeError


class DeterminismTests(unittest.TestCase):
    def test_manifest_byte_identical(self):
        first = canonical_encoding(generate_sample(seed=123).manifest_payload())
        second = canonical_encoding(generate_sample(seed=123).manifest_payload())
        self.assertEqual(first, second)

    def test_grid_hash_stable_and_partition_signature(self):
        sample = generate_sample(seed=9)
        self.assertEqual(sample.grid.content_hash(), sample.grid.content_hash())
        report = sample.grid.region_checksum_report()
        self.assertEqual(set(report), {str(rid) for rid in sample.grid.region_ids()})
        self.assertEqual(
            sample.grid.region_partition_signature(),
            sample.grid.region_partition_signature(),
        )

    def test_map_shape_and_terrain_range_errors(self):
        with self.assertRaises(MapShapeError):
            TacticalGrid(np.zeros((3,), dtype=np.uint8), np.ones((3,), bool),
                         np.ones((3,), bool), RegionPartition(1, 1))
        cost = np.ones((4, 4), dtype=np.uint8)
        cost[0, 0] = 99
        with self.assertRaises(TerrainRangeError):
            TacticalGrid(cost, np.ones((4, 4), bool), np.ones((4, 4), bool),
                         RegionPartition(2, 2))

    def test_illegal_runtime_terrain_change(self):
        sample = generate_sample(seed=1)
        with self.assertRaises(TerrainRangeError):
            sample.grid.set_terrain_cost((0, 0), 0)
        with self.assertRaises(MapShapeError):
            sample.grid.require_cell(999, 999)


if __name__ == "__main__":
    unittest.main()
