from __future__ import annotations

import unittest

import numpy as np

from volumetric_editor.scene import generate_scene
from volumetric_editor.session import IncrementalSession


class IncrementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = generate_scene(seed=7070, width=32, height=24)

    def setUp(self) -> None:
        self.session = IncrementalSession(self.scene, tile_size=8)
        self.session.render_full(threads=3)

    def test_ior_edit_uses_exact_tile_closure_and_reuses_trace(self) -> None:
        before = self.session.snapshot()
        old_traces = {key: tile.volume_trace.copy() for key, tile in self.session.tiles.items()}
        report = self.session.edit_ior("lens_left", 1.68, threads=2)
        self.assertEqual(report["actual_recomputed_tiles"], report["closure"]["tiles"])
        self.assertTrue(report["reuse"]["tile_content_hashes_unchanged"])
        self.assertTrue(report["outside_volume_trace_preserved"])
        self.assertEqual(report["actual_recomputed_volume_segment_count"], 0)
        for key in sorted(set(self.session.tiles) - set(report["closure"]["tiles"])):
            self.assertEqual(before.tile_hashes[key], self.session.snapshot().tile_hashes[key])
            self.assertTrue(np.array_equal(old_traces[key], self.session.tiles[key].volume_trace))
        self.assertNotEqual(before.image_sha256, self.session.snapshot().image_sha256)

    def test_scatter_edit_uses_exact_segment_and_tile_closure(self) -> None:
        before = self.session.snapshot()
        old_traces = {key: tile.volume_trace.copy() for key, tile in self.session.tiles.items()}
        report = self.session.edit_scatter(0.72, threads=2)
        self.assertEqual(report["actual_recomputed_tiles"], report["closure"]["tiles"])
        self.assertEqual(report["actual_volume_segment_digest"], report["closure"]["digest"])
        self.assertTrue(report["actual_matches_closure"])
        self.assertTrue(report["outside_volume_trace_preserved"])
        self.assertTrue(report["reuse"]["tile_content_hashes_unchanged"])
        self.assertGreater(report["closure"]["volume_segment_count"], 0)
        self.assertEqual(
            report["closure"]["volume_segment_count"]
            + report["reuse"]["volume_segment_count"],
            before.segments_total,
        )
        for key in sorted(set(self.session.tiles) - set(report["closure"]["tiles"])):
            self.assertTrue(np.array_equal(old_traces[key], self.session.tiles[key].volume_trace))
            self.assertEqual(before.tile_hashes[key], self.session.snapshot().tile_hashes[key])

    def test_retile_reuses_canonical_content_without_recompute(self) -> None:
        before = self.session.snapshot()
        report = self.session.edit_tile_size(12)
        after = self.session.snapshot()
        self.assertEqual(report["actual_recomputed_tiles"], [])
        self.assertEqual(report["recompute_closure"]["tiles"], [])
        self.assertEqual(before.image_sha256, after.image_sha256)
        self.assertEqual(before.trace_sha256, after.trace_sha256)
        self.assertEqual(report["reused"]["pixels"], 32 * 24)
        self.assertTrue(report["reused"]["canonical_image_sha256_unchanged"])
        self.assertTrue(report["reused"]["canonical_volume_trace_sha256_unchanged"])
        self.assertTrue(all(tile.status == "REUSED_RETILE" for tile in self.session.tiles.values()))


if __name__ == "__main__":
    unittest.main()
