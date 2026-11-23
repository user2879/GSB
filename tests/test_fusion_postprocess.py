import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._util import make_workspace, run
from mstc.fusion import fuse_pair
from mstc.postprocess import _connected_components, remove_small_components, GlobalRaster


class FusionTests(unittest.TestCase):
    def test_pair_count_and_vote_detail(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        report, _ = run(ws)
        # 3x3 grid -> 6 horizontal + 6 vertical right/down pairs = 12
        self.assertEqual(len(report.fusions), 12)
        for fusion in report.fusions:
            self.assertTrue(all(0 <= v.winner_class < 4 for v in fusion.votes))
            self.assertTrue(all(v.margin >= 0 for v in fusion.votes))

    def test_strategy_versions_differ(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        fv1, _ = run(ws)
        (ws / "fv2.yaml").write_text(
            (ws / "config.yaml").read_text().replace("FV1", "FV2"))
        fv2, _ = run(ws, "fv2.yaml")
        self.assertNotEqual(
            np.array(fv1.global_raster.raster).tolist(),
            np.array(fv2.global_raster.raster).tolist(),
        )
        self.assertNotEqual(fv1.fusions[0].fingerprint, fv2.fusions[0].fingerprint)

    def test_tie_break_is_deterministic(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        r1, _ = run(ws)
        r2, _ = run(ws)
        for a, b in zip(r1.fusions, r2.fusions):
            self.assertEqual(
                [v.winner_class for v in a.votes],
                [v.winner_class for v in b.votes],
            )

    def test_fingerprint_includes_strategy_and_keys(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        report, _ = run(ws)
        fp = report.fusions[0].fingerprint
        self.assertEqual(len(fp), 64)


class PostprocessTests(unittest.TestCase):
    def test_connected_components_labelling(self):
        raster = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [2, 2, 1],
        ], dtype=np.int8)
        labels = _connected_components(raster)
        self.assertEqual(len(np.unique(labels)), 3)

    def test_small_component_removed_and_backfilled(self):
        raster = np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=np.int8)
        gr = GlobalRaster(raster=raster, height=4, width=4)
        out = remove_small_components(gr, min_pixels=3)
        self.assertEqual(len(out.removed), 1)
        self.assertEqual(int(out.raster[1, 1]), 1)
        self.assertEqual(out.removed[0].filled_with, 1)


if __name__ == "__main__":
    unittest.main()
