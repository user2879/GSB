import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._util import make_workspace, run
from mstc.errors import Code
from mstc.paths import to_path


class CacheInvalidationTests(unittest.TestCase):
    def test_warm_run_fully_cached(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        first, _ = run(ws)
        second, _ = run(ws)
        self.assertEqual(len(first.recomputed_tiles), 9)
        self.assertEqual(len(second.cached_tiles), 9)
        self.assertEqual(second.recomputed_tiles, [])
        self.assertEqual(len(second.cached_fusions), 12)

    def test_band_patch_only_invalidates_touched_tile_and_its_fusions(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        run(ws)
        band = to_path("samples\\tiles\\t11\\R.npy", ws)
        arr = np.load(band)
        arr[2, 3] = 250
        np.save(band, arr)
        report, _ = run(ws)
        self.assertEqual(report.invalidated_tiles, ["t11"])
        self.assertEqual(report.recomputed_tiles, ["t11"])
        affected = {
            "t01|t11", "t10|t11", "t11|t12", "t11|t21",
        }
        self.assertEqual(set(report.recomputed_fusions), affected)
        self.assertEqual(len(report.cached_tiles), 8)

    def test_weight_change_invalidates_all_tiles(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        run(ws)
        cfg = ws / "config.yaml"
        cfg.write_text(cfg.read_text().replace("vegetation: 1.0", "vegetation: 3.0"))
        report, _ = run(ws)
        self.assertEqual(len(report.recomputed_tiles), 9)

    def test_incremental_equals_clean_rebuild(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        run(ws)
        band = to_path("samples\\tiles\\t02\\G.npy", ws)
        arr = np.load(band)
        arr[0, 0] = 12
        np.save(band, arr)
        incremental, _ = run(ws)

        clean = make_workspace(Path(tempfile.mkdtemp()))
        np.save(to_path("samples\\tiles\\t02\\G.npy", clean), arr)
        clean_run, _ = run(clean)
        np.testing.assert_array_equal(
            incremental.global_raster.raster, clean_run.global_raster.raster)
        for tid in incremental.tile_results:
            self.assertEqual(
                incremental.tile_results[tid].score_summary,
                clean_run.tile_results[tid].score_summary,
            )


class SnapshotTests(unittest.TestCase):
    def test_stale_version_triggers_rebuild_code(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        run(ws, version=1)
        report, _ = run(ws, version=2)
        self.assertIn(Code.E_SNAPSHOT_STALE, report.failures.codes())

    def test_same_version_resumes_without_stale(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        run(ws, version=1)
        report, _ = run(ws, version=1)
        self.assertNotIn(Code.E_SNAPSHOT_STALE, report.failures.codes())

    def test_manifest_records_version_and_quota(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        report, _ = run(ws)
        doc = report.manifest_doc
        self.assertEqual(doc["global_version"], 1)
        self.assertEqual(len(doc["config_hash"]), 64)
        self.assertGreater(doc["consumed_pixel_quota"], 0)
        self.assertIn("postprocess", doc["stages"])


if __name__ == "__main__":
    unittest.main()
