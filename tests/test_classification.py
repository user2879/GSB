import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._util import make_workspace
from mstc.classifier import classify_tile, window_origins
from mstc.config import load_config
from mstc.fixedpoint import FixedConfig, Gear
from mstc.loader import load_manifest, validate_and_load
from mstc.model import CLASSES
from mstc.paths import to_path


class WindowOwnershipTests(unittest.TestCase):
    def test_origins_cover_every_pixel_once(self):
        origins = window_origins(16, 8, 4)
        covered = np.zeros((16, 16), dtype=bool)
        for _, _, cr0, cc0 in origins:
            covered[cr0:cr0 + 4, cc0:cc0 + 4] = True
        self.assertTrue(covered.all())

    def test_centre_regions_disjoint(self):
        origins = window_origins(16, 8, 4)
        covered = np.zeros((16, 16), dtype=int)
        for _, _, cr0, cc0 in origins:
            covered[cr0:cr0 + 4, cc0:cc0 + 4] += 1
        self.assertTrue((covered == 1).all())


class ClassificationTests(unittest.TestCase):
    def _tile(self, tmp):
        ws = make_workspace(tmp)
        cfg = load_config(ws / "config.yaml")
        manifest = load_manifest(to_path(cfg.manifest_path, ws), ws)
        tiles, _ = validate_and_load(manifest, cfg, ws)
        return cfg, tiles["t00"]

    def test_deterministic_pixel_identical(self):
        cfg, tile = self._tile(Path("/tmp/cl_det"))
        r1 = classify_tile(tile, cfg)
        r2 = classify_tile(tile, cfg)
        np.testing.assert_array_equal(r1.class_raster, r2.class_raster)
        np.testing.assert_array_equal(r1.score_raster, r2.score_raster)
        self.assertEqual(r1.cache_key, r2.cache_key)

    def test_all_pixels_classified_and_valid_codes(self):
        cfg, tile = self._tile(Path("/tmp/cl_codes"))
        result = classify_tile(tile, cfg)
        self.assertEqual(result.pixels_classified, 256)
        self.assertTrue(set(np.unique(result.class_raster)).issubset(range(len(CLASSES))))

    def test_residuals_and_rounding_recorded(self):
        cfg, tile = self._tile(Path("/tmp/cl_res"))
        result = classify_tile(tile, cfg)
        for key in ("ndvi", "ndwi", "texture", "slope"):
            self.assertEqual(result.residual_raster[key].shape, (16, 16))
            self.assertTrue(set(np.unique(result.rounding_raster[key])).issubset({-1, 0, 1}))

    def test_quantization_gear_changes_scores(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        cfg = load_config(ws / "config.yaml")
        manifest = load_manifest(to_path(cfg.manifest_path, ws), ws)
        tiles, _ = validate_and_load(manifest, cfg, ws)
        tile = tiles["t00"]
        fine = classify_tile(tile, cfg)
        coarse_cfg = load_config(
            _write_gear(ws, "COARSE"))
        coarse = classify_tile(tile, coarse_cfg)
        self.assertFalse(np.array_equal(fine.score_raster, coarse.score_raster))
        self.assertNotEqual(fine.cache_key, coarse.cache_key)

    def test_probabilities_sum_to_unit(self):
        cfg, tile = self._tile(Path("/tmp/cl_prob"))
        result = classify_tile(tile, cfg)
        fcfg = FixedConfig(cfg.fixed.fractional_bits, cfg.fixed.gear)
        unit = fcfg.scale >> fcfg.gear_shift
        sums = result.prob_raster.sum(axis=2)
        np.testing.assert_array_equal(sums, np.full((16, 16), unit))


def _write_gear(ws: Path, gear: str) -> Path:
    text = (ws / "config.yaml").read_text().replace("NORMAL", gear)
    out = ws / f"config_{gear}.yaml"
    out.write_text(text)
    return out


if __name__ == "__main__":
    unittest.main()
