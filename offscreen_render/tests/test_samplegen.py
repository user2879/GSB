import os
import tempfile
import unittest

import numpy as np

from offscreen_render.samplegen import (
    SEED,
    build_medium_grid,
    expected_probe_transmittance,
    generate_sample,
)
from offscreen_render.scene import load_scene


class TestSampleDeterminism(unittest.TestCase):
    def test_repeatable_generation(self):
        import json
        roots = [tempfile.mkdtemp() for _ in range(2)]
        a = generate_sample(os.path.join(roots[0], "s"),
                            os.path.join(roots[0], "state"),
                            os.path.join(roots[0], "out"))
        b = generate_sample(os.path.join(roots[1], "s"),
                            os.path.join(roots[1], "state"),
                            os.path.join(roots[1], "out"))
        da = json.load(open(a["scene"], encoding="utf-8"))
        db = json.load(open(b["scene"], encoding="utf-8"))
        # 路径随临时目录变化，剔除后其余场景实质参数必须逐字节一致。
        da["render"].pop("state_dir"); db["render"].pop("state_dir")
        da["render"].pop("output_dir"); db["render"].pop("output_dir")
        self.assertEqual(da, db)
        ga = np.load(a["grid"])
        gb = np.load(b["grid"])
        for key in ("sigma_t", "albedo", "ior", "x", "y", "z"):
            np.testing.assert_array_equal(ga[key], gb[key])
        self.assertEqual(
            open(a["expected"], encoding="utf-8").read(),
            open(b["expected"], encoding="utf-8").read())

    def test_expected_transmittance_matches_chain(self):
        scene = load_scene(_scene_path())
        expected = expected_probe_transmittance(scene)
        for probe in expected["probes"]:
            tau = sum(seg["optical_depth"] for seg in probe["segments"])
            self.assertAlmostEqual(np.exp(-tau), probe["expected_transmittance"])
            self.assertGreaterEqual(probe["expected_transmittance"], 0.0)
            self.assertLessEqual(probe["expected_transmittance"], 1.0)

    def test_grid_priority_matches_declaration(self):
        tmp = tempfile.mkdtemp()
        paths = generate_sample(os.path.join(tmp, "s"),
                                os.path.join(tmp, "state"), os.path.join(tmp, "out"))
        scene = load_scene(paths["scene"])
        grid = build_medium_grid(scene)
        # 栅格有限且消光非负
        self.assertTrue(np.isfinite(grid["sigma_t"]).all())
        self.assertTrue((grid["sigma_t"] >= 0).all())
        self.assertTrue(((grid["albedo"] >= 0) & (grid["albedo"] <= 1)).all())
        self.assertTrue((grid["ior"] >= 1.0).all())

    def test_seed_fixed(self):
        self.assertEqual(SEED, 20261119)


def _scene_path():
    tmp = tempfile.mkdtemp()
    paths = generate_sample(os.path.join(tmp, "s"),
                            os.path.join(tmp, "state"), os.path.join(tmp, "out"))
    return paths["scene"]


if __name__ == "__main__":
    unittest.main()
