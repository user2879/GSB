import json
import os
import tempfile
import unittest

from offscreen_render.errors import RenderError
from offscreen_render.scene import load_scene
from offscreen_render.samplegen import generate_sample


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.state = os.path.join(self.tmp, "state")
        self.out = os.path.join(self.tmp, "out")
        self.smp = os.path.join(self.tmp, "sample")
        generate_sample(self.smp, self.state, self.out, width=16, height=16,
                        supersample=2)
        self.scene_path = os.path.join(self.smp, "scene.json")
        with open(self.scene_path, encoding="utf-8") as fh:
            self.doc = json.load(fh)

    def _load(self, doc):
        path = os.path.join(self.tmp, "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        return load_scene(path)

    def test_valid_scene_loads(self):
        scene = load_scene(self.scene_path)
        self.assertEqual(len(scene.volumes), 5)
        self.assertEqual(len(scene.lights), 3)

    def test_bad_sigma(self):
        self.doc["volumes"][0]["sigma_t"] = -1.0
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1003")

    def test_bad_albedo(self):
        self.doc["volumes"][0]["albedo"] = 1.5
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1003")

    def test_bad_ior(self):
        self.doc["volumes"][0]["ior"] = 0.8
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1003")

    def test_degenerate_box(self):
        self.doc["volumes"][0]["max_corner"] = self.doc["volumes"][0]["min_corner"]
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1003")

    def test_light_outside_bbox(self):
        self.doc["lights"][0]["position"] = [100.0, 100.0, 100.0]
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1004")

    def test_bad_fov(self):
        self.doc["camera"]["fov_v_deg"] = 180.0
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1005")

    def test_bad_max_steps_type(self):
        self.doc["render"]["max_steps"] = 2.5
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1006")

    def test_tile_does_not_cover_grid(self):
        self.doc["render"]["tile_width"] = 7  # 网格宽 32，不能整除
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1006")

    def test_malformed_json(self):
        path = os.path.join(self.tmp, "broken.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        with self.assertRaises(RenderError) as ctx:
            load_scene(path)
        self.assertEqual(ctx.exception.code, "E_1001")

    def test_missing_field(self):
        del self.doc["camera"]["aspect"]
        with self.assertRaises(RenderError) as ctx:
            self._load(self.doc)
        self.assertEqual(ctx.exception.code, "E_1002")

    def test_validation_produces_no_outputs(self):
        bad = dict(self.doc)
        bad["render"] = dict(bad["render"])
        bad["render"]["step_length"] = 0
        before = set(os.listdir(self.out))
        with self.assertRaises(RenderError):
            self._load(bad)
        after = set(os.listdir(self.out))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
