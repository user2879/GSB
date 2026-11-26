import json
import os
import tempfile
import unittest

from fixedpoint_render.config import RenderConfig
from fixedpoint_render.errors import ErrorBundle
from fixedpoint_render.samplegen import default_config, generate_workspace
from fixedpoint_render.workspace import load_workspace


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp(prefix="fpr-val-")
        generate_workspace(self.temp, default_config(), 3)

    def test_valid_workspace_loads(self):
        loaded = load_workspace(self.temp)
        self.assertEqual(loaded["frame_indices"], [0, 1, 2])
        self.assertEqual(len(loaded["motions"][0]["tiles"]),
                         default_config().tile_rows)

    def _break(self, relpath, transform):
        path = os.path.join(self.temp, relpath)
        transform(path)

    def test_bad_png_header(self):
        def corrupt(path):
            with open(path, "wb") as fh:
                fh.write(b"NOTPNG" + b"0" * 32)
        self._break("images/frame_00.png", corrupt)
        with self.assertRaises(ErrorBundle) as ctx:
            load_workspace(self.temp)
        self.assertIn("E_PNG_HEADER", ctx.exception.codes())

    def test_bad_resolution(self):
        from PIL import Image
        def shrink(path):
            Image.new("RGB", (4, 4)).save(path)
        self._break("images/frame_01.png", shrink)
        with self.assertRaises(ErrorBundle) as ctx:
            load_workspace(self.temp)
        self.assertIn("E_PNG_RESOLUTION", ctx.exception.codes())

    def test_vocab_violation(self):
        def bad_prompt(path):
            with open(path, "w") as fh:
                fh.write("city word_not_in_vocab")
        self._break("prompts/frame_00.txt", bad_prompt)
        with self.assertRaises(ErrorBundle) as ctx:
            load_workspace(self.temp)
        self.assertIn("E_VOCAB_TERM", ctx.exception.codes())
        err = [e for e in ctx.exception.errors if e.code == "E_VOCAB_TERM"][0]
        self.assertIn("frame_00", err.location)
        self.assertIn("word_not_in_vocab", err.field)

    def test_motion_tile_count(self):
        path = os.path.join(self.temp, "motion/frame_00.json")
        with open(path) as fh:
            data = json.load(fh)
        data["tiles"].pop()
        with open(path, "w") as fh:
            json.dump(data, fh)
        with self.assertRaises(ErrorBundle) as ctx:
            load_workspace(self.temp)
        self.assertIn("E_MOTION_TILES", ctx.exception.codes())

    def test_bad_qmode_and_budget(self):
        path = os.path.join(self.temp, "config.json")
        with open(path) as fh:
            cfg = json.load(fh)
        cfg["qmode"] = "weird"
        cfg["per_frame_budget"] = 0
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        with self.assertRaises(ErrorBundle) as ctx:
            load_workspace(self.temp)
        codes = ctx.exception.codes()
        self.assertIn("E_QMODE_ENUM", codes)
        self.assertIn("E_BUDGET_POSITIVE", codes)

    def test_render_blocked_on_invalid_input(self):
        # Validation errors must prevent any snapshot from being produced.
        path = os.path.join(self.temp, "config.json")
        with open(path) as fh:
            cfg = json.load(fh)
        cfg["frac_bits"] = 7
        with open(path, "w") as fh:
            json.dump(cfg, fh)
        with self.assertRaises(ErrorBundle):
            load_workspace(self.temp)
        self.assertFalse(os.path.exists(os.path.join(self.temp, "snapshots")))


if __name__ == "__main__":
    unittest.main()
