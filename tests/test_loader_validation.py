import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from fxrender.codes import BizCode
from fxrender.config import RenderConfig
from fxrender.loader import (inspect_png_container, load_edit_stream,
                             load_motion_table, load_prompts,
                             load_vocabulary)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.cfg = RenderConfig()

    def test_prompt_oov_is_skipped_with_code(self):
        vocab = self.work / "vocab.txt"
        vocab.write_text("a\nb\n" + "\n".join(
            f"t{i}" for i in range(10)), encoding="utf-8")
        prompts = self.work / "prompts.txt"
        prompts.write_text("a b\nzzz b\na t1\n", encoding="utf-8")
        terms, _ = load_vocabulary(vocab)
        accepted, checks = load_prompts(prompts, terms)
        self.assertEqual(len(accepted), 2)
        codes = [c.code for c in checks]
        self.assertIn(BizCode.E_PROMPT_OOV, codes)
        self.assertTrue(all(c.accepted for c in checks
                            if c.code == BizCode.OK))

    def test_png_header_size_mode_codes(self):
        good = self.work / "good.png"
        Image.new("RGB", (self.cfg.image_width, self.cfg.image_height),
                  (1, 2, 3)).save(good, format="PNG")
        checks = inspect_png_container(good, self.cfg)
        self.assertFalse(any(not c.accepted for c in checks))

        bad_sig = self.work / "bad.png"
        bad_sig.write_bytes(b"not-a-png" + b"0" * 40)
        checks = inspect_png_container(bad_sig, self.cfg)
        self.assertIn(BizCode.E_PNG_HEADER, [c.code for c in checks])

        wrong_size = self.work / "size.png"
        Image.new("RGB", (8, 8)).save(wrong_size, format="PNG")
        checks = inspect_png_container(wrong_size, self.cfg)
        self.assertIn(BizCode.E_PNG_SIZE, [c.code for c in checks])

    def test_motion_shape_validation(self):
        good = self.work / "motion.json"
        rows = [[{"dx": 1, "dy": -1} for _ in range(self.cfg.tile_cols)]
                for _ in range(self.cfg.tile_rows)]
        good.write_text(json.dumps({"rows": rows}), encoding="utf-8")
        table, checks = load_motion_table(good, self.cfg)
        self.assertIsNotNone(table)
        bad = self.work / "bad.json"
        bad.write_text(json.dumps({"rows": rows[:-1]}), encoding="utf-8")
        table, checks = load_motion_table(bad, self.cfg)
        self.assertIsNone(table)
        self.assertIn(BizCode.E_MOTION_SHAPE, [c.code for c in checks])

    def test_config_enum_and_positive_int(self):
        with self.assertRaises(ValueError) as ctx:
            RenderConfig.from_dict({"fp_width": 24})
        self.assertIn(BizCode.E_FP_ENUM.value, str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            RenderConfig.from_dict({"denoise_steps": 0})
        self.assertIn(BizCode.E_POSITIVE_INT.value, str(ctx.exception))
        with self.assertRaises(ValueError):
            RenderConfig.from_dict({"unknown_key": 1})

    def test_edit_stream_rejects_unknown_op(self):
        path = self.work / "edits.json"
        path.write_text(json.dumps({"commands": [
            {"op": "set_weight", "condition": "text", "value": 0.2},
            {"op": "explode"}]}), encoding="utf-8")
        commands, checks = load_edit_stream(path)
        self.assertEqual(len(commands), 1)
        self.assertIn(BizCode.E_BAD_COMMAND, [c.code for c in checks])


if __name__ == "__main__":
    unittest.main()
