from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from renderdiag.config import load_config
from renderdiag.errors import DiagnosticError, E_CONFIG_INVALID
from renderdiag.resources import (
    load_command_stream,
    load_image,
    load_motion_table,
    load_prompt_set,
)
from tests.support import SampleWorkspace


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = SampleWorkspace()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.sample.close()

    def test_valid_sample_contracts(self) -> None:
        config = self.sample.config
        prompts = load_prompt_set(config.resources.prompts, config)
        image = load_image(config.resources.image, config)
        motion = load_motion_table(config.resources.motion, config)
        commands = load_command_stream(self.sample.paths["commands"])
        self.assertEqual(len(prompts.prompts), 3)
        self.assertEqual((image.width, image.height, image.mode), (32, 32, "RGB"))
        self.assertEqual(motion.shape, (3, 4, 4, 2))
        self.assertEqual(len(commands), 13)

    def test_unknown_prompt_term_is_rejected_with_entry(self) -> None:
        prompts_path = self.sample.root / "bad_prompts.json"
        data = json.loads(Path(self.sample.paths["prompts"]).read_text(encoding="utf-8"))
        data["prompts"]["p0"]["attributes"].append("zebra")
        prompts_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(DiagnosticError) as caught:
            load_prompt_set(prompts_path, self.sample.config)
        self.assertEqual(caught.exception.code, E_CONFIG_INVALID)
        self.assertIn("prompts.prompts.p0.attributes[2]", caught.exception.entry)

    def test_quantization_enumeration_and_block_divisibility(self) -> None:
        raw = yaml.safe_load(Path(self.sample.paths["config"]).read_text(encoding="utf-8"))
        raw["condition"]["quantization_level"] = 9
        bad_quant = self.sample.root / "bad_quant.yaml"
        bad_quant.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with self.assertRaises(DiagnosticError) as caught:
            load_config(bad_quant)
        self.assertEqual(caught.exception.code, E_CONFIG_INVALID)
        self.assertEqual(caught.exception.entry, "condition.quantization_level")

        raw["condition"]["quantization_level"] = 2
        raw["render"]["block_rows"] = 3
        bad_rows = self.sample.root / "bad_rows.yaml"
        bad_rows.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        with self.assertRaises(DiagnosticError) as caught:
            load_config(bad_rows)
        self.assertEqual(caught.exception.code, E_CONFIG_INVALID)
        self.assertEqual(caught.exception.entry, "render.block_rows")

    def test_png_pixel_mode_is_validated(self) -> None:
        path = self.sample.root / "rgba.png"
        Image.new("RGBA", (32, 32), (1, 2, 3, 255)).save(path)
        with self.assertRaises(DiagnosticError) as caught:
            load_image(path, self.sample.config)
        self.assertEqual(caught.exception.code, E_CONFIG_INVALID)
        self.assertIn("image.pixel_mode", caught.exception.entry)

    def test_motion_shape_and_command_op_are_validated(self) -> None:
        path = self.sample.root / "bad_motion.npy"
        np.save(path, np.zeros((3, 4, 3, 2), dtype=np.int32), allow_pickle=False)
        with self.assertRaises(DiagnosticError) as caught:
            load_motion_table(path, self.sample.config)
        self.assertEqual(caught.exception.code, E_CONFIG_INVALID)
        self.assertEqual(caught.exception.entry, "motion.shape")

        commands_path = self.sample.root / "bad_commands.jsonl"
        commands_path.write_text('{"op":"teleport"}\n', encoding="utf-8")
        with self.assertRaises(DiagnosticError) as caught:
            load_command_stream(commands_path)
        self.assertEqual(caught.exception.code, E_CONFIG_INVALID)
        self.assertEqual(caught.exception.entry, "commands[0].op")


if __name__ == "__main__":
    unittest.main()
