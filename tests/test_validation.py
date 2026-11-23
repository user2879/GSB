from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_sample, write_json

from inpaint_pyramid.config import apply_command, load_config
from inpaint_pyramid.errors import CommandError, ConfigError


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.samples = Path(cls.temp.name) / "samples"
        cls.bundle, cls.stream, _ = make_sample(cls.samples)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_sample_validates_and_invalid_command_is_explicit(self) -> None:
        self.assertEqual(self.bundle.spec.levels, 3)
        mutable = self.bundle.new_mutable()
        with self.assertRaises(CommandError) as context:
            apply_command(
                mutable,
                {
                    "id": "bad-quant",
                    "type": "set_quant_levels",
                    "layer": 0,
                    "levels": 1,
                },
                self.bundle.spec,
            )
        self.assertEqual(context.exception.code, "E_COMMAND_INVALID")
        before = mutable.to_metadata()
        valid, change = apply_command(
            mutable,
            {"id": "next", "type": "set_iterations", "layer": 1, "value": 6},
            self.bundle.spec,
        )
        self.assertEqual(valid["value"], 6)
        self.assertTrue(change.changed)
        self.assertNotEqual(before, mutable.to_metadata())

    def test_non_monotonic_factors_are_rejected(self) -> None:
        config = json.loads((self.samples / "config.json").read_text(encoding="utf-8"))
        config["pyramid"]["downsample_factors"] = [1, 4, 2]
        path = Path(self.temp.name) / "bad-factors.json"
        write_json(path, config)
        with self.assertRaises(ConfigError) as context:
            load_config(path, self.samples)
        self.assertEqual(context.exception.code, "E_CONFIG_INVALID")
        self.assertTrue(any("monotonically" in item["message"] for item in context.exception.errors))

    def test_non_divisible_tile_size_is_rejected(self) -> None:
        config = json.loads((self.samples / "config.json").read_text(encoding="utf-8"))
        config["pyramid"]["tile_size"] = 12
        path = Path(self.temp.name) / "bad-tile.json"
        write_json(path, config)
        with self.assertRaises(ConfigError) as context:
            load_config(path, self.samples)
        self.assertTrue(any("divisible" in item["message"] for item in context.exception.errors))

    def test_bad_png_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            for name in ("image_01.png", "style_ref.png"):
                (directory / name).write_bytes(b"not a png")
            config = json.loads((self.samples / "config.json").read_text(encoding="utf-8"))
            path = directory / "config.json"
            write_json(path, config)
            with self.assertRaises(ConfigError) as context:
                load_config(path, directory)
            self.assertTrue(any("PNG" in item["message"] for item in context.exception.errors))


if __name__ == "__main__":
    unittest.main()
