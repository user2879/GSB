from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from layout_diag.editing import apply_edit_stream, load_edit_commands
from layout_diag.generator import ImageSpec
from tests.helpers import make_dataset, read_config


class EditingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.samples = make_dataset(
            cls.root / "samples",
            specs=[ImageSpec("small_layout_01.png", 2, 3, 0.0012, 5.0)],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_command_loader_rejects_non_object_entries(self) -> None:
        path = self.root / "bad-commands.json"
        path.write_text(json.dumps({"commands": [{"op": "set_tier", "tier": "small"}, "bad"]}))
        with self.assertRaises(ValueError):
            load_edit_commands(path)

    def test_missing_replacement_source_does_not_mutate_image(self) -> None:
        root = self.root / "missing-source"
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(self.samples, root)
        config_path = root / "config.json"
        image_path = root / "images" / "small_layout_01.png"
        before = image_path.read_bytes()
        failures = apply_edit_stream(
            root,
            config_path,
            [{"op": "replace_image", "image": "small_layout_01.png", "source": str(root / "does-not-exist.png")}],
        )
        self.assertIn("E_EDIT_COMMAND", [failure.code for failure in failures.items])
        self.assertEqual(before, image_path.read_bytes())

    def test_config_edit_changes_custom_parameters_and_tier(self) -> None:
        root = self.root / "config-edits"
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(self.samples, root)
        config_path = root / "config.json"
        failures = apply_edit_stream(
            root,
            config_path,
            [
                {"op": "set_tier", "tier": "medium"},
                {"op": "set_window", "width": 32, "height": 32, "step_x": 16, "step_y": 16},
                {"op": "set_merge_threshold", "value": 4},
                {"op": "set_quantization", "fixed_bits": 10, "quant_step": 1},
            ],
        )
        self.assertFalse(failures.items)
        config = read_config(config_path)
        self.assertEqual(config["tier"], "medium")
        self.assertEqual(config["window"]["width"], 32)
        self.assertEqual(config["cc"]["merge_gap"], 4)
        self.assertEqual(config["strategies"]["medium"]["window"]["width"], 32)
        self.assertEqual(config["strategies"]["medium"]["cc"]["merge_gap"], 4)
        self.assertEqual(config["strategies"]["medium"]["precision"]["fixed_bits"], 10)


if __name__ == "__main__":
    unittest.main()
