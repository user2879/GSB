from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.helpers import create_sample, write_json

import repair_studio  # noqa: F401
from repair_studio.config import apply_command
from repair_studio.imaging import load_sample_set


class ValidationTests(unittest.TestCase):
    def test_failures_use_required_business_codes_and_continue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_sample(root, "00_good", width=48, height=40)

            bad_mask = root / "10_bad_mask"
            shutil.copytree(root / "00_good", bad_mask)
            mask_config = json.loads((bad_mask / "config.json").read_text(encoding="utf-8"))
            mask_config["mask"] = {"x": 40, "y": 2, "width": 20, "height": 10}
            write_json(bad_mask / "config.json", mask_config)

            bad_config = root / "20_bad_config"
            shutil.copytree(root / "00_good", bad_config)
            config_data = json.loads((bad_config / "config.json").read_text(encoding="utf-8"))
            del config_data["repair"]["radius"]
            write_json(bad_config / "config.json", config_data)

            bad_image = root / "30_bad_image"
            shutil.copytree(root / "00_good", bad_image)
            (bad_image / "image.png").write_bytes(b"not a png")

            broken_chunk = root / "40_bad_chunk"
            shutil.copytree(root / "00_good", broken_chunk)
            chunk_config = json.loads((broken_chunk / "config.json").read_text(encoding="utf-8"))
            chunk_config["interaction"]["chunk_rows"] = 1000
            write_json(broken_chunk / "config.json", chunk_config)

            loaded, failures = load_sample_set(root)
            self.assertEqual([sample.sample_id for sample in loaded], ["00_good"])
            self.assertEqual(
                [item["sample_id"] for item in failures],
                ["10_bad_mask", "20_bad_config", "30_bad_image", "40_bad_chunk"],
            )
            self.assertEqual(failures[0]["code"], "E_MASK_OUT_OF_RANGE")
            self.assertEqual(failures[1]["code"], "E_CONFIG_INVALID")
            self.assertEqual(failures[2]["code"], "E_IMAGE_DECODE")
            self.assertEqual(failures[3]["code"], "E_CONFIG_INVALID")

    def test_invalid_fixed_width_and_quant_levels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root, "invalid_fixed", width=40, height=32)
            config_path = sample.directory / "config.json"
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
            config_data["fixed_bits"] = 99
            write_json(config_path, config_data)
            loaded, failures = load_sample_set(root)
            self.assertEqual(loaded, [])
            self.assertEqual(failures[0]["code"], "E_CONFIG_INVALID")

            config_data["fixed_bits"] = 16
            config_data["style"]["quant_levels"] = 12
            write_json(config_path, config_data)
            loaded, failures = load_sample_set(root)
            self.assertEqual(loaded, [])
            self.assertEqual(failures[0]["code"], "E_CONFIG_INVALID")

    def test_mask_tuple_channel_object_and_aliases_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = create_sample(root, "aliases", width=48, height=40)
            config_path = sample.directory / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["mask"] = [8, 6, 24, 18]
            config["style"]["channel_weights"] = {"r": 0.1, "g": 0.5, "b": 0.9}
            config["fixed_point_bits"] = config.pop("fixed_bits")
            config["interaction"]["threads"] = config["interaction"].pop("thread_workers")
            write_json(config_path, config)
            commands_path = sample.directory / "commands.json"
            write_json(
                commands_path,
                {
                    "commands": [
                        {"op": "set_style_weights", "channel_weights": {"red": 0.2, "green": 0.6, "blue": 0.8}},
                        {"op": "set_mask_rect", "x": 10, "y": 8, "w": 20, "h": 14},
                    ]
                },
            )
            loaded, failures = load_sample_set(root)
            self.assertEqual(failures, [])
            self.assertEqual((loaded[0].config.params.mask.x, loaded[0].config.params.mask.width), (8, 24))
            self.assertEqual(loaded[0].config.thread_workers, 4)
            self.assertEqual(loaded[0].config.fixed_bits, 16)
            changed = apply_command(
                loaded[0].config.params,
                loaded[0].config.width,
                loaded[0].config.height,
                {"op": "set_style_weights", "channel_weights": {"r": 0.2, "g": 0.6, "b": 0.8}},
            )
            self.assertEqual(changed.channel_weights_q, (13107, 39322, 52429))
            changed = apply_command(
                changed,
                loaded[0].config.width,
                loaded[0].config.height,
                {"op": "set_mask_rect", "x": 10, "y": 8, "w": 20, "h": 14},
            )
            self.assertEqual((changed.mask.x, changed.mask.y, changed.mask.width, changed.mask.height), (10, 8, 20, 14))


if __name__ == "__main__":
    unittest.main()
