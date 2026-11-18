import json
import tempfile
import unittest
from pathlib import Path

from texmat.config import load_generation_config, load_materials
from texmat.errors import ConfigInvalid


def valid_config() -> dict:
    return {
        "resolution_tiers": [8, 16, 32],
        "active_tier": 16,
        "noise_layers": [
            {"source_texture": "tex_a", "frequency": 4, "amplitude": 0.25, "seed": 1},
        ],
        "blend_mode": "weighted",
        "fractional_bits": 8,
        "rounding": "half_even",
        "workers": 2,
        "point_budget": 1000,
        "snapshot_interval": 1,
    }


def good_material() -> dict:
    return {
        "name": "m1", "layers": ["layer00"], "weights": [1.0], "blend_mode": "weighted",
        "channels": {c: {"base": 0.1, "scale": 0.5, "bias": 0.0}
                     for c in ("diffuse", "roughness", "metalness", "normal")},
    }


class ConfigValidationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_config(self, payload) -> Path:
        path = self.tmp / "generation_config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_config(self):
        config = load_generation_config(self._write_config(valid_config()))
        self.assertEqual(config.active_tier, 16)
        self.assertEqual(config.noise_layers[0].frequency, 4)

    def test_non_monotonic_tiers(self):
        payload = valid_config()
        payload["resolution_tiers"] = [32, 16]
        with self.assertRaisesRegex(ConfigInvalid, "monotonically increasing"):
            load_generation_config(self._write_config(payload))

    def test_bad_frequency_amplitude(self):
        payload = valid_config()
        payload["noise_layers"][0]["frequency"] = 0
        with self.assertRaisesRegex(ConfigInvalid, "frequency"):
            load_generation_config(self._write_config(payload))
        payload = valid_config()
        payload["noise_layers"][0]["amplitude"] = 0
        with self.assertRaisesRegex(ConfigInvalid, "amplitude"):
            load_generation_config(self._write_config(payload))
        payload = valid_config()
        payload["noise_layers"][0]["amplitude"] = float("inf")
        with self.assertRaisesRegex(ConfigInvalid, "amplitude"):
            load_generation_config(self._write_config(payload))

    def test_bits_range(self):
        payload = valid_config()
        payload["fractional_bits"] = 20
        with self.assertRaisesRegex(ConfigInvalid, "fractional_bits"):
            load_generation_config(self._write_config(payload))

    def test_material_channels_and_weights(self):
        config = load_generation_config(self._write_config(valid_config()))
        materials_path = self.tmp / "materials.json"
        materials_path.write_text(json.dumps([good_material()]), encoding="utf-8")
        self.assertEqual(load_materials(materials_path, config)[0].name, "m1")

        bad = [json.loads(json.dumps(good_material()))]
        del bad[0]["channels"]["normal"]
        materials_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ConfigInvalid, "missing channels"):
            load_materials(materials_path, config)

        bad = [json.loads(json.dumps(good_material()))]
        bad[0]["weights"] = [0.0]
        materials_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ConfigInvalid, "positive sum"):
            load_materials(materials_path, config)

    def test_unknown_layer_reference(self):
        config = load_generation_config(self._write_config(valid_config()))
        materials_path = self.tmp / "materials.json"
        bad = [good_material()]
        bad[0]["layers"] = ["layer99"]
        materials_path.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ConfigInvalid, "unknown layer"):
            load_materials(materials_path, config)

    def test_bad_json(self):
        path = self.tmp / "generation_config.json"
        path.write_text("{not valid json", encoding="utf-8")
        with self.assertRaisesRegex(ConfigInvalid, "invalid json"):
            load_generation_config(path)


if __name__ == "__main__":
    unittest.main()
