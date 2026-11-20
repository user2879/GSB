import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import numpy as np

from texmat_pipeline.errors import (
    E_CONFIG_INVALID,
    E_LAYER_INVALID,
    E_TEX_DECODE,
    E_WEIGHT_INVALID,
)
from texmat_pipeline.loader import scan_textures
from texmat_pipeline.models import load_generation_config, load_materials

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_samples.py"


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="texmat-validate-"))
        cls.sample = cls.tmp / "sample"
        subprocess.run(
            [sys.executable, str(GENERATOR), "--out", str(cls.sample), "--tiers", "16"],
            check=True, capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_valid_inputs_have_no_failures(self):
        config, cfg_fail = load_generation_config(self.sample / "generation_config.json")
        materials, mat_fail = load_materials(self.sample / "materials.json")
        self.assertEqual(cfg_fail + mat_fail, [])
        self.assertGreater(len(config.layers), 0)
        self.assertEqual(len(materials), 2)
        textures, failures = scan_textures(self.sample, 16)
        self.assertEqual(failures, [])
        self.assertEqual({t.mode for t in textures.values()}, {"L", "RGB"})
        for tex in textures.values():
            self.assertEqual(tex.gray.dtype, np.uint8)
            self.assertEqual(tex.gray.shape, (16, 16))

    def test_undecodable_and_bad_signature(self):
        bad_dir = self.tmp / "bad"
        shutil.copytree(self.sample, bad_dir)
        (bad_dir / "broken.png").write_bytes(b"not a real png file")
        img = Image.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
        img.save(bad_dir / "good_16.png")
        # corrupt a valid PNG's signature
        valid = bad_dir / "stone_16.png"
        raw = bytearray(valid.read_bytes())
        raw[0] = 0
        valid.write_bytes(bytes(raw))
        textures, failures = scan_textures(bad_dir, 16)
        codes = {f.code for f in failures}
        targets = {Path(f.target).name for f in failures}
        self.assertIn(E_TEX_DECODE, codes)
        self.assertIn("broken.png", targets)
        self.assertIn("stone_16.png", targets)
        # remaining valid textures still decode
        self.assertIn("wood_16", textures)

    def test_size_and_mode_validation(self):
        odd_dir = self.tmp / "odd"
        odd_dir.mkdir()
        Image.fromarray(np.zeros((12, 16), dtype=np.uint8), mode="L").save(odd_dir / "rect_16.png")
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="P").save(odd_dir / "pal_16.png")
        _, failures = scan_textures(odd_dir, 16)
        self.assertTrue(all(f.code == E_TEX_DECODE for f in failures))
        self.assertEqual(len(failures), 2)

    def test_layer_invalid_frequency_amplitude(self):
        cfg = json.loads((self.sample / "generation_config.json").read_text())
        cfg["layers"][0]["frequency"] = 0.0
        cfg["layers"][1]["amplitude"] = float("nan")
        path = self.tmp / "bad_layers.json"
        path.write_text(json.dumps(cfg))
        config, failures = load_generation_config(path)
        self.assertIsNone(config)
        self.assertTrue(all(f.code == E_LAYER_INVALID for f in failures))
        self.assertEqual(len(failures), 2)

    def test_resolution_tiers_must_increase(self):
        cfg = json.loads((self.sample / "generation_config.json").read_text())
        cfg["resolution_tiers"] = [32, 32, 16]
        path = self.tmp / "bad_tiers.json"
        path.write_text(json.dumps(cfg))
        config, failures = load_generation_config(path)
        self.assertIsNone(config)
        self.assertEqual(failures[0].code, E_CONFIG_INVALID)

    def test_fixed_point_width_validation(self):
        cfg = json.loads((self.sample / "generation_config.json").read_text())
        cfg["fixed_point"]["frac_bits"] = 64
        path = self.tmp / "bad_fp.json"
        path.write_text(json.dumps(cfg))
        config, failures = load_generation_config(path)
        self.assertIsNone(config)
        self.assertEqual(failures[0].code, E_CONFIG_INVALID)

    def test_non_positive_weight_sum(self):
        mats = json.loads((self.sample / "materials.json").read_text())
        mats["materials"][0]["layer_weights"]["diffuse"] = {"0": 0.0, "1": 0.0}
        path = self.tmp / "bad_weights.json"
        path.write_text(json.dumps(mats))
        materials, failures = load_materials(path)
        self.assertTrue(any(f.code == E_WEIGHT_INVALID for f in failures))
        # the other material still parses
        self.assertEqual([m.material_id for m in materials], ["mat_bronze"])


if __name__ == "__main__":
    unittest.main()
