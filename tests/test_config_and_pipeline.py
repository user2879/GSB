import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from render_workbench.config import (
    load_prompts, validate_config,
)
from render_workbench import encoder as encoder_mod
from render_workbench import latent as latent_mod
from render_workbench import decoder as decoder_mod
from render_workbench.errors import (
    E_CONFIG_INVALID, E_PATCH_OUT_OF_RANGE, E_TOKEN_OUT_OF_VOCAB,
)
from render_workbench.hashing import fingerprint
from render_workbench.png_codec import encode_png


def base_config():
    return {
        "vocab": ["a", "b", "c"],
        "embedding_dim": 6,
        "latent_channels": 3,
        "seed": 42,
        "seed_min": 1,
        "seed_max": 1000,
        "denoise_steps": 3,
        "guidance_scale": 1.0,
        "img2img_replace_steps": 1,
        "patch_tolerance": 0.01,
        "concurrency": 1,
        "snapshot_interval": 1,
        "tiers": [
            {"id": "t1", "width": 4, "height": 4,
             "noise": [0.1, 0.2, 0.3], "guidance": [0.4, 0.5, 0.6]},
            {"id": "t2", "width": 8, "height": 6,
             "noise": [0.2, 0.3, 0.4], "guidance": [0.5, 0.6, 0.7]},
        ],
        "noise_tables": {
            "t1": [0.1, 0.2, 0.3], "t2": [0.2, 0.3, 0.4]},
        "guidance_tables": {
            "t1": [0.4, 0.5, 0.6], "t2": [0.5, 0.6, 0.7]},
    }


class ConfigValidationTests(unittest.TestCase):
    def test_valid_config_passes(self):
        cfg, errors = validate_config(base_config())
        self.assertEqual(errors, [])
        self.assertEqual([t["id"] for t in cfg["tiers"]], ["t1", "t2"])

    def test_bad_vocab_and_seed(self):
        raw = base_config()
        raw["vocab"] = []
        raw["seed"] = 9999
        cfg, errors = validate_config(raw)
        codes = {e["code"] for e in errors}
        self.assertIn(E_CONFIG_INVALID, codes)

    def test_non_monotonic_tiers(self):
        raw = base_config()
        raw["tiers"][1]["width"], raw["tiers"][1]["height"] = 4, 4
        _, errors = validate_config(raw)
        self.assertTrue(any("monotonic" in e["message"] for e in errors))

    def test_table_length_mismatch(self):
        raw = base_config()
        raw["noise_tables"]["t1"] = [0.1, 0.2]
        _, errors = validate_config(raw)
        self.assertTrue(any("denoise_steps" in e["message"] for e in errors))

    def test_prompt_token_out_of_vocab(self):
        cfg, _ = validate_config(base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.jsonl")
            with open(path, "w", encoding="utf-8") as h:
                h.write(json.dumps({"id": "p1", "tokens": ["a", "zzz"]}) + "\n")
            prompts, errors = load_prompts(path, cfg)
            self.assertEqual(prompts, [])
            self.assertEqual(errors[0]["code"], E_TOKEN_OUT_OF_VOCAB)

    def test_duplicate_prompt_id_collected_not_raised(self):
        cfg, _ = validate_config(base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.jsonl")
            with open(path, "w", encoding="utf-8") as h:
                h.write(json.dumps({"id": "p1", "tokens": ["a"]}) + "\n")
                h.write(json.dumps({"id": "p1", "tokens": ["b"]}) + "\n")
            prompts, errors = load_prompts(path, cfg)
            self.assertEqual(len(prompts), 1)
            self.assertTrue(errors)

    def test_patch_out_of_range(self):
        cfg, _ = validate_config(base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.jsonl")
            record = {
                "id": "p1", "tokens": ["a"],
                "patches": [{"step": 0, "x": 3, "y": 3, "width": 4,
                             "height": 2,
                             "values": [0.0] * (4 * 2 * 3)}],
            }
            with open(path, "w", encoding="utf-8") as h:
                h.write(json.dumps(record) + "\n")
            prompts, errors = load_prompts(path, cfg)
            self.assertEqual(prompts, [])
            self.assertEqual(errors[0]["code"], E_PATCH_OUT_OF_RANGE)

    def test_missing_init_image_recorded(self):
        cfg, _ = validate_config(base_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "p.jsonl")
            with open(path, "w", encoding="utf-8") as h:
                h.write(json.dumps({"id": "p1", "tokens": ["a"],
                                    "init_image": "nope.png"}) + "\n")
            prompts, errors = load_prompts(path, cfg)
            self.assertEqual(prompts, [])
            self.assertEqual(errors[0]["code"], "E_INPUT_MISSING")


class AlgorithmDeterminismTests(unittest.TestCase):
    def setUp(self):
        cfg, _ = validate_config(base_config())
        self.cfg = cfg
        self.tier = cfg["tiers"][0]
        self.config_fp = fingerprint({"v": 1})
        self.prompt = {"id": "p", "tokens": ["a", "b", "a"]}

    def test_embedding_normalized_and_deterministic(self):
        r1 = encoder_mod.encode_prompt(self.prompt, self.cfg["vocab"], 6)
        r2 = encoder_mod.encode_prompt(self.prompt, self.cfg["vocab"], 6)
        np.testing.assert_array_equal(r1["embedding"], r2["embedding"])
        self.assertAlmostEqual(float(np.linalg.norm(r1["embedding"])), 1.0)
        self.assertAlmostEqual(float(r1["token_counts"][0]), 2 / 3)

    def test_latent_uses_isolated_rng(self):
        prompt_fp = encoder_mod.prompt_fingerprint(self.prompt)
        a = latent_mod.initialize_latent(
            prompt_fp, 42, self.config_fp, self.tier, 3)
        b = latent_mod.initialize_latent(
            prompt_fp, 42, self.config_fp, self.tier, 3)
        c = latent_mod.initialize_latent(
            prompt_fp, 43, self.config_fp, self.tier, 3)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_denoise_fixed_order_and_path_independence(self):
        prompt_fp = encoder_mod.prompt_fingerprint(self.prompt)
        emb = encoder_mod.encode_prompt(
            self.prompt, self.cfg["vocab"], 6)["embedding"]
        latent = latent_mod.initialize_latent(
            prompt_fp, 42, self.config_fp, self.tier, 3)
        out_a = latent_mod.denoise_step(
            latent, 0, emb, self.tier, 6, 3, 1.0, self.config_fp)
        out_b = latent_mod.denoise_step(
            latent, 0, emb, self.tier, 6, 3, 1.0, self.config_fp)
        np.testing.assert_array_equal(out_a, out_b)

    def test_patch_noop_tolerance(self):
        latent = np.zeros((4, 4, 3), dtype=np.float64)
        patch = {"step": 0, "x": 0, "y": 0, "width": 2, "height": 2,
                 "values": [0.001] * (2 * 2 * 3)}
        self.assertTrue(latent_mod.patch_is_noop(latent, patch, 0.01))
        patch["values"] = [1.0] * (2 * 2 * 3)
        self.assertFalse(latent_mod.patch_is_noop(latent, patch, 0.01))
        patched = latent_mod.apply_patch(latent, patch)
        self.assertEqual(float(patched[0, 0, 0]), 1.0)
        self.assertEqual(float(latent[0, 0, 0]), 0.0)

    def test_decoder_tier_changes_pixels(self):
        latent = np.random.RandomState(1).standard_normal((4, 4, 3))
        f1 = decoder_mod.decode_frame(
            latent, 0, self.cfg["tiers"][0], 3, self.config_fp)
        f2 = decoder_mod.decode_frame(
            latent, 0, self.cfg["tiers"][0], 3, self.config_fp)
        np.testing.assert_array_equal(f1["pixels"], f2["pixels"])
        self.assertEqual(f1["pixels"].dtype, np.uint8)
        latent_big = np.random.RandomState(1).standard_normal((6, 8, 3))
        f_big = decoder_mod.decode_frame(
            latent_big, 0, self.cfg["tiers"][1], 3, self.config_fp)
        self.assertEqual(f_big["pixels"].shape, (6, 8, 3))
        self.assertNotEqual(f1["pixel_fp"], f_big["pixel_fp"])

    def test_image_encode_deterministic(self):
        pixels = np.random.RandomState(2).randint(
            0, 256, size=(10, 10, 3)).astype(np.uint8)
        a = latent_mod.encode_init_image(
            pixels, self.tier, 3, "pfp", "ifp", self.config_fp)
        b = latent_mod.encode_init_image(
            pixels, self.tier, 3, "pfp", "ifp", self.config_fp)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (4, 4, 3))


if __name__ == "__main__":
    unittest.main()
