import tempfile
import unittest
from pathlib import Path

import numpy as np

from fxrender.config import RenderConfig
from fxrender.encoding import (encode_image, encode_motion, encode_text,
                               fuse_conditions)
from fxrender.fixedpoint import FpSpec
from fxrender.kernel import (TileAccessGuard, TileBorrowError, decode_tile_row,
                             denoise_step, init_latent)
from fxrender.policy import merged_step_groups, select_policy, tile_decode_order
from fxrender.samplegen import (generate_motion_table, generate_vocabulary,
                                generate_prompts, _draw_png)
from PIL import Image
import io


def motion_array(cfg):
    rows = generate_motion_table(cfg)["rows"]
    return np.asarray(
        [[[cell["dx"], cell["dy"]] for cell in row] for row in rows],
        dtype=np.int64)


class EncodingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RenderConfig()
        self.vocab = generate_vocabulary()
        self.prompts = generate_prompts(self.vocab, 3)
        png = _draw_png(1, self.cfg.image_width, self.cfg.image_height)
        self.pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        self.motion = motion_array(self.cfg)

    def test_text_bow_deterministic_and_hits(self):
        first = encode_text(self.prompts[0], self.vocab, self.cfg)
        second = encode_text(self.prompts[0], self.vocab, self.cfg)
        self.assertEqual(first.vector, second.vector)
        self.assertEqual(first.hits, sorted(first.hits))
        self.assertEqual(len(first.vector), self.cfg.text_dim)

    def test_quant_switch_changes_integers(self):
        coarse_cfg = RenderConfig(fp_width=16, fp_tier="coarse")
        fine_cfg = RenderConfig(fp_width=32, fp_tier="fine")
        a = encode_text(self.prompts[0], self.vocab, coarse_cfg)
        b = encode_text(self.prompts[0], self.vocab, fine_cfg)
        self.assertNotEqual(a.vector, b.vector)

    def test_fusion_order_and_residual_trace(self):
        enc = [encode_text(self.prompts[0], self.vocab, self.cfg),
               encode_image(self.pixels, self.cfg),
               encode_motion(self.motion, self.cfg)]
        fused = fuse_conditions(enc, self.cfg)
        self.assertEqual(fused.trace.order, ["image", "motion", "text"])
        self.assertTrue(all(isinstance(v, int) for v in fused.vector))
        again = fuse_conditions(enc, self.cfg)
        self.assertEqual(again.vector, fused.vector)


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RenderConfig()
        vocab = generate_vocabulary()
        prompts = generate_prompts(vocab, 2)
        png = _draw_png(2, self.cfg.image_width, self.cfg.image_height)
        pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        motion = motion_array(self.cfg)
        enc = [encode_text(prompts[0], vocab, self.cfg),
               encode_image(pixels, self.cfg),
               encode_motion(motion, self.cfg)]
        self.fused = fuse_conditions(enc, self.cfg)

    def test_latent_and_step_deterministic(self):
        latent_a = init_latent(self.cfg, 0, self.fused)
        latent_b = init_latent(self.cfg, 0, self.fused)
        self.assertEqual(latent_a, latent_b)
        from fxrender.kernel import build_noise_table
        noise = build_noise_table(self.cfg, 0, self.fused.trace.content_fp)
        step_a = denoise_step(self.cfg, 0, 0, latent_a, self.fused, noise[0])
        step_b = denoise_step(self.cfg, 0, 0, latent_a, self.fused, noise[0])
        self.assertEqual(step_a.latent, step_b.latent)

    def test_tile_borrow_guard_blocks_neighbours(self):
        latent = init_latent(self.cfg, 0, self.fused)
        guard = TileAccessGuard(latent, 0, self.cfg.tile_rows,
                                self.cfg.latent_dim)
        outside = guard.allowed.stop
        with self.assertRaises(TileBorrowError) as ctx:
            guard.read(outside)
        self.assertEqual(ctx.exception.code.value, "E_TILE_BORROW")

    def test_decode_partitions_change_with_tile_rows(self):
        latent = init_latent(self.cfg, 0, self.fused)
        png_a, digest_a, reads_a = decode_tile_row(self.cfg, 0, 0, latent)
        cfg2 = RenderConfig(tile_rows=2)
        png_b, digest_b, reads_b = decode_tile_row(cfg2, 0, 0, latent)
        self.assertNotEqual(reads_a, reads_b)
        self.assertTrue(digest_a and digest_b)


class PolicyTests(unittest.TestCase):
    def test_tier_transitions_change_path(self):
        cfg = RenderConfig()
        small = select_policy(2, cfg)
        medium = select_policy(8, cfg)
        large = select_policy(20, cfg)
        self.assertEqual((small.tier, medium.tier, large.tier),
                         ("small", "medium", "large"))
        self.assertNotEqual(small.batch_steps, large.batch_steps)
        self.assertEqual(tile_decode_order(4, small), [0, 1, 2, 3])
        self.assertEqual(tile_decode_order(4, large), [0, 3, 1, 2])

    def test_merge_groups(self):
        cfg = RenderConfig()
        policy = select_policy(2, cfg)
        groups = merged_step_groups(cfg.denoise_steps, policy)
        self.assertTrue(any(merged for _, _, merged in groups))
        no_merge = select_policy(20, cfg)
        groups_large = merged_step_groups(cfg.denoise_steps, no_merge)
        self.assertFalse(any(merged for _, _, merged in groups_large))


if __name__ == "__main__":
    unittest.main()
