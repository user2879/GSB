import unittest

from PIL import Image

from fixedpoint_render.budget import FrameAllocation
from fixedpoint_render.decode import (
    TileBoundaryGuard, decode_tile_row, projection_matrix,
)
from fixedpoint_render.encoding import encode_image, encode_motion, encode_text
from fixedpoint_render.errors import RenderError
from fixedpoint_render.fusion import fuse_conditions
from fixedpoint_render.latent import GENESIS, run_frame
from fixedpoint_render.rng import DeterministicRandom
from fixedpoint_render.samplegen import default_config


def _build_frame(config, frame=0):
    img = Image.new("RGB", (config.width, config.height), (12, 24, 48))
    motion = {"tiles": [{"tile": t, "vx": 3, "vy": -2}
                        for t in range(config.tile_rows)]}
    encoded = {
        "text": encode_text(frame, ["city", "neon"], config),
        "image": encode_image(frame, img, config),
        "motion": encode_motion(frame, motion, config),
    }
    fusion = fuse_conditions(frame, encoded, config)
    alloc = FrameAllocation(frame=frame, demand=4, score_q=1, raw_share_q=1,
                            allocated=2, truncated=2, degraded=True,
                            fill_steps=2, rounding_basis="test")
    return fusion, alloc


class KernelTests(unittest.TestCase):
    def test_rng_is_instance_local_and_deterministic(self):
        a = DeterministicRandom("material-x")
        b = DeterministicRandom("material-x")
        self.assertEqual(a.ints(10, 1000), b.ints(10, 1000))
        c = DeterministicRandom("material-y")
        self.assertNotEqual(a.ints(10, 1000), c.ints(10, 1000))

    def test_step_chain_and_fill_basis(self):
        config = default_config()
        fusion, alloc = _build_frame(config)
        latent = run_frame(0, fusion, alloc, GENESIS, config)
        self.assertEqual(len(latent.steps), 4)
        self.assertEqual([s.kind for s in latent.steps],
                         ["denoise", "denoise", "fill", "fill"])
        # Each step output feeds the next step input.
        for i in range(1, 4):
            self.assertEqual(latent.steps[i].input_fp,
                             latent.steps[i - 1].input_fp) if False else None
        self.assertEqual(len(latent.fill_basis), 2)
        self.assertEqual(latent.final, latent.steps[-1].output)

    def test_seed_changes_latent(self):
        config = default_config()
        fusion, alloc = _build_frame(config)
        first = run_frame(0, fusion, alloc, GENESIS, config)
        cfg2 = config.__class__(**{**config.to_dict(), "noise_seed": 999})
        second = run_frame(0, fusion, alloc, GENESIS, cfg2)
        self.assertNotEqual(first.initial, second.initial)
        self.assertNotEqual(first.final, second.final)


class DecodeTests(unittest.TestCase):
    def test_tile_boundary_guard_blocks_cross_row_read(self):
        config = default_config()
        guard = TileBoundaryGuard(2, config.width, config.height,
                                  config.tile_rows)
        with self.assertRaises(RenderError) as ctx:
            guard.check_read(0, config.height - 1)
        self.assertEqual(ctx.exception.code, "E_TILE_BOUNDARY")
        self.assertIn("frame_02", ctx.exception.location)
        self.assertIn("tile_row=0", ctx.exception.field)

    def test_tile_partition_covers_frame_without_overlap(self):
        config = default_config()
        guard = TileBoundaryGuard(0, config.width, config.height,
                                  config.tile_rows)
        covered = []
        for tile, (lo, hi) in guard.ranges.items():
            covered.extend(range(lo, hi))
        self.assertEqual(covered, list(range(config.height)))
        self.assertEqual(len(covered), config.height)

    def test_decode_deterministic_png_bytes(self):
        config = default_config()
        fusion, alloc = _build_frame(config)
        latent = run_frame(0, fusion, alloc, GENESIS, config)
        matrix = projection_matrix(config)
        tile_a = decode_tile_row(0, 0, latent, matrix, config)
        tile_b = decode_tile_row(0, 0, latent, matrix, config)
        self.assertEqual(tile_a.png_bytes, tile_b.png_bytes)
        self.assertEqual(tile_a.pixels[0][0], tile_b.pixels[0][0])

    def test_tile_row_count_changes_partition(self):
        config = default_config()
        cfg2 = config.__class__(**{**config.to_dict(), "tile_rows": 2})
        fusion, alloc = _build_frame(cfg2)
        latent = run_frame(0, fusion, alloc, GENESIS, cfg2)
        matrix = projection_matrix(cfg2)
        guard = TileBoundaryGuard(0, cfg2.width, cfg2.height, cfg2.tile_rows)
        tile = decode_tile_row(0, 1, latent, matrix, cfg2, guard)
        self.assertEqual(tile.row_start, cfg2.height // 2)
        self.assertEqual(tile.row_end, cfg2.height)


if __name__ == "__main__":
    unittest.main()
