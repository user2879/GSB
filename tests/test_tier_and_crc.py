import tempfile
import unittest
from pathlib import Path

from fxrender.app import build_engine
from fxrender.codes import BizCode
from fxrender.config import RenderConfig
from fxrender.loader import inspect_png_container
from fxrender.pipeline import render_all
from fxrender.policy import select_policy, tile_decode_order
from fxrender.samplegen import generate_sample_bundle


class TierTransitionTests(unittest.TestCase):
    def test_policy_table_changes_path_and_ledger(self):
        cfg = RenderConfig()
        small = select_policy(2, cfg)
        medium = select_policy(8, cfg)
        large = select_policy(16, cfg)
        self.assertLess(large.batch_steps, small.batch_steps)
        self.assertEqual(small.decode_order, "topdown")
        self.assertEqual(large.decode_order, "outside-in")
        self.assertNotEqual(tile_decode_order(cfg.tile_rows, small),
                            tile_decode_order(cfg.tile_rows, large))

    def test_render_respects_tier_specific_decode_order(self):
        work = Path(tempfile.mkdtemp())
        cfg = RenderConfig()
        bundle = generate_sample_bundle(work / "small", cfg=cfg, frame_count=2)
        engine, _ = build_engine(bundle, work / "cache", max_workers=1)
        result = render_all(engine)
        self.assertEqual(engine.policy.tier, "small")
        # top-down order must appear in the processed decode keys.
        decode_keys = [tuple(item["key"]) for item in result.processed
                       if item["key"][0] == "decode"]
        tile_rows = [key[3] for key in decode_keys if key[1] == 0]
        self.assertEqual(tile_rows, sorted(tile_rows))
        self.assertEqual(result.consumed_budget,
                         engine.state.frame_count * cfg.denoise_steps
                         * cfg.points_per_step)


class PngCrcTests(unittest.TestCase):
    def test_corrupted_crc_detected(self):
        work = Path(tempfile.mkdtemp())
        cfg = RenderConfig()
        bundle = generate_sample_bundle(work / "bundle", frame_count=1)
        png = bundle / "images" / "cond_000.png"
        raw = bytearray(png.read_bytes())
        # IHDR layout: 8 sig + 4 length + 4 type + 13 data + 4 CRC.
        # Flip the last CRC byte so signature/IHDR parse but checksum fails.
        ihdr_crc_end = 8 + 4 + 4 + 13 + 4
        target = ihdr_crc_end - 1
        raw[target] ^= 0xFF
        corrupted = work / "corrupt.png"
        corrupted.write_bytes(bytes(raw))
        checks = inspect_png_container(corrupted, cfg)
        codes = [item.code for item in checks]
        self.assertIn(BizCode.E_PNG_HEADER, codes)


if __name__ == "__main__":
    unittest.main()
