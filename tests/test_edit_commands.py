import tempfile
import unittest
from pathlib import Path

import numpy as np

from fxrender.app import build_engine
from fxrender.pipeline import apply_command, render_all, render_incremental
from fxrender.samplegen import generate_sample_bundle


def digests(engine):
    out = {}
    for entry in engine.cache.entries.values():
        if entry.stage == "frame" and entry.status in ("computed", "reused"):
            payload = engine.cache.load_artifact(entry.key)
            out[entry.frame_id] = payload["pixel_digest"]
    return dict(sorted(out.items()))


class EditCommandTests(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.bundle = generate_sample_bundle(
            self.work / "samples", frame_count=3)

    def test_image_replace_and_delete(self):
        engine, _ = build_engine(self.bundle, self.work / "cache",
                                 max_workers=1)
        render_all(engine)
        before = digests(engine)
        cfg = engine.cfg
        new_pixels = np.zeros((cfg.image_height, cfg.image_width, 3),
                              dtype=np.uint8)
        new_pixels[:, :, 1] = 200
        invalid, _ = render_incremental(engine, {
            "op": "replace_image", "frame_id": 1, "pixels": new_pixels})
        self.assertTrue(invalid)
        after = digests(engine)
        self.assertNotEqual(before[1], after[1])
        self.assertEqual(before[0], after[0])
        invalid, _ = render_incremental(engine,
                                        {"op": "delete_image", "frame_id": 1})
        self.assertTrue(invalid)
        deleted = digests(engine)
        self.assertNotEqual(after[1], deleted[1])

    def test_motion_adjust_is_global(self):
        engine, _ = build_engine(self.bundle, self.work / "cache_m",
                                 max_workers=1)
        render_all(engine)
        before = digests(engine)
        invalid, _ = render_incremental(engine, {
            "op": "adjust_motion", "tile_row": 0, "tile_col": 0,
            "dx": 6, "dy": 6})
        after = digests(engine)
        for frame_id in before:
            self.assertNotEqual(before[frame_id], after[frame_id])

    def test_invalid_command_reports_error_without_state_change(self):
        engine, _ = build_engine(self.bundle, self.work / "cache_bad",
                                 max_workers=1)
        version = engine.state.version
        change = apply_command(engine.state,
                               {"op": "set_denoise_steps", "value": -1})
        self.assertIn("error", change)
        self.assertEqual(engine.state.version, version)
        change = apply_command(engine.state,
                               {"op": "set_quant", "width": 8, "tier": "fine"})
        self.assertIn("error", change)


if __name__ == "__main__":
    unittest.main()
