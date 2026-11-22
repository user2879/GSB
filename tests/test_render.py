from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from stemscope.cache import CacheStore
from stemscope.config import RemixConfig
from stemscope.model import Track
from stemscope.package import parse_package
from stemscope.render import build_segments, render
from stemscope.samplegen import DEFAULT_SEED, generate_samples
from stemscope.state import ClipRuntime, WorkbenchState


class RenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temp.name) / "sample_data"
        generate_samples(cls.root, seed=DEFAULT_SEED)
        cls.config = RemixConfig.from_path(cls.root / "remix_config.json")
        package = parse_package(cls.root / "segments" / "seg_000", "v1", clip_id="seg_000").package
        cls.state = WorkbenchState(
            config=cls.config,
            clips={"seg_000": ClipRuntime.from_package("seg_000", package)},
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_repeat_render_is_byte_identical(self) -> None:
        first = render(self.state, CacheStore())
        second = render(self.state, CacheStore())
        self.assertEqual(first.pcm_bytes, second.pcm_bytes)
        self.assertEqual(first.frame_detail_sha256, second.frame_detail_sha256)
        self.assertEqual(first.segments, second.segments)

    def test_quantization_changes_fixed_integers_and_output(self) -> None:
        low_config = self.config.clone()
        low_config.gain_quantization_steps = 1000
        high_config = self.config.clone()
        high_config.gain_quantization_steps = 2000
        low_state = WorkbenchState(config=low_config, clips=self.state.clips)
        high_state = WorkbenchState(config=high_config, clips=self.state.clips)
        low = render(low_state, CacheStore())
        high = render(high_state, CacheStore())
        self.assertNotEqual(
            low_config.quantized_gain_int("bass"),
            high_config.quantized_gain_int("bass"),
        )
        self.assertNotEqual(low.pcm_sha256, high.pcm_sha256)

    def test_serial_and_parallel_workers_match(self) -> None:
        serial_config = self.config.clone()
        serial_config.workers = 1
        parallel_config = self.config.clone()
        parallel_config.workers = 4
        serial = render(
            WorkbenchState(config=serial_config, clips=self.state.clips),
            CacheStore(),
        )
        parallel = render(
            WorkbenchState(config=parallel_config, clips=self.state.clips),
            CacheStore(),
        )
        self.assertEqual(serial.pcm_bytes, parallel.pcm_bytes)
        self.assertEqual(serial.frame_detail_sha256, parallel.frame_detail_sha256)
        self.assertEqual(serial.segments, parallel.segments)
        self.assertLessEqual(serial.max_observed_workers, 1)
        self.assertLessEqual(parallel.max_observed_workers, 4)

    def test_output_saturates_to_int16(self) -> None:
        pcm = np.array([32767, -32768, 30000, -30000], dtype="<i2")
        track = Track("bass", pcm, 16000, "v1")
        config = RemixConfig(
            sample_rate=16000,
            gain_quantization_steps=1000,
            render_frames=4,
            workers=1,
            interaction_budget_ms=100,
            snapshot_interval_segments=1,
            format_version="v1",
            tracks={
                "bass": type(self.config.tracks["bass"])(gain=4.0, muted=False, start_ms=0, end_ms=1),
            },
        )
        clip = ClipRuntime("seg_x", parse_package(self.root / "segments" / "seg_000", "v1").package, {"bass": track}, 4, 16000)
        state = WorkbenchState(config=config, clips={"seg_x": clip})
        result = render(state, CacheStore())
        self.assertEqual(result.pcm.tolist(), [32767, -32768, 32767, -32768])


if __name__ == "__main__":
    unittest.main()
