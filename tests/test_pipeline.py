from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from voice_conversion.audio import read_wav
from voice_conversion.config import PipelineConfig
from voice_conversion.dsp import from_fixed, snap_grid, to_fixed
from voice_conversion.editor import Workspace
from voice_conversion.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


class FixedPointTests(unittest.TestCase):
    def test_round_half_away_from_zero(self):
        self.assertEqual(to_fixed(0.5, 1), 1)
        self.assertEqual(to_fixed(-0.5, 1), -1)
        self.assertAlmostEqual(from_fixed(to_fixed(0.123, 12), 12),
                               0.123, delta=2 ** -12)
        self.assertEqual(from_fixed(to_fixed(0.25, 2), 2), 0.25)

    def test_grid_snap_is_deterministic_and_symmetric(self):
        self.assertEqual(snap_grid(17, 4), 16)
        self.assertEqual(snap_grid(-17, 4), -16)
        self.assertEqual(snap_grid(0, 4), 0)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vc-test-"))
        self.config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")
        self.source = SAMPLES / "wav" / "source.wav"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, exchange, version, *, rebuild, workers=None):
        return run_pipeline(
            exchange_path=exchange, version=version,
            source_wav=self.source, config=self.config,
            snapshot_dir=self.tmp / "snap",
            output_wav=self.tmp / "out.wav",
            full_rebuild=rebuild, max_workers=workers)

    def test_repeated_runs_produce_identical_pcm(self):
        exchange = SAMPLES / "exchange" / "segments.v1.txt"
        first = self._run(exchange, "v1", rebuild=True)
        pcm_a, _ = read_wav(self.tmp / "out.wav")
        second = self._run(exchange, "v1", rebuild=True)
        pcm_b, _ = read_wav(self.tmp / "out.wav")
        self.assertTrue(np.array_equal(pcm_a, pcm_b))
        self.assertEqual(first["output_fingerprint"],
                         second["output_fingerprint"])

    def test_serial_and_parallel_identical(self):
        exchange = SAMPLES / "exchange" / "segments.v2.json"
        serial = run_pipeline(
            exchange_path=exchange, version="v2",
            source_wav=self.source, config=self.config,
            snapshot_dir=self.tmp / "serial",
            output_wav=self.tmp / "serial.wav",
            full_rebuild=True, max_workers=1)
        parallel = run_pipeline(
            exchange_path=exchange, version="v2",
            source_wav=self.source, config=self.config,
            snapshot_dir=self.tmp / "parallel",
            output_wav=self.tmp / "parallel.wav",
            full_rebuild=True, max_workers=4)
        pcm_s, _ = read_wav(self.tmp / "serial.wav")
        pcm_p, _ = read_wav(self.tmp / "parallel.wav")
        self.assertTrue(np.array_equal(pcm_s, pcm_p))
        self.assertEqual(serial["assembly_key"], parallel["assembly_key"])
        self.assertEqual(
            [r["cache_keys"] for r in serial["segments"]],
            [r["cache_keys"] for r in parallel["segments"]])
        self.assertEqual(serial["failures"], parallel["failures"])

    def test_incremental_matches_full_rebuild_after_edits(self):
        ws = Workspace(self.tmp / "ws", self.config)
        ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                    SAMPLES / "config" / "acoustic.json", "v1")
        for command in [
            {"op": "switch_version", "version": "v2"},
            {"op": "set_boundary", "segment": 1,
             "start_frame": 63.0, "end_frame": 127.0},
            {"op": "set_intensity", "segment": 0, "intensity": 0.333},
            {"op": "delete_segment", "segment": 4},
            {"op": "append_segment", "segment": 7, "start_frame": 300.0,
             "end_frame": 330.0, "intensity": 0.6, "timbre": "tenor"},
            {"op": "switch_version", "version": "v1"},
            {"op": "set_quant_levels", "levels": 8},
        ]:
            ws.apply(command)
            run_pipeline(
                exchange_path=ws.exchange_path, version=ws.version,
                source_wav=self.source, config=ws.config,
                snapshot_dir=self.tmp / "inc_snap",
                output_wav=self.tmp / "inc.wav",
                full_rebuild=False)
        inc_report = run_pipeline(
            exchange_path=ws.exchange_path, version=ws.version,
            source_wav=self.source, config=ws.config,
            snapshot_dir=self.tmp / "inc_snap",
            output_wav=self.tmp / "inc.wav", full_rebuild=False)
        full_report = run_pipeline(
            exchange_path=ws.exchange_path, version=ws.version,
            source_wav=self.source, config=ws.config,
            snapshot_dir=self.tmp / "full_snap",
            output_wav=self.tmp / "full.wav", full_rebuild=True)
        inc_pcm, _ = read_wav(self.tmp / "inc.wav")
        full_pcm, _ = read_wav(self.tmp / "full.wav")
        self.assertTrue(np.array_equal(inc_pcm, full_pcm))
        self.assertEqual(inc_report["assembly_key"],
                         full_report["assembly_key"])
        self.assertEqual(_strip_runtime(inc_report), _strip_runtime(full_report))

    def test_single_segment_invalidation(self):
        ws = Workspace(self.tmp / "ws", self.config)
        ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                    SAMPLES / "config" / "acoustic.json", "v1")
        common = dict(source_wav=self.source,
                      snapshot_dir=self.tmp / "snap",
                      output_wav=self.tmp / "out.wav")
        before = run_pipeline(exchange_path=ws.exchange_path, version="v1",
                              config=self.config, full_rebuild=True, **common)
        ws.apply({"op": "set_boundary", "segment": 2,
                  "start_frame": 132.0, "end_frame": 199.0})
        after = run_pipeline(exchange_path=ws.exchange_path, version="v1",
                             config=self.config, full_rebuild=False, **common)
        before_keys = {r["segment"]: r["cache_keys"]
                       for r in before["segments"]}
        after_keys = {r["segment"]: r["cache_keys"]
                      for r in after["segments"]}
        for idx in before_keys:
            if idx == 2:
                self.assertNotEqual(before_keys[idx], after_keys[idx])
            else:
                self.assertEqual(before_keys[idx], after_keys[idx])
        self.assertEqual(after["reuse_invalidation"]["recomputed"], [2])

    def test_worker_count_cannot_exceed_config(self):
        exchange = SAMPLES / "exchange" / "segments.v1.txt"
        report = run_pipeline(
            exchange_path=exchange, version="v1",
            source_wav=self.source, config=self.config,
            snapshot_dir=self.tmp / "capped",
            output_wav=self.tmp / "capped.wav",
            full_rebuild=True, max_workers=99)
        self.assertEqual(report["valid_segment_count"], 5)


def _strip_runtime(report):
    report = copy.deepcopy(report)
    for key in ("reused_segments", "recomputed_segments",
                "reuse_invalidation"):
        report.pop(key, None)
    for row in report.get("segments", []):
        row.pop("reused", None)
        row.pop("stage_reuse", None)
    return report


if __name__ == "__main__":
    unittest.main()
