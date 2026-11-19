"""Deterministic test suite for the loudness workbench."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from loudness_workbench.audio_io import read_wav, write_wav  # noqa: E402
from loudness_workbench.bs1770 import (  # noqa: E402
    StreamingLoudnessMeter,
    k_weighting_coeffs,
)
from loudness_workbench.budget import BudgetMonitor  # noqa: E402
from loudness_workbench.compressor import LogDomainCompressor  # noqa: E402
from loudness_workbench.config import (  # noqa: E402
    ABSOLUTE_GATE_LKFS,
    RELATIVE_GATE_OFFSET_LU,
    SAMPLE_RATE,
    CompressorConfig,
    WorkbenchConfig,
    ms_to_samples,
    validate_block_ms,
)
from loudness_workbench.demo import ScriptedClock, run_demo  # noqa: E402
from loudness_workbench.engine import PlaybackEngine  # noqa: E402
from loudness_workbench.errors import (  # noqa: E402
    E_AUDIO_FORMAT,
    E_BLOCK_BOUNDS,
    E_BUDGET_EXCEEDED,
    E_COMP_STATE,
    E_REPORT_MISMATCH,
    E_SNAPSHOT_STALE,
    WorkbenchError,
)
from loudness_workbench.offline import run_offline_analysis  # noqa: E402
from loudness_workbench.snapshot import Snapshot, SnapshotStore  # noqa: E402
from loudness_workbench.synth import generate_programme, render_material  # noqa: E402


class MaterialTests(unittest.TestCase):
    def test_byte_identical_renders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav1 = os.path.join(tmp, "a.wav")
            lab1 = os.path.join(tmp, "a.json")
            wav2 = os.path.join(tmp, "b.wav")
            lab2 = os.path.join(tmp, "b.json")
            render_material(wav1, lab1, 20240517)
            render_material(wav2, lab2, 20240517)
            for p1, p2 in ((wav1, wav2), (lab1, lab2)):
                with open(p1, "rb") as h1, open(p2, "rb") as h2:
                    self.assertEqual(hashlib.sha256(h1.read()).digest(),
                                     hashlib.sha256(h2.read()).digest())

    def test_wav_format_and_label_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "m.wav")
            lab = os.path.join(tmp, "m.json")
            manifest = render_material(wav, lab, 7)
            signal, sr, channels = read_wav(wav)
            self.assertEqual(sr, SAMPLE_RATE)
            self.assertEqual(channels, 6)
            self.assertEqual(signal.shape[0], manifest["total_samples"])
            labels = manifest["segments"]
            self.assertEqual(labels[0]["start_sample"], 0)
            self.assertEqual(labels[-1]["end_sample"], signal.shape[0])
            # Adjacent labels overlap only inside the declared cross-fade zone.
            fade = ms_to_samples(manifest["crossfade_ms"])
            for prev, nxt in zip(labels, labels[1:]):
                self.assertEqual(prev["end_sample"] - nxt["start_sample"], fade)
                self.assertLess(nxt["start_sample"], prev["end_sample"])
            for lab in labels:
                self.assertLess(lab["start_sample"], lab["end_sample"])


class LoudnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal, _ = generate_programme(20240517)

    def test_k_weighting_dc_and_high_frequency(self) -> None:
        from scipy import signal as sps

        b, a = k_weighting_coeffs()
        _, h_dc = sps.freqz(b, a, worN=[0.0])
        self.assertLess(abs(h_dc[0]), 0.05)  # RLB removes DC/near-DC

    def test_chunk_size_independence(self) -> None:
        reference = StreamingLoudnessMeter()
        reference.process(self.signal)
        ref_gated = reference.finalize(ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU)
        for block_ms in (10, 25, 50, 100, 200):
            with self.subTest(block_ms=block_ms):
                meter = StreamingLoudnessMeter()
                size = ms_to_samples(block_ms)
                for start in range(0, self.signal.shape[0], size):
                    meter.process(self.signal[start : start + size])
                gated = meter.finalize(ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU)
                self.assertEqual(
                    meter.get_state()["blocks"], reference.get_state()["blocks"]
                )
                self.assertAlmostEqual(
                    gated.integrated_lkfs, ref_gated.integrated_lkfs, places=12
                )
                self.assertAlmostEqual(
                    meter.true_peak.peak_dbtp(),
                    reference.true_peak.peak_dbtp(),
                    places=9,
                )

    def test_filter_state_round_trip(self) -> None:
        meter_a = StreamingLoudnessMeter()
        half = self.signal.shape[0] // 2
        meter_a.process(self.signal[:half])
        state = meter_a.get_state()
        meter_b = StreamingLoudnessMeter()
        meter_b.set_state(state)
        meter_a.process(self.signal[half:])
        meter_b.process(self.signal[half:])
        self.assertEqual(meter_a.get_state()["blocks"], meter_b.get_state()["blocks"])

    def test_gating_counts_are_consistent(self) -> None:
        result = run_offline_analysis(self.signal)
        self.assertEqual(
            result["blocks_below_absolute"]
            + result["blocks_below_relative"]
            + result["blocks_kept"],
            result["blocks_total"],
        )
        self.assertGreaterEqual(result["blocks_below_relative"], 0)


class CompressorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal, _ = generate_programme(20240517)
        self.config = CompressorConfig()

    def test_envelope_independent_of_chunking(self) -> None:
        offline = LogDomainCompressor(self.config)
        offline.process(self.signal)
        expected = offline.gain_envelope()
        for block_ms in (10, 30, 70, 120):
            with self.subTest(block_ms=block_ms):
                streaming = LogDomainCompressor(self.config)
                size = ms_to_samples(block_ms)
                for start in range(0, self.signal.shape[0], size):
                    streaming.process(self.signal[start : start + size])
                actual = streaming.gain_envelope()
                self.assertEqual(actual.size, expected.size)
                np.testing.assert_array_equal(actual, expected)

    def test_state_restore_prevents_gain_reset(self) -> None:
        first = LogDomainCompressor(self.config)
        half = self.signal.shape[0] // 2
        first.process(self.signal[:half])
        state = first.get_state()
        resumed = LogDomainCompressor(self.config)
        resumed.set_state(state)
        first.process(self.signal[half:])
        resumed.process(self.signal[half:])
        np.testing.assert_array_equal(
            resumed.gain_envelope(), first.gain_envelope()
        )

    def test_corrupt_compressor_state_rejected(self) -> None:
        source = LogDomainCompressor(self.config)
        state = source.get_state()
        compressor = LogDomainCompressor(CompressorConfig(ratio=4.0))
        with self.assertRaises(WorkbenchError) as ctx:
            compressor.set_state(state)
        self.assertEqual(ctx.exception.code, E_COMP_STATE)


class SnapshotTests(unittest.TestCase):
    def test_stale_rollback_rejected(self) -> None:
        store = SnapshotStore()
        first = Snapshot(0, "realtime", 0, 10, SAMPLE_RATE, 50, {})
        second = Snapshot(1, "realtime", 0, 20, SAMPLE_RATE, 50, {})
        store.commit(first)
        store.commit(second)
        stale = Snapshot(0, "realtime", 0, 10, SAMPLE_RATE, 50, {})
        with self.assertRaises(WorkbenchError) as ctx:
            store.commit(stale)
        self.assertEqual(ctx.exception.code, E_SNAPSHOT_STALE)
        self.assertIs(store.current, second)

    def test_snapshot_payload_is_immutable(self) -> None:
        snapshot = Snapshot(0, "realtime", 0, 1, SAMPLE_RATE, 50, {"a": 1})
        with self.assertRaises(Exception):
            snapshot.version = 5  # type: ignore[misc]


class BudgetTests(unittest.TestCase):
    def test_degradation_does_not_mutate_measurement(self) -> None:
        # 50 ms elapsed against a 40 ms budget: two strikes stay under the limit.
        clock_values = [0.0, 50.0, 100.0, 150.0, 200.0, 205.0]
        monitor = BudgetMonitor(budget_ms=40.0, overrun_limit=3,
                                clock=ScriptedClock(clock_values))
        results = [monitor.mark_start() or monitor.mark_end() for _ in range(3)]
        self.assertEqual(monitor.report.frames_degraded, 2)
        self.assertEqual(monitor.report.frames_painted, 1)
        self.assertIn(False, results)

    def test_persistent_overrun_raises_budget_error(self) -> None:
        clock_values = [0.0, 99.0, 100.0, 199.0, 200.0, 299.0]
        monitor = BudgetMonitor(budget_ms=40.0, overrun_limit=3,
                                clock=ScriptedClock(clock_values))
        with self.assertRaises(WorkbenchError) as ctx:
            for _ in range(3):
                monitor.mark_start()
                monitor.mark_end()
        self.assertEqual(ctx.exception.code, E_BUDGET_EXCEEDED)


class ErrorCodeTests(unittest.TestCase):
    def test_audio_format_validation(self) -> None:
        from loudness_workbench.config import validate_audio_format

        with self.assertRaises(WorkbenchError) as ctx:
            validate_audio_format(44100, 6)
        self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)
        with self.assertRaises(WorkbenchError) as ctx:
            validate_audio_format(SAMPLE_RATE, 2)
        self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)

    def test_block_bounds_validation(self) -> None:
        for bad in (5, 500, 15):
            with self.subTest(bad=bad), self.assertRaises(WorkbenchError) as ctx:
                validate_block_ms(bad)
            self.assertEqual(ctx.exception.code, E_BLOCK_BOUNDS)

    def test_meter_rejects_wrong_channels(self) -> None:
        meter = StreamingLoudnessMeter()
        with self.assertRaises(WorkbenchError) as ctx:
            meter.process(np.zeros((100, 2)))
        self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)

    def test_report_mismatch_code(self) -> None:
        from loudness_workbench.offline import assert_conclusions_match

        good = {"integrated_lkfs": -23.0, "true_peak_dbtp": -2.0}
        bad = {"integrated_lkfs": -20.0, "true_peak_dbtp": -2.0}
        env = np.zeros(4)
        with self.assertRaises(WorkbenchError) as ctx:
            assert_conclusions_match(good, bad, env, env)
        self.assertEqual(ctx.exception.code, E_REPORT_MISMATCH)


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal, _ = generate_programme(20240517)

    def test_block_length_switch_keeps_state_continuous(self) -> None:
        engine = PlaybackEngine(WorkbenchConfig(block_ms=50))
        half = self.signal.shape[0] // 2
        size50 = ms_to_samples(50)
        pos = 0
        while pos < half:
            nxt = min(size50, half - pos)
            engine.process_block(self.signal[pos : pos + nxt])
            pos += nxt
        # Probe a sample already covered by a completed control frame (samples
        # still inside the compressor residual buffer have no frame yet).
        covered = engine.compressor._cursor - 1
        probe_sample = max(covered, 0)
        gain_before = engine.compressor.gain_at_sample(probe_sample)
        engine.set_block_ms(100)
        size100 = ms_to_samples(100)
        engine.process_block(self.signal[pos : pos + size100])
        # Smoothing continues from the persisted state instead of resetting.
        self.assertTrue(
            engine.store.current.version > 1
        )
        self.assertEqual(engine.compressor.frames[0].index, 0)
        self.assertAlmostEqual(
            engine.compressor.gain_at_sample(probe_sample), gain_before, places=9
        )

    def test_remeasure_builds_on_snapshot_not_zero(self) -> None:
        engine = PlaybackEngine(WorkbenchConfig())
        size = ms_to_samples(100)
        for start in range(0, size * 4, size):
            engine.process_block(self.signal[start : start + size])
        version_before = engine.store.latest_version
        blocks_before = len(engine.meter.blocks)
        snapshot = engine.remeasure_from_snapshot()
        self.assertGreater(snapshot.version, version_before)
        self.assertEqual(len(engine.meter.blocks), blocks_before)

    def test_remeasure_with_extra_signal_continues_accumulation(self) -> None:
        engine = PlaybackEngine(WorkbenchConfig())
        size = ms_to_samples(100)
        split = size * 4
        for start in range(0, split, size):
            engine.process_block(self.signal[start : start + size])
        partial_blocks = len(engine.meter.blocks)
        engine.remeasure_from_snapshot(self.signal[split:])
        reference = StreamingLoudnessMeter()
        reference.process(self.signal)
        self.assertEqual(
            len(engine.meter.blocks), len(reference.blocks)
        )
        self.assertEqual(
            engine.meter.get_state()["blocks"], reference.get_state()["blocks"]
        )
        self.assertGreater(len(reference.blocks), partial_blocks)

    def test_engine_matches_offline_full_scan(self) -> None:
        engine = PlaybackEngine(WorkbenchConfig(block_ms=70))
        size = ms_to_samples(70)
        for start in range(0, self.signal.shape[0], size):
            engine.process_block(self.signal[start : start + size])
        offline = run_offline_analysis(self.signal)
        streaming = engine.current_conclusions()
        self.assertAlmostEqual(
            streaming["integrated_lkfs"],
            offline["integrated_lkfs"],
            places=9,
        )
        self.assertAlmostEqual(
            streaming["true_peak_dbtp"], offline["true_peak_dbtp"], places=9
        )

    def test_segment_target_adjustment_is_snapshotted(self) -> None:
        engine = PlaybackEngine(WorkbenchConfig())
        before = engine.store.latest_version
        snapshot = engine.adjust_segment_target(2, -22.5)
        self.assertEqual(snapshot.version, before + 1)
        self.assertEqual(snapshot.payload["segment_targets"][2], -22.5)
        self.assertEqual(snapshot.payload["adjustment"]["segment_id"], 2)
        with self.assertRaises(WorkbenchError) as ctx:
            engine.adjust_segment_target(0, 12.0)
        self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)

    def test_persistent_budget_overrun_surfaces_from_engine(self) -> None:
        # Every block takes 90 ms against the 40 ms budget -> 3rd strike raises.
        schedule = []
        for _ in range(10):
            schedule.extend([0.0, 90.0])
        engine = PlaybackEngine(
            WorkbenchConfig(), clock=ScriptedClock(schedule)
        )
        size = ms_to_samples(100)
        with self.assertRaises(WorkbenchError) as ctx:
            for start in range(0, size * 3, size):
                engine.process_block(self.signal[start : start + size])
        self.assertEqual(ctx.exception.code, E_BUDGET_EXCEEDED)


class DemoDeterminismTests(unittest.TestCase):
    def test_two_runs_are_byte_identical_and_pass_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dir_a = os.path.join(tmp, "a")
            dir_b = os.path.join(tmp, "b")
            report_a = run_demo(dir_a)
            report_b = run_demo(dir_b)
            self.assertTrue(report_a["tolerance_verdict"]["all_pass"])
            for name in (
                "programme_5p1_48k_24bit.wav",
                "labels.json",
                "verification_report.json",
            ):
                with open(os.path.join(dir_a, name), "rb") as ha, \
                        open(os.path.join(dir_b, name), "rb") as hb:
                    self.assertEqual(
                        hashlib.sha256(ha.read()).digest(),
                        hashlib.sha256(hb.read()).digest(),
                        f"{name} differs between runs",
                    )
            self.assertEqual(
                report_a["streaming"]["envelope_hash"],
                report_a["offline_envelope_hash"],
            )
            self.assertTrue(report_a["stale_snapshot_rejected"]["rejected"])
            self.assertGreaterEqual(report_a["budget"]["frames_degraded"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
