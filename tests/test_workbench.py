from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from acoustic_workbench.analysis import estimate_delays
from acoustic_workbench.audio import decode_wav_24
from acoustic_workbench.calibration import verify_calibration
from acoustic_workbench.deconvolution import recompute_channel_ir
from acoustic_workbench.errors import ErrorCode, WorkbenchError
from acoustic_workbench.gui import FrequencyDiagnosticsWidget, TimeDiagnosticsWidget, render_snapshot_png
from acoustic_workbench.hashing import canonical_json
from acoustic_workbench.models import default_config, validate_config
from acoustic_workbench.pipeline import WorkbenchPipeline
from acoustic_workbench.report import build_report
from acoustic_workbench.simulation import generate_measurement, validate_sync_marks
from acoustic_workbench.snapshot import SnapshotStore


def fast_config():
    base = default_config()
    return replace(
        base,
        measurement=replace(
            base.measurement,
            sweep_duration_s=0.5,
            repeats=2,
            fade_s=0.01,
        ),
    )


class WorkbenchCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="acoustic-workbench-tests-")
        cls.root = Path(cls.temporary.name)
        cls.config = fast_config()
        cls.pipeline = WorkbenchPipeline(cls.root / "cache", cls.root / "current_snapshot.json")
        cls.base_run = cls.pipeline.run(cls.config)
        cls.snapshot = cls.base_run.snapshot

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_wav_format_and_sample_determinism(self) -> None:
        first = generate_measurement(self.config.measurement)
        second = generate_measurement(self.config.measurement)
        self.assertEqual(first.wav_bytes, second.wav_bytes)
        self.assertEqual(first.wav_hash, second.wav_hash)
        decoded, fmt = decode_wav_24(
            first.wav_bytes,
            expected_sample_rate=48_000,
            expected_channels=4,
            expected_bit_depth=24,
        )
        self.assertEqual(fmt.sample_rate, 48_000)
        self.assertEqual(fmt.channels, 4)
        self.assertEqual(fmt.bit_depth, 24)
        self.assertEqual(decoded.shape, first.recorded.shape)
        detected = validate_sync_marks(first)
        self.assertEqual(len(detected), self.config.measurement.repeats)
        self.assertTrue(first.sync_markers)

    def test_delay_truth_estimate_and_bound(self) -> None:
        delay_data = self.snapshot.data["delay"]
        for item in delay_data["channels"]:
            self.assertEqual(item["status"], "OK")
            self.assertLessEqual(abs(item["bias_samples"]), item["error_bound_samples"] + 1.0e-12)
            self.assertLessEqual(abs(item["truth_relative_samples"]), self.config.analysis.delay_tolerance_samples)
        self.assertEqual(delay_data["channels"][0]["estimated_relative_samples"], 0.0)

    def test_gain_truth_interval_and_band_narrowing(self) -> None:
        initial = self.snapshot.data["mismatch"]["channels"]
        initial_widths = [item["interval_high_db"] - item["interval_low_db"] for item in initial]
        for item in initial:
            self.assertTrue(item["truth_within_interval"])
        narrow_config = self.config.with_analysis(band_low_hz=300.0, band_high_hz=8_000.0)
        narrow = self.pipeline.run(narrow_config)
        narrow_widths = [
            item["interval_high_db"] - item["interval_low_db"]
            for item in narrow.snapshot.data["mismatch"]["channels"]
        ]
        self.assertLess(max(narrow_widths), max(initial_widths))
        self.assertNotEqual(initial_widths, narrow_widths)
        self.assertGreater(min(max(0.0, value) for value in initial_widths), 0.0)

    def test_calibration_residual_is_compatible(self) -> None:
        verification = self.snapshot.data["verification"]
        self.assertEqual(verification["status"], "OK")
        for item in verification["channels"]:
            self.assertTrue(item["compatible"])
            self.assertLessEqual(item["max_amplitude_excess_db"], 0.0)
            self.assertLessEqual(item["max_phase_excess_deg"], 0.0)

    def test_snapshot_contains_recomputable_ir_metadata(self) -> None:
        channels = self.snapshot.data["deconvolution"]["channels"]
        self.assertEqual(len(channels), 4)
        for channel in channels:
            self.assertGreater(channel["nfft"], channel["truncation_samples"])
            self.assertEqual(channel["window"], "tukey")
            self.assertTrue(channel["window_starts"])
            self.assertIn("ir_hash", channel)
        recomputed = recompute_channel_ir(self.base_run.measurement, self.config, 1)
        self.assertEqual(recomputed.ir_hash, self.base_run.deconvolution.channels[1].ir_hash)

    def test_report_and_snapshot_numbers_are_identical(self) -> None:
        report = build_report(self.snapshot)
        stored = report["snapshot"]["frequency_view"]
        for snapshot_channel, report_channel in zip(
            self.snapshot.data["frequency_view"]["channels"], stored["channels"]
        ):
            self.assertEqual(
                list(snapshot_channel["amplitude_ratio_db"]),
                list(report_channel["amplitude_ratio_db"]),
            )
            self.assertEqual(
                list(snapshot_channel["phase_difference_deg"]),
                list(report_channel["phase_difference_deg"]),
            )
        with self.assertRaises(TypeError):
            self.snapshot.data["delay"]["channels"][0]["status"] = "BAD"  # type: ignore[index]

    def test_incremental_reuse_graph(self) -> None:
        with tempfile.TemporaryDirectory(prefix="acoustic-workbench-inc-") as temporary:
            root = Path(temporary)
            pipeline = WorkbenchPipeline(root / "cache", root / "snapshot.json")
            base = pipeline.run(self.config)
            first_reuse = {stage.stage: stage.compute_source for stage in base.stages}
            self.assertTrue(all(source == "computed" for source in first_reuse.values()))

            reg_config = self.config.with_deconvolution(regularization=2.0e-6)
            regularization = pipeline.run(reg_config)
            reuse = {stage.stage: stage.compute_source for stage in regularization.stages}
            self.assertNotEqual(reuse["measurement"], "computed")
            self.assertNotEqual(reuse["sync"], "computed")
            self.assertEqual(reuse["deconvolution"], "computed")

            band_config = reg_config.with_analysis(band_low_hz=300.0, band_high_hz=8_000.0)
            band = pipeline.run(band_config)
            reuse = {stage.stage: stage.compute_source for stage in band.stages}
            self.assertNotEqual(reuse["deconvolution"], "computed")
            self.assertNotEqual(reuse["delay"], "computed")
            self.assertEqual(reuse["mismatch"], "computed")

            filter_config = band_config.with_calibration(filter_length=513)
            filtered = pipeline.run(filter_config)
            reuse = {stage.stage: stage.compute_source for stage in filtered.stages}
            self.assertNotEqual(reuse["mismatch"], "computed")
            self.assertEqual(reuse["calibration"], "computed")
            self.assertEqual(reuse["remeasurement"], "computed")
            self.assertEqual(reuse["verification"], "computed")

    def test_business_error_codes(self) -> None:
        with self.assertRaises(WorkbenchError) as audio_error:
            decode_wav_24(self.base_run.measurement.wav_bytes, expected_sample_rate=44_100)
        self.assertEqual(audio_error.exception.code, ErrorCode.AUDIO_FORMAT)

        broken = replace(self.base_run.measurement, recorded=np.zeros_like(self.base_run.measurement.recorded))
        with self.assertRaises(WorkbenchError) as sync_error:
            validate_sync_marks(broken)
        self.assertEqual(sync_error.exception.code, ErrorCode.SYNC_MARK)

        delay_config = self.config.with_analysis(delay_tolerance_samples=0.10)
        with self.assertRaises(WorkbenchError) as delay_error:
            estimate_delays(self.base_run.measurement, self.base_run.deconvolution, delay_config, raise_on_bias=True)
        self.assertEqual(delay_error.exception.code, ErrorCode.DELAY_BIAS)

        tight = replace(
            self.config,
            calibration=replace(self.config.calibration, residual_tolerance_factor=0.0),
        )
        with self.assertRaises(WorkbenchError) as calibration_error:
            verify_calibration(
                self.base_run.measurement,
                self.base_run.calibrated_measurement,
                self.base_run.deconvolution,
                self.base_run.mismatch,
                self.base_run.delays,
                self.base_run.calibration,
                tight,
                raise_on_mismatch=True,
            )
        self.assertEqual(calibration_error.exception.code, ErrorCode.CALIB_MISMATCH)
        self.assertTrue(calibration_error.exception.details["bands"])

        with self.assertRaises(WorkbenchError) as snapshot_error:
            self.pipeline.snapshot_store.load(expected_hash="0" * 64)
        self.assertEqual(snapshot_error.exception.code, ErrorCode.SNAPSHOT_STALE)

        with self.assertRaises(WorkbenchError) as band_error:
            validate_config(self.config.with_analysis(band_low_hz=1_000.0, band_high_hz=100.0))
        self.assertEqual(band_error.exception.code, ErrorCode.BAND_INVALID)

        with self.assertRaises(WorkbenchError) as filter_error:
            validate_config(self.config.with_calibration(filter_length=32))
        self.assertEqual(filter_error.exception.code, ErrorCode.FILTER_LEN)

    def test_snapshot_tamper_and_cache_tamper_are_stale(self) -> None:
        with tempfile.TemporaryDirectory(prefix="acoustic-workbench-stale-") as temporary:
            root = Path(temporary)
            snapshot_path = root / "snapshot.json"
            cache_dir = root / "cache"
            pipeline = WorkbenchPipeline(cache_dir, snapshot_path)
            pipeline.run(self.config)
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            payload["delay"]["channels"][0]["estimated_relative_samples"] = 0.5
            snapshot_path.write_text(canonical_json(payload), encoding="utf-8")
            with self.assertRaises(WorkbenchError) as stale:
                SnapshotStore(snapshot_path).load()
            self.assertEqual(stale.exception.code, ErrorCode.SNAPSHOT_STALE)

        with tempfile.TemporaryDirectory(prefix="acoustic-workbench-cache-") as temporary:
            root = Path(temporary)
            snapshot_path = root / "snapshot.json"
            cache_dir = root / "cache"
            pipeline = WorkbenchPipeline(cache_dir, snapshot_path)
            pipeline.run(self.config)
            cache_file = sorted(cache_dir.glob("measurement-*.json"))[0]
            envelope = json.loads(cache_file.read_text(encoding="utf-8"))
            envelope["content_hash"] = "0" * 64
            cache_file.write_text(canonical_json(envelope), encoding="utf-8")
            with self.assertRaises(WorkbenchError) as cache_stale:
                WorkbenchPipeline(cache_dir, snapshot_path).run(self.config)
            self.assertEqual(cache_stale.exception.code, ErrorCode.SNAPSHOT_STALE)

    def test_gui_renders_only_the_snapshot_and_is_deterministic(self) -> None:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        frequency = FrequencyDiagnosticsWidget(self.snapshot)
        time_view = TimeDiagnosticsWidget(self.snapshot)
        frequency_numbers = frequency.diagnostic_numbers()
        time_numbers = time_view.diagnostic_numbers()
        self.assertEqual(
            list(frequency_numbers["frequencies_hz"]),
            list(self.snapshot.data["frequency_view"]["frequencies_hz"]),
        )
        self.assertEqual(
            list(frequency_numbers["channels"][0]["amplitude_ratio_db"]),
            list(self.snapshot.data["frequency_view"]["channels"][0]["amplitude_ratio_db"]),
        )
        self.assertEqual(
            list(time_numbers["channels"][0]["sync_marker_relative_samples"]),
            list(self.snapshot.data["time_view"]["channels"][0]["sync_marker_relative_samples"]),
        )
        with tempfile.TemporaryDirectory(prefix="acoustic-workbench-png-") as temporary:
            first = Path(temporary) / "first.png"
            second = Path(temporary) / "second.png"
            render_snapshot_png(self.snapshot, first)
            render_snapshot_png(self.snapshot, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertGreater(first.stat().st_size, 1000)
        frequency.close()
        time_view.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
