from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sync_diag.config import load_config
from sync_diag.detect import detect_and_pair
from sync_diag.errors import SyncError
from sync_diag.estimate import DriftCurve, estimate_drift
from sync_diag.hashing import sha256_file
from sync_diag.manifest import load_manifest
from sync_diag.pipeline import run_analysis_atomic
from sync_diag.resample import compensate_wav
from sync_diag.snapshot import load_current_snapshot, publish_snapshot, render_diagnostic_view
from sync_diag.strategy import choose_strategy
from sync_diag.synth import synthesize_dataset
from sync_diag.verify import verify_alignment


ROOT = Path(__file__).resolve().parents[1]


def tree_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.config_path = ROOT / "configs" / "low_drift.json"
        cls.config = load_config(cls.config_path, duration_s=30.0)
        cls.data_dir = cls.root / "data"
        synthesize_dataset(cls.config, cls.data_dir)
        cls.run_dir = cls.root / "run"
        cls.report = run_analysis_atomic(cls.data_dir, cls.run_dir)
        cls.manifest, cls.manifest_sha = load_manifest(cls.data_dir / "manifest.json", cls.config)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_synthesis_is_byte_deterministic(self) -> None:
        cfg = load_config(ROOT / "configs" / "low_drift.json", duration_s=2.0)
        left = self.root / "det_a"
        right = self.root / "det_b"
        synthesize_dataset(cfg, left)
        synthesize_dataset(cfg, right)
        self.assertEqual(sha256_file(left / "capture.wav"), sha256_file(right / "capture.wav"))
        self.assertEqual(sha256_file(left / "manifest.json"), sha256_file(right / "manifest.json"))
        self.assertEqual(sha256_file(left / "config.resolved.json"), sha256_file(right / "config.resolved.json"))

    def test_02_end_to_end_accuracy_and_artifacts(self) -> None:
        self.assertEqual(self.report["strategy"], "full_exact")
        self.assertTrue(self.report["summary"]["target_met"])
        self.assertLess(self.report["summary"]["after_rmse_samples_max"], 0.75)
        self.assertGreater(self.report["summary"]["before_rmse_samples_mean"], 100.0)
        self.assertTrue(self.report["resample"]["comparison"]["consistent"])
        self.assertTrue(self.report["resample"]["comparison"]["byte_identical"])
        for name in (
            "aligned.wav",
            "drift_report.json",
            "decision.json",
            "residual_report.json",
            "pulse_pairing.json",
            "diagnostic_view.svg",
            "hashes.json",
            "run_report.json",
            "snapshots/current_snapshot.json",
        ):
            self.assertTrue((self.run_dir / name).exists(), name)
        residual = json.loads((self.run_dir / "residual_report.json").read_text(encoding="utf-8"))
        slave = [channel for channel in residual["channels"] if channel["role"] == "slave"]
        self.assertEqual(len(slave), 7)
        for channel in slave:
            self.assertEqual(len(channel["pulses"]), len(self.manifest["reference_pulse_samples"]))
            self.assertEqual(channel["status"], "IMPROVED")

    def test_03_analysis_outputs_are_byte_deterministic(self) -> None:
        second = self.root / "run_repeat"
        run_analysis_atomic(self.data_dir, second)
        self.assertEqual(tree_hashes(self.run_dir), tree_hashes(second))

    def test_04_strategy_scale_selection(self) -> None:
        short = choose_strategy(30.0, 40, self.config)
        self.assertEqual(short["selected"], "full_exact")
        long = choose_strategy(1800.0, 2400, self.config)
        self.assertEqual(long["selected"], "block_accumulation")
        self.assertLessEqual(long["error_bound_samples"], self.config["accuracy_target_samples"])
        self.assertLessEqual(long["resource_model"]["memory_mb"], self.config["strategy"]["memory_budget_mb"])
        tight = copy.deepcopy(self.config)
        tight["strategy"]["memory_budget_mb"] = 16.0
        sparse = choose_strategy(30.0, 40, tight)
        self.assertEqual(sparse["selected"], "sparse_anchor_interpolation")

    def test_05_strategy_budget_errors(self) -> None:
        too_small = copy.deepcopy(self.config)
        too_small["strategy"]["memory_budget_mb"] = 2.0
        with self.assertRaises(SyncError) as caught:
            choose_strategy(30.0, 40, too_small)
        self.assertEqual(caught.exception.code, "E_STRATEGY_BUDGET")
        impossible = copy.deepcopy(self.config)
        impossible["strategy"]["memory_budget_mb"] = 16.0
        impossible["strategy"]["override"] = "full_exact"
        with self.assertRaises(SyncError) as caught:
            choose_strategy(30.0, 40, impossible)
        self.assertEqual(caught.exception.code, "E_STRATEGY_BUDGET")

    def test_06_drift_exceed_is_hard_error(self) -> None:
        cfg = load_config(ROOT / "configs" / "over_tolerance.json", duration_s=10.0)
        data = self.root / "over_data"
        synthesize_dataset(cfg, data)
        manifest, _ = load_manifest(data / "manifest.json", cfg)
        pairs = detect_and_pair(data / "capture.wav", manifest, cfg, frames=manifest["frames"])
        decision = choose_strategy(10.0, len(manifest["reference_pulse_samples"]), cfg)
        with self.assertRaises(SyncError) as caught:
            estimate_drift(pairs, manifest, cfg, decision)
        self.assertEqual(caught.exception.code, "E_DRIFT_EXCEED")
        self.assertIn("estimated_rate_ppm", caught.exception.context)

    def test_07_pulse_missing_error(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["detector"]["peak_threshold"] = 2.0
        with self.assertRaises(SyncError) as caught:
            detect_and_pair(self.data_dir / "capture.wav", self.manifest, cfg, frames=self.manifest["frames"])
        self.assertEqual(caught.exception.code, "E_PULSE_MISSING")

    def test_08_channel_format_error(self) -> None:
        cfg = copy.deepcopy(self.config)
        cfg["channel_count"] = 7
        with self.assertRaises(SyncError) as caught:
            detect_and_pair(self.data_dir / "capture.wav", self.manifest, cfg, frames=self.manifest["frames"])
        self.assertEqual(caught.exception.code, "E_CH_FORMAT")

    def test_09_manifest_hash_error(self) -> None:
        target = self.root / "bad_manifest"
        target.mkdir(exist_ok=True)
        manifest_path = target / "manifest.json"
        shutil.copyfile(self.data_dir / "manifest.json", manifest_path)
        (target / "manifest.json.sha256").write_text("0" * 64 + "\n", encoding="ascii")
        with self.assertRaises(SyncError) as caught:
            load_manifest(manifest_path)
        self.assertEqual(caught.exception.code, "E_MANIFEST_HASH")

    def test_10_resample_phase_error(self) -> None:
        drift = json.loads((self.run_dir / "drift_report.json").read_text(encoding="utf-8"))
        curves = {
            int(channel["channel"]): DriftCurve(
                np.asarray(channel["curve"]["x_samples"], dtype=np.float64),
                np.asarray(channel["curve"]["offset_samples"], dtype=np.float64),
            )
            for channel in drift["channels"]
        }
        with self.assertRaises(SyncError) as caught:
            compensate_wav(
                self.data_dir / "capture.wav",
                self.root / "phase_break.wav",
                self.manifest,
                self.config,
                curves,
                block_frames=48000,
                phase_break_samples=1.0,
                phase_break_at=48000,
            )
        self.assertEqual(caught.exception.code, "E_RESAMPLE_PHASE")

    def test_11_snapshot_stale_and_view_read_only(self) -> None:
        store = self.root / "snapshot_store"
        publish_snapshot(store, 2, {"value": 2})
        with self.assertRaises(SyncError) as caught:
            publish_snapshot(store, 1, {"value": 1})
        self.assertEqual(caught.exception.code, "E_SNAPSHOT_STALE")
        with self.assertRaises(SyncError) as caught:
            publish_snapshot(store, 2, {"value": "different"})
        self.assertEqual(caught.exception.code, "E_SNAPSHOT_STALE")
        same = publish_snapshot(store, 2, {"value": 2})
        self.assertTrue(same["idempotent"])
        store = self.run_dir / "snapshots"
        snapshot, pointer = load_current_snapshot(store)
        snapshot_file = store / pointer["file"]
        before = sha256_file(snapshot_file)
        view_path = self.root / "view.svg"
        render_diagnostic_view(store, view_path)
        after = sha256_file(snapshot_file)
        self.assertEqual(before, after)
        self.assertEqual(sha256_file(view_path), json.loads((self.run_dir / "hashes.json").read_text(encoding="utf-8"))["step_hashes"]["diagnostic_view_sha256"])

    def test_12_no_improvement_is_explicit(self) -> None:
        cfg = self.config
        manifest, _ = load_manifest(self.data_dir / "manifest.json", cfg)
        pairs = detect_and_pair(self.data_dir / "capture.wav", manifest, cfg, frames=manifest["frames"])
        decision = choose_strategy(30.0, len(manifest["reference_pulse_samples"]), cfg)
        estimate = estimate_drift(pairs, manifest, cfg, decision)
        verification = verify_alignment(pairs, pairs, manifest, estimate, cfg)
        self.assertEqual(verification["summary"]["status"], "DEGRADED_OR_NO_IMPROVEMENT")
        self.assertFalse(verification["summary"]["all_channels_improved"])

    def test_13_block_and_sparse_strategies_remain_accurate(self) -> None:
        pairs = detect_and_pair(self.data_dir / "capture.wav", self.manifest, self.config, frames=self.manifest["frames"])
        cases = (("block_accumulation", 128.0), ("sparse_anchor_interpolation", 16.0))
        for override, memory_mb in cases:
            cfg = copy.deepcopy(self.config)
            cfg["strategy"]["override"] = override
            cfg["strategy"]["memory_budget_mb"] = memory_mb
            decision = choose_strategy(30.0, len(self.manifest["reference_pulse_samples"]), cfg)
            self.assertEqual(decision["selected"], override)
            estimate = estimate_drift(pairs, self.manifest, cfg, decision)
            aligned = self.root / f"{override}.wav"
            compensate_wav(self.data_dir / "capture.wav", aligned, self.manifest, cfg, estimate["curves_by_channel"])
            aligned_pairs = detect_and_pair(aligned, self.manifest, cfg, frames=self.manifest["analysis_frames"])
            verification = verify_alignment(pairs, aligned_pairs, self.manifest, estimate, cfg)
            self.assertTrue(verification["summary"]["target_met"])
            self.assertLess(verification["summary"]["after_rmse_samples_max"], cfg["accuracy_target_samples"])

    def test_14_streamed_long_mode_matches_retained_alignment(self) -> None:
        streamed = self.root / "run_streamed"
        report = run_analysis_atomic(self.data_dir, streamed, keep_aligned=False)
        self.assertFalse((streamed / "aligned.wav").exists())
        self.assertFalse(report["resample"]["aligned_wav_retained"])
        self.assertEqual(report["aligned_wav_sha256"], self.report["aligned_wav_sha256"])
        self.assertTrue(report["summary"]["target_met"])
        self.assertTrue(report["resample"]["comparison"]["byte_identical"])
        streamed_residual = json.loads((streamed / "residual_report.json").read_text(encoding="utf-8"))
        retained_residual = json.loads((self.run_dir / "residual_report.json").read_text(encoding="utf-8"))
        self.assertLess(
            abs(
                streamed_residual["summary"]["after_rmse_samples_max"]
                - retained_residual["summary"]["after_rmse_samples_max"]
            ),
            1.0e-5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
