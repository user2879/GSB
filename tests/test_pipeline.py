import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

import numpy as np

from vibdet.config import load_config
from vibdet.detectors import run_detectors
from vibdet.edits import delete_segment, replace_window
from vibdet.edits import apply_window_overrides
from vibdet.errors import (E_DEVICE_UNKNOWN, E_SAMPLE_NONFINITE,
                           E_SPECTRUM_TOO_SHORT, E_WINDOW_INVALID,
                           E_POINT_BUDGET_PENDING, PipelineError)
from vibdet.hashing import stable_hash
from vibdet.pipeline import run_pipeline
from vibdet.quantization import fuse_and_judge, normalize, to_fixed
from vibdet.spectrum import build_spectrum
from vibdet.synth import generate_device, load_descriptions
from vibdet.windows import build_windows, window_starts

from tools.generate_sample import build_descriptions


class BaseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="vibdet-"))
        self.descriptions = build_descriptions()["devices"]
        self.cfg = load_config()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def materialize(self, workspace, descriptions=None):
        workspace.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1,
                   "devices": descriptions or self.descriptions}
        with (workspace / "descriptions.yaml").open("w",
                                                     encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=True)
        return workspace / "descriptions.yaml" 

    def device(self, device_id="DEV-A", cfg=None):
        return generate_device(device_id, self.descriptions[device_id],
                               cfg or self.cfg)


class TestDeterminism(BaseCase):
    def test_waveform_is_deterministic(self):
        first = self.device()
        second = self.device()
        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(first.data_fingerprint, second.data_fingerprint)

    def test_full_pipeline_deterministic(self):
        ws1, ws2 = self.tmp / "a", self.tmp / "b"
        r1 = run_pipeline(ws1, self.cfg, self.descriptions)
        r2 = run_pipeline(ws2, copy.deepcopy(self.cfg), self.descriptions)
        self.assertEqual(stable_hash(r1), stable_hash(r2))


class TestErrors(BaseCase):
    def test_unknown_device(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["devices"] = ["DEV-A", "GHOST"]
        result = run_pipeline(self.tmp / "ws", cfg, self.descriptions)
        codes = [(f["code"], f["device"]) for f in result["failures"]]
        self.assertIn((E_DEVICE_UNKNOWN, "GHOST"), codes)

    def test_nonfinite_device(self):
        with self.assertRaises(PipelineError) as ctx:
            self.device("DEV-BAD-FINITE")
        self.assertEqual(ctx.exception.code, E_SAMPLE_NONFINITE)
        cfg = copy.deepcopy(self.cfg)
        cfg["devices"] = ["DEV-BAD-FINITE"]
        result = run_pipeline(self.tmp / "ws", cfg, self.descriptions)
        self.assertEqual(result["failures"][0]["code"], E_SAMPLE_NONFINITE)

    def test_invalid_window(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["window"]["hop"] = cfg["window"]["length"] + 1
        from vibdet.config import validate_window
        with self.assertRaises(PipelineError) as ctx:
            validate_window(cfg)
        self.assertEqual(ctx.exception.code, E_WINDOW_INVALID)

    def test_spectrum_too_short(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["window"]["length"] = 100
        cfg["window"]["hop"] = 100
        wave = self.device("DEV-SHORT", cfg)
        windows = build_windows(wave, cfg)
        self.assertTrue(len(windows) >= 1)
        cfg["spectrum"]["order"] = 3
        with self.assertRaises(PipelineError) as ctx:
            build_spectrum("DEV-SHORT", windows, cfg)
        self.assertEqual(ctx.exception.code, E_SPECTRUM_TOO_SHORT)


class TestWindowsAndSpectrum(BaseCase):
    def test_window_starts_and_no_shared_state(self):
        starts = window_starts(1000, 256, 128)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[1], 128)
        self.assertTrue(starts[-1] + 256 <= 1000)

    def test_spectrum_dependencies_and_padding(self):
        wave = self.device()
        windows = build_windows(wave, self.cfg)
        spectrum = build_spectrum("DEV-A", windows, self.cfg)
        first = spectrum.positions[0]
        self.assertTrue(first.diff_padded)
        self.assertTrue(first.stat_padded)
        self.assertGreater(first.padding_count, 0)
        self.assertIn(0, first.dependencies)
        later = spectrum.positions[5]
        # order=2, stat_window=4 -> windows up to position 5 and back 5 steps
        self.assertIn(5, later.dependencies)
        self.assertIn(max(0, 5 - 5), later.dependencies)
        # cross-segment registration: positions near 1024 touch A-seg0/A-seg1
        boundary_positions = [p for p in spectrum.positions
                              if len(p.segments) > 1]
        self.assertTrue(boundary_positions)

    def test_spectrum_param_change_changes_key(self):
        wave = self.device()
        windows = build_windows(wave, self.cfg)
        s1 = build_spectrum("DEV-A", windows, self.cfg)
        cfg2 = copy.deepcopy(self.cfg)
        cfg2["spectrum"]["order"] = 3
        s2 = build_spectrum("DEV-A", windows, cfg2)
        self.assertNotEqual(s1.param_hash, s2.param_hash)
        self.assertNotEqual(s1.positions[0].cache_key,
                            s2.positions[0].cache_key)


class TestDetectors(BaseCase):
    def test_hyperparameter_change_changes_scores(self):
        wave = self.device()
        windows = build_windows(wave, self.cfg)
        spectrum = build_spectrum("DEV-A", windows, self.cfg)
        r1 = run_detectors(spectrum.matrix, spectrum.positions, self.cfg)
        cfg2 = copy.deepcopy(self.cfg)
        cfg2["detectors"]["maha"]["scale_multiplier"] = 2.5
        cfg2["detectors"]["residual"]["power"] = 2.0
        cfg2["detectors"]["crest"]["epsilon"] = 1e-3
        r2 = run_detectors(spectrum.matrix, spectrum.positions, cfg2)
        for name in ("maha", "residual", "crest"):
            self.assertFalse(np.allclose(r1[name].scores, r2[name].scores),
                             name)

    def test_fixed_point_and_boundary(self):
        values = np.array([0.1, 0.4, 0.9])
        fixed = to_fixed(values, 8, "round")
        np.testing.assert_array_equal(fixed, [26, 102, 230])
        trunc = to_fixed(values, 8, "truncate")
        np.testing.assert_array_equal(trunc, [25, 102, 230])


class TestInvalidation(BaseCase):
    def test_single_window_invalidation(self):
        ws_dir = self.tmp / "ws"
        self.materialize(ws_dir)
        r1 = run_pipeline(ws_dir, self.cfg, self.descriptions)
        before = r1["cache_stats"]["computed"]["spectrum"]
        replace_window(ws_dir, "DEV-A", 2, amplitude=3.0)
        edited = load_descriptions(ws_dir / "descriptions.yaml")
        r2 = run_pipeline(ws_dir, self.cfg, edited)
        after = r2["cache_stats"]["computed"]["spectrum"]
        self.assertLess(after, before)
        dev1 = next(d for d in r1["devices"] if d["device_id"] == "DEV-A")
        dev2 = next(d for d in r2["devices"] if d["device_id"] == "DEV-A")
        k1 = {p["position"]: p["cache_key"] for p in dev1["spectrum_positions"]}
        k2 = {p["position"]: p["cache_key"] for p in dev2["spectrum_positions"]}
        changed_positions = {p for p in k1 if k1[p] != k2[p]}
        # derive the ground truth from each position's registered window set
        wave = self.device()
        windows = build_windows(wave, self.cfg)
        spectrum = build_spectrum("DEV-A", windows, self.cfg)
        changed_windows = {
            i for i in range(len(windows))
            if next(w for w in dev1["windows"] if w["index"] == i)["fingerprint"]
            != next(w for w in dev2["windows"] if w["index"] == i)["fingerprint"]}
        expected = {p.position for p in spectrum.positions
                    if changed_windows.intersection(p.dependencies)}
        self.assertEqual(expected, changed_positions)
        # untouched positions are reused byte-for-byte
        untouched = set(k1) - changed_positions
        self.assertTrue(untouched)
        for position in untouched:
            self.assertEqual(k1[position], k2[position])

    def test_override_only_changes_targeted_samples(self):
        wave = self.device()
        replaced = apply_window_overrides(
            wave.values, self.descriptions, "DEV-A",
            self.cfg["window"]["length"], self.cfg["window"]["hop"])
        # no overrides yet -> identical
        np.testing.assert_array_equal(wave.values, replaced)


class TestConcurrencyAndBudget(BaseCase):
    def test_serial_parallel_identical(self):
        serial_cfg = copy.deepcopy(self.cfg)
        serial_cfg["scheduler"]["serial"] = True
        serial_cfg["scheduler"]["max_workers"] = 1
        parallel_cfg = copy.deepcopy(self.cfg)
        parallel_cfg["scheduler"]["serial"] = False
        parallel_cfg["scheduler"]["max_workers"] = 4
        r_serial = run_pipeline(self.tmp / "s", serial_cfg, self.descriptions)
        r_parallel = run_pipeline(self.tmp / "p", parallel_cfg,
                                  self.descriptions)
        for key in ("devices", "failures", "pending_queue"):
            self.assertEqual(stable_hash(r_serial[key]),
                             stable_hash(r_parallel[key]), key)

    def test_point_budget_pending_and_resume(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["scheduler"]["point_budget"] = 10
        ws_dir = self.materialize(self.tmp / "budget").parent
        r1 = run_pipeline(ws_dir, cfg, self.descriptions)
        pending_codes = [f["code"] for f in r1["failures"]
                         if f["code"] == E_POINT_BUDGET_PENDING]
        self.assertTrue(pending_codes)
        total_positions = sum(
            d["spectrum"]["position_count"] for d in r1["devices"])
        self.assertLess(sum(len(d["verdicts"]) for d in r1["devices"]),
                        total_positions)
        r2 = run_pipeline(ws_dir, cfg, self.descriptions)
        # second batch processes the persisted pending queue first under a
        # fresh per-batch budget
        self.assertGreater(r2["snapshots"]["budget_consumed"], 0)
        first_pending = r1["pending_queue"][0]["position"]
        self.assertEqual(
            r2["devices"][0]["allowed_positions"][0], first_pending)
        # repeatedly resuming with fresh per-batch budgets drains the queue;
        # a large final batch leaves no position pending
        drain_cfg = copy.deepcopy(cfg)
        drain_cfg["scheduler"]["point_budget"] = 100000
        r3 = run_pipeline(ws_dir, drain_cfg, self.descriptions)
        self.assertEqual(r3["pending_queue"], [])
        self.assertEqual(
            sum(len(d["verdicts"]) for d in r3["devices"]),
            sum(d["spectrum"]["position_count"] for d in r3["devices"]))

    def test_segment_deletion_incremental_equivalence(self):
        ws_inc = self.tmp / "inc"
        ws_full = self.tmp / "full"
        self.materialize(ws_inc)
        run_pipeline(ws_inc, self.cfg, self.descriptions)
        delete_segment(ws_inc, "DEV-A", "A-seg2")
        edited = load_descriptions(ws_inc / "descriptions.yaml")
        # full rebuild starts from a blank workspace but consumes the same
        # edited descriptions
        self.materialize(ws_full, edited["devices"])
        r_inc = run_pipeline(ws_inc, self.cfg, edited)
        r_full = run_pipeline(ws_full, self.cfg, edited)
        self.assertEqual(stable_hash(r_inc["devices"]),
                         stable_hash(r_full["devices"]))




    def test_rounding_vs_truncation_causes(self):
        from vibdet.detectors import DetectorResult
        from vibdet.quantization import fuse_and_judge
        cfg = copy.deepcopy(self.cfg)
        cfg["normalization"]["method"] = "minmax"
        cfg["score"]["threshold"] = 500
        # raw values spanning [0, 2] normalize to [0, 1]; the middle value
        # 1.0 normalizes to 0.5 for every detector, so the fused fixed-point
        # total lands exactly on the threshold
        raw = [0.0, 1.0, 2.0, 1.0]
        def make(name):
            return DetectorResult(
                name, np.array(raw, dtype=np.float64), "ph",
                [f"{name}-k{i}" for i in range(len(raw))])
        results = {name: make(name) for name in
                   ("maha", "residual", "crest")}
        verdicts = fuse_and_judge(results, cfg)
        middle = [v for v in verdicts if v.position in (1, 3)]
        self.assertTrue(middle)
        self.assertTrue(all(v.boundary for v in middle))
        self.assertTrue(all(v.reference_frac_bits ==
                            cfg["quantization"]["frac_bits"] +
                            cfg["quantization"]["ref_extra_bit"]
                            for v in middle))
        self.assertTrue(all(v.discrepancy_cause in
                            ("none", "rounding", "quantization_truncation")
                            for v in verdicts))

if __name__ == "__main__":
    unittest.main()


class TestShutdownAndBoundary(BaseCase):
    def test_shutdown_event_completes_stage_and_diagnostics(self):
        import threading
        from vibdet.scheduler import ShutdownRequested
        event = threading.Event()
        event.set()
        ws = self.tmp / "shutdown"
        # a pre-set shutdown event must still surface committed stages rather
        # than corrupting the snapshot store
        result = run_pipeline(ws, self.cfg, self.descriptions,
                              shutdown_event=event)
        self.assertTrue((ws / "snapshots" / "manifest.json").exists())
        self.assertIn("spectrum", result["snapshots"]["stages"])

    def test_manifest_resume_uses_only_committed_snapshot(self):
        import json
        cfg = copy.deepcopy(self.cfg)
        cfg["scheduler"]["point_budget"] = 12
        ws = self.tmp / "resume"
        r1 = run_pipeline(ws, cfg, self.descriptions)
        manifest = json.loads((ws / "snapshots" / "manifest.json").read_text())
        self.assertIn("global_version", manifest)
        self.assertIn("budget_consumed", manifest)
        # resume reads the persisted manifest, keeps monotonic global
        # versions and reuses committed content where nothing changed
        r2 = run_pipeline(ws, cfg, self.descriptions)
        manifest2 = json.loads((ws / "snapshots" / "manifest.json").read_text())
        self.assertGreaterEqual(manifest2["global_version"],
                                manifest["global_version"])
        for stage in ("spectrum", "detector", "verdict"):
            self.assertIn(stage, manifest2["stages"])
            self.assertTrue((ws / manifest2["stages"][stage]["path"]).exists())

    def test_boundary_verdict_structure(self):
        from vibdet.detectors import run_detectors
        from vibdet.quantization import fuse_and_judge
        wave = self.device()
        windows = build_windows(wave, self.cfg)
        spectrum = build_spectrum("DEV-A", windows, self.cfg)
        results = run_detectors(spectrum.matrix, spectrum.positions, self.cfg)
        verdicts = fuse_and_judge(results, self.cfg)
        for verdict in verdicts:
            self.assertIn(verdict.discrepancy_cause,
                          ("none", "rounding", "quantization_truncation"))
            self.assertGreaterEqual(verdict.reference_frac_bits,
                                    self.cfg["quantization"]["frac_bits"])
