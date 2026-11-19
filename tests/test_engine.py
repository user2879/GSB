from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from audiofp.atomic import atomic_write_json_group
from audiofp.errors import BusinessError
from audiofp.runtime import CancelToken
from audiofp.samplegen import generate_demo
from audiofp.task import load_task
from audiofp.wavio import read_wav, write_pcm16_wav
from audiofp import run_task


ROOT = Path(__file__).resolve().parents[1]


class EngineAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="audiofp-tests-")
        cls.workspace = Path(cls.temp.name)
        cls.task_path = generate_demo(cls.workspace, clean=True)
        cls.task = load_task(cls.task_path)
        cls.report = run_task(cls.task)
        cls.results_bytes = cls.task.results_path.read_bytes()
        cls.report_bytes = cls.task.report_path.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_injected_track_errors_and_partial_report(self) -> None:
        self.assertEqual(self.report["business_code"], "E_PARTIAL")
        self.assertEqual(self.report["status"], "complete_with_track_errors")
        expected = {
            "q_truncated": "E_TRUNC",
            "q_silent": "E_SILENT",
            "q_rate_mismatch": "E_RATE",
            "q_bad_band": "E_BAND",
        }
        actual = {
            pair["query_id"]: pair["business_code"]
            for pair in self.report["pairs"]
            if pair["query_id"] in expected
        }
        self.assertEqual(actual, expected)
        self.assertTrue(all(item["matched"] for item in self.report["injections"]))

    def test_fixed_score_ranking_and_dual_accumulators(self) -> None:
        for pair in self.report["pairs"]:
            if pair["business_code"] != "E_OK":
                continue
            scores = [fragment["score_fixed_units"] for fragment in pair["fragments"]]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for fragment in pair["fragments"]:
                self.assertAlmostEqual(
                    fragment["score"],
                    fragment["score_fixed_units"] / 1000,
                    places=12,
                )
                self.assertIsNotNone(fragment["first_exceed_score_tolerance"])
                trace = fragment["score_trace"]
                self.assertEqual(trace["format"], "parallel_arrays")
                self.assertGreater(len(trace["abs_difference"]), 0)

    def test_speed_noise_localization_and_insertion(self) -> None:
        raw = json.loads(self.task_path.read_text(encoding="utf-8"))
        truth = raw["ground_truth"]["q_speed_noise"]["segments"]
        pair = next(
            pair
            for pair in self.report["pairs"]
            if pair["query_id"] == "q_speed_noise" and pair["reference_id"] == "ref_song"
        )
        self.assertGreaterEqual(len(pair["fragments"]), 2)
        truth_rate = raw["ground_truth"]["q_speed_noise"]["rate"]
        query_rate = 48000
        reference_rate = 44100
        for segment in truth:
            expected = (
                segment["reference_start_sample"] / truth_rate,
                segment["reference_end_sample"] / truth_rate,
                segment["query_start_sample"] / truth_rate,
                segment["query_end_sample"] / truth_rate,
            )
            best_error = float("inf")
            best_fragment = None
            for fragment in pair["fragments"]:
                observed = (
                    fragment["reference_start_sample"] / reference_rate,
                    fragment["reference_end_sample"] / reference_rate,
                    fragment["query_start_sample"] / query_rate,
                    fragment["query_end_sample"] / query_rate,
                )
                error = max(abs(a - b) for a, b in zip(observed, expected))
                if error < best_error:
                    best_error = error
                    best_fragment = fragment
            self.assertIsNotNone(best_fragment)
            self.assertLess(best_error, 0.36, best_fragment)
            self.assertEqual(best_fragment["drift"]["direction"], "decreasing")

    def test_frame_mapping_and_offset_conversion_tolerances(self) -> None:
        for group in ("references", "queries"):
            for track in self.report["tracks"][group]:
                for frame in track.get("frame_map", []):
                    self.assertLessEqual(abs(frame["start_mapping_error_samples"]), 0.5)
                    self.assertLessEqual(abs(frame["end_mapping_error_samples"]), 1.0)
                    self.assertLess(frame["source_start_sample"], frame["source_end_sample"])
        tolerance = self.report["parameters"]["tolerance"]["time_offset_tolerance_samples"]
        for pair in self.report["pairs"]:
            for fragment in pair["fragments"]:
                self.assertLessEqual(fragment["offset_conversion_error_samples"], tolerance)
                self.assertTrue(fragment["alignment"]["within_time_offset_tolerance"])

    def test_spectrum_budget_and_concurrency_metadata(self) -> None:
        resource = self.report["resource_control"]
        self.assertEqual(resource["max_workers"], 3)
        self.assertLessEqual(
            resource["peak_in_flight_spectrum_frames"],
            resource["in_flight_spectrum_capacity_frames"],
        )
        self.assertFalse(resource["budget_exceeded"])

    def test_results_are_byte_deterministic_across_repeated_run(self) -> None:
        second = run_task(self.task)
        self.assertEqual(self.results_bytes, self.task.results_path.read_bytes())
        self.assertEqual(
            [pair["fragments"] for pair in self.report["pairs"]],
            [pair["fragments"] for pair in second["pairs"]],
        )

    def test_cancel_then_resume(self) -> None:
        interrupted = replace(
            self.task,
            state_dir=self.workspace / "state_interrupt",
            report_path=self.workspace / "output_interrupt" / "report.json",
            results_path=self.workspace / "output_interrupt" / "results.json",
        )
        token = CancelToken()
        timer = threading.Timer(0.03, token.cancel, args=("test interruption",))
        timer.start()
        with self.assertRaises(BusinessError) as context:
            run_task(interrupted, cancel_token=token)
        timer.cancel()
        self.assertEqual(context.exception.code, "E_INTERRUPTED")
        self.assertFalse(interrupted.report_path.exists())
        resumed = run_task(interrupted)
        self.assertEqual(resumed["status"], "complete_with_track_errors")
        self.assertTrue(interrupted.report_path.exists())

    def test_cli_sigterm_and_resume(self) -> None:
        raw = json.loads(self.task_path.read_text(encoding="utf-8"))
        raw["state_dir"] = "state_cli"
        raw["outputs"] = {
            "report": "output_cli/report.json",
            "results": "output_cli/results.json",
        }
        cli_task = self.workspace / "task_cli.json"
        cli_task.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        process = subprocess.Popen(
            [sys.executable, "-m", "audiofp", "run", str(cli_task), "--quiet"],
            cwd=self.workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        state_path = self.workspace / "state_cli" / "status.json"
        deadline = time.time() + 10
        while not state_path.exists() and process.poll() is None and time.time() < deadline:
            time.sleep(0.02)
        self.assertIsNone(process.poll(), "CLI exited before the interruption test could signal it")
        self.assertTrue(state_path.exists(), "CLI did not publish its running state")
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 130, (stdout, stderr))
        self.assertFalse((self.workspace / "output_cli" / "report.json").exists())
        state = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(state["status"], "interrupted")
        resumed = subprocess.run(
            [sys.executable, "-m", "audiofp", "run", str(cli_task), "--quiet"],
            cwd=self.workspace,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        self.assertEqual(resumed.returncode, 0, (resumed.stdout, resumed.stderr))
        self.assertTrue((self.workspace / "output_cli" / "report.json").exists())


class ValidationAndAtomicityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="audiofp-unit-")
        self.workspace = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _task_raw(self) -> dict:
        return {
            "allowed_sample_rates": [16000],
            "references": [{"id": "r", "path": "ref.wav"}],
            "queries": [{"id": "q", "path": "q.wav"}],
            "fingerprint": {
                "frame_length": 256,
                "hop_length": 64,
                "base_sample_rate": 16000,
                "bands": [[100, 1000], [1000, 7000]],
                "peak_neighborhood": {"frequency_bins": 2, "time_frames": 1},
                "min_peak_db": -40,
                "fanout": 2,
                "max_pair_delta_frames": 8,
                "peaks_per_band": 1,
            },
            "retrieval": {
                "bucket_capacity": 64,
                "voting_window_frames": 8,
                "min_matches": 1,
                "max_candidates": 3,
            },
            "tolerance": {
                "score_scale": 100,
                "score_tolerance": 1e-9,
                "time_offset_tolerance_samples": 64,
            },
            "concurrency": {"max_workers": 1, "spectrum_buffer_bytes": 65536},
            "outputs": {"report": "out/report.json", "results": "out/results.json"},
        }

    def _write_task(self, raw: dict) -> Path:
        path = self.workspace / "task.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return path

    def test_schema_rate_and_band_errors(self) -> None:
        raw = self._task_raw()
        raw["fingerprint"]["hop_length"] = 300
        with self.assertRaises(BusinessError) as context:
            load_task(self._write_task(raw))
        self.assertEqual(context.exception.code, "E_SCHEMA")
        raw = self._task_raw()
        raw["fingerprint"]["base_sample_rate"] = 22050
        with self.assertRaises(BusinessError) as context:
            load_task(self._write_task(raw))
        self.assertEqual(context.exception.code, "E_RATE")
        raw = self._task_raw()
        raw["fingerprint"]["bands"] = [[100, 1000], [1000, 9000]]
        with self.assertRaises(BusinessError) as context:
            load_task(self._write_task(raw))
        self.assertEqual(context.exception.code, "E_BAND")

    def test_malformed_json_and_no_output(self) -> None:
        path = self.workspace / "bad.json"
        path.write_text("{", encoding="utf-8")
        with self.assertRaises(BusinessError) as context:
            load_task(path)
        self.assertEqual(context.exception.code, "E_JSON")
        self.assertFalse((self.workspace / "out").exists())

    def test_wav_truncated_silent_and_declared_rate_errors(self) -> None:
        rate = 16000
        signal = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
        good = self.workspace / "good.wav"
        write_pcm16_wav(good, signal, rate)
        truncated = self.workspace / "truncated.wav"
        shutil.copyfile(good, truncated)
        with truncated.open("r+b") as handle:
            handle.truncate(truncated.stat().st_size - 100)
        with self.assertRaises(BusinessError) as context:
            read_wav(
                truncated,
                track_id="t",
                role="query",
                allowed_sample_rates=(16000,),
            )
        self.assertEqual(context.exception.code, "E_TRUNC")
        silent = self.workspace / "silent.wav"
        write_pcm16_wav(silent, np.zeros(rate), rate)
        with self.assertRaises(BusinessError) as context:
            read_wav(silent, track_id="s", role="query", allowed_sample_rates=(16000,))
        self.assertEqual(context.exception.code, "E_SILENT")
        with self.assertRaises(BusinessError) as context:
            read_wav(
                good,
                track_id="r",
                role="query",
                allowed_sample_rates=(16000,),
                declared_sample_rate=48000,
            )
        self.assertEqual(context.exception.code, "E_RATE")

    def test_atomic_group_rollback_preserves_old_files(self) -> None:
        report = self.workspace / "report.json"
        results = self.workspace / "results.json"
        report.write_text('{"old":"report"}', encoding="utf-8")
        results.write_text('{"old":"results"}', encoding="utf-8")
        original_replace = os.replace

        def flaky_replace(source: str | bytes | os.PathLike, destination: str | bytes | os.PathLike) -> None:
            if str(destination).endswith("results.json") and str(source).endswith(".tmp"):
                raise OSError("simulated publish failure")
            original_replace(source, destination)

        os.replace = flaky_replace
        try:
            with self.assertRaises(BusinessError) as context:
                atomic_write_json_group({report: {"new": 1}, results: {"new": 2}})
        finally:
            os.replace = original_replace
        self.assertEqual(context.exception.code, "E_WR")
        self.assertEqual(report.read_text(encoding="utf-8"), '{"old":"report"}')
        self.assertEqual(results.read_text(encoding="utf-8"), '{"old":"results"}')


if __name__ == "__main__":
    unittest.main()
