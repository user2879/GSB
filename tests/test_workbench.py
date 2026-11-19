from dataclasses import asdict
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from loudbench.bufferpool import RAW_GRID
from loudbench.demo import run_demo
from loudbench.errors import (
    E_AUDIO_FORMAT,
    E_BLOCK_BOUNDS,
    E_BUDGET_DENIED,
    E_MERGE_ORDER,
    E_PEAK_OVER,
    E_THREAD_RANGE,
    E_WINDOW_INVALID,
)
from loudbench.kweight import (
    _rbj_high_pass,
    _rbj_high_shelf,
    fftconvolve_chunk,
    kweight_impulse,
    to_db_tp,
    true_peak_fir,
    upsample_max,
)
from loudbench.meter import measure_whole_program
from loudbench.report import canonical_json, merge_results
from loudbench.scheduler import SchedulerConfig, run_scheduler
from loudbench.synth import generate_program
from loudbench.tasks import serialize_manifest, split_tasks, validate_window_params
from loudbench.wavio import read_wav24, write_wav24

SMALL_PROGRAM = [
    ("speech", 2.0, -24.0),
    ("music", 2.0, -20.0),
    ("transient", 1.0, -26.0),
]


def _audio():
    audio, _, _ = generate_program(program=SMALL_PROGRAM)
    return audio


def _biquad_direct(x, coef):
    b0, b1, b2, a0, a1, a2 = coef
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(x.shape[0]):
        xn = x[n]
        yn = (b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        y[n] = yn
        x1, x2, y1, y2 = xn, x1, yn, y1
    return y


class TestSynthesis(unittest.TestCase):
    def test_deterministic_and_wav_roundtrip(self):
        a1, seg1, meta1 = generate_program(program=SMALL_PROGRAM)
        a2, seg2, meta2 = generate_program(program=SMALL_PROGRAM)
        np.testing.assert_array_equal(a1, a2)
        self.assertEqual(canonical_json(meta1), canonical_json(meta2))
        self.assertEqual(a1.shape[0], 6)
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = Path(d) / "a.wav", Path(d) / "b.wav"
            write_wav24(str(p1), a1)
            write_wav24(str(p2), a2)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
            back, rate, channels = read_wav24(str(p1))
            self.assertEqual(rate, 48000)
            self.assertEqual(channels, 6)
            # 24bit 量化往返误差有界
            self.assertLess(np.max(np.abs(back - a1)), 1.0 / 2**22)
            for sg in meta1["segments"]:
                self.assertGreater(sg["sample_count"], 0)
                self.assertLess(sg["peak_linear"], 1.0)
                self.assertIn("source", sg)

    def test_illegal_channel_format(self):
        audio = _audio()[:3, :]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.wav"
            write_wav24(str(p), audio)
            with self.assertRaises(Exception) as ctx:
                read_wav24(str(p))
            self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)


class TestTaskSplit(unittest.TestCase):
    def test_manifest_deterministic_and_partition(self):
        audio = _audio()
        t1 = split_tasks(audio.shape[1], 6, 1.0)
        t2 = split_tasks(audio.shape[1], 6, 1.0)
        m1 = serialize_manifest(t1, {"window_seconds": 1.0})
        m2 = serialize_manifest(t2, {"window_seconds": 1.0})
        self.assertEqual(m1, m2)
        self.assertEqual([t.task_key for t in t1], [f"T{i:04d}" for i in range(len(t1))])
        # 连续无重叠、全覆盖
        self.assertEqual(t1[0].start, 0)
        self.assertEqual(t1[-1].end, audio.shape[1])
        for a, b in zip(t1, t1[1:]):
            self.assertEqual(a.end, b.start)

    def test_window_validation_errors(self):
        with self.assertRaises(Exception) as ctx:
            validate_window_params(0.2, 0.0)
        self.assertEqual(ctx.exception.code, E_WINDOW_INVALID)
        with self.assertRaises(Exception) as ctx:
            validate_window_params(2.0, 2.0)
        self.assertEqual(ctx.exception.code, E_WINDOW_INVALID)
        with self.assertRaises(Exception) as ctx:
            validate_window_params(3.0, 1.0)
        self.assertEqual(ctx.exception.code, E_BLOCK_BOUNDS)


class TestMetering(unittest.TestCase):
    def test_kweight_matches_direct_biquad(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(3000)
        shelf = _rbj_high_shelf(48000, 4.0, 1681.7632, 0.707175236)
        hp = _rbj_high_pass(48000, 38.135470876, 0.5003270373)
        direct = _biquad_direct(_biquad_direct(x, shelf), hp)
        impulse = kweight_impulse()
        conv = fftconvolve_chunk(x, impulse)
        # 截断冲激响应在 0.5s 后，前段误差远低于门控量化精度
        self.assertLess(np.max(np.abs(conv[1000:] - direct[1000:])), 1e-6)

    def test_true_peak_sine(self):
        n = 48000
        t = np.arange(n) / 48000
        sine = 0.5 * np.sin(2 * np.pi * 1000 * t)
        fir = true_peak_fir()
        peak, _ = upsample_max(sine, fir)
        self.assertLess(abs(to_db_tp(peak) - (-6.0206)), 0.1)

    def test_split_invariant_and_parallel_equivalence(self):
        audio = _audio()
        reference = measure_whole_program(audio)
        for window in (1.0, 2.0):
            tasks = split_tasks(audio.shape[1], 6, window)
            results = {}
            for workers in (1, 2, 4):
                run = run_scheduler(audio, tasks, SchedulerConfig(max_workers=workers))
                report = merge_results(run.results, reference)
                self.assertTrue(report["reference_comparison"]["all_pass"])
                results[workers] = canonical_json([asdict(r) for r in run.results])
            self.assertEqual(results[1], results[2])
            self.assertEqual(results[2], results[4])
            self.assertEqual(sum(r.block_count for r in run.results), reference["block_count"])


class TestBudgetAndErrors(unittest.TestCase):
    def test_budget_degradation_completeness(self):
        audio = _audio()
        tasks = split_tasks(audio.shape[1], 6, 1.0)
        reference = measure_whole_program(audio)
        chunk = RAW_GRID * 6 * 8

        tight = run_scheduler(audio, tasks, SchedulerConfig(max_workers=4, memory_budget_bytes=chunk * 2))
        self.assertGreater(tight.cache_plan.total_evictions, 0)
        self.assertGreater(tight.cache_plan.total_recomputes, 0)
        tight_report = merge_results(tight.results, reference)
        self.assertTrue(tight_report["reference_comparison"]["all_pass"])
        self.assertEqual(len(tight.results), len(tasks))

        denied = run_scheduler(audio, tasks, SchedulerConfig(max_workers=2, memory_budget_bytes=chunk - 1))
        self.assertTrue(any(e.code == E_BUDGET_DENIED for e in denied.errors))
        self.assertGreater(denied.cache_plan.total_uncached, 0)
        denied_report = merge_results(denied.results, reference)
        self.assertTrue(denied_report["reference_comparison"]["all_pass"])
        # 账本中 RECOMPUTE 与 HIT 语义互斥，不允许把淘汰误记为复用
        for event in denied.cache_plan.ledger:
            self.assertNotIn(event["event"], ("STORE", "HIT")) if event["event"] == "DENY" else None

        serial = run_scheduler(audio, tasks, SchedulerConfig(max_workers=8, task_time_budget_seconds=1e-12))
        self.assertEqual(serial.effective_workers, 1)
        self.assertTrue(any(d["axis"] == "time" for d in serial.degradation))

    def test_thread_and_merge_and_peak_errors(self):
        audio = _audio()
        tasks = split_tasks(audio.shape[1], 6, 1.0)
        with self.assertRaises(Exception) as ctx:
            run_scheduler(audio, tasks, SchedulerConfig(max_workers=0))
        self.assertEqual(ctx.exception.code, E_THREAD_RANGE)
        with self.assertRaises(Exception) as ctx:
            run_scheduler(audio, tasks, SchedulerConfig(max_workers=2, inject_merge_fault=True))
        self.assertEqual(ctx.exception.code, E_MERGE_ORDER)
        hot = audio.copy()
        hot[0, :] = 1.0
        with self.assertRaises(Exception) as ctx:
            run_scheduler(hot, tasks, SchedulerConfig(max_workers=1, raise_on_peak_over=True))
        self.assertEqual(ctx.exception.code, E_PEAK_OVER)


class TestReproducibility(unittest.TestCase):
    def test_demo_two_runs_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            ws1 = Path(d) / "run1"
            ws2 = Path(d) / "run2"
            run_demo(str(ws1))
            first = (ws1 / "hash_manifest.json").read_bytes()
            run_demo(str(ws2))
            second = (ws2 / "hash_manifest.json").read_bytes()
            self.assertEqual(first, second)
            for name in [
                "sample_5ch1_48k24.wav",
                "ground_truth_labels.json",
                "task_manifest.json",
                "reference_whole.json",
                "demo_index.json",
            ]:
                self.assertEqual((ws1 / name).read_bytes(), (ws2 / name).read_bytes())
            for rp in sorted((ws1 / "runs").glob("*.json")):
                self.assertEqual(rp.read_bytes(), (ws2 / "runs" / rp.name).read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
