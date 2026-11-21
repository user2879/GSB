import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vocoder.config import apply_edit, load_config
from vocoder.engine import VocoderEngine, stable_task_key
from vocoder.generate_samples import generate
from vocoder.ledger import Ledger
from vocoder.loader import load_inputs
from vocoder.runner import PipelineRunner
from vocoder.session import InteractiveSession
from vocoder.wav_io import read_wav


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vocoder_test_"))
        cls.samples = cls.tmp / "samples"
        generate(cls.samples)
        cls.cfg = load_config(cls.samples / "vocoder_config.yaml")
        cls.utters, cls.failures = load_inputs(cls.samples)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_failure_codes(self):
        codes = {f.utterance_id: f.code for f in self.failures}
        self.assertEqual(codes["utt_badaudio"], "E_AUDIO_FORMAT")
        self.assertEqual(codes["utt_badfeat"], "E_FEATURE_INVALID")
        ids = {u.utterance_id for u in self.utters}
        self.assertNotIn("utt_badfeat", ids)
        self.assertNotIn("utt_badaudio", ids)

    def test_02_float64_matches_reference_sample_exact(self):
        for utter in self.utters:
            result = VocoderEngine(copy.deepcopy(self.cfg), Ledger()).run_utterance(utter)
            ref, _ = read_wav(utter.wav_path)
            np.testing.assert_array_equal(result.wave_float64_pcm, ref)
            self.assertEqual(result.alignment["best_offset_samples"], 0)
            self.assertGreater(result.metrics["normalized_cross_correlation"], 0.9999)

    def test_03_deterministic_frame_repeat(self):
        cfg = copy.deepcopy(self.cfg)
        r1 = VocoderEngine(cfg, Ledger()).run_utterance(self.utters[0])
        r2 = VocoderEngine(copy.deepcopy(cfg), Ledger()).run_utterance(self.utters[0])
        np.testing.assert_array_equal(r1.wave_pcm, r2.wave_pcm)

    def test_04_phase_bits_change_waveform(self):
        fine = VocoderEngine(copy.deepcopy(self.cfg), Ledger()).run_utterance(self.utters[0])
        coarse_cfg = copy.deepcopy(self.cfg)
        coarse_cfg["phase_bits"] = 5
        coarse_cfg["phase_quant_step"] = 4
        coarse = VocoderEngine(coarse_cfg, Ledger()).run_utterance(self.utters[0])
        self.assertFalse(np.array_equal(fine.wave_pcm, coarse.wave_pcm))
        self.assertTrue(coarse.attribution["phase_dominant"])
        self.assertTrue(coarse.attribution["offending_harmonics"])
        self.assertEqual(len(coarse.attribution["offending_frame_range"]), 2)
        self.assertEqual(coarse.attribution["first_dominant_stage"],
                         "phase_quantization")

    def test_05_harmonic_truncation_record(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["harmonic_limit"] = 8
        result = VocoderEngine(cfg, Ledger()).run_utterance(self.utters[0])
        self.assertEqual(result.truncation["dropped_harmonics"], [9, 10, 11, 12])
        self.assertTrue(result.truncation["affected_frames"])

    def test_06_incremental_equals_full_sample_exact(self):
        commands = json.loads((self.samples / "edit_commands.json").read_text())
        edits = [c for c in commands if c["type"] not in ("snapshot", "shutdown")]

        inc_cfg = load_config(self.samples / "vocoder_config.yaml")
        inc = PipelineRunner(self.samples, inc_cfg, self.tmp / "inc")
        inc.reload_inputs()
        inc.run_full()
        for command in edits:
            inc.run_incremental(command)
        inc_results = {
            utter.utterance_id: VocoderEngine(inc_cfg, Ledger()).run_utterance(utter)
            for utter in inc.utterances
        }

        full_cfg = load_config(self.samples / "vocoder_config.yaml")
        for command in edits:
            apply_edit(full_cfg, command)
        full = PipelineRunner(self.samples, full_cfg, self.tmp / "full")
        full.reload_inputs()
        full.run_full()
        full_results = {
            utter.utterance_id: VocoderEngine(full_cfg, Ledger()).run_utterance(utter)
            for utter in full.utterances
        }

        self.assertEqual(inc_cfg, full_cfg)
        self.assertEqual([u.utterance_id for u in inc.utterances],
                         [u.utterance_id for u in full.utterances])
        for utter_id, inc_result in inc_results.items():
            np.testing.assert_array_equal(
                inc_result.wave_pcm, full_results[utter_id].wave_pcm
            )
            np.testing.assert_array_equal(
                inc_result.wave_float64_pcm, full_results[utter_id].wave_float64_pcm
            )
        inc.shutdown()
        full.shutdown()

    def test_07_cache_reuse_and_invalidation_set(self):
        cfg = load_config(self.samples / "vocoder_config.yaml")
        runner = PipelineRunner(self.samples, cfg, self.tmp / "reuse")
        runner.reload_inputs()
        first = runner.run_full()
        self.assertEqual(first["reuse"]["total_reused"], 0)
        second = runner.run_full()
        self.assertEqual(second["reuse"]["total_recomputed"], 0)
        self.assertEqual(second["reuse"]["total_reused"],
                         second["reuse"]["total_lookups"])
        noop = runner.run_incremental({"seq": 1, "type": "phase_bits",
                                       "value": int(cfg["phase_bits"])})
        self.assertEqual(noop["reuse"]["total_recomputed"], 0)
        edited = runner.run_incremental({"seq": 2, "type": "phase_bits", "value": 12})
        self.assertGreater(edited["reuse"]["total_recomputed"], 0)
        restored = runner.run_incremental({"seq": 3, "type": "phase_bits",
                                           "value": int(cfg["phase_bits"])})
        self.assertGreater(restored["reuse"]["total_reused"], 0)
        runner.shutdown()

    def test_08_stable_ordering_under_concurrency(self):
        keys = [stable_task_key(u.utterance_id, f["frame_index"])
                for u in self.utters for f in u.frames]
        self.assertEqual(keys, sorted(keys))
        outputs = []
        for workers in (1, 2, 4):
            cfg = copy.deepcopy(self.cfg)
            cfg["max_workers"] = workers
            runner = PipelineRunner(self.samples, cfg,
                                    self.tmp / f"conc{workers}")
            runner.reload_inputs()
            report = runner.run_full()
            outputs.append([u["metrics"] for u in report["utterances"]])
            runner.shutdown()
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_09_snapshot_recover_and_manifest_versioning(self):
        cfg = load_config(self.samples / "vocoder_config.yaml")
        runner = PipelineRunner(self.samples, cfg, self.tmp / "snap")
        runner.reload_inputs()
        runner.run_full()
        manifest = runner.snapshot_store.recover()
        self.assertIn("metric", manifest["available_stages"])
        self.assertGreater(manifest["version"], 0)
        (runner.snapshot_store.tmp_root / "phase_stray").mkdir()
        recovered = runner.snapshot_store.recover()
        self.assertNotIn("phase_stray", recovered["available_stages"])
        with self.assertRaises(RuntimeError):
            runner.snapshot_store._write_global_manifest(
                0, "x", 24, 1, [], 0, "metric"
            )
        runner.shutdown()

    def test_10_interactive_budget_and_shutdown(self):
        cfg = load_config(self.samples / "vocoder_config.yaml")
        cfg["interactive_budget_ms"] = 1.0
        session = InteractiveSession(self.samples, cfg, self.tmp / "interact")
        commands = json.loads((self.samples / "edit_commands.json").read_text())
        responses = session.run_command_stream(commands)
        codes = {r.get("code") for r in responses}
        self.assertIn("SHUTDOWN", codes)
        shutdown = next(r for r in responses if r.get("code") == "SHUTDOWN")
        self.assertEqual(shutdown["exit_code"], 0)
        self.assertTrue(any(
            r.get("code") in ("OK", "E_INTERACT_BUDGET_PENDING") for r in responses
        ))

    def test_11_affected_stages_per_edit_type(self):
        runner = PipelineRunner(self.samples, self.cfg, self.tmp / "stages")
        runner.reload_inputs()
        cases = {
            "phase_bits": ("phase", "harmonic", "overlap_add", "metric"),
            "phase_quant_step": ("phase", "harmonic", "overlap_add", "metric"),
            "f0_ratio": ("phase", "harmonic", "overlap_add", "metric"),
            "harmonic_limit": ("harmonic", "overlap_add", "metric"),
            "window_type": ("overlap_add", "metric"),
            "energy_rounding": ("overlap_add", "metric"),
        }
        for edit_type, expected in cases.items():
            command = {"seq": 0, "type": edit_type, "value": self.cfg.get(edit_type)}
            self.assertEqual(runner.affected_stages(command), expected)
        self.assertEqual(runner.affected_stages({"type": "snapshot"}), ())
        runner.shutdown()


class InvalidationEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="vocoder_inval_"))
        cls.samples = cls.tmp / "samples"
        generate(cls.samples)
        cls.commands = json.loads((cls.samples / "edit_commands.json").read_text())
        cls.edits = [c for c in cls.commands
                     if c["type"] not in ("snapshot", "shutdown")]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_restored_config_fully_reuses_and_matches_cold_full(self):
        inc_cfg = load_config(self.samples / "vocoder_config.yaml")
        inc = PipelineRunner(self.samples, inc_cfg, self.tmp / "inc")
        inc.reload_inputs()
        inc.run_full()
        for command in self.edits:
            report = inc.run_incremental(command)
        self.assertEqual(report["reuse"]["total_recomputed"], 0)
        self.assertEqual(report["reuse"]["total_reused"],
                         report["reuse"]["total_lookups"])

        full_cfg = load_config(self.samples / "vocoder_config.yaml")
        for command in self.edits:
            apply_edit(full_cfg, command)
        self.assertEqual(inc_cfg, full_cfg)

        full = PipelineRunner(self.samples, full_cfg, self.tmp / "full")
        full.reload_inputs()
        full.run_full()
        for utter in full.utterances:
            inc_wave = VocoderEngine(inc_cfg, Ledger()).run_utterance(utter).wave_pcm
            full_wave = VocoderEngine(full_cfg, Ledger()).run_utterance(utter).wave_pcm
            np.testing.assert_array_equal(inc_wave, full_wave)
        inc.shutdown()
        full.shutdown()

    def test_invalidation_set_matches_cold_recompute_set(self):
        prefix = self.edits[:4]
        warm_cfg = load_config(self.samples / "vocoder_config.yaml")
        warm = PipelineRunner(self.samples, warm_cfg, self.tmp / "warm")
        warm.reload_inputs()
        warm.run_full()
        for command in prefix:
            warm.run_incremental(command)
        warm_keys = set()
        for utter in warm.utterances:
            warm_keys.update(
                VocoderEngine(warm_cfg, Ledger()).run_utterance(utter).recompute_keys
            )

        cold_cfg = load_config(self.samples / "vocoder_config.yaml")
        for command in prefix:
            apply_edit(cold_cfg, command)
        cold = PipelineRunner(self.samples, cold_cfg, self.tmp / "cold")
        cold.reload_inputs()
        cold_report = cold.run_full()
        cold_keys = set(k for u in cold_report["utterances"]
                        for k in u["recomputed_cache_keys"])
        self.assertEqual(warm_cfg, cold_cfg)
        self.assertTrue(cold_keys.issubset(warm_keys))
        warm.shutdown()
        cold.shutdown()


if __name__ == "__main__":
    unittest.main()
