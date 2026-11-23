import json
import os
import shutil
import tempfile
import unittest
import wave

import numpy as np

from asr_pipeline import config as C
from asr_pipeline import decoder as D
from asr_pipeline import audio as A
from asr_pipeline import features as F
from asr_pipeline.cli import run_once, parse_args
from asr_pipeline.errors import (
    E_AUDIO_FORMAT, E_CONFIG_INVALID, E_POINT_BUDGET_PENDING,
    E_SHARD_BUDGET_EXHAUSTED, E_SNAPSHOT_STALE, PipelineError,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples")


def make_args(sample_dir, workdir, schedule=None, score=None, injections=None):
    argv = [
        "--sample-dir", sample_dir,
        "--score-config", score or os.path.join(SAMPLE, "score_params.json"),
        "--schedule-config", schedule or os.path.join(SAMPLE, "schedule.json"),
        "--phoneme-map", os.path.join(SAMPLE, "phoneme_map.json"),
        "--workdir", workdir,
        "--report", os.path.join(workdir, "report.json"),
    ]
    if injections:
        argv += ["--injections", injections]
    return parse_args(argv)


def write_config(dst, mutator):
    with open(os.path.join(SAMPLE, "schedule.json")) as handle:
        sch = json.load(handle)
    mutator(sch)
    with open(dst, "w") as handle:
        json.dump(sch, handle, indent=2)
    return dst


def comparable(report):
    report = dict(report)
    for key in ("shard_timings_ms", "mode", "max_concurrency"):
        report.pop(key, None)
    # Cache keys are fingerprints of the exact config; concurrency/batch are
    # part of the config hash, so compare the reuse actions instead of keys.
    report["reuse_recompute_log"] = [
        {k: v for k, v in entry.items() if k != "cache_key"}
        for entry in report["reuse_recompute_log"]
    ]
    return report


class DecoderBudgetTests(unittest.TestCase):
    def setUp(self):
        self.sp = C.load_score_params(os.path.join(SAMPLE, "score_params.json"))
        self.sch = C.load_schedule_config(os.path.join(SAMPLE, "schedule.json"))
        item = A.validate_wav(os.path.join(SAMPLE, "wav", "utter_001.wav"))
        span = A.plan_shards(
            item["n_samples"], self.sch["shard_duration_ms"],
            self.sch["overlap_ms"], self.sp["frame_shift_ms"],
        )[0]
        samples = item["samples"][span["sample_start"]:span["sample_end"]]
        self.feat = F.compute_emissions(samples, self.sp, self.sch)

    def test_resume_matches_single_run(self):
        emissions, costs = self.feat["emissions"], self.feat["point_costs"]
        full, _full_inc, full_consumed, resume_full = D.run_viterbi(emissions, costs, self.sp, 10 ** 9)
        self.assertIsNone(resume_full)
        # Split the same decode across several small allowances via checkpoints.
        checkpoint = None
        stitched_consumed = 0
        result = None
        allowance = costs[0] * 7
        while True:
            result, incremental, cumulative, resume = D.run_viterbi(
                emissions, costs, self.sp, allowance, checkpoint=checkpoint
            )
            stitched_consumed += incremental
            if resume is None:
                break
            checkpoint = resume
        self.assertEqual(stitched_consumed, full_consumed)
        self.assertEqual(result["words"], full["words"])
        self.assertEqual(result["path"], full["path"])

    def test_resume_from_empty_checkpoint_matches_single_run(self):
        # First call gets an allowance too small for even one frame, producing
        # an empty checkpoint; the resume must still equal a full run.
        emissions, costs = self.feat["emissions"], self.feat["point_costs"]
        full, _, _, _ = D.run_viterbi(emissions, costs, self.sp, 10 ** 9)
        _decoded, _inc, _cum, checkpoint = D.run_viterbi(
            emissions, costs, self.sp, 0
        )
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint["next_frame"], 0)
        result = None
        guard = 0
        while True:
            result, _inc, _cum, resume = D.run_viterbi(
                emissions, costs, self.sp, costs[0] * 11, checkpoint=checkpoint
            )
            guard += 1
            if resume is None:
                break
            checkpoint = resume
            self.assertLess(guard, 50)
        self.assertEqual(result["words"], full["words"])
        self.assertEqual(result["path"], full["path"])

    def test_quota_below_min_points_raises(self):
        with self.assertRaises(PipelineError) as ctx:
            D.run_viterbi(
                self.feat["emissions"], self.feat["point_costs"], self.sp, 1,
                injection={"type": "quota_below_min_points", "id": "X"},
            )
        self.assertEqual(ctx.exception.code, E_SHARD_BUDGET_EXHAUSTED)


class ParallelSerialConsistencyTests(unittest.TestCase):
    def test_matches_serial_across_concurrency_and_batches(self):
        work_root = tempfile.mkdtemp(prefix="asr_e2e_")
        wav_dir = os.path.join(work_root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        _, serial_report, _, _ = run_once(make_args(wav_dir, os.path.join(work_root, "serial"),
                                                     ))
        baseline = comparable(serial_report)
        for conc, batch in [(1, 2), (8, 5), (16, 7), (2, 1)]:
            cfg_path = write_config(
                os.path.join(work_root, f"cfg_{conc}_{batch}.json"),
                lambda d, c=conc, b=batch: d.update(max_concurrency=c, shards_per_batch=b),
            )
            _, report, _, _ = run_once(
                make_args(wav_dir, os.path.join(work_root, f"run_{conc}_{batch}"), cfg_path),
            )
            body = comparable(report)
            body.pop("config_hash", None)
            base = dict(baseline)
            base.pop("config_hash", None)
            self.assertEqual(body["transcripts"], base["transcripts"])
            self.assertEqual(body["budget_ledger"], base["budget_ledger"])
            self.assertEqual(body["pending_queue"], base["pending_queue"])
            self.assertEqual(body["failures"], base["failures"])
            self.assertEqual(body["reuse_recompute_log"], base["reuse_recompute_log"])

    def test_jitter_does_not_change_results(self):
        work_root = tempfile.mkdtemp(prefix="asr_jit_")
        wav_dir = os.path.join(work_root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        args = make_args(wav_dir, os.path.join(work_root, "w"))
        args.jitter_ms = 1.0
        _, jittered, _, _ = run_once(args)
        _, clean, _, _ = run_once(make_args(wav_dir, os.path.join(work_root, "c")))
        self.assertEqual(comparable(jittered)["transcripts"], comparable(clean)["transcripts"])
        self.assertEqual(comparable(jittered)["budget_ledger"], comparable(clean)["budget_ledger"])


class GlobalBudgetResumeTests(unittest.TestCase):
    def test_pending_then_resume_equals_full_run(self):
        work_root = tempfile.mkdtemp(prefix="asr_budget_")
        wav_dir = os.path.join(work_root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        tight = write_config(
            os.path.join(work_root, "tight.json"),
            lambda d: d.update(global_point_budget=2500, shard_point_quota=700),
        )
        workdir = os.path.join(work_root, "work")
        first_args = make_args(wav_dir, workdir, tight)
        _, first, _, _ = run_once(first_args)
        self.assertTrue(all(x["remaining_after"] >= 0 for x in first["budget_ledger"]))
        self.assertTrue(first["shards_pending"] > 0)
        self.assertIn(E_POINT_BUDGET_PENDING,
                      [item["code"] for item in first["pending_queue"]])
        # Resume repeatedly with fresh budgets until complete.
        report = first
        runs = 0
        while report["pending_queue"] and runs < 8:
            _, report, _, _ = run_once(make_args(wav_dir, workdir, tight))
            runs += 1
        self.assertEqual(report["shards_pending"], 0)
        _, full, _, _ = run_once(
            make_args(wav_dir, os.path.join(work_root, "full"))
        )
        self.assertEqual(report["transcripts"], full["transcripts"])


class IncrementalInvalidationTests(unittest.TestCase):
    def _setup(self, root):
        wav_dir = os.path.join(root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        workdir = os.path.join(root, "work")
        run_once(make_args(wav_dir, workdir))
        return wav_dir, workdir

    def test_replace_one_audio_recomputes_only_that_audio(self):
        root = tempfile.mkdtemp(prefix="asr_inc_")
        wav_dir, workdir = self._setup(root)
        target = os.path.join(wav_dir, "utter_002.wav")
        with wave.open(target, "rb") as wav:
            params = wav.getparams()
            data = wav.readframes(wav.getnframes())
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64)
        rng = np.random.default_rng(99)
        mid = samples.shape[0] // 2
        samples[mid:mid + 800] += rng.normal(0, 4000, 800)
        with wave.open(target, "wb") as wav:
            wav.setparams(params)
            wav.writeframes(np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes())
        _, report, _, _ = run_once(make_args(wav_dir, workdir))
        computed = [x for x in report["reuse_recompute_log"] if x["action"] == "computed"]
        self.assertTrue(computed)
        self.assertTrue(all(x["audio"] == "utter_002.wav" for x in computed))
        stages = {x["stage"] for x in computed}
        self.assertEqual(stages, {"audio", "emission", "decode", "stitch"})

    def test_backoff_change_reuses_emissions(self):
        root = tempfile.mkdtemp(prefix="asr_bo_")
        wav_dir, workdir = self._setup(root)
        with open(os.path.join(SAMPLE, "score_params.json")) as handle:
            sp = json.load(handle)
        sp["backoff_weight"] = -1.5
        score_path = os.path.join(root, "score.json")
        with open(score_path, "w") as handle:
            json.dump(sp, handle)
        _, report, _, _ = run_once(make_args(wav_dir, workdir, score=score_path))
        actions = {(x["stage"], x["action"]) for x in report["reuse_recompute_log"]}
        self.assertIn(("emission", "reused"), actions)
        self.assertIn(("decode", "computed"), actions)
        self.assertIn(("audio", "reused"), actions)

    def test_quant_change_invalidates_emissions(self):
        root = tempfile.mkdtemp(prefix="asr_q_")
        wav_dir, workdir = self._setup(root)
        cfg_path = write_config(
            os.path.join(root, "q.json"), lambda d: d.update(quant_mode="q16")
        )
        _, report, _, _ = run_once(make_args(wav_dir, workdir, cfg_path))
        actions = {(x["stage"], x["action"]) for x in report["reuse_recompute_log"]}
        self.assertIn(("emission", "computed"), actions)
        self.assertIn(("audio", "reused"), actions)
        self.assertTrue(any(e["code"] == E_SNAPSHOT_STALE for e in report["snapshot_events"]))


class InjectionIsolationTests(unittest.TestCase):
    def test_all_injection_types_map_to_business_codes(self):
        root = tempfile.mkdtemp(prefix="asr_inj_")
        wav_dir = os.path.join(root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        injections = os.path.join(SAMPLE, "injections.example.json")
        _, report, _, _ = run_once(
            make_args(wav_dir, os.path.join(root, "work"), injections=injections)
        )
        codes = {h["injection_id"]: h["business_code"] for h in report["injections_hit"]}
        self.assertEqual(codes["INJ-01"], E_AUDIO_FORMAT)
        self.assertEqual(codes["INJ-02"], E_AUDIO_FORMAT)
        self.assertEqual(codes["INJ-03"], E_CONFIG_INVALID)
        self.assertEqual(codes["INJ-04"], E_SHARD_BUDGET_EXHAUSTED)
        self.assertTrue(all(h["hit"] for h in report["injections_hit"]))
        self.assertNotIn("utter_004.wav", report["transcripts"])
        self.assertIn("utter_001.wav", report["transcripts"])
        failed = {(f["audio"], f["shard_index"]) for f in report["failures"]}
        self.assertIn(("utter_004.wav", 0), failed)
        self.assertIn(("utter_004.wav", 3), failed)


if __name__ == "__main__":
    unittest.main()


class SnapshotAndAudioFormatTests(unittest.TestCase):
    def test_stale_snapshot_version_emits_event_and_rebuilds(self):
        root = tempfile.mkdtemp(prefix="asr_stale_")
        wav_dir = os.path.join(root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        workdir = os.path.join(root, "work")
        run_once(make_args(wav_dir, workdir))
        manifest_path = os.path.join(workdir, "snapshot", "manifest.json")
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        manifest["version"] = 999
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle)
        _, report, _, _ = run_once(make_args(wav_dir, workdir))
        self.assertTrue(
            any(event["code"] == E_SNAPSHOT_STALE for event in report["snapshot_events"])
        )
        self.assertEqual(report["shards_completed"], report["shards_total"])

    def test_corrupt_wav_is_flagged_and_others_continue(self):
        from asr_pipeline import audio as audio_mod
        from asr_pipeline.errors import PipelineError
        root = tempfile.mkdtemp(prefix="asr_corrupt_")
        wav_dir = os.path.join(root, "wav")
        shutil.copytree(os.path.join(SAMPLE, "wav"), wav_dir)
        bad = os.path.join(wav_dir, "broken.wav")
        with open(bad, "wb") as handle:
            handle.write(b"RIFF\x00\x00\x00\x00WAVEjunk")
        with self.assertRaises(PipelineError) as ctx:
            audio_mod.validate_wav(bad)
        self.assertEqual(ctx.exception.code, E_AUDIO_FORMAT)
        # The remaining valid WAV files are still discoverable and valid.
        valid = [p for p in audio_mod.list_wav_files(wav_dir) if not p.endswith("broken.wav")]
        for path in valid:
            item = audio_mod.validate_wav(path)
            self.assertEqual(item["n_samples"] > 0, True)
