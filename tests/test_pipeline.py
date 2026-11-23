"""组件单元测试：切分、特征、量化、归属、聚合、增量、预算、快照与注入。"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from timbre_pipeline import audio as audio_mod
from timbre_pipeline import cache as cache_mod
from timbre_pipeline import quantize as quantize_mod
from timbre_pipeline.attach import _tie_decision, normalized_l1_similarity
from timbre_pipeline.audio import read_wav_strict, scan_wav_dir, segment_audio
from timbre_pipeline.budget import BudgetExecutor, BudgetTask
from timbre_pipeline.config import Config
from timbre_pipeline.errors import (
    E_AUDIO_FORMAT, E_DIM_QUANT_CONFLICT, E_INJECT_NON_FINITE,
    E_INJECT_THRESHOLD_RANGE, E_LATENCY_FALLBACK, E_SEGMENT_OVERFLOW,
)
from timbre_pipeline.features import compute_segment_feature
from timbre_pipeline.inject import InjectionBook
from timbre_pipeline.pipeline import TimbrePipeline
from timbre_pipeline.snapshot import SnapshotStore
from timbre_pipeline.synthesize import generate_sample_workspace
from timbre_pipeline.utils import quantile


class WorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="timbre_test_")
        cls.layout = generate_sample_workspace(os.path.join(cls.tmp, "work"))
        cls.config = Config.load(cls.layout["config_path"])
        cls.truth = json.load(open(cls.layout["label_path"], encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _pipeline(self, injections=None, snapshot_dir=None, budget_ms=None):
        config = self.config
        if budget_ms is not None:
            data = config.data
            data["first_response_budget_ms"] = budget_ms
            config = Config(data)
        book = InjectionBook(injections if injections is not None else [])
        store = SnapshotStore(snapshot_dir) if snapshot_dir else None
        return TimbrePipeline(config, book, store, deterministic=True)

    def _run_clean(self, **kwargs):
        pipeline = self._pipeline(**kwargs)
        report = pipeline.run_full(scan_wav_dir(self.layout["wav_dir"]), self.truth)
        return pipeline, report


class TestAudioSegmentation(WorkspaceTest):
    def test_strict_reader_accepts_good_wav(self):
        audio = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_01.wav"))
        self.assertEqual(audio["sample_rate"], 16000)
        self.assertEqual(audio["samples"].ndim, 1)

    def test_strict_reader_rejects_malformed_wav(self):
        with self.assertRaises(Exception) as context:
            read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_bad.wav"))
        self.assertEqual(context.exception.code, E_AUDIO_FORMAT)

    def test_segmentation_deterministic_and_five_segments(self):
        audio = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_01.wav"))
        first = segment_audio(self.config, audio)
        second = segment_audio(self.config, audio)
        self.assertEqual(len(first["segments"]), 5)
        self.assertEqual(
            [s["content_fingerprint"] for s in first["segments"]],
            [s["content_fingerprint"] for s in second["segments"]])
        for segment in first["segments"]:
            self.assertLess(segment["start_sample"], segment["end_sample"])
            self.assertEqual(segment["frame_count"],
                             segment["frame_end"] - segment["frame_start"] + 1)


class TestFeatures(WorkspaceTest):
    def test_feature_deterministic_and_speaker_separable(self):
        audio = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_01.wav"))
        segmentation = segment_audio(self.config, audio)
        segment = segmentation["segments"][0]
        first = compute_segment_feature(self.config, segment, audio["samples"])
        second = compute_segment_feature(self.config, segment, audio["samples"])
        self.assertEqual(first["feature_fingerprint"], second["feature_fingerprint"])
        self.assertGreater(first["valid_frames"], 0)
        self.assertEqual(len(first["descriptor_inputs"]), len(self.config.dimensions()))
        f0_values = [first["descriptor_inputs"]["f0_median"]]
        audio_gamma = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_03.wav"))
        gamma_seg = segment_audio(self.config, audio_gamma)["segments"][0]
        gamma_feature = compute_segment_feature(self.config, gamma_seg, audio_gamma["samples"])
        self.assertNotAlmostEqual(
            f0_values[0], gamma_feature["descriptor_inputs"]["f0_median"], delta=40.0)

    def test_quantile_methods_are_stable_with_ties(self):
        values = [1.0, 1.0, 2.0, 2.0, 3.0]
        self.assertEqual(quantile(values, 0.5, "lower"), 2.0)
        self.assertEqual(quantile(values, 0.5, "higher"), 2.0)
        self.assertAlmostEqual(quantile(values, 0.25, "linear"), 1.0)


class TestQuantization(WorkspaceTest):
    def test_level_switch_changes_integers_or_steps(self):
        audio = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_02.wav"))
        segmentation = segment_audio(self.config, audio)
        features = [compute_segment_feature(self.config, segment, audio["samples"])
                    for segment in segmentation["segments"]]
        rows = [feature["descriptor_inputs"] for feature in features]
        schemas = quantize_mod.build_dimension_schemas(self.config, rows)
        probe_feature = features[min(2, len(features) - 1)]
        first = quantize_mod.quantize_with_schema(probe_feature, schemas)
        switched_data = self.config.data
        for dimension in switched_data["dimensions"]:
            dimension["levels"] = [level + 4 for level in dimension["levels"]]
        switched_config = Config(switched_data)
        switched_schemas = quantize_mod.build_dimension_schemas(switched_config, rows)
        second = quantize_mod.quantize_with_schema(probe_feature, switched_schemas)
        self.assertNotEqual(
            [d["selected_level"] for d in first["dimensions"]],
            [d["selected_level"] for d in second["dimensions"]])
        self.assertTrue(
            first["integer_vector"] != second["integer_vector"]
            or [d["step"] for d in first["dimensions"]]
            != [d["step"] for d in second["dimensions"]])

    def test_dimension_conflict_keeps_dimension_list(self):
        audio = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_02.wav"))
        segmentation = segment_audio(self.config, audio)
        feature = compute_segment_feature(self.config, segmentation["segments"][0],
                                          audio["samples"])
        injection = {"id": "inj-x", "type": "dim_mismatch",
                     "drop_dim": self.config.dimension_names()[-1]}
        with self.assertRaises(Exception) as context:
            quantize_mod.quantize_segment(self.config, feature, injection)
        self.assertEqual(context.exception.code, E_DIM_QUANT_CONFLICT)
        self.assertEqual(context.exception.extra["conflict_dims"],
                         [self.config.dimension_names()[-1]])


class TestAttachAndAggregation(WorkspaceTest):
    def test_tie_threshold_rule_and_boundary_flag(self):
        merge, boundary_merge = _tie_decision(0.9, 0.9, "merge")
        split, boundary_split = _tie_decision(0.9, 0.9, "split")
        self.assertTrue(merge and boundary_merge)
        self.assertFalse(split)
        self.assertTrue(boundary_split)

    def test_same_speaker_segments_cluster_without_conflict(self):
        _, report = self._run_clean()
        for audio_id in ("clip_01", "clip_02", "clip_04", "clip_05"):
            attach = report["audio_reports"][audio_id]["attach"]
            self.assertEqual(len(attach["speaker_ids"]), 1)
            self.assertEqual(attach["conflicts"], [])
            aggregate = report["audio_reports"][audio_id]["aggregate"]
            self.assertEqual(len(aggregate["speakers"]), 1)
            self.assertFalse(aggregate["speakers"][0]["degraded_single_segment"])

    def test_boundary_sample_at_exact_threshold(self):
        data = self.config.data
        data["similarity_threshold"] = 1.0
        data["threshold_tie_rule"] = "split"
        config = Config(data)
        pipeline = TimbrePipeline(config, InjectionBook([]), None, deterministic=True)
        report = pipeline.run_full(
            [os.path.join(self.layout["wav_dir"], "clip_04.wav")], self.truth)
        attach = report["audio_reports"]["clip_04"]["attach"]
        self.assertTrue(attach["boundary_segments"])
        self.assertTrue(all(
            trace["boundary"] for trace in attach["merge_trace"]
            if trace["similarity"] == attach["threshold"]))

    def test_overflow_guard(self):
        data = self.config.data
        data["max_consecutive_segments"] = 2
        config = Config(data)
        pipeline = TimbrePipeline(config, InjectionBook([]), None, deterministic=True)
        report = pipeline.run_full(
            [os.path.join(self.layout["wav_dir"], "clip_01.wav")], self.truth)
        codes = [item["code"] for item in report["failure_list"]]
        self.assertIn(E_SEGMENT_OVERFLOW, codes)

    def test_weighted_aggregate_uses_fixed_point_and_weights(self):
        _, report = self._run_clean()
        speaker = report["audio_reports"]["clip_02"]["aggregate"]["speakers"][0]
        self.assertEqual(sum(speaker["weights"].values()), speaker["total_weight"])
        self.assertEqual(len(speaker["per_dimension"]), len(self.config.dimensions()))
        for dimension in speaker["per_dimension"]:
            weighted_sum = sum(item["weight"] * item["integer"]
                               for item in dimension["weights"])
            expected = int(np.floor(weighted_sum / speaker["total_weight"] + 0.5))
            self.assertEqual(dimension["aggregated_integer"], expected)


class TestIncrementalAndSnapshot(WorkspaceTest):
    def _commands(self):
        return [
            {"command_id": "cmd-001", "type": "split", "audio": "clip_01",
             "segment_index": 1, "sample_offset": 1600},
            {"command_id": "cmd-002", "type": "modify_feature", "audio": "clip_02",
             "segment_index": 0, "dimension": "f0_median", "value": 190.0},
            {"command_id": "cmd-003", "type": "merge", "audio": "clip_03",
             "segment_index": 2},
        ]

    def test_incremental_matches_full_replay_field_by_field(self):
        wav_paths = scan_wav_dir(self.layout["wav_dir"])
        incremental = self._pipeline()
        incremental.run_full(wav_paths, self.truth)
        incremental.apply_edit_commands(self._commands(), self.truth)

        replay = self._pipeline()
        replay.run_full(wav_paths, self.truth)
        replay.apply_edit_commands(self._commands(), self.truth)
        self.assertEqual(
            json.dumps(incremental.build_report()["audio_reports"], sort_keys=True),
            json.dumps(replay.build_report()["audio_reports"], sort_keys=True))

        reuse_events = [event for event in incremental.ledger.events
                        if event["action"] == "reuse"]
        recompute_events = [event for event in incremental.ledger.events
                            if event["action"] == "recompute"]
        self.assertTrue(reuse_events)
        self.assertTrue(recompute_events)
        seq = [event["seq"] for event in incremental.ledger.events]
        self.assertEqual(seq, sorted(seq))

    def test_only_affected_audio_recomputes(self):
        wav_paths = scan_wav_dir(self.layout["wav_dir"])
        pipeline = self._pipeline()
        pipeline.run_full(wav_paths, self.truth)
        before = pipeline.build_report()["audio_reports"]["clip_03"]
        pipeline.apply_edit_commands([
            {"command_id": "cmd-only", "type": "modify_feature",
             "audio": "clip_02", "segment_index": 0,
             "dimension": "f0_median", "value": 180.0}
        ], self.truth)
        after = pipeline.build_report()["audio_reports"]["clip_03"]
        self.assertEqual(
            before["aggregate"]["aggregate_fingerprint"],
            after["aggregate"]["aggregate_fingerprint"])

    def test_snapshot_restore_resumes_and_matches(self):
        snapshot_dir = os.path.join(self.tmp, "snapshot")
        wav_paths = scan_wav_dir(self.layout["wav_dir"])
        first = self._pipeline(snapshot_dir=snapshot_dir)
        first.run_full(wav_paths, self.truth)
        revived = self._pipeline(snapshot_dir=snapshot_dir)
        decision = revived.restore_from_snapshot()
        self.assertEqual(decision["action"], "resume")
        first_reports = first.build_report()["audio_reports"]
        revived_reports = revived.build_report()["audio_reports"]
        self.assertEqual(set(first_reports), set(revived_reports))
        for audio_id in first_reports:
            self.assertEqual(
                [item["code"] for item in first_reports[audio_id]["failures"]],
                [item["code"] for item in revived_reports[audio_id]["failures"]])
            self.assertEqual(
                first_reports[audio_id]["segments"],
                revived_reports[audio_id]["segments"])
            self.assertEqual(
                first_reports[audio_id]["descriptors"],
                revived_reports[audio_id]["descriptors"])
            self.assertEqual(
                first_reports[audio_id]["aggregate"],
                revived_reports[audio_id]["aggregate"])

    def test_config_hash_change_forces_cold_start(self):
        snapshot_dir = os.path.join(self.tmp, "snapshot_cold")
        wav_paths = scan_wav_dir(self.layout["wav_dir"])
        first = self._pipeline(snapshot_dir=snapshot_dir)
        first.run_full(wav_paths, self.truth)
        changed_data = self.config.data
        changed_data["similarity_threshold"] = 0.66
        changed_config = Config(changed_data)
        revived = TimbrePipeline(changed_config, InjectionBook([]),
                                 SnapshotStore(snapshot_dir), deterministic=True)
        decision = revived.restore_from_snapshot()
        self.assertEqual(decision["action"], "cold_start")
        self.assertEqual(decision["reason"], "config_hash_changed")

    def test_budget_fallback_keeps_pending_and_code(self):
        executor = BudgetExecutor(max_workers=2, budget_ms=0.0)
        tasks = [BudgetTask("clip_01", index, "feature", lambda index=index: index)
                 for index in range(4)]
        results, pending, fallback = executor.run_sequential_under_budget(tasks, {})
        self.assertTrue(pending)
        self.assertEqual(fallback["code"], E_LATENCY_FALLBACK)
        self.assertIn("pending_sort_keys", fallback)
        self.assertEqual([task.sort_key for task in tasks],
                         sorted(task.sort_key for task in tasks))


class TestInjectionIsolation(WorkspaceTest):
    def test_all_four_injection_families_are_recorded(self):
        injections = json.load(open(self.layout["injection_path"], encoding="utf-8"))
        pipeline = self._pipeline(injections=injections)
        report = pipeline.run_full(scan_wav_dir(self.layout["wav_dir"]), self.truth)
        codes = {item["code"] for item in report["failure_list"]}
        self.assertIn("E_INJECT_NON_FINITE", codes)
        self.assertIn(E_DIM_QUANT_CONFLICT, codes)
        self.assertIn(E_INJECT_THRESHOLD_RANGE, codes)
        self.assertIn("E_INJECT_SEGMENT_OUT_OF_RANGE", codes)
        self.assertIn(E_AUDIO_FORMAT, codes)
        hit_ids = {
            hit.get("injection_id")
            for state in pipeline.states.values()
            for hit in state.injection_hits
        }
        self.assertEqual(hit_ids, {"inj-01", "inj-02", "inj-03", "inj-04"})
        # 单个音频故障不阻断其余音频。
        self.assertIsNotNone(report["audio_reports"]["clip_05"]["attach"])

    def test_non_finite_injection_raises_business_error(self):
        audio = read_wav_strict(os.path.join(self.layout["wav_dir"], "clip_01.wav"))
        segmentation = segment_audio(self.config, audio)
        with self.assertRaises(Exception) as context:
            compute_segment_feature(
                self.config, segmentation["segments"][1], audio["samples"],
                {"id": "inj-01", "type": "non_finite", "target_frame": 2})
        self.assertEqual(context.exception.code, E_INJECT_NON_FINITE)
        self.assertEqual(context.exception.extra["injection_id"], "inj-01")


class TestBudgetConcurrency(unittest.TestCase):
    def test_max_workers_is_respected(self):
        import threading
        import time
        active = {"current": 0, "peak": 0}
        lock = threading.Lock()

        def workload():
            with lock:
                active["current"] += 1
                active["peak"] = max(active["peak"], active["current"])
            time.sleep(0.02)
            with lock:
                active["current"] -= 1
            return True

        executor = BudgetExecutor(max_workers=2, budget_ms=5000.0)
        tasks = [BudgetTask("audio", index, "feature", workload) for index in range(6)]
        results, pending, fallback = executor.run(tasks)
        self.assertIsNone(fallback)
        self.assertEqual(len(results), 6)
        self.assertLessEqual(active["peak"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
