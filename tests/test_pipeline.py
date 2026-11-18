"""端到端确定性测试：真值命中、增量复用、并发一致、快照恢复、异常码。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from proofread_app.app import ProofreadApp
from proofread_app.config import load_config
from proofread_app.errors import ProofreadError
from proofread_app.loader import load_samples, load_rules, load_scoring_params, validate_rule_hints
from proofread_app.detection import segment_sample, detect_candidates
from proofread_app.scoring import score_all
from proofread_app.pipeline import Pipeline
from proofread_app.cache import ContentCache
from proofread_app.snapshot import SnapshotStore


def build_app(tmp: Path) -> ProofreadApp:
    return ProofreadApp(str(tmp))


def copy_workspace(tmp: Path) -> None:
    import shutil
    for name in ("samples", "resources", "config"):
        shutil.copytree(ROOT / name, tmp / name)


class TruthMatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)
        self.app = build_app(self.tmp)
        self.app.clean_state()
        self.report = self.app.run()

    def test_all_truth_matches(self) -> None:
        truth = {}
        with (self.tmp / "samples" / "sentences.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if obj.get("truth"):
                    truth[obj["sample_id"]] = obj["truth"]
        corrected = {}
        with (self.tmp / "output" / "corrected.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                corrected[obj["sample_id"]] = obj["corrected_text"]
        mismatches = [sid for sid, value in truth.items() if corrected[sid] != value]
        self.assertEqual(mismatches, [])

    def test_report_contains_required_sections(self) -> None:
        for key in ("error_type_distribution", "histograms", "max_deviation_candidate",
                    "boundary_sensitive_samples", "decision_flips",
                    "ranking_contradictions", "cache", "failures", "fingerprints",
                    "snapshot", "precision_audits"):
            self.assertIn(key, self.report)
        self.assertTrue(self.report["boundary_sensitive_samples"])
        self.assertTrue(self.report["decision_flips"])


class IncrementalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)
        self.app = build_app(self.tmp)
        self.app.clean_state()
        self.app.run()

    def test_replace_single_sample_only_recomputes_that_sample(self) -> None:
        target = next(s for s in self.app.samples if s.sample_id == "S005")
        self.app.cache.reset_counters()
        self.app.apply_command({
            "cmd": "replace_sample",
            "sample": {"sample_id": "S005", "text": "一一这是", "context": target.context,
                       "truth": None, "rule_hints": ["R001"]},
        })
        self.app.snapshots.reset()
        report = self.app.run()
        recomputed_samples = {item.split(":", 1)[0]
                              for item in report["cache"]["recomputed"]}
        reused_samples = {item.split(":", 1)[0]
                          for item in report["cache"]["reused"]}
        self.assertIn("S005", recomputed_samples)
        self.assertNotIn("S005", reused_samples)
        # 其他样本必须有缓存复用
        self.assertIn("S007", reused_samples)
        self.assertNotIn("S007", recomputed_samples)

    def test_remove_rule_only_invalidates_referencing_samples(self) -> None:
        rule_id = "W003"
        affected = set(self.app.cache.samples_using_rule(rule_id))
        self.assertTrue(affected)
        self.app.cache.reset_counters()
        self.app.apply_command({"cmd": "remove_confusion_pair", "rule_id": rule_id})
        invalidated = {item.split(":", 1)[0] for item in self.app.invalidated_keys}
        self.assertEqual(invalidated, affected)
        # 受影响样本之外的缓存必须保留
        untouched = next(s for s in self.app.samples
                         if s.sample_id not in affected)
        self.assertTrue(any(
            key[0] == untouched.sample_id
            for key in self.app.cache._entries))


class ConcurrencyParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)

    def _run_with_workers(self, workers: int):
        app = build_app(self.tmp)
        app.config = app.config.with_updates(max_workers=workers)
        app.clean_state()
        report = app.run()
        corrected = (self.tmp / "output" / "corrected.jsonl").read_text(encoding="utf-8")
        diagnostics = json.loads((self.tmp / "output" / "diagnostics.json").read_text(
            encoding="utf-8"))
        return corrected, diagnostics, app

    def test_worker_count_does_not_change_results(self) -> None:
        seq_text, seq_report, _ = self._run_with_workers(1)
        for workers in (2, 4, 8):
            par_text, par_report, _ = self._run_with_workers(workers)
            self.assertEqual(seq_text, par_text)
            self.assertEqual(
                json.dumps(seq_report["precision_audits"], sort_keys=True),
                json.dumps(par_report["precision_audits"], sort_keys=True))
            self.assertEqual(seq_report["error_type_distribution"],
                             par_report["error_type_distribution"])


class SnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)

    def test_resume_guard_rejects_config_mismatch(self) -> None:
        app = build_app(self.tmp)
        app.clean_state()
        app.run()
        app.config = app.config.with_updates(similarity_threshold=0.10)
        with self.assertRaises(ProofreadError) as ctx:
            app.run(resume=True)
        self.assertEqual(ctx.exception.code, "E_CONFIG_MISMATCH")

    def test_recovery_point_after_commit(self) -> None:
        app = build_app(self.tmp)
        app.clean_state()
        app.run()
        recovery = app.snapshots.recovery_point()
        self.assertTrue(recovery["recoverable"])
        self.assertIn("detect", recovery["intact_stages"])
        self.assertTrue(recovery["completed_cursor"])

    def test_snapshot_files_use_versioned_atomic_commits(self) -> None:
        app = build_app(self.tmp)
        app.clean_state()
        app.run()
        stage_files = list((self.tmp / "state" / "snapshots" / "stages").glob("*.json"))
        self.assertTrue(stage_files)
        for path in stage_files:
            self.assertIn(".v", path.name)


class FaultInjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)

    def test_invalid_jsonl_produces_failure_codes(self) -> None:
        samples, failures = load_samples(self.tmp / "samples" / "sentences_invalid.jsonl")
        codes = {item["code"] for item in failures}
        self.assertIn("E_EMPTY_TEXT", codes)
        self.assertIn("E_PARSE_INVALID", codes)
        self.assertIn("E_DUPLICATE_ID", codes)
        resource = load_rules(self.tmp / "resources" / "rules.json")
        hint_failures = validate_rule_hints(samples, resource, "invalid.jsonl")
        self.assertIn("E_RULE_UNKNOWN", {item["code"] for item in hint_failures})

    def test_invalid_config_aborts_startup(self) -> None:
        with self.assertRaises(ProofreadError) as ctx:
            load_config(self.tmp / "config" / "invalid_config.yaml")
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

    def test_unknown_rule_weight_rejected(self) -> None:
        rules_path = self.tmp / "resources" / "rules.json"
        obj = json.loads(rules_path.read_text(encoding="utf-8"))
        obj["rule_weights"]["GHOST_RULE"] = 1.0
        rules_path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(ProofreadError) as ctx:
            load_rules(rules_path)
        self.assertEqual(ctx.exception.code, "E_RULE_UNKNOWN")



class IncrementalReuseCountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)
        self.app = build_app(self.tmp)
        self.app.clean_state()
        self.app.run()

    def test_unchanged_stage_keys_are_reused_not_recomputed(self) -> None:
        # 仅改聚合权重：切分/检测阶段键不变，必须命中复用
        self.app.cache.reset_counters()
        self.app.apply_command({"cmd": "set_aggregation_weights",
                                "value": {"detect": 0.6, "confidence": 0.4}})
        self.app.snapshots.reset()
        report = self.app.run()
        reused_stages = {item.split(":", 2)[1] for item in report["cache"]["reused"]}
        recomputed_stages = {item.split(":", 2)[1]
                             for item in report["cache"]["recomputed"]}
        self.assertIn("segment", reused_stages)
        self.assertIn("detect", reused_stages)
        self.assertNotIn("segment", recomputed_stages)
        self.assertNotIn("detect", recomputed_stages)


class GracefulSignalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)

    def test_signal_completes_sample_then_exits_zero(self) -> None:
        app = build_app(self.tmp)
        app.clean_state()
        app.controller.requested = True  # 模拟首个样本处理后收到退出信号
        results, manifest, exited = app._build_pipeline().run_batch(app.samples)
        self.assertTrue(exited)
        self.assertEqual(len(results), 1)
        recovery = app.snapshots.recovery_point()
        self.assertTrue(recovery["recoverable"])
        self.assertEqual(manifest.completed_cursor, [results[0].sample.sample_id])


class SuppressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        copy_workspace(self.tmp)
        self.app = build_app(self.tmp)

    def test_overlap_suppression_keeps_rule_ids(self) -> None:
        sample = next(s for s in self.app.samples if s.sample_id == "S011")
        candidates, suppressed = detect_candidates(
            sample, segment_sample(sample, self.app.config),
            self.app.resource, self.app.config)
        self.assertTrue(suppressed)
        self.assertTrue(all(item.suppressed_by and item.rule_id for item in suppressed))
        keeper_ids = {item.rule_id for item in candidates}
        self.assertIn("R004", keeper_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
