import json
import os
import shutil
import tempfile
import unittest

from diar.codes import (
    E_AUDIO_TRUNCATED, E_NONFINITE_SAMPLE, E_RANGE_VIOLATION,
    E_THRESHOLD_ZERO, E_TRUTH_MISMATCH,
)
from diar.config import PipelineConfig
from diar.injection import InjectionRegistry
from diar.pipeline import DiarizationPipeline
from diar.run_demo import _comparable, run_full_demo
from diar.synth import generate_dataset


class EndToEndDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.mkdtemp(prefix="diar_e2e_")
        cls.consistency, cls.canonical = run_full_demo(cls.workspace)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workspace, ignore_errors=True)

    def test_incremental_equals_canonical_field_by_field(self):
        self.assertTrue(self.consistency["incremental_equals_canonical"])

    def test_all_injection_business_codes_and_traceability(self):
        failures = self.canonical["failures"]
        codes = {item["code"] for item in failures}
        expected = {
            E_AUDIO_TRUNCATED, E_NONFINITE_SAMPLE, E_THRESHOLD_ZERO,
            E_RANGE_VIOLATION, E_TRUTH_MISMATCH,
        }
        self.assertTrue(expected.issubset(codes))
        # 每条失败都带注入编号、音频名、阶段名与业务码
        for item in failures:
            self.assertTrue(item["injection_id"].startswith("INJ-"))
            self.assertTrue(item["audio"].endswith(".wav"))
            self.assertTrue(item["stage"])
            self.assertTrue(item["code"].startswith("E_"))
        # 失败清单顺序按文件名+阶段稳定排序
        ordered = [(f["audio"], f["stage"]) for f in failures]
        self.assertEqual(ordered, sorted(ordered))
        # 单个音频失败不影响其余音频产出
        self.assertIn("mix_006.wav", self.canonical["results"])

    def test_verification_metrics_present(self):
        # mix_006 是带重叠区的干净音频；新增的 mix_007 不参与该指标断言
        result = self.canonical["results"]["mix_006.wav"]
        verification = result["verification"]
        self.assertEqual(verification["speaker_count_deviation"], 0)
        self.assertIsNotNone(verification["boundary_deviation_ms"])
        self.assertGreater(verification["overlap_total_frames"], 0)
        self.assertGreaterEqual(verification["overlap_frame_accuracy"], 0.8)
        self.assertTrue(verification["within_tolerance"])
        self.assertIsNone(verification["diagnosis"])

    def test_rule_switch_changes_adjudication(self):
        changed = self.consistency["rule_switch_changed_frames"]
        self.assertTrue(any(item["changed_frames"] > 0 for item in changed))

    def test_selective_recompute_counts(self):
        counts = self.consistency["recompute_counts"]
        # 无改动续跑零重算
        self.assertEqual(counts["resume"], 0)
        # 五阶段（特征/切分/聚类/归属/核验）下的确定重算数：
        # 档位切换令全部整数阶段失效，干净音频 5 阶段 + 注入音频已成功阶段共 12 个
        self.assertEqual(counts["after_gear"], 12)
        self.assertGreater(counts["after_gear"], counts["after_cluster"])
        # 规则切换只令归属与核验失效（聚类复用），故重算量小于全档位切换
        self.assertGreater(counts["after_rule"], 0)
        self.assertLess(counts["after_rule"], counts["after_gear"])
        # 能量门限只影响切分及下游，特征复用
        self.assertGreater(counts["after_energy"], 0)
        self.assertLess(counts["after_energy"], counts["after_gear"])


class ConcurrencyDeterminism(unittest.TestCase):
    def test_worker_counts_identical_output(self):
        workspace = tempfile.mkdtemp(prefix="diar_conc_")
        input_dir = os.path.join(workspace, "input")
        generate_dataset(input_dir, n_files=6)
        with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as fh:
            base_cfg = PipelineConfig.from_dict(json.load(fh))
        with open(os.path.join(input_dir, "truth.json"), encoding="utf-8") as fh:
            truth = json.load(fh)
        with open(os.path.join(input_dir, "injections.json"), encoding="utf-8") as fh:
            injections = json.load(fh)
        reports = {}
        for workers in (1, 2, 4, 8):
            import dataclasses
            cfg = dataclasses.replace(base_cfg, max_workers=workers)
            out = os.path.join(workspace, f"out_{workers}")
            report = DiarizationPipeline(
                input_dir, out, cfg, truth, InjectionRegistry(injections), fresh=True
            ).run()
            reports[workers] = report
        reference = _comparable(reports[1])
        for workers in (2, 4, 8):
            self.assertEqual(_comparable(reports[workers]), reference)
            self.assertEqual(reports[workers]["audio_order"], reports[1]["audio_order"])
            self.assertEqual(
                [(f["audio"], f["stage"], f["code"]) for f in reports[workers]["failures"]],
                [(f["audio"], f["stage"], f["code"]) for f in reports[1]["failures"]],
            )
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
