"""端到端管线：增量一致性、并发一致、快照恢复与异常注入。"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import unittest

from xmodal.pipeline import Pipeline
from xmodal.png_codec import encode_png
from xmodal.samplegen import generate_sample
from xmodal.tests._helpers import DEFAULT_CONFIG, Workspace

CONFIG = "config/alignment.json"
SPEC = "config/format_spec.json"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def run_pipeline(input_dir, work_dir, config_path=CONFIG, spec_path=SPEC,
                 injections=None, version=None, workers=None):
    pipe = Pipeline(input_dir, work_dir,
                    os.path.join(REPO_ROOT, config_path)
                    if not os.path.isabs(config_path) else config_path,
                    os.path.join(REPO_ROOT, spec_path)
                    if not os.path.isabs(spec_path) else spec_path,
                    injection_path=injections,
                    format_version=version,
                    max_workers_override=workers)
    return pipe.run()


BUSINESS_KEYS = ("alignment", "retrieval", "failures", "cross_format",
                 "roundtrip_fingerprints", "content_hash")


class IncrementalConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(count=6)

    def tearDown(self):
        self.ws.cleanup()

    def test_full_reuse_then_sample_replacement(self):
        first = run_pipeline(self.ws.input_dir, self.ws.work_dir,
                             self.ws.config_path, self.ws.spec_path)
        self.assertEqual(first["reuse_stats"]["extract"]["recompute"], 6)
        second = run_pipeline(self.ws.input_dir, self.ws.work_dir,
                              self.ws.config_path, self.ws.spec_path)
        self.assertEqual(second["reuse_stats"]["extract"]["reuse"], 6)
        for key in BUSINESS_KEYS:
            self.assertEqual(first[key], second[key])

        # 替换一个样本：仅该样本及其下游失效
        edited = os.path.join(self.ws.root, "edited")
        shutil.copytree(self.ws.input_dir, edited)
        canvas, regions, text = generate_sample(9, (64, 64))
        target = os.path.join(edited, "sample_002")
        with open(os.path.join(target, "image.png"), "wb") as fh:
            fh.write(encode_png(canvas))
        with open(os.path.join(target, "regions.json"), "w") as fh:
            json.dump(regions, fh, sort_keys=True)
        with open(os.path.join(target, "text.json"), "w") as fh:
            json.dump(text, fh, sort_keys=True)
        inc_work = os.path.join(self.ws.root, "work-inc")
        shutil.copytree(self.ws.work_dir, inc_work)
        inc = run_pipeline(edited, inc_work, self.ws.config_path,
                           self.ws.spec_path)
        full_work = os.path.join(self.ws.root, "work-full")
        full = run_pipeline(edited, full_work, self.ws.config_path,
                            self.ws.spec_path)
        self.assertEqual(inc["invalidation"]["extract"], ["sample_002"])
        self.assertEqual(inc["reuse_stats"]["extract"]["reuse"], 5)
        for key in BUSINESS_KEYS:
            self.assertEqual(inc[key], full[key])

    def test_config_and_format_invalidation_scope(self):
        run_pipeline(self.ws.input_dir, self.ws.work_dir,
                     self.ws.config_path, self.ws.spec_path)
        # 权重变更只失效 align/retrieve
        weight_cfg = dict(DEFAULT_CONFIG)
        weight_cfg["align_weights"] = dict(DEFAULT_CONFIG["align_weights"])
        weight_cfg["align_weights"]["mean"] = 0.5
        weight_path = os.path.join(self.ws.root, "weight.json")
        with open(weight_path, "w") as fh:
            json.dump(weight_cfg, fh)
        w_work = os.path.join(self.ws.root, "work-weight")
        shutil.copytree(self.ws.work_dir, w_work)
        report = run_pipeline(self.ws.input_dir, w_work, weight_path,
                              self.ws.spec_path)
        self.assertEqual(report["invalidation"]["extract"], [])
        self.assertEqual(len(report["invalidation"]["align"]), 6)

        # 格式切换只失效 convert
        v1_work = os.path.join(self.ws.root, "work-v1")
        shutil.copytree(self.ws.work_dir, v1_work)
        v1_report = run_pipeline(self.ws.input_dir, v1_work,
                                 self.ws.config_path, self.ws.spec_path,
                                 version=1)
        self.assertEqual(v1_report["invalidation"]["extract"], [])
        self.assertEqual(len(v1_report["invalidation"]["convert"]), 6)
        self.assertEqual(v1_report["format_version"], 1)


class ConcurrencyConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(count=6)

    def tearDown(self):
        self.ws.cleanup()

    def test_worker_counts_identical(self):
        reports = {}
        for workers in (1, 4, 8):
            work = os.path.join(self.ws.root, "work-%d" % workers)
            reports[workers] = run_pipeline(
                self.ws.input_dir, work, self.ws.config_path,
                self.ws.spec_path, workers=workers)
        for workers in (4, 8):
            for key in BUSINESS_KEYS + ("invalidation", "reuse_stats"):
                self.assertEqual(reports[1][key], reports[workers][key],
                                 "字段 %s 在并发度 %d 下不一致"
                                 % (key, workers))


class InjectionIsolationTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(count=8)
        self.injection_path = os.path.join(REPO_ROOT, "config",
                                           "injections.json")

    def tearDown(self):
        self.ws.cleanup()

    def test_all_injection_ids_hit_with_business_codes(self):
        report = run_pipeline(self.ws.input_dir, self.ws.work_dir,
                              self.ws.config_path, self.ws.spec_path,
                              injections=self.injection_path, version=1)
        hits = {h["id"]: h for h in report["injections"]}
        for inj_id in ("INJ-001", "INJ-002", "INJ-003", "INJ-004",
                       "INJ-005", "INJ-006"):
            self.assertIn(inj_id, hits)
            self.assertTrue(hits[inj_id]["hit"])
            self.assertEqual(hits[inj_id]["code"],
                             hits[inj_id]["expected_code"])
        codes = {f["code"] for f in report["failures"]["ordered"]}
        self.assertIn("E_SAMPLE_INVALID", codes)
        self.assertIn("E_RETRIEVAL_LOW_SCORE",
                      {h["code"] for h in report["injections"]})
        # 其余样本继续执行：8 个样本中仅 5 个进入失败清单
        self.assertEqual(report["failures"]["count"], 5)
        # 失败清单顺序确定（样本名 + 阶段排序）
        names = [(f["sample"], f["stage"])
                 for f in report["failures"]["ordered"]]
        self.assertEqual(names, sorted(names))


class SnapshotRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace(count=8)
        self.signal_cfg = os.path.join(REPO_ROOT, "config",
                                       "injections_signal.json")

    def tearDown(self):
        self.ws.cleanup()

    def _spawn(self, work_dir):
        cmd = [sys.executable, "-m", "xmodal.cli", "run",
               "--input", self.ws.input_dir, "--workdir", work_dir,
               "--config", self.ws.config_path, "--spec",
               self.ws.spec_path, "--injections", self.signal_cfg]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=REPO_ROOT)

    def test_sigterm_graceful_exit_zero_and_resume(self):
        work = os.path.join(self.ws.root, "work-sig")
        proc = self._spawn(work)
        time.sleep(0.6)
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, 0)
        with open(os.path.join(work, "report.json")) as fh:
            interrupted = json.load(fh)
        self.assertTrue(interrupted["interrupted"])
        # 续跑后与空白全量结果一致
        resumed = run_pipeline(self.ws.input_dir, work,
                               self.ws.config_path, self.ws.spec_path)
        full = run_pipeline(
            self.ws.input_dir,
            os.path.join(self.ws.root, "work-full"),
            self.ws.config_path, self.ws.spec_path)
        for key in BUSINESS_KEYS:
            self.assertEqual(resumed[key], full[key])

    def test_sigkill_leaves_tmp_and_recovers(self):
        work = os.path.join(self.ws.root, "work-kill")
        proc = self._spawn(work)
        time.sleep(0.7)
        proc.kill()
        proc.wait(timeout=10)
        self.assertEqual(proc.returncode, -9)
        stages = os.listdir(os.path.join(work, "stages"))
        self.assertTrue(any(n.endswith(".tmp") for n in stages))
        resumed = run_pipeline(self.ws.input_dir, work,
                               self.ws.config_path, self.ws.spec_path)
        self.assertIn("extract.tmp",
                      resumed["snapshot"]["recovery_point"]["cleaned_tmp"])
        full = run_pipeline(
            self.ws.input_dir,
            os.path.join(self.ws.root, "work-full"),
            self.ws.config_path, self.ws.spec_path)
        for key in BUSINESS_KEYS:
            self.assertEqual(resumed[key], full[key])
        stages_after = os.listdir(os.path.join(work, "stages"))
        self.assertFalse(any(n.endswith(".tmp") for n in stages_after))


if __name__ == "__main__":
    unittest.main()
