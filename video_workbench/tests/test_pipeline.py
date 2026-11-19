"""多阶段渲染工作台单元测试。

覆盖：阶段化产物与指纹、图生视频分支、精确失效传播、补丁边界、
增量/全量逐帧一致、快照恢复、并发/串行一致、配置与失败隔离。
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vbench.config import apply_command, validate_config
from vbench.engine import RenderEngine
from vbench.errors import WorkbenchError
from vbench.persistence import SnapshotStore
from vbench.stages import RESOLUTION_TIERS, encode_prompt
from vbench.verify import assert_snapshots_equal, render_full

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default_config.json"
COMMANDS = PROJECT_ROOT / "config" / "commands.json"
INJECTIONS = PROJECT_ROOT / "config" / "injections.json"


class WorkbenchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="vbench-test-")
        self.raw = json.loads(DEFAULT_CONFIG.read_text())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cfg(self, overrides: dict | None = None):
        raw = copy.deepcopy(self.raw)
        if overrides:
            raw.update(overrides)
        return validate_config(raw, str(PROJECT_ROOT))

    def run_engine(self, workdir: str, cfg=None, injections=None, serial=False):
        cfg = cfg or self.cfg()
        store = SnapshotStore(workdir)
        engine = RenderEngine(cfg, store, injections=injections)
        return cfg, store, engine.run(serial=serial)

    # ---- 需求 1：阶段化产物与指纹 ------------------------------------------
    def test_01_stage_artifacts_and_fingerprints(self):
        workdir = os.path.join(self.tmp, "w")
        _, store, report = self.run_engine(workdir)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["stats"]["recomputed_count"], 13)
        for name in ("encode", "latent_init", "denoise_000", "decode_000"):
            artifact = store.load_artifact(name)
            self.assertIn("content_fingerprint", artifact)
            self.assertIn("dependency_fingerprints", artifact)
            self.assertIn("cache_key", artifact)
            self.assertIn("summary", artifact)
            self.assertTrue(artifact["array"].size > 0)
        # 第二跑全部命中，不重算。
        _, _, second = self.run_engine(workdir)
        self.assertEqual(second["stats"]["recomputed_count"], 0)
        self.assertEqual(second["stats"]["reused_count"], 13)
        self.assertEqual(second["reused"], sorted(second["reused"],
                         key=lambda n: store.load_artifact(n)["sort_key"]))

    # ---- 需求 2：图生视频分支与初始图校验 ----------------------------------
    def test_02_i2v_branch_and_init_image_validation(self):
        raw = copy.deepcopy(self.raw)
        raw["mode"] = "i2v"
        cfg = validate_config(raw, str(PROJECT_ROOT))
        workdir = os.path.join(self.tmp, "i2v")
        _, store, report = self.run_engine(workdir, cfg=cfg)
        self.assertEqual(report["status"], "ok")
        for index in range(cfg["replace_steps"]):
            artifact = store.load_artifact(f"denoise_{index:03d}")
            self.assertTrue(artifact["replaced"])
            self.assertEqual(artifact["origin"], "image_latent_replace")
        self.assertFalse(store.load_artifact("denoise_003")["replaced"])

        bad = copy.deepcopy(raw)
        bad_path = os.path.join(self.tmp, "bad_image.npy")
        np.save(bad_path, np.zeros((10, 10, 3), dtype=np.uint8))
        bad["i2v"]["init_image"] = bad_path
        with self.assertRaises(WorkbenchError) as caught:
            validate_config(bad, str(PROJECT_ROOT))
        self.assertEqual(caught.exception.code, "E_INIT_IMAGE_INVALID")

    # ---- 需求 3：精确失效传播 ----------------------------------------------
    def test_03_precise_invalidation_propagation(self):
        workdir = os.path.join(self.tmp, "incr")
        self.run_engine(workdir)
        # 只改帧数：去噪链全部命中，解码全部重算。
        raw = copy.deepcopy(self.raw)
        raw["frames"] = 6
        cfg = validate_config(raw, str(PROJECT_ROOT))
        _, _, report = self.run_engine(workdir, cfg=cfg)
        self.assertNotIn("denoise_000", report["recomputed"])
        self.assertIn("denoise_000", report["reused"])
        self.assertTrue(all(name.startswith("decode_") for name in report["recomputed"]))
        # 改引导强度：encode/latent_init 命中，去噪与解码重算。
        raw["guidance"] = 2.5
        cfg = validate_config(raw, str(PROJECT_ROOT))
        _, _, report = self.run_engine(workdir, cfg=cfg)
        self.assertIn("encode", report["reused"])
        self.assertIn("latent_init", report["reused"])
        self.assertIn("denoise_000", report["recomputed"])

    # ---- 需求 4：补丁精确失效边界与越界 ------------------------------------
    def test_04_patch_boundary_and_out_of_range(self):
        workdir = os.path.join(self.tmp, "patch")
        self.run_engine(workdir)
        raw = copy.deepcopy(self.raw)
        raw["patches"] = [{
            "step": 3,
            "patch": {
                "top": 1, "left": 2,
                "values": np.zeros((2, 2, 4)).tolist(),
            },
        }]
        cfg = validate_config(raw, str(PROJECT_ROOT))
        _, _, report = self.run_engine(workdir, cfg=cfg)
        for upstream in ("encode", "latent_init", "denoise_000",
                         "denoise_001", "denoise_002"):
            self.assertIn(upstream, report["reused"])
        for downstream in ("denoise_003", "denoise_004", "denoise_005",
                           "decode_000"):
            self.assertIn(downstream, report["recomputed"])

        bad = copy.deepcopy(raw)
        bad["patches"][0]["patch"]["top"] = 7
        bad["patches"][0]["patch"]["values"] = np.zeros((2, 1, 4)).tolist()
        with self.assertRaises(WorkbenchError) as caught:
            validate_config(bad, str(PROJECT_ROOT))
        self.assertEqual(caught.exception.code, "E_PATCH_OUT_OF_RANGE")

    # ---- 需求 5：编辑序列增量与全量逐帧一致 --------------------------------
    def test_05_incremental_matches_full_render(self):
        commands = json.loads(COMMANDS.read_text())
        current = copy.deepcopy(self.raw)
        incr_dir = os.path.join(self.tmp, "incr")
        store = SnapshotStore(incr_dir)
        for command in commands:
            current = apply_command(current, command)
            steps = current["steps"]
            candidate = PROJECT_ROOT / "samples" / "noise" / f"noise_{steps}.npy"
            if candidate.exists():
                current["noise_table"] = os.path.relpath(candidate, PROJECT_ROOT)
            cfg = validate_config(current, str(PROJECT_ROOT))
            RenderEngine(cfg, store).run()
        full_dir = os.path.join(self.tmp, "full")
        final_cfg, _ = render_full(current, str(PROJECT_ROOT), full_dir)
        result = assert_snapshots_equal(incr_dir, full_dir, final_cfg)
        self.assertTrue(result["equal"])
        self.assertEqual(result["frames_checked"], final_cfg["frames"])

    # ---- 需求 6：快照恢复点 ------------------------------------------------
    def test_06_recovery_point_uses_committed_snapshot_only(self):
        workdir = os.path.join(self.tmp, "rec")
        cfg, store, _ = self.run_engine(workdir)
        recovery = store.recovery_point()
        self.assertTrue(recovery["recoverable"])
        self.assertEqual(recovery["step_cursor"], cfg["steps"] - 1)
        self.assertEqual(recovery["frame_cursor"], cfg["frames"] - 1)
        # 未提交的暂存目录不得参与恢复。
        orphan = os.path.join(store.snapshot_root, "tmp-orphan")
        os.makedirs(os.path.join(orphan, "artifacts"))
        cleaned = store.cleanup_orphans()
        self.assertIn("tmp-orphan", cleaned)
        self.assertTrue(store.recovery_point()["recoverable"])

    # ---- 需求 7：并发与串行逐字段一致 --------------------------------------
    def test_07_concurrent_matches_serial(self):
        serial_dir = os.path.join(self.tmp, "serial")
        parallel_dir = os.path.join(self.tmp, "parallel")
        cfg_s, _, serial_report = self.run_engine(serial_dir, serial=True)
        cfg_p, _, parallel_report = self.run_engine(parallel_dir, serial=False)
        self.assertEqual(serial_report["recomputed"], parallel_report["recomputed"])
        result = assert_snapshots_equal(parallel_dir, serial_dir, cfg_s)
        self.assertTrue(result["equal"])
        self.assertGreaterEqual(cfg_p["max_workers"], 1)

    # ---- 需求 8：配置校验与失败隔离 ----------------------------------------
    def test_08_config_validation_matrix(self):
        cases = json.loads(INJECTIONS.read_text())["config_injections"]
        for case in cases:
            mutated = copy.deepcopy(self.raw)
            for key in [k for k in case.get("remove", "").split(",") if k]:
                mutated.pop(key, None)
            for key, value in case.get("set", {}).items():
                if key == "guidance_nan":
                    mutated["guidance"] = float("nan")
                else:
                    mutated[key] = value
            with self.assertRaises(WorkbenchError) as caught:
                validate_config(mutated, str(PROJECT_ROOT))
            self.assertEqual(caught.exception.code, case["expect"], msg=case["id"])

    def test_09_stage_failure_isolation(self):
        injections = [{
            "id": "inj-x", "type": "stage_exception",
            "stage": "denoise_002", "code": "E_STAGE_FAILED",
            "message": "unit injected failure",
        }]
        workdir = os.path.join(self.tmp, "fail")
        cfg, _, report = self.run_engine(workdir, injections=injections)
        self.assertEqual(report["status"], "failed")
        root = [f for f in report["failures"]
                if f["exception_type"] == "WorkbenchError"]
        self.assertEqual(len(root), 1)
        self.assertEqual(root[0]["stage"], "denoise_002")
        # 无关节点（encode）仍成功，下游全部隔离跳过。
        self.assertIn("encode", report["recomputed"])
        skipped = {f["stage"] for f in report["failures"]
                   if f["exception_type"] == "UpstreamSkipped"}
        self.assertIn("decode_000", skipped)
        self.assertIn("denoise_003", skipped)
        # 注入命中编号写入报告。
        self.assertEqual(report["injection_hits"][0]["id"], "inj-x")

    def test_10_prompt_encoding_independent_of_seed(self):
        first = encode_prompt("a calm ocean dawn", 1)["embedding"]
        second = encode_prompt("a calm ocean dawn", 99999)["embedding"]
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
