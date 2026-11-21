"""端到端与单元测试：固定种子样本，纯本地、确定性、无网络。"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from rs_calib.calibration import calibrate_tile_band, stable_key
from rs_calib.edits import EditEngine
from rs_calib.executor import run_keyed
from rs_calib.fingerprint import content_fingerprint
from rs_calib.fixedpoint import to_fixed_point
from rs_calib.loader import load_calibration_table, load_regression_config, load_tile_pack
from rs_calib.pipeline import Pipeline
from rs_calib.runner import compare_reports, prepare_inputs, run
from rs_calib.samplegen import generate_sample_dataset
from rs_calib.store import StateStore
from rs_calib import yaml_lite

SEED = 20261122


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="rscalib-test-")
        cls.sample = os.path.join(cls.tmp, "sample")
        generate_sample_dataset(cls.sample, seed=SEED)
        cls.table, cls.table_failures = load_calibration_table(
            os.path.join(cls.sample, "calibration_table.json"))
        cls.config, cls.config_failures = load_regression_config(
            os.path.join(cls.sample, "regression_config.yaml"), cls.table)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fresh_run(self, out_name, mode, injection=None):
        out = os.path.join(self.tmp, out_name)
        return run(self.sample, out, mode=mode,
                   injection_path=os.path.join(self.sample, "injection_config.json")
                   if injection else None)

    # 1. 加载校验 ------------------------------------------------------
    def test_01_loader_validation(self):
        packs = [os.path.join(self.sample, f) for f in sorted(os.listdir(self.sample))
                 if f.endswith(".jsonl") and f.startswith("tiles_")]
        tiles, failures = load_tile_pack(packs, self.table)
        self.assertEqual(len(tiles), 72)
        self.assertEqual(failures, [])
        ids = [t.tile_id for t in tiles]
        self.assertEqual(len(ids), len(set(ids)))
        # YAML 解析与原始配置一致
        self.assertEqual(self.config["source_whitelist"], ["S2", "S3"])
        self.assertEqual(self.config["feature_orders"], [0, 1, 2])
        self.assertEqual(self.table_failures + self.config_failures, [])

    # 2. 定点定标值的独立手工复算 -------------------------------------
    def test_02_fixed_point_calibration(self):
        packs = [os.path.join(self.sample, "tiles_batch0.jsonl")]
        tiles, _ = load_tile_pack(packs, self.table)
        tile = next(t for t in tiles if t.tile_id == "tile-0001")
        record = calibrate_tile_band(tile, self.table, self.config)
        sensor = self.table[tile.sensor_id]["bands"][tile.band]
        scale = 1 << self.config["fixed_point_frac_bits"]
        expected = []
        for count, sat in zip(tile.counts, tile.saturated_flags):
            eff = self.table[tile.sensor_id]["saturation_limit"] if sat else count
            physical = eff * sensor["gain"] + sensor["offset"]
            expected.append(int(np.rint(physical * scale)))
        self.assertEqual(record.fixed_values, expected)
        self.assertEqual(record.saturation_count, 2)
        # 位宽越界
        with self.assertRaises(Exception) as ctx:
            to_fixed_point(np.array([1e9]), 8, 4, tile_id="x", band=1, stage="calib")
        self.assertEqual(ctx.exception.code, "E_FIXEDPOINT_OVERFLOW")

    # 3. 回归对齐与残差 ------------------------------------------------
    def test_03_regression_alignment(self):
        inc = self._fresh_run("out-inc", "incremental")
        self.assertEqual(len(inc["regression_models"]), 6)
        self.assertEqual(len(inc["alignment_records"]), 48)
        for model in inc["regression_models"]:
            self.assertEqual(len(model["weights"]), 3)
            self.assertIn(model["source_sensor"], model["referenced_sensors"])
        for align in inc["alignment_records"]:
            self.assertEqual(len(align["aligned_values"]), 16)
            self.assertGreaterEqual(align["rmse"], 0.0)

    # 4/6. 增量 vs 全量逐字段一致（含/不含注入） ----------------------
    def test_04_incremental_equals_full(self):
        inc = self._fresh_run("eq-inc", "incremental")
        full = self._fresh_run("eq-full", "full")
        result = compare_reports(inc, full)
        self.assertTrue(result["equal"], result["different_fields"])
        self.assertEqual(inc["content_fingerprints"], full["content_fingerprints"])

    def test_05_incremental_equals_full_with_injection(self):
        inc = self._fresh_run("eq-inc-inj", "incremental", injection=True)
        full = self._fresh_run("eq-full-inj", "full", injection=True)
        result = compare_reports(inc, full)
        self.assertTrue(result["equal"], result["different_fields"])
        self.assertTrue(all(h["hit"] for h in inc["injection_report"]))
        codes = {f["code"] for f in inc["failure_list"]}
        for expected in ("E_PIXEL_OVERFLOW", "E_TILE_BAND_UNDECLARED",
                         "E_BACKFILL_TIMESTAMP_ORDER", "E_REG_SOURCE_UNKNOWN",
                         "E_FIXEDPOINT_OVERFLOW"):
            self.assertIn(expected, codes)

    # 5. 预算待办与账本核对 -------------------------------------------
    def test_06_budget_pending_and_ledger(self):
        inc = self._fresh_run("bud-inc", "incremental")
        ledger = inc["budget_ledger"]
        self.assertTrue(ledger["reconciled"])
        statuses = {e["status"] for e in ledger["entries"]}
        self.assertIn("PENDING", statuses)
        self.assertIn("PROCESSED", statuses)
        self.assertIn("REUSED", statuses)
        pending_codes = [f["code"] for f in inc["failure_list"]
                         if f["code"] == "E_CALIB_BUDGET_PENDING"]
        # 待办在最终均被消化（无残留 pending 失败）
        self.assertEqual(pending_codes, [])
        grants = sum(s["budget_grant"] for s in inc["batch_summaries"])
        self.assertEqual(ledger["total_budget_grant"], grants)
        self.assertLessEqual(ledger["total_processed_pixels"], grants)

    # 4. 精确失效：传感器系数变更只失效引用方 --------------------------
    def test_07_precise_invalidation(self):
        # 构造两阶段场景：先对初始瓦片建立模型，再仅修改 S2 系数
        import copy
        from rs_calib.edits import EditEngine
        packs = [os.path.join(self.sample, "tiles_batch0.jsonl")]
        tiles, _ = load_tile_pack(packs, self.table)
        config = copy.deepcopy(self.config)
        config["pixel_budget"] = 10_000
        table = copy.deepcopy(self.table)
        out = os.path.join(self.tmp, "inv")
        store = StateStore(os.path.join(out, "snapshot"), config, table)
        engine = EditEngine(table, config, tiles)
        pipeline = Pipeline(store, config, table)
        pipeline.run_batch(1, engine.tile_list())
        model_keys_before = {m["model_id"]: m["key"] for m in store.models.values()}
        self.assertEqual(len(model_keys_before), 6)

        s2 = table["S2"]
        entry = copy.deepcopy(s2)
        entry["bands"][1] = {"band": 1, "gain": s2["bands"][1]["gain"] * 1.1,
                             "offset": s2["bands"][1]["offset"] - 0.02}
        table["S2"] = entry
        pipeline.run_batch(2, engine.tile_list())
        actions = {(r.get("stage"), r.get("action"))
                   for r in store.reuse_log}
        self.assertIn(("calib", "REUSED"), actions)
        self.assertIn(("regress", "RECOMPUTED"), actions)
        s2_recompute = [r for r in store.reuse_log
                        if r.get("stage") == "regress" and "S2" in r.get("model_id", "")
                        and r["action"] == "RECOMPUTED"]
        s3_recompute = [r for r in store.reuse_log
                        if r.get("stage") == "regress" and "S3" in r.get("model_id", "")
                        and r["action"] == "RECOMPUTED"]
        self.assertTrue(s2_recompute)
        self.assertEqual(s3_recompute, [])
        # 未失效产物键保持不变
        s3_after = {m["model_id"]: m["key"] for m in store.models.values()
                    if m["source_sensor"] == "S3"}
        s3_before = {k: v for k, v in model_keys_before.items() if k.startswith("model:S3")}
        self.assertEqual(s3_after, s3_before)

    # 7. 并发结果与单线程一致 -----------------------------------------
    def test_08_concurrency_determinism(self):
        out_a = os.path.join(self.tmp, "par-1")
        out_b = os.path.join(self.tmp, "par-8")
        cfg1 = dict(self.config); cfg1["concurrency"] = 1
        cfg8 = dict(self.config); cfg8["concurrency"] = 8
        report1 = run(self.sample, out_a, mode="incremental")
        report8 = run(self.sample, out_b, mode="incremental")
        self.assertEqual(report1["calibration_records"], report8["calibration_records"])
        self.assertEqual(report1["alignment_records"], report8["alignment_records"])
        self.assertEqual(report1["content_fingerprints"], report8["content_fingerprints"])
        # 执行器顺序保证
        tasks = [(stable_key(f"t-{i:03d}", i % 3 + 1), i) for i in range(20)]
        ordered = run_keyed(tasks, lambda x: x * 2, 4)
        self.assertEqual([k for k, _ in ordered], sorted(k for k, _ in tasks))

    # 7. 快照整体替换与 stale 恢复 ------------------------------------
    def test_09_snapshot_stale_and_resume(self):
        out = os.path.join(self.tmp, "snap")
        run(self.sample, out, mode="incremental")
        snap = os.path.join(out, "snapshot", "manifest.json")
        self.assertTrue(os.path.exists(snap))
        with open(snap, encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.assertIn("config_hash", manifest)
        self.assertIn("artifact_versions", manifest)
        bad = os.path.join(self.tmp, "snap-bad")
        shutil.copytree(os.path.join(out, "snapshot"), os.path.join(bad, "snapshot"))
        mp = os.path.join(bad, "snapshot", "manifest.json")
        with open(mp, encoding="utf-8") as fh:
            m = json.load(fh)
        m["snapshot_version"] = -1
        with open(mp, "w", encoding="utf-8") as fh:
            json.dump(m, fh)
        rebuilt = run(self.sample, bad, mode="incremental", resume=True)
        self.assertEqual(rebuilt["snapshot_events"][0]["event"], "E_SNAPSHOT_STALE")
        self.assertEqual(len(rebuilt["calibration_records"]), 72)

    # 8. 注入诊断报告完整性 -------------------------------------------
    def test_10_injection_diagnostic_report(self):
        inc = self._fresh_run("diag-inj", "incremental", injection=True)
        self.assertEqual(len(inc["injection_report"]), 5)
        for hit in inc["injection_report"]:
            self.assertTrue(hit["hit"], hit["injection_id"])
            self.assertTrue(hit["matched_failures"])
        for failure in inc["failure_list"]:
            self.assertIn("stage", failure)
            self.assertIn("code", failure)
        required = {"calibration_records", "alignment_records",
                    "reuse_invalidation_log", "budget_ledger", "failure_list"}
        self.assertTrue(required.issubset(inc.keys()))

    # YAML 子集解析器 -------------------------------------------------
    def test_11_yaml_lite(self):
        parsed = yaml_lite.loads(
            "a: 1\nb: [1, 2, 3]\nc:\n  x: true\n  y:\n    - 1\n    - 2\n")
        self.assertEqual(parsed, {"a": 1, "b": [1, 2, 3],
                                  "c": {"x": True, "y": [1, 2]}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
