"""业务要求 4-8：簇账本、分量级失效、增量/全量一致、预算调度、快照与并发。"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xhash_ledger.engine import Engine
from xhash_ledger.generator import (
    build_command_stream,
    default_config,
    generate_sample,
    write_sample,
)


class EnginePipelineTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.workspace, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        for idx in range(6):
            sid = f"s{idx:03d}"
            write_sample(self.sample_dir, sid, generate_sample(1000 + idx, sid))
        # 完全副本，强制产生距离 0 的合并
        write_sample(self.sample_dir, "s100", generate_sample(1000, "s100"))
        with open(os.path.join(self.workspace, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(default_config(), fh)
        with open(os.path.join(self.workspace, "commands.json"), "w", encoding="utf-8") as fh:
            json.dump(build_command_stream(6), fh)

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _stream(self):
        return json.load(open(os.path.join(self.workspace, "commands.json")))

    def test_duplicate_samples_merge_into_one_cluster(self):
        engine = Engine(self.workspace)
        reports = engine.execute_stream(
            [{"cmd": "add", "sample_id": sid} for sid in
             ["s000", "s001", "s002", "s003", "s004", "s005", "s100"]]
        )
        ledger = reports[-1]["cluster_ledger"]
        merged = [c for c in ledger.values() if len(c["members"]) > 1]
        self.assertTrue(merged)
        cluster = next(c for c in merged if {m["sample_id"] for m in c["members"]} == {"s000", "s100"})
        self.assertEqual(cluster["cluster_id"], "s000")
        self.assertEqual([m["rank"] for m in cluster["members"]], [0, 1])
        self.assertEqual(cluster["merge_log"][0]["hamming_distance"], 0)
        self.assertEqual(cluster["merge_log"][0]["merged_cluster_id"], "s100")

    def test_cluster_id_reassignment_after_delete(self):
        engine = Engine(self.workspace)
        engine.execute_stream([
            {"cmd": "add", "sample_id": "s000"},
            {"cmd": "add", "sample_id": "s100"},
        ])
        report = engine.execute_stream([{"cmd": "delete", "sample_id": "s000"}])[-1]
        self.assertIn("s100", report["cluster_ledger"])
        self.assertNotIn("s000", report["cluster_ledger"])
        self.assertTrue(any(r["new_cluster_id"] == "s100" for r in report["cluster_reassignments"]))
        # 无悬空成员
        for cluster in report["cluster_ledger"].values():
            self.assertTrue(cluster["members"])

    def test_component_level_invalidation_on_replace(self):
        engine = Engine(self.workspace)
        engine.execute_stream([{"cmd": "add", "sample_id": f"s{idx:03d}"} for idx in range(4)])
        os.makedirs(os.path.join(self.workspace, "replacements"), exist_ok=True)
        write_sample(os.path.join(self.workspace, "replacements"), "s002", generate_sample(9999, "s002"))
        report = engine.execute_stream([{"cmd": "replace", "sample_id": "s002"}])[-1]
        status = {(e["stage"], e["subject"]): e["status"] for e in report["reuse_log"]}
        self.assertEqual(status[("feature", "s002")], "recomputed")
        self.assertEqual(status[("signature", "s002")], "recomputed")
        self.assertEqual(status[("bucket", "s002")], "recomputed")
        for other in ("s000", "s001", "s003"):
            self.assertEqual(status[("feature", other)], "reused")
        affected = next(e for e in report["reuse_log"]
                        if e["stage"] == "feature" and e["subject"] == "s002")
        self.assertTrue(affected["affected_component_indices"])

    def test_bucket_bits_change_reuses_signatures(self):
        engine = Engine(self.workspace)
        engine.execute_stream([{"cmd": "add", "sample_id": f"s{idx:03d}"} for idx in range(4)])
        cfg = {**default_config(), "bucket_bits": 16}
        report = engine.execute_stream([{"cmd": "set_config", "config": cfg}])[-1]
        status = {(e["stage"], e["subject"]): e["status"] for e in report["reuse_log"]}
        for sid in ("s000", "s001", "s002", "s003"):
            self.assertEqual(status[("signature", sid)], "reused")
            self.assertEqual(status[("bucket", sid)], "recomputed")

    def test_seed_version_changes_bit_pattern(self):
        engine = Engine(self.workspace)
        engine.execute_stream([{"cmd": "add", "sample_id": "s000"}])
        before = engine.signatures["s000"].bits
        cfg = {**default_config(), "seed_version": "seed-v2"}
        engine.execute_stream([{"cmd": "set_config", "config": cfg}])
        self.assertNotEqual(engine.signatures["s000"].bits, before)

    def test_incremental_equals_full_rebuild_field_by_field(self):
        commands = self._stream()
        engine = Engine(self.workspace)
        reports = engine.execute_stream(commands)
        incremental = reports[-1]

        final_ids = self._final_ids(commands)
        rebuild = Engine(self.workspace, rebuild=True)
        rebuild.apply_final_config(commands)
        rebuilt = rebuild.execute_rebuild(final_ids)

        self.assertEqual(incremental["signatures"], rebuilt["signatures"])
        self.assertEqual(incremental["cluster_ledger"], rebuilt["cluster_ledger"])
        inc_keys = {sid: d["cache_key"] for sid, d in incremental["signatures"].items()}
        reb_keys = {sid: d["cache_key"] for sid, d in rebuilt["signatures"].items()}
        self.assertEqual(inc_keys, reb_keys)
        self.assertEqual(
            incremental["manifest"]["config_hash"], rebuilt["manifest"]["config_hash"]
        )

    def test_snapshot_resume_and_manifest_fingerprint(self):
        engine = Engine(self.workspace)
        engine.execute_stream([{"cmd": "add", "sample_id": "s000"}])
        ok, _ = engine.store.can_resume(engine.config.config_hash())
        self.assertTrue(ok)
        # 重新实例化引擎应恢复已完成样本与缓存
        resumed = Engine(self.workspace)
        self.assertIn("s000", resumed.bundles)
        self.assertIn("s000", resumed.feature_cache)
        manifest = json.load(open(os.path.join(self.workspace, "state", "manifest.json")))
        self.assertIn("manifest_fingerprint", manifest)

    def test_budget_pending_priority_and_determinism(self):
        cfg = {**default_config(), "global_budget": 1}
        with open(os.path.join(self.workspace, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        engine = Engine(self.workspace)
        report = engine.execute_stream(
            [{"cmd": "add", "sample_id": f"s{idx:03d}"} for idx in range(3)]
        )[-1]
        self.assertTrue(any(f["code"] == "E_SIGNATURE_BUDGET_PENDING" for f in report["failures"]))
        self.assertTrue(report["pending_queue"])

    def test_quota_exceeded_failure(self):
        engine = Engine(self.workspace)
        report = engine.execute_stream([
            {"cmd": "add", "sample_id": "s000", "per_sample_quota": 1}
        ])[-1]
        self.assertTrue(
            any(f["code"] == "E_SIGNATURE_QUOTA_EXCEEDED" for f in report["failures"])
        )

    def test_concurrent_and_serial_identical(self):
        ids = [f"s{idx:03d}" for idx in range(6)] + ["s100"]
        serial_cfg = {**default_config(), "max_workers": 1}
        parallel_cfg = {**default_config(), "max_workers": 4}

        eng_serial = Engine(self.workspace)
        eng_serial.execute_stream([{"cmd": "set_config", "config": serial_cfg}])
        serial = eng_serial.execute_stream([{"cmd": "add", "sample_id": sid} for sid in ids])[-1]

        ws2 = tempfile.mkdtemp()
        try:
            shutil.copytree(self.sample_dir, os.path.join(ws2, "samples"))
            shutil.copy(os.path.join(self.workspace, "config.json"), os.path.join(ws2, "config.json"))
            eng_parallel = Engine(ws2)
            eng_parallel.execute_stream([{"cmd": "set_config", "config": parallel_cfg}])
            parallel = eng_parallel.execute_stream(
                [{"cmd": "add", "sample_id": sid} for sid in ids]
            )[-1]
            self.assertEqual(serial["signatures"], parallel["signatures"])
            self.assertEqual(serial["cluster_ledger"], parallel["cluster_ledger"])
            self.assertEqual(serial["point_deductions"], parallel["point_deductions"])
        finally:
            shutil.rmtree(ws2, ignore_errors=True)

    def _final_ids(self, commands):
        active = set()
        for command in commands:
            cmd, sid = command.get("cmd"), command.get("sample_id")
            if cmd in ("add", "replace"):
                active.add(sid)
            elif cmd == "delete":
                active.discard(sid)
        return sorted(sid for sid in active if os.path.exists(os.path.join(self.sample_dir, f"{sid}.png")))


if __name__ == "__main__":
    unittest.main()


class BudgetSchedulerTests(unittest.TestCase):
    def setUp(self):
        from xhash_ledger.scheduler import total_cost
        self.workspace = tempfile.mkdtemp()
        self.sample_dir = os.path.join(self.workspace, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)
        for idx in range(4):
            sid = f"s{idx:03d}"
            write_sample(self.sample_dir, sid, generate_sample(700 + idx, sid))
        json.dump(default_config(), open(os.path.join(self.workspace, "config.json"), "w"))
        engine = Engine(self.workspace)
        engine._install_sample("s000")
        _, layout = engine._ensure_projection()
        self.cost = total_cost(layout.dim, default_config()["signature_bits"],
                               default_config()["bucket_bits"])
        shutil.rmtree(os.path.join(self.workspace, "state"), ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_pending_retained_and_processed_first_next_run(self):
        cfg = {**default_config(), "global_budget": self.cost * 2 + 1}
        json.dump(cfg, open(os.path.join(self.workspace, "config.json"), "w"))
        engine = Engine(self.workspace)
        first = engine.apply_batch([
            {"cmd": "add", "sample_id": f"s{idx:03d}"} for idx in range(4)
        ])
        self.assertEqual(first["manifest"]["completed_samples"], ["s000", "s001"])
        self.assertEqual(first["pending_queue"], ["s002", "s003"])
        self.assertTrue(
            all(f["code"] == "E_SIGNATURE_BUDGET_PENDING" for f in first["failures"])
        )
        second = engine.run_pipeline(None)
        self.assertEqual(second["pending_queue"], [])
        self.assertEqual(
            second["manifest"]["completed_samples"],
            ["s000", "s001", "s002", "s003"],
        )
        # 待办样本在点扣减序列中最先出现
        self.assertEqual(
            [d["sample_id"] for d in second["point_deductions"]][:3],
            ["s002", "s002", "s002"],
        )
