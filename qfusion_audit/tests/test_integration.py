import json
import os
import shutil
import tempfile
import unittest

from qfusion.config import load_config
from qfusion.errors import E_SNAPSHOT_STALE
from qfusion.loader import load_stream
from qfusion.pipeline import Pipeline
from qfusion.sample_generator import generate
from qfusion.serialization import canonical_json, read_json
from qfusion.snapshot import SnapshotStore


class IntegrationBase(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.input = os.path.join(self.work, "input")
        self.tmp = os.path.join(self.work, "tmp")
        self.commit = os.path.join(self.work, "commit")
        info = generate(self.input)
        self.stream_path = info["stream"]
        self.config_path = info["config"]

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _run(self, config_path=None, stream_path=None, budget=None,
             pipeline=None):
        config_path = config_path or self.config_path
        stream_path = stream_path or self.stream_path
        cfg, cf = load_config(config_path)
        self.assertEqual(cf, [])
        store = SnapshotStore(self.tmp, self.commit)
        manifest, stale = store.validate(cfg.hash())
        start = 1 if manifest is None else manifest["global_version"] + 1
        stream = load_stream(stream_path, cfg)
        self.assertEqual(stream.failures, [])
        pipe = pipeline or Pipeline(cfg, store)
        report = pipe.run(stream, budget or cfg.recompute_budget,
                          start_version=start)
        return cfg, report, pipe

    def _current_payloads(self):
        cur = os.path.realpath(os.path.join(self.commit, "current"))
        return {
            stage: read_json(os.path.join(cur, stage, "data.json"))
            for stage in ("ledger", "flips", "boundary")
        }


class IncrementalEqualityTest(IntegrationBase):
    def _edit_stream(self, edits):
        with open(self.stream_path, encoding="utf-8") as src:
            rows = [json.loads(line) for line in src if line.strip()]
        for edit in edits:
            rows = edit(rows)
        path = os.path.join(self.work, "stream_edited.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        return path

    def test_edit_sequence_matches_clean_rebuild(self):
        # 基线全量。
        _, _, _ = self._run()
        baseline = self._current_payloads()

        # 编辑 1：分片追加（新增一个请求的新物品，保持各通道一致）。
        def append_shard(rows):
            req = "req-000"
            new_item = "item-000-777"
            chans = sorted({r["channel"] for r in rows})
            for idx, chan in enumerate(chans):
                rows.append({
                    "request_id": req, "channel": chan,
                    "item_id": new_item,
                    "raw_score": 0.42 + idx * 0.01,
                    "features": [0.1, -0.2, 0.3, 0.05],
                    "channel_rank": 90 + idx, "shard": 1})
            return rows

        # 编辑 2：单请求候选替换（改一个物品的原始得分）。
        def replace_candidate(rows):
            for row in rows:
                if row["request_id"] == "req-001" and                         row["item_id"] == "item-001-000":
                    row["raw_score"] = 0.7654
            return rows

        edited = self._edit_stream([append_shard, replace_candidate])

        # 增量运行：复用同一 Pipeline 缓存。
        cfg, _, pipe = self._run()
        _, inc_report, _ = self._run(stream_path=edited, pipeline=pipe)
        incremental = self._current_payloads()

        # 空白全量重建：全新目录 + 全新 Pipeline。
        rebuild_work = tempfile.mkdtemp()
        r_input = os.path.join(rebuild_work, "input")
        os.makedirs(r_input)
        shutil.copy(self.config_path, os.path.join(r_input,
                                                   "fusion_config.json"))
        shutil.copy(edited, os.path.join(r_input, "candidates.jsonl"))
        r_cfg, _ = load_config(os.path.join(r_input, "fusion_config.json"))
        r_store = SnapshotStore(os.path.join(rebuild_work, "tmp"),
                                os.path.join(rebuild_work, "commit"))
        r_stream = load_stream(os.path.join(r_input, "candidates.jsonl"),
                               r_cfg)
        r_pipe = Pipeline(r_cfg, r_store)
        r_pipe.run(r_stream, r_cfg.recompute_budget)
        rebuild = {
            stage: read_json(os.path.join(
                os.path.realpath(os.path.join(rebuild_work, "commit",
                                              "current")),
                stage, "data.json"))
            for stage in ("ledger", "flips", "boundary")
        }
        shutil.rmtree(rebuild_work, ignore_errors=True)

        for stage in ("ledger", "flips", "boundary"):
            self.assertEqual(canonical_json(incremental[stage]),
                             canonical_json(rebuild[stage]),
                             f"{stage} 增量与全量重建不一致")
        # 编辑确实改变了产物，避免"假一致"。
        self.assertNotEqual(canonical_json(baseline["ledger"]),
                            canonical_json(incremental["ledger"]))

    def test_config_weight_and_gear_edits_match_rebuild(self):
        # 权重调整 + 档位/位宽增删。
        with open(self.config_path, encoding="utf-8") as fh:
            edited_cfg = json.load(fh)
        edited_cfg["channels"]["hotness"]["weight"] = 0.3
        edited_cfg["gears"] = [1, 2, 4, 8]
        edited_cfg["frac_bits_choices"] = [6, 10]
        cfg_path = os.path.join(self.work, "config_edited.json")
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(edited_cfg, fh, sort_keys=True)

        self._run()
        self._run(config_path=cfg_path)
        incremental = self._current_payloads()

        rebuild_work = tempfile.mkdtemp()
        r_input = os.path.join(rebuild_work, "input")
        os.makedirs(r_input)
        shutil.copy(cfg_path, os.path.join(r_input, "fusion_config.json"))
        shutil.copy(self.stream_path,
                    os.path.join(r_input, "candidates.jsonl"))
        r_cfg, _ = load_config(os.path.join(r_input, "fusion_config.json"))
        r_store = SnapshotStore(os.path.join(rebuild_work, "tmp"),
                                os.path.join(rebuild_work, "commit"))
        r_stream = load_stream(os.path.join(r_input, "candidates.jsonl"),
                               r_cfg)
        Pipeline(r_cfg, r_store).run(r_stream, r_cfg.recompute_budget)
        rebuild = {
            stage: read_json(os.path.join(
                os.path.realpath(os.path.join(rebuild_work, "commit",
                                              "current")),
                stage, "data.json"))
            for stage in ("ledger", "flips", "boundary")
        }
        shutil.rmtree(rebuild_work, ignore_errors=True)
        for stage in ("ledger", "flips", "boundary"):
            self.assertEqual(canonical_json(incremental[stage]),
                             canonical_json(rebuild[stage]))


    def test_shard_append_only_matches_rebuild(self):
        # 仅追加一个新分片（shard=1）的记录，其余输入不变。
        rows = [json.loads(line)
                for line in open(self.stream_path, encoding="utf-8")
                if line.strip()]
        chans = sorted({r["channel"] for r in rows})
        new_item = "item-002-500"
        for idx, chan in enumerate(chans):
            rows.append({
                "request_id": "req-002", "channel": chan,
                "item_id": new_item, "raw_score": 0.33 + 0.01 * idx,
                "features": [0.05, 0.15, -0.25, 0.35],
                "channel_rank": 50 + idx, "shard": 1})
        edited = os.path.join(self.work, "shard_append.jsonl")
        with open(edited, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        cfg, _, pipe = self._run()
        self._run(stream_path=edited, pipeline=pipe)
        incremental = self._current_payloads()

        rebuild = tempfile.mkdtemp()
        r_in = os.path.join(rebuild, "input")
        os.makedirs(r_in)
        shutil.copy(self.config_path,
                    os.path.join(r_in, "fusion_config.json"))
        shutil.copy(edited, os.path.join(r_in, "candidates.jsonl"))
        r_cfg, _ = load_config(os.path.join(r_in, "fusion_config.json"))
        r_store = SnapshotStore(os.path.join(rebuild, "tmp"),
                                os.path.join(rebuild, "commit"))
        r_stream = load_stream(os.path.join(r_in, "candidates.jsonl"), r_cfg)
        Pipeline(r_cfg, r_store).run(r_stream, r_cfg.recompute_budget)
        rebuilt = {
            stage: read_json(os.path.join(
                os.path.realpath(os.path.join(rebuild, "commit", "current")),
                stage, "data.json"))
            for stage in ("ledger", "flips", "boundary")
        }
        shutil.rmtree(rebuild, ignore_errors=True)
        for stage in ("ledger", "flips", "boundary"):
            self.assertEqual(canonical_json(incremental[stage]),
                             canonical_json(rebuilt[stage]))

class ConcurrencyEqualityTest(IntegrationBase):
    def test_serial_equals_parallel(self):
        serial_cfg = json.load(open(self.config_path, encoding="utf-8"))
        serial_cfg["concurrency"] = 1
        serial_path = os.path.join(self.work, "cfg_serial.json")
        open(serial_path, "w").close()
        with open(serial_path, "w") as fh:
            json.dump(serial_cfg, fh, sort_keys=True)

        parallel_cfg = dict(serial_cfg)
        parallel_cfg["concurrency"] = 8
        parallel_path = os.path.join(self.work, "cfg_parallel.json")
        with open(parallel_path, "w") as fh:
            json.dump(parallel_cfg, fh, sort_keys=True)

        self._run(config_path=serial_path)
        serial = self._current_payloads()
        self._run(config_path=parallel_path)
        parallel = self._current_payloads()
        for stage in ("ledger", "flips", "boundary"):
            self.assertEqual(canonical_json(serial[stage]),
                             canonical_json(parallel[stage]))


class SnapshotRecoveryTest(IntegrationBase):
    def test_stale_config_hash_triggers_rebuild(self):
        self._run()
        # 篡改配置哈希：构造一个同结构不同权重的配置副本。
        with open(self.config_path, encoding="utf-8") as fh:
            changed = json.load(fh)
        changed["channels"]["lexical"]["weight"] = 0.99
        changed_path = os.path.join(self.work, "cfg_changed.json")
        with open(changed_path, "w") as fh:
            json.dump(changed, fh, sort_keys=True)
        cfg, _ = load_config(changed_path)
        store = SnapshotStore(self.tmp, self.commit)
        manifest, failures = store.validate(cfg.hash())
        self.assertIsNone(manifest)
        self.assertTrue(any(f.code == E_SNAPSHOT_STALE for f in failures))
        # 旧产物已被隔离，current 链接移除。
        self.assertFalse(os.path.exists(
            os.path.join(self.commit, "current")))
        stale_dir = os.path.join(self.commit, "stale")
        self.assertTrue(os.listdir(stale_dir))

    def test_version_increments_on_valid_rerun(self):
        _, r1, _ = self._run()
        _, r2, _ = self._run()
        # 单次运行按提交间隔产生多个阶段版本；再次运行继续递增。
        self.assertLess(r1["manifest"]["global_version"],
                        r2["manifest"]["global_version"])
        versions = os.listdir(os.path.join(self.commit, "versions"))
        self.assertTrue(all(v.startswith("v") for v in versions))
        self.assertGreaterEqual(len(versions),
                                r2["manifest"]["global_version"])
        # current 指向最新版本目录。
        cur = os.path.basename(os.path.realpath(
            os.path.join(self.commit, "current")))
        self.assertEqual(
            cur, "v" + str(r2["manifest"]["global_version"]))

    def test_manifest_tracks_completed_and_points(self):
        _, report, _ = self._run()
        manifest = report["manifest"]
        self.assertTrue(manifest["completed_requests"])
        self.assertGreater(manifest["consumed_recompute_points"], 0)
        self.assertEqual(set(manifest["stage_versions"]),
                         {"ledger", "flips", "boundary", "report"})


class BudgetContinuationTest(IntegrationBase):
    def test_tiny_budget_equals_full_budget_payloads(self):
        # 小预算多批续跑。
        cfg, report_tiny, _ = self._run(budget=7)
        tiny = self._current_payloads()
        self.assertTrue(len(report_tiny["batch_ledgers"]) > 1)
        self.assertGreater(report_tiny["budget_pending_count"], 0)

        # 全新目录一次性足额预算。
        rebuild = tempfile.mkdtemp()
        r_in = os.path.join(rebuild, "input")
        os.makedirs(r_in)
        shutil.copy(self.config_path, os.path.join(r_in,
                                                   "fusion_config.json"))
        shutil.copy(self.stream_path,
                    os.path.join(r_in, "candidates.jsonl"))
        r_cfg, _ = load_config(os.path.join(r_in, "fusion_config.json"))
        r_store = SnapshotStore(os.path.join(rebuild, "tmp"),
                                os.path.join(rebuild, "commit"))
        r_stream = load_stream(os.path.join(r_in, "candidates.jsonl"), r_cfg)
        Pipeline(r_cfg, r_store).run(r_stream, 100000)
        full = {
            stage: read_json(os.path.join(
                os.path.realpath(os.path.join(rebuild, "commit", "current")),
                stage, "data.json"))
            for stage in ("ledger", "flips", "boundary")
        }
        shutil.rmtree(rebuild, ignore_errors=True)
        for stage in ("ledger", "flips", "boundary"):
            self.assertEqual(canonical_json(tiny[stage]),
                             canonical_json(full[stage]))


if __name__ == "__main__":
    unittest.main()
