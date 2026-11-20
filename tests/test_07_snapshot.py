"""业务要求 8：阶段快照整体替换、版本识别与恢复点。"""

from __future__ import annotations

import json
import os

from vrep.config import Config
from vrep.engine import Engine
from vrep.injection import InjectionManager
from vrep.jsonio import read_json, write_json
from vrep.storage import Store, SNAPSHOT_PREFIX

from ._base import SampleTestBase


class SnapshotTests(SampleTestBase):
    def _engine(self, name: str) -> Engine:
        engine = Engine(
            os.path.join(self.temp_root, name), InjectionManager(None)
        )
        engine.seed_library(self.samples)
        return engine

    def _drain(self, engine: Engine, config: Config) -> None:
        batch_no = 1
        engine.run_batch(config, batch_no=batch_no)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            batch_no += 1
            engine.run_batch(config, batch_no=batch_no)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break

    def test_snapshot_manifest_fields_and_versioning(self) -> None:
        config = Config(
            **{
                **Config.load(self.config_path(), image_size=self.IMAGE_SIZE).to_dict(),
                "snapshot_commit_interval": 1,
            }
        )
        engine = self._engine("snap-engine")
        self._drain(engine, config)
        store = engine.store
        snapshot_names = sorted(
            name
            for name in os.listdir(store.snapshot_dir)
            if name.startswith(SNAPSHOT_PREFIX)
        )
        self.assertGreaterEqual(len(snapshot_names), 1)
        manifest = read_json(
            os.path.join(store.snapshot_dir, snapshot_names[0], "manifest.json")
        )
        for field in ("global_version", "config_hash", "quant_level",
                      "completed_images", "stage_versions"):
            self.assertIn(field, manifest)
        # 版本号单调递增且不覆盖
        versions = [int(name.replace(SNAPSHOT_PREFIX, "")) for name in snapshot_names]
        self.assertEqual(versions, sorted(set(versions)))

    def test_recovery_uses_only_completed_snapshot(self) -> None:
        config = Config(
            **{
                **Config.load(self.config_path(), image_size=self.IMAGE_SIZE).to_dict(),
                "snapshot_commit_interval": 1,
            }
        )
        ws = os.path.join(self.temp_root, "recover-engine")
        engine = self._engine_from(ws)
        engine.run_batch(config, batch_no=1)
        self.assertIsNotNone(engine.store.latest_snapshot_version())
        # 模拟强制终止：写入一个未完成的临时快照目录（无 latest 指针）
        tmp_dir = os.path.join(engine.store.tmp_dir, "snap-v999-broken")
        os.makedirs(tmp_dir, exist_ok=True)
        write_json(os.path.join(tmp_dir, "manifest.json"), {"global_version": 999})

        restarted = Engine(ws, InjectionManager(None))
        manifest = restarted.recover()
        self.assertIsNotNone(manifest)
        # 恢复点绝不能指向未完成的 v999
        self.assertNotEqual(manifest["global_version"], 999)
        self.assertEqual(
            restarted.ledger["global_version"], manifest["global_version"]
        )
        self.assertIsNotNone(restarted.ledger["last_recovery"])

    def _engine_from(self, workspace: str) -> Engine:
        engine = Engine(workspace, InjectionManager(None))
        engine.seed_library(self.samples)
        return engine

    def test_stage_version_change_detected_not_file_overwrite(self) -> None:
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        engine = self._engine_from(os.path.join(self.temp_root, "version-engine"))
        engine.run_batch(config, batch_no=1)
        first_index_key = engine.ledger["index"]["index_key"]
        threshold_config = Config(
            **{**config.to_dict(), "similarity_threshold": 0.1}
        )
        engine.run_batch(threshold_config, batch_no=2)
        second_index_key = engine.ledger["index"]["index_key"]
        self.assertNotEqual(first_index_key, second_index_key)
        # 全局版本递增，旧快照保持不可变
        self.assertEqual(engine.ledger["global_version"], 2)

    def test_blob_immutability_conflict_detection(self) -> None:
        store = Store(os.path.join(self.temp_root, "store-unit"))
        store.put_blob("key-1", {"a": 1})
        with self.assertRaises(RuntimeError):
            store.put_blob("key-1", {"a": 2})
