"""业务要求 4、5：内容指纹失效判定、复用账本、增量与全量逐字段一致。"""

from __future__ import annotations

import os

from vrep.config import Config
from vrep.engine import Engine
from vrep.injection import InjectionManager
from vrep.jsonio import canonical

from ._base import SampleTestBase


class IncrementalConsistencyTests(SampleTestBase):
    def _run_full(self, workspace: str, config_overrides: dict | None = None):
        engine = Engine(workspace, InjectionManager(None))
        engine.seed_library(self.samples)
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        if config_overrides:
            config = Config(**{**config.to_dict(), **config_overrides})
        engine.run_batch(config, batch_no=1)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            engine.run_batch(config, batch_no=2)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break
        return engine

    def _run_incremental(self, workspace: str, edits: list[dict],
                         config_overrides: dict | None = None):
        engine = Engine(workspace, InjectionManager(None))
        engine.seed_library(self.samples)
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        if config_overrides:
            config = Config(**{**config.to_dict(), **config_overrides})
        batch_no = 0
        for edit in edits:
            batch_no += 1
            if edit["op"] == "config":
                config = Config(**{**config.to_dict(), **edit["changes"]})
            else:
                engine.apply_command(edit, self.samples)
            engine.run_batch(config, batch_no=batch_no)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            batch_no += 1
            engine.run_batch(config, batch_no=batch_no)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break
        return engine

    def _comparable(self, ledger: dict) -> dict:
        return {
            "config_hash": ledger["config_hash"],
            "quant_level": ledger["quant_level"],
            "images": ledger["images"],
            "index": ledger["index"],
            "failures": ledger["failures"],
            "pending": ledger["runtime"]["pending"],
        }

    def test_replace_single_image_consistency(self) -> None:
        full_ws = os.path.join(self.temp_root, "full-replace")
        incr_ws = os.path.join(self.temp_root, "incr-replace")
        full = self._run_full(full_ws)

        # 增量：先完成初始全量构建（含预算排空），再替换一张图
        incr = Engine(incr_ws, InjectionManager(None))
        incr.seed_library(self.samples)
        config0 = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        incr.run_batch(config0, batch_no=1)
        batch_cursor = 1
        while incr.ledger["runtime"]["pending"]:
            before = len(incr.ledger["runtime"]["pending"])
            batch_cursor += 1
            incr.run_batch(config0, batch_no=batch_cursor)
            if len(incr.ledger["runtime"]["pending"]) >= before:
                break
        recompute_before = set(canonical(e) for e in incr.ledger["runtime"]["recompute"])
        reuse_before = set(canonical(e) for e in incr.ledger["runtime"]["reuse"])
        incr.apply_command(
            {"op": "replace", "image": "img_0002",
             "source": "pool/img_0002_v2.png"},
            self.samples,
        )
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        next_batch = batch_cursor + 1
        incr.run_batch(config, batch_no=next_batch)
        while incr.ledger["runtime"]["pending"]:
            before = len(incr.ledger["runtime"]["pending"])
            next_batch += 1
            incr.run_batch(config, batch_no=next_batch)
            if len(incr.ledger["runtime"]["pending"]) >= before:
                break

        # 用替换后的库做全量对照
        full_engine = Engine(
            os.path.join(self.temp_root, "full-replace-2"), InjectionManager(None)
        )
        full_engine.seed_library(self.samples)
        full_engine.apply_command(
            {"op": "replace", "image": "img_0002",
             "source": "pool/img_0002_v2.png"},
            self.samples,
        )
        full_engine.run_batch(config, batch_no=1)
        next_batch = 2
        while full_engine.ledger["runtime"]["pending"]:
            before = len(full_engine.ledger["runtime"]["pending"])
            full_engine.run_batch(config, batch_no=next_batch)
            next_batch += 1
            if len(full_engine.ledger["runtime"]["pending"]) >= before:
                break

        self.assertEqual(
            canonical(self._comparable(incr.ledger)),
            canonical(self._comparable(full_engine.ledger)),
        )
        # 仅检查替换之后新增的账本事件（此前事件属于初始构建）
        new_recompute = [
            event for event in incr.ledger["runtime"]["recompute"]
            if canonical(event) not in recompute_before
        ]
        new_reuse = [
            event for event in incr.ledger["runtime"]["reuse"]
            if canonical(event) not in reuse_before
        ]
        recomputed = {(event["stage"], event["image"]) for event in new_recompute}
        self.assertIn(("feature", "img_0002"), recomputed)
        self.assertIn(("embedding", "img_0002"), recomputed)
        # 未替换图像的特征不得重算
        self.assertNotIn(("feature", "img_0001"), recomputed)
        reused_images = {event["image"] for event in new_reuse
                         if event["stage"] == "embedding"}
        self.assertIn("img_0001", reused_images)

    def test_add_delete_consistency(self) -> None:
        edits = [
            {"op": "add", "image": "img_0099", "source": "pool/img_0099.png"},
            {"op": "delete", "image": "img_0004"},
        ]
        full_engine = Engine(
            os.path.join(self.temp_root, "full-add-del"), InjectionManager(None)
        )
        full_engine.seed_library(self.samples)
        for edit in edits:
            full_engine.apply_command(edit, self.samples)
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        full_engine.run_batch(config, 1)
        while full_engine.ledger["runtime"]["pending"]:
            before = len(full_engine.ledger["runtime"]["pending"])
            full_engine.run_batch(config, 2)
            if len(full_engine.ledger["runtime"]["pending"]) >= before:
                break

        incr = self._run_incremental(
            os.path.join(self.temp_root, "incr-add-del"), edits
        )
        self.assertEqual(
            canonical(self._comparable(incr.ledger)),
            canonical(self._comparable(full_engine.ledger)),
        )

    def test_config_changes_invalidate_only_affected_stages(self) -> None:
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        engine = Engine(
            os.path.join(self.temp_root, "cfg-engine"), InjectionManager(None)
        )
        engine.seed_library(self.samples)
        engine.run_batch(config, batch_no=1)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            engine.run_batch(config, batch_no=2)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break

        # 仅调整阈值：嵌入与分块必须复用，只有索引重算
        threshold_config = Config(**{**config.to_dict(), "similarity_threshold": 0.5})
        engine.run_batch(threshold_config, batch_no=10)
        reuse_stages = {event["stage"] for event in engine.ledger["runtime"]["reuse"]}
        recompute_stages = {
            event["stage"] for event in engine.ledger["runtime"]["recompute"]
        }
        self.assertIn("embedding", reuse_stages)
        self.assertIn("feature", reuse_stages)
        self.assertIn("index", recompute_stages)

        # 调整嵌入维度：嵌入与索引失效，分块特征仍可复用
        embed_config = Config(**{**config.to_dict(), "embed_dim": config.embed_dim + 4})
        engine.run_batch(embed_config, batch_no=11)
        reuse_stages = {event["stage"] for event in engine.ledger["runtime"]["reuse"]}
        recompute_stages = {
            event["stage"] for event in engine.ledger["runtime"]["recompute"]
        }
        self.assertIn("feature", reuse_stages)
        self.assertIn("embedding", recompute_stages)
        self.assertIn("index", recompute_stages)

    def test_quant_switch_changes_cache_keys(self) -> None:
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        engine = Engine(
            os.path.join(self.temp_root, "quant-engine"), InjectionManager(None)
        )
        engine.seed_library(self.samples)
        engine.run_batch(config, batch_no=1)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            engine.run_batch(config, batch_no=2)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break
        before_keys = {
            name: entry["embedding"]["embedding_key"]
            for name, entry in engine.ledger["images"].items()
            if entry["embedding"]
        }
        quant_config = Config(
            **{**config.to_dict(), "quant_level": 8, "fixed_bits": 8}
        )
        engine.run_batch(quant_config, batch_no=20)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            engine.run_batch(quant_config, batch_no=21)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break
        after_keys = {
            name: entry["embedding"]["embedding_key"]
            for name, entry in engine.ledger["images"].items()
            if entry["embedding"]
        }
        self.assertTrue(set(before_keys).issubset(after_keys.keys()))
        self.assertNotEqual(before_keys, after_keys)
        for name in before_keys:
            self.assertNotEqual(before_keys[name], after_keys[name])
