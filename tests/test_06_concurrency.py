"""业务要求 7：并发与串行逐字段一致、稳定排序键、并发上限。"""

from __future__ import annotations

import os

from vrep.config import Config
from vrep.engine import Engine
from vrep.features import TileIndex
from vrep.injection import InjectionManager
from vrep.jsonio import canonical

from ._base import SampleTestBase


class ConcurrencyTests(SampleTestBase):
    def _build(self, workspace: str, worker_count: int) -> dict:
        config = Config(
            **{
                **Config.load(self.config_path(), image_size=self.IMAGE_SIZE).to_dict(),
                "worker_count": worker_count,
            }
        )
        engine = Engine(workspace, InjectionManager(None))
        engine.seed_library(self.samples)
        batch_no = 1
        engine.run_batch(config, batch_no=batch_no)
        while engine.ledger["runtime"]["pending"]:
            before = len(engine.ledger["runtime"]["pending"])
            batch_no += 1
            engine.run_batch(config, batch_no=batch_no)
            if len(engine.ledger["runtime"]["pending"]) >= before:
                break
        return engine.ledger

    def test_worker_one_vs_many_fieldwise_equal(self) -> None:
        serial = self._build(
            os.path.join(self.temp_root, "serial"), worker_count=1
        )
        parallel = self._build(
            os.path.join(self.temp_root, "parallel"), worker_count=8
        )
        comparable_fields = ("images", "index", "failures")
        for field in comparable_fields:
            self.assertEqual(
                canonical(serial[field]), canonical(parallel[field])
            )
        self.assertEqual(
            canonical(serial["runtime"]["pending"]),
            canonical(parallel["runtime"]["pending"]),
        )

    def test_repeated_parallel_runs_identical(self) -> None:
        first = self._build(os.path.join(self.temp_root, "p1"), worker_count=8)
        second = self._build(os.path.join(self.temp_root, "p2"), worker_count=8)
        self.assertEqual(canonical(first["images"]), canonical(second["images"]))
        self.assertEqual(canonical(first["index"]), canonical(second["index"]))

    def test_stable_sort_key_format(self) -> None:
        tile = TileIndex(row=2, col=3, seq=11)
        key = tile.sort_key("img_0001", "feature")
        self.assertEqual(key, "feature|img_0001|r0002|c0003|s0011")
        # 字典序与行列序号顺序一致
        keys = [
            TileIndex(row=r, col=c, seq=r * 4 + c).sort_key("img", "feature")
            for r in range(2)
            for c in range(4)
        ]
        self.assertEqual(keys, sorted(keys))

    def test_worker_count_capped_by_config(self) -> None:
        config = Config.load(self.config_path(), image_size=self.IMAGE_SIZE)
        self.assertLessEqual(config.worker_count, 32)
