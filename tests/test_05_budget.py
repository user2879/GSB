"""业务要求 6：单批特征点数预算、待办保留与审计。"""

from __future__ import annotations

import os

from vrep.config import Config
from vrep.engine import Engine
from vrep.errors import E_POINT_BUDGET_PENDING
from vrep.injection import InjectionManager
from vrep.scheduler import BudgetScheduler

from ._base import SampleTestBase


class SchedulerUnitTests(SampleTestBase):
    def test_reservation_never_exceeds_budget(self) -> None:
        scheduler = BudgetScheduler(budget_per_batch=5)
        scheduler.reset_batch(batch_no=1)
        accepted = 0
        for seq in range(12):
            if scheduler.reserve("img", seq, "feature", 1):
                accepted += 1
        self.assertEqual(accepted, 5)
        self.assertEqual(scheduler.remaining, 0)
        self.assertEqual(len(scheduler.pending), 7)
        self.assertEqual(scheduler.reserved_total, 5)

    def test_audit_trail_is_complete_and_ordered(self) -> None:
        scheduler = BudgetScheduler(budget_per_batch=3)
        scheduler.reset_batch(batch_no=1)
        for seq in range(5):
            scheduler.reserve("img", seq, "feature", 1)
        events = scheduler.audit
        self.assertEqual(events[0]["event"], "reset")
        consume = [e for e in events if e["event"] == "consume"]
        defer = [e for e in events if e["event"] == "defer"]
        self.assertEqual(len(consume), 3)
        self.assertEqual(len(defer), 2)
        self.assertEqual([e["seq"] for e in consume], [0, 1, 2])
        self.assertEqual([e["seq"] for e in defer], [3, 4])

    def test_pending_failures_have_business_code(self) -> None:
        scheduler = BudgetScheduler(budget_per_batch=1)
        scheduler.reset_batch(1)
        scheduler.reserve("img", 0, "feature", 1)
        scheduler.reserve("img", 1, "feature", 1)
        failures = scheduler.pending_failures()
        self.assertTrue(all(f.code == E_POINT_BUDGET_PENDING for f in failures))


class BudgetIntegrationTests(SampleTestBase):
    def test_pending_drains_across_batches_and_never_indexed(self) -> None:
        config = Config(
            **{
                **Config.load(self.config_path(), image_size=self.IMAGE_SIZE).to_dict(),
                "point_budget_per_batch": 8,
            }
        )
        engine = Engine(
            os.path.join(self.temp_root, "budget-engine"), InjectionManager(None)
        )
        engine.seed_library(self.samples)

        engine.run_batch(config, batch_no=1)
        self.assertGreater(len(engine.ledger["runtime"]["pending"]), 0)
        # 待办图像不得进入索引
        pending_images = {item["image"] for item in engine.ledger["runtime"]["pending"]}
        self.assertTrue(pending_images.isdisjoint(set(engine.ledger["index"]["images"])))

        # 预算审计：任何批次累计扣减不超过预算
        for batch_no in range(2, 40):
            engine.run_batch(config, batch_no=batch_no)
            if not engine.ledger["runtime"]["pending"]:
                break
        self.assertEqual(engine.ledger["runtime"]["pending"], [])
        # 排空后所有图像入索引
        self.assertEqual(
            sorted(engine.ledger["index"]["images"]),
            sorted(engine._load_manifest()["images"]),
        )

        # 审计中每个批次 consume 数 <= 预算
        per_batch: dict[int, int] = {}
        for event in engine.ledger["runtime"]["budget_audit"]:
            if event["event"] == "consume":
                per_batch[event["batch"]] = per_batch.get(event["batch"], 0) + 1
        self.assertTrue(all(count <= 8 for count in per_batch.values()))

    def test_pending_queue_is_stable_sorted(self) -> None:
        config = Config(
            **{
                **Config.load(self.config_path(), image_size=self.IMAGE_SIZE).to_dict(),
                "point_budget_per_batch": 4,
            }
        )
        engine = Engine(
            os.path.join(self.temp_root, "budget-sort"), InjectionManager(None)
        )
        engine.seed_library(self.samples)
        engine.run_batch(config, batch_no=1)
        pending = engine.ledger["runtime"]["pending"]
        keys = [(item["image"], item["seq"]) for item in pending]
        self.assertEqual(keys, sorted(keys))
