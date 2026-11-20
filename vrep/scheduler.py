"""单批特征点数预算调度器（业务要求 6）。

- 一个分块特征点消耗 1 个预算；
- 按“图像名字典序 + 分块 seq 升序”的稳定顺序预订预算；
- 超出预算的分块进入持久化待办队列，标记 E_POINT_BUDGET_PENDING；
- 每个批次先消耗历史待办，再处理新任务；
- 扣减序列逐条审计，任何时刻累计扣减不超过批次预算。
"""

from __future__ import annotations

from .errors import E_POINT_BUDGET_PENDING, Failure


class BudgetScheduler:
    def __init__(self, budget_per_batch: int):
        self.budget_per_batch = int(budget_per_batch)
        self.remaining = int(budget_per_batch)
        self.reserved_total = 0
        self.audit: list[dict] = []
        self.pending: list[dict] = []

    def reset_batch(self, batch_no: int) -> None:
        self.remaining = self.budget_per_batch
        self.reserved_total = 0
        self.audit.append(
            {
                "batch": batch_no,
                "event": "reset",
                "budget": self.budget_per_batch,
                "remaining": self.remaining,
            }
        )

    def reserve(self, image: str, seq: int, stage: str, batch_no: int) -> bool:
        """按稳定顺序申请 1 个特征点预算。"""
        if self.remaining <= 0:
            self.pending.append(
                {
                    "image": image,
                    "seq": seq,
                    "stage": stage,
                    "batch_deferred": batch_no,
                }
            )
            self.audit.append(
                {
                    "batch": batch_no,
                    "event": "defer",
                    "image": image,
                    "seq": seq,
                    "stage": stage,
                    "remaining": self.remaining,
                }
            )
            return False
        self.remaining -= 1
        self.reserved_total += 1
        if self.reserved_total > self.budget_per_batch:
            raise AssertionError("预算扣减总量超过批次上限")
        self.audit.append(
            {
                "batch": batch_no,
                "event": "consume",
                "image": image,
                "seq": seq,
                "stage": stage,
                "remaining": self.remaining,
                "reserved_total": self.reserved_total,
            }
        )
        return True

    def pending_failures(self) -> list[Failure]:
        result = []
        for item in sorted(self.pending, key=lambda x: (x["image"], x["seq"], x["stage"])):
            result.append(
                Failure(
                    item["image"],
                    item["stage"],
                    E_POINT_BUDGET_PENDING,
                    f"分块 seq={item['seq']} 等待特征点预算，"
                    f"自批次 {item['batch_deferred']} 起挂起",
                )
            )
        return result

    def state(self) -> dict:
        return {
            "budget_per_batch": self.budget_per_batch,
            "remaining": self.remaining,
            "reserved_total": self.reserved_total,
            "pending": sorted(
                (dict(item) for item in self.pending),
                key=lambda x: (x["image"], x["seq"], x["stage"]),
            ),
        }
