"""全局重建点数预算与单视图配额的确定性调度账本。

线程池只负责无状态计算；预算的分配 / 扣减始终在主线程按视图序号的
稳定顺序执行，因此无论线程调度顺序与预算耗尽时机如何，扣减序列、
待处理队列与最终结果都与单线程串行执行完全一致，且全程可审计。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import E_POINT_BUDGET_EXHAUSTED


@dataclass
class BudgetLedger:
    global_budget: int
    per_view_quota: int
    remaining: int = field(init=False)
    consumed: int = field(default=0)
    entries: list[dict] = field(default_factory=list)
    queue_order: list[int] = field(default_factory=list)
    exhausted_views: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.remaining = self.global_budget

    @classmethod
    def restore(cls, global_budget: int, per_view_quota: int, consumed: int) -> "BudgetLedger":
        ledger = cls(global_budget, per_view_quota)
        ledger.remaining = max(0, global_budget - consumed)
        ledger.consumed = consumed
        return ledger

    def reserve(self, view_id: int) -> dict:
        """按稳定顺序为视图登记待处理队列与单视图配额。"""
        self.queue_order.append(view_id)
        entry = {
            "view_id": view_id,
            "per_view_quota": self.per_view_quota,
            "remaining_before": self.remaining,
            "requested_points": None,
            "accepted_points": None,
            "truncated_points": None,
            "remaining_after": self.remaining,
            "status": "queued",
        }
        self.entries.append(entry)
        return entry

    def settle(self, view_id: int, requested_points: int) -> tuple[int, bool, dict]:
        """登记视图实际请求的重建点数，返回接受数量与是否触发预算耗尽。"""
        entry = next(item for item in self.entries if item["view_id"] == view_id)
        accepted = min(requested_points, self.remaining)
        truncated = requested_points - accepted
        exhausted = truncated > 0
        self.remaining -= accepted
        self.consumed += accepted
        entry.update(
            requested_points=requested_points,
            accepted_points=accepted,
            truncated_points=truncated,
            remaining_after=self.remaining,
            status="exhausted" if exhausted else "settled",
        )
        if exhausted and view_id not in self.exhausted_views:
            self.exhausted_views.append(view_id)
        return accepted, exhausted, entry

    def budget_failure(self, view_id: int) -> dict:
        return {
            "code": E_POINT_BUDGET_EXHAUSTED,
            "message": (
                f"视图 {view_id} 重建点数超出全局预算：剩余 {self.remaining}，"
                "该视图保留在待处理队列稳定位置，跳过三角化 / 深度 / 点云阶段"
            ),
            "stage": "budget",
            "view_id": view_id,
        }

    def to_dict(self) -> dict:
        return {
            "global_budget": self.global_budget,
            "per_view_quota": self.per_view_quota,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "queue_order": list(self.queue_order),
            "deduction_sequence": [
                {
                    "view_id": entry["view_id"],
                    "requested_points": entry["requested_points"],
                    "accepted_points": entry["accepted_points"],
                    "truncated_points": entry["truncated_points"],
                    "remaining_before": entry["remaining_before"],
                    "remaining_after": entry["remaining_after"],
                    "status": entry["status"],
                }
                for entry in self.entries
            ],
            "exhausted_views": list(self.exhausted_views),
        }
