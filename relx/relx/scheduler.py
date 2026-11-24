"""单批特征点数预算调度器与预算账本。

调度器按句子 ID 稳定顺序消耗批次预算，单个任务超过整批预算时直接以
E_FEATURE_BUDGET_OVERSIZE 进入失败清单；句子在当前批预算不足时保留在
待办队列并标记 E_FEATURE_BUDGET_PENDING，下一批优先处理待办。
预算扣减顺序严格等价于单线程按句子 ID 升序执行。
"""

from dataclasses import dataclass

from .errors import E_FEATURE_BUDGET_OVERSIZE, E_FEATURE_BUDGET_PENDING, FailureRecord


@dataclass(frozen=True)
class ScheduledTask:
    sentence_id: str
    stage: str
    points: int

    @property
    def sort_key(self):
        return (self.sentence_id, self.stage)


@dataclass
class LedgerRow:
    batch_no: int
    sentence_id: str
    stage: str
    points: int
    batch_remaining_after: int
    global_consumed_after: int
    status: str

    def to_dict(self):
        return {
            "batch_no": self.batch_no,
            "sentence_id": self.sentence_id,
            "stage": self.stage,
            "points": self.points,
            "batch_remaining_after": self.batch_remaining_after,
            "global_consumed_after": self.global_consumed_after,
            "status": self.status,
        }


class BudgetScheduler:
    def __init__(self, budget_per_batch):
        self.budget_per_batch = budget_per_batch
        self.pending = []
        self.ledger = []
        self.global_consumed = 0
        self.batch_no = 0

    def plan_wave(self, planned_tasks):
        """planned_tasks: 当前批候选任务（含历史待办），返回本批执行任务。"""
        self.batch_no += 1
        deduped = {}
        for task in self.pending + list(planned_tasks):
            deduped.setdefault(task.sort_key, task)
        merged = sorted(deduped.values(), key=lambda task: task.sort_key)
        self.pending = []
        remaining = self.budget_per_batch
        runnable = []
        pending_rows = []
        oversize = []
        for task in merged:
            if task.points > self.budget_per_batch:
                oversize.append(FailureRecord(
                    E_FEATURE_BUDGET_OVERSIZE, task.sentence_id, task.stage,
                    f"任务点数 {task.points} 超过单批预算 {self.budget_per_batch}",
                ))
                continue
            if task.points > remaining:
                self.pending.append(task)
                pending_rows.append(FailureRecord(
                    E_FEATURE_BUDGET_PENDING, task.sentence_id, task.stage,
                    f"批次预算不足，任务保留待办（需要 {task.points}，剩余 {remaining}）",
                ))
                continue
            remaining -= task.points
            self.global_consumed += task.points
            runnable.append(task)
            self.ledger.append(LedgerRow(
                batch_no=self.batch_no,
                sentence_id=task.sentence_id,
                stage=task.stage,
                points=task.points,
                batch_remaining_after=remaining,
                global_consumed_after=self.global_consumed,
                status="charged",
            ))
        return runnable, pending_rows, oversize, remaining

    def has_pending(self):
        return bool(self.pending)

    def pending_summary(self):
        return [
            {"sentence_id": task.sentence_id, "stage": task.stage, "points": task.points}
            for task in sorted(self.pending, key=lambda item: item.sort_key)
        ]

    def verify_ledger(self):
        total = sum(row.points for row in self.ledger)
        return total == self.global_consumed
