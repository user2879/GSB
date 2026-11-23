"""全局/单批对数预算调度与待办保留。

预算只对“从未计入预算的对”扣减一次；已计入预算的对重算不重复扣减。
全局预算耗尽时，未处理对进入待办队列并产出 E_TRAIN_BUDGET_EXHAUSTED。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import E_TRAIN_BUDGET_EXHAUSTED, FailureEntry, FailureLog
from .pairs import PreferencePair


@dataclass
class BudgetState:
    global_budget: int
    consumed: int = 0
    # 已计入预算的对（历史账本，替换/删除不退款也不重复计费）
    charged_pairs: set[str] = field(default_factory=set)
    # 待办队列（pair_id），跨运行保留，下批优先处理
    todo: list[str] = field(default_factory=list)
    # 每批游标：batch_id -> 该批已处理到的 pair 序号（按处理顺序）
    batch_cursor: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(0, self.global_budget - self.consumed)


@dataclass
class BatchPlan:
    batch_id: str
    process_pairs: list[PreferencePair]   # 本次真正处理（含复用与重算）
    newly_charged: list[PreferencePair]   # 其中首次计入预算的对
    deferred: list[PreferencePair]        # 因配额/预算保留的待办
    budget_blocked: list[PreferencePair]  # 因全局预算耗尽保留
    charged_count: int
    quota_used: int
    global_consumed: int
    global_remaining: int


class Scheduler:
    def __init__(self, state: BudgetState, per_batch_quota: int) -> None:
        self.state = state
        self.per_batch_quota = per_batch_quota

    def plan_batch(self, batch_id: str, members: list[PreferencePair],
                   failures: FailureLog) -> BatchPlan:
        """按确定性顺序规划一个批次的处理集合。

        顺序：待办优先（按 pair_id），再按成员 pair_id 字典序。
        - 单批处理对数量不超过 per_batch_quota；
        - 全局预算只约束“新计费对”数量，耗尽即阻断后续未计费对。
        """
        members_by_id = {p.pair_id: p for p in members}
        # 待办中仍属于本批且仍存在的，优先处理
        todo_here = [
            pid for pid in self.state.todo
            if pid in members_by_id
        ]
        ordered_ids = list(dict.fromkeys(
            todo_here + sorted(p.pair_id for p in members)
        ))
        # 从待办中移除本批将考虑的（无论处理还是继续保留，稍后重建）
        for pid in todo_here:
            self.state.todo.remove(pid)

        process: list[PreferencePair] = []
        newly: list[PreferencePair] = []
        deferred: list[PreferencePair] = []
        blocked: list[PreferencePair] = []
        exhausted_emitted = False

        for pid in ordered_ids:
            pair = members_by_id[pid]
            if len(process) >= self.per_batch_quota:
                deferred.append(pair)
                continue
            if pid in self.state.charged_pairs:
                # 已计费：处理但不重复扣减
                process.append(pair)
                continue
            # 未计费：需要占用全局预算
            if self.state.remaining <= 0:
                blocked.append(pair)
                if not exhausted_emitted:
                    failures.add(FailureEntry(
                        sort_key=batch_id, code=E_TRAIN_BUDGET_EXHAUSTED,
                        stage="schedule",
                        message="global pair budget exhausted; pairs retained in todo",
                        context=f"batch={batch_id}",
                    ))
                    exhausted_emitted = True
                continue
            process.append(pair)
            newly.append(pair)
            self.state.charged_pairs.add(pid)
            self.state.consumed += 1

        # 保留待办：配额截断 + 预算阻断，统一按 pair_id 排序入队
        retained = sorted({p.pair_id for p in deferred + blocked})
        self.state.todo = retained + [
            pid for pid in self.state.todo if pid not in members_by_id
        ]
        self.state.batch_cursor[batch_id] = len(process)

        return BatchPlan(
            batch_id=batch_id,
            process_pairs=process,
            newly_charged=newly,
            deferred=deferred,
            budget_blocked=blocked,
            charged_count=len(newly),
            quota_used=len(process),
            global_consumed=self.state.consumed,
            global_remaining=self.state.remaining,
        )
