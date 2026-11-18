"""Deterministic logical planner.

The planner simulates the same FIFO schedule the runtime executes, using the
declared ``estimated_ms`` of every task and whole-millisecond arithmetic.  It
produces a *reproducible* plan (connection slot, queue wait and predicted
outcome) that the runtime follows and that snapshots replay against.

Constraints modelled here, identical to the runtime:

* total live connections never exceed ``max_connections``;
* per-workload active connections never exceed its quota;
* a READ_ONLY task may share an idle read-write connection, write tasks need a
  READ_WRITE connection;
* queue wait beyond ``acquire_timeout_ms`` is an E_GETTIMEOUT outcome;
* duration beyond ``hold_timeout_ms`` is an E_HOLDTIMEOUT outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import READ_WRITE, PoolConfig, Task

PLAN_SUCCESS = "SUCCESS"
PLAN_GET_TIMEOUT = "GET_TIMEOUT"
PLAN_HOLD_TIMEOUT = "HOLD_TIMEOUT"


@dataclass
class PlannedTask:
    task_id: str
    workload: str
    conn_type: str
    estimated_ms: int
    start_ms: int | None = None
    end_ms: int | None = None
    slot_seq: int | None = None
    slot_type: str | None = None
    wait_ms: int = 0
    outcome: str = PLAN_SUCCESS
    queue_enter_ms: int = 0


@dataclass
class Plan:
    tasks: list[PlannedTask] = field(default_factory=list)
    open_events: list[dict] = field(default_factory=list)
    peak_active: int = 0
    peak_workload: dict[str, int] = field(default_factory=dict)

    def by_id(self) -> dict[str, PlannedTask]:
        return {item.task_id: item for item in self.tasks}


class _Slot:
    def __init__(self, seq: int, conn_type: str) -> None:
        self.seq = seq
        self.conn_type = conn_type
        self.free_at = 0

    def supports(self, requested: str) -> bool:
        return requested != READ_WRITE or self.conn_type == READ_WRITE


def _candidate_times(plan: Plan, slots: list[_Slot], task: Task) -> list[int]:
    times: set[int] = {0}
    for slot in slots:
        if slot.supports(task.conn_type):
            times.add(slot.free_at)
    for item in plan.tasks:
        if item.outcome != PLAN_GET_TIMEOUT and item.end_ms is not None:
            times.add(item.end_ms)
    return sorted(times)


def build_plan(config: PoolConfig, tasks: list[Task]) -> Plan:
    plan = Plan()
    slots: list[_Slot] = []
    next_seq = 0
    peak_workload = {workload: 0 for workload in config.workload_quotas}
    head_ready = 0

    def active_at(moment: int, workload: str) -> int:
        return sum(
            1
            for item in plan.tasks
            if item.workload == workload
            and item.outcome != PLAN_GET_TIMEOUT
            and item.start_ms is not None
            and item.start_ms <= moment < item.end_ms  # type: ignore[operator]
        )

    for task in tasks:
        queue_enter = head_ready
        quota = config.workload_quotas[task.workload]
        chosen: tuple[int, _Slot] | None = None
        for moment in _candidate_times(plan, slots, task):
            if moment < head_ready:
                continue
            wait = moment - queue_enter
            if wait > config.acquire_timeout_ms:
                break
            if active_at(moment, task.workload) >= quota:
                continue
            reusable = [slot for slot in slots
                        if slot.supports(task.conn_type) and slot.free_at <= moment]
            if reusable:
                chosen = (moment, min(reusable, key=lambda slot: slot.seq))
                break
            if len(slots) < config.max_connections:
                preferred_type = READ_WRITE if task.conn_type == READ_WRITE else task.conn_type
                next_seq += 1
                new_slot = _Slot(next_seq, preferred_type)
                plan.open_events.append(
                    {"seq": next_seq, "conn_type": preferred_type, "at_ms": moment,
                     "task_id": task.task_id}
                )
                slots.append(new_slot)
                chosen = (moment, new_slot)
                break

        if chosen is None:
            timeout_at = queue_enter + config.acquire_timeout_ms
            plan.tasks.append(PlannedTask(
                task_id=task.task_id,
                workload=task.workload,
                conn_type=task.conn_type,
                estimated_ms=task.estimated_ms,
                start_ms=None,
                end_ms=timeout_at,
                slot_seq=None,
                wait_ms=config.acquire_timeout_ms,
                outcome=PLAN_GET_TIMEOUT,
                queue_enter_ms=queue_enter,
            ))
            head_ready = timeout_at
            continue

        start_ms, slot = chosen
        end_ms = start_ms + task.estimated_ms
        predicted = PLAN_HOLD_TIMEOUT if task.estimated_ms > config.hold_timeout_ms else PLAN_SUCCESS
        plan.tasks.append(PlannedTask(
            task_id=task.task_id,
            workload=task.workload,
            conn_type=task.conn_type,
            estimated_ms=task.estimated_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            slot_seq=slot.seq,
            slot_type=slot.conn_type,
            wait_ms=start_ms - queue_enter,
            outcome=predicted,
            queue_enter_ms=queue_enter,
        ))
        slot.free_at = end_ms
        head_ready = start_ms
        peak_workload[task.workload] = max(
            peak_workload[task.workload], active_at(start_ms, task.workload)
        )

    plan.peak_workload = peak_workload
    plan.peak_active = min(config.max_connections, len(slots))
    return plan
