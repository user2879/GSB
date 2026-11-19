"""Dual-budget parallel scheduler with a deterministic logical clock.

Hard guarantees:
  * running tasks never exceed the global concurrency limit;
  * running tasks of one stage never exceed that stage's quota;
  * every task records queue/start/end ticks and the slot used;
  * identical inputs produce identical hit-outcome sets regardless of the
    interleaving (the scheduling order is fully deterministic).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class TaskSpec:
    key: str
    stage_id: str
    order: int
    payload: object = None


@dataclass
class TaskResult:
    key: str
    stage_id: str
    outcome: str
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass
class ScheduleRecord:
    key: str
    stage_id: str
    order: int
    queued_tick: int
    start_tick: int
    end_tick: int
    slot: int
    wait_ticks: int
    outcome: str
    reason: str


class BudgetScheduler:
    def __init__(self, global_limit, stage_quotas, *, jitter_ms=0):
        if global_limit < 1:
            raise ValueError("global_limit must be >= 1")
        self.global_limit = global_limit
        self.stage_quotas = dict(stage_quotas)
        self.jitter_ms = jitter_ms
        self._cond = threading.Condition()
        self._tick = 0
        self._global_slots = set(range(global_limit))
        self._stage_running: dict[str, set[int]] = {}
        self._records: list[ScheduleRecord] = []
        self._cancelled = False

    def run(self, tasks, worker):
        ordered = sorted(tasks, key=lambda t: (t.order, t.key))
        pending = list(ordered)
        running: dict[str, dict] = {}
        finished: list[ScheduleRecord] = []
        queue_ticks = {t.key: self._bump() for t in ordered}
        results: dict[str, TaskResult] = {}

        def slot_of(stage_id):
            stage_slots = self._stage_running.setdefault(stage_id, set())
            quota = self.stage_quotas.get(stage_id, 1)
            free_global = sorted(self._global_slots)
            if not free_global or len(stage_slots) >= quota:
                return None
            slot = free_global[0]
            return slot

        def launch(task):
            slot = slot_of(task.stage_id)
            if slot is None:
                return False
            self._global_slots.discard(slot)
            self._stage_running[task.stage_id].add(slot)
            start = self._bump()
            event = threading.Event()
            ctx = {
                "task": task,
                "slot": slot,
                "start": start,
                "event": event,
                "thread": None,
            }

            def target():
                try:
                    result = worker(task, slot, start)
                except Exception as exc:  # pragma: no cover - surfaced below
                    result = TaskResult(
                        key=task.key,
                        stage_id=task.stage_id,
                        outcome="ERROR",
                        reason=type(exc).__name__,
                        detail={"message": str(exc)},
                    )
                ctx["result"] = result
                event.set()

            thread = threading.Thread(
                target=target, name=f"pcl-{task.key}", daemon=True
            )
            ctx["thread"] = thread
            running[task.key] = ctx
            thread.start()
            return True

        def reap(ctx):
            task = ctx["task"]
            result = ctx["result"]
            slot = ctx["slot"]
            end = self._bump()
            self._global_slots.add(slot)
            self._stage_running[task.stage_id].discard(slot)
            record = ScheduleRecord(
                key=task.key,
                stage_id=task.stage_id,
                order=task.order,
                queued_tick=queue_ticks[task.key],
                start_tick=ctx["start"],
                end_tick=end,
                slot=slot,
                wait_ticks=ctx["start"] - queue_ticks[task.key],
                outcome=result.outcome,
                reason=result.reason,
            )
            finished.append(record)
            results[task.key] = result

        while pending or running:
            progressed = False
            for task in list(pending):
                if launch(task):
                    pending.remove(task)
                    progressed = True
                else:
                    break
            if not running:
                if pending and not progressed:  # pragma: no cover - config guard
                    raise RuntimeError("scheduler deadlocked: no task could start")
                continue
            # Deterministic wait: the lowest-key running task sets the pace.
            watch_key = sorted(running)[0]
            ctx = running.pop(watch_key)
            ctx["event"].wait()
            ctx["thread"].join()
            reap(ctx)
            if self.jitter_ms:
                time.sleep(self.jitter_ms / 1000.0)

        finished.sort(key=lambda r: (r.order, r.key))
        self._records = finished
        return finished, results

    def _bump(self):
        with self._cond:
            self._tick += 1
            return self._tick

    @property
    def records(self):
        return list(self._records)

    def slot_timeline(self):
        """Stable per-slot occupancy view for the diagnostic UI."""
        timeline: dict[int, list[dict]] = {}
        for rec in sorted(self._records, key=lambda r: (r.slot, r.start_tick)):
            timeline.setdefault(rec.slot, []).append(
                {
                    "key": rec.key,
                    "stage_id": rec.stage_id,
                    "start_tick": rec.start_tick,
                    "end_tick": rec.end_tick,
                }
            )
        return [{"slot": s, "occupancy": timeline[s]} for s in sorted(timeline)]
