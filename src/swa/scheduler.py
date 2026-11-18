"""Triple-budget parallel shard scheduler (short-task, non-blocking coordinator).

Design that is deadlock-free under all three budgets at once:

* Shard aggregate state lives in coordinator-owned dictionaries; a worker task
  only borrows a concurrency slot, applies one pending event batch, then
  releases the slot immediately. Tasks never block while holding a slot.
* The global cap and per-rule quota are bounded semaphores acquired *inside*
  each short task. Blocked tasks sit in the executor queue without pinning a
  thread, so they can never starve runnable tasks.
* The per-window distinct-group budget is coordinator admission state. A new
  group is never created by evicting an existing one; when the budget is full
  the coordinator parks the event (back-pressure) and records monotonic-clock
  wait start/end. Sustained exhaustion beyond the threshold raises E_BUDGET.

Because aggregation (count/sum/max) is order-independent and the final window
output is rebuilt from canonical state sorted by group key, any scheduling
interleaving yields byte-identical results.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .spec import RuleSpec
from .util import monotonic_ms


@dataclass
class WaitRecord:
    reason: str
    rule_id: str
    window_start_ms: int
    group_key: str
    wait_start_monotonic_ms: float
    wait_end_monotonic_ms: float

    @property
    def waited_ms(self) -> float:
        return self.wait_end_monotonic_ms - self.wait_start_monotonic_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "rule_id": self.rule_id,
            "window_start_ms": self.window_start_ms,
            "group_key": self.group_key,
            "wait_start_monotonic_ms": round(self.wait_start_monotonic_ms, 3),
            "wait_end_monotonic_ms": round(self.wait_end_monotonic_ms, 3),
            "waited_ms": round(self.waited_ms, 3),
        }


@dataclass
class Shard:
    rule_id: str
    window_start_ms: int
    group_key: str
    state: dict[str, int] = field(default_factory=dict)
    pending: list = field(default_factory=list)
    scheduled: bool = False
    admitted: bool = False
    event_count: int = 0


class Scheduler:
    def __init__(
        self,
        rules: tuple[RuleSpec, ...],
        global_limit: int,
        *,
        budget_wait_threshold_ms: float | None = None,
        max_workers: int | None = None,
        compute_hook: Callable[["Shard", list], None] | None = None,
    ):
        self.rules = {rule.rule_id: rule for rule in rules}
        self.global_limit = global_limit
        self.global_slots = threading.BoundedSemaphore(global_limit)
        self.rule_slots = {
            rule.rule_id: threading.BoundedSemaphore(rule.concurrency) for rule in rules
        }
        self.budget_wait_threshold_ms = budget_wait_threshold_ms
        self.compute_hook = compute_hook
        self.memory_items: dict[tuple[str, int], set[str]] = {}
        self.memory_peak: dict[tuple[str, int], int] = {}
        self.waits: list[WaitRecord] = []
        self.submitted_shards = 0
        self.completed_tasks = 0
        self.queued_tasks_with_wait = 0
        self.peak_running_global = 0
        self.peak_running_by_rule = {rule.rule_id: 0 for rule in rules}
        self._running_global = 0
        self._running_by_rule = {rule.rule_id: 0 for rule in rules}
        self._active: set[int] = set()
        self._active_by_rule: dict[str, set[int]] = {rule.rule_id: set() for rule in rules}
        self._metrics_lock = threading.Lock()
        self._error: BaseException | None = None
        self._error_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers or 64, thread_name_prefix="shard"
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _fail(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = exc

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise self._error

    def memory_full(self, rule_id: str, window_start_ms: int, group_key: str) -> bool:
        """True when a NEW distinct group cannot be admitted right now."""
        rule = self.rules[rule_id]
        items = self.memory_items.setdefault((rule_id, window_start_ms), set())
        return group_key not in items and len(items) >= rule.memory_budget

    def admit(self, shard: Shard) -> bool:
        """Admit a shard's group into the per-window memory budget (coordinator)."""
        key = (shard.rule_id, shard.window_start_ms)
        items = self.memory_items.setdefault(key, set())
        if shard.group_key in items:
            shard.admitted = True
            return True
        rule = self.rules[shard.rule_id]
        if len(items) >= rule.memory_budget:
            return False
        items.add(shard.group_key)
        shard.admitted = True
        self.memory_peak[key] = max(self.memory_peak.get(key, 0), len(items))
        return True

    def dispatch(self, shard: Shard, events: list, on_complete: Callable[[Shard], None]) -> None:
        """Schedule one short processing task; never blocks the coordinator."""
        if not shard.scheduled:
            shard.scheduled = True
            self.submitted_shards += 1
        self._executor.submit(self._run_task, shard, list(events), on_complete)

    def _run_task(self, shard: Shard, events: list, on_complete: Callable[[Shard], None]) -> None:
        wait_start = monotonic_ms()
        rule = self.rules[shard.rule_id]
        with self._metrics_lock:
            must_wait = (
                self._running_global >= self.global_limit
                or self._running_by_rule[shard.rule_id] >= rule.concurrency
            )
        if must_wait:
            self.queued_tasks_with_wait += 1
        try:
            self.global_slots.acquire()
            self.rule_slots[shard.rule_id].acquire()
            wait_end = monotonic_ms()
            task_id = id(object())
            with self._metrics_lock:
                self._running_global += 1
                self._active.add(task_id)
                self.peak_running_global = max(self.peak_running_global, len(self._active))
                self._running_by_rule[shard.rule_id] += 1
                self._active_by_rule[shard.rule_id].add(task_id)
                self.peak_running_by_rule[shard.rule_id] = max(
                    self.peak_running_by_rule[shard.rule_id],
                    len(self._active_by_rule[shard.rule_id]),
                )
            if must_wait:
                self.waits.append(
                    WaitRecord("slot", shard.rule_id, shard.window_start_ms, shard.group_key, wait_start, wait_end)
                )
            # Pure function over local copies; state merge happens once at end.
            if self.compute_hook is not None:
                self.compute_hook(shard, events)
            local_count = shard.state.get("count", 0)
            local_sum = shard.state.get("sum", 0)
            local_max = shard.state.get("max")
            for event in events:
                local_count += 1
                local_sum += event.value
                local_max = event.value if local_max is None else max(local_max, event.value)
            shard.state["count"] = local_count
            shard.state["sum"] = local_sum
            if local_max is not None:
                shard.state["max"] = local_max
            shard.event_count += len(events)
        except BaseException as exc:
            self._fail(exc)
        finally:
            try:
                self.rule_slots[shard.rule_id].release()
                self.global_slots.release()
            except ValueError:
                pass
            with self._metrics_lock:
                self._running_global = max(0, self._running_global - 1)
                self._active.discard(task_id)
                self._running_by_rule[shard.rule_id] = max(0, self._running_by_rule[shard.rule_id] - 1)
                self._active_by_rule[shard.rule_id].discard(task_id)
                self.completed_tasks += 1
            shard.scheduled = False
            on_complete(shard)

    def record_memory_wait(self, record: WaitRecord) -> None:
        self.waits.append(record)

    def release_window(self, rule_id: str, window_start_ms: int) -> None:
        self.memory_items.pop((rule_id, window_start_ms), None)

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            return {
                "global_limit": self.global_limit,
                "submitted_shards": self.submitted_shards,
                "completed_shards": self.completed_tasks,
                "queued_shards_with_wait": self.queued_tasks_with_wait,
                "pending_shards": 0,
                "max_running_global": self.peak_running_global,
                "max_running_by_rule": dict(self.peak_running_by_rule),
                "max_memory_items_by_window": {
                    f"{rule_id}:{window_start}": peak
                    for (rule_id, window_start), peak in sorted(self.memory_peak.items())
                },
                "memory_budget_by_rule": {
                    rule_id: rule.memory_budget for rule_id, rule in sorted(self.rules.items())
                },
                "wait_records": [record.to_dict() for record in self.waits],
            }
