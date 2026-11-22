from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .errors import DriftError
from .models import AttemptRecord, CollectionTask, Criteria, Observation
from .protocol import monotonic_ms


@dataclass
class ScheduledOutcome:
    successes: dict[str, Observation] = field(default_factory=dict)
    attempts: list[AttemptRecord] = field(default_factory=list)
    max_observed_global: int = 0
    max_observed_per_agent: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_count": len(self.successes),
            "attempt_count": len(self.attempts),
            "max_observed_global": self.max_observed_global,
            "max_observed_per_agent": dict(sorted(self.max_observed_per_agent.items())),
        }


@dataclass
class _Work:
    task: CollectionTask
    attempt_no: int
    queued_at_ms: int
    not_before_ms: int
    order: int


class ConcurrencyScheduler:
    def __init__(self, criteria: Criteria) -> None:
        self.criteria = criteria

    def run(
        self,
        tasks: Iterable[CollectionTask],
        collect: Callable[[CollectionTask], Observation],
        *,
        batch_id: str,
    ) -> ScheduledOutcome:
        condition = threading.Condition()
        pending: list[_Work] = []
        order_counter = [0]

        def next_order() -> int:
            value = order_counter[0]
            order_counter[0] += 1
            return value

        for task in tasks:
            now = monotonic_ms()
            pending.append(_Work(task, 1, now, now, next_order()))
        active_total = 0
        active_by_agent: dict[str, int] = {}
        outcome = ScheduledOutcome()
        threads: list[threading.Thread] = []

        def on_complete(work: _Work, wait_end: int, run_start: int, run_end: int, result: Any) -> None:
            nonlocal active_total
            with condition:
                active_total -= 1
                active_by_agent[work.task.agent.agent_id] -= 1
                if isinstance(result, Observation):
                    outcome.successes[work.task.task_key] = result
                    status = "SUCCESS"
                    error_code = None
                    message = ""
                    value = result.value
                else:
                    error = result if isinstance(result, BaseException) else RuntimeError(str(result))
                    if isinstance(error, DriftError):
                        error_code = error.code
                        message = error.message
                    else:
                        error_code = "E_RECORD"
                        message = f"unexpected collector error: {error}"
                    status = "FAILED"
                    value = None
                    if work.attempt_no <= self.criteria.max_retries:
                        retry_at = monotonic_ms() + self.criteria.retry_backoff_ms
                        pending.append(
                            _Work(
                                task=work.task,
                                attempt_no=work.attempt_no + 1,
                                queued_at_ms=monotonic_ms(),
                                not_before_ms=retry_at,
                                order=next_order(),
                            )
                        )
                outcome.attempts.append(
                    AttemptRecord(
                        batch_id=batch_id,
                        task_key=work.task.task_key,
                        agent_id=work.task.agent.agent_id,
                        snapshot_seq=work.task.snapshot_seq,
                        host_id=work.task.host_id,
                        metric_id=work.task.metric_id,
                        observation_seq=work.task.observation_seq,
                        attempt_no=work.attempt_no,
                        is_retry=work.attempt_no > 1,
                        wait_start_ms=work.queued_at_ms,
                        wait_end_ms=wait_end,
                        run_start_ms=run_start,
                        run_end_ms=run_end,
                        status=status,
                        error_code=error_code,
                        message=message,
                        value=value,
                    )
                )
                condition.notify_all()

        def execute(work: _Work, wait_end: int) -> None:
            run_start = monotonic_ms()
            try:
                result: Any = collect(work.task)
            except BaseException as exc:
                result = exc
            run_end = monotonic_ms()
            on_complete(work, wait_end, run_start, run_end, result)

        with condition:
            while pending or active_total:
                now = monotonic_ms()
                selected: _Work | None = None
                if active_total < self.criteria.global_concurrency:
                    for index, work in enumerate(pending):
                        if work.not_before_ms > now:
                            continue
                        limit = min(
                            self.criteria.per_agent_concurrency,
                            work.task.agent.per_agent_concurrency,
                        )
                        if active_by_agent.get(work.task.agent.agent_id, 0) < limit:
                            selected = pending.pop(index)
                            break
                if selected is not None:
                    agent_id = selected.task.agent.agent_id
                    wait_end = now
                    active_total += 1
                    active_by_agent[agent_id] = active_by_agent.get(agent_id, 0) + 1
                    outcome.max_observed_global = max(outcome.max_observed_global, active_total)
                    outcome.max_observed_per_agent[agent_id] = max(
                        outcome.max_observed_per_agent.get(agent_id, 0),
                        active_by_agent[agent_id],
                    )
                    thread = threading.Thread(
                        target=execute,
                        args=(selected, wait_end),
                        name=f"collect-{agent_id}-{selected.order}",
                    )
                    threads.append(thread)
                    thread.start()
                    continue
                if not pending and active_total:
                    condition.wait(timeout=0.05)
                    continue
                if pending:
                    future = min(work.not_before_ms for work in pending)
                    delay_ms = max(0, future - monotonic_ms())
                    condition.wait(timeout=min(0.05, max(0.001, delay_ms / 1000.0)))
                elif active_total:
                    condition.wait(timeout=0.05)
        for thread in threads:
            thread.join()
        outcome.attempts.sort(key=lambda item: (item.task_key, item.attempt_no))
        return outcome
