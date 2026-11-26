"""Budget-constrained deterministic scheduler with todo retention."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

from .cache import ConclusionCache
from .config import DecodeConfig
from .errors import BizCode, BizError
from .models import Request, stable_task_key
from .pipeline import DecodePipeline, StepOutcome


@dataclass
class Task:
    request_id: str
    seq_id: int
    step_index: int
    points: int

    @property
    def key(self) -> str:
        return stable_task_key(self.request_id, self.seq_id, self.step_index)

    def to_dict(self) -> dict:
        return asdict(self) | {"key": self.key}


@dataclass
class BatchResult:
    outcomes: list[StepOutcome]
    completed: list[Task]
    todos: list[Task]
    failures: list[BizError]
    budget_used: int
    cache: ConclusionCache

    def sorted_outcomes(self) -> list[StepOutcome]:
        return sorted(self.outcomes, key=lambda o: o.task_key)


class Scheduler:
    """Runs stable-ordered tasks under three independent constraints.

    - batch budget: total decoding points consumable in one batch
    - per-request quota: points one request may consume within one batch
    - step recompute cap: how many times one step may be (re)computed

    A constraint breach aborts only the current batch.  Unfinished tasks are
    retained as todos and processed first in the next batch, where the batch
    budget and per-request quota reset.
    """

    def __init__(self, pipeline: DecodePipeline, requests: list[Request],
                 cache: ConclusionCache,
                 batch_budget: int, per_request_quota: int,
                 step_recompute_cap: int, max_workers: int = 1):
        self.pipeline = pipeline
        self.requests = requests
        self.cache = cache
        self.batch_budget = batch_budget
        self.per_request_quota = per_request_quota
        self.step_recompute_cap = step_recompute_cap
        self.max_workers = max_workers
        self.todos: list[Task] = []
        self.recompute_counts: dict[str, int] = {}
        self.completed_keys: set[str] = set()

    def build_tasks(self) -> list[Task]:
        tasks: list[Task] = []
        for request in sorted(self.requests, key=lambda r: r.request_id):
            for seq_id, step in request.all_steps():
                points = max(1, len(step.logprobs))
                tasks.append(Task(request.request_id, seq_id, step.index,
                                  points))
        tasks.sort(key=lambda t: t.key)
        return tasks

    def run_batch(self, carried_todos: list[Task] | None = None) -> BatchResult:
        fresh = self.build_tasks()
        queue = list(carried_todos if carried_todos is not None
                     else self.todos)
        known = {t.key for t in queue}
        for task in fresh:
            if task.key not in known and task.key not in self.completed_keys:
                queue.append(task)
                known.add(task.key)
        queue.sort(key=lambda t: t.key)

        request_index = {r.request_id: r for r in self.requests}
        outcomes: list[StepOutcome] = []
        completed: list[Task] = []
        todos: list[Task] = []
        failures: list[BizError] = []
        budget_used = 0
        request_spend: dict[str, int] = 0 and {}  # type: ignore[var-annotated]
        request_spend = {}
        aborted = False

        for position, task in enumerate(queue):
            key = task.key
            request = request_index.get(task.request_id)
            if request is None:
                failures.append(BizError(
                    BizCode.UNKNOWN_REQUEST,
                    f"request={task.request_id}/sequence={task.seq_id}"
                    f"/step={task.step_index}",
                    "task references unknown request"))
                continue

            will_recompute = self.cache.peek(
                request, task.seq_id, task.step_index,
                self.pipeline.config) is None
            if will_recompute:
                count = self.recompute_counts.get(key, 0) + 1
                if count > self.step_recompute_cap:
                    failures.append(BizError(
                        BizCode.STEP_RECOMPUTE_CAP,
                        f"request={task.request_id}/sequence={task.seq_id}"
                        f"/step={task.step_index}",
                        f"recompute count {count} > cap "
                        f"{self.step_recompute_cap}"))
                    todos.append(task)
                    continue
                self.recompute_counts[key] = count

            spend = request_spend.get(task.request_id, 0)
            if spend + task.points > self.per_request_quota:
                failures.append(BizError(
                    BizCode.REQUEST_QUOTA_EXCEEDED,
                    f"request={task.request_id}",
                    f"per-request quota {self.per_request_quota} reached"))
                # Abort this batch; this task and all later ones carry over.
                todos.extend(queue[position:])
                aborted = True
                break

            if budget_used + task.points > self.batch_budget:
                failures.append(BizError(
                    BizCode.BATCH_BUDGET_EXHAUSTED,
                    f"request={task.request_id}/sequence={task.seq_id}"
                    f"/step={task.step_index}",
                    f"batch budget {self.batch_budget} exhausted at "
                    f"{budget_used} points"))
                todos.extend(queue[position:])
                aborted = True
                break

            seq = request.sequence_map()[task.seq_id]
            step = seq.step_map()[task.step_index]
            if self.max_workers > 1:
                outcome = self._run_concurrent(request, seq, step)
            else:
                outcome = self.pipeline.process_step(request, seq, step)

            outcomes.append(outcome)
            completed.append(task)
            self.completed_keys.add(key)
            budget_used += task.points
            request_spend[task.request_id] = spend + task.points

        todos = self._dedupe(todos)
        self.todos = todos
        return BatchResult(
            outcomes=outcomes, completed=completed, todos=todos,
            failures=sorted(failures, key=lambda e: (e.code, e.location)),
            budget_used=budget_used, cache=self.cache)

    def _run_concurrent(self, request, seq, step) -> StepOutcome:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future = pool.submit(self.pipeline.process_step, request, seq, step)
            return future.result()

    @staticmethod
    def _dedupe(tasks: list[Task]) -> list[Task]:
        seen = set()
        out = []
        for task in sorted(tasks, key=lambda t: t.key):
            if task.key not in seen:
                seen.add(task.key)
                out.append(task)
        return out

    def run_to_completion(self) -> list[BatchResult]:
        batches: list[BatchResult] = []
        carried: list[Task] = []
        guard = 0
        stale_rounds = 0
        previous_todo_keys: set[str] | None = None
        while True:
            guard += 1
            result = self.run_batch(carried_todos=carried)
            batches.append(result)
            todo_keys = {t.key for t in result.todos}
            if not result.todos:
                return batches
            carried = result.todos
            # A permanently blocked task (recompute cap) yields zero
            # completions with an identical todo set two rounds in a row.
            if not result.completed and todo_keys == previous_todo_keys:
                stale_rounds += 1
            else:
                stale_rounds = 0
            if stale_rounds >= 2 or guard > 10000:
                return batches
            previous_todo_keys = todo_keys
