"""Deterministic tick-based GPU shard scheduler with three-fold constraints.

The virtual monotonic clock advances in fixed integer ticks.  A shard stays
RUNNING across several ticks until its declared duration elapses, so task
submissions that arrive while resources are in flight can trigger priority
preemption at the nearest shard boundary.

Constraints enforced simultaneously:
  1. global GPU quota; 2. per-pool concurrent task limit;
  3. per-task memory budget and per-pool GPU/memory capacity.

All state transitions and numeric water levels are integers; identical inputs
produce byte-identical round ledgers.  Shard work is projected onto a real
local worker thread pool, while timing uses the deterministic virtual clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..digest import chain_digest, digest_of
from ..fixedpoint import quantize_ratio
from ..models import Declaration, Shard, ShardState
from .propagation import propagate_shard_invalidation
from .worker import ShardThreadPool

TICK_MS = 10


@dataclass
class TaskRuntime:
    task_id: str
    completed: set[int] = field(default_factory=set)
    next_index: int = 0
    suspended: bool = False
    failed: bool = False
    pool_id: str | None = None
    wait_start_ms: int | None = None
    preemption_count: int = 0


@dataclass
class RunningShard:
    shard: Shard
    task_id: str
    pool_id: str
    gpu_need: int
    mem_need: int
    start_ms: int
    remaining_ms: int


class SchedulerEngine:
    def __init__(self, decl: Declaration, max_workers: int = 4):
        self.decl = decl
        self.params = decl.params
        self.pool = ShardThreadPool(max_workers=max_workers)
        self.clock_ms = 0
        self.round_no = 0
        self.running: dict[str, RunningShard] = {}
        self._futures: dict[str, Any] = {}
        self.runtime: dict[str, TaskRuntime] = {
            tid: TaskRuntime(tid) for tid in decl.tasks}
        self.unknown_shards: list[dict[str, Any]] = []
        self.rounds: list[dict[str, Any]] = []
        self.chain = "0" * 64
        self.fail_inject: set[str] = set()
        self.crash_after_round: int | None = None
        self._worker_seen: set[str] = set()

    # ------------------------------------------------------------------ usage
    def _task_memory_used(self, task_id: str) -> int:
        return sum(run.mem_need for run in self.running.values()
                   if run.task_id == task_id)

    def _pool_usage(self, pool_id: str) -> tuple[int, int, set[str]]:
        gpu_used = mem_used = 0
        task_ids: set[str] = set()
        for run in self.running.values():
            if run.pool_id == pool_id:
                gpu_used += run.gpu_need
                mem_used += run.mem_need
                task_ids.add(run.task_id)
        return gpu_used, mem_used, task_ids

    def _global_gpu_used(self) -> int:
        return sum(run.gpu_need for run in self.running.values())

    def _is_ready(self, shard: Shard) -> bool:
        if shard.state in (ShardState.COMPLETED, ShardState.RUNNING,
                           ShardState.UNKNOWN, ShardState.AFFECTED,
                           ShardState.FAILED):
            return False
        rt = self.runtime[shard.task_id]
        if rt.failed or rt.suspended:
            return False
        if shard.index != rt.next_index:
            return False
        if self.clock_ms < self.decl.tasks[shard.task_id].submit_time_ms:
            return False
        return all(self.decl.shards[ref].state == ShardState.COMPLETED
                   for ref in shard.deps)

    def _resume_suspended(self, decisions: list[dict[str, Any]]) -> None:
        """Clear suspension for tasks whose shard can now be considered again.

        Completed shards are retained: only the next unfinished shard re-enters
        scheduling, so finished GPU occupancy is never counted twice.
        """
        for tid in sorted(self.runtime):
            rt = self.runtime[tid]
            if not rt.suspended or rt.failed:
                continue
            task = self.decl.tasks[tid]
            if rt.next_index >= task.shard_count:
                rt.suspended = False
                continue
            candidate = self.decl.shards[f"{tid}#{rt.next_index}"]
            if self.clock_ms < task.submit_time_ms:
                continue
            deps_done = all(self.decl.shards[ref].state == ShardState.COMPLETED
                            for ref in candidate.deps)
            if deps_done:
                rt.suspended = False
                candidate.state = ShardState.PENDING
                decisions.append({
                    "action": "resume", "task_id": tid,
                    "next_shard": candidate.global_id,
                    "reused_completed": sorted(rt.completed),
                    "reason": "resume after preemption; rerun only unfinished shards",
                })

    def _ready_shards(self) -> list[Shard]:
        ready = [s for s in self.decl.shards.values() if self._is_ready(s)]
        ready.sort(key=lambda s: (
            -self.decl.tasks[s.task_id].priority,
            self.decl.tasks[s.task_id].submit_time_ms,
            s.task_id, s.index))
        return ready

    # ------------------------------------------------------------- water level
    def water_levels(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for pid, pool in sorted(self.decl.pools.items()):
            gpu_used, mem_used, task_ids = self._pool_usage(pid)
            report[pid] = {
                "gpu_used": gpu_used,
                "gpu_free": pool.gpu_count - gpu_used,
                "gpu_capacity": pool.gpu_count,
                "memory_used_mib": mem_used,
                "memory_free_mib": pool.total_memory_mib - mem_used,
                "memory_capacity_mib": pool.total_memory_mib,
                "running_task_count": len(task_ids),
                "running_tasks": sorted(task_ids),
                "water_ratio_q": quantize_ratio(
                    gpu_used, pool.gpu_count, self.params.scale_factor),
                "scale_factor": self.params.scale_factor,
                "truncation": self.params.truncation,
            }
        report["__global__"] = {
            "gpu_used": self._global_gpu_used(),
            "gpu_free": self.params.global_gpu_quota - self._global_gpu_used(),
            "gpu_capacity": self.params.global_gpu_quota,
            "water_ratio_q": quantize_ratio(
                self._global_gpu_used(), self.params.global_gpu_quota,
                self.params.scale_factor),
            "scale_factor": self.params.scale_factor,
            "truncation": self.params.truncation,
        }
        return report

    # --------------------------------------------------------------- admission
    def _can_admit(self, shard: Shard, pool_id: str) -> tuple[bool, str]:
        task = self.decl.tasks[shard.task_id]
        pool = self.decl.pools[pool_id]
        if task.gpu_need > pool.gpu_count:
            return False, f"pool {pool_id} gpu capacity {pool.gpu_count} < need {task.gpu_need}"
        gpu_used, mem_used, task_ids = self._pool_usage(pool_id)
        if gpu_used + task.gpu_need > pool.gpu_count:
            return False, "pool gpu capacity exceeded"
        if mem_used + task.memory_per_gpu_mib * task.gpu_need > pool.total_memory_mib:
            return False, "pool memory capacity exceeded"
        if shard.task_id not in task_ids and len(task_ids) >= self.params.pool_concurrency_limit:
            return False, f"pool concurrency limit {self.params.pool_concurrency_limit} reached"
        if self._global_gpu_used() + task.gpu_need > self.params.global_gpu_quota:
            return False, f"global gpu quota {self.params.global_gpu_quota} exceeded"
        if task.memory_budget_mib:
            if self._task_memory_used(shard.task_id) + \
                    task.memory_per_gpu_mib * task.gpu_need > task.memory_budget_mib:
                return False, f"task memory budget {task.memory_budget_mib} MiB exceeded"
        return True, "fits global quota, pool concurrency and memory budget"

    def _candidate_pools(self, shard: Shard) -> list[str]:
        task = self.decl.tasks[shard.task_id]
        pools = [pid for pid in sorted(self.decl.pools)
                 if task.memory_per_gpu_mib <= self.decl.pools[pid].memory_per_gpu_mib
                 and task.gpu_need <= self.decl.pools[pid].gpu_count]
        pools.sort(key=lambda pid: (self._pool_usage(pid)[0], pid))
        return pools

    def _victim_for(self, shard: Shard) -> TaskRuntime | None:
        """Pick the lowest-priority running task on a pool the shard can fit.

        Explicit total order: ascending priority, then most recent start
        (largest start_ms), then largest GPU footprint, then task id.
        """
        task = self.decl.tasks[shard.task_id]
        candidates: list[tuple[int, int, int, str]] = []
        fitting = set(self._candidate_pools(shard))
        for run in self.running.values():
            if run.pool_id not in fitting:
                continue
            victim_task = self.decl.tasks[run.task_id]
            if victim_task.priority >= task.priority:
                continue
            candidates.append((victim_task.priority, -run.start_ms,
                               -run.gpu_need, run.task_id))
        if not candidates:
            return None
        tid = sorted(candidates)[0][3]
        return self.runtime[tid]

    def _preempt(self, rt: TaskRuntime, decisions: list[dict[str, Any]]) -> None:
        victim_shards = [run.shard for run in self.running.values()
                         if run.task_id == rt.task_id]
        rt.suspended = True
        rt.preemption_count += 1
        for shard in victim_shards:
            shard.state = ShardState.SUSPENDED
            self._futures.pop(shard.shard_id, None)
            self.running.pop(shard.shard_id, None)
        decisions.append({
            "action": "preempt", "task_id": rt.task_id,
            "pool_id": rt.pool_id,
            "boundary_shard": max(s.global_id for s in victim_shards),
            "completed_preserved": sorted(rt.completed),
            "reason": "higher priority task blocked; suspend at shard boundary",
        })

    # ------------------------------------------------------------- transitions
    def _dispatch(self, shard: Shard, pool_id: str, decisions: list[dict[str, Any]]) -> None:
        task = self.decl.tasks[shard.task_id]
        pool = self.decl.pools[pool_id]
        shard.state = ShardState.RUNNING
        shard.pool_id = pool_id
        shard.start_ms = self.clock_ms
        shard.attempts += 1
        rt = self.runtime[shard.task_id]
        rt.suspended = False
        rt.pool_id = pool_id
        if rt.wait_start_ms is not None:
            shard.wait_start_ms = rt.wait_start_ms
            shard.wait_end_ms = self.clock_ms
            rt.wait_start_ms = None
        run = RunningShard(
            shard=shard, task_id=shard.task_id, pool_id=pool_id,
            gpu_need=task.gpu_need,
            mem_need=task.memory_per_gpu_mib * task.gpu_need,
            start_ms=self.clock_ms, remaining_ms=task.duration_ms)
        self.running[shard.shard_id] = run
        self._futures[shard.shard_id] = self.pool.submit(
            shard.shard_id, shard.task_id, shard.index, pool.compute_units)
        decisions.append({"action": "admit", "shard_id": shard.shard_id,
                          "pool_id": pool_id,
                          "basis": self._can_admit(shard, pool_id)[1]})

    def _finish_due(self, finish_ms: int,
                    decisions: list[dict[str, Any]]) -> list[str]:
        completed: list[str] = []
        for sid in sorted(self.running):
            run = self.running[sid]
            if run.remaining_ms > 0:
                continue
            result = self._futures.pop(sid).result()
            self._worker_seen.add(result.worker)
            shard = run.shard
            if sid in self.fail_inject:
                shard.state = ShardState.FAILED
                shard.finish_ms = finish_ms
                self.runtime[shard.task_id].failed = True
                decisions.append({"action": "shard_failed", "shard_id": sid,
                                  "pool_id": run.pool_id,
                                  "reason": "injected shard failure"})
                ordered, rationale = propagate_shard_invalidation(
                    self.decl, [sid])
                decisions.append({"action": "propagation",
                                  "affected_shards": ordered,
                                  "rationale": rationale})
                self.running.pop(sid)
                continue
            shard.state = ShardState.COMPLETED
            shard.result_digest = result.digest
            shard.finish_ms = finish_ms
            rt = self.runtime[shard.task_id]
            rt.completed.add(shard.index)
            if shard.index == rt.next_index:
                rt.next_index += 1
            while rt.next_index in rt.completed:
                rt.next_index += 1
            completed.append(sid)
            # The shard genuinely executed on a pool worker thread; only the
            # deterministic digest enters the ledger, not the thread name.
            self._worker_seen.add(result.worker)
            decisions.append({"action": "shard_completed", "shard_id": sid,
                              "pool_id": run.pool_id, "digest": result.digest,
                              "off_main_thread":
                                  result.worker != "MainThread"})
            self.running.pop(sid)
        return completed

    def run_round(self) -> dict[str, Any]:
        self.round_no += 1
        decisions: list[dict[str, Any]] = []
        start_ms = self.clock_ms

        completed = self._finish_due(start_ms, decisions)
        # A preempted task may resume only in a window where a shard just
        # finished and released resources; this prevents same-priority flapping.
        if completed:
            self._resume_suspended(decisions)

        ready_before = self._ready_shards()
        admitted: list[str] = []
        queued: list[dict[str, Any]] = []
        for shard in ready_before:
            rt = self.runtime[shard.task_id]
            fitting = self._candidate_pools(shard)
            chosen = next((pid for pid in fitting
                           if self._can_admit(shard, pid)[0]), None)
            if chosen is None and fitting:
                victim = self._victim_for(shard)
                if victim is not None:
                    self._preempt(victim, decisions)
                    chosen = next((pid for pid in fitting
                                   if self._can_admit(shard, pid)[0]), None)
            if chosen is None:
                if rt.wait_start_ms is None:
                    rt.wait_start_ms = start_ms
                reasons = sorted({self._can_admit(shard, pid)[1]
                                  for pid in fitting}) or ["no fitting pool"]
                shard.state = ShardState.QUEUED
                queued.append({"shard_id": shard.shard_id, "reasons": reasons})
                decisions.append({"action": "queued",
                                  "shard_id": shard.shard_id,
                                  "reasons": reasons})
                continue
            self._dispatch(shard, chosen, decisions)
            admitted.append(shard.shard_id)

        water = self.water_levels()
        for run in self.running.values():
            run.remaining_ms -= TICK_MS
        self.clock_ms += TICK_MS

        round_record = {
            "round": self.round_no,
            "clock_start_ms": start_ms,
            "clock_end_ms": self.clock_ms,
            "ready_set": [s.shard_id for s in ready_before],
            "admitted": admitted,
            "running_set": sorted(self.running.keys()),
            "queued": queued,
            "completed": completed,
            "decisions": decisions,
            "water_levels": water,
            "preemptions": [d for d in decisions if d.get("action") == "preempt"],
        }
        self.chain = chain_digest(self.chain, round_record)
        round_record["round_digest"] = self.chain
        self.rounds.append(round_record)
        if self.crash_after_round == self.round_no:
            self._simulate_crash(round_record)
        return round_record

    def _simulate_crash(self, round_record: dict[str, Any]) -> None:
        # The dispatched-but-uncommitted round is kept OUT of self.rounds and
        # surfaced via pending_crash_record, mirroring kill -9 between the
        # dispatch log and the atomic round commit.
        self.rounds.pop()
        self.round_no -= 1
        self.pending_crash_record = round_record
        for shard in self.decl.shards.values():
            if shard.state == ShardState.RUNNING:
                shard.state = ShardState.UNKNOWN
                self.unknown_shards.append({
                    "shard_id": shard.shard_id, "pool_id": shard.pool_id,
                    "reason": "dispatch intent logged but round never committed",
                })
        raise RuntimeError("E_STATE: simulated forced termination before commit")

    def is_done(self) -> bool:
        if self.running:
            return False
        if all(rt.failed or rt.next_index >= self.decl.tasks[tid].shard_count
               for tid, rt in self.runtime.items()):
            return True
        # Failure propagation can leave dependents permanently un-ready; once
        # nothing is running and nothing is ready, the session has reached its
        # terminal state (affected shards stay marked AFFECTED/FAILED).
        return not self._ready_shards()

    def run_all(self, max_ticks: int | None = None) -> list[dict[str, Any]]:
        limit = max_ticks or (
            sum(t.shard_count for t in self.decl.tasks.values()) * 200 + 100)
        guard = 0
        while not self.is_done():
            self.run_round()
            guard += 1
            if guard > limit:  # pragma: no cover - defensive
                raise RuntimeError("scheduler failed to converge")
        return self.rounds

    def shard_outcomes(self) -> dict[str, dict[str, Any]]:
        return {
            sid: {"state": shard.state.value,
                  "digest": shard.result_digest,
                  "attempts": shard.attempts}
            for sid, shard in sorted(self.decl.shards.items())
        }

    def declaration_digest(self) -> str:
        return digest_of({
            "pools": {pid: pool.__dict__ for pid, pool in self.decl.pools.items()},
            "tasks": {tid: task.__dict__ for tid, task in self.decl.tasks.items()},
            "params": self.params.__dict__,
            "shards": {sid: {"deps": shard.deps, "index": shard.index}
                       for sid, shard in self.decl.shards.items()},
        })
