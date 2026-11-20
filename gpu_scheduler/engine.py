"""Deterministic in-memory GPU quota and shard scheduling engine.

The engine is deliberately free of any persistence concern: given an identical
:class:`~gpu_scheduler.model.Declaration` it always produces the same sequence
of :class:`~gpu_scheduler.events.RoundRecord` objects, regardless of the order
in which shards happen to appear in the input.  Ordering always comes from
explicit integer/string keys, never from dictionary or hash iteration.

Scheduling proceeds in millisecond clock rounds.  Each round:

1. every shard whose finish time equals the current clock is finalized;
2. the stable priority queue is rebuilt from the ready shards;
3. at most one shard is dispatched, triggering at most one preemption when a
   strictly higher priority shard is blocked by lower priority work;
4. the clock advances to the next finish event when no dispatch is possible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from . import events as ev
from .errors import E_MEMORY, E_QUOTA, Problem, SchedulerError
from .graph import (
    ShardKey,
    build_global_graph,
    shard_key,
    topological_order,
    transitive_downstream,
)
from .model import (
    FAILED,
    INVALIDATED,
    Pool,
    PREEMPTED,
    READY,
    RUNNING,
    SUCCESS,
    UNKNOWN,
    WAITING,
    Declaration,
    Shard,
    Task,
    content_digest,
)


@dataclass
class _Runtime:
    declaration: Declaration
    graph: Dict[ShardKey, List[ShardKey]]
    clock: int = 0
    round_no: int = 0
    running: Dict[ShardKey, Tuple[int, str]] = field(default_factory=dict)
    wait_start: Dict[ShardKey, int] = field(default_factory=dict)
    total_wait_ms: Dict[ShardKey, int] = field(default_factory=dict)
    fail_remaining: Dict[ShardKey, int] = field(default_factory=dict)
    rounds: List[ev.RoundRecord] = field(default_factory=list)
    preemptions: List[ev.PreemptionRecord] = field(default_factory=list)

    @property
    def shards(self) -> Dict[ShardKey, Shard]:
        return self.declaration.shard_index()

    @property
    def pools(self) -> Dict[str, Pool]:
        return self.declaration.pool_map()

    @property
    def tasks(self) -> Dict[str, Task]:
        return self.declaration.task_map()


def pool_shares_overflow(pool: Pool,
                         running: Dict[ShardKey, Tuple[int, str]],
                         shards: Dict[ShardKey, Shard],
                         needed_shares: int) -> bool:
    used = 0
    for key, (_, pool_id) in running.items():
        if pool_id == pool.pool_id:
            used += shards[key].gpu_need * pool.shares_per_gpu
    return used + needed_shares > pool.quota_shares


def _success_digest(shard: Shard, finish_ms: int, pool_id: str,
                    attempt: int) -> str:
    payload = {
        "task_id": shard.task_id,
        "shard_index": shard.shard_index,
        "version": shard.version + 1,
        "finish_ms": finish_ms,
        "pool_id": pool_id,
        "attempt": attempt,
    }
    return content_digest(payload)


class SimulationEngine:
    """Pure, replayable scheduler used by both live runs and crash recovery."""

    def __init__(self, declaration: Declaration) -> None:
        self.declaration = declaration
        self.graph = build_global_graph(declaration)
        self.rt = _Runtime(declaration=declaration, graph=self.graph)
        for shard in declaration.all_shards():
            if shard.shard_index in self.rt.tasks[shard.task_id].fail_once_shards:
                self.rt.fail_remaining[shard_key(shard)] = 1
        # Upstream-first ordering drives deterministic invalidation.
        self.topo_order = topological_order(self.graph)

    # ------------------------------------------------------------------
    # Readiness and eligibility
    # ------------------------------------------------------------------
    def _upstreams_succeeded(self, shard: Shard) -> bool:
        index = self.rt.shards
        for upstream_key in self.graph[shard_key(shard)]:
            upstream = index[upstream_key]
            if upstream.state != SUCCESS:
                return False
        return True

    def _ready_shards(self) -> List[Shard]:
        ready: List[Shard] = []
        for key in self.topo_order:
            shard = self.rt.shards[key]
            if self.rt.tasks[shard.task_id].arrival_ms > self.rt.clock:
                continue
            if shard.state in (RUNNING, SUCCESS, FAILED):
                continue
            if shard.state == INVALIDATED:
                # Re-arm automatically once every upstream output is available;
                # successful sibling shards stay untouched.
                if not self._upstreams_succeeded(shard):
                    continue
                shard.state = READY
            if not self._upstreams_succeeded(shard):
                continue
            ready.append(shard)
        return ready

    def _task_memory_used(self, task_id: str, pool: Optional[Pool] = None,
                          extra_memory: int = 0) -> int:
        used = 0
        for key, (finish_ms, pool_id) in self.rt.running.items():
            if key[0] == task_id:
                shard = self.rt.shards[key]
                used += shard.total_memory_mib
        return used + extra_memory

    def _pool_usage(self) -> List[ev.PoolUsage]:
        usage: List[ev.PoolUsage] = []
        busy_gpu: Dict[str, int] = {pid: 0 for pid in self.rt.pools}
        busy_mem: Dict[str, int] = {pid: 0 for pid in self.rt.pools}
        busy_shares: Dict[str, int] = {pid: 0 for pid in self.rt.pools}
        busy_tasks: Dict[str, Set[str]] = {
            pid: set() for pid in self.rt.pools
        }
        for key, (finish_ms, pool_id) in self.rt.running.items():
            shard = self.rt.shards[key]
            pool = self.rt.pools[pool_id]
            busy_gpu[pool_id] += shard.gpu_need
            busy_mem[pool_id] += shard.total_memory_mib
            busy_shares[pool_id] += shard.gpu_need * pool.shares_per_gpu
            busy_tasks[pool_id].add(key[0])
        for pool_id in sorted(self.rt.pools):
            pool = self.rt.pools[pool_id]
            usage.append(ev.PoolUsage(
                pool_id=pool_id,
                gpu_busy=busy_gpu[pool_id],
                gpu_capacity=pool.gpu_count,
                memory_busy_mib=busy_mem[pool_id],
                memory_capacity_mib=pool.total_memory_mib,
                shares_busy=busy_shares[pool_id],
                shares_capacity=pool.quota_shares,
                concurrent_tasks=len(busy_tasks[pool_id]),
                concurrent_cap=pool.max_concurrent_tasks,
            ))
        return usage

    def _global_busy(self) -> Tuple[int, int]:
        gpu = 0
        shares = 0
        for key, (finish_ms, pool_id) in self.rt.running.items():
            shard = self.rt.shards[key]
            pool = self.rt.pools[pool_id]
            gpu += shard.gpu_need
            shares += shard.gpu_need * pool.shares_per_gpu
        return gpu, shares

    # ------------------------------------------------------------------
    # Stable queue and resource feasibility
    # ------------------------------------------------------------------
    def _queue_key(self, shard: Shard) -> Tuple:
        # Higher priority first, longer accumulated wait first, then stable ids.
        wait = self.rt.total_wait_ms.get(shard_key(shard), 0)
        return (
            -self.rt.tasks[shard.task_id].priority,
            -wait,
            shard.task_id,
            shard.shard_index,
        )

    def _feasible_pool(self, shard: Shard) -> Optional[Pool]:
        """First feasible pool under all three constraints, or None.

        Pools are scanned in ascending pool id so the choice never depends on
        dict ordering.  All arithmetic stays integer.
        """
        config = self.declaration.config
        task = self.rt.tasks[shard.task_id]
        global_gpu, global_shares = self._global_busy()
        for pool_id in sorted(self.rt.pools):
            pool = self.rt.pools[pool_id]
            if shard.gpu_need > pool.gpu_count:
                continue
            if shard.total_memory_mib > pool.total_memory_mib:
                continue
            # Per-pool GPU headroom.
            pool_gpu = sum(
                self.rt.shards[key].gpu_need
                for key, (_, pid) in self.rt.running.items() if pid == pool_id)
            if pool_gpu + shard.gpu_need > pool.gpu_count:
                continue
            # Per-pool memory headroom.
            pool_mem = sum(
                self.rt.shards[key].total_memory_mib
                for key, (_, pid) in self.rt.running.items() if pid == pool_id)
            if pool_mem + shard.total_memory_mib > pool.total_memory_mib:
                continue
            # Per-pool shares headroom.
            pool_shares = sum(
                self.rt.shards[key].gpu_need * pool.shares_per_gpu
                for key, (_, pid) in self.rt.running.items() if pid == pool_id)
            needed_shares = shard.gpu_need * pool.shares_per_gpu
            if pool_shares + needed_shares > pool.quota_shares:
                continue
            # Per-pool concurrent task cap.
            active_tasks = {key[0] for key, (_, pid) in self.rt.running.items()
                            if pid == pool_id}
            if shard.task_id not in active_tasks and \
                    len(active_tasks) >= pool.max_concurrent_tasks:
                continue
            # Per-task memory budget across all running shards of the task.
            if self._task_memory_used(shard.task_id) + shard.total_memory_mib \
                    > task.memory_budget_mib:
                continue
            # Global GPU cap and global quota shares.
            if global_gpu + shard.gpu_need > config.global_gpu_cap:
                continue
            if global_shares + needed_shares > config.total_quota_shares:
                continue
            return pool
        return None

    def _blocking_victims(self, shard: Shard) -> List[Tuple[Shard, Pool, str]]:
        """Running shards that are strictly lower priority and could be removed.

        Returns candidate ``(victim_shard, pool, reason)`` tuples in the stable
        victim ordering.  Only shards belonging to tasks with a strictly lower
        integer priority are eligible, which guarantees the non-decreasing
        priority rule for every preemption.
        """
        trigger_priority = self.rt.tasks[shard.task_id].priority
        candidates: List[Tuple[Shard, Pool, str]] = []
        for key in sorted(self.rt.running):
            victim = self.rt.shards[key]
            victim_priority = self.rt.tasks[victim.task_id].priority
            if victim_priority >= trigger_priority:
                continue
            _, pool_id = self.rt.running[key]
            pool = self.rt.pools[pool_id]
            reason = self._removal_helps(shard, victim, pool)
            if reason:
                candidates.append((victim, pool, reason))
        # Explicit, deterministic victim rule: lowest priority first, then the
        # earliest dispatch (smallest start), then smallest shard identifier.
        candidates.sort(key=lambda item: (
            self.rt.tasks[item[0].task_id].priority,
            item[0].started_ms if item[0].started_ms is not None else 0,
            item[0].task_id,
            item[0].shard_index,
        ))
        return candidates

    def _removal_helps(self, shard: Shard, victim: Shard,
                       pool: Pool) -> Optional[str]:
        """Identify which constraint the victim's removal would relieve."""
        config = self.declaration.config
        task = self.rt.tasks[shard.task_id]
        pool_id = pool.pool_id
        pool_gpu = sum(
            self.rt.shards[key].gpu_need
            for key, (_, pid) in self.rt.running.items() if pid == pool_id)
        pool_mem = sum(
            self.rt.shards[key].total_memory_mib
            for key, (_, pid) in self.rt.running.items() if pid == pool_id)
        active_tasks = {key[0] for key, (_, pid) in self.rt.running.items()
                        if pid == pool_id}
        global_gpu, global_shares = self._global_busy()
        needed_shares = shard.gpu_need * pool.shares_per_gpu
        victim_shares = victim.gpu_need * pool.shares_per_gpu

        if shard.gpu_need <= pool.gpu_count and \
                pool_gpu - victim.gpu_need + shard.gpu_need <= pool.gpu_count \
                and pool_mem - victim.total_memory_mib + shard.total_memory_mib \
                <= pool.total_memory_mib \
                and (shard.task_id in active_tasks or
                     len(active_tasks - {victim.task_id}) <
                     pool.max_concurrent_tasks) \
                and self._task_memory_used(shard.task_id) + \
                shard.total_memory_mib <= task.memory_budget_mib \
                and global_gpu - victim.gpu_need + shard.gpu_need \
                <= config.global_gpu_cap \
                and global_shares - victim_shares + needed_shares \
                <= config.total_quota_shares:
            if pool_gpu + shard.gpu_need > pool.gpu_count or \
                    global_gpu + shard.gpu_need > config.global_gpu_cap:
                return "GPU"
            if pool_mem + shard.total_memory_mib > pool.total_memory_mib:
                return "MEMORY"
            if global_shares + needed_shares > config.total_quota_shares or \
                    pool_shares_overflow(pool, self.rt.running, self.rt.shards,
                                         needed_shares):
                return "QUOTA"
            return "CONCURRENCY"
        return None

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _dispatch(self, shard: Shard, pool: Pool,
                  trace: List[ev.TraceEvent]) -> None:
        key = shard_key(shard)
        shard.attempts += 1
        shard.state = RUNNING
        shard.pool_id = pool.pool_id
        shard.started_ms = self.rt.clock
        shard.finish_ms = self.rt.clock + shard.duration_ms
        if key in self.rt.wait_start:
            waited = self.rt.clock - self.rt.wait_start.pop(key)
            shard.wait_duration_ms += waited
        self.rt.running[key] = (shard.finish_ms, pool.pool_id)
        trace.append(ev.TraceEvent(
            round_no=self.rt.round_no, clock_ms=self.rt.clock,
            kind=ev.DISPATCH, task_id=shard.task_id,
            shard_index=shard.shard_index, pool_id=pool.pool_id,
            gpu_count=shard.gpu_need, memory_mib=shard.total_memory_mib,
            detail=f"dispatch attempt={shard.attempts}"))

    def _finalize_due(self, trace: List[ev.TraceEvent]) -> None:
        due = sorted(key for key, (finish, _) in self.rt.running.items()
                     if finish == self.rt.clock)
        for key in due:
            shard = self.rt.shards[key]
            _, pool_id = self.rt.running.pop(key)
            will_fail = self.rt.fail_remaining.get(key, 0) > 0
            if will_fail:
                self.rt.fail_remaining[key] -= 1
                self._fail_shard(shard, pool_id, trace)
            else:
                shard.state = SUCCESS
                shard.version += 1
                shard.ended_ms = self.rt.clock
                shard.output_digest = _success_digest(
                    shard, self.rt.clock, pool_id, shard.attempts)
                trace.append(ev.TraceEvent(
                    round_no=self.rt.round_no, clock_ms=self.rt.clock,
                    kind=ev.COMPLETE, task_id=shard.task_id,
                    shard_index=shard.shard_index, pool_id=pool_id,
                    gpu_count=shard.gpu_need,
                    memory_mib=shard.total_memory_mib,
                    detail=f"success version={shard.version}"))

    def _fail_shard(self, shard: Shard, pool_id: str,
                    trace: List[ev.TraceEvent]) -> None:
        shard.state = FAILED
        shard.ended_ms = self.rt.clock
        shard.output_digest = None
        trace.append(ev.TraceEvent(
            round_no=self.rt.round_no, clock_ms=self.rt.clock,
            kind=ev.FAIL, task_id=shard.task_id,
            shard_index=shard.shard_index, pool_id=pool_id,
            gpu_count=shard.gpu_need, memory_mib=shard.total_memory_mib,
            detail="injected deterministic failure"))
        self._invalidate_downstream([shard_key(shard)], trace)

    def _invalidate_downstream(self, origins: List[ShardKey],
                               trace: List[ev.TraceEvent]) -> None:
        origin_set = set(origins)
        affected = [key for key in transitive_downstream(self.graph, origins)
                    if key not in origin_set]
        for key in affected:
            shard = self.rt.shards[key]
            if shard.state == RUNNING:
                # A still-running downstream shard is suspended and must rerun.
                self.rt.running.pop(key, None)
            shard.state = INVALIDATED
            shard.output_digest = None
            shard.pool_id = None
            shard.started_ms = None
            shard.finish_ms = None
            trace.append(ev.TraceEvent(
                round_no=self.rt.round_no, clock_ms=self.rt.clock,
                kind=ev.INVALIDATE, task_id=key[0], shard_index=key[1],
                pool_id=None, gpu_count=0, memory_mib=0,
                detail="invalidated by upstream outcome change"))

    def _preempt(self, shard: Shard, victim: Shard, pool: Pool,
                 reason: str, trace: List[ev.TraceEvent]) -> ev.PreemptionRecord:
        victim_key = shard_key(victim)
        finish, victim_pool_id = self.rt.running.pop(victim_key)
        victim.state = PREEMPTED
        victim.ended_ms = self.rt.clock
        victim.finish_ms = None
        # Completed shard progress (digests/versions) is untouched; only the
        # in-flight shard is suspended at this boundary.
        trace.append(ev.TraceEvent(
            round_no=self.rt.round_no, clock_ms=self.rt.clock,
            kind=ev.PREEMPT, task_id=victim.task_id,
            shard_index=victim.shard_index, pool_id=victim_pool_id,
            gpu_count=victim.gpu_need, memory_mib=victim.total_memory_mib,
            detail=f"suspended for {shard.task_id} reason={reason}"))
        record = ev.PreemptionRecord(
            round_no=self.rt.round_no, clock_ms=self.rt.clock,
            trigger_task_id=shard.task_id,
            trigger_shard_index=shard.shard_index,
            victim_task_id=victim.task_id,
            victim_shard_index=victim.shard_index,
            pool_id=victim_pool_id,
            released_gpu=victim.gpu_need,
            released_memory_mib=victim.total_memory_mib,
            trigger_priority=self.rt.tasks[shard.task_id].priority,
            victim_priority=self.rt.tasks[victim.task_id].priority,
        )
        self.rt.preemptions.append(record)
        return record

    # ------------------------------------------------------------------
    # Main scheduling loop
    # ------------------------------------------------------------------
    def _mark_waiting(self, shard: Shard, trace: List[ev.TraceEvent]) -> None:
        key = shard_key(shard)
        if key not in self.rt.wait_start:
            self.rt.wait_start[key] = self.rt.clock
            trace.append(ev.TraceEvent(
                round_no=self.rt.round_no, clock_ms=self.rt.clock,
                kind=ev.ENQUEUE, task_id=shard.task_id,
                shard_index=shard.shard_index, pool_id=None,
                gpu_count=0, memory_mib=0,
                detail="blocked by quota/memory; queued"))
        if shard.state != WAITING:
            shard.state = WAITING

    def _build_round(self, trace: List[ev.TraceEvent],
                     preemption: Optional[ev.PreemptionRecord]) -> ev.RoundRecord:
        waiting = sorted(self.rt.wait_start,
                         key=lambda key: self._queue_key(self.rt.shards[key]))
        running = sorted(self.rt.running)
        global_gpu, global_shares = self._global_busy()
        return ev.RoundRecord(
            round_no=self.rt.round_no,
            clock_ms=self.rt.clock,
            events=list(trace),
            preemption=preemption,
            pool_usage=self._pool_usage(),
            running=[f"{task}#S{index:02d}" for task, index in running],
            waiting=[f"{task}#S{index:02d}" for task, index in waiting],
            global_gpu_busy=global_gpu,
            global_shares_busy=global_shares,
        )

    def run_until(self, max_rounds: Optional[int] = None) -> List[ev.RoundRecord]:
        """Advance the simulation until every shard reaches a terminal state."""
        guard = 0
        hard_limit = max_rounds if max_rounds is not None else 10_000_000
        while not self._all_terminal() and guard < hard_limit:
            guard += 1
            self.rt.round_no += 1
            trace: List[ev.TraceEvent] = []
            preemption: Optional[ev.PreemptionRecord] = None
            self._finalize_due(trace)
            preemption = self._schedule_pass(trace)
            record = self._build_round(trace, preemption)
            self.rt.rounds.append(record)
            if not self._advance_clock(trace):
                break
        return list(self.rt.rounds)

    def _schedule_pass(self,
                       trace: List[ev.TraceEvent]) -> Optional[ev.PreemptionRecord]:
        """Dispatch every currently feasible shard, then at most one preemption.

        Recomputing the ready set after every dispatch makes the round fill the
        machine with parallel shards while preserving deterministic ordering.
        A round performs at most one preemption so the freed resources are
        re-evaluated in a fresh round and the preemption sequence is stable.
        """
        made_preemption: Optional[ev.PreemptionRecord] = None
        while True:
            ready = sorted(self._ready_shards(), key=self._queue_key)
            dispatched = False
            for shard in ready:
                pool = self._feasible_pool(shard)
                if pool is not None:
                    self._dispatch(shard, pool, trace)
                    dispatched = True
                    break
                self._mark_waiting(shard, trace)
            if dispatched:
                continue
            # Nothing left is immediately feasible: try one preemption for the
            # highest-priority blocked shard, otherwise end the round.
            blocked = sorted(self._ready_shards(), key=self._queue_key)
            for shard in blocked:
                victims = self._blocking_victims(shard)
                if victims:
                    victim, victim_pool, reason = victims[0]
                    made_preemption = self._preempt(
                        shard, victim, victim_pool, reason, trace)
                    return made_preemption
            return made_preemption

    def _advance_clock(self, trace: List[ev.TraceEvent]) -> bool:
        if not self.rt.running:
            # Nothing running: if work remains but is permanently blocked by a
            # failure graph, the terminal check ends the run; otherwise this is
            # an infeasible declaration.
            if self._all_terminal():
                return False
            ready = self._ready_shards()
            if not ready:
                return False
            # A recoverable shard should still be dispatchable; if none is
            # feasible, surface a precise business error rather than spin.
            head = sorted(ready, key=self._queue_key)[0]
            if self._feasible_pool(head) is None and not \
                    self._blocking_victims(head):
                raise SchedulerError(
                    E_MEMORY if self._memory_bound(head) else E_QUOTA,
                    "scheduler deadlocked under declared constraints",
                    [Problem(f"task:{head.task_id}.shard:{head.shard_index}",
                             "feasible resource under triple constraints",
                             "no feasible pool")])
            return True
        next_finish = min(finish for finish, _ in self.rt.running.values())
        self.rt.clock = next_finish
        return True

    def _memory_bound(self, shard: Shard) -> bool:
        task = self.rt.tasks[shard.task_id]
        return shard.total_memory_mib > task.memory_budget_mib or all(
            shard.total_memory_mib > pool.total_memory_mib
            for pool in self.rt.pools.values())

    def _all_terminal(self) -> bool:
        for shard in self.declaration.all_shards():
            if shard.state not in (SUCCESS, FAILED, INVALIDATED):
                return False
        return True

    # ------------------------------------------------------------------
    # Recovery helpers used after a hard kill
    # ------------------------------------------------------------------
    def mark_unknown_uncommitted(self,
                                 committed_shard_ids: Set[ShardKey],
                                 trace: List[ev.TraceEvent]) -> List[ShardKey]:
        """Shards left RUNNING without a committed round become UNKNOWN.

        Unknown shards must be rerun; no exit report is fabricated for them.
        Already successful shards keep their digest/version and are untouched.
        """
        unknown: List[ShardKey] = []
        for key in sorted(self.rt.running):
            if key not in committed_shard_ids:
                shard = self.rt.shards[key]
                shard.state = UNKNOWN
                shard.finish_ms = None
                shard.pool_id = None
                shard.started_ms = None
                unknown.append(key)
                trace.append(ev.TraceEvent(
                    round_no=self.rt.round_no, clock_ms=self.rt.clock,
                    kind=ev.MARK_UNKNOWN, task_id=key[0],
                    shard_index=key[1], pool_id=None, gpu_count=0,
                    memory_mib=0,
                    detail="uncommitted after forced termination; rerun"))
        for key in unknown:
            self.rt.running.pop(key, None)
        return unknown

    def rearm_recoverable(self) -> None:
        """Convert UNKNOWN/PREEMPTED/INVALIDATED shards back to READY/WAITING.

        Successful shards are left intact so recovery only recomputes the lost
        or invalidated shards, preserving output digests and versions.
        """
        for key in self.topo_order:
            shard = self.rt.shards[key]
            if shard.state in (UNKNOWN, PREEMPTED):
                shard.state = READY
                shard.finish_ms = None
                shard.pool_id = None
                shard.started_ms = None
            elif shard.state == INVALIDATED and self._upstreams_succeeded(shard):
                shard.state = READY

    def shard_summary(self) -> List[dict]:
        """Stable per-shard result used by snapshots, stats and the UI."""
        summary: List[dict] = []
        for key in self.topo_order:
            shard = self.rt.shards[key]
            summary.append({
                "task_id": shard.task_id,
                "shard_index": shard.shard_index,
                "state": shard.state,
                "version": shard.version,
                "output_digest": shard.output_digest,
                "pool_id": shard.pool_id,
                "started_ms": shard.started_ms,
                "finish_ms": shard.finish_ms,
                "ended_ms": shard.ended_ms,
                "wait_duration_ms": shard.wait_duration_ms,
                "attempts": shard.attempts,
            })
        return summary

    def persistence_snapshot(self) -> List[dict]:
        """Full mutable shard state for atomic ledger persistence."""
        rows = []
        for key in self.topo_order:
            shard = self.rt.shards[key]
            rows.append({
                "task_id": shard.task_id,
                "shard_index": shard.shard_index,
                "state": shard.state,
                "version": shard.version,
                "output_digest": shard.output_digest,
                "pool_id": shard.pool_id,
                "started_ms": shard.started_ms,
                "finish_ms": shard.finish_ms,
                "ended_ms": shard.ended_ms,
                "wait_duration_ms": shard.wait_duration_ms,
                "attempts": shard.attempts,
            })
        return rows
