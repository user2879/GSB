"""Deterministic lock/version engine projected onto real worker threads.

The lock arbiter advances in deterministic rounds. Transaction work is submitted
to a ThreadPoolExecutor and every granted operation is acknowledged by its real
worker thread. Wall-clock time is used only for timeout/budget enforcement and is
persisted separately from the byte-stable canonical trajectory.
"""

from __future__ import annotations

import os
import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_text, digest
from .model import Operation, RuntimeConfig, Spec, TransactionSpec


READ = "read"
WRITE = "write"


@dataclass
class Version:
    row: str
    version_no: int
    value: Any
    txid: str | None
    status: str
    created_round: int
    committed_round: int | None = None

    def public(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "version": self.version_no,
            "value": self.value,
            "txid": self.txid,
            "status": self.status,
            "created_round": self.created_round,
            "committed_round": self.committed_round,
        }


@dataclass
class TxRuntime:
    spec: TransactionSpec
    status: str = "pending"
    next_index: int = 0
    fixed_reads: dict[str, int] = field(default_factory=dict)
    owned_versions: list[Version] = field(default_factory=list)
    read_events: list[dict[str, Any]] = field(default_factory=list)
    admitted_round: int | None = None
    quota_wait_start_round: int | None = None
    quota_wait_start_ns: int | None = None
    completed_round: int | None = None
    abort_reason: str | None = None
    step_done_round: int | None = None
    operations_done: int = 0
    command_q: queue.Queue[dict[str, Any] | None] = field(default_factory=queue.Queue)
    future: Future[Any] | None = None
    worker_closed: bool = False
    worker_name: str | None = None

    @property
    def txid(self) -> str:
        return self.spec.txid

    @property
    def priority(self) -> tuple[int, str]:
        return self.spec.priority

    def next_operation(self) -> Operation | None:
        if self.next_index >= len(self.spec.operations):
            return None
        return self.spec.operations[self.next_index]


@dataclass
class LockWait:
    txid: str
    row: str
    modes: tuple[str, ...]
    mode_index: int
    start_round: int
    start_ns: int
    limit_ms: int
    op: Operation
    priority: tuple[int, str]


@dataclass
class LockDecision:
    granted: bool
    acquired: list[tuple[str, str]] = field(default_factory=list)
    wait: LockWait | None = None
    blockers: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    queue_ahead: list[str] = field(default_factory=list)


@dataclass
class WaitGrant:
    wait: LockWait
    acquired: list[tuple[str, str]]
    complete: bool
    wait_ms: int


@dataclass
class ExecutionResult:
    status: str
    path: str
    rounds: list[dict[str, Any]]
    observed_rounds: list[dict[str, Any]]
    result: dict[str, Any]
    isolation_conclusions: list[dict[str, Any]]
    deadlocks: list[dict[str, Any]]
    timeouts: list[dict[str, Any]]
    worker_threads: list[str]
    range_plan_rounds: list[dict[str, Any]]
    stale_rounds: list[int] = field(default_factory=list)


class LockManager:
    def __init__(self, spec: Spec, path: str, priorities: dict[str, tuple[int, str]],
                 logical_tick_ms: int) -> None:
        self.spec = spec
        self.path = path
        self.priorities = priorities
        self.tick = logical_tick_ms
        self.row_index = spec.row_index
        self.holders: dict[str, dict[str, set[str]]] = {row: {} for row in spec.row_keys}
        self.waiters: dict[str, list[LockWait]] = {row: [] for row in spec.row_keys}
        self._intervals: list[dict[str, Any]] = []
        self._recompute_intervals()

    # ---- path/range representation -------------------------------------------------
    def _recompute_intervals(self) -> None:
        intervals: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        previous_idx: int | None = None
        for idx, row in enumerate(self.spec.row_keys):
            holder_map = self.holders[row]
            signature = tuple(
                (txid, tuple(sorted(modes))) for txid, modes in sorted(holder_map.items())
                if modes
            )
            if not signature:
                current = None
                previous_idx = None
                continue
            if (current is not None and previous_idx == idx - 1
                    and current["signature"] == signature):
                current["end"] = idx
                current["rows"].append(row)
            else:
                current = {
                    "start": idx,
                    "end": idx,
                    "rows": [row],
                    "signature": signature,
                    "holders": {
                        txid: sorted(modes) for txid, modes in sorted(holder_map.items())
                        if modes
                    },
                }
                intervals.append(current)
            previous_idx = idx
        self._intervals = intervals

    def range_plan(self) -> list[dict[str, Any]]:
        return [
            {
                "start": item["start"],
                "end": item["end"],
                "rows": list(item["rows"]),
                "holders": {txid: list(modes) for txid, modes in item["holders"].items()},
            }
            for item in self._intervals
        ]

    def _holder_map(self, row: str) -> dict[str, set[str]]:
        if self.path == "row":
            return self.holders[row]
        idx = self.row_index[row]
        for interval in self._intervals:
            if interval["start"] <= idx <= interval["end"]:
                return {
                    txid: set(modes)
                    for txid, modes in interval["holders"].items()
                }
        return {}

    # ---- compatibility/queue -------------------------------------------------------
    def _has_mode(self, txid: str, row: str, mode: str) -> bool:
        modes = self.holders[row].get(txid, set())
        return mode in modes or (mode == "S" and "X" in modes)

    def blockers(self, txid: str, row: str, mode: str) -> list[tuple[str, tuple[str, ...]]]:
        out: list[tuple[str, tuple[str, ...]]] = []
        for holder, modes in sorted(self._holder_map(row).items()):
            if holder == txid or not modes:
                continue
            if mode == "S" and "X" in modes:
                out.append((holder, tuple(sorted(modes))))
            elif mode == "X":
                out.append((holder, tuple(sorted(modes))))
        return out

    def _can_acquire(self, txid: str, row: str, mode: str) -> bool:
        if self._has_mode(txid, row, mode):
            return True
        return not self.blockers(txid, row, mode)

    def _acquire_mode(self, txid: str, row: str, mode: str) -> None:
        self.holders[row].setdefault(txid, set()).add(mode)
        self._recompute_intervals()

    def _waiters_sorted(self, row: str) -> list[LockWait]:
        return sorted(self.waiters[row], key=lambda wait: wait.priority)

    def primary_waiter(self, row: str) -> LockWait | None:
        ordered = self._waiters_sorted(row)
        return ordered[0] if ordered else None

    def waiting_rows(self) -> list[str]:
        return [row for row in self.spec.row_keys if self.waiters[row]]

    def all_waits(self) -> list[LockWait]:
        return [
            wait
            for row in self.spec.row_keys
            for wait in self._waiters_sorted(row)
        ]

    def request(self, txid: str, row: str, modes: list[str], op: Operation, *,
                round_no: int, limit_ms: int) -> LockDecision:
        acquired: list[tuple[str, str]] = []
        priority = self.priorities[txid]
        if all(self._has_mode(txid, row, mode) for mode in modes):
            return LockDecision(True, acquired, None, [], [])
        existing_ahead = [
            wait.txid for wait in self._waiters_sorted(row) if wait.priority < priority
        ]
        if self.waiters[row]:
            wait = LockWait(txid, row, tuple(modes), 0, round_no, time.monotonic_ns(),
                            limit_ms, op, priority)
            self.waiters[row].append(wait)
            return LockDecision(False, acquired, wait, [], existing_ahead)

        mode_index = 0
        while mode_index < len(modes):
            mode = modes[mode_index]
            if self._has_mode(txid, row, mode):
                mode_index += 1
                continue
            if not self._can_acquire(txid, row, mode):
                blockers = self.blockers(txid, row, mode)
                wait = LockWait(txid, row, tuple(modes), mode_index, round_no,
                                time.monotonic_ns(), limit_ms, op, priority)
                self.waiters[row].append(wait)
                return LockDecision(False, acquired, wait, blockers, [])
            self._acquire_mode(txid, row, mode)
            acquired.append((row, mode))
            mode_index += 1
        return LockDecision(True, acquired, None, [], [])

    def advance_head(self, row: str, round_no: int) -> WaitGrant | None:
        head = self.primary_waiter(row)
        if head is None:
            return None
        acquired: list[tuple[str, str]] = []
        while head.mode_index < len(head.modes):
            mode = head.modes[head.mode_index]
            if self._has_mode(head.txid, row, mode):
                head.mode_index += 1
                continue
            if not self._can_acquire(head.txid, row, mode):
                if acquired:
                    return WaitGrant(head, acquired, False, 0)
                return None
            self._acquire_mode(head.txid, row, mode)
            acquired.append((row, mode))
            head.mode_index += 1
        self.waiters[row].remove(head)
        wait_ms = max(self.tick, (round_no - head.start_round) * self.tick)
        return WaitGrant(head, acquired, True, wait_ms)

    def release_mode(self, txid: str, row: str, mode: str) -> bool:
        modes = self.holders[row].get(txid)
        if not modes or mode not in modes or "X" in modes:
            return False
        modes.remove(mode)
        if not modes:
            self.holders[row].pop(txid, None)
        self._recompute_intervals()
        return True

    def release_all(self, txid: str) -> list[tuple[str, str]]:
        released: list[tuple[str, str]] = []
        for row in self.spec.row_keys:
            modes = self.holders[row].pop(txid, None)
            if modes:
                for mode in sorted(modes):
                    released.append((row, mode))
        self.remove_waits(txid)
        self._recompute_intervals()
        return released

    def remove_waits(self, txid: str) -> list[LockWait]:
        removed: list[LockWait] = []
        for row in self.spec.row_keys:
            kept: list[LockWait] = []
            for wait in self.waiters[row]:
                if wait.txid == txid:
                    removed.append(wait)
                else:
                    kept.append(wait)
            self.waiters[row] = kept
        return removed

    def wait_graph(self) -> dict[str, Any]:
        nodes = sorted(self.priorities, key=lambda txid: self.priorities[txid])
        edges: set[tuple[str, str]] = set()
        for wait in self.all_waits():
            for blocker, _modes in self.blockers(wait.txid, wait.row, wait.modes[wait.mode_index]):
                if blocker != wait.txid:
                    edges.add((wait.txid, blocker))
        edge_list = [list(edge) for edge in sorted(edges)]
        cycles = find_wait_cycles(edge_list, self.priorities)
        return {
            "nodes": nodes,
            "edges": edge_list,
            "cycles": cycles,
        }

    def holder_snapshot(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self.spec.row_keys:
            for txid, modes in sorted(self.holders[row].items()):
                if modes:
                    out.append({"row": row, "txid": txid, "modes": sorted(modes)})
        return out


def _canonical_cycle(cycle: Iterable[str], priorities: dict[str, tuple[int, str]]) -> tuple[str, ...]:
    values = list(cycle)
    if not values:
        return ()
    smallest = min(range(len(values)), key=lambda idx: priorities[values[idx]])
    rotated = tuple(values[smallest:] + values[:smallest])
    return rotated


def find_wait_cycles(edges: Iterable[Iterable[str]], priorities: dict[str, tuple[int, str]]) -> list[list[str]]:
    adjacency: dict[str, list[str]] = {txid: [] for txid in priorities}
    for raw in edges:
        source, target = list(raw)
        adjacency.setdefault(source, []).append(target)
        adjacency.setdefault(target, [])
    for txid in adjacency:
        adjacency[txid] = sorted(set(adjacency[txid]), key=lambda t: priorities.get(t, (0, t)))
    nodes = sorted(adjacency, key=lambda t: priorities.get(t, (0, t)))
    found: set[tuple[str, ...]] = set()

    def dfs(start: str, node: str, stack: list[str], in_stack: set[str]) -> None:
        for nxt in adjacency.get(node, []):
            if nxt == start and len(stack) >= 2:
                found.add(_canonical_cycle(stack, priorities))
                continue
            if nxt in in_stack:
                continue
            if priorities.get(nxt, (0, nxt)) < priorities.get(start, (0, start)):
                continue
            stack.append(nxt)
            in_stack.add(nxt)
            dfs(start, nxt, stack, in_stack)
            in_stack.remove(nxt)
            stack.pop()

    for start in nodes:
        dfs(start, start, [start], {start})
    return [list(cycle) for cycle in sorted(found, key=lambda c: tuple(priorities[t] for t in c))]


class Engine:
    def __init__(self, spec: Spec, config: RuntimeConfig, path: str,
                 hooks: Any | None = None) -> None:
        self.spec = spec
        self.config = config
        self.path = path
        self.hooks = hooks
        self.tick = config.logical_tick_ms
        self.logical_ms = 0
        self.round_no = 0
        self.priorities = {tx.txid: tx.priority for tx in spec.transactions}
        self.txs: dict[str, TxRuntime] = {
            tx.txid: TxRuntime(tx) for tx in spec.transactions
        }
        self.lock_manager = LockManager(spec, path, self.priorities, self.tick)
        self.versions: dict[str, list[Version]] = {}
        self.next_version: dict[str, int] = {}
        for row in spec.row_keys:
            initial = spec.initial_versions[row]
            self.versions[row] = [Version(row, initial.version_no, initial.value, None,
                                          initial.status, 0,
                                          0 if initial.status == "committed" else None)]
            self.next_version[row] = initial.version_no + 1
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, config.max_running),
            thread_name_prefix="txdiag",
        )
        self._ack_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._worker_names: set[str] = set()
        self._rounds: list[dict[str, Any]] = []
        self._observed_rounds: list[dict[str, Any]] = []
        self._range_plan_rounds: list[dict[str, Any]] = []
        self._deadlocks: list[dict[str, Any]] = []
        self._timeouts: list[dict[str, Any]] = []
        self._wait_budget_used = 0
        self._actual_wait_spent_ms = 0.0
        self._hold_tx, self._hold_after = self._parse_hold()
        self._crash_after = self._parse_crash()
        self._crash_in_round = self._parse_crash_in_round()
        self._inject_diverge = "diverge" in config.inject
        self._injection_unlocked = False
        self._session_start_ns = time.monotonic_ns()
        self._idle_gate = threading.Event()
        self._abort_during_shutdown = False

    # ---- injection/config parsing --------------------------------------------------
    def _parse_crash(self) -> int | None:
        for item in self.config.inject:
            if item.startswith("crash-after-round="):
                try:
                    return int(item.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _parse_crash_in_round(self) -> int | None:
        for item in self.config.inject:
            if item.startswith("crash-in-round="):
                try:
                    return int(item.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _parse_hold(self) -> tuple[str | None, int]:
        for item in self.config.inject:
            if item.startswith("hold-lock="):
                value = item.split("=", 1)[1]
                txid, _, after = value.partition(":")
                try:
                    count = int(after) if after else 1
                except ValueError:
                    count = 1
                return txid, count
        return None, 1

    @staticmethod
    def _actual_ms() -> float:
        return round(time.monotonic_ns() / 1_000_000.0, 6)

    # ---- public ----------------------------------------------------------------
    def execute(self) -> ExecutionResult:
        try:
            self._loop()
            self._close_all_workers()
            self.executor.shutdown(wait=True)
            result = self._final_result()
            if self.hooks is not None and hasattr(self.hooks, "complete"):
                self.hooks.complete(result)
            return result
        except BaseException:
            self._abort_during_shutdown = True
            self._close_all_workers()
            self.executor.shutdown(wait=False, cancel_futures=True)
            raise

    def _loop(self) -> None:
        while self._has_work():
            self.round_no += 1
            self.logical_ms = self.round_no * self.tick
            started = self._actual_ms()
            events: list[dict[str, Any]] = []
            observed: dict[str, Any] = {
                "round": self.round_no,
                "started_monotonic_ms": started,
                "lock_observations": [],
                "wait_observations": [],
                "quota_observations": [],
                "worker_acks": [],
            }
            self._events = events
            self._observed = observed
            commands: list[tuple[TxRuntime, dict[str, Any]]] = []

            self._admit_round(events, observed)
            running_at_start = self._running_txids()
            if self.hooks is not None and hasattr(self.hooks, "begin_round"):
                self.hooks.begin_round(self.round_no, running_at_start)
            if (self._crash_in_round is not None
                    and self.round_no >= self._crash_in_round):
                if self.hooks is not None and hasattr(self.hooks, "flush"):
                    self.hooks.flush()
                os._exit(70)

            self._commit_phase(events)
            self._service_waiters(events, commands)

            for tx in self._ordered_running():
                if (tx.status == "ready"
                        and tx.step_done_round != self.round_no
                        and not (self._hold_tx == tx.txid
                                 and tx.operations_done >= self._hold_after
                                 and not self._injection_unlocked)):
                    self._process_operation(tx, events, commands)

            self._resolve_deadlocks(events, commands)
            self._resolve_timeouts(events, commands)
            self._maybe_stall_until_timeout(events, commands)
            self._resolve_deadlocks(events, commands)
            self._resolve_timeouts(events, commands)
            self._service_waiters(events, commands)

            if commands:
                self._dispatch_commands(commands, observed)

            self._close_finished_workers()
            self._unlock_injection_if_done()

            wait_graph = self.lock_manager.wait_graph()
            range_plan = self.lock_manager.range_plan()
            payload = {
                "round": self.round_no,
                "logical_ms": self.logical_ms,
                "running": running_at_start,
                "quota": self._quota_snapshot(running_at_start),
                "admission_queue": self._admission_queue_snapshot(),
                "events": events,
                "wait_graph": wait_graph,
                "locks": self.lock_manager.holder_snapshot(),
                "waiters": [
                    {
                        "txid": wait.txid,
                        "row": wait.row,
                        "mode": wait.modes[wait.mode_index],
                    }
                    for wait in sorted(
                        self.lock_manager.all_waits(),
                        key=lambda item: (item.priority, item.row, item.mode_index),
                    )
                ],
                "wait_budget": {
                    "limit_ms": self.config.wait_budget_ms,
                    "used_ms": min(self._wait_budget_used, self.config.wait_budget_ms),
                    "remaining_ms": max(0, self.config.wait_budget_ms - self._wait_budget_used),
                },
            }
            if self._inject_diverge and self.path == "range":
                payload["path"] = self.path
                first_tx = self.spec.transactions[0]
                row = first_tx.operations[0].row
                events.append({
                    "event": "injected_divergence",
                    "event_seq": len(events),
                    "at_ms": self.logical_ms,
                    "txid": first_tx.txid,
                    "row": row,
                    "field": "lock_trajectory",
                })
            observed["finished_monotonic_ms"] = self._actual_ms()
            observed["wait_budget"] = {
                "actual_spent_ms": round(self._actual_wait_spent_ms, 6),
                "remaining_ms": max(0.0, round(self.config.wait_budget_ms - self._actual_wait_spent_ms, 6)),
            }
            self._rounds.append(payload)
            self._observed_rounds.append(observed)
            self._range_plan_rounds.append({"round": self.round_no, "intervals": range_plan})
            if self.hooks is not None and hasattr(self.hooks, "commit_round"):
                self.hooks.commit_round(payload, observed)
            if self._crash_after is not None and self.round_no >= self._crash_after:
                if self.hooks is not None and hasattr(self.hooks, "flush"):
                    self.hooks.flush()
                os._exit(70)

    # ---- rounds/state -----------------------------------------------------------
    def _has_work(self) -> bool:
        return any(tx.status not in ("done", "aborted") for tx in self.txs.values())

    def _ordered_running(self) -> list[TxRuntime]:
        return sorted(
            (tx for tx in self.txs.values()
             if tx.status in ("ready", "lock_wait", "commit_ready", "inject_hold")),
            key=lambda tx: tx.priority,
        )

    def _running_txids(self) -> list[str]:
        return [tx.txid for tx in self._ordered_running()]

    def _admit_round(self, events: list[dict[str, Any]], observed: dict[str, Any]) -> None:
        running = self._ordered_running()
        global_used = len(running)
        iso_used = {iso: 0 for iso in sorted({tx.spec.isolation for tx in self.txs.values()})}
        for tx in running:
            iso_used[tx.spec.isolation] = iso_used.get(tx.spec.isolation, 0) + 1
        for tx in sorted((tx for tx in self.txs.values() if tx.status == "pending"),
                         key=lambda tx: tx.priority):
            iso = tx.spec.isolation
            reason: str | None = None
            if global_used >= self.config.max_running:
                reason = "global"
            elif iso_used.get(iso, 0) >= self.config.quota_for(iso):
                reason = f"isolation:{iso}"
            if reason is not None:
                if tx.quota_wait_start_round is None:
                    tx.quota_wait_start_round = self.round_no
                    tx.quota_wait_start_ns = time.monotonic_ns()
                continue
            if tx.quota_wait_start_ns is not None:
                actual_wait_ms = round(
                    (time.monotonic_ns() - tx.quota_wait_start_ns) / 1_000_000.0, 6)
                observed["quota_observations"].append({
                    "txid": tx.txid,
                    "start_monotonic_ms": round(tx.quota_wait_start_ns / 1_000_000.0, 6),
                    "end_monotonic_ms": self._actual_ms(),
                    "observed_wait_ms": actual_wait_ms,
                })
            tx.status = "ready"
            tx.admitted_round = self.round_no
            global_used += 1
            iso_used[iso] = iso_used.get(iso, 0) + 1
            tx.future = self.executor.submit(self._worker_loop, tx.txid)
            event = self._emit(events, "admit", txid=tx.txid, isolation=iso)
            observed["lock_observations"].append({
                "event_seq": event["event_seq"],
                "kind": "admission",
                "monotonic_ms": self._actual_ms(),
            })

    def _quota_snapshot(self, running: list[str]) -> dict[str, Any]:
        iso_used: dict[str, int] = {}
        for txid in running:
            iso = self.txs[txid].spec.isolation
            iso_used[iso] = iso_used.get(iso, 0) + 1
        isos = sorted({tx.spec.isolation for tx in self.txs.values()})
        return {
            "global": {"used": len(running), "limit": self.config.max_running},
            "isolation": {
                iso: {"used": iso_used.get(iso, 0), "limit": self.config.quota_for(iso)}
                for iso in isos
            },
        }

    def _admission_queue_snapshot(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for tx in sorted((tx for tx in self.txs.values() if tx.status == "pending"),
                         key=lambda tx: tx.priority):
            start = tx.quota_wait_start_round or self.round_no
            out.append({
                "txid": tx.txid,
                "isolation": tx.spec.isolation,
                "wait_ms": max(0, (self.round_no - start) * self.tick),
            })
        return out

    # ---- event helpers ----------------------------------------------------------
    def _emit(self, events: list[dict[str, Any]], event: str, **fields: Any) -> dict[str, Any]:
        value = {
            "event": event,
            "event_seq": len(events),
            "at_ms": self.logical_ms,
            **fields,
        }
        events.append(value)
        return value

    def _record_lock(self, observed: dict[str, Any], event: dict[str, Any], action: str) -> None:
        observed["lock_observations"].append({
            "event_seq": event["event_seq"],
            "kind": action,
            "row": event.get("row"),
            "txid": event.get("txid"),
            "mode": event.get("mode"),
            "monotonic_ms": self._actual_ms(),
        })

    def _emit_acquired(self, events: list[dict[str, Any]], txid: str, acquired: list[tuple[str, str]]) -> None:
        for row, mode in acquired:
            ev = self._emit(events, "lock_acquired", txid=txid, row=row, mode=mode,
                            holder=txid)
            self._record_lock(self._observed, ev, "acquired")

    # ---- operations -------------------------------------------------------------
    @staticmethod
    def _lock_modes(isolation: str, kind: str) -> list[str]:
        if kind == READ:
            if isolation == "read_uncommitted":
                return []
            if isolation == "serializable":
                return ["S", "X"]
            return ["S"]
        if isolation == "serializable":
            return ["S", "X"]
        return ["X"]

    def _effective_wait_limit(self) -> int:
        remaining_budget = max(0, self.config.wait_budget_ms - self._wait_budget_used)
        return max(0, int(min(float(self.config.wait_timeout_ms), remaining_budget)))

    def _process_operation(self, tx: TxRuntime, events: list[dict[str, Any]],
                           commands: list[tuple[TxRuntime, dict[str, Any]]]) -> None:
        op = tx.next_operation()
        if op is None:
            tx.status = "commit_ready"
            return
        modes = self._lock_modes(tx.spec.isolation, op.kind)
        limit = self._effective_wait_limit()
        if not modes:
            self._complete_operation(tx, op, events, commands)
            return
        self._emit(events, "lock_request", txid=tx.txid, row=op.row, modes=modes,
                   kind=op.kind)
        decision = self.lock_manager.request(
            tx.txid, op.row, modes, op, round_no=self.round_no, limit_ms=limit)
        self._emit_acquired(events, tx.txid, decision.acquired)
        if decision.granted:
            self._complete_operation(tx, op, events, commands)
            return
        wait = decision.wait
        assert wait is not None
        tx.status = "lock_wait"
        blockers = [
            {"txid": blocker, "modes": list(blocker_modes)}
            for blocker, blocker_modes in decision.blockers
        ]
        self._emit(events, "lock_conflict", txid=tx.txid, row=op.row,
                   requested_modes=modes, blockers=blockers,
                   queue_ahead=decision.queue_ahead)
        self._emit(events, "wait_start", txid=tx.txid, row=op.row,
                   mode=modes[wait.mode_index], start_round=self.round_no,
                   limit_ms=limit)
        self._observed["wait_observations"].append({
            "txid": tx.txid,
            "row": op.row,
            "mode": modes[wait.mode_index],
            "start_monotonic_ms": round(wait.start_ns / 1_000_000.0, 6),
            "limit_ms": limit,
            "event": "start",
        })

    def _complete_operation(self, tx: TxRuntime, op: Operation,
                            events: list[dict[str, Any]],
                            commands: list[tuple[TxRuntime, dict[str, Any]]]) -> None:
        if op.kind == WRITE:
            old = self._select_visible(tx, op.row, for_write=True)
            version = Version(op.row, self.next_version[op.row], op.value, tx.txid,
                              "uncommitted", self.round_no)
            self.next_version[op.row] += 1
            self.versions[op.row].append(version)
            tx.owned_versions.append(version)
            self._emit(events, "write", txid=tx.txid, row=op.row,
                       isolation=tx.spec.isolation, version=version.version_no,
                       value=version.value, status=version.status,
                       previous_version=old.version_no if old else None,
                       previous_value=old.value if old else None)
            command = {
                "txid": tx.txid,
                "round": self.round_no,
                "op_seq": op.seq,
                "kind": op.kind,
                "row": op.row,
                "version": version.version_no,
            }
        else:
            visible, source, fixed = self._read_visible(tx, op.row)
            lock_mode = "none" if tx.spec.isolation == "read_uncommitted" else (
                "S+X" if tx.spec.isolation == "serializable" else "S")
            event = self._emit(events, "read", txid=tx.txid, row=op.row,
                               isolation=tx.spec.isolation,
                               version=visible.version_no if visible else None,
                               value=visible.value if visible else None,
                               version_status=visible.status if visible else None,
                               source=source, fixed=fixed, lock_mode=lock_mode)
            tx.read_events.append(dict(event))
            command = {
                "txid": tx.txid,
                "round": self.round_no,
                "op_seq": op.seq,
                "kind": op.kind,
                "row": op.row,
                "version": visible.version_no if visible else None,
            }
            if tx.spec.isolation in ("read_committed", "repeatable_read"):
                if self.lock_manager.release_mode(tx.txid, op.row, "S"):
                    release = self._emit(events, "lock_release", txid=tx.txid,
                                         row=op.row, mode="S", reason="statement_end")
                    self._record_lock(self._observed, release, "released")

        tx.next_index += 1
        tx.operations_done += 1
        tx.step_done_round = self.round_no
        tx.status = "commit_ready" if tx.next_index >= len(tx.spec.operations) else "ready"
        commands.append((tx, command))
        if (self._hold_tx == tx.txid and tx.operations_done >= self._hold_after
                and not self._injection_unlocked):
            tx.status = "inject_hold"

    def _select_visible(self, tx: TxRuntime, row: str, *, for_write: bool = False,
                        fixed_version: int | None = None) -> Version | None:
        own = [v for v in tx.owned_versions if v.row == row and v.status == "uncommitted"]
        if own:
            return max(own, key=lambda v: v.version_no)
        versions = [v for v in self.versions[row] if v.status != "aborted"]
        if tx.spec.isolation == "read_uncommitted":
            return max(versions, key=lambda v: v.version_no) if versions else None
        if fixed_version is not None:
            for version in versions:
                if version.version_no == fixed_version:
                    return version
        committed = [v for v in versions if v.status == "committed"]
        return max(committed, key=lambda v: v.version_no) if committed else None

    def _read_visible(self, tx: TxRuntime, row: str) -> tuple[Version | None, str, bool]:
        own = [v for v in tx.owned_versions if v.row == row and v.status == "uncommitted"]
        if own:
            return max(own, key=lambda v: v.version_no), "own_write", False
        if tx.spec.isolation == "repeatable_read" and row in tx.fixed_reads:
            version = self._select_visible(tx, row, fixed_version=tx.fixed_reads[row])
            return version, "repeatable_snapshot", True
        if tx.spec.isolation == "read_uncommitted":
            version = self._select_visible(tx, row)
            source = "uncommitted_latest" if version and version.status == "uncommitted" else "committed_latest"
            return version, source, False
        version = self._select_visible(tx, row)
        source = "committed_latest"
        if tx.spec.isolation == "repeatable_read":
            if version is not None:
                tx.fixed_reads[row] = version.version_no
            source = "first_read"
        return version, source, False

    # ---- waiting/deadlock/timeout ------------------------------------------------
    def _service_waiters(self, events: list[dict[str, Any]],
                         commands: list[tuple[TxRuntime, dict[str, Any]]]) -> None:
        for row in self.lock_manager.waiting_rows():
            while True:
                grant = self.lock_manager.advance_head(row, self.round_no)
                if grant is None:
                    break
                wait = grant.wait
                tx = self.txs[wait.txid]
                self._emit_acquired(events, tx.txid, grant.acquired)
                if not grant.complete:
                    break
                self._emit(events, "wait_end", txid=tx.txid, row=wait.row,
                           mode=wait.modes[wait.mode_index] if wait.mode_index < len(wait.modes)
                           else wait.modes[-1], result="granted", wait_ms=grant.wait_ms)
                end_ns = time.monotonic_ns()
                actual_ms = max(0.0, (end_ns - wait.start_ns) / 1_000_000.0)
                self._actual_wait_spent_ms += actual_ms
                self._wait_budget_used += grant.wait_ms
                self._observed["wait_observations"].append({
                    "txid": tx.txid,
                    "row": wait.row,
                    "mode": wait.modes[-1],
                    "start_monotonic_ms": round(wait.start_ns / 1_000_000.0, 6),
                    "end_monotonic_ms": round(end_ns / 1_000_000.0, 6),
                    "observed_wait_ms": round(actual_ms, 6),
                    "event": "granted",
                    "canonical_wait_ms": grant.wait_ms,
                })
                tx.status = "ready"
                if grant.complete:
                    self._complete_operation(tx, wait.op, events, commands)
                if tx.status == "inject_hold":
                    break

    def _resolve_deadlocks(self, events: list[dict[str, Any]],
                           commands: list[tuple[TxRuntime, dict[str, Any]]]) -> None:
        while True:
            graph = self.lock_manager.wait_graph()
            cycles = graph["cycles"]
            if not cycles:
                return
            cycle = max(cycles, key=lambda c: (
                max(self.priorities[txid] for txid in c),
                tuple(self.priorities[txid] for txid in c),
            ))
            victim = max(cycle, key=lambda txid: self.priorities[txid])
            basis = "max(start_seq,txid)"
            victim_tx = self.txs[victim]
            self._emit(events, "deadlock_cycle", cycle=list(cycle),
                       wait_graph_edges=graph["edges"], selected_victim=victim,
                       basis=basis)
            self._abort_tx(victim_tx, "deadlock_victim", events)
            record = {
                "round": self.round_no,
                "logical_ms": self.logical_ms,
                "cycle": list(cycle),
                "wait_graph_edges": graph["edges"],
                "victim": victim,
                "basis": basis,
                "detected_monotonic_ms": self._actual_ms(),
            }
            self._deadlocks.append(record)
            self._emit(events, "deadlock_victim", txid=victim, cycle=list(cycle),
                       basis=basis)
            self._service_waiters(events, commands)

    def _resolve_timeouts(self, events: list[dict[str, Any]],
                          commands: list[tuple[TxRuntime, dict[str, Any]]]) -> None:
        now_ns = time.monotonic_ns()
        due: list[LockWait] = []
        for wait in self.lock_manager.all_waits():
            deadline = wait.start_ns + wait.limit_ms * 1_000_000
            if now_ns >= deadline:
                due.append(wait)
        for wait in sorted(due, key=lambda item: (item.priority, item.row, item.mode_index)):
            if wait not in self.lock_manager.waiters[wait.row]:
                continue
            tx = self.txs[wait.txid]
            self.lock_manager.waiters[wait.row].remove(wait)
            end_ns = time.monotonic_ns()
            actual_ms = max(0.0, (end_ns - wait.start_ns) / 1_000_000.0)
            self._actual_wait_spent_ms += actual_ms
            self._wait_budget_used += wait.limit_ms
            self._emit(events, "wait_end", txid=tx.txid, row=wait.row,
                       mode=wait.modes[wait.mode_index], result="timeout",
                       wait_ms=wait.limit_ms, code="E_TIMEOUT")
            self._emit(events, "E_TIMEOUT", txid=tx.txid, row=wait.row,
                       waited_ms=wait.limit_ms)
            self._observed["wait_observations"].append({
                "txid": tx.txid,
                "row": wait.row,
                "mode": wait.modes[-1],
                "start_monotonic_ms": round(wait.start_ns / 1_000_000.0, 6),
                "end_monotonic_ms": round(end_ns / 1_000_000.0, 6),
                "observed_wait_ms": round(actual_ms, 6),
                "event": "timeout",
                "canonical_wait_ms": wait.limit_ms,
            })
            record = {
                "round": self.round_no,
                "logical_ms": self.logical_ms,
                "txid": tx.txid,
                "row": wait.row,
                "mode": wait.modes[wait.mode_index],
                "wait_ms": wait.limit_ms,
                "observed_wait_ms": round(actual_ms, 6),
                "code": "E_TIMEOUT",
            }
            self._timeouts.append(record)
            self._injection_unlocked = True
            self._abort_tx(tx, "wait_timeout", events)
            self._service_waiters(events, commands)

    def _maybe_stall_until_timeout(self, events: list[dict[str, Any]],
                                   commands: list[tuple[TxRuntime, dict[str, Any]]]) -> None:
        if commands:
            return
        waits = self.lock_manager.all_waits()
        if not waits:
            return
        # No runnable step exists. Block on a condition until the next real monotonic
        # deadline; this branch is entered only for an externally injected stalled
        # holder or a fully blocked wait graph.
        while True:
            now = time.monotonic_ns()
            next_deadline = min(w.start_ns + w.limit_ms * 1_000_000 for w in waits)
            if now >= next_deadline:
                self._resolve_timeouts(events, commands)
                return
            remaining = min((next_deadline - now) / 1_000_000_000.0, 0.01)
            self._idle_gate.wait(max(0.0001, remaining))
            waits = self.lock_manager.all_waits()
            if not waits:
                return
            if any(tx.status in ("ready", "commit_ready") for tx in self.txs.values()):
                return

    def _abort_tx(self, tx: TxRuntime, reason: str, events: list[dict[str, Any]]) -> None:
        if tx.status in ("done", "aborted"):
            return
        for version in tx.owned_versions:
            if version.status == "uncommitted":
                version.status = "aborted"
        released = self.lock_manager.release_all(tx.txid)
        for row, mode in released:
            ev = self._emit(events, "lock_release", txid=tx.txid, row=row, mode=mode,
                            reason=reason)
            self._record_lock(self._observed, ev, "released")
        tx.status = "aborted"
        tx.abort_reason = reason
        tx.completed_round = self.round_no
        self._emit(events, "abort", txid=tx.txid, reason=reason)

    # ---- commit/workers ----------------------------------------------------------
    def _commit_phase(self, events: list[dict[str, Any]]) -> None:
        for tx in self._ordered_running():
            if tx.status != "commit_ready":
                continue
            for version in tx.owned_versions:
                if version.status == "uncommitted":
                    version.status = "committed"
                    version.committed_round = self.round_no
            released = self.lock_manager.release_all(tx.txid)
            tx.status = "done"
            tx.completed_round = self.round_no
            self._emit(events, "commit", txid=tx.txid,
                       versions=[v.version_no for v in tx.owned_versions])
            for row, mode in released:
                ev = self._emit(events, "lock_release", txid=tx.txid, row=row,
                                mode=mode, reason="commit")
                self._record_lock(self._observed, ev, "released")

    def _worker_loop(self, txid: str) -> None:
        tx = self.txs[txid]
        tx.worker_name = threading.current_thread().name
        self._worker_names.add(tx.worker_name)
        while True:
            command = tx.command_q.get()
            if command is None:
                return
            token = digest({
                "txid": command["txid"],
                "round": command["round"],
                "op_seq": command["op_seq"],
                "kind": command["kind"],
                "row": command["row"],
                "version": command["version"],
            })
            self._ack_q.put({
                "txid": txid,
                "round": command["round"],
                "op_seq": command["op_seq"],
                "thread": tx.worker_name,
                "token": token,
                "monotonic_ms": self._actual_ms(),
            })

    def _dispatch_commands(self, commands: list[tuple[TxRuntime, dict[str, Any]]],
                           observed: dict[str, Any]) -> None:
        for tx, command in commands:
            tx.command_q.put(command)
        expected = len(commands)
        received = 0
        while received < expected:
            try:
                ack = self._ack_q.get(timeout=max(1.0, self.config.wait_timeout_ms / 1000.0 + 2.0))
            except queue.Empty as exc:
                raise RuntimeError("worker acknowledgement timeout") from exc
            observed["worker_acks"].append(ack)
            received += 1
        observed["worker_acks"].sort(key=lambda item: (item["round"], item["txid"], item["op_seq"]))

    def _close_finished_workers(self) -> None:
        for tx in sorted(self.txs.values(), key=lambda item: item.priority):
            if tx.status in ("done", "aborted") and not tx.worker_closed:
                tx.command_q.put(None)
                tx.worker_closed = True

    def _close_all_workers(self) -> None:
        for tx in sorted(self.txs.values(), key=lambda item: item.priority):
            if not tx.worker_closed:
                tx.command_q.put(None)
                tx.worker_closed = True

    def _unlock_injection_if_done(self) -> None:
        if not self._injection_unlocked:
            return
        for tx in self.txs.values():
            if tx.status == "inject_hold":
                tx.status = ("commit_ready" if tx.next_index >= len(tx.spec.operations)
                             else "ready")

    # ---- final -------------------------------------------------------------------
    def _final_result(self) -> ExecutionResult:
        ordered = sorted(self.txs.values(), key=lambda tx: tx.priority)
        all_versions = {
            row: [v.public() for v in sorted(self.versions[row], key=lambda v: v.version_no)]
            for row in self.spec.row_keys
        }
        conclusions: list[dict[str, Any]] = []
        for tx in ordered:
            conclusions.append({
                "txid": tx.txid,
                "start_seq": tx.spec.start_seq,
                "isolation": tx.spec.isolation,
                "outcome": tx.status,
                "abort_reason": tx.abort_reason,
                "admitted_round": tx.admitted_round,
                "completed_round": tx.completed_round,
                "fixed_reads": dict(sorted(tx.fixed_reads.items())),
                "reads": [dict(item) for item in tx.read_events],
                "owned_versions": [v.public() for v in tx.owned_versions],
            })
        status = "COMPLETED"
        if self._deadlocks and self._timeouts:
            status = "COMPLETED_WITH_DEADLOCK_AND_TIMEOUT"
        elif self._deadlocks:
            status = "COMPLETED_WITH_DEADLOCK"
        elif self._timeouts:
            status = "COMPLETED_WITH_TIMEOUTS"
        result = {
            "format_version": 1,
            "status": status,
            "spec_sha256": self.spec.spec_sha256,
            "path": self.path,
            "round_count": len(self._rounds),
            "transactions": {tx.txid: tx.status for tx in ordered},
            "final_versions": all_versions,
            "final_locks": self.lock_manager.holder_snapshot(),
            "deadlocks": self._deadlocks,
            "timeouts": self._timeouts,
            "wait_budget": {
                "limit_ms": self.config.wait_budget_ms,
                "canonical_used_ms": self._wait_budget_used,
                "observed_used_ms": round(self._actual_wait_spent_ms, 6),
            },
            "worker_threads": sorted(self._worker_names),
            "isolations": conclusions,
            "canonical_sha256": digest({"rounds": self._rounds, "final": {
                "versions": all_versions,
                "transactions": {tx.txid: tx.status for tx in ordered},
            }}),
        }
        return ExecutionResult(
            status=status,
            path=self.path,
            rounds=self._rounds,
            observed_rounds=self._observed_rounds,
            result=result,
            isolation_conclusions=conclusions,
            deadlocks=self._deadlocks,
            timeouts=self._timeouts,
            worker_threads=sorted(self._worker_names),
            range_plan_rounds=self._range_plan_rounds,
        )


def execute(spec: Spec, config: RuntimeConfig, path: str, hooks: Any | None = None) -> ExecutionResult:
    return Engine(spec, config, path, hooks).execute()


def compare_canonical(left: ExecutionResult, right: ExecutionResult) -> dict[str, Any]:
    left_bytes = canonical_text(left.rounds).encode("utf-8")
    right_bytes = canonical_text(right.rounds).encode("utf-8")
    if left_bytes == right_bytes:
        return {"equal": True, "left_sha256": digest(left.rounds),
                "right_sha256": digest(right.rounds)}
    from .canonical import first_diff_location, structured_diff
    diffs = structured_diff(left.rounds, right.rounds, "$.rounds")
    txid, row, field = _locate_round_diff(diffs, left.rounds, right.rounds)
    return {
        "equal": False,
        "left_sha256": digest(left.rounds),
        "right_sha256": digest(right.rounds),
        "first_txid": txid,
        "first_row": row,
        "first_field": field,
        "differences": diffs[:16],
    }


def _locate_round_diff(diffs: list[dict[str, Any]], left_rounds: list[dict[str, Any]],
                       right_rounds: list[dict[str, Any]]) -> tuple[str | None, str | None, str]:
    if not diffs:
        return None, None, "$"
    path = str(diffs[0]["path"])
    round_match = re.search(r"rounds\[(\d+)\]", path)
    if not round_match:
        return None, None, path
    round_idx = int(round_match.group(1))
    if round_idx >= max(len(left_rounds), len(right_rounds)):
        return None, None, path
    payload = right_rounds[round_idx] if round_idx < len(right_rounds) else left_rounds[round_idx]
    if path.endswith(".events.length"):
        left_events = (left_rounds[round_idx].get("events", [])
                       if round_idx < len(left_rounds) else [])
        right_events = (right_rounds[round_idx].get("events", [])
                        if round_idx < len(right_rounds) else [])
        idx = min(len(left_events), len(right_events))
        source = right_events if len(right_events) > len(left_events) else left_events
        if idx < len(source):
            event = source[idx]
            return (str(event.get("txid")) if event.get("txid") is not None else None,
                    str(event.get("row")) if event.get("row") is not None else None, "event")
    event_match = re.search(r"events\[(\d+)\]", path)
    if event_match:
        event_idx = int(event_match.group(1))
        events = payload.get("events", [])
        if event_idx < len(events):
            event = events[event_idx]
            txid = event.get("txid")
            row = event.get("row")
            field = path.rsplit(".", 1)[-1]
            return (str(txid) if txid is not None else None,
                    str(row) if row is not None else None, field)
    running = payload.get("running", [])
    return (str(running[0]) if running else None, None, path)
