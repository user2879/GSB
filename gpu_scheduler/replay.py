"""Crash recovery and deterministic replay comparison.

Recovery trusts only fully committed ledger rounds.  A shard that was running
without a committed completion is labelled UNKNOWN and rerun; the component
never fabricates an exit report for it.  The recovered continuation is then
compared shard-by-shard against an uninterrupted reference run over the same
round interval, and the first divergence is reported as ``E_REPLAY`` with the
round, task id and shard index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .engine import SimulationEngine
from .errors import E_REPLAY, Problem, SchedulerError
from .events import (
    COMPLETE,
    DISPATCH,
    FAIL,
    MARK_UNKNOWN,
    PREEMPT,
    RoundRecord,
    TraceEvent,
)
from .graph import ShardKey
from .ledger import Ledger
from .model import (
    RUNNING,
    UNKNOWN,
    Declaration,
)


@dataclass
class ReplayDelta:
    round_no: int
    task_id: str
    shard_index: int
    field: str
    expected: object
    actual: object

    def to_dict(self) -> dict:
        return {
            "round_no": self.round_no,
            "task_id": self.task_id,
            "shard_index": self.shard_index,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
        }


def _event_index(round_record: RoundRecord) -> Dict[ShardKey, List[TraceEvent]]:
    index: Dict[ShardKey, List[TraceEvent]] = {}
    for event in round_record.events:
        index.setdefault((event.task_id, event.shard_index), []).append(event)
    return index


def _shard_outcome_sequence(rounds: List[RoundRecord]):
    """Collapse rounds into per-shard dispatch usage, terminal outcome and
    preemption sequences.  Timestamps and extra rerun rounds are intentionally
    excluded because an UNKNOWN in-flight shard legitimately re-dispatches and
    shifts the recovered clock, while outcomes, resource usage and preemption
    decisions must still match the uninterrupted reference.
    """
    dispatch: Dict[ShardKey, List[Tuple[str, int, int]]] = {}
    terminal: Dict[ShardKey, str] = {}
    preemptions = []
    for record in rounds:
        if record.preemption is not None:
            preemptions.append(_preemption_tuple(record.preemption))
        for event in record.events:
            key = (event.task_id, event.shard_index)
            if event.kind == DISPATCH:
                dispatch.setdefault(key, []).append(
                    (event.pool_id or "", event.gpu_count, event.memory_mib))
            elif event.kind in (COMPLETE, FAIL):
                terminal[key] = event.kind
    return dispatch, terminal, sorted(preemptions)


def compare_round_intervals(
    reference: List[RoundRecord],
    recovered: List[RoundRecord],
    committed_count: int,
) -> None:
    """Compare shard outcomes, resource usage and preemptions across runs.

    Only work performed by the recovered suffix is compared.  UNKNOWN in-flight
    shards legitimately add rerun dispatches, so dispatch usage is compared as
    multisets (recovered usages must be a sub-multiset of the reference usage
    choices for the same shard), terminal outcomes must match exactly for every
    shard finalized in the suffix, and the recovered preemption sequence must
    be a prefix-consistent subsequence of the reference sequence.  The first
    mismatch carries its round, task id and shard index.
    """
    from collections import Counter
    recovered_suffix = recovered[committed_count:]
    if not recovered_suffix:
        return
    first_round = recovered_suffix[0].round_no
    ref_dispatch, ref_terminal, ref_preempt = _shard_outcome_sequence(
        reference)
    rec_dispatch, rec_terminal, rec_preempt = _shard_outcome_sequence(
        recovered_suffix)

    for key in sorted(rec_terminal):
        if ref_terminal.get(key) != rec_terminal.get(key):
            raise _divergence(first_round, key, "terminal_outcome",
                               ref_terminal.get(key), rec_terminal.get(key))

    for key in sorted(rec_dispatch):
        reference_counter = Counter(ref_dispatch.get(key, []))
        recovered_counter = Counter(rec_dispatch[key])
        for usage, count in recovered_counter.items():
            if reference_counter.get(usage, 0) < count:
                raise _divergence(first_round, key, "dispatch_usage",
                                   sorted(ref_dispatch.get(key, [])),
                                   sorted(rec_dispatch[key]))

    # Every preemption observed during recovery must occur in the same order
    # as the deterministic reference preemption sequence.
    cursor = 0
    for entry in rec_preempt:
        matched = False
        while cursor < len(ref_preempt):
            if ref_preempt[cursor] == entry:
                matched = True
                cursor += 1
                break
            cursor += 1
        if not matched:
            raise _divergence(first_round, ("preemption", 0),
                              "preemption_sequence", ref_preempt, rec_preempt)


def _preemption_tuple(record) -> Optional[tuple]:
    if record is None:
        return None
    return (record.trigger_task_id, record.trigger_shard_index,
            record.victim_task_id, record.victim_shard_index,
            record.released_gpu, record.released_memory_mib, record.clock_ms)


def _divergence(round_no: int, key: ShardKey, field_name: str,
                expected: object, actual: object) -> SchedulerError:
    return SchedulerError(
        E_REPLAY,
        "recovered schedule diverges from uninterrupted reference",
        [Problem(f"round:{round_no}.task:{key[0]}.shard:{key[1]}.{field_name}",
                 "identical replay outcome",
                 {"expected": expected, "actual": actual}, E_REPLAY)])


def committed_shard_keys(ledger: Ledger) -> Set[ShardKey]:
    """Shard keys whose latest committed event proves a safe boundary state."""
    safe: Set[ShardKey] = set()
    for record in ledger.load_rounds():
        for event in record.events:
            key = (event.task_id, event.shard_index)
            if event.kind in (COMPLETE, FAIL, PREEMPT, MARK_UNKNOWN):
                safe.add(key)
            elif event.kind == DISPATCH:
                safe.discard(key)
    return safe


def reference_run(declaration: Declaration) -> SimulationEngine:
    engine = SimulationEngine(declaration)
    engine.run_until()
    return engine


def reconstruct_engine(declaration: Declaration,
                       ledger: Ledger) -> Tuple[SimulationEngine, List[ShardKey],
                                                int]:
    """Rebuild engine state from the latest committed per-shard state.

    The ledger stores a full shard snapshot inside every round transaction, so
    recovery restores exact finish timestamps, wait totals and the running set
    rather than guessing them from an event stream.  Shards still RUNNING at
    the last committed round have no completion event and become UNKNOWN.
    """
    engine = SimulationEngine(declaration)
    index = engine.rt.shards
    rows = ledger.latest_shard_states()
    committed_rounds = ledger.last_round_no()
    unknown: List[ShardKey] = []
    for row in rows:
        key = (row["task_id"], row["shard_index"])
        shard = index[key]
        shard.state = row["state"]
        shard.version = int(row["version"])
        shard.output_digest = row["output_digest"]
        shard.pool_id = row["pool_id"]
        shard.started_ms = row["started_ms"]
        shard.finish_ms = row["finish_ms"]
        shard.ended_ms = row["ended_ms"]
        shard.wait_duration_ms = int(row["wait_duration_ms"])
        shard.attempts = int(row["attempts"])
        if row["state"] == RUNNING:
            # A running shard at the last committed round is exactly the
            # uncommitted work that must be rerun after a hard kill.
            unknown.append(key)

    trace: List[TraceEvent] = []
    engine.rt.round_no = committed_rounds
    cursor = ledger.last_cursor()
    engine.rt.clock = cursor["last_clock"] if cursor else 0
    unknown_set = set(unknown)
    # Force the UNKNOWN label directly: reconstruction starts with an empty
    # running map, so the generic mark_unknown_uncommitted scan cannot see the
    # in-flight shards recorded as RUNNING in the final committed snapshot.
    for key in sorted(unknown_set):
        shard = index[key]
        shard.state = UNKNOWN
        shard.finish_ms = None
        shard.pool_id = None
        shard.started_ms = None
        trace.append(TraceEvent(
            round_no=committed_rounds, clock_ms=engine.rt.clock,
            kind=MARK_UNKNOWN, task_id=key[0], shard_index=key[1],
            pool_id=None, gpu_count=0, memory_mib=0,
            detail="uncommitted after forced termination; rerun"))
    marked = sorted(unknown_set)
    engine.rearm_recoverable()
    engine.rt.rounds = ledger.load_rounds()
    engine.rt.running = {}
    return engine, marked, committed_rounds


def continue_recovered(engine: SimulationEngine,
                       ledger: Ledger) -> List[RoundRecord]:
    """Run remaining rounds onto a new ledger, returning the full round list."""
    prev_digest = ledger.verify()
    while not engine._all_terminal():
        engine.rt.round_no += 1
        trace: List[TraceEvent] = []
        engine._finalize_due(trace)
        preemption = engine._schedule_pass(trace)
        record = engine._build_round(trace, preemption)
        prev_digest = ledger.append_round(
            record, prev_digest, engine.persistence_snapshot())
        engine.rt.rounds.append(record)
        if not engine._advance_clock(trace):
            break
    return engine.rt.rounds
