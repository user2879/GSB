"""Batch runner that couples state advancement with SQLite transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .clock import MonotonicClock
from .engine import CallDecision, DiagnosticEngine
from .models import ChainSpec, TraceStream
from .storage import RecoveryReport, SessionStore, SnapshotRecord


@dataclass(frozen=True)
class RunOutcome:
    session_id: str
    recovery: RecoveryReport
    processed_calls: int
    final_cursor: int
    decisions: tuple[CallDecision, ...]

    def as_dict(self, *, include_decisions: bool = False) -> dict[str, Any]:
        value = {
            "session_id": self.session_id,
            "recovery": self.recovery.as_dict(),
            "processed_calls": self.processed_calls,
            "final_cursor": self.final_cursor,
        }
        if include_decisions:
            value["decisions"] = [item.as_dict() for item in self.decisions]
        return value


class BatchRunner:
    """Resume only from committed batches and commit one batch atomically."""

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        chain: ChainSpec,
        trace: TraceStream,
        *,
        clock: Any | None = None,
        follow_trace_time: bool | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.chain = chain
        self.trace = trace
        self.clock = clock if clock is not None else MonotonicClock()
        if follow_trace_time is None:
            follow_trace_time = hasattr(self.clock, "follow")
        self.follow_trace_time = follow_trace_time
        self.engine = DiagnosticEngine(chain, self.clock)
        self.recovery = store.recover(session_id)
        self._restore_committed_state()

    def _restore_committed_state(self) -> None:
        session = self.store.get_session(self.session_id)
        cursor = int(session["committed_cursor"])
        if cursor == 0:
            self.engine.reset()
            return
        state = self.store.load_latest_state(self.session_id)
        if state is None:
            raise RuntimeError(
                f"session {self.session_id} has cursor {cursor} but no committed state"
            )
        self.engine.restore_state(state)
        if self.engine.cursor != cursor:
            raise RuntimeError(
                "committed cursor and committed state cursor disagree: "
                f"{cursor} != {self.engine.cursor}"
            )

    def _follow_clock(self, timestamp_ms: int) -> None:
        if self.follow_trace_time and hasattr(self.clock, "follow"):
            self.clock.follow(timestamp_ms)

    def run(self, max_batches: int | None = None, batch_size: int = 8) -> RunOutcome:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        total = len(self.trace)
        processed = 0
        decisions: list[CallDecision] = []
        batches = 0
        while self.engine.cursor < total:
            if max_batches is not None and batches >= max_batches:
                break
            start = self.engine.cursor + 1
            end_exclusive = min(total + 1, start + batch_size)
            batch_no = self.store.start_batch_attempt(
                self.session_id,
                start_call=start,
                end_call_exclusive=end_exclusive,
            )
            batch_decisions: list[CallDecision] = []
            for call in self.trace.calls[start - 1 : end_exclusive - 1]:
                self._follow_clock(call.timestamp_ms)
                decision = self.engine.process_call(call)
                batch_decisions.append(decision)
                decisions.append(decision)
                processed += 1
            self.store.commit_batch(
                self.session_id,
                batch_no,
                [decision.as_dict() for decision in batch_decisions],
                self.engine,
            )
            batches += 1
        return RunOutcome(
            session_id=self.session_id,
            recovery=self.recovery,
            processed_calls=processed,
            final_cursor=self.engine.cursor,
            decisions=tuple(decisions),
        )

    def create_snapshot(self) -> SnapshotRecord:
        return self.store.create_snapshot(
            self.session_id, self.engine, self.chain, self.trace
        )
