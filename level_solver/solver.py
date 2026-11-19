"""Deterministic constraint solver with backtracking and durable snapshots."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from . import constraints as constraint_lib
from .errors import E_CONFLICT, E_DIGEST, SolverError
from .grid import Grid
from .precheck_pool import BoundedPrecheckPool, PrecheckJob
from .prng import DeterministicRNG, derive_seed
from .snapshot import (
    FrameRecord,
    Snapshot,
    SnapshotStore,
    build_version_id,
)


@dataclass
class SolveResult:
    grid: Grid
    rooms: list[dict[str, int]]
    constraint_counts: dict[str, int]
    conflicts: list[dict]
    applied: list[FrameRecord]
    sequence: int
    interrupted: bool = False
    digest_warnings: list[dict] = field(default_factory=list)
    resumed: bool = False


class LevelSolver:
    """Apply constraints in declaration order with bounded rollback."""

    def __init__(self, spec, snapshot_dir: str) -> None:
        self.spec = spec
        self.version_id = build_version_id(spec)
        self.store = SnapshotStore(snapshot_dir, self.version_id)
        self.grid = Grid(spec.width, spec.height, spec.blocked)
        self.rooms: list[dict[str, int]] = []
        self.frames: list[FrameRecord] = []
        self.retries = [0] * len(spec.constraints)
        self.constraint_counts: dict[str, int] = {c.cid: 0 for c in spec.constraints}
        self.conflicts: list[dict] = []
        self.checkpoints: list[Snapshot] = []
        self.sequence = 0
        self.backtracks = 0
        self.rng = DeterministicRNG(derive_seed(("level", spec.seed)))
        self.pool = BoundedPrecheckPool(spec.concurrency, spec.memory_budget)
        self.digest_warnings: list[dict] = []
        self.resumed = False
        self._interrupt_event = threading.Event()

    # ------------------------------------------------------------- interrupts
    def request_interrupt(self, reason: str = "signal") -> None:
        self._interrupt_event.reason = reason  # type: ignore[attr-defined]
        self._interrupt_event.set()

    def _interrupt_reason(self) -> str:
        return getattr(self._interrupt_event, "reason", "signal")

    # --------------------------------------------------------------- snapshots
    def _make_snapshot(self, next_index: int) -> Snapshot:
        return Snapshot(
            version_id=self.version_id,
            sequence=self.sequence,
            next_index=next_index,
            grid=self.grid.to_list(),
            rooms=[dict(room) for room in self.rooms],
            frames=list(self.frames),
            retries=list(self.retries),
            constraint_counts=dict(self.constraint_counts),
            conflicts=list(self.conflicts),
            rng_state=self.rng.state,
            seed=self.spec.seed,
        )

    def _commit(self, next_index: int) -> None:
        snapshot = self._make_snapshot(next_index)
        self.store.save_snapshot(snapshot)
        self.checkpoints.append(snapshot)
        self.store.clear_interrupted()

    def _restore_snapshot(self, snapshot: Snapshot) -> None:
        self.grid = Grid.from_list(snapshot.grid)
        self.rooms = [dict(room) for room in snapshot.rooms]
        self.frames = [FrameRecord(**frame.__dict__) for frame in snapshot.frames]
        self.retries = list(snapshot.retries) if snapshot.retries else [0] * len(self.spec.constraints)
        self.constraint_counts = dict(snapshot.constraint_counts)
        self.conflicts = list(snapshot.conflicts)
        self.sequence = snapshot.sequence
        self.rng.state = snapshot.rng_state
        self.checkpoints = self.checkpoints[: self.sequence]

    # ------------------------------------------------------------------- setup
    def prepare(self, allow_resume: bool = True) -> None:
        self.store.initialize()
        if not allow_resume:
            return
        snapshot, digest_error = self.store.load_latest_valid()
        interrupted = self.store.read_interrupted()
        if digest_error is not None:
            self.digest_warnings.append(digest_error.to_dict()["error"])
        if snapshot is None:
            if digest_error is not None and interrupted is not None:
                # No intact state survived: restart from scratch is the only safe path.
                self.digest_warnings.append(
                    {
                        "code": E_DIGEST,
                        "message": "all snapshots invalid; restarting from scratch",
                    }
                )
            return
        if interrupted is None or interrupted.get("version_id") != self.version_id:
            # A clean run left snapshots but no suspend marker; keep them only
            # as history and start fresh to preserve deterministic semantics.
            return
        self._restore_snapshot(snapshot)
        self.resumed = True

    # ---------------------------------------------------------------- precheck
    def _precheck_remaining(self, index: int) -> None:
        jobs = []
        for order, constraint in enumerate(self.spec.constraints[index:], start=index):
            jobs.append(
                PrecheckJob(
                    order=order,
                    cid=constraint.cid,
                    spec=constraint,
                    grid=self.grid.clone(),
                    rooms=list(self.rooms),
                )
            )
        self.pool.run(jobs)

    # --------------------------------------------------------------- conflicts
    def _record_conflict(self, constraint_id: str, reason: str, other_id: str | None) -> None:
        pair = tuple(sorted(pid for pid in (constraint_id, other_id) if pid))
        entry = {"constraint": constraint_id, "with": other_id, "pair": list(pair), "reason": reason}
        if entry not in self.conflicts:
            self.conflicts.append(entry)

    def _first_unsatisfied(self, current_index: int) -> tuple[str, str] | None:
        for constraint in self.spec.constraints[:current_index]:
            ok, reason = constraint_lib.is_satisfied(constraint, self.grid, self.rooms)
            if not ok:
                return constraint.cid, reason
        return None

    # ---------------------------------------------------------------- backtrack
    def _backtrack(self, failing_cid: str, reason: str) -> bool:
        if self.backtracks >= self.spec.max_rollback or self.sequence == 0:
            raise SolverError(
                E_CONFLICT,
                f"constraint '{failing_cid}' cannot be satisfied: {reason}",
                {
                    "constraint": failing_cid,
                    "reason": reason,
                    "conflicts": self.conflicts,
                    "backtracks": self.backtracks,
                    "max_rollback": self.spec.max_rollback,
                },
            )
        self.backtracks += 1
        target_snapshot = self.checkpoints[-1]
        self._restore_snapshot(target_snapshot)
        self.retries[self.sequence] += 1
        return True

    # --------------------------------------------------------------------- run
    def solve(
        self,
        interrupt_after: int | None = None,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> SolveResult:
        if self.sequence == 0:
            self._commit(next_index=0)
        total = len(self.spec.constraints)
        while self.sequence < total:
            if self._interrupt_event.is_set() or (should_interrupt and should_interrupt()):
                self._suspend()
                return self._result(interrupted=True)
            if interrupt_after is not None and self.sequence >= interrupt_after:
                self.request_interrupt("interrupt_after")
                self._suspend()
                return self._result(interrupted=True)

            index = self.sequence
            constraint = self.spec.constraints[index]
            frame = FrameRecord(cid=constraint.cid, ctype=constraint.type, attempts=0, label="")
            self.frames.append(frame)

            # Concurrent feasibility precheck over the remaining constraints.
            try:
                self._precheck_remaining(index)
            except SolverError as exc:
                self._record_conflict(constraint.cid, exc.message, None)
                self.frames.pop()
                self._backtrack(constraint.cid, exc.message)
                continue

            already_ok, _ = constraint_lib.is_satisfied(constraint, self.grid, self.rooms)
            if already_ok:
                frame.label = "already-satisfied"
                self.constraint_counts[constraint.cid] += 1
                self.rng.state = self.rng.next64()
                self.sequence += 1
                self._commit(next_index=self.sequence)
                continue

            rng = self.rng.substream(index * 1_000_003 + self.retries[index])
            actions = constraint_lib.enumerate_actions(
                constraint,
                self.grid,
                self.rooms,
                rng,
                self.spec.alternative_limit,
            )

            applied = False
            last_reason = "no viable alternative"
            for action in actions:
                frame.attempts += 1
                frame.label = action.label
                constraint_lib.apply_action(self.grid, self.rooms, action)
                broken = self._first_unsatisfied(index)
                if broken is not None:
                    other_id, reason = broken
                    self._record_conflict(constraint.cid, reason, other_id)
                    last_reason = reason
                    continue
                ok, reason = constraint_lib.is_satisfied(constraint, self.grid, self.rooms)
                if not ok:
                    self._record_conflict(constraint.cid, reason, None)
                    last_reason = reason
                    continue
                applied = True
                break

            if not applied:
                self.frames.pop()
                self._backtrack(constraint.cid, last_reason)
                continue

            self.constraint_counts[constraint.cid] += 1
            self.rng.state = self.rng.next64()
            self.sequence += 1
            self._commit(next_index=self.sequence)

        return self._result()

    def _suspend(self) -> None:
        snapshot = self._make_snapshot(next_index=self.sequence)
        self.store.save_snapshot(snapshot)
        self.store.write_interrupted(snapshot, self._interrupt_reason(), 130)

    def _result(self, interrupted: bool = False) -> SolveResult:
        return SolveResult(
            grid=self.grid,
            rooms=list(self.rooms),
            constraint_counts=dict(self.constraint_counts),
            conflicts=list(self.conflicts),
            applied=list(self.frames),
            sequence=self.sequence,
            interrupted=interrupted,
            digest_warnings=list(self.digest_warnings),
            resumed=self.resumed,
        )
