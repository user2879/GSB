"""Interactive frame-driven online replay with latency budget and degradation.

The player advances the same :class:`~rts_replay.engine.TickEngine` used by the
offline path, but ticks are consumed in *frames* of ``base_ticks_per_frame``
steps multiplied by the playback speed. Each frame is measured against a
declared latency budget:

* if the frame fits, the step allowance is gradually recovered;
* if it overshoots, the allowance is halved (floor of one) and a skipped-frame
  refresh is recorded with ``E_BUDGET_EXCEEDED``.

Degradation only changes *how many ticks are displayed per frame* (the refresh
rhythm). Already synthesized ticks, snapshots, ledger entries and the final
report are produced by the deterministic engine and therefore stay byte-stable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .engine import TickEngine
from .errors import ErrorCode, RtsError
from .events import Event
from .rules import RuleConfig
from .state import SNAPSHOT_FORMAT_VERSION, GameState, build_snapshot
from .store import nearest_snapshot_at_or_before, save_snapshot

SPEED_MULTIPLIERS = (1, 2, 4, 8)
RECOVERY_STEP = 1


class Clock(Protocol):
    def now(self) -> float: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class ScriptedClock:
    """Deterministic clock whose per-tick cost is supplied up front.

    Used by the repeatable demo so budget statistics are themselves reproducible
    regardless of host machine speed.
    """

    def __init__(self, per_tick_costs: dict[int, float]) -> None:
        self._costs = per_tick_costs
        self._now = 0.0

    def now(self) -> float:
        return self._now

    def charge(self, ticks: list[int]) -> float:
        cost = sum(self._costs.get(tick, self._costs.get(-1, 0.001)) for tick in ticks)
        self._now += cost
        return cost


@dataclass
class FrameStat:
    frame_index: int
    ticks_attempted: int
    ticks_synthesized: int
    elapsed_ms: float
    budget_ms: float
    degraded: bool
    skipped_refresh: bool
    speed: int
    allowance_before: int
    allowance_after: int
    first_tick: int | None
    last_tick: int | None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "ticks_attempted": self.ticks_attempted,
            "ticks_synthesized": self.ticks_synthesized,
            "elapsed_ms": round(self.elapsed_ms, 4),
            "budget_ms": round(self.budget_ms, 4),
            "degraded": self.degraded,
            "skipped_refresh": self.skipped_refresh,
            "speed": self.speed,
            "allowance_before": self.allowance_before,
            "allowance_after": self.allowance_after,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "note": self.note,
        }


@dataclass
class BudgetReport:
    budget_ms: float
    base_ticks_per_frame: int
    frames: list[FrameStat] = field(default_factory=list)
    over_budget_count: int = 0
    skipped_refresh_count: int = 0
    min_allowance: int = 1 << 30
    max_elapsed_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "budget_ms": self.budget_ms,
            "base_ticks_per_frame": self.base_ticks_per_frame,
            "frame_count": len(self.frames),
            "over_budget_count": self.over_budget_count,
            "skipped_refresh_count": self.skipped_refresh_count,
            "min_allowance": self.min_allowance if self.frames else 0,
            "max_elapsed_ms": round(self.max_elapsed_ms, 4),
            "frames": [frame.to_dict() for frame in self.frames],
        }


class ReplayPlayer:
    def __init__(
        self,
        config: RuleConfig,
        events: list[Event],
        *,
        budget_ms: float = 16.0,
        base_ticks_per_frame: int = 4,
        snapshot_interval: int = 10,
        snapshot_dir: str | None = None,
        clock: Clock | None = None,
        workload: Callable[[int], None] | None = None,
    ) -> None:
        if budget_ms <= 0:
            raise RtsError(
                ErrorCode.E_BUDGET_EXCEEDED,
                "budget_ms must be positive",
                context={"budget_ms": budget_ms},
            )
        if base_ticks_per_frame < 1:
            raise RtsError(
                ErrorCode.E_BUDGET_EXCEEDED,
                "base_ticks_per_frame must be >= 1",
                context={"base_ticks_per_frame": base_ticks_per_frame},
            )
        self.config = config
        self.events = events
        self.budget_ms = budget_ms
        self.base_ticks_per_frame = base_ticks_per_frame
        self.snapshot_interval = snapshot_interval
        self.snapshot_dir = snapshot_dir
        self.clock = clock or MonotonicClock()
        self.workload = workload
        self.engine = TickEngine(
            config, events, snapshot_interval=snapshot_interval
        )
        self.paused = True
        self.speed = 1
        self.allowance = base_ticks_per_frame
        self.frame_index = 0
        self.budget_report = BudgetReport(budget_ms, base_ticks_per_frame)
        self.resource_history: list[dict[str, int]] = [
            {name: faction.resources for name, faction in self.engine.state.factions.items()}
        ]
        self._persisted_ticks: set[int] = set()
        if snapshot_dir:
            self._restore_or_seed_snapshots()

    def _persist(self) -> None:
        if not self.snapshot_dir:
            return
        for tick, snapshot in list(self.engine.snapshots.items()):
            if tick not in self._persisted_ticks:
                save_snapshot(self.snapshot_dir, snapshot)
                self._persisted_ticks.add(tick)

    def _restore_or_seed_snapshots(self) -> None:
        snapshot = nearest_snapshot_at_or_before(
            self.snapshot_dir, self.engine.last_tick
        )
        if snapshot is None:
            self._persist()
            return
        if snapshot.format_version != SNAPSHOT_FORMAT_VERSION:
            raise RtsError(
                ErrorCode.E_SNAPSHOT_STALE,
                "persisted snapshot has an unsupported format version",
                tick=snapshot.tick,
            )
        rebuilt = _engine_from_state(
            self.config, self.events, snapshot, self.snapshot_interval
        )
        self.engine = rebuilt
        self._persisted_ticks = set(self.engine.snapshots)
        self.resource_history = _rebuild_resource_history(rebuilt)

    @property
    def current_tick(self) -> int:
        return self.engine.state.tick

    @property
    def finished(self) -> bool:
        return self.current_tick >= self.engine.last_tick

    def _remaining_ticks(self) -> int:
        return len(self.engine._tick_order) - self.engine._cursor

    def set_speed(self, speed: int) -> None:
        if speed not in SPEED_MULTIPLIERS:
            raise RtsError(
                ErrorCode.E_INPUT_MALFORMED,
                f"unsupported speed multiplier {speed}",
                context={"allowed": list(SPEED_MULTIPLIERS)},
            )
        self.speed = speed

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def step_one(self) -> GameState:
        """Manual single tick; always allowed while paused and never degraded."""
        if self.finished:
            return self.engine.state
        self.engine.step()
        self.resource_history.append(
            {name: faction.resources for name, faction in self.engine.state.factions.items()}
        )
        self._persist()
        return self.engine.state

    def run_frame(self) -> FrameStat:
        """Synthesize up to ``allowance * speed`` ticks within one frame budget."""
        attempted = min(self.allowance * self.speed, self._remaining_ticks())
        allowance_before = self.allowance
        first_tick = self.current_tick + 1 if attempted else None
        start = self.clock.now()
        synthesized_ticks: list[int] = []
        for _ in range(attempted):
            if self.finished:
                break
            tick = self.engine._tick_order[self.engine._cursor]
            self.engine.step()
            self.resource_history.append(
                {
                    name: faction.resources
                    for name, faction in self.engine.state.factions.items()
                }
            )
            synthesized_ticks.append(tick)
            if self.workload is not None:
                self.workload(tick)
        if isinstance(self.clock, ScriptedClock):
            elapsed_ms = self.clock.charge(synthesized_ticks) * 1000.0
        else:
            elapsed_ms = (self.clock.now() - start) * 1000.0

        degraded = False
        skipped = False
        note = ""
        if synthesized_ticks and elapsed_ms > self.budget_ms:
            degraded = True
            skipped = True
            self.allowance = max(1, self.allowance // 2)
            note = ErrorCode.E_BUDGET_EXCEEDED.value
            self.budget_report.over_budget_count += 1
            self.budget_report.skipped_refresh_count += 1
        elif self.allowance < self.base_ticks_per_frame:
            self.allowance = min(
                self.base_ticks_per_frame, self.allowance + RECOVERY_STEP
            )

        stat = FrameStat(
            frame_index=self.frame_index,
            ticks_attempted=attempted,
            ticks_synthesized=len(synthesized_ticks),
            elapsed_ms=elapsed_ms,
            budget_ms=self.budget_ms,
            degraded=degraded,
            skipped_refresh=skipped,
            speed=self.speed,
            allowance_before=allowance_before,
            allowance_after=self.allowance,
            first_tick=first_tick,
            last_tick=self.current_tick,
            note=note,
        )
        self.frame_index += 1
        self.budget_report.frames.append(stat)
        self.budget_report.min_allowance = min(
            self.budget_report.min_allowance, self.allowance
        )
        self.budget_report.max_elapsed_ms = max(
            self.budget_report.max_elapsed_ms, elapsed_ms
        )
        self._persist()
        return stat

    def run_to_end(self) -> BudgetReport:
        self.paused = False
        while not self.finished:
            self.run_frame()
        return self.budget_report

    def jump_to(self, target_tick: int) -> GameState:
        """Rebuild from the nearest snapshot and resume without full replay."""
        if target_tick < 0 or target_tick > self.engine.last_tick:
            raise RtsError(
                ErrorCode.E_INPUT_MALFORMED,
                "jump target tick is outside the event stream range",
                tick=target_tick,
                context={"last_tick": self.engine.last_tick},
            )

        snapshot = self._snapshot_for_jump(target_tick)
        rebuilt = _engine_from_state(
            self.config,
            self.events,
            snapshot,
            self.snapshot_interval,
        )
        rebuilt.advance_to(target_tick)
        self.engine = rebuilt
        self.resource_history = _rebuild_resource_history(rebuilt)
        self._persist()
        return self.engine.state

    def _snapshot_for_jump(self, target_tick: int):
        candidates = [
            snapshot
            for tick, snapshot in self.engine.snapshots.items()
            if tick <= target_tick
        ]
        in_memory = max(candidates, key=lambda item: item.tick, default=None)
        if self.snapshot_dir is None or in_memory is None:
            if in_memory is not None:
                return in_memory
        if self.snapshot_dir is not None:
            persisted = nearest_snapshot_at_or_before(self.snapshot_dir, target_tick)
            if persisted is not None and (
                in_memory is None or persisted.tick >= in_memory.tick
            ):
                if persisted.format_version != SNAPSHOT_FORMAT_VERSION:
                    raise RtsError(
                        ErrorCode.E_SNAPSHOT_STALE,
                        "snapshot used for jump has a stale format version",
                        tick=persisted.tick,
                    )
                return persisted
        if in_memory is not None:
            return in_memory
        return build_snapshot(GameState.initial(self.config))

    def current_view(self) -> dict:
        return {
            "tick": self.current_tick,
            "paused": self.paused,
            "speed": self.speed,
            "allowance": self.allowance,
            "state": self.engine.state.to_dict(),
            "state_hash": self.engine.state.state_hash(),
        }


def _rebuild_resource_history(engine: TickEngine) -> list[dict[str, int]]:
    history = [
        {name: faction.resources for name, faction in engine.state.factions.items()}
        if not engine.records
        else {name: 0 for name in engine.config.faction_names()}
    ]
    if not engine.records:
        return history
    scratch = TickEngine(
        engine.config, engine.events, snapshot_interval=engine.snapshot_interval
    )
    for record in engine.records:
        scratch.advance_to(record.tick)
        history.append(
            {name: faction.resources for name, faction in scratch.state.factions.items()}
        )
    return history


def _engine_from_state(
    config: RuleConfig,
    events: list[Event],
    snapshot,
    snapshot_interval: int,
) -> TickEngine:
    engine = TickEngine(config, events, snapshot_interval=snapshot_interval)
    engine.state = snapshot.state.clone()
    engine.ledger = []
    engine.records = []
    engine.snapshots = {snapshot.tick: snapshot}
    engine._cursor = len([tick for tick in engine._tick_order if tick <= snapshot.tick])
    return engine
