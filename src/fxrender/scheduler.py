"""Single-batch denoise point budget scheduler with retained todo queue."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .codes import BizCode
from .policy import Policy, merged_step_groups, tile_decode_order


@dataclass(frozen=True)
class WorkItem:
    phase: str          # "denoise" or "decode"
    frame_id: int
    step: int
    tile_row: int
    group_id: int
    merged: bool
    cost: int

    @property
    def sort_key(self) -> tuple:
        # Phase first so retained denoise work always precedes new decode.
        return (self.phase, self.frame_id, self.step, self.tile_row,
                self.group_id)


@dataclass
class BatchResult:
    processed: list[WorkItem] = field(default_factory=list)
    pending: list[WorkItem] = field(default_factory=list)
    deductions: list[dict] = field(default_factory=list)
    pending_events: list[dict] = field(default_factory=list)
    budget_remaining: int = 0


def build_denoise_items(frame_count: int, cfg, policy: Policy) -> list[WorkItem]:
    items: list[WorkItem] = []
    groups = merged_step_groups(cfg.denoise_steps, policy)
    for frame_id in range(frame_count):
        for step, group_id, merged in groups:
            items.append(WorkItem(
                phase="denoise", frame_id=frame_id, step=step, tile_row=-1,
                group_id=group_id, merged=merged,
                cost=cfg.points_per_step))
    return sorted(items, key=lambda item: item.sort_key)


def build_decode_items(frame_count: int, cfg, policy: Policy) -> list[WorkItem]:
    order = tile_decode_order(cfg.tile_rows, policy)
    items: list[WorkItem] = []
    for frame_id in range(frame_count):
        for tile_row in order:
            items.append(WorkItem(
                phase="decode", frame_id=frame_id, step=-1,
                tile_row=tile_row, group_id=-1, merged=False, cost=0))
    return sorted(items, key=lambda item: item.sort_key)


class Scheduler:
    """Allocates one budget batch at a time and retains over-budget work."""

    def __init__(self, cfg, policy: Policy, frame_count: int) -> None:
        self.cfg = cfg
        self.policy = policy
        self.frame_count = frame_count
        self.todo: list[WorkItem] = build_denoise_items(frame_count, cfg, policy)
        self.total_consumed = 0
        self.deductions: list[dict] = []
        self.pending_events: list[dict] = []
        self._deducted_keys: set[tuple] = set()

    @property
    def denoise_done(self) -> bool:
        return all(item.phase != "denoise" for item in self.todo)

    def append_decode_phase(self) -> None:
        if self.denoise_done:
            self.todo.extend(build_decode_items(
                self.frame_count, self.cfg, self.policy))
            self.todo.sort(key=lambda item: item.sort_key)

    def run_batch(self, budget: int | None = None) -> BatchResult:
        """Consume one fresh budget; over-budget steps stay retained.

        Each call receives a refreshed point budget while the retained todo
        queue survives across batches, so pending work is revisited first and
        already-deducted points are never charged again.
        """
        budget = self.cfg.batch_budget if budget is None else budget
        remaining = budget
        processed: list[WorkItem] = []
        pending: list[WorkItem] = []
        deductions: list[dict] = []
        events: list[dict] = []
        first_blocker: WorkItem | None = None
        for item in list(self.todo):
            key = item.sort_key
            if item.phase == "decode":
                processed.append(item)
                self.todo.remove(item)
                continue
            if item.cost <= remaining:
                remaining -= item.cost
                processed.append(item)
                self.todo.remove(item)
                if key not in self._deducted_keys:
                    record = {"key": list(key), "cost": item.cost,
                              "remaining": remaining, "frame_id": item.frame_id,
                              "step": item.step, "merged": item.merged}
                    deductions.append(record)
                    self.deductions.append(record)
                    self._deducted_keys.add(key)
                    self.total_consumed += item.cost
            else:
                # Stable order: this item and every later one remain queued.
                first_blocker = item
                event = {"code": BizCode.E_BUDGET_PENDING.value,
                         "key": list(key), "cost": item.cost,
                         "remaining": remaining}
                events.append(event)
                self.pending_events.append(event)
                break
        if first_blocker is not None:
            index = self.todo.index(first_blocker)
            pending = list(self.todo[index:])
        return BatchResult(processed, pending, deductions, events, remaining)

    def state(self) -> dict:
        return {
            "todo": [asdict(item) for item in self.todo],
            "total_consumed": self.total_consumed,
            "deductions": self.deductions,
            "pending_events": self.pending_events,
        }

    def load_state(self, state: dict) -> None:
        self.todo = [WorkItem(**item) for item in state["todo"]]
        self.total_consumed = state["total_consumed"]
        self.deductions = list(state["deductions"])
        self.pending_events = list(state["pending_events"])
        self._deducted_keys = {
            tuple(record["key"]) for record in self.deductions}
