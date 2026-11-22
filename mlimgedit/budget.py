"""Per-batch feature-point budget scheduler with replayable deductions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .errors import E_FEATURE_BUDGET_PENDING


@dataclass(frozen=True)
class FeatureTask:
    sample: str
    cost: int
    sort_key: str

    @property
    def stage(self) -> str:
        return "features"


@dataclass
class BudgetDeduction:
    sequence: int
    sample: str
    cost: int
    remaining_before: int
    remaining_after: int

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "sample": self.sample,
            "cost": self.cost,
            "remaining_before": self.remaining_before,
            "remaining_after": self.remaining_after,
        }


@dataclass
class PendingItem:
    sample: str
    stage: str
    code: str
    message: str
    sort_key: str

    def to_dict(self) -> dict:
        return {
            "sample": self.sample,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "sort_key": self.sort_key,
        }


@dataclass
class BudgetSchedule:
    ready: list[FeatureTask] = field(default_factory=list)
    pending: list[FeatureTask] = field(default_factory=list)
    deductions: list[BudgetDeduction] = field(default_factory=list)
    pending_records: list[PendingItem] = field(default_factory=list)
    initial_budget: int = 0
    remaining_budget: int = 0


def make_feature_tasks(sample_names: list[str], cost: int) -> list[FeatureTask]:
    names = sorted(sample_names)
    return [
        FeatureTask(sample=name, cost=cost, sort_key=f"features|0|{name}")
        for name in names
    ]


def schedule_budget(
    tasks: list[FeatureTask], budget: int, carried_pending: list[FeatureTask] | None = None
) -> BudgetSchedule:
    """Consume budget in stable (label/sample) order.

    Carried pending tasks are always placed first. A task that does not fit is
    retained together with every later task; already completed deductions are
    never altered.
    """
    ordered = sorted(carried_pending or [], key=lambda task: task.sort_key)
    ordered += sorted(tasks, key=lambda task: task.sort_key)
    # De-duplicate by sample while preserving the first stable occurrence.
    seen: set[str] = set()
    unique: list[FeatureTask] = []
    for task in ordered:
        if task.sample in seen:
            continue
        seen.add(task.sample)
        unique.append(task)

    schedule = BudgetSchedule(initial_budget=budget, remaining_budget=budget)
    sequence = 0
    exhausted = False
    for task in unique:
        if exhausted:
            schedule.pending.append(task)
            schedule.pending_records.append(
                PendingItem(
                    sample=task.sample,
                    stage="features",
                    code=E_FEATURE_BUDGET_PENDING,
                    message=(
                        "feature point budget exhausted; task retained for the "
                        "next batch"
                    ),
                    sort_key=task.sort_key,
                )
            )
            continue
        before = schedule.remaining_budget
        if task.cost > schedule.remaining_budget:
            exhausted = True
            schedule.pending.append(task)
            schedule.pending_records.append(
                PendingItem(
                    sample=task.sample,
                    stage="features",
                    code=E_FEATURE_BUDGET_PENDING,
                    message=(
                        f"task cost {task.cost} exceeds remaining budget "
                        f"{schedule.remaining_budget}"
                    ),
                    sort_key=task.sort_key,
                )
            )
            continue
        schedule.remaining_budget -= task.cost
        schedule.deductions.append(
            BudgetDeduction(
                sequence=sequence,
                sample=task.sample,
                cost=task.cost,
                remaining_before=before,
                remaining_after=schedule.remaining_budget,
            )
        )
        sequence += 1
        schedule.ready.append(task)
    return schedule
