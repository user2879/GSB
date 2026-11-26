"""Concurrent scheduler under global/channel/batch budget constraints."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

from .errors import (
    E_CHANNEL_QUOTA_EXCEEDED,
    E_TEXT_BUDGET_EXCEEDED,
    Failure,
)
from .models import RoutingConfig, TextItem


def match_points(text: TextItem) -> int:
    """Deterministic candidate match-point cost of one text.

    Each potential start position for each distinct trigger length costs one
    point; this bounds the total candidate enumeration work before matching.
    """
    length = len(text.tokens)
    # Conservative per-text point cost scales with text length plus one point
    # per token slot, keeping it independent of library contents for stability.
    return length + 1


@dataclass
class PendingItem:
    text_id: str
    channel: str
    cost: int
    code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulePlan:
    accepted: list[TextItem] = field(default_factory=list)
    pending: list[PendingItem] = field(default_factory=list)
    consumed_global: int = 0
    consumed_channel: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [t.text_id for t in self.accepted],
            "pending": [p.to_dict() for p in self.pending],
            "consumed_global": self.consumed_global,
            "consumed_channel": dict(self.consumed_channel),
        }


class BudgetScheduler:
    """Stable-order budget reservation with a thread pool execution bound.

    Reservation decisions run single-threaded in deterministic order so that
    any concurrency level yields identical plans; only the pure per-text
    computation is handed to a bounded ThreadPoolExecutor.
    """

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config
        self.global_remaining = config.global_match_budget
        self.channel_remaining = dict(config.channel_quotas)
        self.backlog: list[PendingItem] = []

    def reset_budget(self) -> None:
        """Restore per-batch budgets while preserving the backlog queue."""
        self.global_remaining = self.config.global_match_budget
        self.channel_remaining = dict(self.config.channel_quotas)

    def _stable_order(self, texts: Iterable[TextItem]) -> list[TextItem]:
        # Backlog items always come first, then stable (channel, text_id) order.
        fresh = sorted(texts, key=lambda t: (t.channel, t.text_id))
        return fresh

    def reserve(
        self,
        texts: list[TextItem],
        carried_backlog: list[PendingItem] | None = None,
        text_lookup: dict[str, TextItem] | None = None,
    ) -> SchedulePlan:
        plan = SchedulePlan()
        plan.consumed_channel = {}
        lookup = text_lookup or {t.text_id: t for t in texts}

        # Priority: previously pending items first (stable within backlog).
        backlog = sorted(
            carried_backlog or self.backlog,
            key=lambda p: (p.channel, p.text_id),
        )
        queue: list[TextItem] = []
        for item in backlog:
            if item.text_id in lookup:
                queue.append(lookup[item.text_id])
        queued_ids = {t.text_id for t in queue}
        for text in self._stable_order(texts):
            if text.text_id not in queued_ids:
                queue.append(text)

        batch_count = 0
        for text in queue:
            if batch_count >= self.config.per_batch_recompute_limit:
                self._add_pending(
                    plan, text, E_TEXT_BUDGET_EXCEEDED, "per_batch_limit"
                )
                continue

            cost = match_points(text)
            channel = text.channel
            # Global budget.
            if cost > self.global_remaining:
                self._add_pending(
                    plan, text, E_TEXT_BUDGET_EXCEEDED, "global_budget"
                )
                continue
            # Channel quota.
            channel_cap = self.channel_remaining.get(channel)
            if channel_cap is not None and cost > channel_cap:
                self._add_pending(
                    plan, text, E_CHANNEL_QUOTA_EXCEEDED, "channel_quota"
                )
                continue
            self._consume(plan, channel, cost)
            plan.accepted.append(text)
            batch_count += 1

        self.backlog = list(plan.pending)
        return plan

    def _consume(self, plan: SchedulePlan, channel: str, cost: int) -> None:
        self.global_remaining -= cost
        self.channel_remaining[channel] = (
            self.channel_remaining.get(channel, 0) - cost
        )
        plan.consumed_global += cost
        plan.consumed_channel[channel] = (
            plan.consumed_channel.get(channel, 0) + cost
        )

    def _add_pending(
        self,
        plan: SchedulePlan,
        text: TextItem,
        code: str,
        reason: str,
    ) -> None:
        item = PendingItem(
            text_id=text.text_id,
            channel=text.channel,
            cost=match_points(text),
            code=code,
            reason=reason,
        )
        plan.pending.append(item)

    def run(
        self,
        texts: list[TextItem],
        worker: Callable[[TextItem], Any],
        carried_backlog: list[PendingItem] | None = None,
        text_lookup: dict[str, TextItem] | None = None,
    ) -> tuple[list[Any], SchedulePlan]:
        """Reserve budget then execute accepted texts in a bounded thread pool."""
        lookup = text_lookup or {t.text_id: t for t in texts}
        plan = self.reserve(texts, carried_backlog, lookup)
        executable: list[TextItem] = list(plan.accepted)

        results: list[Any] = []
        if executable:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
                futures = [pool.submit(worker, item) for item in executable]
                for future in futures:
                    results.append(future.result())
        # Stable result order regardless of completion order.
        order = {t.text_id: i for i, t in enumerate(executable)}
        results.sort(key=lambda r: order.get(getattr(r, "text_id", ""), 1 << 30))
        return results, plan

    def failures(self, plan: SchedulePlan) -> list[Failure]:
        return [
            Failure(
                code=item.code,
                message=f"deferred: {item.reason}",
                entity_type="text",
                entity_id=item.text_id,
                detail={"channel": item.channel, "cost": item.cost},
            )
            for item in plan.pending
        ]
