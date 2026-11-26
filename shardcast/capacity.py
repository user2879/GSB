"""Capacity-version views: whole replacement, stale flags and diffs."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .spec import NodeSpec, Spec


@dataclass(frozen=True)
class NodeQuota:
    node_id: str
    concurrency: int
    lease_limit: int


@dataclass(frozen=True)
class CapacityView:
    version: int
    global_budget: int
    quotas: dict[str, NodeQuota]

    def quota(self, node_id: str) -> NodeQuota:
        return self.quotas[node_id]


class CapacityManager:
    """Maintains the current view and immutable read-only backups."""

    def __init__(self, spec: Spec):
        self._spec = spec
        initial_version = min(spec.global_budget)
        self.current = self._build_view(initial_version)
        self.backups: list[CapacityView] = []
        self.diffs: list[dict] = []

    def _build_view(self, version: int) -> CapacityView:
        quotas: dict[str, NodeQuota] = {}
        for node in self._spec.nodes:
            level = node.level(version)
            quotas[node.node_id] = NodeQuota(
                node_id=node.node_id,
                concurrency=level.concurrency,
                lease_limit=level.lease_limit,
            )
        return CapacityView(
            version=version,
            global_budget=self._spec.global_budget[version],
            quotas=quotas,
        )

    def replace(self, version: int) -> tuple[CapacityView, CapacityView, dict]:
        """Whole-view replacement; never an incremental merge."""
        previous = self.current
        backup = CapacityView(
            version=previous.version,
            global_budget=previous.global_budget,
            quotas=copy.deepcopy(previous.quotas),
        )
        self.backups.append(backup)
        new_view = self._build_view(version)
        diff = self._diff(previous, new_view)
        self.diffs.append(diff)
        self.current = new_view
        return previous, new_view, diff

    @staticmethod
    def _diff(old: CapacityView, new: CapacityView) -> dict:
        changed: list[dict] = []
        for node_id in sorted(set(old.quotas) | set(new.quotas)):
            before = old.quotas.get(node_id)
            after = new.quotas.get(node_id)
            if before is None or after is None:
                changed.append(
                    {
                        "node_id": node_id,
                        "before": None if before is None else before.__dict__,
                        "after": None if after is None else after.__dict__,
                    }
                )
                continue
            if (
                before.concurrency != after.concurrency
                or before.lease_limit != after.lease_limit
            ):
                changed.append(
                    {
                        "node_id": node_id,
                        "before": before.__dict__,
                        "after": after.__dict__,
                    }
                )
        return {
            "from_version": old.version,
            "to_version": new.version,
            "global_budget": {"before": old.global_budget, "after": new.global_budget},
            "node_changes": changed,
        }


def mark_stale(record: dict, current_version: int) -> dict:
    """Return a shallow copy flagged stale when bound to an older version."""
    flagged = dict(record)
    flagged["stale"] = int(record.get("capacity_version", -1)) < current_version
    return flagged
