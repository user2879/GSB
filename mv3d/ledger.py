"""Deterministic invalidation and reuse ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import STAGE_ORDER


@dataclass
class Ledger:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        stage: str,
        artifact: str,
        sort_key: str,
        action: str,
        reason: str,
        key: str,
    ) -> None:
        self.entries.append(
            {
                "stage": stage,
                "artifact": artifact,
                "sort_key": sort_key,
                "action": action,
                "reason": reason,
                "key": key,
            }
        )

    def extend(self, entries: list[dict[str, Any]]) -> None:
        self.entries.extend(entries)

    def normalized(self) -> list[dict[str, Any]]:
        stage_index = {stage: index for index, stage in enumerate(STAGE_ORDER)}
        return sorted(
            self.entries,
            key=lambda item: (
                stage_index.get(item["stage"], len(STAGE_ORDER)),
                item["sort_key"],
                item["artifact"],
                item["action"],
                item["reason"],
            ),
        )

    def summary(self) -> dict[str, Any]:
        normalized = self.normalized()
        by_stage: dict[str, dict[str, int]] = {}
        by_action: dict[str, int] = {}
        for entry in normalized:
            stage_stats = by_stage.setdefault(
                entry["stage"],
                {"computed": 0, "invalidated": 0, "reused": 0},
            )
            stage_stats[entry["action"]] = stage_stats.get(entry["action"], 0) + 1
            by_action[entry["action"]] = by_action.get(entry["action"], 0) + 1
        return {
            "entry_count": len(normalized),
            "by_stage": by_stage,
            "by_action": by_action,
            "reuse_hits": by_action.get("reused", 0),
            "invalidations": by_action.get("invalidated", 0),
            "computations": by_action.get("computed", 0),
        }

