"""Deterministic partition and global watermark advancement strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .canonical import canonical_json
from .errors import WatermarkError
from .spec import StrategySpec


@dataclass(frozen=True)
class WatermarkPoint:
    event_seq: int
    event_id: str
    phase: str
    partition_label: str
    partition_json: str
    partition_watermark: int
    global_watermark: int
    strategy: str

    def to_db_dict(self) -> dict[str, object]:
        return {
            "event_seq": self.event_seq,
            "event_id": self.event_id,
            "phase": self.phase,
            "partition_label": self.partition_label,
            "partition_json": self.partition_json,
            "partition_watermark": self.partition_watermark,
            "global_watermark": self.global_watermark,
            "strategy": self.strategy,
        }


class WatermarkTracker:
    """Maintains per-partition watermarks and one global logical watermark.

    Missing partitions start at zero. This is intentional: ``partition_min``
    cannot advance until every declared partition has supplied an event, and
    the fixed-step strategy can only catch up by its configured step.
    """

    def __init__(
        self,
        strategy: StrategySpec,
        partitions: Mapping[str, dict[str, object]],
    ) -> None:
        self.strategy = strategy
        self.partition_json = {
            label: canonical_json(dict(payload))
            for label, payload in sorted(partitions.items())
        }
        self.partition_watermarks = {
            label: 0 for label in sorted(self.partition_json)
        }
        self.global_watermark = 0

    def _recompute_global(self) -> int:
        values = [
            self.partition_watermarks[label]
            for label in sorted(self.partition_watermarks)
        ]
        if not values:
            return self.global_watermark
        if self.strategy.kind == "source_max":
            return max(values)
        if self.strategy.kind == "partition_min":
            return min(values)
        if self.strategy.kind == "fixed_step":
            target = min(values)
            step = int(self.strategy.step_ms or 1)
            return min(target, self.global_watermark + step)
        raise WatermarkError(
            "unknown watermark strategy",
            details={"strategy": self.strategy.kind},
        )

    def advance(self, partition_label: str, event_time_ms: int) -> int:
        if partition_label not in self.partition_watermarks:
            raise WatermarkError(
                "event references an unregistered partition",
                details={"partition_label": partition_label},
            )
        self.partition_watermarks[partition_label] = max(
            self.partition_watermarks[partition_label], int(event_time_ms)
        )
        candidate = self._recompute_global()
        self.validate_candidate(candidate)
        self.global_watermark = candidate
        return candidate

    def validate_candidate(self, candidate: int) -> None:
        values = [
            self.partition_watermarks[label]
            for label in sorted(self.partition_watermarks)
        ]
        if not values:
            return
        if self.strategy.kind == "source_max" and candidate > max(values):
            raise WatermarkError(
                "source_max watermark exceeded the maximum partition watermark",
                details={
                    "candidate": candidate,
                    "maximum_partition_watermark": max(values),
                    "strategy": self.strategy.kind,
                },
            )
        if self.strategy.kind == "partition_min" and candidate < min(values):
            raise WatermarkError(
                "partition_min watermark fell below a partition watermark",
                details={
                    "candidate": candidate,
                    "minimum_partition_watermark": min(values),
                    "strategy": self.strategy.kind,
                },
            )
        if self.strategy.kind == "fixed_step":
            target = min(values)
            step = int(self.strategy.step_ms or 1)
            if candidate < self.global_watermark:
                raise WatermarkError(
                    "fixed_step watermark moved backwards",
                    details={
                        "candidate": candidate,
                        "previous_global_watermark": self.global_watermark,
                        "strategy": self.strategy.kind,
                    },
                )
            if candidate - self.global_watermark > step:
                raise WatermarkError(
                    "fixed_step watermark advanced by more than one configured step",
                    details={
                        "candidate": candidate,
                        "previous_global_watermark": self.global_watermark,
                        "step_ms": step,
                        "strategy": self.strategy.kind,
                    },
                )
            if candidate > target:
                raise WatermarkError(
                    "fixed_step watermark exceeded the partition-min target",
                    details={
                        "candidate": candidate,
                        "target": target,
                        "strategy": self.strategy.kind,
                    },
                )

    def inject_watermark_gap(self, partition_label: str) -> int:
        """Return an intentionally invalid candidate and validate it."""

        values = [
            self.partition_watermarks[label]
            for label in sorted(self.partition_watermarks)
        ]
        if self.strategy.kind == "source_max":
            illegal = max(values) + int(self.strategy.step_ms or 1)
        elif self.strategy.kind == "partition_min":
            illegal = max(0, min(values) - 1)
        else:
            illegal = self.global_watermark + int(self.strategy.step_ms or 1) * 2
        try:
            self.validate_candidate(illegal)
        except WatermarkError as exc:
            exc.details.update(
                {
                    "partition_label": partition_label,
                    "injected": True,
                    "illegal_candidate": illegal,
                }
            )
            raise
        raise WatermarkError(
            "watermark anomaly injection did not create an invalid gap",
            details={
                "partition_label": partition_label,
                "injected": True,
                "candidate": illegal,
            },
        )

    def snapshot(
        self,
        *,
        event_seq: int,
        event_id: str,
        phase: str,
        global_override: int | None = None,
    ) -> list[WatermarkPoint]:
        points: list[WatermarkPoint] = []
        for label in sorted(self.partition_json):
            points.append(
                WatermarkPoint(
                    event_seq=event_seq,
                    event_id=event_id,
                    phase=phase,
                    partition_label=label,
                    partition_json=self.partition_json[label],
                    partition_watermark=self.partition_watermarks[label],
                    global_watermark=(
                        self.global_watermark
                        if global_override is None
                        else int(global_override)
                    ),
                    strategy=self.strategy.kind,
                )
            )
        return points

    def flush_globals(self, target: int) -> list[int]:
        """Return logical end-of-stream watermark levels for finalization."""

        target = max(self.global_watermark, int(target))
        if self.strategy.kind != "fixed_step":
            self.global_watermark = target
            return [target]
        step = int(self.strategy.step_ms or 1)
        levels: list[int] = []
        current = self.global_watermark
        while current < target:
            current = min(target, current + step)
            levels.append(current)
        if not levels:
            levels = [current]
        self.global_watermark = levels[-1]
        return levels
