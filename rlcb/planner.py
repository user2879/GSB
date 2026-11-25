"""Full vs incremental recompute planning with explicit selection rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Explicit thresholds; the boundary values are reported alongside the choice.
INCREMENTAL_SERVICE_LIMIT = 4      # services > this favor incremental savings
INCREMENTAL_BATCH_LIMIT = 6        # at least this many batches makes reuse matter
INCREMENTAL_WINDOW_LIMIT = 8       # many windows make full recompute costly
# Incremental is only chosen when the changed fraction is at or below this
# fraction of (service, batch) work units.
MAX_CHANGED_FRACTION_NUM = 1
MAX_CHANGED_FRACTION_DEN = 2


@dataclass(frozen=True)
class Plan:
    mode: str
    reason: str
    boundaries: dict[str, int]
    changed_services: tuple[str, ...]
    changed_batches: tuple[str, ...]
    unchanged_batches: tuple[str, ...]
    metrics: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "boundaries": self.boundaries,
            "changed_services": list(self.changed_services),
            "changed_batches": list(self.changed_batches),
            "unchanged_batches": list(self.unchanged_batches),
            "metrics": self.metrics,
        }


def choose_plan(*, service_count: int, batch_count: int, window_count: int,
                changed_services: list[str],
                changed_batches: list[str],
                all_batches: list[str],
                force_mode: str | None = None) -> Plan:
    boundaries = {
        "incremental_service_limit": INCREMENTAL_SERVICE_LIMIT,
        "incremental_batch_limit": INCREMENTAL_BATCH_LIMIT,
        "incremental_window_limit": INCREMENTAL_WINDOW_LIMIT,
        "max_changed_fraction_num": MAX_CHANGED_FRACTION_NUM,
        "max_changed_fraction_den": MAX_CHANGED_FRACTION_DEN,
    }
    metrics = {
        "service_count": service_count,
        "batch_count": batch_count,
        "window_count": window_count,
        "changed_service_count": len(changed_services),
        "changed_batch_count": len(changed_batches),
    }
    unchanged = sorted(set(all_batches) - set(changed_batches))

    if force_mode in ("full", "incremental"):
        return Plan(
            mode=force_mode,
            reason=f"forced by caller ({force_mode})",
            boundaries=boundaries,
            changed_services=tuple(sorted(changed_services)),
            changed_batches=tuple(sorted(changed_batches)),
            unchanged_batches=tuple(unchanged),
            metrics=metrics,
        )

    work_units = max(1, service_count * max(1, batch_count))
    changed_units = max(1, len(changed_batches))
    fraction_ok = (
        changed_units * MAX_CHANGED_FRACTION_DEN
        <= work_units * MAX_CHANGED_FRACTION_NUM
    )
    scale_ok = (
        service_count > INCREMENTAL_SERVICE_LIMIT
        or batch_count >= INCREMENTAL_BATCH_LIMIT
        or window_count >= INCREMENTAL_WINDOW_LIMIT
    )
    if changed_services and fraction_ok and scale_ok:
        reason = (
            f"changed_fraction {changed_units}/{work_units} <= 1/2 and scale "
            f"threshold met (services>{INCREMENTAL_SERVICE_LIMIT} or "
            f"batches>={INCREMENTAL_BATCH_LIMIT} or windows>={INCREMENTAL_WINDOW_LIMIT})"
        )
        mode = "incremental"
    else:
        if not changed_services:
            reason = "no parameter changes; full recompute for a clean baseline"
        elif not fraction_ok:
            reason = (f"changed_fraction {changed_units}/{work_units} > 1/2; "
                      "recompute dominates, full run is cheaper")
        else:
            reason = ("workload below scale thresholds; fixed overhead of "
                      "invalidation exceeds a full run")
        mode = "full"
    return Plan(
        mode=mode,
        reason=reason,
        boundaries=boundaries,
        changed_services=tuple(sorted(changed_services)),
        changed_batches=tuple(sorted(changed_batches)),
        unchanged_batches=tuple(unchanged),
        metrics=metrics,
    )
