"""Scale-adaptive, deterministic selection among the three computation paths."""

from __future__ import annotations

from typing import Any

from .engine import PATH_DIFF, PATH_FULL, PATH_INCREMENTAL

PATH_PRIORITY = (PATH_FULL, PATH_INCREMENTAL, PATH_DIFF)
TIE_DELTA_RATIO = 0.05
MERGE_FACTOR = 0.25

try:
    import numpy as _numpy

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    _numpy = None
    NUMPY_AVAILABLE = False


def estimate_work(
    scale: dict[str, int],
    change_ratio: float,
) -> dict[str, dict[str, float]]:
    """Pure integer/float cost model; numpy only trims the vector arithmetic."""

    service_count = max(1, scale["service_count"])
    instance_count = max(1, scale["instance_count"])
    rows = max(0, scale["observation_rows"])
    round_span = max(1, scale["round_span"])
    ratio = min(1.0, max(0.0, change_ratio))
    touched = max(1.0, instance_count * ratio)
    touched_services = max(1.0, service_count * ratio)

    costs = {
        PATH_FULL: {
            "state_work": float(round_span * instance_count),
            "decision_work": float(round_span * service_count),
            "diff_work": 0.0,
        },
        PATH_INCREMENTAL: {
            "state_work": float(rows),
            "decision_work": float(round_span * touched_services),
            "diff_work": 0.0,
        },
        PATH_DIFF: {
            "state_work": float(rows),
            "decision_work": float(round_span * touched_services),
            "diff_work": float(round_span * touched * MERGE_FACTOR),
        },
    }
    if NUMPY_AVAILABLE:
        for name in costs:
            vector = _numpy.array(
                [
                    costs[name]["state_work"],
                    costs[name]["decision_work"],
                    costs[name]["diff_work"],
                ],
                dtype=_numpy.float64,
            )
            costs[name]["estimated_total"] = float(_numpy.sum(vector))
    else:
        for name in costs:
            costs[name]["estimated_total"] = sum(costs[name].values())
    return costs


def select_path(
    scale: dict[str, int],
    change_ratio: float,
    tie_delta_ratio: float = TIE_DELTA_RATIO,
) -> dict[str, Any]:
    costs = estimate_work(scale, change_ratio)
    totals = {name: costs[name]["estimated_total"] for name in PATH_PRIORITY}
    minimum = min(totals.values())
    near_tie = [
        name
        for name in PATH_PRIORITY
        if totals[name] - minimum <= minimum * tie_delta_ratio
    ]
    chosen = near_tie[0]
    influenced = len(near_tie) > 1
    ranking = [
        {
            "path": name,
            "estimated_total": totals[name],
            "components": costs[name],
        }
        for name in sorted(PATH_PRIORITY, key=lambda item: (totals[item], PATH_PRIORITY.index(item)))
    ]
    reasons = [
        f"minimum estimated work among {list(PATH_PRIORITY)}",
        f"change_ratio={change_ratio:.4f}",
    ]
    if influenced:
        reasons.append(
            f"tie band +-{tie_delta_ratio:.2%} applied; stable priority {list(PATH_PRIORITY)} resolved to {chosen}"
        )
    return {
        "selected_path": chosen,
        "threshold_influenced": influenced,
        "tie_delta_ratio": tie_delta_ratio,
        "scale": dict(scale),
        "change_ratio": change_ratio,
        "ranking": ranking,
        "reasons": reasons,
        "numpy_used": NUMPY_AVAILABLE,
    }


def change_ratio_from_diffs(diffs: list[dict[str, Any]], instance_count: int) -> float:
    if not diffs or instance_count <= 0:
        return 0.0
    changed = sum(len(diff["changed_instances"]) for diff in diffs)
    total = len(diffs) * instance_count
    return changed / total if total else 0.0
