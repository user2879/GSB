"""Scheduler wait statistics; numpy is used when available."""

from __future__ import annotations

from typing import Any, Iterable


def wait_statistics(wait_records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = [int(record["wait_ms"]) for record in wait_records]
    if not values:
        return {"count": 0, "backend": "none"}

    ordered = sorted(values)
    summary = {
        "count": len(values),
        "min_ms": ordered[0],
        "median_ms": ordered[len(ordered) // 2],
        "max_ms": ordered[-1],
    }
    try:
        import numpy as np

        array = np.asarray(values, dtype=np.int64)
        summary.update(
            {
                "backend": "numpy",
                "mean_ms": float(array.mean()),
                "p95_ms": float(np.percentile(array, 95)),
            }
        )
    except ImportError:
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        total = sum(ordered)
        summary.update(
            {
                "backend": "pure-python",
                "mean_ms": total / len(ordered),
                "p95_ms": ordered[index],
            }
        )
    return summary
