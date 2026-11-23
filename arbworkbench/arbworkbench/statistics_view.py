"""NumPy-backed observation distribution statistics.

NumPy is used strictly for read-only display statistics (mean / spread /
median over quantized observations). All arbitration decisions remain pure
Python integer arithmetic, so NumPy version differences cannot affect
authority values, accepted sets or rejection reasons.
"""
from __future__ import annotations

import numpy as np

from .engine import KeyConclusion


def distribution_stats(kc: KeyConclusion) -> dict:
    scalar_values: list[int] = []
    interval_widths: list[int] = []
    for env_id, rows in sorted(kc.distribution_q.items()):
        for row in rows:
            qv = row["q"]
            if isinstance(qv, list):
                scalar_values.extend(qv)
                interval_widths.append(qv[1] - qv[0])
            else:
                scalar_values.append(qv)
    if not scalar_values:
        return {"sample_count": 0}
    arr = np.asarray(scalar_values, dtype=np.int64)
    stats = {
        "sample_count": int(arr.size),
        "min_q": int(arr.min()),
        "max_q": int(arr.max()),
        "mean_q": float(arr.mean()),
        "median_q": float(np.median(arr)),
        "spread_q": int(arr.max() - arr.min()),
        "std_q": float(arr.std(ddof=0)),
    }
    if interval_widths:
        widths = np.asarray(interval_widths, dtype=np.int64)
        stats["interval_width_min_q"] = int(widths.min())
        stats["interval_width_max_q"] = int(widths.max())
        stats["interval_width_mean_q"] = float(widths.mean())
    return stats


def session_distribution_report(session) -> dict:
    report: dict[str, dict[str, dict]] = {}
    for view_id in sorted(session.conclusions):
        report[view_id] = {}
        for key, kc in sorted(session.conclusions[view_id].keys.items()):
            report[view_id][key] = distribution_stats(kc)
    return report
