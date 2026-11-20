from __future__ import annotations

import numpy as np

from .model import Config


def summarize_ledger(entries, cfg: Config):
    """Per-unit peak usage, overflow counts and per-step resource curve."""
    per_unit = {}
    for entry in entries:
        per_unit.setdefault(entry["unit_id"], []).append(int(entry["balance"]))
    summary = {}
    for unit_id in sorted(per_unit):
        curve = np.asarray(per_unit[unit_id], dtype=np.int64)
        summary[unit_id] = {
            "peak_balance": int(curve.max()),
            "min_balance": int(curve.min()),
            "overflows": int(np.count_nonzero((curve < 0) | (curve > cfg.supply_cap))),
            "curve": [int(v) for v in curve],
        }
    return summary
