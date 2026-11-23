"""Numpy-backed diagnostic histograms and reason restoration statistics."""

import numpy as np


def deviation_histogram(deviations):
    """Histogram of integer score deviations after serialize round trips."""
    values = np.asarray(deviations, dtype=np.int64)
    if values.size == 0:
        return {"bins": [], "counts": [], "max_abs": 0}
    unique, counts = np.unique(values, return_counts=True)
    return {
        "bins": [int(v) for v in unique],
        "counts": [int(c) for c in counts],
        "max_abs": int(np.abs(values).max()),
        "mean_abs": float(np.abs(values).astype(float).mean()),
    }


def reason_restore_stats(ok_count, total_count):
    if total_count == 0:
        return {"ok": 0, "total": 0, "rate": 1.0}
    return {
        "ok": int(ok_count),
        "total": int(total_count),
        "rate": round(ok_count / total_count, 6),
    }
