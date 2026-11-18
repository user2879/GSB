"""float64 reference path and numerical error metrics."""

import numpy as np

from .fixedpoint import dequantize_scalar, ulp


def reference_gather(table, ids, mode):
    """float64 reference aggregation in ID sequence order."""
    acc = np.zeros(table.config.dim, dtype=np.float64)
    count = 0
    for wid in ids:
        row = table.rows.get(wid)
        if row is None:
            return None
        acc += np.asarray(row.vector, dtype=np.float64)
        count += 1
    if mode == "mean" and count:
        acc = acc / count
    return acc


def result_metrics(result, table):
    """Max abs error vs float64 and ID positions with rounding deviation
    beyond half an ulp (i.e. clipping or abnormal rounding)."""
    if result.status != "ok":
        return {"max_abs_error": None, "deviation_positions": []}
    ref = reference_gather(table, result.ids, result.mode)
    if ref is None:
        return {"max_abs_error": None, "deviation_positions": []}
    fixed = np.asarray(
        [dequantize_scalar(q, result.frac_bits) for q in result.acc],
        dtype=np.float64)
    max_abs_error = float(np.max(np.abs(fixed - ref))) if ref.size else 0.0
    half_ulp = 0.5 * ulp(result.frac_bits)
    deviations = []
    for pos, (wid, contrib) in enumerate(
            zip(result.ids, result.contributions)):
        row = table.rows.get(wid)
        if row is None:
            continue
        deq = np.asarray(
            [dequantize_scalar(q, result.frac_bits) for q in contrib],
            dtype=np.float64)
        err = np.max(np.abs(deq - np.asarray(row.vector,
                                             dtype=np.float64)))
        if err > half_ulp + 1e-15:
            deviations.append({
                "position": pos,
                "word_id": wid,
                "max_element_error": float(err),
            })
    return {
        "max_abs_error": max_abs_error,
        "deviation_positions": deviations,
    }
