"""Per-partition and global numeric statistics with fixed-point error bounds."""

import numpy as np

from . import fixedpoint as fp


def _numeric_columns(projections, tables):
    out = []
    for ref in projections:
        table_name, col_name = ref.split(".")
        dtype = tables[table_name].dtype_map[col_name]
        out.append((ref, dtype))
    return out


def _column_values(rows, ref, dtype):
    if dtype == "float":
        float_values = np.array(
            [
                row["projection"][ref]["float_ref"]
                if row["projection"][ref]["float_ref"] is not None
                else np.nan
                for row in rows
            ],
            dtype=np.float64,
        )
        q_values = np.array(
            [row["projection"][ref]["fixed_point"] for row in rows],
            dtype=np.int64,
        )
        return float_values, q_values
    q_values = np.array(
        [row["projection"][ref]["fixed_point"] for row in rows],
        dtype=np.int64,
    )
    return None, q_values


def _trunc_mean_q(sum_q, count):
    if count == 0:
        return 0.0
    truncated = sum_q // count if sum_q >= 0 else -((-sum_q) // count)
    return truncated


def compute_stats(rows, projections, tables, fraction_bits):
    """Aggregate count/sum/fixed-point mean over projection numeric columns."""
    numeric = _numeric_columns(projections, tables)
    stats = {"row_count": len(rows), "columns": {}}
    for ref, dtype in numeric:
        float_values, q_values = _column_values(rows, ref, dtype)
        count = int(len(rows))
        sum_q = int(np.sum(q_values)) if count else 0
        mean_q = _trunc_mean_q(sum_q, count)
        entry = {
            "dtype": dtype,
            "count": count,
            "sum_fixed_point": sum_q,
            "mean_fixed_point_q": mean_q,
            "mean_fixed_point": fp.dequantize(mean_q, fraction_bits),
        }
        if dtype == "float":
            finite = float_values[~np.isnan(float_values)]
            sum_float = float(np.sum(finite))
            entry.update(
                {
                    "sum_float_ref": sum_float,
                    "quantization_error_bound": fp.error_bound(fraction_bits),
                    "sum_error_bound": fp.sum_error_bound(count, fraction_bits),
                    "actual_sum_error": (
                        fp.dequantize(sum_q, fraction_bits) - sum_float
                        if count else 0.0
                    ),
                }
            )
        else:
            entry.update(
                {
                    "sum_float_ref": float(sum_q >> fraction_bits)
                    if fraction_bits else float(sum_q),
                    "quantization_error_bound": 0.0,
                    "sum_error_bound": 0.0,
                    "actual_sum_error": 0.0,
                }
            )
        stats["columns"][ref] = entry
    return stats


def merge_partition_stats(partition_stats, projections, tables, fraction_bits):
    """Merge partition stats into global stats without reading rows twice."""
    numeric = _numeric_columns(projections, tables)
    total_rows = sum(s["row_count"] for s in partition_stats.values())
    global_stats = {"row_count": total_rows, "columns": {}}
    for ref, dtype in numeric:
        count = sum(s["columns"][ref]["count"] for s in partition_stats.values())
        sum_float = sum(
            s["columns"][ref]["sum_float_ref"] for s in partition_stats.values()
        )
        sum_q = sum(
            s["columns"][ref]["sum_fixed_point"]
            for s in partition_stats.values()
        )
        mean_q = _trunc_mean_q(sum_q, count)
        entry = {
            "dtype": dtype,
            "count": count,
            "sum_float_ref": sum_float,
            "sum_fixed_point": int(sum_q),
            "mean_fixed_point_q": mean_q,
            "mean_fixed_point": fp.dequantize(mean_q, fraction_bits),
            "quantization_error_bound": (
                fp.error_bound(fraction_bits) if dtype == "float" else 0.0
            ),
            "sum_error_bound": (
                fp.sum_error_bound(count, fraction_bits)
                if dtype == "float"
                else 0.0
            ),
            "actual_sum_error": (
                fp.dequantize(sum_q, fraction_bits) - sum_float
                if count and dtype == "float"
                else 0.0
            ),
        }
        global_stats["columns"][ref] = entry
    return global_stats
