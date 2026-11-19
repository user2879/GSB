"""Independent verification: canonical references and divergence checks."""

import json

import numpy as np

from .errors import PipelineError
from .fixedpoint import mean_scaled, to_scaled
from .recovery import _first_diff, state_digest
from .state import _aggregate_window


def canonical_agg_results(tables, query_set, param_version, scale, direction):
    """Order-independent grouped aggregation computed in one pass."""
    results = {}
    for query in query_set.queries:
        if query.kind != "aggregate":
            continue
        table = tables[query.table]
        value_col = table.column(query.value_column)
        key_cols = [table.column(c) for c in query.group_by]
        groups = {}
        for idx in range(table.row_count):
            if value_col.is_null(idx) or any(c.is_null(idx)
                                             for c in key_cols):
                continue
            key = json.dumps([c.get(idx) for c in key_cols],
                             ensure_ascii=False)
            scaled = to_scaled(value_col.get(idx), scale, direction)
            entry = groups.setdefault(key, [])
            entry.append(scaled)
        results[query.qid] = {
            key: {"count": len(vals), "sum": sum(vals),
                  "min": min(vals), "max": max(vals),
                  "mean": mean_scaled(sum(vals), len(vals), direction),
                  "rows_processed": len(vals),
                  "group_version": param_version}
            for key, vals in groups.items()}
    return results


def canonical_window_results(tables, query_set, scale, direction):
    """Order-independent window results from fully sorted data."""
    results = {}
    for query in query_set.queries:
        if query.kind != "window":
            continue
        table = tables[query.table]
        ts_col = table.column(table.timestamp_column)
        value_col = table.column(query.value_column)
        key_cols = [table.column(c) for c in query.partition_by]
        partitions = {}
        for idx in range(table.row_count):
            if ts_col.is_null(idx) or value_col.is_null(idx) or \
                    any(c.is_null(idx) for c in key_cols):
                continue
            key = json.dumps([c.get(idx) for c in key_cols],
                             ensure_ascii=False)
            partitions.setdefault(key, []).append(
                (ts_col.get(idx), idx,
                 to_scaled(value_col.get(idx), scale, direction)))
        query_results = {}
        for key, rows in partitions.items():
            ts_arr = np.array([r[0] for r in rows], dtype=np.int64)
            seq_arr = np.array([r[1] for r in rows], dtype=np.int64)
            order = np.lexsort((seq_arr, ts_arr))
            sorted_rows = [rows[i] for i in order]
            preceding = query.interval["preceding"]
            following = query.interval["following"]
            row_results = {}
            for ts, seq, _val in sorted_rows:
                members = [r for r in sorted_rows
                           if ts - preceding <= r[0] <= ts + following]
                window = members[-query.max_rows:]
                row_results[str(seq)] = _aggregate_window(
                    [m[2] for m in window], direction)
            query_results[key] = row_results
        results[query.qid] = query_results
    return results


def check_window_results(engine_results, tables, query_set, scale,
                         direction):
    """E_WINDOW: locate the first row deviating from the canonical result."""
    canonical = canonical_window_results(tables, query_set, scale, direction)
    for qid, parts in canonical.items():
        engine_parts = engine_results.get(qid, {})
        for key, rows in parts.items():
            engine_rows = engine_parts.get(key, {})
            for row_seq in sorted(rows, key=lambda s: int(s)):
                expected = rows[row_seq]
                actual = engine_rows.get(row_seq)
                if actual != expected:
                    raise PipelineError(
                        "E_WINDOW", "window result deviates from canonical",
                        query=qid, partition_key=key, row=row_seq,
                        expected=expected, actual=actual)
    return True


def check_agg_results(engine_aggs, tables, query_set, param_version, scale,
                      direction):
    canonical = canonical_agg_results(tables, query_set, param_version,
                                      scale, direction)
    for qid, groups in canonical.items():
        engine_groups = engine_aggs.get(qid, {})
        for key in sorted(groups):
            expected = groups[key]
            actual = engine_groups.get(key)
            if actual is None:
                raise PipelineError("E_DIVERGE", "missing aggregation group",
                                    query=qid, group_key=key)
            for field in ("count", "sum", "min", "max", "mean",
                          "rows_processed"):
                if actual.get(field) != expected[field]:
                    raise PipelineError(
                        "E_DIVERGE", "aggregation result mismatch",
                        query=qid, group_key=key, field=field,
                        expected=expected[field], actual=actual.get(field))
    return True


def compare_runs(engine_a, engine_b):
    """E_DIVERGE: bitwise comparison of two run outcomes."""
    res_a, res_b = engine_a.results(), engine_b.results()
    if res_a == res_b and \
            state_digest(engine_a.state) == state_digest(engine_b.state):
        return True
    path = _first_diff(res_a, res_b, "") or \
        _first_diff(engine_a.state.to_dict(), engine_b.state.to_dict(), "")
    details = {"first_diff_path": path}
    parts = [p for p in (path or "").split("/") if p]
    if "aggs" in parts:
        idx = parts.index("aggs")
        if len(parts) > idx + 2:
            details["group_key"] = parts[idx + 2]
        if len(parts) > idx + 3:
            details["field"] = parts[idx + 3].split("[")[0]
    if "windows" in parts:
        idx = parts.index("windows")
        if len(parts) > idx + 2:
            details["partition_key"] = parts[idx + 2]
        if len(parts) > idx + 3:
            details["row"] = parts[idx + 3].split("[")[0]
            row = details["row"]
            if row.isdigit():
                details["timestamp"] = _row_timestamp(engine_a, row)
    raise PipelineError("E_DIVERGE",
                        "resumed run diverges from uninterrupted run",
                        **details)


def compare_results(engine_a, engine_b):
    """E_DIVERGE on results only: invariant across batch sizes and
    arrival orders (bookkeeping such as applied intervals is excluded)."""
    res_a, res_b = engine_a.results(), engine_b.results()
    if res_a == res_b:
        return True
    path = _first_diff(res_a, res_b, "")
    raise PipelineError("E_DIVERGE",
                        "results differ across batch/arrival combinations",
                        first_diff_path=path)


def _row_timestamp(engine, row_seq):
    for table in engine.tables.values():
        idx = int(row_seq)
        if idx < table.row_count:
            return table.column(table.timestamp_column).get(idx)
    return None
