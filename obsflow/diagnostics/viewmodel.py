"""Headless view model backing the Tkinter diagnostics UI.

The view model is pure: rendering refreshes never mutate the ledger. The Tk
widget layer is imported lazily so the package works on machines without a
Tk runtime (it degrades to this structured view model).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..lease import LeaseManager
from ..models import WindowResult


@dataclass(frozen=True, slots=True)
class WindowRow:
    metric_key: str
    window_index: int
    start_seq: int
    end_seq: int
    count: int
    sum_q: int
    mean_q: int
    value_error_bound_q: int
    sum_error_bound_q: int
    window_version: int
    status: str
    lease_state: str
    lease_id: Optional[str]


def build_window_rows(windows: list[WindowResult],
                      lease_manager: LeaseManager) -> list[WindowRow]:
    rows: list[WindowRow] = []
    for window in sorted(windows, key=lambda w: (w.metric_key,
                                                 w.window_index)):
        lease = lease_manager.active_lease(window.metric_key,
                                           window.window_index)
        if lease is None:
            all_leases = {l.lease_id: l for l in lease_manager.all_leases()}
            lease = all_leases.get(window.lease_id) if window.lease_id else None
        rows.append(WindowRow(
            metric_key=window.metric_key,
            window_index=window.window_index,
            start_seq=window.start_seq,
            end_seq=window.end_seq,
            count=window.count,
            sum_q=window.sum_q,
            mean_q=window.mean_q,
            value_error_bound_q=window.value_error_bound_q,
            sum_error_bound_q=window.sum_error_bound_q,
            window_version=window.window_version,
            status=window.status,
            lease_state=lease.state if lease is not None else "none",
            lease_id=lease.lease_id if lease is not None else None,
        ))
    return rows


def render_text_table(rows: list[WindowRow]) -> str:
    headers = ("metric", "win", "start", "end", "count", "sum_q", "mean_q",
               "val_err<", "sum_err<", "ver", "status", "lease")
    table = [headers]
    for row in rows:
        table.append((
            row.metric_key, str(row.window_index), str(row.start_seq),
            str(row.end_seq), str(row.count), str(row.sum_q),
            str(row.mean_q), str(row.value_error_bound_q),
            str(row.sum_error_bound_q), str(row.window_version),
            row.status, row.lease_state,
        ))
    widths = [max(len(line[i]) for line in table)
              for i in range(len(headers))]
    lines = []
    for line_no, line in enumerate(table):
        lines.append("  ".join(cell.ljust(widths[i])
                               for i, cell in enumerate(line)))
        if line_no == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
