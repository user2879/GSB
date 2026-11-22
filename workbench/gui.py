"""Tkinter diagnostic workbench with a headless-testable view model.

Tkinter is imported lazily so the rest of the toolkit (and the test suite)
works on machines without the Tk native library installed. On standard
Windows CPython, Tkinter ships with the installer and :func:`launch` works
directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .latency import FixedPointStats
from .ledger import Ledger
from .models import MinuteDecision


@dataclass(frozen=True)
class DecisionRow:
    minute: int
    service_id: str
    requests: int
    demand: int
    allocated: int
    rejected: int
    trim_reason: str


@dataclass(frozen=True)
class CurvePoint:
    minute: int
    total_allocated: int
    total_rejected: int
    budget: int


class WorkbenchViewModel:
    """Pure data shaping; no Tk dependency, fully deterministic ordering."""

    def __init__(self, decisions: list[MinuteDecision],
                 latency_overall: FixedPointStats,
                 latency_by_service: dict[str, FixedPointStats],
                 snapshots: list[dict],
                 stale_versions: list[int]) -> None:
        self.decisions = sorted(decisions, key=lambda item: item.minute)
        self.latency_overall = latency_overall
        self.latency_by_service = dict(sorted(latency_by_service.items()))
        self.snapshots = sorted(snapshots, key=lambda item: (item["version"], item["minute"]))
        self.stale_versions = sorted(stale_versions)

    def decision_rows(self) -> list[DecisionRow]:
        rows: list[DecisionRow] = []
        for decision in self.decisions:
            estimate_map = {item.service_id: item for item in decision.estimates}
            for allocation in sorted(decision.allocations, key=lambda item: item.service_id):
                estimate = estimate_map[allocation.service_id]
                rows.append(
                    DecisionRow(
                        minute=decision.minute,
                        service_id=allocation.service_id,
                        requests=estimate.requests,
                        demand=allocation.demand,
                        allocated=allocation.allocated,
                        rejected=allocation.rejected,
                        trim_reason=allocation.trim_reason,
                    )
                )
        return rows

    def budget_curve(self) -> list[CurvePoint]:
        points: list[CurvePoint] = []
        for decision in self.decisions:
            points.append(
                CurvePoint(
                    minute=decision.minute,
                    total_allocated=decision.total_allocated,
                    total_rejected=sum(item.rejected for item in decision.allocations),
                    budget=decision.global_budget,
                )
            )
        return points

    def trim_reason_rows(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            for allocation in decision.allocations:
                counts[allocation.trim_reason] = counts.get(allocation.trim_reason, 0) + 1
        return sorted(counts.items(), key=lambda item: item[0])

    def snapshot_rows(self) -> list[tuple[int, int, str]]:
        return [
            (int(item["version"]), int(item["minute"]), str(item["digest"])[:16])
            for item in self.snapshots
        ]


def build_view_model(ledger: Ledger, version: int, recorder) -> WorkbenchViewModel:
    decisions = ledger.load_decisions(version)
    overall = recorder.aggregate()
    by_service = recorder.aggregate_by_service()
    snapshots = ledger.load_snapshots(version)
    return WorkbenchViewModel(
        decisions=decisions,
        latency_overall=overall,
        latency_by_service=by_service,
        snapshots=snapshots,
        stale_versions=ledger.stale_versions(),
    )


def launch(ledger: Ledger, version: int, recorder) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Tkinter is unavailable in this environment. On Windows install "
            "standard CPython (Tcl/Tk bundled); headless Linux needs python3-tk."
        ) from exc

    view = build_view_model(ledger, version, recorder)
    root = tk.Tk()
    root.title("Model Service Elastic-Scaling Workbench")
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    decision_tab = ttk.Frame(notebook)
    notebook.add(decision_tab, text="Decisions")
    columns = ("minute", "service", "requests", "demand", "allocated", "rejected", "reason")
    tree = ttk.Treeview(decision_tab, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
    for row in view.decision_rows():
        tree.insert("", tk.END, values=(
            row.minute, row.service_id, row.requests, row.demand,
            row.allocated, row.rejected, row.trim_reason,
        ))
    tree.pack(fill=tk.BOTH, expand=True)

    curve_tab = ttk.Frame(notebook)
    notebook.add(curve_tab, text="Budget Curve")
    canvas = tk.Canvas(curve_tab, bg="white", width=760, height=320)
    canvas.pack(fill=tk.BOTH, expand=True)
    points = view.budget_curve()
    if points:
        width, height = 760, 320
        max_budget = max(point.budget for point in points) or 1
        step_x = max(1, (width - 80) // max(1, len(points)))
        budget_coords, used_coords = [], []
        for index, point in enumerate(points):
            x = 40 + index * step_x
            budget_coords.extend([x, height - 40])
            used_coords.extend([
                x,
                height - 40 - int(point.total_allocated * (height - 80) / max_budget),
            ])
        canvas.create_line(*budget_coords, fill="#999999", width=2)
        canvas.create_line(*used_coords, fill="#1f77b4", width=2)
        canvas.create_text(60, 20, text="blue=allocated grey=budget", fill="#333333")

    reject_tab = ttk.Frame(notebook)
    notebook.add(reject_tab, text="Reject & Trim")
    reject_tree = ttk.Treeview(
        reject_tab, columns=("minute", "service", "rejected", "reason"), show="headings"
    )
    for column in ("minute", "service", "rejected", "reason"):
        reject_tree.heading(column, text=column)
    for row in view.decision_rows():
        reject_tree.insert("", tk.END, values=(
            row.minute, row.service_id, row.rejected, row.trim_reason,
        ))
    reject_tree.pack(fill=tk.BOTH, expand=True)

    latency_tab = ttk.Frame(notebook)
    notebook.add(latency_tab, text="Latency")
    stats = view.latency_overall.as_dict() if view.latency_overall.count else {
        "note": "no latency samples"
    }
    tk.Label(latency_tab, text=str(stats), justify=tk.LEFT, anchor="w").pack(
        fill=tk.X, padx=8, pady=8
    )
    for service_id, service_stats in view.latency_by_service.items():
        tk.Label(
            latency_tab,
            text=f"{service_id}: {service_stats.as_dict()}",
            justify=tk.LEFT,
            anchor="w",
        ).pack(fill=tk.X, padx=8)

    snapshot_tab = ttk.Frame(notebook)
    notebook.add(snapshot_tab, text="Snapshots")
    snapshot_tree = ttk.Treeview(
        snapshot_tab, columns=("version", "minute", "digest16"), show="headings"
    )
    for column in ("version", "minute", "digest16"):
        snapshot_tree.heading(column, text=column)
    for version, minute, digest16 in view.snapshot_rows():
        snapshot_tree.insert("", tk.END, values=(version, minute, digest16))
    snapshot_tree.pack(fill=tk.BOTH, expand=True)
    if view.stale_versions:
        tk.Label(
            snapshot_tab,
            text=f"stale versions (excluded from current view): {view.stale_versions}",
            fg="#aa3333",
        ).pack(anchor="w", padx=8, pady=4)

    root.mainloop()
