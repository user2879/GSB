"""Tkinter workbench with a headless view-model fallback.

The GUI only reads the engine; refreshing it never writes the ledger. When
Tcl/Tk is unavailable (for example a minimal Linux CI image), importing this
module still works and :class:`ScalerWorkbench` exposes the same view model
through its headless mode so the demo and tests run everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stats import LatencyRecorder


@dataclass(frozen=True)
class ServiceRow:
    minute: int
    service: str
    requests: int
    raw_demand: int
    demand: int
    allocated_replicas: int
    rejected_requests: int
    trim_reasons: tuple[str, ...]


def build_view_model(engine: Any) -> dict[str, Any]:
    """Stable-order, read-only projection of the current ledger view."""
    run_id = engine.run_id
    minutes = engine.ledger.committed_minutes(run_id)
    rows: list[ServiceRow] = []
    budget_curve: list[dict[str, int]] = []
    for minute in minutes:
        decision = engine.fast_query(minute)
        budget_curve.append({
            "minute": minute,
            "budget_used": decision.budget_used,
            "budget": decision.budget,
            "budget_remaining": decision.budget_remaining,
            "total_rejected": decision.total_rejected,
        })
        for alloc in decision.allocations:  # lexicographic service order
            rows.append(ServiceRow(
                minute=minute,
                service=alloc.service,
                requests=alloc.estimate.requests,
                raw_demand=alloc.estimate.raw_demand,
                demand=alloc.estimate.demand,
                allocated_replicas=alloc.allocated_replicas,
                rejected_requests=alloc.rejected_requests,
                trim_reasons=alloc.trim_reasons,
            ))
    return {
        "run_id": run_id,
        "spec_version": engine.spec_version,
        "cache_basis_spec_version": engine.cache_basis_version,
        "spec_digest": engine.spec.digest if engine.spec else "",
        "minutes": minutes,
        "service_order": [svc.name for svc in engine.spec.services],
        "service_rows": [row.__dict__ for row in rows],
        "budget_curve": budget_curve,
        "versions": engine.list_versions(),
        "snapshots": engine.ledger.list_snapshots(),
    }


class ScalerWorkbench:
    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.overall_recorder = LatencyRecorder()
        self.per_service_recorders: dict[str, LatencyRecorder] = {}
        self._tk = None
        self._root = None
        self.headless = False
        try:
            import tkinter as tk  # lazy import: core never requires a display
            self._tk = tk
        except Exception:  # pragma: no cover - environment dependent
            self.headless = True

    def refresh(self) -> dict[str, Any]:
        """One measured read; identical inputs always produce identical rows."""
        with self.overall_recorder.measure():
            view = build_view_model(self.engine)
        for service in view["service_order"]:
            recorder = self.per_service_recorders.setdefault(service, LatencyRecorder())
            with recorder.measure():
                self.engine.fast_query(view["minutes"][-1]) if view["minutes"] else None
        return view

    def latency_report(self) -> dict[str, Any]:
        per_service = {
            name: recorder.snapshot().to_dict()
            for name, recorder in sorted(self.per_service_recorders.items())
        }
        return {
            "overall": self.overall_recorder.snapshot().to_dict(),
            "per_service": per_service,
            "clock": "time.perf_counter_ns (monotonic)",
            "unit": "millisecond fixed-point (3 decimals)",
            "quantization_error_bound": "each raw sample floored to microseconds, "
                                        "per-sample error < 1 microsecond",
        }

    def run(self) -> dict[str, Any] | None:
        if self.headless:
            return self.refresh()
        self._root = self._tk.Tk()
        self._root.title("Elastic Scaler Workbench")
        self._render(self.refresh())
        self._root.mainloop()
        return None

    def _render(self, view: dict[str, Any]) -> None:  # pragma: no cover - GUI
        tk = self._tk
        frame = tk.Frame(self._root, padx=8, pady=8)
        frame.pack(fill="both", expand=True)

        header = tk.Label(
            frame,
            f"run={view['run_id']}  spec_version={view['spec_version']}  "
            f"cache_basis={view['cache_basis_spec_version']}  digest={view['spec_digest'][:12]}",
            anchor="w",
        )
        header.pack(fill="x")

        columns = ("minute", "service", "requests", "raw_demand", "demand",
                   "allocated", "rejected", "reasons")
        tree = _treeview(tk, frame, columns)
        for row in view["service_rows"]:
            tree.insert(
                "", "end",
                values=(
                    row["minute"], row["service"], row["requests"],
                    row["raw_demand"], row["demand"], row["allocated_replicas"],
                    row["rejected_requests"], ",".join(row["trim_reasons"]),
                ),
            )

        curve_lines = [
            f"m {point['minute']:>2}: used {point['budget_used']}/{point['budget']} "
            f"rejected={point['total_rejected']}"
            for point in view["budget_curve"]
        ]
        curve = tk.Label(frame, text="\n".join(curve_lines), justify="left", anchor="w")
        curve.pack(fill="x", pady=(8, 0))

        version_lines = [
            f"v{item['spec_version']} {item['status']} {item['label']} {item['digest'][:10]}"
            for item in view["versions"]
        ]
        snapshot_lines = [
            f"{item['snapshot_id']} -> v{item['spec_version']} minute {item['target_minute']}"
            for item in view["snapshots"]
        ]
        tk.Label(frame, text="VERSIONS\n" + "\n".join(version_lines),
                 justify="left", anchor="w").pack(fill="x", pady=(8, 0))
        tk.Label(frame, text="SNAPSHOTS\n" + "\n".join(snapshot_lines),
                 justify="left", anchor="w").pack(fill="x", pady=(8, 0))

        tk.Button(frame, text="Refresh (read-only)",
                  command=lambda: self._render(self.refresh())).pack(pady=(8, 0))


def _treeview(tk: Any, parent: Any, columns: tuple[str, ...]) -> Any:  # pragma: no cover
    from tkinter import ttk

    tree = ttk.Treeview(parent, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=110, anchor="w")
    tree.pack(fill="both", expand=True)
    return tree
