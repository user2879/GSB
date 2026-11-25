"""Optional Tkinter dashboard for a run report.

Tk is part of the Python standard library but needs native Tcl/Tk shared
libraries and, for a real window, a display. When no display is available the
dashboard falls back to a structured text rendering instead of crashing.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_report(path: str) -> dict:
    return json.loads(Path(path).read_text("utf-8"))


def render_text(report: dict) -> str:
    lines = []
    plan = report.get("plan", {})
    lines.append(f"mode={plan.get('mode')}  reason={plan.get('reason')}")
    lines.append(f"decisions={len(report.get('decisions', []))}  "
                 f"digest={report.get('content_digest')}")
    outcomes: dict[str, int] = {}
    for decision in report.get("decisions", []):
        outcomes[decision["outcome"]] = outcomes.get(decision["outcome"], 0) + 1
    lines.append("outcomes=" + json.dumps(outcomes, sort_keys=True))
    lines.append("lease_slots=" + str(len(report.get("lease_ownership", {}))))
    inv = report.get("invalidation", {})
    lines.append("affected_requests="
                 + str(len(inv.get("affected_requests", []))))
    return "\n".join(lines)


def launch_ui(report_path: str) -> int:
    report = _load_report(report_path)
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        print(render_text(report))
        print(f"[ui] tkinter unavailable ({exc}); text fallback used")
        return 0
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(render_text(report))
        print(f"[ui] no display ({exc}); text fallback used")
        return 0

    root.title("RLCB run dashboard")
    frame = ttk.Frame(root, padding=10)
    frame.grid(row=0, column=0, sticky="nsew")

    ttk.Label(frame, text="Rate-limit / breaker chain",
              font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0,
                                                        sticky="w")
    ttk.Label(frame, text=render_text(report), justify="left").grid(
        row=1, column=0, sticky="w", pady=(8, 8))

    columns = ("request", "service", "outcome", "window", "rate_bp")
    table = ttk.Treeview(frame, columns=columns, show="headings", height=16)
    headers = {"request": "Request", "service": "Service",
               "outcome": "Outcome", "window": "Window", "rate_bp": "Rate(bp)"}
    for column in columns:
        table.heading(column, text=headers[column])
    for decision in report.get("decisions", [])[:200]:
        table.insert("", "end", values=(
            decision["request_id"], decision["service_id"],
            decision["outcome"], decision["window_index"],
            decision["failure_rate_bp"],
        ))
    table.grid(row=2, column=0, sticky="nsew")
    root.mainloop()
    return 0
