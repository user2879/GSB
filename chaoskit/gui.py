"""Tkinter diagnostic window for an experiment ledger.

The window renders, in stable order: ledger identity and version, the timeline
with injection status, a success-rate/latency curve with error-bound glyphs,
unknown injections and whole-snapshot replacements. Tkinter is imported lazily
so importing this module never breaks headless deployments or unit tests.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict

from .errors import ChaosError
from .viewmodel import build_view_model, render_text_report


def _load_view(ledger_path: str) -> Dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        raise ChaosError(f"ledger not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        return build_view_model(conn)
    finally:
        conn.close()


def render_headless(ledger_path: str) -> str:
    return render_text_report(_load_view(ledger_path))


def launch(ledger_path: str) -> None:  # pragma: no cover - requires a display
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise ChaosError(
            "tkinter is unavailable on this interpreter; use render_headless() instead",
            details={"ledger_path": ledger_path},
        ) from exc

    view = _load_view(ledger_path)
    root = tk.Tk()
    root.title(f"Chaos Ledger - {view['experiment_id']}")
    root.geometry("980x720")

    header = ttk.Frame(root, padding=(8, 6))
    header.pack(fill="x")
    ttk.Label(
        header,
        text=(
            f"{view['experiment_id']}  |  ledger v{view['ledger_format_version']} "
            f"spec v{view['spec_version']}  seed={view['seed']}  "
            f"unknown={view['unknown_indexes'] or 'none'}"
        ),
        font=("TkDefaultFont", 10, "bold"),
    ).pack(anchor="w")
    ttk.Label(header, text=f"timeline digest: {view['timeline_digest']}").pack(anchor="w")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=6)

    timeline_tab = ttk.Frame(notebook)
    notebook.add(timeline_tab, text="Timeline & Status")
    timeline_box = tk.Listbox(timeline_tab, font=("Courier", 10))
    timeline_box.pack(fill="both", expand=True)
    for row in view["timeline_rows"]:
        timeline_box.insert("end", row.render())

    chart_tab = ttk.Frame(notebook)
    notebook.add(chart_tab, text="Observation Curve")
    canvas = tk.Canvas(chart_tab, bg="white")
    canvas.pack(fill="both", expand=True)
    _draw_curve(canvas, view)

    obs_tab = ttk.Frame(notebook)
    notebook.add(obs_tab, text="Observations")
    obs_box = tk.Listbox(obs_tab, font=("Courier", 10))
    obs_box.pack(fill="both", expand=True)
    for point in view["observation_points"]:
        obs_box.insert("end", point.render())

    replace_tab = ttk.Frame(notebook)
    notebook.add(replace_tab, text="Ledger Replacements")
    replace_box = tk.Listbox(replace_tab, font=("Courier", 10))
    replace_box.pack(fill="both", expand=True)
    if view["replacements"]:
        for event in view["replacements"]:
            replace_box.insert(
                "end",
                f"v{event['old_format_version']} -> v{event['new_format_version']} "
                f"skipped={event['skipped_injections']} backup={event['backup_path']}",
            )
    else:
        replace_box.insert("end", "no replacement events")

    root.mainloop()


def _draw_curve(canvas: "tk.Canvas", view: Dict[str, Any]) -> None:  # pragma: no cover
    import tkinter as tk

    width, height = 920, 480
    left, right, top, bottom = 60, 20, 30, 40
    canvas.create_text((width // 2, 14), text="success rate (bps) and p95 latency (ms)",
                       fill="#222")
    canvas.create_line(left, top, left, height - bottom, fill="#333")
    canvas.create_line(left, height - bottom, width - right, height - bottom, fill="#333")

    valid = [point for point in view["observation_points"] if not point.insufficient]
    if not valid:
        canvas.create_text((width // 2, height // 2),
                           text="insufficient samples: no curve fabricated", fill="#888")
        return

    span = max(1, len(valid) - 1)
    x_step = (width - left - right) / span

    def x_pos(index: int) -> float:
        return left + index * x_step

    max_latency = max(point.max_ms or 1 for point in valid)
    plot_height = height - top - bottom

    rate_points, latency_points = [], []
    for index, point in enumerate(valid):
        x = x_pos(index)
        rate_y = top + plot_height * (1 - (point.success_rate_bps or 0) / 10000)
        rate_points.append((x, rate_y))
        latency_y = top + plot_height * (1 - (point.p95_ms or 0) / max(1, max_latency))
        latency_points.append((x, latency_y))
        # Error-bound tick for the rate, using the stored bps quantization bound.
        tick = plot_height * (point.rate_error_bps / 10000)
        canvas.create_line(x, rate_y - tick, x, rate_y + tick, fill="#1f77b4")

    for series, color in ((rate_points, "#1f77b4"), (latency_points, "#d62728")):
        if len(series) > 1:
            canvas.create_line(*[coordinate for point in series for coordinate in point], fill=color)
        for x, y in series:
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)

    canvas.create_text((12, top), text="10000bps", fill="#1f77b4", anchor="w")
    canvas.create_text((12, height - bottom), text="0bps", fill="#1f77b4", anchor="w")
    canvas.create_text((width - right, top), text=f"p95 max {max_latency}ms",
                       fill="#d62728", anchor="e")
    legend = "blue=success rate (+/- bound ticks)   red=p95 latency"
    canvas.create_text((width // 2, height - 12), text=legend, fill="#444")
