from __future__ import annotations

from typing import Any

from .renderer import (
    render_affected,
    render_budget,
    render_chains,
    render_decisions,
    render_overview,
    render_schedule,
)


class UiUnavailable(RuntimeError):
    pass


def launch_ui(workbench: Any, report: dict[str, Any] | None = None) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - depends on Windows/Tk installation
        raise UiUnavailable(f"tkinter is unavailable: {exc}") from exc

    try:
        root = tk.Tk()
    except Exception as exc:  # pragma: no cover - headless CI
        raise UiUnavailable(f"Tk window could not start: {exc}") from exc

    root.title("MVCC Visibility Diagnostics Workbench")
    root.geometry("1180x760")
    root.minsize(920, 600)
    initial = report if report is not None else workbench.refresh_view()

    frame = ttk.Frame(root, padding=8)
    frame.pack(fill="both", expand=True)
    toolbar = ttk.Frame(frame)
    toolbar.pack(fill="x", pady=(0, 6))
    status_var = tk.StringVar(value=f"status={initial.get('status')} code={initial.get('code')}")
    ttk.Label(toolbar, textvariable=status_var).pack(side="left")

    notebook = ttk.Notebook(frame)
    notebook.pack(fill="both", expand=True)
    tabs: dict[str, tk.Text] = {}
    titles = ("Overview", "Chains", "Decisions", "Schedule", "Budget & Ledger", "Affected")
    for title in titles:
        text = tk.Text(notebook, wrap="none", font=("Consolas", 10), undo=False)
        text.configure(state="disabled")
        notebook.add(text, text=title)
        tabs[title] = text

    def apply_report(current: dict[str, Any]) -> None:
        contents = {
            "Overview": render_overview(current),
            "Chains": render_chains(current, workbench.dataset),
            "Decisions": render_decisions(current),
            "Schedule": render_schedule(current),
            "Budget & Ledger": render_budget(current),
            "Affected": render_affected(current),
        }
        for title_text, text_widget in tabs.items():
            text_widget.configure(state="normal")
            text_widget.delete("1.0", "end")
            text_widget.insert("1.0", contents[title_text])
            text_widget.configure(state="disabled")
        status_var.set(f"status={current.get('status')} code={current.get('code')}")

    def refresh_only() -> None:
        apply_report(workbench.refresh_view())

    def execute_run() -> None:
        apply_report(workbench.execute())

    ttk.Button(toolbar, text="Refresh view (no ledger write)", command=refresh_only).pack(side="right", padx=(6, 0))
    ttk.Button(toolbar, text="Run / resume", command=execute_run).pack(side="right")
    apply_report(initial)
    root.mainloop()
