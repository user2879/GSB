"""Small local Tkinter viewer for diagnostic reports."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .pipeline import PipelinePaths, run_pipeline


class DiagnosticApp:
    def __init__(
        self,
        root: tk.Tk,
        *,
        report_path: Path,
        run_paths: PipelinePaths | None = None,
    ) -> None:
        self.root = root
        self.report_path = report_path
        self.run_paths = run_paths
        self.report: dict[str, Any] = {}
        root.title("IntentLab Diagnostic Viewer")
        root.geometry("1180x760")
        self._build()
        self.load_report()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Reload report", command=self.load_report).pack(side="left")
        if self.run_paths is not None:
            ttk.Button(toolbar, text="Run pipeline", command=self.run_and_reload).pack(side="left", padx=6)
        self.summary_var = tk.StringVar(value="No report loaded")
        ttk.Label(toolbar, textvariable=self.summary_var).pack(side="left", padx=16)

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.notebook = notebook

        query_frame = ttk.Frame(notebook)
        notebook.add(query_frame, text="Queries")
        columns = ("query", "session", "source", "intent", "boundary", "reused", "segment")
        self.query_tree = ttk.Treeview(query_frame, columns=columns, show="headings")
        headings = {
            "query": "Query",
            "session": "Session",
            "source": "Source",
            "intent": "Intent",
            "boundary": "Boundary",
            "reused": "Reused M/S",
            "segment": "Segment",
        }
        for column in columns:
            self.query_tree.heading(column, text=headings[column])
            self.query_tree.column(column, width=150 if column != "segment" else 210)
        scrollbar = ttk.Scrollbar(query_frame, orient="vertical", command=self.query_tree.yview)
        self.query_tree.configure(yscrollcommand=scrollbar.set)
        self.query_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.detail = tk.Text(notebook, wrap="none")
        notebook.add(self.detail, text="Query detail")
        self.query_tree.bind("<<TreeviewSelect>>", self.show_selected_detail)

        self.segments = tk.Text(notebook, wrap="none")
        notebook.add(self.segments, text="Segments / ledger")

    def load_report(self) -> None:
        try:
            self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.summary_var.set(f"Cannot load {self.report_path}: {exc}")
            return
        summary = self.report.get("summary", {})
        self.summary_var.set(
            f"processed {summary.get('queries_processed', 0)}/{summary.get('queries_loaded', 0)} · "
            f"segments {summary.get('segments', 0)} · boundary {summary.get('boundary_intents', 0)} · "
            f"reuse {summary.get('reuse_hits', 0)}"
        )
        self.query_tree.delete(*self.query_tree.get_children())
        for index, item in enumerate(self.report.get("queries", [])):
            self.query_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    item.get("query_id"),
                    item.get("session_id"),
                    item.get("source"),
                    item.get("score", {}).get("predicted_intent"),
                    item.get("score", {}).get("boundary"),
                    f"{item.get('match_reused')}/{item.get('score_reused')}",
                    item.get("segment_id"),
                ),
            )
        self.detail.delete("1.0", "end")
        self.segments.delete("1.0", "end")
        self.segments.insert(
            "end",
            json.dumps(
                {
                    "segments": self.report.get("segments", []),
                    "ledger": self.report.get("ledger", {}),
                    "pending": self.report.get("pending", []),
                    "issues": self.report.get("issues", []),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )

    def show_selected_detail(self, _event: object) -> None:
        selection = self.query_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        queries = self.report.get("queries", [])
        if not 0 <= index < len(queries):
            return
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", json.dumps(queries[index], ensure_ascii=False, indent=2, sort_keys=True))

    def run_and_reload(self) -> None:
        if self.run_paths is None:
            return
        try:
            result = run_pipeline(self.run_paths)
        except Exception as exc:
            messagebox.showerror("Pipeline failed", f"{type(exc).__name__}: {exc}")
            return
        self.report_path = result.json_path
        self.load_report()


def main(*, report_path: Path | None = None, paths: PipelinePaths | None = None) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"Tkinter unavailable: {exc}")
        return 2
    chosen = report_path or Path("reports") / "latest.json"
    DiagnosticApp(root, report_path=chosen, run_paths=paths)
    root.mainloop()
    return 0

