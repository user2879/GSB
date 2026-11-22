from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .canonical import normalize
from .ledger import Ledger
from .pipeline import run_pipeline
from .spec import load_spec
from .verify import verify_ledger


class QPlanWindow(tk.Tk):
    def __init__(
        self,
        *,
        default_spec: str | None = None,
        workspace: str = "workspace",
        ledger: str | None = None,
        database: str | None = None,
    ) -> None:
        super().__init__()
        self.title("qplan - local query plan and index diagnostics")
        self.geometry("1280x820")
        self.minsize(980, 620)
        self.result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.report: dict[str, Any] | None = None
        workspace_path = Path(workspace)
        self.spec_var = tk.StringVar(value=default_spec or "")
        self.workspace_var = tk.StringVar(value=str(workspace_path))
        self.ledger_var = tk.StringVar(value=ledger or str(workspace_path / "qplan_ledger.sqlite"))
        self.db_var = tk.StringVar(value=database or str(workspace_path / "sample.sqlite"))
        self.status_var = tk.StringVar(value="Ready. Select a declaration JSON and run.")
        self._build_ui()
        self.after(120, self._poll_queue)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Spec JSON").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Entry(top, textvariable=self.spec_var).grid(row=0, column=1, sticky=tk.EW, padx=4)
        ttk.Button(top, text="Browse...", command=self._browse_spec).grid(row=0, column=2, padx=4)
        ttk.Label(top, text="Workspace").grid(row=1, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        ttk.Entry(top, textvariable=self.workspace_var).grid(row=1, column=1, sticky=tk.EW, padx=4, pady=(6, 0))
        ttk.Button(top, text="Browse...", command=self._browse_workspace).grid(row=1, column=2, padx=4, pady=(6, 0))
        ttk.Label(top, text="Ledger").grid(row=2, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        ttk.Entry(top, textvariable=self.ledger_var).grid(row=2, column=1, sticky=tk.EW, padx=4, pady=(6, 0))
        ttk.Label(top, text="Sample DB").grid(row=3, column=0, sticky=tk.W, padx=(0, 6), pady=(6, 0))
        ttk.Entry(top, textvariable=self.db_var).grid(row=3, column=1, sticky=tk.EW, padx=4, pady=(6, 0))
        top.columnconfigure(1, weight=1)

        buttons = ttk.Frame(self, padding=(10, 0, 10, 8))
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text="Validate Spec", command=self.validate_spec).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(buttons, text="Run Diagnostics", command=self.run_diagnostics).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Refresh Ledger", command=self.refresh_ledger).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Recover", command=self.recover).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Independent Verify", command=self.verify).pack(side=tk.LEFT, padx=6)
        ttk.Label(buttons, textvariable=self.status_var).pack(side=tk.LEFT, padx=16, fill=tk.X, expand=True)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.summary_text = self._add_text_tab(notebook, "Summary")
        self.query_tree = self._add_tree_tab(
            notebook,
            "Queries",
            ("query", "selected", "kind", "est_cost", "actual_ms", "rows", "result", "bias"),
            (130, 180, 120, 90, 100, 80, 80, 170),
        )
        self.index_tree = self._add_tree_tab(
            notebook,
            "Index Search",
            ("index", "table", "columns", "status", "reason"),
            (220, 120, 220, 90, 360),
        )
        self.diag_tree = self._add_tree_tab(
            notebook,
            "Diagnostics",
            ("code", "query", "detail"),
            (130, 130, 800),
        )
        self.json_text = self._add_text_tab(notebook, "JSON")
        self.ledger_text = self._add_text_tab(notebook, "Ledger")

    def _add_text_tab(self, notebook: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10))
        y_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        x_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
        text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        text.grid(row=0, column=0, sticky=tk.NSEW)
        y_scroll.grid(row=0, column=1, sticky=tk.NS)
        x_scroll.grid(row=1, column=0, sticky=tk.EW)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return text

    def _add_tree_tab(
        self,
        notebook: ttk.Notebook,
        title: str,
        columns: tuple[str, ...],
        widths: tuple[int, ...],
    ) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for column, width in zip(columns, widths):
            tree.heading(column, text=column.replace("_", " ").title())
            tree.column(column, width=width, stretch=True)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        scroll.grid(row=0, column=1, sticky=tk.NS)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def _browse_spec(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select declaration JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.spec_var.set(selected)

    def _browse_workspace(self) -> None:
        selected = filedialog.askdirectory(title="Select workspace directory")
        if selected:
            self.workspace_var.set(selected)
            self.ledger_var.set(str(Path(selected) / "qplan_ledger.sqlite"))
            self.db_var.set(str(Path(selected) / "sample.sqlite"))

    def _paths(self) -> tuple[Path, Path]:
        workspace = Path(self.workspace_var.get())
        workspace.mkdir(parents=True, exist_ok=True)
        return Path(self.ledger_var.get()), Path(self.db_var.get())

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def validate_spec(self) -> None:
        try:
            spec = load_spec(self.spec_var.get())
            message = (
                f"Spec OK. tables={len(spec.tables)} queries={len(spec.queries)} "
                f"indexes={len(spec.candidate_indexes)}\ndigest={spec.digest}\n"
                f"stats={spec.stats.caliber_version}"
            )
            self._set_status("Spec validation passed.")
            messagebox.showinfo("Spec validation", message)
        except Exception as exc:
            self._set_status("Spec validation failed.")
            messagebox.showerror("Spec validation", str(exc))

    def run_diagnostics(self) -> None:
        if not self.spec_var.get():
            messagebox.showwarning("Missing spec", "Select a declaration JSON first.")
            return
        self._set_status("Running deterministic data, estimates, index search and SQLite execution...")
        ledger, database = self._paths()

        def worker() -> None:
            try:
                spec = load_spec(self.spec_var.get())
                report = run_pipeline(
                    spec,
                    spec_path=self.spec_var.get(),
                    ledger_path=ledger,
                    db_path=database,
                    batch_size=2,
                )
                self.result_queue.put(("run", report))
            except Exception as exc:  # displayed on the Tk thread
                self.result_queue.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()

    def refresh_ledger(self) -> None:
        ledger_path = Path(self.ledger_var.get())
        try:
            ledger = Ledger.read_only(ledger_path)
            try:
                view = ledger.load_current_view()
                history = ledger.history()
            finally:
                ledger.close()
            self.ledger_text.delete("1.0", tk.END)
            self.ledger_text.insert(
                tk.END,
                json.dumps(normalize({"view": view, "history": history}), ensure_ascii=False, indent=2),
            )
            self._set_status(f"Loaded ledger snapshot view from {ledger_path}.")
        except Exception as exc:
            self._set_status("Ledger refresh failed.")
            messagebox.showerror("Ledger", str(exc))

    def recover(self) -> None:
        try:
            ledger = Ledger(self.ledger_var.get())
            try:
                result = ledger.recover()
            finally:
                ledger.close()
            self._set_status(f"Recovery status: {result.get('status')}")
            messagebox.showinfo("Recovery", json.dumps(normalize(result), ensure_ascii=False, indent=2))
            self.refresh_ledger()
        except Exception as exc:
            messagebox.showerror("Recovery", str(exc))

    def verify(self) -> None:
        try:
            result = verify_ledger(spec_path=self.spec_var.get(), ledger_path=self.ledger_var.get())
            self._set_status(f"Verification: {result.get('status')}")
            messagebox.showinfo("Independent verify", json.dumps(normalize(result), ensure_ascii=False, indent=2))
        except Exception as exc:
            messagebox.showerror("Independent verify", str(exc))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, value = self.result_queue.get_nowait()
                if kind == "run":
                    self.report = value
                    self._render_report(value)
                elif kind == "error":
                    self._set_status("Run failed.")
                    messagebox.showerror("Diagnostics", str(value))
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for item in tree.get_children():
            tree.delete(item)

    def _render_report(self, report: dict[str, Any]) -> None:
        self._set_status(
            f"Snapshot {report['snapshot']['snapshot_version']} committed; "
            f"diagnostics={','.join(report.get('diagnostic_codes', [])) or 'none'}"
        )
        self.summary_text.delete("1.0", tk.END)
        summary = {
            "snapshot": report.get("snapshot"),
            "invalidation": report.get("invalidation"),
            "statistics_summary": report.get("statistics_summary"),
            "index_search": {
                key: report.get("index_search", {}).get(key)
                for key in (
                    "primary_mode",
                    "truth_source",
                    "selected_indexes",
                    "candidate_count",
                    "total_estimated_cost",
                    "diagnostics",
                )
            },
            "diagnostic_codes": report.get("diagnostic_codes"),
        }
        self.summary_text.insert(tk.END, json.dumps(normalize(summary), ensure_ascii=False, indent=2))
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(tk.END, json.dumps(normalize(report), ensure_ascii=False, indent=2))

        self._clear_tree(self.query_tree)
        for query in report.get("queries", []):
            selected = query["selected_path"]
            estimate = query["selected_estimate"]
            execution = next(
                item["execution"]
                for item in query["paths"]
                if item["path"]["id"] == query["selected_path_id"]
            )
            attribution = (query.get("bias") or {}).get("attribution") or {}
            self.query_tree.insert(
                "",
                tk.END,
                values=(
                    query["query_id"],
                    query["selected_path_id"],
                    selected["kind"],
                    estimate["cost"],
                    execution["real_elapsed_ms"],
                    estimate["estimated_output_rows"],
                    "match" if query["result_match"] else "MISMATCH",
                    attribution.get("category", "OK"),
                ),
            )

        self._clear_tree(self.index_tree)
        for item in report.get("index_search", {}).get("selected_candidates", []):
            self.index_tree.insert(
                "",
                tk.END,
                values=(item["index"], item["table"], ", ".join(item["columns"]), "selected", item["reason"]),
            )
        for item in report.get("index_search", {}).get("unselected_candidates", []):
            self.index_tree.insert(
                "",
                tk.END,
                values=(item["index"], item["table"], ", ".join(item["columns"]), "not selected", item["reason"]),
            )

        self._clear_tree(self.diag_tree)
        for diagnostic in report.get("diagnostics", []):
            self.diag_tree.insert(
                "",
                tk.END,
                values=(
                    diagnostic.get("code", ""),
                    diagnostic.get("query_id", "workload"),
                    json.dumps(normalize(diagnostic), ensure_ascii=False, sort_keys=True),
                ),
            )
        self.refresh_ledger()


def run_gui(
    *,
    default_spec: str | None = None,
    workspace: str = "workspace",
    ledger: str | None = None,
    database: str | None = None,
) -> None:
    app = QPlanWindow(
        default_spec=default_spec,
        workspace=workspace,
        ledger=ledger,
        database=database,
    )
    app.mainloop()
