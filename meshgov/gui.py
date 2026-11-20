from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .snapshot import load_snapshot
from .verify import verify_snapshot
from .viewmodel import build_view_model


class WorkbenchApp:
    def __init__(self, root: Any, snapshot_path: Path) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.snapshot_path = snapshot_path
        self.root.title("Service Mesh Traffic Workbench")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Treeview", rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        self.header = ttk.Label(root, text="", style="Title.TLabel")
        self.header.pack(fill="x", padx=16, pady=(12, 2))
        self.meta = ttk.Label(root, text="")
        self.meta.pack(fill="x", padx=16, pady=(0, 8))
        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=16, pady=(0, 10))
        self.progress_text = ttk.Label(root, text="")
        self.progress_text.pack(fill="x", padx=16, pady=(0, 8))

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.overview_tab = ttk.Frame(notebook)
        self.instances_tab = ttk.Frame(notebook)
        self.budget_tab = ttk.Frame(notebook)
        self.circuit_tab = ttk.Frame(notebook)
        self.routes_tab = ttk.Frame(notebook)
        notebook.add(self.overview_tab, text="Summary")
        notebook.add(self.instances_tab, text="Instances")
        notebook.add(self.budget_tab, text="Concurrency Budget")
        notebook.add(self.circuit_tab, text="Circuits")
        notebook.add(self.routes_tab, text="Routes")

        self.conclusion = ttk.Label(
            self.overview_tab,
            text="",
            wraplength=1180,
            justify="left",
            padding=12,
        )
        self.conclusion.pack(fill="x")
        self.summary_tree = self._make_tree(
            self.overview_tab,
            ("metric", "value"),
            {"metric": ("Metric", 320), "value": ("Value", 760)},
        )

        self.instances_tree = self._make_tree(
            self.instances_tab,
            (
                "service",
                "instance",
                "port",
                "weight",
                "targeted",
                "selected",
                "success",
                "failed",
                "fastfail",
                "circuit",
                "threshold",
                "cooldown",
            ),
            {
                "service": ("Service", 120),
                "instance": ("Instance", 150),
                "port": ("Port", 70),
                "weight": ("Weight", 70),
                "targeted": ("Targeted", 80),
                "selected": ("Selected", 80),
                "success": ("Success", 80),
                "failed": ("Failed", 75),
                "fastfail": ("E_CIRCUIT", 90),
                "circuit": ("Circuit", 100),
                "threshold": ("Threshold", 90),
                "cooldown": ("Cooldown ms", 100),
            },
        )

        self.budget_tree = self._make_tree(
            self.budget_tab,
            ("scope", "limit", "peak", "active", "utilization"),
            {
                "scope": ("Scope", 260),
                "limit": ("Limit", 160),
                "peak": ("Peak", 160),
                "active": ("Active at snapshot", 180),
                "utilization": ("Peak utilization", 180),
            },
        )

        self.circuit_tree = self._make_tree(
            self.circuit_tab,
            ("service", "port", "event", "state", "reason", "opened", "until"),
            {
                "service": ("Service", 130),
                "port": ("Port", 70),
                "event": ("Event", 120),
                "state": ("State", 100),
                "reason": ("Reason", 480),
                "opened": ("Opened mono ms", 150),
                "until": ("Cooldown until mono ms", 170),
            },
        )

        self.routes_tree = self._make_tree(
            self.routes_tab,
            ("prefix", "service", "timeout", "retries", "version"),
            {
                "prefix": ("Prefix", 300),
                "service": ("Target service", 180),
                "timeout": ("Timeout ms", 120),
                "retries": ("Retry budget", 120),
                "version": ("Rule version", 480),
            },
        )

        actions = ttk.Frame(root)
        actions.pack(fill="x", padx=16, pady=(0, 12))
        ttk.Button(actions, text="Reload latest", command=self.reload).pack(
            side="left"
        )
        ttk.Button(actions, text="Verify snapshot", command=self.verify).pack(
            side="left", padx=8
        )
        ttk.Button(actions, text="Exit", command=root.destroy).pack(side="right")
        self.status = ttk.Label(actions, text="")
        self.status.pack(side="left", padx=16)
        self.reload()

    def _make_tree(
        self,
        parent: Any,
        columns: tuple[str, ...],
        settings: dict[str, tuple[str, int]],
    ) -> Any:
        frame = self.ttk.Frame(parent)
        frame.pack(fill="both", expand=True, padx=8, pady=8)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings")
        for column in columns:
            heading, width = settings[column]
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="w", stretch=True)
        scroll_y = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scroll_x = self.ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        return tree

    def reload(self) -> None:
        try:
            snapshot = load_snapshot(self.snapshot_path)
        except Exception as exc:
            self.status.configure(text=f"Load failed: {exc}")
            return
        view = build_view_model(snapshot)
        meta = view["meta"]
        progress = view["progress"]
        self.header.configure(text=f"Round {meta['round_id']}")
        self.meta.configure(
            text=(
                f"Rule version {meta['rule_version']}  |  Snapshot {meta['snapshot_id']}  |  "
                f"Created {meta['created_at_utc']}"
            )
        )
        self.progress["value"] = progress["percent"]
        self.progress_text.configure(
            text=(
                f"Progress {progress['completed']}/{progress['total']} "
                f"({progress['percent']:.1f}%)  |  Status {meta['status']}"
            )
        )
        self._clear(self.summary_tree)
        summary = view["summary"]
        self.conclusion.configure(text=str(summary.get("conclusion", "")))
        summary_rows = [
            ("Requests total", summary.get("requests_total")),
            ("Requests succeeded", summary.get("requests_succeeded")),
            ("Requests failed", summary.get("requests_failed")),
            ("Attempts", summary.get("attempts_total")),
            ("Selected attempts", summary.get("selected_attempts")),
            ("Retries used", summary.get("retries_used")),
            ("Circuit fast-failures", summary.get("circuit_fast_failures")),
            ("Open / half-open circuits", summary.get("circuits_open")),
            ("Final codes", summary.get("final_code_counts")),
            ("Rules changed during round", summary.get("rules_changed_during_round")),
            ("Observed rule version", summary.get("observed_rule_version")),
        ]
        for metric, value in summary_rows:
            self.summary_tree.insert("", "end", values=(metric, value))

        self._clear(self.instances_tree)
        for row in view["instances"]:
            self.instances_tree.insert(
                "",
                "end",
                values=(
                    row["service"],
                    row["instance"],
                    row["port"],
                    row["weight"],
                    row["targeted_count"],
                    row["selected_count"],
                    row["success_count"],
                    row["failure_count"],
                    row["fast_fail_count"],
                    row["circuit_state"],
                    row["failure_threshold"],
                    row["cooldown_ms"],
                ),
            )

        self._clear(self.budget_tree)
        global_budget = view["budget"]["global"]
        self.budget_tree.insert(
            "",
            "end",
            values=(
                "GLOBAL",
                global_budget.get("limit"),
                global_budget.get("peak"),
                global_budget.get("active"),
                f"{global_budget.get('utilization_percent', 0):.1f}%",
            ),
        )
        for row in view["budget"]["services"]:
            self.budget_tree.insert(
                "",
                "end",
                values=(
                    row.get("service"),
                    row.get("limit"),
                    row.get("peak"),
                    row.get("active"),
                    f"{row.get('utilization_percent', 0):.1f}%",
                ),
            )

        self._clear(self.circuit_tree)
        for row in view["circuit_events"]:
            self.circuit_tree.insert(
                "",
                "end",
                values=(
                    row.get("service"),
                    row.get("port"),
                    row.get("event"),
                    row.get("state"),
                    row.get("reason"),
                    _format_ms(row.get("opened_at_mono_ms")),
                    _format_ms(row.get("cooldown_until_mono_ms")),
                ),
            )

        self._clear(self.routes_tree)
        for row in view["routes"]:
            self.routes_tree.insert(
                "",
                "end",
                values=(
                    row.get("prefix"),
                    row.get("service"),
                    row.get("timeout_ms"),
                    row.get("max_retries"),
                    row.get("rule_version"),
                ),
            )
        self.status.configure(text=f"Loaded {self.snapshot_path}")

    def verify(self) -> None:
        from tkinter import messagebox

        try:
            snapshot = load_snapshot(self.snapshot_path)
            result = verify_snapshot(snapshot)
        except Exception as exc:
            messagebox.showerror("Verification error", str(exc))
            return
        if result["status"] == "pass":
            messagebox.showinfo("Verification", "PASS: all budget and trace checks passed.")
            self.status.configure(text="Verification PASS")
        else:
            first = result["first_deviation"]
            messagebox.showerror(
                "E_VERIFY",
                f"First deviation: {first['request_id']} / {first['check']}\n"
                f"{first['detail']}",
            )
            self.status.configure(text="Verification E_VERIFY")

    @staticmethod
    def _clear(tree: Any) -> None:
        for item in tree.get_children():
            tree.delete(item)


def _format_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def launch(snapshot_path: str | Path) -> int:
    try:
        import tkinter as tk
    except ImportError:
        raise RuntimeError(
            "Tkinter is unavailable. Install the official Windows Python package "
            "with Tcl/Tk support."
        )
    root = tk.Tk()
    WorkbenchApp(root, Path(snapshot_path))
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tkinter service-mesh workbench")
    parser.add_argument("--snapshot", default="artifacts/snapshots/latest.json")
    args = parser.parse_args(argv)
    return launch(args.snapshot)


if __name__ == "__main__":
    raise SystemExit(main())

