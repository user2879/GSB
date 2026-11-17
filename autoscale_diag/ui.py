"""Tkinter diagnostic workbench.  Tk is imported lazily for headless use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_USAGE
from .viewmodel import TABLE_COLUMNS, DashboardModel, load_dashboard_model, self_check


def _tk_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
    except ImportError as exc:
        raise BusinessError(
            E_USAGE,
            "tkinter is unavailable; install the Python Tk component for this interpreter",
            "tkinter",
            {"error": str(exc)},
        )
    return tk, ttk, messagebox, ScrolledText


class WorkbenchApp:
    def __init__(self, root: Any, model: DashboardModel, db_path: str, run_id: str):
        self.tk, self.ttk, self.messagebox, self.ScrolledText = _tk_modules()
        self.root = root
        self.model = model
        self.db_path = db_path
        self.run_id = run_id
        self.row_lookup: dict[str, dict[str, Any]] = {}
        self.root.title("弹性伸缩指标诊断工作台")
        self.root.geometry("1320x840")
        self.root.minsize(980, 660)
        self._build()
        self._populate()

    def _build(self) -> None:
        ttk = self.ttk
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="弹性伸缩指标驱动诊断工作台", font=("Segoe UI", 16, "bold")).pack(
            side="left"
        )
        self.status_var = self.tk.StringVar(value=self.model.status_line)
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(8, 8))
        ttk.Label(controls, text="SQLite").pack(side="left")
        self.db_var = self.tk.StringVar(value=self.db_path)
        ttk.Entry(controls, textvariable=self.db_var, width=48).pack(side="left", padx=6)
        ttk.Label(controls, text="Run").pack(side="left", padx=(8, 0))
        self.run_var = self.tk.StringVar(value=self.run_id)
        ttk.Entry(controls, textvariable=self.run_var, width=16).pack(side="left", padx=6)
        ttk.Button(controls, text="重新加载账本", command=self.reload).pack(side="left", padx=4)
        ttk.Button(controls, text="展开所选依据链", command=self.show_selected_evidence).pack(
            side="left", padx=4
        )
        self.path_var = self.tk.StringVar()
        ttk.Label(controls, textvariable=self.path_var).pack(side="right")

        summary = ttk.Frame(outer)
        summary.pack(fill="x", pady=(0, 8))
        self.summary_var = self.tk.StringVar()
        ttk.Label(summary, textvariable=self.summary_var, font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )

        paned = ttk.Panedwindow(outer, orient="vertical")
        paned.pack(fill="both", expand=True)
        upper = ttk.Panedwindow(paned, orient="horizontal")
        paned.add(upper, weight=4)

        table_frame = ttk.Frame(upper)
        upper.add(table_frame, weight=3)
        columns = tuple(item[0] for item in TABLE_COLUMNS)
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        for key, label, width in TABLE_COLUMNS:
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, minwidth=45, anchor="w", stretch=False)
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        self.tree.tag_configure("cooling", foreground="#9a5f00")
        self.tree.tag_configure("clipped", foreground="#8a2230")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda _event: self.show_selected_evidence())

        side = ttk.Frame(upper, padding=(8, 0, 0, 0))
        upper.add(side, weight=2)
        self.chart_title = ttk.Label(side, text="预算占用曲线", font=("Segoe UI", 11, "bold"))
        self.chart_title.pack(anchor="w")
        self.canvas = self.tk.Canvas(side, background="#111820", height=230, highlightthickness=0)
        self.canvas.pack(fill="x", pady=(4, 8))
        self.canvas.bind("<Configure>", lambda _event: self._draw_chart())
        ttk.Label(side, text="冷却抑制记录", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.cooling = ttk.Treeview(
            side, columns=("minute", "workload", "trigger", "attempted", "final"), show="headings"
        )
        for key, label, width in (
            ("minute", "分钟", 55),
            ("workload", "工作负载", 100),
            ("trigger", "触发分钟", 75),
            ("attempted", "尝试", 55),
            ("final", "最终", 55),
        ):
            self.cooling.heading(key, text=label)
            self.cooling.column(key, width=width, anchor="w")
        self.cooling.pack(fill="both", expand=True)

        lower = ttk.Frame(paned)
        paned.add(lower, weight=1)
        self.evidence = self.ScrolledText(
            lower, height=8, wrap="word", font=("Consolas", 10)
        )
        self.evidence.pack(fill="both", expand=True)
        self.evidence.configure(state="disabled")

    def _populate(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.row_lookup.clear()
        first_id = None
        for row in self.model.rows:
            item_id = f"m{row['minute_index']:06d}:{row['workload_id']}"
            tags = []
            if row["cooldown_suppressed"]:
                tags.append("cooling")
            if row["budget_clipped"]:
                tags.append("clipped")
            self.tree.insert(
                "", "end", iid=item_id, values=self.model.table_row(row), tags=tuple(tags)
            )
            self.row_lookup[item_id] = row
            if first_id is None:
                first_id = item_id
        for item in self.cooling.get_children():
            self.cooling.delete(item)
        for item in self.model.cooling_suppressions:
            self.cooling.insert(
                "",
                "end",
                values=(
                    item["minute_index"],
                    item["workload_id"],
                    item["trigger_minute"],
                    item["attempted_replicas"],
                    item["final_replicas"],
                ),
            )
        run = self.model.report["run"]
        self.summary_var.set(
            f"决策 {self.model.report['summary']['decision_counts']} · "
            f"冷却抑制 {self.model.report['summary']['cooling_suppressions']} · "
            f"预算裁剪 {self.model.report['summary']['budget_clipping_events']} · "
            f"窗口容量 {run['window_capacity']} · 账本版本 {run['declaration_version']}"
        )
        self.path_var.set(f"路径 {run['path']} · SHA {run['spec_sha256'][:12]}")
        self.status_var.set(self.model.status_line)
        if first_id:
            self.tree.selection_set(first_id)
            self.tree.focus(first_id)
            self._on_select()
        self._draw_chart()

    def reload(self) -> None:
        try:
            model = load_dashboard_model(self.db_var.get().strip(), self.run_var.get().strip())
        except Exception as exc:
            self.messagebox.showerror("加载失败", str(exc))
            return
        self.model = model
        self.db_path = self.db_var.get().strip()
        self.run_id = self.run_var.get().strip()
        self._populate()

    def _selected_row(self) -> dict[str, Any] | None:
        selected = self.tree.selection()
        return self.row_lookup.get(selected[0]) if selected else None

    def _on_select(self, _event: Any = None) -> None:
        row = self._selected_row()
        if row is None:
            return
        self.evidence.configure(state="normal")
        self.evidence.delete("1.0", "end")
        self.evidence.insert("1.0", self.model.evidence_text(row))
        self.evidence.configure(state="disabled")

    def show_selected_evidence(self) -> None:
        row = self._selected_row()
        if row is None:
            self.messagebox.showinfo("依据链", "请先选择一分钟决策。")
            return
        self.messagebox.showinfo(
            f"分钟 {row['minute']} · {row['workload_id']}", self.model.evidence_text(row)
        )

    def _draw_chart(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        curve = self.model.budget_curve
        width = max(360, canvas.winfo_width())
        height = max(190, canvas.winfo_height())
        left, right, top, bottom = 42, 18, 18, 30
        plot_w = width - left - right
        plot_h = height - top - bottom
        budget = max(1, int(self.model.report["run"]["budget"]))
        canvas.create_line(left, top, left, top + plot_h, fill="#506077")
        canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="#506077")
        canvas.create_text(left + 4, top + 6, anchor="w", text=f"budget {budget}", fill="#f0b45c")
        if not curve:
            return
        count = len(curve)

        def x(index: int) -> float:
            return left + (plot_w * index / max(1, count - 1))

        def y(value: int) -> float:
            return top + plot_h - (plot_h * value / budget)

        budget_y = y(budget)
        canvas.create_line(left, budget_y, left + plot_w, budget_y, fill="#f0b45c", dash=(6, 4))
        points: list[float] = []
        for index, item in enumerate(curve):
            points.extend((x(index), y(item["budget_used"])))
        if len(points) >= 4:
            canvas.create_line(*points, fill="#57c2a6", width=3, smooth=False)
        for index, item in enumerate(curve):
            px, py = x(index), y(item["budget_used"])
            canvas.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#57c2a6", outline="")
            if index % max(1, count // 10) == 0 or index == count - 1:
                canvas.create_text(px, top + plot_h + 13, text=str(item["minute"]), fill="#9fb0c3")


def launch_ui(db_path: str, run_id: str) -> None:
    tk, _ttk, _messagebox, _ScrolledText = _tk_modules()
    model = load_dashboard_model(db_path, run_id)
    root = tk.Tk()
    WorkbenchApp(root, model, db_path, run_id)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tkinter scaling diagnostic workbench")
    parser.add_argument("--db", default=str(Path("artifacts/demo/workbench.db")))
    parser.add_argument("--run-id", default="demo")
    parser.add_argument("--check", action="store_true", help="run a headless view-model check")
    parser.add_argument(
        "--require-tk",
        action="store_true",
        help="with --check, also require a usable Tk display",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.check:
            result = self_check(args.db, args.run_id)
            root = None
            tk_imported = False
            try:
                tk, _ttk, _messagebox, _ScrolledText = _tk_modules()
                tk_imported = True
                model = load_dashboard_model(args.db, args.run_id)
                root = tk.Tk()
                callback_errors: list[str] = []
                root.report_callback_exception = (
                    lambda exc_type, exc_value, exc_traceback: callback_errors.append(
                        f"{exc_type.__name__}: {exc_value}"
                    )
                )
                app = WorkbenchApp(root, model, args.db, args.run_id)
                root.update_idletasks()
                root.update()
                if callback_errors:
                    raise RuntimeError("Tk callback failed: " + "; ".join(callback_errors))
                result["tkinter_available"] = True
                result["tk_version"] = str(tk.TkVersion)
                result["workbench_ready"] = True
                result["window_size"] = [root.winfo_width(), root.winfo_height()]
                result["tree_rows"] = len(app.tree.get_children())
                result["cooling_rows"] = len(app.cooling.get_children())
                result["canvas_items"] = len(app.canvas.find_all())
                result["evidence_chars"] = len(app.evidence.get("1.0", "end"))
            except Exception as exc:
                result["tkinter_available"] = False
                result["tkinter_error"] = str(exc)
                if tk_imported:
                    result["tkinter_available"] = True
                if args.require_tk or result.get("tkinter_available"):
                    result["ok"] = False
                    result["business_code"] = "E_USAGE"
            finally:
                if root is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0 if result["ok"] else 2
        launch_ui(args.db, args.run_id)
        return 0
    except BusinessError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "business_code": "E_USAGE",
                    "message": "cannot start the Tkinter workbench",
                    "details": {"error": f"{type(exc).__name__}: {exc}"},
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
