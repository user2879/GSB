"""Tkinter desktop workbench for the graph traversal diagnostics engine.

Windows official Python ships Tkinter.  On minimal Linux interpreters without
the ``tkinter`` extension, importing the module raises a readable message
instead of a raw traceback; the CLI remains fully functional in that case.
"""

from __future__ import annotations

import os
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TK_AVAILABLE = True
    TK_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # minimal interpreters without tk
    TK_AVAILABLE = False
    TK_IMPORT_ERROR = exc

    class _WidgetStub:  # pragma: no cover - only present when tk is missing
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "tkinter is not available in this interpreter"
            ) from TK_IMPORT_ERROR

    tk = type("tk", (), {"Tk": _WidgetStub, "TclError": Exception})()
    filedialog = messagebox = ttk = None

from ..core.engine import Engine
from ..core.errors import GraphDiagError
from ..core.fixedpoint import format_micro
from ..core.traversal import (
    CONFIRMED_OVER,
    ERROR_CROSS,
    UNREACHABLE,
    WITHIN,
)
from ..sample_data import generate

VERDICT_TEXT = {
    WITHIN: "在限内",
    CONFIRMED_OVER: "确实超界",
    ERROR_CROSS: "误差跨越(不可区分)",
    UNREACHABLE: "限内不可达",
}


class GraphDiagApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("图遍历可达距离定点估算与误差传播诊断台")
        self.geometry("1180x760")
        self.engine: Engine | None = None
        self.db_path = os.path.abspath("graphdiag_gui.db")
        self._build_widgets()

    # -- layout -------------------------------------------------------------
    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Button(top, text="生成样本", command=self.on_generate).pack(side=tk.LEFT)
        ttk.Button(top, text="加载声明 JSON", command=self.on_load).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top, text="执行遍历", command=self.on_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="保存快照", command=self.on_snapshot).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(top, text="恢复并重放校验", command=self.on_restore).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Label(top, text="预算(访问节点)").pack(side=tk.LEFT, padx=(16, 4))
        self.budget_var = tk.StringVar(value="1000")
        ttk.Entry(top, textvariable=self.budget_var, width=10).pack(side=tk.LEFT)
        ttk.Label(top, text="并发").pack(side=tk.LEFT, padx=(12, 4))
        self.workers_var = tk.StringVar(value="2")
        ttk.Entry(top, textvariable=self.workers_var, width=6).pack(side=tk.LEFT)
        ttk.Label(top, text="误差口径").pack(side=tk.LEFT, padx=(12, 4))
        self.mode_var = tk.StringVar(value="exact")
        ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["exact", "conservative"],
            width=12,
            state="readonly",
        ).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="就绪。数据库: " + self.db_path)
        ttk.Label(self, textvariable=self.status_var, padding=(8, 2)).pack(fill=tk.X)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="查询结论")
        columns = (
            "query", "target", "verdict", "dist_fp", "dist", "err_exact",
            "err_cons", "cap", "affected", "dominant", "path",
        )
        self.tree = ttk.Treeview(summary_frame, columns=columns, show="headings")
        headings = {
            "query": ("查询", 110),
            "target": ("目标", 60),
            "verdict": ("判定", 150),
            "dist_fp": ("距离fp", 80),
            "dist": ("距离", 110),
            "err_exact": ("精确误差界(μ)", 120),
            "err_cons": ("保守界(ulp)", 100),
            "cap": ("上限", 110),
            "affected": ("受误差影响", 90),
            "dominant": ("主导误差段", 120),
            "path": ("最优路径", 220),
        }
        for key, (text, width) in headings.items():
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor=tk.W)
        tree_scroll = ttk.Scrollbar(
            summary_frame, orient=tk.VERTICAL, command=self.tree.yview
        )
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        detail_frame = ttk.Frame(notebook)
        notebook.add(detail_frame, text="误差中间项 / 剪枝 / 调度")
        self.detail = tk.Text(detail_frame, wrap=tk.NONE)
        detail_scroll = ttk.Scrollbar(
            detail_frame, orient=tk.VERTICAL, command=self.detail.yview
        )
        self.detail.configure(yscrollcommand=detail_scroll.set)
        self.detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ledger_frame = ttk.Frame(notebook)
        notebook.add(ledger_frame, text="遍历账本 / 版本")
        self.ledger = tk.Text(ledger_frame, wrap=tk.NONE)
        ledger_scroll = ttk.Scrollbar(
            ledger_frame, orient=tk.VERTICAL, command=self.ledger.yview
        )
        self.ledger.configure(yscrollcommand=ledger_scroll.set)
        self.ledger.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ledger_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # -- helpers ------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _engine(self) -> Engine:
        if self.engine is None:
            self.engine = Engine(self.db_path, error_mode=self.mode_var.get())
        else:
            self.engine.error_mode = self.mode_var.get()
        return self.engine

    def _run_async(self, action) -> None:
        thread = threading.Thread(target=action, daemon=True)
        thread.start()

    # -- actions ------------------------------------------------------------
    def on_generate(self) -> None:
        def task() -> None:
            try:
                paths = generate()
                self.after(0, lambda: self._set_status("已生成: " + ", ".join(paths.values())))
            except Exception as exc:  # pragma: no cover - GUI guard
                self.after(0, lambda: messagebox.showerror("生成失败", str(exc)))
        self._run_async(task)

    def on_load(self) -> None:
        path = filedialog.askopenfilename(
            title="选择图声明 JSON", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return

        def task() -> None:
            try:
                info = self._engine().load(path)
                self.after(0, lambda: self._after_load(info))
            except GraphDiagError as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        exc.code, f"{exc.message}\n{exc.detail}"
                    ),
                )
        self._run_async(task)

    def _after_load(self, info: dict) -> None:
        self._set_status(
            f"已加载 graph=v{info['graph_version']} query=v{info['query_version']} "
            f"scale=1e{info['scale_exponent']}；旧结论已整体标记为陈旧"
        )
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.ledger.delete("1.0", tk.END)
        self.ledger.insert(
            tk.END,
            f"graph_digest={info['graph_digest']}\nqueries_digest={info['queries_digest']}\n",
        )

    def on_run(self) -> None:
        if self.engine is None or self.engine.declaration is None:
            messagebox.showwarning("提示", "请先加载声明 JSON")
            return
        try:
            budget = int(self.budget_var.get())
            workers = int(self.workers_var.get())
        except ValueError:
            messagebox.showerror("输入错误", "预算与并发必须是整数")
            return

        def task() -> None:
            try:
                self._set_status("遍历执行中……")
                view = self.engine.run(budget_limit=budget, max_workers=workers)
                self.after(0, lambda: self._after_run(view))
            except GraphDiagError as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        exc.code, f"{exc.message}\n{exc.detail}"
                    ),
                )
        self._run_async(task)

    def _after_run(self, view) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        detail_lines: list[str] = []
        for schedule in view.report.ordered():
            result = schedule.result
            detail_lines.append(
                f"[{result.query_id}] start_ms={schedule.start_monotonic_ms} "
                f"end_ms={schedule.end_monotonic_ms} visited={schedule.visited_nodes} "
                f"pruned={schedule.pruned_branches} budget={schedule.budget_consumed} "
                f"exhausted={schedule.budget_exhausted} wait_ms={schedule.wait_duration_ms}\n"
                f"  visit sequence: {result.visited_sequence}\n"
            )
            for target, tr in sorted(result.targets.items()):
                if not tr.reached:
                    self.tree.insert(
                        "", tk.END,
                        values=(result.query_id, target, VERDICT_TEXT[tr.classification],
                                "-", "-", "-", "-", format_micro(tr.cap_micro),
                                "-", "-", "-"),
                    )
                    continue
                self.tree.insert(
                    "", tk.END,
                    values=(
                        result.query_id, target, VERDICT_TEXT[tr.classification],
                        tr.dist_fp, format_micro(tr.dist_micro),
                        tr.exact_bound_micro, tr.conservative_bound_ulp,
                        format_micro(tr.cap_micro), tr.error_affected,
                        f"{tr.dominant_edge[0]}->{tr.dominant_edge[1]}",
                        "->".join(str(n) for n in tr.path_nodes),
                    ),
                )
                for term in tr.terms:
                    detail_lines.append(
                        f"  target {target} edge#{term.edge_index} "
                        f"{term.source}->{term.target} q_fp={term.quantized_fp} "
                        f"remainder={term.edge_remainder_micro} "
                        f"running_cons_ulp={term.conservative_bound_ulp} "
                        f"running_exact={term.exact_bound_micro}\n"
                    )
            for prune in result.pruned:
                detail_lines.append(
                    f"  prune {prune.source}->{prune.target} reason={prune.reason} "
                    f"candidate_fp={prune.candidate_fp} cap_fp={prune.cap_fp}\n"
                )
            detail_lines.append("\n")
        self.detail.delete("1.0", tk.END)
        self.detail.insert(tk.END, "".join(detail_lines))

        changes = self.engine.diff_views()
        change_text = (
            "整体替换，无变化"
            if not changes
            else f"整体替换，变化目标 {len(changes)} 个"
        )
        self._set_status(
            f"graph=v{view.graph_version} query=v{view.query_version} "
            f"scale=1e{view.scale_exponent}；{change_text}"
        )
        self.ledger.delete("1.0", tk.END)
        for row in self.engine.store.read_ledger():
            self.ledger.insert(tk.END, f"{row}\n")

    def on_snapshot(self) -> None:
        if self.engine is None or self.engine.current_view is None:
            messagebox.showwarning("提示", "请先加载并执行遍历")
            return

        def task() -> None:
            try:
                digest = self.engine.save_snapshot("snapshot-gui")
                self.after(
                    0,
                    lambda: self._set_status(f"快照已保存 digest={digest[:16]}…"),
                )
            except GraphDiagError as exc:
                self.after(0, lambda: messagebox.showerror(exc.code, exc.message))
        self._run_async(task)

    def on_restore(self) -> None:
        def task() -> None:
            try:
                outcome = self._engine().restore_snapshot("snapshot-gui")
                self.after(
                    0,
                    lambda: (
                        self._set_status("快照恢复并重放一致: " + outcome["digest"][:16]),
                        messagebox.showinfo("重放校验", "逐项比较通过，E_REPLAY 未触发"),
                    ),
                )
            except GraphDiagError as exc:
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        exc.code, f"{exc.message}\n{exc.detail}"
                    ),
                )
        self._run_async(task)


def launch() -> int:
    if not TK_AVAILABLE:
        raise SystemExit(
            "Tkinter 不可用（本机 Python 缺少 tk 扩展）。Windows 官方安装包自带；"
            "Linux 可安装 python3-tk。CLI 仍可使用: python -m graphdiag.cli --help"
        )
    try:
        app = GraphDiagApp()
    except tk.TclError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Tkinter 不可用（本机 Python 缺少 tk 扩展）。Windows 官方安装包自带；"
            "Linux 可安装 python3-tk。CLI 仍可使用: python -m graphdiag.cli --help"
        ) from exc
    app.mainloop()
    return 0
