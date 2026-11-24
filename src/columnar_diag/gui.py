from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .errors import AppError
from .models import BudgetConfig
from .sample import build_sample_spec, sample_changes
from .snapshot import restore_snapshot, save_snapshot
from .verifier import authoritative_replay
from .workspace import Workspace


class DiagnosticApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("列式段压缩与扫描诊断")
        self.root.geometry("1240x820")
        self.root.minsize(1020, 680)
        self.workspace: Workspace | None = None
        self.workspace_path = tk.StringVar(value=str(Path.cwd() / "diagnostic_workspace"))
        self.parallelism = tk.IntVar(value=4)
        self.memory_budget = tk.IntVar(value=128)
        self.segment_quota = tk.IntVar(value=3)
        self.batch_size = tk.IntVar(value=5)
        self.delay_ms = tk.DoubleVar(value=2.0)
        self.snapshot_path = tk.StringVar(value=str(Path.cwd() / "session_snapshot.json"))
        self.encoding_column = tk.StringVar()
        self.encoding_name = tk.StringVar(value="dictionary")
        self.encoding_compressed = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="请初始化样本工作区或打开已有工作区")
        self._build_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Hint.TLabel", foreground="#4d5966")
        style.configure("Metric.TLabel", font=("Consolas", 10))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="列式段压缩与扫描诊断台", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="Windows 本地 · 确定性重放", style="Hint.TLabel").pack(
            side="left", padx=16, pady=(6, 0)
        )

        path_row = ttk.Frame(outer)
        path_row.pack(fill="x", pady=(0, 8))
        ttk.Label(path_row, text="工作区").pack(side="left")
        ttk.Entry(path_row, textvariable=self.workspace_path).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(path_row, text="初始化样本", command=self._init_sample).pack(side="left")
        ttk.Button(path_row, text="打开", command=self._open_workspace).pack(side="left", padx=6)
        ttk.Button(path_row, text="刷新", command=self._refresh_all).pack(side="left")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self._build_overview_tab()
        self._build_scan_tab()
        self._build_incremental_tab()
        self._build_ledger_tab()
        self._build_snapshot_tab()
        self._build_version_tab()

        status_bar = ttk.Frame(outer)
        status_bar.pack(fill="x", pady=(8, 0))
        ttk.Label(status_bar, textvariable=self.status, style="Hint.TLabel").pack(
            side="left"
        )

    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="概览与计划")
        toolbar = ttk.Frame(frame)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="刷新计划", command=self._refresh_overview).pack(side="left")
        ttk.Label(
            toolbar,
            text="展示裁列、候选块、推导依据与当前查询结论摘要",
            style="Hint.TLabel",
        ).pack(side="left", padx=12)
        self.overview_text = ScrolledText(frame, font=("Consolas", 10), wrap="none")
        self.overview_text.pack(fill="both", expand=True)

    def _build_scan_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="预算调度扫描")
        controls = ttk.LabelFrame(frame, text="三重约束", padding=8)
        controls.pack(fill="x", pady=(0, 8))
        fields = (
            ("全局并行度", self.parallelism, 1, 64),
            ("每列解码内存预算(B)", self.memory_budget, 8, 1_000_000),
            ("每段并发配额", self.segment_quota, 1, 64),
            ("批大小", self.batch_size, 1, 1000),
        )
        for index, (label, variable, minimum, maximum) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=index * 2, sticky="e", padx=(0, 4))
            ttk.Spinbox(
                controls,
                from_=minimum,
                to=maximum,
                textvariable=variable,
                width=10,
            ).grid(row=0, column=index * 2 + 1, sticky="w", padx=(0, 14))
        ttk.Label(controls, text="任务延迟(ms)").grid(row=1, column=0, sticky="e", pady=(8, 0))
        ttk.Spinbox(
            controls,
            from_=0,
            to=5000,
            increment=0.5,
            textvariable=self.delay_ms,
            width=10,
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))
        ttk.Button(controls, text="应用预算版本并重扫", command=self._change_budget).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=(0, 12), pady=(8, 0)
        )
        ttk.Button(controls, text="从游标继续扫描", command=self._run_scan).grid(
            row=1, column=4, columnspan=2, sticky="w", pady=(8, 0)
        )
        self.scan_text = ScrolledText(frame, font=("Consolas", 10), wrap="none")
        self.scan_text.pack(fill="both", expand=True)

    def _build_incremental_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="增量失效")
        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(
            top,
            text="提交块级变更集；仅重编码直接变更块，并只重算受影响查询的失效子集。",
            style="Hint.TLabel",
        ).pack(side="left")
        ttk.Button(top, text="填入样本变更", command=self._fill_sample_change).pack(side="right")
        ttk.Button(top, text="应用变更集", command=self._apply_change).pack(side="right", padx=6)
        self.change_text = ScrolledText(frame, font=("Consolas", 10), height=14)
        self.change_text.pack(fill="both", expand=False)
        ttk.Label(frame, text="传播与逐字节保持校验", style="Hint.TLabel").pack(
            anchor="w", pady=(8, 4)
        )
        self.incremental_text = ScrolledText(frame, font=("Consolas", 10), wrap="none")
        self.incremental_text.pack(fill="both", expand=True)
        self._fill_sample_change()

    def _build_ledger_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="账本与权威重放")
        bar = ttk.Frame(frame)
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="显示当前账本", command=self._refresh_ledger).pack(side="left")
        ttk.Button(bar, text="权威逐块重放校验", command=self._verify).pack(
            side="left", padx=6
        )
        ttk.Label(
            bar,
            text="重放只从 segment.seg 解码，不把账本摘要作为计算输入",
            style="Hint.TLabel",
        ).pack(side="left", padx=10)
        self.ledger_text = ScrolledText(frame, font=("Consolas", 10), wrap="none")
        self.ledger_text.pack(fill="both", expand=True)

    def _build_snapshot_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="快照恢复")
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(0, 8))
        ttk.Label(row, text="快照文件").pack(side="left")
        ttk.Entry(row, textvariable=self.snapshot_path).pack(
            side="left", fill="x", expand=True, padx=8
        )
        ttk.Button(row, text="选择", command=self._choose_snapshot).pack(side="left")
        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(0, 8))
        ttk.Button(actions, text="保存快照", command=self._save_snapshot).pack(side="left")
        ttk.Button(actions, text="整体替换恢复", command=self._restore_snapshot).pack(
            side="left", padx=6
        )
        ttk.Button(actions, text="从记录游标继续", command=self._resume_scan).pack(side="left")
        ttk.Button(
            actions, text="恢复后重放校验", command=self._verify_snapshot_replay
        ).pack(
            side="left", padx=6
        )
        self.snapshot_text = ScrolledText(frame, font=("Consolas", 10), wrap="none")
        self.snapshot_text.pack(fill="both", expand=True)

    def _build_version_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="版本演进")
        policy = ttk.LabelFrame(frame, text="编码选择策略", padding=8)
        policy.pack(fill="x", pady=(0, 8))
        ttk.Label(policy, text="列").grid(row=0, column=0, sticky="e")
        self.column_combo = ttk.Combobox(
            policy, textvariable=self.encoding_column, state="readonly", width=18
        )
        self.column_combo.grid(row=0, column=1, padx=6)
        ttk.Label(policy, text="编码").grid(row=0, column=2, sticky="e")
        ttk.Combobox(
            policy,
            textvariable=self.encoding_name,
            values=("raw", "dictionary", "rle"),
            state="readonly",
            width=14,
        ).grid(row=0, column=3, padx=6)
        ttk.Checkbutton(
            policy, text="启用压缩", variable=self.encoding_compressed
        ).grid(row=0, column=4, padx=6)
        ttk.Button(policy, text="提升编码版本并更新", command=self._change_encoding).grid(
            row=0, column=5, padx=6
        )
        self.version_text = ScrolledText(frame, font=("Consolas", 10), wrap="none")
        self.version_text.pack(fill="both", expand=True)

    def _set_output(self, widget: ScrolledText, value: Any) -> None:
        widget.delete("1.0", "end")
        if isinstance(value, str):
            widget.insert("end", value)
        else:
            widget.insert("end", json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        widget.see("1.0")

    def _require_workspace(self) -> Workspace | None:
        if self.workspace is None:
            messagebox.showwarning("未打开工作区", "请先初始化样本或打开已有工作区。")
            return None
        return self.workspace

    def _budget(self) -> BudgetConfig:
        return BudgetConfig(
            global_parallelism=int(self.parallelism.get()),
            column_memory_budget_bytes=int(self.memory_budget.get()),
            segment_concurrency=int(self.segment_quota.get()),
        )

    def _run_async(
        self,
        label: str,
        work: Callable[[], Any],
        on_success: Callable[[Any], None],
    ) -> None:
        self.status.set(f"{label}…")

        def runner() -> None:
            try:
                result = work()
            except AppError as error:
                payload = error.to_dict()
                self.root.after(0, lambda payload=payload: self._show_error(label, payload))
                return
            except Exception as error:  # pragma: no cover - UI safety net
                payload = {
                    "code": "E_UI",
                    "message": str(error),
                    "trace": traceback.format_exc(),
                }
                self.root.after(
                    0,
                    lambda payload=payload: self._show_error(label, payload),
                )
                return
            self.root.after(0, lambda: self._finish_async(label, result, on_success))

        threading.Thread(target=runner, daemon=True).start()

    def _finish_async(
        self, label: str, result: Any, on_success: Callable[[Any], None]
    ) -> None:
        on_success(result)
        self.status.set(f"{label}完成")

    def _show_error(self, label: str, error: dict[str, Any]) -> None:
        self.status.set(f"{label}失败: {error.get('code')}")
        messagebox.showerror(error.get("code", "ERROR"), error.get("message", "未知错误"))

    def _init_sample(self) -> None:
        path = Path(self.workspace_path.get())
        budget = self._budget()
        if self.workspace is not None:
            self.workspace.close()
            self.workspace = None

        def work() -> dict[str, Any]:
            workspace = Workspace.create(
                path,
                build_sample_spec(),
                budget=budget,
                overwrite=True,
            )
            try:
                scan = workspace.run_full_scan(batch_size=5, delay_ms=1.0)
                return {
                    "workspace": str(path),
                    "task_count": len(workspace.bundle.tasks),
                    "scan": scan,
                    "statistics": workspace.ledger_statistics(),
                }
            finally:
                workspace.close()

        def done(result: dict[str, Any]) -> None:
            self._open_workspace(show_error=False)
            self._set_output(self.scan_text, result)

        self._run_async("初始化样本并扫描", work, done)

    def _open_workspace(self, show_error: bool = True) -> None:
        path = Path(self.workspace_path.get())

        def work() -> Workspace:
            return Workspace(path)

        def done(workspace: Workspace) -> None:
            if self.workspace is not None:
                self.workspace.close()
            self.workspace = workspace
            self.column_combo.configure(values=list(workspace.spec.column_names))
            if workspace.spec.columns and not self.encoding_column.get():
                self.encoding_column.set(workspace.spec.columns[0].name)
            self.parallelism.set(workspace.budget.global_parallelism)
            self.memory_budget.set(workspace.budget.column_memory_budget_bytes)
            self.segment_quota.set(workspace.budget.segment_concurrency)
            self.status.set(f"已打开 {workspace.root}")
            self._refresh_all()

        self._run_async("打开工作区", work, done)

    def _refresh_all(self) -> None:
        if self.workspace is None:
            return
        self._refresh_overview()
        self._refresh_ledger()

    def _refresh_overview(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        self._set_output(
            self.overview_text,
            {
                "segment": workspace.segment_set_summary(),
                "versions": workspace.versions,
                "budget": workspace.budget.to_dict(),
                "plans": [plan.to_dict() for plan in workspace.bundle.plans],
                "query_results": workspace.query_view(),
            },
        )

    def _run_scan(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        batch_size = int(self.batch_size.get())
        delay = float(self.delay_ms.get())
        self._run_async(
            "扫描",
            lambda: workspace.scan_pending(batch_size=batch_size, delay_ms=delay),
            lambda result: (
                self._set_output(self.scan_text, result),
                self._refresh_ledger(),
            ),
        )

    def _change_budget(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        budget = self._budget()
        batch_size = int(self.batch_size.get())
        delay = float(self.delay_ms.get())
        self._run_async(
            "预算版本演进",
            lambda: workspace.change_budget(budget, batch_size=batch_size, delay_ms=delay),
            lambda result: (
                self._set_output(self.version_text, result),
                self._refresh_overview(),
            ),
        )

    def _fill_sample_change(self) -> None:
        self.change_text.delete("1.0", "end")
        self.change_text.insert(
            "end", json.dumps(sample_changes(), ensure_ascii=False, indent=2)
        )

    def _apply_change(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        try:
            payload = json.loads(self.change_text.get("1.0", "end"))
            changes = payload if isinstance(payload, list) else payload["changes"]
        except (ValueError, KeyError, TypeError) as error:
            messagebox.showerror("变更集格式错误", str(error))
            return
        delay = float(self.delay_ms.get())
        self._run_async(
            "增量更新",
            lambda: workspace.apply_changeset(changes, delay_ms=delay),
            lambda result: (
                self._set_output(self.incremental_text, result),
                self._refresh_overview(),
            ),
        )

    def _refresh_ledger(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        rows = workspace.ledger.current_task_rows()
        self._set_output(
            self.ledger_text,
            {
                "statistics": workspace.ledger_statistics(),
                "batches": workspace.ledger.current_batches(),
                "tasks": rows,
            },
        )

    def _verify(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return

        def done(result: Any) -> None:
            self._set_output(self.ledger_text, result)
            self._set_output(self.snapshot_text, result)
            self.status.set("权威重放校验通过")

        self._run_async(
            "权威重放校验",
            lambda: authoritative_replay(workspace, error_code="E_DIVERGE"),
            done,
        )

    def _verify_snapshot_replay(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return

        def done(result: Any) -> None:
            self._set_output(self.snapshot_text, result)
            self.status.set("快照恢复重放校验通过")

        self._run_async(
            "快照恢复重放校验",
            lambda: authoritative_replay(workspace, error_code="E_REPLAY"),
            done,
        )

    def _choose_snapshot(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择快照文件",
            defaultextension=".json",
            filetypes=[("JSON snapshot", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.snapshot_path.set(path)

    def _save_snapshot(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        path = self.snapshot_path.get()
        self._run_async(
            "保存快照",
            lambda: save_snapshot(workspace, path),
            lambda result: self._set_output(self.snapshot_text, result),
        )

    def _restore_snapshot(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        path = self.snapshot_path.get()

        def done(result: Any) -> None:
            self._set_output(self.snapshot_text, result)
            if self.workspace is not None:
                self.parallelism.set(self.workspace.budget.global_parallelism)
                self.memory_budget.set(
                    self.workspace.budget.column_memory_budget_bytes
                )
                self.segment_quota.set(self.workspace.budget.segment_concurrency)
            self._refresh_all()

        self._run_async(
            "整体替换恢复", lambda: restore_snapshot(workspace, path), done
        )

    def _resume_scan(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        batch_size = int(self.batch_size.get())
        delay = float(self.delay_ms.get())

        def done(result: Any) -> None:
            self._set_output(self.snapshot_text, result)
            self._refresh_ledger()

        self._run_async(
            "从快照游标继续扫描",
            lambda: workspace.scan_pending(batch_size=batch_size, delay_ms=delay),
            done,
        )

    def _change_encoding(self) -> None:
        workspace = self._require_workspace()
        if workspace is None:
            return
        column = self.encoding_column.get()
        if not column:
            messagebox.showwarning("缺少列", "请选择列。")
            return
        encoding = self.encoding_name.get()
        compressed = bool(self.encoding_compressed.get())
        delay = float(self.delay_ms.get())

        def done(result: Any) -> None:
            self._set_output(self.version_text, result)
            self._refresh_overview()

        self._run_async(
            "编码策略演进",
            lambda: workspace.change_encoding_policy(
                column=column,
                encoding=encoding,
                compressed=compressed,
                delay_ms=delay,
            ),
            done,
        )

    def _on_close(self) -> None:
        if self.workspace is not None:
            self.workspace.close()
        self.root.destroy()


def launch_gui() -> None:
    root = tk.Tk()
    DiagnosticApp(root)
    root.mainloop()
