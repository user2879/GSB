"""Tkinter workbench: transactional editors, run timeline, resource curve, results."""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any

from .canonical import canonical_json_text
from .errors import AppError
from .ledger import ExecutionLedger, SnapshotStore
from .logging_utils import JsonLogger
from .model import Workspace
from .planner import compile_workspace
from .service import ExecutionService, RunReport, load_injected_workspace

try:  # Tk is optional at import time for headless core testing.
    import tkinter as tk
    from tkinter import messagebox, scrolledtext, ttk
except Exception as _tk_exc:  # pragma: no cover - environment dependent
    tk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    scrolledtext = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    _TK_IMPORT_ERROR = _tk_exc
else:
    _TK_IMPORT_ERROR = None


def _report_dict(report: Any) -> dict[str, Any] | None:
    if isinstance(report, RunReport):
        return report.to_dict()
    if isinstance(report, dict):
        return report
    return None


class VecFlowApp:
    def __init__(
        self,
        root: Any,
        *,
        workspace: Workspace,
        ledger: ExecutionLedger,
        snapshot_store: SnapshotStore,
        logger: JsonLogger,
    ) -> None:
        self.root = root
        self.workspace = workspace
        self.ledger = ledger
        self.snapshot_store = snapshot_store
        self.logger = logger
        self.service = ExecutionService(workspace, ledger, snapshot_store=snapshot_store, logger=logger)
        self._events: list[dict[str, Any]] = []
        self._worker_thread: threading.Thread | None = None
        self._worker_queue: queue.Queue[Any] = queue.Queue()
        self._closed = False

        self.root.title("VecFlow · 向量化执行与资源预算工作台")
        self.root.geometry("1360x880")
        self.root.minsize(1080, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.version_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        self.ledger_var = tk.StringVar(value="账本：未读取")
        self.budget_var = tk.StringVar()
        self.threads_var = tk.StringVar()
        self.task_var = tk.StringVar()
        self.quota_var = tk.StringVar()
        self.expression_var = tk.StringVar()

        self._configure_style()
        self._build_layout()
        self._refresh_all(reset_text=True)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Header.TLabel", font=("TkDefaultFont", 14, "bold"))
        style.configure("Meta.TLabel", foreground="#4c566a")
        style.configure("Error.TLabel", foreground="#b00020")
        style.configure("Treeview", rowheight=22)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 6))
        ttk.Label(header, text="VecFlow", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="  确定性列式执行 · 双层资源约束 · 可重放账本", style="Meta.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.version_var, style="Meta.TLabel").pack(side="right")

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)
        self._build_data_tab()
        self._build_tasks_tab()
        self._build_execution_tab()

        status_bar = ttk.Frame(outer)
        status_bar.pack(fill="x", pady=(6, 0))
        self.status_label = ttk.Label(status_bar, textvariable=self.status_var)
        self.status_label.pack(side="left", fill="x", expand=True)
        ttk.Label(status_bar, textvariable=self.ledger_var, style="Meta.TLabel").pack(side="right")

    def _build_data_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab, text="数据块")
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="校验并保存数据", command=self._save_data).pack(side="left")
        ttk.Button(toolbar, text="重新载入已校验状态", command=lambda: self._refresh_all(reset_text=True)).pack(side="left", padx=6)
        ttk.Label(toolbar, text="非法编辑整体拒绝为 E_DATA，不覆盖已加载状态", style="Meta.TLabel").pack(side="left", padx=10)

        paned = ttk.Panedwindow(tab, orient="vertical")
        paned.pack(fill="both", expand=True)
        summary_frame = ttk.Frame(paned)
        ttk.Label(summary_frame, text="数据块与列摘要（按标识稳定排序）").pack(anchor="w")
        self.data_tree = ttk.Treeview(summary_frame, columns=("block", "rows", "columns"), show="headings", height=7)
        for name, label, width in (("block", "数据块", 180), ("rows", "行数", 90), ("columns", "列（类型）", 800)):
            self.data_tree.heading(name, text=label)
            self.data_tree.column(name, width=width, anchor="w")
        self.data_tree.pack(fill="both", expand=True)
        paned.add(summary_frame, weight=1)

        editor_frame = ttk.Frame(paned)
        ttk.Label(editor_frame, text="列式数据 JSON（values / null_bitmap 可编辑）").pack(anchor="w")
        self.data_text = scrolledtext.ScrolledText(editor_frame, wrap="none", font=("TkFixedFont", 10), undo=True)
        self.data_text.pack(fill="both", expand=True)
        paned.add(editor_frame, weight=3)

    def _build_tasks_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab, text="任务与预算")
        settings = ttk.LabelFrame(tab, text="全局参数（修改后参数版本单调递增并整体失效旧运行）", padding=6)
        settings.pack(fill="x", pady=(0, 6))
        ttk.Label(settings, text="全局预算单位").pack(side="left")
        ttk.Entry(settings, textvariable=self.budget_var, width=8).pack(side="left", padx=(4, 14))
        ttk.Label(settings, text="工作线程数").pack(side="left")
        ttk.Entry(settings, textvariable=self.threads_var, width=8).pack(side="left", padx=(4, 14))
        ttk.Button(settings, text="应用预算与线程", command=self._apply_runtime_settings).pack(side="left")

        quick = ttk.Frame(tab)
        quick.pack(fill="x", pady=(0, 6))
        ttk.Label(quick, text="任务").pack(side="left")
        self.task_combo = ttk.Combobox(quick, textvariable=self.task_var, state="readonly", width=22)
        self.task_combo.pack(side="left", padx=4)
        self.task_combo.bind("<<ComboboxSelected>>", self._on_task_selected)
        ttk.Label(quick, text="配额").pack(side="left")
        ttk.Entry(quick, textvariable=self.quota_var, width=6).pack(side="left", padx=(4, 8))
        ttk.Button(quick, text="应用配额", command=self._apply_task_quota).pack(side="left")
        ttk.Label(quick, text="表达式").pack(side="left", padx=(14, 4))
        ttk.Entry(quick, textvariable=self.expression_var, width=58).pack(side="left", fill="x", expand=True)
        ttk.Button(quick, text="应用表达式", command=self._apply_expression).pack(side="left", padx=(4, 0))

        paned = ttk.Panedwindow(tab, orient="vertical")
        paned.pack(fill="both", expand=True)
        summary = ttk.Frame(paned)
        ttk.Label(summary, text="任务、规范化表达式与分片规划").pack(anchor="w")
        self.task_tree = ttk.Treeview(
            summary,
            columns=("id", "block", "expr", "quota", "agg", "shards", "digest"),
            show="headings",
            height=8,
        )
        widths = {"id": 160, "block": 120, "expr": 520, "quota": 70, "agg": 90, "shards": 130, "digest": 170}
        for column in self.task_tree["columns"]:
            self.task_tree.heading(column, text=column)
            self.task_tree.column(column, width=widths[str(column)], anchor="w")
        self.task_tree.bind("<<TreeviewSelect>>", self._on_task_selected)
        self.task_tree.pack(fill="both", expand=True)
        paned.add(summary, weight=2)

        editor = ttk.Frame(paned)
        edit_toolbar = ttk.Frame(editor)
        edit_toolbar.pack(fill="x")
        ttk.Button(edit_toolbar, text="校验并保存任务", command=self._save_tasks).pack(side="left")
        ttk.Label(edit_toolbar, text="任务 JSON（含 parameter_version / global_budget / threads）", style="Meta.TLabel").pack(side="left", padx=8)
        self.tasks_text = scrolledtext.ScrolledText(editor, wrap="none", font=("TkFixedFont", 10), undo=True)
        self.tasks_text.pack(fill="both", expand=True)
        paned.add(editor, weight=3)

    def _build_execution_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=6)
        self.notebook.add(tab, text="执行与账本")
        toolbar = ttk.Frame(tab)
        toolbar.pack(fill="x", pady=(0, 6))
        self.run_button = ttk.Button(toolbar, text="执行当前计划", command=self._start_run)
        self.run_button.pack(side="left")
        ttk.Button(toolbar, text="刷新状态", command=lambda: self._refresh_all(reset_text=False)).pack(side="left", padx=6)
        ttk.Button(toolbar, text="账本恢复检查", command=self._recover).pack(side="left")
        ttk.Button(toolbar, text="整体恢复快照", command=self._restore_snapshot).pack(side="left", padx=6)
        ttk.Label(toolbar, text="时间线按任务/分片稳定顺序渲染，不按线程完成顺序", style="Meta.TLabel").pack(side="left", padx=10)

        self.run_status_var = tk.StringVar(value="尚未执行")
        ttk.Label(tab, textvariable=self.run_status_var, style="Meta.TLabel").pack(anchor="w", pady=(0, 4))

        paned = ttk.Panedwindow(tab, orient="vertical")
        paned.pack(fill="both", expand=True)
        timeline_frame = ttk.Frame(paned)
        ttk.Label(timeline_frame, text="分片排队、运行与资源时间线").pack(anchor="w")
        columns = ("task", "shard", "rows", "res", "queued", "start", "end", "wait", "status", "outcome", "digest")
        self.timeline_tree = ttk.Treeview(timeline_frame, columns=columns, show="headings", height=10)
        labels = {
            "task": "任务",
            "shard": "分片",
            "rows": "行区间",
            "res": "资源",
            "queued": "排队开始ms",
            "start": "开始ms",
            "end": "结束ms",
            "wait": "等待ms",
            "status": "状态",
            "outcome": "结局",
            "digest": "结果摘要",
        }
        widths = {"task": 150, "shard": 190, "rows": 90, "res": 60, "queued": 110, "start": 105, "end": 105, "wait": 75, "status": 95, "outcome": 80, "digest": 220}
        for column in columns:
            self.timeline_tree.heading(column, text=labels[column])
            self.timeline_tree.column(column, width=widths[column], anchor="w")
        self.timeline_tree.tag_configure("stale", background="#f0d0d0")
        self.timeline_tree.tag_configure("failed", background="#ffe0b2")
        timeline_scroll = ttk.Scrollbar(timeline_frame, orient="vertical", command=self.timeline_tree.yview)
        self.timeline_tree.configure(yscrollcommand=timeline_scroll.set)
        self.timeline_tree.pack(side="left", fill="both", expand=True)
        timeline_scroll.pack(side="right", fill="y")
        paned.add(timeline_frame, weight=2)

        visual = ttk.Frame(paned)
        chart_frame = ttk.LabelFrame(visual, text="资源占用曲线（活动资源单位）", padding=4)
        chart_frame.pack(fill="x")
        self.chart = tk.Canvas(chart_frame, height=170, background="#fbfbfd", highlightthickness=0)
        self.chart.pack(fill="x", expand=True)
        self.chart.bind("<Configure>", lambda event: self._draw_chart())

        result_frame = ttk.LabelFrame(visual, text="任务结果（规范 JSON；陈旧运行不参与当前汇总）", padding=4)
        result_frame.pack(fill="both", expand=True, pady=(6, 0))
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap="none", font=("TkFixedFont", 9), height=12)
        self.result_text.pack(fill="both", expand=True)
        paned.add(visual, weight=3)

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_var.set(text)
        self.status_label.configure(style="Error.TLabel" if error else "TLabel")

    def _refresh_all(self, *, reset_text: bool) -> None:
        state = self.workspace.state
        self.version_var.set(
            f"参数版本 {state.parameter_version} | content {self.workspace.content_digest[:16]} | "
            f"budget {state.global_budget} | threads {state.threads}"
        )
        self.budget_var.set(str(state.global_budget))
        self.threads_var.set(str(state.threads))
        if reset_text:
            self._set_text(self.data_text, self.workspace.data_json())
            self._set_text(self.tasks_text, self.workspace.tasks_json())
            self._set_status(f"已加载 {len(state.blocks.blocks)} 个数据块、{len(state.tasks.tasks)} 个任务")
        self._refresh_data_summary()
        self._refresh_task_summary()
        self._refresh_timeline()
        self._refresh_results()
        self._refresh_ledger()

    def _set_text(self, widget: Any, text: str) -> None:
        widget.delete("1.0", "end")
        widget.insert("1.0", text)

    def _refresh_data_summary(self) -> None:
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        for block in sorted(self.workspace.state.blocks.blocks, key=lambda item: item.id):
            columns = ", ".join(f"{column.name}:{column.type}" for column in block.columns)
            self.data_tree.insert("", "end", iid=block.id, values=(block.id, block.row_count, columns))

    def _refresh_task_summary(self) -> None:
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        state = self.workspace.state
        block_map = state.by_block()
        task_ids: list[str] = []
        for task in sorted(state.tasks.tasks, key=lambda item: item.id):
            block = block_map[task.block_id]
            if task.shards is not None:
                shard_text = f"显式 {len(task.shards)}"
            else:
                count = math.ceil(block.row_count / int(task.shard_rows or 1))
                shard_text = f"生成 {count} × {task.shard_rows} 行"
            self.task_tree.insert(
                "",
                "end",
                iid=task.id,
                values=(
                    task.id,
                    task.block_id,
                    task.parsed.normalized,
                    task.quota,
                    task.aggregation,
                    shard_text,
                    task.parsed.structural_digest[:16],
                ),
            )
            task_ids.append(task.id)
        self.task_combo.configure(values=task_ids)
        if task_ids:
            if self.task_var.get() not in task_ids:
                self.task_var.set(task_ids[0])
            self._on_task_selected()
        else:
            self.task_var.set("")

    def _on_task_selected(self, _event: Any = None) -> None:
        task_id = self.task_var.get()
        tree_selection = self.task_tree.selection()
        if tree_selection and tree_selection[0] != task_id:
            task_id = tree_selection[0]
            self.task_var.set(task_id)
        task_map = self.workspace.state.tasks.by_id()
        task = task_map.get(task_id)
        if task is None:
            return
        self.quota_var.set(str(task.quota))
        self.expression_var.set(task.parsed.normalized)

    def _refresh_timeline(self) -> None:
        for item in self.timeline_tree.get_children():
            self.timeline_tree.delete(item)
        report = _report_dict(self.workspace.current_run)
        if report and report.get("schedule") and report["schedule"].get("records"):
            stale = bool(report.get("stale")) or report.get("status") == "stale"
            for record in report["schedule"]["records"]:
                tag = "stale" if stale else ("failed" if record.get("outcome") not in (None, "success") else "")
                self.timeline_tree.insert(
                    "",
                    "end",
                    iid=record["shard_id"],
                    values=(
                        record["task_id"],
                        record["shard_id"],
                        f"[{record['start']},{record['end']})",
                        record["resources"],
                        record["queued_at_ms"],
                        record.get("started_at_ms") or "",
                        record.get("ended_at_ms") or "",
                        record.get("wait_duration_ms") if record.get("wait_duration_ms") is not None else "",
                        record["status"],
                        record.get("outcome") or "",
                        (record.get("result_digest") or "")[:20],
                    ),
                    tags=(tag,) if tag else (),
                )
            self._events = list(report["schedule"].get("resource_events") or [])
            self.run_status_var.set(
                f"运行 {report['run_id']} | 状态 {report.get('status')} | 总摘要 {report.get('overall_digest') or '-'} | "
                f"峰值资源 {report['schedule'].get('peak_resources')}"
            )
        else:
            try:
                plan = compile_workspace(self.workspace.state)
            except AppError as exc:
                self.run_status_var.set(f"计划错误 {exc.code}: {exc.message}")
                self._events = []
                self._draw_chart()
                return
            for task in plan.tasks:
                for shard in task.shards:
                    self.timeline_tree.insert(
                        "",
                        "end",
                        iid=shard.shard_id,
                        values=(
                            task.task_id,
                            shard.shard_id,
                            f"[{shard.start_row},{shard.end_row})",
                            shard.resources,
                            "",
                            "",
                            "",
                            "",
                            "queued",
                            "",
                            "",
                        ),
                    )
            self._events = []
            self.run_status_var.set(f"已编译计划：{plan.shard_count} 个分片，等待执行")
        self._draw_chart()

    def _draw_chart(self) -> None:
        canvas = self.chart
        canvas.delete("all")
        width = max(400, canvas.winfo_width())
        height = max(140, canvas.winfo_height())
        left, right, top, bottom = 52, width - 18, 18, height - 30
        canvas.create_line(left, bottom, right, bottom, fill="#8892a0")
        canvas.create_line(left, top, left, bottom, fill="#8892a0")
        if not self._events:
            canvas.create_text(width / 2, height / 2, text="尚无资源事件", fill="#6b7280")
            return
        events = sorted(self._events, key=lambda event: (event["at_ms"], 0 if event["kind"] == "end" else 1, event["sequence"]))
        start_ms = events[0]["at_ms"]
        end_ms = max(start_ms + 1, events[-1]["at_ms"])
        max_units = max(max(int(event["active_units"]) for event in events), self.workspace.global_budget, 1)
        budget = self.workspace.global_budget

        def x_of(ms: int) -> float:
            return left + (ms - start_ms) / max(1, end_ms - start_ms) * (right - left)

        def y_of(units: int) -> float:
            return bottom - units / max_units * (bottom - top)

        if budget > 0:
            budget_y = y_of(budget)
            canvas.create_line(left, budget_y, right, budget_y, fill="#d1495b", dash=(4, 3))
            canvas.create_text(right - 4, max(top, budget_y - 9), text=f"预算 {budget}", anchor="e", fill="#d1495b", font=("TkDefaultFont", 8))
        points: list[float] = []
        last_y = y_of(int(events[0]["active_units"]) if events[0]["kind"] == "start" else 0)
        for event in events:
            x = x_of(int(event["at_ms"]))
            points.extend([x, last_y])
            last_y = y_of(int(event["active_units"]))
            points.extend([x, last_y])
        if len(points) >= 4:
            canvas.create_line(*points, fill="#2f6f9f", width=2)
        for tick in range(0, max_units + 1):
            if max_units <= 8 or tick % max(1, math.ceil(max_units / 6)) == 0:
                y = y_of(tick)
                canvas.create_text(left - 20, y, text=str(tick), fill="#4c566a", font=("TkDefaultFont", 8))
        canvas.create_text((left + right) / 2, height - 10, text=f"单调时钟毫秒：{start_ms} → {end_ms}", fill="#4c566a", font=("TkDefaultFont", 8))

    def _refresh_results(self) -> None:
        report = _report_dict(self.workspace.current_run)
        if report is None:
            self._set_text(self.result_text, "尚未生成任务结果。\n")
            return
        lines: list[str] = []
        stale = bool(report.get("stale")) or report.get("status") == "stale"
        if stale:
            self._set_text(
                self.result_text,
                "STALE：该结果属于旧参数版本，已从当前结果展示与汇总中排除；请在新版本上整体重跑。\n",
            )
            return
        lines.append(f"run_id: {report.get('run_id')}")
        lines.append(f"status: {report.get('status')}")
        lines.append(f"parameter_version: {report.get('parameter_version')}")
        lines.append(f"content_digest: {report.get('content_digest')}")
        lines.append(f"plan_digest: {report.get('plan_digest')}")
        lines.append(f"overall_digest: {report.get('overall_digest')}")
        lines.append("")
        task_results = report.get("task_results") or {}
        if not task_results:
            lines.append("没有可展示的任务结果（失败或陈旧运行不参与汇总）。")
        for task_id in sorted(task_results):
            lines.append(f"## {task_id}")
            lines.append(canonical_json_text(task_results[task_id], pretty=True))
        self._set_text(self.result_text, "\n".join(lines))

    def _refresh_ledger(self) -> None:
        try:
            info = self.ledger.ledger_info()
        except AppError as exc:
            self.ledger_var.set(f"账本错误 {exc.code}")
            return
        latest = info.get("latest")
        if latest is None:
            self.ledger_var.set(f"账本格式 v{info['format_version']} | 空")
        else:
            self.ledger_var.set(
                f"账本格式 v{info['format_version']} | 参数 v{latest['parameter_version']} | "
                f"{latest['status']} | {latest['batch_id'][:10]}"
            )

    def _handle_edit_result(self, result: Any, label: str) -> None:
        message = f"{label}成功：参数版本 {result.parameter_version}，changed={result.changed}"
        if result.affected_running or result.affected_queued:
            message += (
                f"；旧运行已失效。运行中 {list(result.affected_running)}，"
                f"排队中 {list(result.affected_queued)}"
            )
        self._set_status(message)
        self.logger.log(
            "gui_edit_committed",
            parameter_version=result.parameter_version,
            changed=result.changed,
            affected_running=list(result.affected_running),
            affected_queued=list(result.affected_queued),
            stale_run_ids=list(result.stale_run_ids),
        )
        self._refresh_all(reset_text=True)

    def _show_error(self, exc: AppError, label: str) -> None:
        self._set_status(f"{label}失败 {exc.code}: {exc.message}", error=True)
        if messagebox is not None:
            messagebox.showerror(exc.code, exc.message)

    def _save_data(self) -> None:
        try:
            result = self.workspace.replace_data_json(self.data_text.get("1.0", "end-1c"), persist=True)
        except AppError as exc:
            self._show_error(exc, "保存数据")
            return
        self._handle_edit_result(result, "保存数据")

    def _save_tasks(self) -> None:
        try:
            result = self.workspace.replace_tasks_json(self.tasks_text.get("1.0", "end-1c"), persist=True)
        except AppError as exc:
            self._show_error(exc, "保存任务")
            return
        self._handle_edit_result(result, "保存任务")

    def _apply_runtime_settings(self) -> None:
        try:
            budget = int(self.budget_var.get().strip())
            threads = int(self.threads_var.get().strip())
            result = self.workspace.update_runtime_settings(budget, threads, persist=True)
        except (ValueError, AppError) as exc:
            app_error = exc if isinstance(exc, AppError) else AppError("E_DATA", "预算与线程必须是整数", {"value": str(exc)})
            self._show_error(app_error, "应用预算")
            return
        self._handle_edit_result(result, "应用预算")

    def _apply_task_quota(self) -> None:
        task_id = self.task_var.get().strip()
        if not task_id:
            self._show_error(AppError("E_DATA", "请选择任务"), "应用配额")
            return
        try:
            quota = int(self.quota_var.get().strip())
            result = self.workspace.update_task_quota(task_id, quota, persist=True)
        except (ValueError, AppError) as exc:
            app_error = exc if isinstance(exc, AppError) else AppError("E_DATA", "配额必须是整数", {"value": str(exc)})
            self._show_error(app_error, "应用配额")
            return
        self._handle_edit_result(result, "应用配额")

    def _apply_expression(self) -> None:
        task_id = self.task_var.get().strip()
        if not task_id:
            self._show_error(AppError("E_DATA", "请选择任务"), "应用表达式")
            return
        try:
            result = self.workspace.update_expression(task_id, self.expression_var.get(), persist=True)
        except AppError as exc:
            self._show_error(exc, "应用表达式")
            return
        self._handle_edit_result(result, "应用表达式")

    def _start_run(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._set_status("已有执行正在进行", error=True)
            return
        self.run_button.configure(state="disabled")
        self._set_status("执行中：分片按预算与任务配额调度")
        self.run_status_var.set("运行中…")

        def worker() -> None:
            try:
                report = self.service.execute(save_snapshot=True)
                self._worker_queue.put(("ok", report))
            except AppError as exc:
                self._worker_queue.put(("error", exc, self.service.last_report))
            except Exception as exc:
                self._worker_queue.put(("error", AppError("E_INTERNAL", repr(exc)), self.service.last_report))

        self._worker_thread = threading.Thread(target=worker, name="vecflow-gui-run", daemon=True)
        self._worker_thread.start()
        self.root.after(100, self._poll_run)

    def _poll_run(self) -> None:
        if self._closed:
            return
        try:
            message = self._worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_run)
            return
        self.run_button.configure(state="normal")
        if message[0] == "ok":
            report = message[1]
            self._refresh_all(reset_text=False)
            self._set_status(f"执行完成：{report.status}，总摘要 {report.overall_digest or '-'}")
        else:
            _, error, report = message
            self._refresh_all(reset_text=False)
            if report is not None:
                self.workspace.current_run = report
                self._refresh_timeline()
                self._refresh_results()
            self._show_error(error, "执行")

    def _recover(self) -> None:
        try:
            interrupted = self.ledger.recover_incomplete()
            latest = self.ledger.latest_batch(recover=False)
        except AppError as exc:
            self._show_error(exc, "账本恢复")
            return
        if interrupted:
            text = "; ".join(f"{item['batch_id']} unknown={len(item['unknown_shards'])}" for item in interrupted)
            self._set_status(f"恢复检查：中断批次已标记未知，禁止退出报告；{text}")
        else:
            self._set_status("恢复检查：没有未完成批次")
        if latest is not None:
            self.run_status_var.set(f"账本最新批次 {latest.batch_id} | {latest.status} | exit_report={latest.exit_report_allowed}")
        self._refresh_ledger()

    def _restore_snapshot(self) -> None:
        path = self.snapshot_store.default_path
        if not path.exists():
            self._show_error(AppError("E_DATA", f"快照不存在: {path}"), "恢复快照")
            return
        try:
            restored = self.service.restore_snapshot(path)
            self.workspace.persist()
        except AppError as exc:
            self._show_error(exc, "恢复快照")
            return
        self._refresh_all(reset_text=True)
        self._set_status(
            f"快照已整体替换：参数版本 {restored.parameter_version}，差异字段 {list(restored.mismatched_fields)}"
        )

    def _on_close(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive() and messagebox is not None:
            if not messagebox.askyesno("执行仍在进行", "执行线程仍在运行。退出会放弃当前界面展示，是否继续？"):
                return
        self._closed = True
        self.root.destroy()


def run_gui(
    *,
    data_path: str,
    tasks_path: str,
    ledger_path: str,
    snapshot_dir: str,
    logger: JsonLogger,
    smoke: bool = False,
    smoke_ms: int = 800,
) -> dict[str, Any]:
    if tk is None:
        raise AppError(
            "E_GUI",
            f"Tkinter 不可用: {_TK_IMPORT_ERROR}；Windows 官方 Python 通常内置 Tk",
            {"tkinter_available": False},
        )
    workspace, _ = load_injected_workspace(data_path, tasks_path, injection="none")
    ledger = ExecutionLedger(ledger_path)
    snapshot_store = SnapshotStore(snapshot_dir)
    app: VecFlowApp | None = None
    try:
        try:
            root = tk.Tk()
        except Exception as exc:
            raise AppError("E_GUI", f"无法创建 Tk 窗口: {exc}", {"display": __import__("os").environ.get("DISPLAY")}) from exc
        app = VecFlowApp(
            root,
            workspace=workspace,
            ledger=ledger,
            snapshot_store=snapshot_store,
            logger=logger,
        )
        logger.log("gui_started", data=data_path, tasks=tasks_path, ledger=ledger_path, smoke=smoke)
        if smoke:
            root.update_idletasks()
            root.update()
            time.sleep(max(0, smoke_ms) / 1000.0)
            root.update_idletasks()
            root.update()
            result = {
                "ok": True,
                "command": "gui",
                "smoke": True,
                "tk_version": float(tk.TkVersion),
                "blocks": len(workspace.state.blocks.blocks),
                "tasks": len(workspace.state.tasks.tasks),
                "timeline_rows": len(app.timeline_tree.get_children()),
                "parameter_version": workspace.parameter_version,
            }
            root.destroy()
            return result
        root.mainloop()
        return {"ok": True, "command": "gui", "closed": True}
    finally:
        if app is not None and app._worker_thread is not None and app._worker_thread.is_alive():
            app._worker_thread.join(timeout=5)
        ledger.close()
