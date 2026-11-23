"""Tkinter diagnostic workbench."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import tkinter.font as tkfont
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .canonical import canonical_json
from .engine import DiagnosisRound, DiagnosticEngine
from .errors import WorkbenchError
from .ledger import Ledger
from .loader import load_spec, validate_spec_object
from .model import (
    Spec,
    SpecIndex,
    budget_version,
    build_index,
    chain_digest,
    read_point_set_version,
    transaction_set_version,
)
from .planner import select_plan
from .snapshot import (
    create_snapshot,
    load_snapshot,
    restore_snapshot,
    save_snapshot,
    session_identity,
)
from .visibility import QueryResult, run_per_query


class WorkbenchApp(tk.Tk):
    def __init__(
        self,
        *,
        spec_path: str | None = None,
        ledger_path: str | None = None,
        snapshot_path: str | None = None,
    ) -> None:
        super().__init__()
        self.font_family = self._choose_font_family()
        for named_font in (
            "TkDefaultFont",
            "TkTextFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
        ):
            try:
                tkfont.nametofont(named_font).configure(family=self.font_family)
            except tk.TclError:
                pass
        self.ui_font = (self.font_family, 10)
        self.title("MVCC 可见性诊断工作台 · Windows 本地")
        self.geometry("1440x900")
        self.minsize(1120, 720)
        self.spec: Spec | None = None
        self.index: SpecIndex | None = None
        self.round: DiagnosisRound | None = None
        self.ledger_cursor = 0
        self.snapshot_path = snapshot_path or "data/session_snapshot.json"
        self._chain_items: dict[str, Any] = {}
        self._query_items: dict[str, QueryResult] = {}
        self._build_ui()
        if spec_path:
            self.spec_var.set(spec_path)
        if ledger_path:
            self.ledger_var.set(ledger_path)
        self.after(100, self._initial_load)

    def _choose_font_family(self) -> str:
        families = set(tkfont.families(self))
        for candidate in (
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "WenQuanYi Zen Hei",
            "Segoe UI",
            "DejaVu Sans",
        ):
            if candidate in families:
                return candidate
        return "TkDefaultFont"

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista" if "vista" in style.theme_names() else "clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=(self.font_family, 15, "bold"))
        style.configure("Code.TLabel", font=(self.font_family, 10, "bold"))
        style.configure(
            "Budget.Horizontal.TProgressbar",
            troughcolor="#e2e8f0",
            background="#0f766e",
            lightcolor="#0f766e",
            darkcolor="#0f766e",
            bordercolor="#cbd5e1",
        )

        root = ttk.Frame(self, padding=8)
        root.pack(fill=tk.BOTH, expand=True)

        title_row = ttk.Frame(root)
        title_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(title_row, text="MVCC 快照可见性诊断", style="Title.TLabel").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="业务码: INIT")
        self.status_label = ttk.Label(
            title_row, textvariable=self.status_var, style="Code.TLabel"
        )
        self.status_label.pack(side=tk.RIGHT)

        path_row = ttk.Frame(root)
        path_row.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(path_row, text="规格").pack(side=tk.LEFT)
        self.spec_var = tk.StringVar(value="data/demo_large.json")
        ttk.Entry(path_row, textvariable=self.spec_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 4)
        )
        ttk.Button(path_row, text="浏览…", command=self._browse_spec).pack(side=tk.LEFT)
        ttk.Button(path_row, text="加载", command=self.load_spec).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(path_row, text="账本").pack(side=tk.LEFT)
        self.ledger_var = tk.StringVar(value="data/ledger.sqlite3")
        ttk.Entry(path_row, textvariable=self.ledger_var, width=30).pack(
            side=tk.LEFT, padx=(6, 4)
        )
        ttk.Button(path_row, text="浏览…", command=self._browse_ledger).pack(side=tk.LEFT)

        action_row = ttk.Frame(root)
        action_row.pack(fill=tk.X, pady=(0, 6))
        self.diagnose_button = ttk.Button(
            action_row, text="运行诊断", command=self.diagnose
        )
        self.diagnose_button.pack(side=tk.LEFT)
        ttk.Button(action_row, text="保存快照", command=self.save_snapshot).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action_row, text="快照恢复/重放", command=self.restore_snapshot).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(action_row, text="账本恢复", command=self.recover_ledger).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Separator(action_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        ttk.Button(action_row, text="注入环 E_CYCLE", command=self.inject_cycle).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            action_row, text="注入快照不符 E_SNAPSHOT", command=self.inject_snapshot
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(action_row, text="退出", command=self.destroy).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self._build_overview_tab()
        self._build_chains_tab()
        self._build_queries_tab()
        self._build_schedule_tab()
        self._build_state_tab()

        self.detail_var = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.detail_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            fill=tk.X, pady=(6, 0)
        )

    def _build_overview_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="诊断概览")
        paned = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)
        ttk.Label(left, text="选路、阈值边界与一致性", style="Title.TLabel").pack(anchor=tk.W)
        self.overview_text = tk.Text(left, wrap=tk.WORD, font=self.ui_font)
        self.overview_text.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        ttk.Label(right, text="关键规模参数", style="Title.TLabel").pack(anchor=tk.W)
        self.metrics_tree = ttk.Treeview(
            right, columns=("metric", "value"), show="headings", height=12
        )
        self.metrics_tree.heading("metric", text="参数")
        self.metrics_tree.heading("value", text="值")
        self.metrics_tree.column("metric", width=220)
        self.metrics_tree.column("value", width=170, anchor=tk.E)
        self.metrics_tree.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

    def _build_chains_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="版本链结构")
        paned = ttk.Panedwindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)
        columns = ("row", "version", "tx", "outcome", "commit", "prev", "digest")
        self.chain_tree = ttk.Treeview(left, columns=columns, show="tree headings")
        headings = {
            "row": "行键 / 版本",
            "version": "版本",
            "tx": "事务",
            "outcome": "结局",
            "commit": "提交序号",
            "prev": "前版本",
            "digest": "值摘要",
        }
        widths = {
            "row": 220,
            "version": 70,
            "tx": 90,
            "outcome": 95,
            "commit": 90,
            "prev": 80,
            "digest": 260,
        }
        for column in columns:
            self.chain_tree.heading(column, text=headings[column])
            self.chain_tree.column(column, width=widths[column], anchor=tk.W)
        tree_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.chain_tree.yview)
        self.chain_tree.configure(yscrollcommand=tree_scroll.set)
        self.chain_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.chain_tree.bind("<<TreeviewSelect>>", self._show_chain_detail)
        ttk.Label(right, text="链结构详情", style="Title.TLabel").pack(anchor=tk.W)
        detail_frame = ttk.Frame(right)
        detail_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.chain_detail = tk.Text(detail_frame, wrap=tk.WORD, font=self.ui_font)
        detail_scroll = ttk.Scrollbar(
            detail_frame, orient=tk.VERTICAL, command=self.chain_detail.yview
        )
        self.chain_detail.configure(yscrollcommand=detail_scroll.set)
        self.chain_detail.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        detail_scroll.pack(side=tk.LEFT, fill=tk.Y)

    def _build_queries_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="读点与判定理由")
        ttk.Label(frame, text="读点快照线", style="Title.TLabel").pack(anchor=tk.W)
        self.snapshot_canvas = tk.Canvas(
            frame, height=126, background="#fbfdff", highlightthickness=1,
            highlightbackground="#c7d4e2"
        )
        self.snapshot_canvas.pack(fill=tk.X, pady=(4, 8))
        self.snapshot_canvas.bind("<Configure>", lambda _event: self._draw_snapshot_lines())
        columns = ("query", "rp", "row", "snapshot", "visible", "tx", "steps", "result", "digest")
        self.query_tree = ttk.Treeview(frame, columns=columns, show="headings", height=11)
        headings = {
            "query": "查询",
            "rp": "读点",
            "row": "行键",
            "snapshot": "快照",
            "visible": "可见版本",
            "tx": "写入事务",
            "steps": "回退步数",
            "result": "谓词",
            "digest": "理由链摘要",
        }
        widths = {
            "query": 90,
            "rp": 80,
            "row": 130,
            "snapshot": 70,
            "visible": 80,
            "tx": 90,
            "steps": 75,
            "result": 70,
            "digest": 330,
        }
        for column in columns:
            self.query_tree.heading(column, text=headings[column])
            self.query_tree.column(column, width=widths[column], anchor=tk.W)
        self.query_tree.pack(fill=tk.BOTH, expand=True)
        self.query_tree.bind("<<TreeviewSelect>>", self._show_reason_detail)
        ttk.Label(frame, text="逐步排除理由链", style="Title.TLabel").pack(
            anchor=tk.W, pady=(8, 0)
        )
        reason_frame = ttk.Frame(frame)
        reason_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.reason_text = tk.Text(reason_frame, height=9, wrap=tk.WORD, font=self.ui_font)
        reason_scroll = ttk.Scrollbar(
            reason_frame, orient=tk.VERTICAL, command=self.reason_text.yview
        )
        self.reason_text.configure(yscrollcommand=reason_scroll.set)
        self.reason_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        reason_scroll.pack(side=tk.LEFT, fill=tk.Y)

    def _build_schedule_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="调度时间线与预算")
        self.budget_frame = ttk.LabelFrame(frame, text="预算占用（实测并发峰值 / 上限）")
        self.budget_frame.pack(fill=tk.X, pady=(0, 8))
        self.timeline_canvas = tk.Canvas(
            frame, background="#fcfcfc", highlightthickness=1,
            highlightbackground="#c8c8c8"
        )
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.timeline_canvas.yview)
        self.timeline_canvas.configure(yscrollcommand=scroll.set)
        self.timeline_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.LEFT, fill=tk.Y)

    def _build_state_tab(self) -> None:
        frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(frame, text="快照 / 账本")
        ttk.Label(
            frame,
            text="版本识别、内容摘要、账本恢复与重放结论",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        self.state_text = tk.Text(frame, wrap=tk.NONE, font=self.ui_font)
        horizontal = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.state_text.xview)
        vertical = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.state_text.yview)
        self.state_text.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        self.state_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))
        horizontal.pack(side=tk.BOTTOM, fill=tk.X)
        vertical.pack(side=tk.RIGHT, fill=tk.Y)

    def _set_status(self, code: str, detail: str = "") -> None:
        self.status_var.set(f"业务码: {code}")
        self.detail_var.set(detail or code)
        self.status_label.configure(foreground="#0a7a35" if code in {"OK", "OK_RECOVERED"} else "#b42318")

    def _write_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)

    def _initial_load(self) -> None:
        self.load_spec()
        if self.spec is not None:
            self.after(200, self.diagnose)

    def _browse_spec(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 MVCC 规格",
            filetypes=[("JSON/NDJSON", "*.json *.ndjson"), ("所有文件", "*.*")],
        )
        if path:
            self.spec_var.set(path)
            self.load_spec()

    def _browse_ledger(self) -> None:
        path = filedialog.asksaveasfilename(
            title="选择 SQLite 账本",
            defaultextension=".sqlite3",
            filetypes=[("SQLite", "*.sqlite3 *.db"), ("所有文件", "*.*")],
        )
        if path:
            self.ledger_var.set(path)

    def load_spec(self) -> None:
        path = self.spec_var.get().strip()
        try:
            spec = load_spec(path)
            self.spec = spec
            self.index = build_index(spec)
            self.round = None
            plan = select_plan(spec, self.index)
            self._populate_overview(plan.to_dict(), spec)
            self._populate_chains()
            self._draw_snapshot_lines()
            self._clear_results_views()
            self._set_status("OK", f"已加载 {path}；{len(spec.queries)} 个查询")
        except WorkbenchError as exc:
            self._show_error(exc)

    def _clear_results_views(self) -> None:
        for item in self.query_tree.get_children():
            self.query_tree.delete(item)
        self._write_text(self.reason_text, "尚无判定结果。")
        self.timeline_canvas.delete("all")

    def _populate_overview(self, plan: dict[str, Any], spec: Spec) -> None:
        metrics = plan.get("metrics", {})
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for key in sorted(metrics):
            self.metrics_tree.insert("", tk.END, values=(key, metrics[key]))
        text = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True)
        self._write_text(self.overview_text, text)

    def _populate_chains(self) -> None:
        for item in self.chain_tree.get_children():
            self.chain_tree.delete(item)
        self._chain_items.clear()
        if self.spec is None or self.index is None:
            return
        for row_key in self.index.row_keys:
            row_node = self.chain_tree.insert(
                "", tk.END, text=row_key, open=False,
                values=(row_key, "", "", "", "", "", "")
            )
            records = sorted(self.index.chain(row_key), key=lambda item: item.version, reverse=True)
            for record in records:
                tx = self.index.tx_by_id[record.tx_id]
                node = self.chain_tree.insert(
                    row_node,
                    tk.END,
                    text=f"v{record.version}",
                    values=(
                        row_key,
                        record.version,
                        record.tx_id,
                        tx.outcome,
                        tx.commit_seq if tx.commit_seq is not None else "—",
                        record.prev_version if record.prev_version is not None else "—",
                        record.value_digest,
                    ),
                    tags=(tx.outcome,),
                )
                self._chain_items[node] = (row_key, record)
        self.chain_tree.tag_configure("rolledback", foreground="#9a3412")
        self.chain_tree.tag_configure("committed", foreground="#14532d")
        self._write_text(self.chain_detail, "选择版本查看结构详情。")

    def _show_chain_detail(self, _event: tk.Event[Any] | None = None) -> None:
        selected = self.chain_tree.selection()
        if not selected or selected[0] not in self._chain_items or self.index is None:
            return
        row_key, record = self._chain_items[selected[0]]
        tx = self.index.tx_by_id[record.tx_id]
        rows = [
            f"行键: {row_key}",
            f"版本: {record.version}",
            f"事务: {record.tx_id}",
            f"开始序号: {tx.start_seq}",
            f"结局: {tx.outcome}",
            f"提交序号: {tx.commit_seq if tx.commit_seq is not None else '—'}",
            f"前一版本: {record.prev_version if record.prev_version is not None else '—'}",
            f"值摘要: {record.value_digest}",
            "",
            "完整链（版本降序）:",
        ]
        for item in sorted(self.index.chain(row_key), key=lambda x: x.version, reverse=True):
            marker = " ← 当前" if item.version == record.version else ""
            rows.append(f"  v{item.version}  {item.tx_id}  digest={item.value_digest}{marker}")
        self._write_text(self.chain_detail, "\n".join(rows))

    def _draw_snapshot_lines(self) -> None:
        canvas = self.snapshot_canvas
        canvas.delete("all")
        if self.spec is None or not self.spec.read_points:
            canvas.create_text(12, 20, text="无读点", anchor=tk.W, fill="#64748b")
            return
        width = max(700, canvas.winfo_width())
        left, right = 42, width - 20
        y_axis = 78
        snapshots = [item.snapshot_seq for item in self.spec.read_points]
        commit_values = [tx.commit_seq for tx in self.spec.transactions if tx.commit_seq is not None]
        maximum = max([*snapshots, *commit_values, 1])
        canvas.create_line(left, y_axis, right, y_axis, fill="#334155", width=2)
        for tick in range(6):
            seq = round(maximum * tick / 5)
            x = left + (right - left) * tick / 5
            canvas.create_line(x, y_axis - 5, x, y_axis + 5, fill="#64748b")
            canvas.create_text(x, y_axis + 16, text=str(seq), fill="#475569", font=("Consolas", 8))
        colors = ["#dc2626", "#2563eb", "#7c3aed", "#059669", "#d97706", "#0891b2"]
        for index, read_point in enumerate(sorted(self.spec.read_points, key=lambda x: x.id)):
            x = left + (right - left) * read_point.snapshot_seq / maximum
            color = colors[index % len(colors)]
            canvas.create_line(x, 18, x, y_axis + 6, fill=color, width=2, dash=(4, 2))
            canvas.create_text(
                x + 4, 18, text=f"{read_point.id}@{read_point.snapshot_seq}",
                anchor=tk.NW, fill=color, font=("Consolas", 8)
            )
        canvas.create_text(
            left, 8, text="快照序号 →", anchor=tk.W, fill="#334155",
            font=(self.font_family, 9)
        )

    def diagnose(self) -> None:
        path = self.spec_var.get().strip()
        if not path:
            return
        self.diagnose_button.configure(state=tk.DISABLED)
        self._set_status("RUNNING", "后台线程池正在推进判定…")

        def worker() -> None:
            ledger: Ledger | None = None
            try:
                spec = load_spec(path)
                ledger = Ledger(self.ledger_var.get().strip() or "data/ledger.sqlite3")
                result = DiagnosticEngine(spec, ledger=ledger).run()
                cursor = ledger.ledger_cursor()
                self.after(0, lambda: self._apply_round(spec, result, cursor))
            except WorkbenchError as exc:
                self.after(0, lambda exc=exc: self._show_error(exc))
            except BaseException as exc:
                self.after(0, lambda exc=exc: self._show_unexpected(exc))
            finally:
                if ledger is not None:
                    ledger.close()
                self.after(0, lambda: self.diagnose_button.configure(state=tk.NORMAL))

        threading.Thread(target=worker, daemon=True, name="gui-diagnose").start()

    def _apply_round(self, spec: Spec, result: DiagnosisRound, cursor: int) -> None:
        self.spec = spec
        self.index = build_index(spec)
        self.round = result
        self.ledger_cursor = cursor
        self._populate_overview(result.plan.to_dict(), spec)
        self._populate_chains()
        self._draw_snapshot_lines()
        self._populate_query_results()
        self._draw_timeline()
        self._draw_budget()
        self._populate_state()
        self._set_status(
            "OK",
            f"{result.plan.selected_path} · {len(result.results)} 条判定 · 两路径 {result.consistency.code}",
        )

    def _populate_query_results(self) -> None:
        for item in self.query_tree.get_children():
            self.query_tree.delete(item)
        self._query_items.clear()
        if self.round is None or self.index is None:
            return
        for result in self.round.results:
            node = self.query_tree.insert(
                "",
                tk.END,
                values=(
                    result.query_id,
                    result.read_point_id,
                    result.row_key,
                    result.snapshot_seq,
                    result.visible_version if result.visible_version is not None else "—",
                    result.visible_tx_id or "—",
                    result.backtrack_steps,
                    "true" if result.predicate_result else "false",
                    result.reason_chain_digest,
                ),
                tags=("visible" if result.visible_version is not None else "none",),
            )
            self._query_items[node] = result
        self.query_tree.tag_configure("visible", foreground="#14532d")
        self.query_tree.tag_configure("none", foreground="#9a3412")
        self._write_text(self.reason_text, "选择查询查看逐步理由链。")

    def _show_reason_detail(self, _event: tk.Event[Any] | None = None) -> None:
        selected = self.query_tree.selection()
        if not selected or selected[0] not in self._query_items:
            return
        result = self._query_items[selected[0]]
        lines = [
            f"{result.query_id} | {result.read_point_id}@{result.snapshot_seq} | {result.row_key}",
            (
                f"结论 {result.terminal_code} | predicate_result={result.predicate_result} | "
                f"predicate={canonical_json(result.predicate)}"
            ),
            (
                f"可见版本 {result.visible_version if result.visible_version is not None else '—'}"
                f" tx={result.visible_tx_id or '—'} | 依据 {result.visibility_basis or '—'}"
            ),
            (
                f"回退/检查 {result.backtrack_steps}/{result.inspected_versions} | "
                f"摘要 {result.reason_chain_digest}"
            ),
            "",
            "逐步理由:",
        ]
        if not result.exclusion_reasons:
            lines.append("  (无排除，链头即满足可见性)")
        for reason in result.exclusion_reasons:
            lines.append(
                f"  step {reason.step}: v{reason.version} tx={reason.tx_id} "
                f"[{reason.code}] {reason.detail}"
            )
        lines.extend(["", "完整理由对象:", json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)])
        self._write_text(self.reason_text, "\n".join(lines))

    def _draw_timeline(self) -> None:
        canvas = self.timeline_canvas
        canvas.delete("all")
        if self.round is None:
            canvas.create_text(20, 20, text="尚无调度轨迹", anchor=tk.W)
            return
        traces = list(self.round.schedule.traces)
        if not traces:
            canvas.create_text(20, 20, text="本轮无任务", anchor=tk.W)
            return
        starts = [trace["queued_at_ms"] for trace in traces]
        ends = [trace["finished_at_ms"] for trace in traces if trace["finished_at_ms"] is not None]
        minimum = min(starts)
        maximum = max(ends or starts) or minimum + 1.0
        span = max(0.001, maximum - minimum)
        width = max(900, canvas.winfo_width() - 20)
        left, right = 200, width - 30
        row_height = 22
        top = 34
        for tick in range(6):
            x = left + (right - left) * tick / 5
            value = minimum + span * tick / 5
            canvas.create_line(x, top - 14, x, top + len(traces) * row_height, fill="#e2e8f0")
            canvas.create_text(x, 8, text=f"{value:.1f}ms", fill="#64748b", font=("Consolas", 8))
        for index, trace in enumerate(traces):
            y = top + index * row_height
            queued = trace["queued_at_ms"]
            started = trace["started_at_ms"] or queued
            finished = trace["finished_at_ms"] or started
            x0 = left + (queued - minimum) / span * (right - left)
            x1 = left + (started - minimum) / span * (right - left)
            x2 = left + (finished - minimum) / span * (right - left)
            canvas.create_text(
                8, y + 7, text=f"{trace['task_id']} @{trace['read_point_id']}",
                anchor=tk.W, fill="#334155", font=("Consolas", 8)
            )
            if trace["quota_wait_ms"] and trace["quota_wait_ms"] > 0:
                canvas.create_rectangle(
                    x0, y, max(x1, x0 + 4), y + 14, fill="#fed7aa", outline=""
                )
            if trace["blocked_by_global"] and trace["blocked_by_read_point"]:
                fill = "#c2410c"
            elif trace["blocked_by_global"]:
                fill = "#d97706"
            elif trace["blocked_by_read_point"]:
                fill = "#7c3aed"
            else:
                fill = "#0f766e"
            canvas.create_rectangle(
                max(x1, x0), y, max(x2, x1 + 5), y + 14,
                fill=fill, outline="#ffffff"
            )
        canvas.configure(scrollregion=(0, 0, width, top + len(traces) * row_height + 10))

    def _draw_budget(self) -> None:
        for child in self.budget_frame.winfo_children():
            child.destroy()
        if self.round is None:
            return
        budget = self.round.schedule.budget
        peaks = self.round.schedule.observed_peaks
        ttk.Label(
            self.budget_frame,
            text=(
                f"全局 {peaks['global']}/{budget['global_concurrency']} · "
                f"约束={'满足' if budget['constraints_ok'] else '违反'} · "
                f"阻塞任务={self.round.schedule.totals['blocked_task_count']} · "
                f"最大等待={self.round.schedule.totals['max_quota_wait_ms']}ms"
            ),
        ).pack(anchor=tk.W, padx=6, pady=2)
        global_row = ttk.Frame(self.budget_frame)
        global_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(global_row, text="全局", width=10).pack(side=tk.LEFT)
        global_bar = ttk.Progressbar(
            global_row, maximum=budget["global_concurrency"],
            value=peaks["global"], length=260,
            style="Budget.Horizontal.TProgressbar",
        )
        global_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            global_row,
            text=f"{peaks['global']}/{budget['global_concurrency']}",
            width=8,
        ).pack(side=tk.LEFT)
        for read_point_id, peak in sorted(peaks["per_read_point"].items()):
            row = ttk.Frame(self.budget_frame)
            row.pack(fill=tk.X, padx=6, pady=1)
            ttk.Label(row, text=read_point_id, width=10).pack(side=tk.LEFT)
            bar = ttk.Progressbar(
                row,
                maximum=budget["per_read_point_concurrency"],
                value=peak,
                length=260,
                style="Budget.Horizontal.TProgressbar",
            )
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(
                row,
                text=f"{peak}/{budget['per_read_point_concurrency']}",
                width=8,
            ).pack(side=tk.LEFT)

    def _populate_state(self) -> None:
        if self.spec is None:
            return
        cursor = self.ledger_cursor
        results: tuple[QueryResult, ...] = self.round.results if self.round else run_per_query(self.index) if self.index else ()
        identity = session_identity(self.spec, results, ledger_cursor=cursor)
        state = {
            "spec_digest_components": {
                "transaction_set_version": transaction_set_version(self.spec),
                "chain_digest": chain_digest(self.spec),
                "budget_version": budget_version(self.spec),
                "read_point_set_version": read_point_set_version(self.spec),
            },
            "session_identity": identity.to_dict(),
            "ledger": self.round.ledger.summary() if self.round and self.round.ledger else None,
            "schedule_peak": self.round.schedule.observed_peaks if self.round else None,
            "consistency": self.round.consistency.to_dict() if self.round else None,
            "snapshot_path": self.snapshot_path,
        }
        self._write_text(self.state_text, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))

    def save_snapshot(self) -> None:
        if self.round is None or self.spec is None:
            self._set_status("E_STATE", "请先运行诊断再保存快照")
            return
        path = filedialog.asksaveasfilename(
            title="保存会话快照",
            initialfile=Path(self.snapshot_path).name,
            initialdir=str(Path(self.snapshot_path).parent),
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        snapshot = create_snapshot(
            self.spec,
            self.round.plan.to_dict(),
            self.round.results,
            ledger_cursor=self.ledger_cursor,
        )
        summary = save_snapshot(path, snapshot)
        self.snapshot_path = path
        self._set_status("OK", f"快照已保存: {summary['content_digest']}")
        self._populate_state()

    def restore_snapshot(self) -> None:
        path = filedialog.askopenfilename(
            title="恢复并重放会话快照",
            initialdir=str(Path(self.snapshot_path).parent),
            filetypes=[("JSON", "*.json"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            current_identity = None
            if self.spec is not None:
                results = (
                    self.round.results
                    if self.round is not None
                    else run_per_query(self.index) if self.index is not None else ()
                )
                current_identity = session_identity(
                    self.spec, results, ledger_cursor=self.ledger_cursor
                )
            try:
                snapshot, replay = restore_snapshot(
                    path, expected_identity=current_identity
                )
            except WorkbenchError as exc:
                if exc.code == "E_SNAPSHOT" and exc.details.get("action") == "REPLACE_ALL":
                    replace = messagebox.askyesno(
                        "快照版本不一致",
                        "快照版本或摘要与当前会话不一致。是否按 REPLACE_ALL 整体替换，"
                        "而不是增量合并？",
                    )
                    if not replace:
                        raise
                    snapshot, replay = restore_snapshot(
                        path, expected_identity=current_identity, force_replace=True
                    )
                else:
                    raise
            spec = validate_spec_object(snapshot.spec, source_path=f"snapshot:{snapshot.snapshot_id}")
            self.spec = spec
            self.index = build_index(spec)
            self.round = None
            self.ledger_cursor = snapshot.identity.ledger_cursor
            self.snapshot_path = path
            restored_spec_path = Path("data/restored_session_spec.json")
            restored_spec_path.parent.mkdir(parents=True, exist_ok=True)
            restored_spec_path.write_text(
                json.dumps(spec.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            self.spec_var.set(str(restored_spec_path))
            plan_payload = snapshot.plan or select_plan(spec, self.index).to_dict()
            self._populate_overview(plan_payload, spec)
            self._populate_chains()
            self._draw_snapshot_lines()
            self._clear_results_views()
            self._populate_state()
            self._set_status(
                "OK",
                f"快照 {snapshot.snapshot_id} 已整体恢复，重放 {replay.query_count if replay else 0} 条一致",
            )
        except WorkbenchError as exc:
            self._show_error(exc)

    def recover_ledger(self) -> None:
        path = self.ledger_var.get().strip()
        if not path:
            self._set_status("E_STATE", "未配置账本路径")
            return
        try:
            ledger = Ledger(path)
            try:
                report = ledger.recover()
            finally:
                ledger.close()
            payload = report.to_dict()
            self._write_text(self.state_text, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
            self._set_status(
                report.code,
                f"账本游标={report.ledger_cursor}，未知轮次={len(report.unknown_rounds)}",
            )
        except WorkbenchError as exc:
            self._show_error(exc)

    def inject_cycle(self) -> None:
        path = self.spec_var.get().strip()
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            grouped: dict[str, list[dict[str, Any]]] = {}
            for record in value["records"]:
                grouped.setdefault(record["row_key"], []).append(record)
            chosen = next(items for _, items in sorted(grouped.items()) if len(items) >= 2)
            chosen[0]["prev_version"] = chosen[1]["version"]
            chosen[1]["prev_version"] = chosen[0]["version"]
            output = Path("artifacts/gui_injected_cycle.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            load_spec(output)
            self._set_status("E_INTERNAL", "环注入未触发异常")
        except WorkbenchError as exc:
            self._show_error(exc)
        except Exception as exc:
            self._show_unexpected(exc)

    def inject_snapshot(self) -> None:
        if self.round is None or self.spec is None:
            self._set_status("E_STATE", "请先运行诊断")
            return
        try:
            output = Path("artifacts/gui_injected_snapshot.json")
            output.parent.mkdir(parents=True, exist_ok=True)
            snapshot = create_snapshot(
                self.spec,
                self.round.plan.to_dict(),
                self.round.results,
                ledger_cursor=self.ledger_cursor,
                snapshot_id="gui-tampered-snapshot",
            )
            save_snapshot(output, snapshot)
            tampered = json.loads(output.read_text(encoding="utf-8"))
            tampered["results"][0]["visible_version"] = 999999
            output.write_text(
                json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            load_snapshot(output)
            self._set_status("E_INTERNAL", "快照注入未触发异常")
        except WorkbenchError as exc:
            self._show_error(exc)
        except Exception as exc:
            self._show_unexpected(exc)

    def _show_error(self, exc: WorkbenchError) -> None:
        text = json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        self._set_status(exc.code, exc.message)
        if hasattr(self, "state_text"):
            self._write_text(self.state_text, text)

    def _show_unexpected(self, exc: BaseException) -> None:
        self._set_status("E_INTERNAL", f"{type(exc).__name__}: {exc}")


def run_gui(
    *,
    spec_path: str | None = None,
    ledger_path: str | None = None,
    snapshot_path: str | None = None,
    auto_close_ms: int | None = None,
    screenshot_path: str | None = None,
) -> int:
    try:
        app = WorkbenchApp(
            spec_path=spec_path,
            ledger_path=ledger_path,
            snapshot_path=snapshot_path,
        )
    except tk.TclError as exc:
        raise WorkbenchError(
            "E_GUI",
            f"无法创建 Tk 窗口: {exc}",
            details={"hint": "Windows 官方 Python 应包含 tkinter/tcl/tk；无显示环境请使用命令行验证。"},
        ) from exc

    if screenshot_path:
        def capture() -> None:
            try:
                app.update_idletasks()
                x = app.winfo_rootx()
                y = app.winfo_rooty()
                width = app.winfo_width()
                height = app.winfo_height()
                _capture_tk_window(app, screenshot_path, (x, y, width, height))
                app._set_status("OK", f"界面截图已保存: {screenshot_path}")
            except Exception as exc:  # screenshot is an optional validation aid
                app._set_status("WARN_SCREENSHOT", str(exc))
                print(f"WARN_SCREENSHOT: {exc}", file=sys.stderr, flush=True)

        app.after(2200, capture)
    if auto_close_ms is not None:
        app.after(auto_close_ms, app.destroy)
    app.mainloop()
    return 0


def _capture_tk_window(
    app: tk.Tk,
    destination: str,
    bbox: tuple[int, int, int, int],
) -> None:
    """Capture a Tk window, preferring portable Pillow and falling back to xwd."""

    from PIL import Image

    x, y, width, height = bbox
    try:
        from PIL import ImageGrab

        image = ImageGrab.grab(bbox=(x, y, x + width, y + height))
        image.save(destination)
        return
    except Exception:
        pass

    xwd = shutil.which("xwd")
    if not xwd:
        raise RuntimeError("Pillow ImageGrab 不可用，且未找到 xwd 截图工具")
    descriptor, raw_path = tempfile.mkstemp(prefix="mvcc-ui-", suffix=".xwd")
    os.close(descriptor)
    try:
        completed = subprocess.run(
            [xwd, "-silent", "-id", str(app.winfo_id()), "-out", raw_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", errors="replace").strip())
        with open(raw_path, "rb") as handle:
            data = handle.read()
        if len(data) < 100:
            raise RuntimeError("xwd 输出不完整")
        values = struct.unpack(">25I", data[:100])
        header_size = values[0]
        pixel_width = values[4]
        pixel_height = values[5]
        byte_order = values[7]
        bits_per_pixel = values[11]
        bytes_per_line = values[12]
        color_count = values[19]
        offset = header_size + color_count * 12
        raw_pixels = data[offset : offset + bytes_per_line * pixel_height]
        if bits_per_pixel == 32:
            mode = "BGRX" if byte_order == 0 else "RGBX"
            image = Image.frombytes(
                "RGB", (pixel_width, pixel_height), raw_pixels, "raw", mode
            )
            image.save(destination)
            return
        raise RuntimeError(f"暂不支持 {bits_per_pixel} bpp 的 xwd 截图")
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass
