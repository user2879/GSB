"""Tkinter desktop application for segment editing, scans and snapshot review.

Tkinter is imported lazily so importing this module (or running the test suite
on a headless box without tk) never breaks the core engine.  On the Windows
official Python distribution Tkinter ships with the installer and works as-is.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .engine import ScanEngine
from .errors import ColStoreError
from .model import BudgetParams, Encoding, SegmentDefinition
from .prepare_samples import DATA_DIR, SAMPLE_DIR
from .snapshot import SnapshotStore


def launch() -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    class ColumnarApp(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("列式存储压缩与扫描系统")
            self.geometry("1080x720")
            self.engine = ScanEngine(DATA_DIR, BudgetParams(4, 1 << 16, 2))
            self.snapshot_store = SnapshotStore(os.path.join(DATA_DIR, "sessions_gui.db"))
            self._build_widgets()
            self._reload_definitions()

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        def _build_widgets(self) -> None:
            notebook = ttk.Notebook(self)
            notebook.pack(fill="both", expand=True, padx=8, pady=8)
            self.tab_segments = ttk.Frame(notebook)
            self.tab_scan = ttk.Frame(notebook)
            self.tab_timeline = ttk.Frame(notebook)
            self.tab_snapshot = ttk.Frame(notebook)
            notebook.add(self.tab_segments, text="段定义编辑")
            notebook.add(self.tab_scan, text="扫描与预算")
            notebook.add(self.tab_timeline, text="任务时间线/资源")
            notebook.add(self.tab_snapshot, text="会话快照")
            self._build_segment_tab()
            self._build_scan_tab()
            self._build_timeline_tab()
            self._build_snapshot_tab()

        def _build_segment_tab(self) -> None:
            toolbar = ttk.Frame(self.tab_segments)
            toolbar.pack(fill="x", padx=6, pady=4)
            ttk.Button(toolbar, text="重新加载样本定义", command=self._reload_definitions).pack(side="left")
            ttk.Button(toolbar, text="构建段文件", command=self._build_segments).pack(side="left", padx=6)
            ttk.Button(toolbar, text="保存当前定义 JSON", command=self._save_definition).pack(side="left")
            self.definition_text = tk.Text(self.tab_segments, wrap="none")
            self.definition_text.pack(fill="both", expand=True, padx=6, pady=6)

        def _build_scan_tab(self) -> None:
            controls = ttk.LabelFrame(self.tab_scan, text="预算参数与查询")
            controls.pack(fill="x", padx=6, pady=6)
            self.var_parallelism = tk.IntVar(value=4)
            self.var_memory = tk.IntVar(value=1 << 16)
            self.var_seg_conc = tk.IntVar(value=2)
            for idx, (label, variable) in enumerate(
                (
                    ("全局并行度", self.var_parallelism),
                    ("解码内存预算(字节)", self.var_memory),
                    ("每段并发配额", self.var_seg_conc),
                )
            ):
                ttk.Label(controls, text=label).grid(row=0, column=idx * 2, padx=4, pady=4)
                ttk.Entry(controls, textvariable=variable, width=12).grid(
                    row=0, column=idx * 2 + 1
                )
            self.query_text = tk.Text(controls, height=6)
            self.query_text.grid(row=1, column=0, columnspan=6, sticky="ew", padx=4, pady=4)
            buttons = ttk.Frame(controls)
            buttons.grid(row=2, column=0, columnspan=6, sticky="w", padx=4, pady=4)
            ttk.Button(buttons, text="执行扫描", command=self._run_scan).pack(side="left")
            ttk.Button(buttons, text="应用预算变更(版本递增)",
                       command=self._change_budget).pack(side="left", padx=6)
            ttk.Button(buttons, text="一致性校验",
                       command=self._consistency).pack(side="left")
            self.scan_output = tk.Text(self.tab_scan, wrap="none")
            self.scan_output.pack(fill="both", expand=True, padx=6, pady=6)

        def _build_timeline_tab(self) -> None:
            columns = ("task", "segment", "rg", "column", "encoding", "memory",
                       "wait_ms", "start", "end", "rows", "matched", "version")
            self.timeline = ttk.Treeview(self.tab_timeline, columns=columns, show="headings")
            for column in columns:
                self.timeline.heading(column, text=column)
                self.timeline.column(column, width=90, anchor="center")
            self.timeline.pack(fill="both", expand=True, padx=6, pady=6)
            self.resource_label = ttk.Label(self.tab_timeline, text="峰值内存: -")
            self.resource_label.pack(anchor="w", padx=6)

        def _build_snapshot_tab(self) -> None:
            bar = ttk.Frame(self.tab_snapshot)
            bar.pack(fill="x", padx=6, pady=4)
            self.var_snapshot_id = tk.StringVar(value="gui-session-1")
            ttk.Entry(bar, textvariable=self.var_snapshot_id, width=22).pack(side="left")
            ttk.Button(bar, text="保存当前会话快照", command=self._save_snapshot).pack(side="left", padx=6)
            ttk.Button(bar, text="打开并比对快照", command=self._open_snapshot).pack(side="left")
            ttk.Button(bar, text="列出快照", command=self._list_snapshots).pack(side="left", padx=6)
            self.snapshot_output = tk.Text(self.tab_snapshot, wrap="none")
            self.snapshot_output.pack(fill="both", expand=True, padx=6, pady=6)

        # ------------------------------------------------------------------
        # Segment actions
        # ------------------------------------------------------------------
        def _reload_definitions(self) -> None:
            merged: Dict[str, Any] = {"segments": []}
            for filename in sorted(os.listdir(SAMPLE_DIR)):
                if filename.startswith("segment_") and filename.endswith(".json"):
                    with open(os.path.join(SAMPLE_DIR, filename), "r", encoding="utf-8") as h:
                        merged["segments"].append(json.load(h))
            self.definition_text.delete("1.0", "end")
            self.definition_text.insert("1.0", json.dumps(merged, indent=2, ensure_ascii=False))

        def _parse_editor_definitions(self) -> List[SegmentDefinition]:
            raw = json.loads(self.definition_text.get("1.0", "end"))
            definitions = [SegmentDefinition.from_dict(item) for item in raw["segments"]]
            return definitions

        def _build_segments(self) -> None:
            try:
                self.engine = ScanEngine(
                    DATA_DIR,
                    BudgetParams(
                        self.var_parallelism.get(),
                        self.var_memory.get(),
                        self.var_seg_conc.get(),
                    ),
                )
                for definition in self._parse_editor_definitions():
                    self.engine.register_definition(definition)
                self.engine.build_segments()
            except (ColStoreError, ValueError, json.JSONDecodeError) as exc:
                messagebox.showerror("段构建失败", str(exc))
                return
            messagebox.showinfo("完成", "段文件已构建并通过全量校验")

        def _save_definition(self) -> None:
            path = os.path.join(SAMPLE_DIR, "gui_edited_segments.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    json.loads(self.definition_text.get("1.0", "end")),
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )
            messagebox.showinfo("已保存", path)

        # ------------------------------------------------------------------
        # Scan actions
        # ------------------------------------------------------------------
        def _current_budget(self) -> BudgetParams:
            return BudgetParams(
                self.var_parallelism.get(),
                self.var_memory.get(),
                self.var_seg_conc.get(),
            )

        def _ensure_engine(self) -> None:
            if not self.engine.segments:
                self._build_segments()

        def _run_scan(self) -> None:
            self._ensure_engine()
            try:
                query = json.loads(self.query_text.get("1.0", "end") or "{}")
                if not query:
                    with open(os.path.join(SAMPLE_DIR, "query_and.json"), "r",
                              encoding="utf-8") as handle:
                        query = json.load(handle)
                result = self.engine.scan(query, budget=self._current_budget())
            except ColStoreError as exc:
                messagebox.showerror(exc.code, exc.message)
                return
            self.scan_output.delete("1.0", "end")
            self.scan_output.insert("1.0", json.dumps(result.to_dict(), indent=2,
                                                       ensure_ascii=False))
            self._populate_timeline(result)

        def _change_budget(self) -> None:
            self._ensure_engine()
            try:
                report = self.engine.update_budget(self._current_budget())
            except ColStoreError as exc:
                messagebox.showerror(exc.code, exc.message)
                return
            messagebox.showinfo(
                "预算版本",
                f"budget_version={report.budget_version}\n受影响块={len(report.affected_blocks)}",
            )

        def _consistency(self) -> None:
            self._ensure_engine()
            try:
                with open(os.path.join(SAMPLE_DIR, "query_eq.json"), "r",
                          encoding="utf-8") as handle:
                    query = json.load(handle)
                report = self.engine.consistency_check(query, self.var_parallelism.get())
            except ColStoreError as exc:
                messagebox.showerror(exc.code, json.dumps(exc.context, ensure_ascii=False))
                return
            messagebox.showinfo("一致性校验", json.dumps(report, ensure_ascii=False))

        def _populate_timeline(self, result) -> None:
            for item in self.timeline.get_children():
                self.timeline.delete(item)
            for task in result.tasks:
                self.timeline.insert(
                    "",
                    "end",
                    values=(
                        task.task_id,
                        task.block.segment,
                        task.block.row_group,
                        task.block.column,
                        task.encoding.value,
                        task.memory_bytes,
                        round(task.wait_ms, 3),
                        round(task.start_ms, 3),
                        round(task.end_ms, 3),
                        task.rows,
                        task.matched_rows,
                        task.budget_version,
                    ),
                )
            self.resource_label.config(
                text=(
                    f"峰值内存: {result.stats.peak_memory_bytes} 字节 | "
                    f"预算版本: {result.budget_version} | "
                    f"匹配行: {result.stats.rows_matched} | "
                    f"下推跳过块: {result.stats.blocks_skipped_by_pushdown}"
                )
            )

        # ------------------------------------------------------------------
        # Snapshot actions
        # ------------------------------------------------------------------
        def _save_snapshot(self) -> None:
            self._ensure_engine()
            if self.engine.current_result is None:
                messagebox.showwarning("无结果", "请先执行一次扫描")
                return
            summaries = [self.engine.segments[n].file_summary() for n in self.engine.segments]
            for item in summaries:
                item["encoding_selection_version"] = self.engine.encoding_selection_version
            self.snapshot_store.save_session(
                self.var_snapshot_id.get(),
                self.engine.current_result,
                self.engine.budget,
                summaries,
                self.engine.encoding_selection_version,
                {"budget": self.engine.budget.to_dict()},
            )
            messagebox.showinfo("已保存", self.var_snapshot_id.get())

        def _open_snapshot(self) -> None:
            current = [self.engine.segments[n].file_summary() for n in self.engine.segments]
            for item in current:
                item["encoding_selection_version"] = self.engine.encoding_selection_version
            snapshot_id = self.var_snapshot_id.get()
            diffs = self.snapshot_store.compare_with_current(
                snapshot_id, current, self.engine.encoding_selection_version
            )
            session = self.snapshot_store.load_session(snapshot_id)
            if diffs:
                self.snapshot_store.mark_superseded(
                    snapshot_id, f"reopened-{snapshot_id}", diffs
                )
            self.snapshot_output.delete("1.0", "end")
            self.snapshot_output.insert(
                "1.0",
                json.dumps(
                    {"replaced": bool(diffs), "diffs": diffs, "session": session},
                    indent=2,
                    ensure_ascii=False,
                ),
            )

        def _list_snapshots(self) -> None:
            self.snapshot_output.delete("1.0", "end")
            self.snapshot_output.insert(
                "1.0", json.dumps(self.snapshot_store.list_snapshots(), indent=2,
                                   ensure_ascii=False, default=str)
            )

    app = ColumnarApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
