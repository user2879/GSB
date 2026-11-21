"""Tkinter diagnostic workbench for local Windows use."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .engine import Engine
from .errors import E_ENV, WorkbenchError
from .ledger import Ledger
from .loader import load_workspace
from .logs import JsonlLogger
from .recovery import RecoveryManager


def _short(value: str | None, length: int = 14) -> str:
    if not value:
        return "-"
    return value if len(value) <= length else value[:length] + "…"


class WorkbenchUI:
    def __init__(
        self,
        root: tk.Tk,
        db_path: str,
        data_path: str,
        pipeline_path: str,
        log_path: str,
    ) -> None:
        self.root = root
        self.db_path = str(Path(db_path).resolve())
        self.data_path = str(Path(data_path).resolve())
        self.pipeline_path = str(Path(pipeline_path).resolve())
        self.log_path = str(Path(log_path).resolve())
        self.logger = JsonlLogger(self.log_path)
        self.ledger = Ledger(self.db_path, self.logger)
        self.engine = Engine(self.ledger)
        self.recovery = RecoveryManager(self.ledger, self.engine)
        self.prepared = None
        self.current_view: dict[str, Any] | None = None
        self.selected_run_id: str | None = None
        self._busy = False

        self.root.title("ML 训练数据管线诊断工作台")
        self.root.geometry("1320x860")
        self.root.minsize(1080, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._configure_style()
        self._build_layout()
        self._load_prepared(show_dialog=False)
        self.refresh_all()
        self.append_activity("工作台已启动。账本: " + self.db_path)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#f4f7fa"
        ink = "#14212f"
        accent = "#0b6e75"
        self.root.configure(background=bg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=ink)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=ink)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 9), foreground="#516171")
        style.configure("Value.TLabel", font=("Consolas", 9), foreground="#173b4d")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground=accent)
        style.configure("Banner.TLabel", font=("Segoe UI", 10, "bold"), foreground="#7a271a")
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 5))
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), padding=(12, 6))
        style.configure("Treeview", rowheight=23, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook.Tab", font=("Segoe UI", 9, "bold"), padding=(12, 6))
        style.configure("Status.TLabel", background="#dfe8ef", foreground="#33475b", padding=6)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=(16, 12, 16, 6))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="ML 训练数据管线诊断工作台", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="确定性执行计划 · 版本化观测 · 崩溃恢复 · 重放一致性",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.banner_var = tk.StringVar(value="业务码: -")
        ttk.Label(header, textvariable=self.banner_var, style="Banner.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="e"
        )

        specbar = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        specbar.grid(row=1, column=0, sticky="ew")
        specbar.columnconfigure(1, weight=1)
        specbar.columnconfigure(3, weight=1)
        ttk.Label(specbar, text="规格版本", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.spec_version_var = tk.StringVar(value="-")
        ttk.Label(specbar, textvariable=self.spec_version_var, style="Value.TLabel").grid(
            row=0, column=1, sticky="w", padx=(8, 24)
        )
        ttk.Label(specbar, text="规格指纹", style="Section.TLabel").grid(row=0, column=2, sticky="w")
        self.fingerprint_var = tk.StringVar(value="-")
        ttk.Label(specbar, textvariable=self.fingerprint_var, style="Value.TLabel").grid(
            row=0, column=3, sticky="w", padx=(8, 0)
        )
        ttk.Label(specbar, text="样本", style="Section.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.data_path_var = tk.StringVar(value=self.data_path)
        ttk.Label(specbar, textvariable=self.data_path_var, style="Value.TLabel").grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(8, 0), pady=(4, 0)
        )

        toolbar = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        toolbar.grid(row=2, column=0, sticky="new")
        self.root.rowconfigure(2, weight=0)
        self.data_button = ttk.Button(toolbar, text="选择样本…", command=self.select_data)
        self.data_button.pack(side="left", padx=(0, 6))
        self.pipeline_button = ttk.Button(toolbar, text="选择规格…", command=self.select_pipeline)
        self.pipeline_button.pack(side="left", padx=(0, 6))
        self.load_button = ttk.Button(toolbar, text="加载 / 校验", command=self.reload_prepared)
        self.load_button.pack(side="left", padx=(0, 14))
        self.run_button = ttk.Button(
            toolbar, text="运行并整体替换", style="Accent.TButton", command=self.run_current
        )
        self.run_button.pack(side="left", padx=(0, 6))
        self.recover_button = ttk.Button(toolbar, text="恢复中断运行", command=self.recover_selected)
        self.recover_button.pack(side="left", padx=(0, 6))
        self.replay_button = ttk.Button(toolbar, text="重放校验", command=self.replay_selected)
        self.replay_button.pack(side="left", padx=(0, 6))
        self.refresh_button = ttk.Button(toolbar, text="刷新", command=self.refresh_all)
        self.refresh_button.pack(side="left")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 8))
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        self.notebook = ttk.Notebook(body)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self._build_overview_tab()
        self._build_ledger_tab()
        self._build_activity_tab()

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel").grid(
            row=4, column=0, sticky="ew"
        )

    def _build_overview_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="管线总览")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=3)
        tab.rowconfigure(5, weight=2)

        self.dependency_var = tk.StringVar(value="依赖链: -")
        ttk.Label(tab, textvariable=self.dependency_var, style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(tab, text="阶段状态与逐阶段摘要", style="Section.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 4)
        )
        stage_frame = ttk.Frame(tab)
        stage_frame.grid(row=2, column=0, sticky="nsew")
        stage_frame.columnconfigure(0, weight=1)
        stage_frame.rowconfigure(0, weight=1)
        stage_columns = ("ordinal", "stage", "type", "version", "upstream", "status", "rows", "digest")
        self.stage_tree = ttk.Treeview(
            stage_frame, columns=stage_columns, show="headings", selectmode="browse"
        )
        headings = {
            "ordinal": "#",
            "stage": "阶段标识",
            "type": "类型",
            "version": "版本",
            "upstream": "上游",
            "status": "状态",
            "rows": "行数",
            "digest": "输出摘要",
        }
        widths = {
            "ordinal": 38,
            "stage": 150,
            "type": 120,
            "version": 64,
            "upstream": 130,
            "status": 120,
            "rows": 70,
            "digest": 410,
        }
        for column in stage_columns:
            self.stage_tree.heading(column, text=headings[column])
            self.stage_tree.column(
                column,
                width=widths[column],
                minwidth=36,
                anchor="center" if column not in ("stage", "upstream", "digest") else "w",
            )
        stage_y = ttk.Scrollbar(stage_frame, orient="vertical", command=self.stage_tree.yview)
        self.stage_tree.configure(yscrollcommand=stage_y.set)
        self.stage_tree.grid(row=0, column=0, sticky="nsew")
        stage_y.grid(row=0, column=1, sticky="ns")
        self.stage_tree.tag_configure("trusted", background="#eef8f4")
        self.stage_tree.tag_configure("planned", background="#f8f5e8")
        self.stage_tree.tag_configure("failed", background="#fdecea")

        ttk.Label(tab, text="训练分片分布", style="Section.TLabel").grid(
            row=4, column=0, sticky="w", pady=(12, 4)
        )
        shard_frame = ttk.Frame(tab)
        shard_frame.grid(row=5, column=0, sticky="nsew")
        shard_frame.columnconfigure(0, weight=1)
        shard_frame.rowconfigure(0, weight=1)
        shard_columns = ("index", "range", "rows", "source", "digest", "reused", "state")
        self.shard_tree = ttk.Treeview(
            shard_frame, columns=shard_columns, show="headings", selectmode="browse"
        )
        shard_headings = {
            "index": "分片",
            "range": "行区间 [start, end)",
            "rows": "行数",
            "source": "来源阶段@版本",
            "digest": "分片指纹",
            "reused": "复用自规格",
            "state": "状态",
        }
        shard_widths = {
            "index": 60,
            "range": 160,
            "rows": 70,
            "source": 180,
            "digest": 440,
            "reused": 110,
            "state": 100,
        }
        for column in shard_columns:
            self.shard_tree.heading(column, text=shard_headings[column])
            self.shard_tree.column(
                column,
                width=shard_widths[column],
                minwidth=50,
                anchor="w" if column in ("range", "source", "digest") else "center",
            )
        shard_y = ttk.Scrollbar(shard_frame, orient="vertical", command=self.shard_tree.yview)
        self.shard_tree.configure(yscrollcommand=shard_y.set)
        self.shard_tree.grid(row=0, column=0, sticky="nsew")
        shard_y.grid(row=0, column=1, sticky="ns")
        self.shard_tree.tag_configure("current", background="#eef8f4")
        self.shard_tree.tag_configure("stale", background="#fdecea")

    def _build_ledger_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="SQLite 账本")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=2)
        tab.rowconfigure(3, weight=1)
        tab.rowconfigure(5, weight=2)

        ttk.Label(tab, text="运行记录", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        run_frame = ttk.Frame(tab)
        run_frame.grid(row=1, column=0, sticky="nsew")
        run_frame.columnconfigure(0, weight=1)
        run_frame.rowconfigure(0, weight=1)
        run_columns = ("run", "status", "spec", "version", "fingerprint", "final", "started")
        self.run_tree = ttk.Treeview(run_frame, columns=run_columns, show="headings", selectmode="browse")
        run_headings = {
            "run": "运行 ID",
            "status": "状态",
            "spec": "规格版本",
            "version": "批大小",
            "fingerprint": "规格指纹",
            "final": "最终摘要",
            "started": "开始时间",
        }
        for column in run_columns:
            self.run_tree.heading(column, text=run_headings[column])
            self.run_tree.column(
                column,
                width={"run": 170, "status": 120, "spec": 80, "version": 70, "fingerprint": 290, "final": 290, "started": 190}[column],
                anchor="w",
            )
        run_y = ttk.Scrollbar(run_frame, orient="vertical", command=self.run_tree.yview)
        self.run_tree.configure(yscrollcommand=run_y.set)
        self.run_tree.grid(row=0, column=0, sticky="nsew")
        run_y.grid(row=0, column=1, sticky="ns")
        self.run_tree.bind("<<TreeviewSelect>>", self.on_run_selected)

        ttk.Label(tab, text="业务码 / 结构化错误", style="Section.TLabel").grid(
            row=2, column=0, sticky="w", pady=(10, 4)
        )
        error_frame = ttk.Frame(tab)
        error_frame.grid(row=3, column=0, sticky="nsew")
        error_frame.columnconfigure(0, weight=1)
        error_frame.rowconfigure(0, weight=1)
        error_columns = ("code", "stage", "param", "value", "rows", "location", "message")
        self.error_tree = ttk.Treeview(error_frame, columns=error_columns, show="headings")
        error_headings = {
            "code": "业务码",
            "stage": "阶段",
            "param": "参数",
            "value": "参数值",
            "rows": "受影响行",
            "location": "首个位置",
            "message": "说明",
        }
        for column in error_columns:
            self.error_tree.heading(column, text=error_headings[column])
            self.error_tree.column(
                column,
                width={"code": 100, "stage": 140, "param": 160, "value": 80, "rows": 80, "location": 300, "message": 430}[column],
                anchor="w",
            )
        error_y = ttk.Scrollbar(error_frame, orient="vertical", command=self.error_tree.yview)
        self.error_tree.configure(yscrollcommand=error_y.set)
        self.error_tree.grid(row=0, column=0, sticky="nsew")
        error_y.grid(row=0, column=1, sticky="ns")
        self.error_tree.tag_configure("business", background="#fff1ee")

        ttk.Label(tab, text="所选运行的最新批次观测", style="Section.TLabel").grid(
            row=4, column=0, sticky="w", pady=(10, 4)
        )
        batch_frame = ttk.Frame(tab)
        batch_frame.grid(row=5, column=0, sticky="nsew")
        batch_frame.columnconfigure(0, weight=1)
        batch_frame.rowconfigure(0, weight=1)
        batch_columns = ("stage", "attempt", "batch", "range", "rows", "input", "output", "trusted")
        self.batch_tree = ttk.Treeview(batch_frame, columns=batch_columns, show="headings")
        batch_headings = {
            "stage": "阶段",
            "attempt": "尝试",
            "batch": "批次",
            "range": "行区间",
            "rows": "行数",
            "input": "输入摘要",
            "output": "输出摘要",
            "trusted": "可信",
        }
        for column in batch_columns:
            self.batch_tree.heading(column, text=batch_headings[column])
            self.batch_tree.column(
                column,
                width={"stage": 150, "attempt": 65, "batch": 65, "range": 130, "rows": 60, "input": 320, "output": 320, "trusted": 60}[column],
                anchor="w",
            )
        batch_y = ttk.Scrollbar(batch_frame, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=batch_y.set)
        self.batch_tree.grid(row=0, column=0, sticky="nsew")
        batch_y.grid(row=0, column=1, sticky="ns")

    def _build_activity_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="执行 / 恢复报告")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        ttk.Label(
            tab,
            text="最近操作、恢复清单与重放结论（按发生顺序追加）",
            style="Section.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        frame = ttk.Frame(tab)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.activity = tk.Text(frame, wrap="word", font=("Consolas", 9), background="#fbfcfd", foreground="#172b3a")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.activity.yview)
        self.activity.configure(yscrollcommand=scroll.set, state="disabled")
        self.activity.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def append_activity(self, text: str) -> None:
        self.activity.configure(state="normal")
        self.activity.insert("end", text.rstrip() + "\n\n")
        self.activity.see("end")
        self.activity.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _load_prepared(self, show_dialog: bool = True) -> bool:
        try:
            self.prepared = load_workspace(self.data_path, self.pipeline_path)
            self.data_path_var.set(self.data_path)
            if show_dialog:
                self.set_status(
                    f"规格校验通过 · {self.prepared.dataset.row_count} 行 · {len(self.prepared.dataset.columns)} 列"
                )
            return True
        except WorkbenchError as error:
            self.prepared = None
            self.banner_var.set(f"业务码: {error.code}")
            self.set_status(error.message)
            if show_dialog:
                messagebox.showerror(
                    "规格加载失败",
                    f"{error.code}\n{error.message}\n首个位置: {error.location or '-'}",
                    parent=self.root,
                )
            return False

    def reload_prepared(self) -> None:
        if self._load_prepared(show_dialog=True):
            self.refresh_all()

    def select_data(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择样本 JSON",
            initialdir=str(Path(self.data_path).parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.data_path = str(Path(path).resolve())
            self._load_prepared(show_dialog=True)
            self.refresh_all()

    def select_pipeline(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择管线 JSON",
            initialdir=str(Path(self.pipeline_path).parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.pipeline_path = str(Path(path).resolve())
            self._load_prepared(show_dialog=True)
            self.refresh_all()

    def refresh_all(self) -> None:
        self.current_view = self.ledger.current_conclusion()
        self.refresh_header()
        self.refresh_overview()
        self.refresh_runs()
        self.refresh_errors()
        errors = self.ledger.list_errors(limit=1)
        if errors:
            error = errors[0]
            self.banner_var.set(f"业务码: {error['code']} · {error['stage_id'] or '-'}")
        elif self.current_view:
            self.banner_var.set("业务码: -")
        self.set_status("就绪")

    def refresh_header(self) -> None:
        if self.current_view:
            self.spec_version_var.set(str(self.current_view["spec_version"]))
            self.fingerprint_var.set(self.current_view["fingerprint"])
        elif self.prepared is not None:
            self.spec_version_var.set("未分配（尚未写入账本）")
            self.fingerprint_var.set(self.prepared.fingerprint)
        else:
            self.spec_version_var.set("-")
            self.fingerprint_var.set("-")

    def refresh_overview(self) -> None:
        for item in self.stage_tree.get_children():
            self.stage_tree.delete(item)
        for item in self.shard_tree.get_children():
            self.shard_tree.delete(item)
        if self.current_view:
            stages = sorted(self.current_view.get("stages", []), key=lambda item: (item["ordinal"], item["stage_id"]))
            for stage in stages:
                trusted = bool(stage.get("trusted"))
                tag = "trusted" if trusted else "failed"
                self.stage_tree.insert(
                    "",
                    "end",
                    iid=stage["stage_id"],
                    values=(
                        stage["ordinal"] + 1,
                        stage["stage_id"],
                        stage["type"],
                        stage["version"],
                        stage.get("upstream") or "ROOT",
                        "COMMITTED / TRUSTED" if trusted else stage.get("status", "UNKNOWN"),
                        stage.get("row_count", "-"),
                        stage.get("output_digest") or "-",
                    ),
                    tags=(tag,),
                )
            self.dependency_var.set(
                "依赖链: " + " → ".join(stage["stage_id"] for stage in stages)
            )
            shards = sorted(self.current_view.get("shards", []), key=lambda item: item["shard_index"])
            for shard in shards:
                state = "STALE" if shard.get("stale") else "CURRENT"
                tag = "stale" if shard.get("stale") else "current"
                reused = shard.get("reused_from_spec_version")
                self.shard_tree.insert(
                    "",
                    "end",
                    values=(
                        shard["shard_index"],
                        f"[{shard['start_row']}, {shard['end_row']})",
                        shard["row_count"],
                        f"{shard['source_stage_id']}@{shard['source_stage_version']}",
                        shard["shard_fingerprint"],
                        f"v{reused}" if reused is not None else "-",
                        state,
                    ),
                    tags=(tag,),
                )
        elif self.prepared is not None:
            for stage in self.prepared.pipeline.stages:
                self.stage_tree.insert(
                    "",
                    "end",
                    iid=stage.id,
                    values=(
                        stage.ordinal + 1,
                        stage.id,
                        stage.type,
                        stage.version,
                        stage.upstream or "ROOT",
                        "PLANNED",
                        "-",
                        "-",
                    ),
                    tags=("planned",),
                )
            self.dependency_var.set(
                "编译计划: " + " → ".join(stage.id for stage in self.prepared.pipeline.stages)
            )

    def refresh_runs(self) -> None:
        for item in self.run_tree.get_children():
            self.run_tree.delete(item)
        for run in self.ledger.list_runs():
            self.run_tree.insert(
                "",
                "end",
                iid=run["run_id"],
                values=(
                    run["run_id"],
                    run["status"],
                    run["spec_version"],
                    run["batch_size"],
                    run["fingerprint"],
                    run["final_data_digest"] or "-",
                    run["started_at"],
                ),
            )
        if self.current_view and self.run_tree.exists(self.current_view["run_id"]):
            self.run_tree.selection_set(self.current_view["run_id"])
            self.run_tree.focus(self.current_view["run_id"])

    def refresh_errors(self) -> None:
        for item in self.error_tree.get_children():
            self.error_tree.delete(item)
        for error in self.ledger.list_errors(limit=200):
            value = error["param_value_json"] if error["param_value_json"] is not None else "-"
            self.error_tree.insert(
                "",
                "end",
                values=(
                    error["code"],
                    error["stage_id"] or "-",
                    error["param_name"] or "-",
                    value,
                    error["affected_rows"] if error["affected_rows"] is not None else "-",
                    error["location"] or "-",
                    error["message"],
                ),
                tags=("business",),
            )

    def on_run_selected(self, _event: object | None = None) -> None:
        selection = self.run_tree.selection()
        if not selection:
            return
        run_id = selection[0]
        self.selected_run_id = run_id
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)
        for stage in self.ledger.latest_stage_runs(run_id):
            batches = self.ledger.get_batches(run_id, stage["stage_id"], int(stage["attempt"]), trusted_only=False)
            for batch in batches:
                self.batch_tree.insert(
                    "",
                    "end",
                    values=(
                        batch["stage_id"],
                        batch["attempt"],
                        batch["batch_index"],
                        f"[{batch['start_row']}, {batch['end_row']})",
                        batch["row_count"],
                        batch["input_digest"],
                        batch["output_digest"],
                        "YES" if batch["trusted"] else "NO",
                    ),
                )

    def _selected_or_latest(self, statuses: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        selection = self.run_tree.selection()
        if selection:
            run = self.ledger.get_run(selection[0])
            if run is not None and (statuses is None or run["status"] in statuses):
                return run
        for run in self.ledger.list_runs():
            if statuses is None or run["status"] in statuses:
                return run
        return None

    def _run_async(
        self,
        label: str,
        action: Callable[[], tuple[str, str]],
        done: Callable[[str, str], None] | None = None,
    ) -> None:
        if self._busy:
            messagebox.showinfo("忙碌", "当前操作尚未完成。", parent=self.root)
            return
        self._busy = True
        self.run_button.configure(state="disabled")
        self.recover_button.configure(state="disabled")
        self.replay_button.configure(state="disabled")
        self.set_status(label)
        self.append_activity(label)

        def worker() -> None:
            try:
                code, message = action()
            except WorkbenchError as error:
                if not error.details.get("ledger_recorded"):
                    self.ledger.record_error(self.selected_run_id, error)
                code, message = error.code, f"{error.code}: {error.message}\n首个位置: {error.location or '-'}"
            except Exception as error:  # defensive UI boundary
                wrapped = WorkbenchError(E_ENV, f"unexpected UI operation failure: {error}")
                code, message = wrapped.code, str(wrapped)
            self.root.after(0, lambda: self._finish_async(code, message, done))

        threading.Thread(target=worker, daemon=True, name="mlpipe-ui-worker").start()

    def _finish_async(
        self,
        code: str,
        message: str,
        done: Callable[[str, str], None] | None,
    ) -> None:
        self._busy = False
        self.run_button.configure(state="normal")
        self.recover_button.configure(state="normal")
        self.replay_button.configure(state="normal")
        if code == "OK":
            self.banner_var.set("业务码: -")
        else:
            self.banner_var.set(f"业务码: {code}")
        self.append_activity(message)
        self.set_status("完成" if code == "OK" else code)
        if done is not None:
            done(code, message)
        self.refresh_all()

    def run_current(self) -> None:
        data_path = self.data_path
        pipeline_path = self.pipeline_path
        run_id = f"ui-{uuid.uuid4().hex[:12]}"

        def action() -> tuple[str, str]:
            prepared = load_workspace(data_path, pipeline_path)
            result = self.engine.run(
                prepared,
                run_id=run_id,
                batch_size=32,
                command={"command": "ui", "data": data_path, "pipeline": pipeline_path},
            )
            evolution = result.evolution or {}
            detail = [
                f"运行完成: {result.run_id}",
                f"状态: {result.status}",
                f"规格版本: {result.spec_version}",
                f"规格指纹: {result.fingerprint}",
                f"阶段摘要: {len(result.stage_statuses)}",
                f"分片数: {len(result.shards)}",
                f"最终摘要: {result.final_data_digest}",
            ]
            if evolution:
                detail.extend(
                    [
                        f"受影响阶段: {','.join(stage['stage_id'] for stage in evolution.get('affected_stages', []))}",
                        f"作废分片: {evolution.get('invalidated_shard_count')}",
                        f"保留分片: {evolution.get('preserved_shard_count')}",
                        "结论视图已整体替换",
                    ]
                )
            return "OK", "\n".join(detail)

        self._run_async("运行管线并生成/替换结论视图…", action)

    def recover_selected(self) -> None:
        run = self._selected_or_latest(("RUNNING", "INTERRUPTED", "FAILED"))
        if run is None:
            messagebox.showinfo("无可恢复运行", "账本中没有 RUNNING / INTERRUPTED / FAILED 运行。", parent=self.root)
            return
        if not messagebox.askyesno(
            "确认恢复",
            f"将仅依据已提交阶段恢复 {run['run_id']}，并重跑未知阶段。继续？",
            parent=self.root,
        ):
            return
        run_id = run["run_id"]
        self.selected_run_id = run_id

        def action() -> tuple[str, str]:
            report = self.recovery.recover(run_id)
            inspection = report["inspection"]
            replay = report.get("replay") or {}
            detail = [
                f"恢复报告: {run_id}",
                f"强制退出报告可用: {str(inspection.get('exit_report_available')).lower()}",
                "已恢复阶段: " + ",".join(stage["stage_id"] for stage in inspection.get("recovered_stages", [])),
                "未知阶段: " + ",".join(stage["stage_id"] for stage in inspection.get("unknown_stages", [])),
                "重跑计划: " + ",".join(inspection.get("rerun_plan", [])),
                f"重放状态: {replay.get('status')}",
                f"校验阶段/分片: {replay.get('stages_verified')}/{replay.get('shards_verified')}",
                f"恢复后状态: {report.get('status')}",
            ]
            return "OK", "\n".join(detail)

        self._run_async("检查已提交阶段、续跑未知阶段并重放校验…", action)

    def replay_selected(self) -> None:
        run = self._selected_or_latest(("COMPLETE", "RECOVERED_COMPLETE"))
        if run is None:
            messagebox.showinfo("无可重放运行", "账本中没有已完成的运行。", parent=self.root)
            return
        run_id = run["run_id"]

        def action() -> tuple[str, str]:
            report = self.recovery.replay(run_id)
            return "OK", "\n".join(
                [
                    f"重放校验通过: {run_id}",
                    f"规格版本: {report['spec_version']}",
                    f"规格指纹: {report['fingerprint']}",
                    f"阶段数: {report['stages_verified']}",
                    f"分片数: {report['shards_verified']}",
                    f"最终摘要: {report['final_data_digest']}",
                ]
            )

        self._run_async("从账本快照重算并逐项比较…", action)

    def on_close(self) -> None:
        try:
            self.ledger.close()
        finally:
            self.root.destroy()


def main(
    db_path: str = "workbench.db",
    data_path: str = "data/sample.json",
    pipeline_path: str = "data/pipeline.json",
    log_path: str = "logs/workbench.jsonl",
) -> int:
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise WorkbenchError(
            E_ENV,
            f"Tkinter display is unavailable: {error}",
            location="environment.tkinter-display",
        ) from error
    WorkbenchUI(root, db_path, data_path, pipeline_path, log_path)
    root.mainloop()
    return 0
