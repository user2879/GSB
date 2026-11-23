"""Tkinter dashboard.  Imported lazily so headless deployments still work."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from .engine import ReplayEngine
from .errors import BusinessError
from .snapshots import create_snapshot, restore_snapshot
from .storage import Storage
from .validation import load_sample
from .verification import verify_replay


def launch_gui(storage: Storage, sample_path: str = "samples/sample.json") -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText
    except Exception as exc:  # missing libtk/_tkinter on headless hosts
        raise BusinessError(
            "E_GUI",
            "Tkinter is unavailable in this Python runtime",
            {"error": str(exc), "hint": "Use the Windows Python installer with Tcl/Tk enabled."},
        )

    class DiagnosticsGUI:
        def __init__(self, root: Any) -> None:
            self.root = root
            self.storage = storage
            self.sample_path = str(Path(sample_path).resolve())
            self.root.title("WAL Crash Recovery Diagnostics")
            self.root.geometry("1380x860")
            self.root.minsize(1080, 680)
            self.root.configure(background="#f4f6f8")
            self.work_queue: queue.Queue[tuple[str, str, Any]] = queue.Queue()
            self.busy = False
            self._buttons: list[Any] = []
            self._build_style()
            self._build_layout()
            self.root.after(100, self._poll_work)
            self.refresh()

        def _build_style(self) -> None:
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("App.TFrame", background="#f4f6f8")
            style.configure("Header.TFrame", background="#132238")
            style.configure(
                "Header.TLabel",
                background="#132238",
                foreground="#f7fafc",
                font=("Segoe UI", 17, "bold"),
            )
            style.configure(
                "Subheader.TLabel",
                background="#132238",
                foreground="#a9c4e3",
                font=("Segoe UI", 9),
            )
            style.configure("Card.TLabelframe", background="#ffffff", padding=10)
            style.configure("Card.TLabelframe.Label", font=("Segoe UI", 10, "bold"))
            style.configure("TNotebook", background="#f4f6f8", borderwidth=0)
            style.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 9))
            style.configure("Treeview", rowheight=24, font=("Consolas", 9))
            style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
            style.configure("Status.TLabel", background="#e6edf5", padding=6)

        def _make_tree(self, parent: Any, columns: tuple[tuple[str, str, int], ...]) -> Any:
            frame = ttk.Frame(parent)
            frame.pack(fill="both", expand=True)
            tree = ttk.Treeview(frame, columns=[col[0] for col in columns], show="headings")
            for key, title, width in columns:
                tree.heading(key, text=title)
                tree.column(key, width=width, minwidth=50, stretch=True, anchor="w")
            y_scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
            tree.grid(row=0, column=0, sticky="nsew")
            y_scroll.grid(row=0, column=1, sticky="ns")
            x_scroll.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            return tree

        def _build_layout(self) -> None:
            header = ttk.Frame(self.root, style="Header.TFrame", padding=(18, 12))
            header.pack(fill="x")
            ttk.Label(header, text="WAL Crash Recovery Diagnostics", style="Header.TLabel").pack(anchor="w")
            ttk.Label(
                header,
                text="Deterministic page versions · budgeted parallel replay · canonical LSN commit · hash-chained ledger",
                style="Subheader.TLabel",
            ).pack(anchor="w", pady=(2, 0))

            toolbar = ttk.Frame(self.root, style="App.TFrame", padding=(12, 9))
            toolbar.pack(fill="x")
            button_specs = (
                ("刷新", self.refresh),
                ("加载样本…", self.load_sample_dialog),
                ("执行重放", self.replay),
                ("断点恢复", self.replay),
                ("校验重放", self.verify),
                ("写检查点", self.save_checkpoint),
                ("创建快照…", self.create_snapshot_dialog),
                ("恢复快照…", self.restore_snapshot_dialog),
                ("强制终止演练", self.crash_drill),
            )
            for text, command in button_specs:
                button = ttk.Button(toolbar, text=text, command=command)
                button.pack(side="left", padx=(0, 6))
                self._buttons.append(button)

            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
            self._build_overview_tab()
            self._build_pages_tab()
            self._build_wal_tab()
            self._build_schedule_tab()
            self._build_checkpoint_ledger_tab()
            self._build_visibility_tab()
            self._build_snapshots_tab()

            self.status_var = tk.StringVar(value="就绪。刷新只读，不改变账本。")
            ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel").pack(fill="x")

        def _new_tab(self, title: str) -> Any:
            tab = ttk.Frame(self.notebook, style="App.TFrame", padding=10)
            self.notebook.add(tab, text=title)
            return tab

        def _build_overview_tab(self) -> None:
            tab = self._new_tab("总览")
            summary_frame = ttk.LabelFrame(tab, text="会话状态", style="Card.TLabelframe")
            summary_frame.pack(fill="both", expand=True)
            self.summary_text = ScrolledText(
                summary_frame,
                height=24,
                wrap="word",
                font=("Consolas", 9),
                background="#fbfcfe",
                relief="flat",
            )
            self.summary_text.pack(fill="both", expand=True)
            self.summary_text.configure(state="disabled")

        def _build_pages_tab(self) -> None:
            tab = self._new_tab("页列表与版本")
            self.pages_tree = self._make_tree(
                tab,
                (
                    ("page_id", "页标识", 110),
                    ("size", "页大小(B)", 90),
                    ("initial", "初始版本", 90),
                    ("visible_version", "可见版本", 90),
                    ("applied_version", "已应用版本", 100),
                    ("visible_digest", "可见内容摘要", 390),
                    ("state", "可见性状态", 220),
                ),
            )

        def _build_wal_tab(self) -> None:
            tab = self._new_tab("WAL 日志区间")
            controls = ttk.Frame(tab)
            controls.pack(fill="x", pady=(0, 7))
            ttk.Label(controls, text="起始 LSN").pack(side="left")
            self.start_var = tk.StringVar(value="1")
            ttk.Entry(controls, textvariable=self.start_var, width=8).pack(side="left", padx=(5, 14))
            ttk.Label(controls, text="结束 LSN").pack(side="left")
            self.end_var = tk.StringVar(value="")
            ttk.Entry(controls, textvariable=self.end_var, width=8).pack(side="left", padx=(5, 9))
            ttk.Button(controls, text="应用区间", command=self.refresh).pack(side="left")
            ttk.Label(
                controls,
                text="  提交/持久化标记与崩溃可见性按 LSN 展示",
            ).pack(side="left", padx=(12, 0))
            self.wal_tree = self._make_tree(
                tab,
                (
                    ("lsn", "LSN", 65),
                    ("txid", "事务", 100),
                    ("page", "页", 75),
                    ("version", "目标版本", 85),
                    ("digest", "新版摘要", 360),
                    ("commit", "提交", 60),
                    ("persist", "持久化", 70),
                    ("visible", "可见", 60),
                    ("group", "事务组", 65),
                ),
            )

        def _build_schedule_tab(self) -> None:
            tab = self._new_tab("重放调度时间线")
            self.budget_var = tk.StringVar(value="预算：—")
            ttk.Label(tab, textvariable=self.budget_var, font=("Segoe UI", 9, "bold")).pack(
                anchor="w", pady=(0, 7)
            )
            self.schedule_tree = self._make_tree(
                tab,
                (
                    ("run", "运行", 85),
                    ("task", "任务", 335),
                    ("txid", "事务", 90),
                    ("page", "页", 65),
                    ("lsn", "LSN 区间", 95),
                    ("queued", "入队(ms)", 90),
                    ("wait_start", "等待开始", 90),
                    ("wait_end", "等待结束", 90),
                    ("wait_ms", "等待(ms)", 80),
                    ("started", "运行开始", 90),
                    ("finished", "运行结束", 90),
                    ("state", "结局", 80),
                    ("planned", "预算字节", 90),
                    ("consumed", "已消耗", 88),
                    ("batch", "批次", 60),
                ),
            )

        def _build_checkpoint_ledger_tab(self) -> None:
            tab = self._new_tab("检查点 · 账本 · 快照版本")
            checkpoint_box = ttk.LabelFrame(tab, text="检查点集合", style="Card.TLabelframe")
            checkpoint_box.pack(fill="both", expand=True, pady=(0, 7))
            self.checkpoint_tree = self._make_tree(
                checkpoint_box,
                (
                    ("id", "检查点", 70),
                    ("run", "运行/标签", 100),
                    ("cursor", "日志游标", 90),
                    ("pages", "页数", 60),
                    ("digest", "已提交状态摘要", 430),
                    ("created", "创建(单调ms)", 120),
                ),
            )
            batch_box = ttk.LabelFrame(tab, text="原子重放批次", style="Card.TLabelframe")
            batch_box.pack(fill="both", expand=True, pady=(0, 7))
            self.batch_tree = self._make_tree(
                batch_box,
                (
                    ("id", "批次", 60),
                    ("run", "运行", 80),
                    ("range", "LSN 区间", 100),
                    ("pages", "页版本", 70),
                    ("bytes", "预算字节", 90),
                    ("cursor", "账本游标", 90),
                    ("checkpoint", "检查点", 70),
                    ("hash", "账本哈希", 480),
                ),
            )
            ledger_box = ttk.LabelFrame(tab, text="哈希链账本事件", style="Card.TLabelframe")
            ledger_box.pack(fill="both", expand=True)
            self.ledger_tree = self._make_tree(
                ledger_box,
                (
                    ("id", "事件", 60),
                    ("type", "类型", 130),
                    ("cursor", "游标", 70),
                    ("batch", "批次", 60),
                    ("prev", "前序哈希", 300),
                    ("hash", "事件哈希", 300),
                ),
            )

        def _build_visibility_tab(self) -> None:
            tab = self._new_tab("崩溃点与不可见版本")
            self.visibility_var = tk.StringVar(value="崩溃点与本视图将在刷新后更新")
            ttk.Label(tab, textvariable=self.visibility_var, font=("Segoe UI", 9, "bold")).pack(
                anchor="w", pady=(0, 8)
            )
            self.invisible_tree = self._make_tree(
                tab,
                (
                    ("page", "页", 90),
                    ("version", "版本", 70),
                    ("lsn", "LSN", 70),
                    ("txid", "事务", 100),
                    ("digest", "内容摘要", 390),
                    ("reason", "不可见原因", 260),
                ),
            )

        def _build_snapshots_tab(self) -> None:
            tab = self._new_tab("快照")
            top = ttk.Frame(tab)
            top.pack(fill="both", expand=True, pady=(0, 8))
            self.snapshot_tree = self._make_tree(
                top,
                (
                    ("id", "快照", 190),
                    ("format", "版本", 60),
                    ("cursor", "账本游标", 90),
                    ("digest", "内容摘要", 430),
                    ("path", "路径", 340),
                ),
            )
            diff_box = ttk.LabelFrame(tab, text="整体替换差异（新增 / 删除 / 内容变化）", style="Card.TLabelframe")
            diff_box.pack(fill="both", expand=False)
            self.snapshot_diff = ScrolledText(
                diff_box,
                height=9,
                wrap="word",
                font=("Consolas", 9),
                background="#fbfcfe",
                relief="flat",
            )
            self.snapshot_diff.pack(fill="both", expand=True)
            self.snapshot_diff.configure(state="disabled")

        def _clear_tree(self, tree: Any) -> None:
            children = tree.get_children()
            if children:
                tree.delete(*children)

        def _set_text(self, widget: Any, text: str) -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")

        def refresh(self) -> None:
            try:
                status = self.storage.status()
            except Exception as exc:
                self.status_var.set(f"刷新失败：{exc}")
                return
            if not status.get("initialized"):
                for tree in (
                    self.pages_tree,
                    self.wal_tree,
                    self.schedule_tree,
                    self.checkpoint_tree,
                    self.batch_tree,
                    self.ledger_tree,
                    self.invisible_tree,
                    self.snapshot_tree,
                ):
                    self._clear_tree(tree)
                summary = (
                    "工作区尚未初始化。\n\n"
                    f"工作区：{status.get('workspace')}\n\n"
                    "点击“加载样本…”选择 JSON，或执行：\n"
                    "python -m walcrash --action load --input samples/sample.json"
                )
                self._set_text(self.summary_text, summary)
                self.visibility_var.set("崩溃点：未初始化")
                self.status_var.set("未初始化；刷新未写入任何账本记录。")
                return

            pages = status["pages"]
            counts = status["counts"]
            last_run = status.get("last_run") or {}
            last_checkpoint = status.get("last_checkpoint") or {}
            summary = [
                f"工作区                : {status['workspace']}",
                f"账本游标              : LSN {status['cursor']} / WAL 末尾 {status['wal_last_lsn']}",
                f"崩溃点                : LSN {status['crash_point_lsn']}",
                f"可见页列表            : {len(pages)} 页",
                f"WAL 条目 / 页版本     : {counts['wal_entries']} / {counts['page_versions']}",
                f"不可见页版本           : {counts['invisible_page_versions']}",
                f"检查点 / 原子批次      : {counts['checkpoints']} / {counts['batches']}",
                f"未知任务 / 活动任务    : {counts['unknown_tasks']} / {counts['active_tasks']}",
                f"页表版本 / 预算参数版本 : {status.get('page_table_version')} / {status.get('budget_params_version')}",
                f"账本版本 / 格式版本    : {status.get('ledger_version')} / {status.get('schema_format_version')}",
                f"最近运行              : {last_run.get('run_id', '—')} · {last_run.get('status', '—')}",
                f"最近检查点            : {last_checkpoint.get('checkpoint_id', '—')} @ LSN {last_checkpoint.get('ledger_cursor', '—')}",
                f"账本 HEAD              : {status.get('ledger_head') or '—'}",
                f"日志摘要              : {status.get('active_log_digest') or '—'}",
                "",
                "语义说明",
                "  1. 并行阶段只生成确定性页版本候选，实际页版本与账本由单线程 LSN 定序器提交。",
                "  2. 崩溃点 = 最后一个“已提交且组内带持久化标记”的完整事务组末端。",
                "  3. 崩溃点之后的页版本保留为诊断记录，但 visible=0，不参与恢复后页视图。",
                "  4. 强制终止后仅完整提交批次可信；活动/排队任务标为 unknown 并要求重跑。",
            ]
            self._set_text(self.summary_text, "\n".join(summary))

            try:
                budget = self.storage.budget()
                total_bytes = sum(row["bytes_consumed"] for row in self.storage.batches())
                self.budget_var.set(
                    f"预算：全局槽位={budget.global_slots}  ·  每事务配额={budget.per_tx_quota}  ·  "
                    f"单页字节预算={budget.page_byte_budget}  ·  检查点间隔={budget.checkpoint_interval}  ·  "
                    f"账本累计重放字节={total_bytes}"
                )
            except BusinessError:
                self.budget_var.set("预算：不可用")

            self._clear_tree(self.pages_tree)
            for page in pages:
                visible_version = page.get("visible_version")
                applied_version = page.get("applied_version")
                state = "可见" if visible_version is not None else "无可见版本"
                if applied_version is not None and visible_version is not None and applied_version > visible_version:
                    state = "存在不可见新版本"
                self.pages_tree.insert(
                    "",
                    "end",
                    values=(
                        page["page_id"],
                        page["size_bytes"],
                        page["initial_version"],
                        visible_version if visible_version is not None else "—",
                        applied_version if applied_version is not None else "—",
                        page.get("visible_digest") or "—",
                        state,
                    ),
                )

            try:
                start = int(self.start_var.get() or 1)
            except ValueError:
                start = 1
            try:
                end = int(self.end_var.get()) if self.end_var.get().strip() else status["wal_last_lsn"]
            except ValueError:
                end = status["wal_last_lsn"]
            self._clear_tree(self.wal_tree)
            for entry in self.storage.log_entries(start, end):
                self.wal_tree.insert(
                    "",
                    "end",
                    values=(
                        entry["lsn"],
                        entry["txid"],
                        entry["page_id"],
                        entry["target_version"],
                        entry["content_digest"],
                        "Y" if entry["commit"] else "N",
                        "Y" if entry["persist"] else "N",
                        "Y" if entry["visible"] else "N",
                        entry["tx_group"],
                    ),
                )

            self._clear_tree(self.schedule_tree)
            for task in self.storage.tasks():
                wait_start = task.get("wait_started_ms")
                wait_end = task.get("wait_ended_ms")
                wait_ms = (
                    wait_end - wait_start
                    if wait_start is not None and wait_end is not None
                    else None
                )
                self.schedule_tree.insert(
                    "",
                    "end",
                    values=(
                        task["run_id"],
                        task["task_id"],
                        task["txid"],
                        task["page_id"],
                        f"{task['lsn_start']}-{task['lsn_end']}",
                        task["queued_ms"],
                        wait_start if wait_start is not None else "—",
                        wait_end if wait_end is not None else "—",
                        wait_ms if wait_ms is not None else "—",
                        task.get("started_ms") or "—",
                        task.get("finished_ms") or "—",
                        task["state"],
                        task["planned_bytes"],
                        task["consumed_bytes"],
                        task.get("batch_id") or "—",
                    ),
                )

            self._clear_tree(self.checkpoint_tree)
            for checkpoint in self.storage.checkpoint_rows():
                self.checkpoint_tree.insert(
                    "",
                    "end",
                    values=(
                        checkpoint["checkpoint_id"],
                        checkpoint["run_id"],
                        checkpoint["ledger_cursor"],
                        checkpoint["page_count_since_checkpoint"],
                        checkpoint["state_digest"],
                        checkpoint["created_ms"],
                    ),
                )
            self._clear_tree(self.batch_tree)
            for batch in self.storage.batches():
                self.batch_tree.insert(
                    "",
                    "end",
                    values=(
                        batch["batch_id"],
                        batch["run_id"],
                        f"{batch['start_lsn']}-{batch['end_lsn']}",
                        batch["page_count"],
                        batch["bytes_consumed"],
                        batch["ledger_cursor"],
                        batch.get("checkpoint_id") or "—",
                        batch["ledger_hash"],
                    ),
                )
            self._clear_tree(self.ledger_tree)
            for event in self.storage.ledger_rows():
                self.ledger_tree.insert(
                    "",
                    "end",
                    values=(
                        event["event_id"],
                        event["event_type"],
                        event["cursor"],
                        event.get("batch_id") or "—",
                        event["prev_hash"],
                        event["event_hash"],
                    ),
                )

            self._clear_tree(self.invisible_tree)
            invisible = [row for row in self.storage.page_versions() if not row["visible"] and row["kind"] == "wal"]
            for row in invisible:
                self.invisible_tree.insert(
                    "",
                    "end",
                    values=(
                        row["page_id"],
                        row["version"],
                        row["lsn"],
                        row["txid"],
                        row["digest"],
                        "LSN 在崩溃点之后或事务未持久化提交",
                    ),
                )
            self.visibility_var.set(
                f"崩溃点：LSN {status['crash_point_lsn']}  ·  不可见页版本：{len(invisible)}  ·  "
                "不可见版本仅用于诊断，不进入恢复后页视图"
            )

            self._clear_tree(self.snapshot_tree)
            for snapshot in status.get("snapshots", []):
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        snapshot["snapshot_id"],
                        snapshot["format_version"],
                        snapshot["ledger_cursor"],
                        snapshot["content_digest"],
                        snapshot["file_path"],
                    ),
                )
            self.status_var.set("刷新完成；只读渲染，账本未改变。")

        def _set_busy(self, busy: bool, text: str = "") -> None:
            self.busy = busy
            state = "disabled" if busy else "normal"
            for button in self._buttons:
                button.configure(state=state)
            self.root.configure(cursor="watch" if busy else "")
            if text:
                self.status_var.set(text)

        def _run_async(self, label: str, work: Callable[[], Any]) -> None:
            if self.busy:
                return
            self._set_busy(True, f"{label}…")

            def runner() -> None:
                try:
                    result = work()
                    self.work_queue.put(("ok", label, result))
                except BusinessError as exc:
                    self.work_queue.put(("error", label, exc))
                except Exception as exc:
                    self.work_queue.put(("exception", label, exc))

            threading.Thread(target=runner, daemon=True).start()

        def _poll_work(self) -> None:
            try:
                while True:
                    kind, label, payload = self.work_queue.get_nowait()
                    self._set_busy(False)
                    if kind == "ok":
                        self.status_var.set(f"{label}完成。")
                        if isinstance(payload, dict) and not payload.get("ok", True):
                            self.status_var.set(f"{label}异常：{payload}")
                        if label == "恢复快照" and isinstance(payload, dict):
                            self._show_snapshot_diff(payload)
                        self.refresh()
                    elif kind == "error":
                        self.status_var.set(f"{label}失败：{payload.code} · {payload.message}")
                        self.storage.record_error(payload, ui_action=label)
                        self.refresh()
                        messagebox.showerror(
                            f"{label} · {payload.code}",
                            f"{payload.message}\n\n{json.dumps(dict(payload.context), ensure_ascii=False, indent=2)}",
                        )
                    else:
                        error = BusinessError("E_STATE", f"{label}失败：{payload}", {"error": str(payload)})
                        self.storage.record_error(error, ui_action=label)
                        self.status_var.set(error.message)
                        messagebox.showerror(label, error.message)
            except queue.Empty:
                pass
            self.root.after(120, self._poll_work)

        def load_sample_dialog(self) -> None:
            selected = filedialog.askopenfilename(
                title="选择页与 WAL 样本 JSON",
                initialdir=str(Path(self.sample_path).parent),
                filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            )
            if not selected:
                return
            self.sample_path = selected

            def work() -> dict[str, Any]:
                sample = load_sample(selected)
                self.storage.load_fresh(sample)
                return {
                    "ok": True,
                    "pages": len(sample.pages),
                    "entries": len(sample.entries),
                    "crash_point_lsn": sample.crash_point_lsn,
                }

            self._run_async("加载样本", work)

        def replay(self) -> None:
            self._run_async("执行重放", lambda: ReplayEngine(self.storage).replay(delay_ms=12))

        def verify(self) -> None:
            self._run_async("校验重放", lambda: verify_replay(self.storage))

        def save_checkpoint(self) -> None:
            self._run_async("写检查点", lambda: self.storage.save_checkpoint("gui-manual"))

        def create_snapshot_dialog(self) -> None:
            selected = filedialog.asksaveasfilename(
                title="创建会话快照",
                initialdir=str(self.storage.snapshots_dir),
                initialfile="session.json",
                defaultextension=".json",
                filetypes=(("JSON", "*.json"),),
            )
            if not selected:
                return
            self._run_async("创建快照", lambda: create_snapshot(self.storage, selected))

        def restore_snapshot_dialog(self) -> None:
            selected = filedialog.askopenfilename(
                title="选择会话快照",
                initialdir=str(self.storage.snapshots_dir),
                filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            )
            if not selected:
                return

            def work() -> dict[str, Any]:
                return restore_snapshot(self.storage, selected)

            self._run_async("恢复快照", work)

        def _show_snapshot_diff(self, result: dict[str, Any]) -> None:
            self._set_text(
                self.snapshot_diff,
                json.dumps(
                    {
                        "restore_mode": result.get("restore_mode"),
                        "log_changed": result.get("log_changed"),
                        "page_diff": result.get("page_diff"),
                        "restored": result.get("restored"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )

        def crash_drill(self) -> None:
            if not messagebox.askyesno(
                "强制终止演练",
                "将重新加载当前样本，并由子进程在首批提交后执行 os._exit(91)。\n"
                "随后不会生成退出报告；请点击“断点恢复”验证 unknown 任务重跑。\n\n继续？",
            ):
                return
            sample = self.sample_path
            workspace = str(self.storage.workspace)
            project_root = str(Path(__file__).resolve().parent.parent)

            def work() -> dict[str, Any]:
                env = dict(os.environ)
                env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
                base = [sys.executable, "-m", "walcrash", "--workspace", workspace]
                load_result = subprocess.run(
                    base + ["--action", "load", "--input", sample],
                    cwd=project_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if load_result.returncode != 0:
                    raise BusinessError(
                        "E_STATE",
                        "crash drill load failed",
                        {"stdout": load_result.stdout, "stderr": load_result.stderr},
                    )
                crash_result = subprocess.run(
                    base
                    + [
                        "--action",
                        "run",
                        "--delay-ms",
                        "90",
                        "--crash-after-batches",
                        "1",
                    ],
                    cwd=project_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                return {
                    "ok": crash_result.returncode == 91,
                    "crash_rc": crash_result.returncode,
                    "note": "hard termination; no exit report is claimed",
                    "stderr": crash_result.stderr.strip(),
                }

            self._run_async("强制终止演练", work)

    root = None
    try:
        root = tk.Tk()
        DiagnosticsGUI(root)
        root.mainloop()
    except BusinessError:
        raise
    except Exception as exc:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        raise BusinessError(
            "E_GUI",
            "Tkinter display or widget initialization failed",
            {"error": str(exc), "workspace": str(storage.workspace)},
        )
