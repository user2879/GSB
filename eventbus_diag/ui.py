"""Tkinter diagnostic workbench.

The Tk modules are imported only by :func:`run_ui`, so the rest of the package
remains usable on machines where Tk is not installed.
"""

from __future__ import annotations

import queue
import threading
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as _TK_IMPORT_ERROR:  # pragma: no cover - platform dependent
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    TK_IMPORT_ERROR: Exception | None = _TK_IMPORT_ERROR
else:
    TK_IMPORT_ERROR = None

from .batching import build_plan
from .engine import DiagnosticEngine, RunReport
from .errors import BusDiagnosticError
from .spec import Dataset, load_dataset


class Workbench:
    def __init__(
        self,
        root: tk.Tk,
        *,
        spec_path: str,
        events_path: str,
        db_path: str,
        log_path: str,
    ) -> None:
        self.root = root
        self.root.title("事件总线流式集成诊断台")
        self.root.geometry("1280x820")
        self.root.minsize(980, 640)

        self.dataset: Dataset | None = None
        self.engine: DiagnosticEngine | None = None
        self.running = False
        self._result_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._last_report: RunReport | None = None
        self._last_snapshot: dict[str, Any] = {}

        self.spec_var = tk.StringVar(value=str(spec_path))
        self.events_var = tk.StringVar(value=str(events_path))
        self.db_var = tk.StringVar(value=str(db_path))
        self.log_var = tk.StringVar(value=str(log_path))
        self.status_var = tk.StringVar(value="尚未装载")
        self.summary_vars = {
            "version": tk.StringVar(value="-"),
            "run": tk.StringVar(value="-"),
            "status": tk.StringVar(value="-"),
            "plan": tk.StringVar(value="-"),
            "ledger": tk.StringVar(value="-"),
            "stats": tk.StringVar(value="-"),
            "replacement": tk.StringVar(value="-"),
        }

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_result_queue)
        self.root.after(500, self._periodic_refresh)
        self.load_declaration(show_dialog=False)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        paths = ttk.LabelFrame(outer, text="本地输入与账本", padding=8)
        paths.pack(fill=tk.X)
        self._path_row(paths, 0, "声明 JSON", self.spec_var, self._choose_spec)
        self._path_row(paths, 1, "事件 NDJSON", self.events_var, self._choose_events)
        self._path_row(paths, 2, "SQLite 账本", self.db_var, self._choose_db)
        self._path_row(paths, 3, "结构化日志", self.log_var, None)

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(8, 8))
        self.load_button = ttk.Button(
            controls, text="校验并装载", command=lambda: self.load_declaration(show_dialog=True)
        )
        self.load_button.pack(side=tk.LEFT, padx=(0, 6))
        self.run_button = ttk.Button(controls, text="启动 / 恢复", command=self.start_run)
        self.run_button.pack(side=tk.LEFT, padx=6)
        self.replace_button = ttk.Button(
            controls, text="整体替换声明版本", command=self.replace_declaration
        )
        self.replace_button.pack(side=tk.LEFT, padx=6)
        self.verify_button = ttk.Button(controls, text="重放校验当前账本", command=self.verify_replay)
        self.verify_button.pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="立即刷新", command=self.refresh_view).pack(side=tk.LEFT, padx=6)
        ttk.Label(controls, textvariable=self.status_var).pack(side=tk.RIGHT)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill=tk.BOTH, expand=True)
        self._build_overview(notebook)
        self._build_subscribers(notebook)
        self._build_channels(notebook)
        self._build_batches(notebook)
        self._build_leases(notebook)
        self._build_errors(notebook)

    def _path_row(self, parent, row: int, label: str, variable: tk.StringVar, chooser) -> None:
        ttk.Label(parent, text=label, width=12).grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky=tk.EW, padx=(4, 4), pady=2
        )
        if chooser is not None:
            ttk.Button(parent, text="浏览…", command=chooser, width=8).grid(
                row=row, column=2, sticky=tk.E, pady=2
            )
        parent.columnconfigure(1, weight=1)

    def _build_overview(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text="总览")
        labels = [
            ("声明版本", "version"),
            ("运行标识", "run"),
            ("当前状态", "status"),
            ("计划摘要", "plan"),
            ("账本摘要", "ledger"),
            ("量化统计", "stats"),
            ("替换差异", "replacement"),
        ]
        for index, (label, key) in enumerate(labels):
            ttk.Label(frame, text=label, width=12).grid(
                row=index, column=0, sticky=tk.NW, pady=4
            )
            ttk.Label(frame, textvariable=self.summary_vars[key], wraplength=1040).grid(
                row=index, column=1, sticky=tk.EW, pady=4
            )
        frame.columnconfigure(1, weight=1)
        ttk.Separator(frame).grid(row=len(labels), column=0, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(
            frame,
            text=(
                "约束：全局活跃租约、每订阅者活跃租约与通道 1 秒滑动派发窗口同时生效。"
                "\n恢复：只信任已提交 SQLite 记录；在途或未提交批标记 UNKNOWN 后重跑。"
                "\n替换：新声明版本会整体废弃旧版本的租约、账本、摘要和统计，不做增量合并。"
            ),
            justify=tk.LEFT,
        ).grid(row=len(labels) + 1, column=0, columnspan=2, sticky=tk.W)

    def _make_tree(self, parent, columns: tuple[tuple[str, str, int], ...]) -> ttk.Treeview:
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)
        tree = ttk.Treeview(
            container, columns=[column[0] for column in columns], show="headings"
        )
        for key, heading, width in columns:
            tree.heading(key, text=heading)
            tree.column(key, width=width, anchor=tk.W, stretch=True)
        vertical = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
        horizontal = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        vertical.grid(row=0, column=1, sticky=tk.NS)
        horizontal.grid(row=1, column=0, sticky=tk.EW)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        return tree

    def _build_subscribers(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="订阅者")
        self.subscriber_tree = self._make_tree(
            frame,
            (
                ("subscriber_id", "订阅者", 120),
                ("port", "回环端口", 80),
                ("budget", "预算 ms", 75),
                ("lease_limit", "租约上限", 75),
                ("seed", "种子", 70),
                ("topics", "主题", 230),
                ("active", "活跃租约", 80),
                ("health", "进程状态", 120),
            ),
        )

    def _build_channels(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="发布通道")
        self.channel_tree = self._make_tree(
            frame,
            (
                ("channel_id", "通道", 130),
                ("topic", "主题", 120),
                ("batch_size", "批量", 65),
                ("rate", "速率/秒", 75),
                ("quota", "订阅者配额", 95),
                ("recipients", "稳定订阅者集合", 300),
            ),
        )

    def _build_batches(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="批次账本")
        self.batch_tree = self._make_tree(
            frame,
            (
                ("ordinal", "序", 50),
                ("batch_id", "批标识", 180),
                ("subscriber", "订阅者", 95),
                ("channel", "通道", 100),
                ("events", "事件序号集合", 160),
                ("state", "三态", 90),
                ("wait", "等待 ms", 75),
                ("processing", "处理 ms", 70),
                ("lease", "租约版本", 70),
                ("ack", "确认摘要", 170),
                ("cursor", "游标", 55),
            ),
        )

    def _build_leases(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="租约")
        self.lease_tree = self._make_tree(
            frame,
            (
                ("lease_id", "租约标识", 190),
                ("batch_id", "批标识", 185),
                ("subscriber", "订阅者", 95),
                ("version", "版本", 55),
                ("granted", "授予单调 ms", 115),
                ("expires", "到期单调 ms", 115),
                ("revoked", "回收单调 ms", 115),
                ("state", "状态", 85),
            ),
        )

    def _build_errors(self, notebook: ttk.Notebook) -> None:
        frame = ttk.Frame(notebook, padding=8)
        notebook.add(frame, text="错误 / 重放")
        self.error_tree = self._make_tree(
            frame,
            (
                ("time", "时间 ms", 105),
                ("code", "错误码", 80),
                ("field", "首个字段", 140),
                ("line", "行", 50),
                ("batch", "批标识", 180),
                ("message", "说明", 650),
            ),
        )

    def _choose_spec(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.spec_var.set(path)

    def _choose_events(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("NDJSON", "*.ndjson"), ("All", "*.*")])
        if path:
            self.events_var.set(path)

    def _choose_db(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".sqlite")
        if path:
            self.db_var.set(path)

    def load_declaration(self, *, show_dialog: bool) -> None:
        if self.running:
            self.status_var.set("运行中不能更换输入")
            return
        try:
            dataset = load_dataset(self.spec_var.get(), self.events_var.get())
            build_plan(dataset.declaration, dataset.events)
        except BusDiagnosticError as exc:
            self.status_var.set(f"{exc.code} {exc.field or ''} 行 {exc.line or '-'}：{exc.message}")
            if show_dialog:
                messagebox.showerror("E_SPEC", str(exc), parent=self.root)
            return
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        self.dataset = dataset
        self.engine = DiagnosticEngine(
            dataset,
            db_path=self.db_var.get(),
            log_path=self.log_var.get(),
        )
        self.status_var.set(
            f"已装载 v{dataset.declaration.declaration_version}，"
            f"{len(dataset.events)} 个事件"
        )
        self._last_report = None
        self.refresh_view()

    def start_run(self) -> None:
        if self.dataset is None or self.engine is None:
            self.load_declaration(show_dialog=True)
            if self.engine is None:
                return
        if self.running:
            return
        self.running = True
        self.status_var.set("派发中…")
        self._set_buttons(False)
        engine = self.engine

        def target() -> None:
            try:
                report = engine.run(resume=True, verify=True)
                self._result_queue.put(("report", report))
            except Exception as exc:
                self._result_queue.put(("error", exc))

        threading.Thread(target=target, name="diagnostic-run", daemon=True).start()

    def replace_declaration(self) -> None:
        if self.running:
            self.status_var.set("运行中不能替换声明")
            return
        try:
            dataset = load_dataset(self.spec_var.get(), self.events_var.get())
        except BusDiagnosticError as exc:
            messagebox.showerror("E_SPEC", str(exc), parent=self.root)
            return
        if self.engine is None:
            self.dataset = dataset
            self.engine = DiagnosticEngine(
                dataset, db_path=self.db_var.get(), log_path=self.log_var.get()
            )
            self.status_var.set("已新建诊断引擎")
            return
        try:
            diff = self.engine.replace_declaration(dataset)
        except BusDiagnosticError as exc:
            messagebox.showerror(exc.code, str(exc), parent=self.root)
            return
        self.dataset = dataset
        self.summary_vars["replacement"].set(
            "新增="
            + ",".join(diff.added_subscribers)
            + " 移除="
            + ",".join(diff.removed_subscribers)
            + " 升级="
            + ",".join(item["subscriber_id"] for item in diff.upgraded_subscribers)
        )
        self.status_var.set(f"声明已整体替换到 v{diff.new_version}")
        self.refresh_view()

    def verify_replay(self) -> None:
        if self.engine is None or self.running:
            return
        try:
            result = self.engine.verify_replay()
        except BusDiagnosticError as exc:
            messagebox.showerror(exc.code, str(exc), parent=self.root)
            self.status_var.set(f"{exc.code}: {exc.message}")
            return
        self.status_var.set(
            f"重放校验通过：{len(result.verified_batches)} 批，"
            f"待完成 {len(result.remaining_batches)} 批"
        )
        self.refresh_view()

    def refresh_view(self) -> None:
        if self.dataset is None:
            return
        snapshot: dict[str, Any] = {}
        if self.engine is not None:
            try:
                snapshot = self.engine.snapshot()
            except Exception as exc:
                self.status_var.set(f"读取账本失败：{exc}")
                return
        self._last_snapshot = snapshot
        run = snapshot.get("run") or {}
        stats = snapshot.get("stats") or {}
        info = snapshot.get("ledger_info") or {}
        plan = build_plan(self.dataset.declaration, self.dataset.events)
        self.summary_vars["version"].set(str(self.dataset.declaration.declaration_version))
        self.summary_vars["run"].set(str(run.get("run_id", "尚无已提交运行")))
        self.summary_vars["status"].set(str(run.get("status", "未运行")))
        self.summary_vars["plan"].set(plan.digest())
        self.summary_vars["ledger"].set(
            f"format={info.get('format_version', '-')} digest={info.get('content_digest', '-')}"
        )
        self.summary_vars["stats"].set(
            f"批次={stats.get('batch_count', 0)} 已确认={stats.get('confirmed_batches', 0)} "
            f"失败={stats.get('failed_batches', 0)} 未知={stats.get('unknown_batches', 0)} "
            f"在途={stats.get('in_flight_batches', 0)} 活跃租约={stats.get('active_leases', 0)}"
        )
        replacement = run.get("replacement")
        if replacement:
            self.summary_vars["replacement"].set(
                "新增="
                + ",".join(replacement.get("added_subscribers", []))
                + " 移除="
                + ",".join(replacement.get("removed_subscribers", []))
                + " 升级="
                + ",".join(
                    item.get("subscriber_id", "")
                    for item in replacement.get("upgraded_subscribers", [])
                )
            )
        self._replace_tree_rows(self.subscriber_tree, self._subscriber_rows(snapshot))
        self._replace_tree_rows(self.channel_tree, self._channel_rows())
        self._replace_tree_rows(self.batch_tree, self._batch_rows(snapshot))
        self._replace_tree_rows(self.lease_tree, self._lease_rows(snapshot))
        self._replace_tree_rows(self.error_tree, self._error_rows(snapshot))

    def _subscriber_rows(self, snapshot: dict[str, Any]) -> list[tuple[str, ...]]:
        live = snapshot.get("live_leases") or {}
        active = live.get("subscriber_active", {}) if isinstance(live, dict) else {}
        startup_errors = {
            str(error.get("payload", {}).get("subscriber_id"))
            for error in snapshot.get("errors", [])
            if error.get("code") in {"E_PORT", "E_TIMEOUT", "E_PROCESS"}
            and error.get("payload")
        }
        rows = []
        for subscriber in sorted(self.dataset.declaration.subscribers, key=lambda item: item.subscriber_id):
            health = "启动异常" if subscriber.subscriber_id in startup_errors else (
                "运行/待命" if self.engine is not None else "未启动"
            )
            rows.append(
                (
                    subscriber.subscriber_id,
                    str(subscriber.port),
                    str(subscriber.processing_budget_ms),
                    str(subscriber.lease_limit),
                    str(subscriber.seed),
                    ",".join(subscriber.topics) or "*",
                    str(active.get(subscriber.subscriber_id, 0)),
                    health,
                )
            )
        return rows

    def _channel_rows(self) -> list[tuple[str, ...]]:
        plan = build_plan(self.dataset.declaration, self.dataset.events)
        rows = []
        for channel in sorted(self.dataset.declaration.channels, key=lambda item: item.channel_id):
            rows.append(
                (
                    channel.channel_id,
                    channel.topic,
                    str(channel.batch_size),
                    str(channel.send_rate_limit),
                    str(channel.subscriber_quota),
                    ",".join(plan.subscribers_by_channel.get(channel.channel_id, ())),
                )
            )
        return rows

    def _batch_rows(self, snapshot: dict[str, Any]) -> list[tuple[str, ...]]:
        rows = []
        for batch in snapshot.get("batches", []):
            ack = batch.get("ack_summary") or {}
            rows.append(
                (
                    str(batch.get("ordinal", "")),
                    str(batch.get("batch_id", "")),
                    str(batch.get("subscriber_id", "")),
                    str(batch.get("channel_id", "")),
                    ",".join(str(value) for value in batch.get("event_ids", [])),
                    str(batch.get("state", "")),
                    f"{float(batch.get('wait_ms') or 0.0):.3f}",
                    str(batch.get("processing_ms") if batch.get("processing_ms") is not None else "-"),
                    str(batch.get("lease_version") if batch.get("lease_version") is not None else "-"),
                    str(ack.get("ack_digest", "-")),
                    str(batch.get("progress_cursor", 0)),
                )
            )
        return rows

    def _lease_rows(self, snapshot: dict[str, Any]) -> list[tuple[str, ...]]:
        rows = []
        for lease in snapshot.get("leases", []):
            rows.append(
                (
                    str(lease.get("lease_id", "")),
                    str(lease.get("batch_id", "")),
                    str(lease.get("subscriber_id", "")),
                    str(lease.get("lease_version", "")),
                    f"{float(lease.get('granted_mono_ms') or 0.0):.3f}",
                    f"{float(lease.get('expires_mono_ms') or 0.0):.3f}",
                    f"{float(lease.get('revoked_mono_ms') or 0.0):.3f}"
                    if lease.get("revoked_mono_ms") is not None
                    else "-",
                    str(lease.get("state", "")),
                )
            )
        return rows

    def _error_rows(self, snapshot: dict[str, Any]) -> list[tuple[str, ...]]:
        rows = []
        for error in snapshot.get("errors", []):
            rows.append(
                (
                    str(error.get("at_wall_ms", "")),
                    str(error.get("code", "")),
                    str(error.get("field") or "-"),
                    str(error.get("source_line") if error.get("source_line") is not None else "-"),
                    str(error.get("batch_id") or "-"),
                    str(error.get("message", "")),
                )
            )
        return rows

    @staticmethod
    def _replace_tree_rows(tree: ttk.Treeview, rows: list[tuple[str, ...]]) -> None:
        # Delete all rows first: the current declaration view is replaced, never merged.
        children = tree.get_children()
        if children:
            tree.delete(*children)
        for row in rows:
            tree.insert("", tk.END, values=row)

    def _periodic_refresh(self) -> None:
        if not self.running:
            self.refresh_view()
        self.root.after(500, self._periodic_refresh)

    def _drain_result_queue(self) -> None:
        try:
            while True:
                kind, payload = self._result_queue.get_nowait()
                self.running = False
                self._set_buttons(True)
                if kind == "report":
                    self._last_report = payload
                    if payload.replay_mismatch:
                        self.status_var.set(
                            f"E_REPLAY: {payload.replay_mismatch.get('message', '重放不一致')}"
                        )
                    else:
                        self.status_var.set(
                            f"运行 {payload.status}：确认 {payload.confirmed_batches}/"
                            f"{payload.batch_count}，未知 {payload.unknown_batches}"
                        )
                    self.refresh_view()
                else:
                    code = getattr(payload, "code", "E_INTERNAL")
                    self.status_var.set(f"{code}: {payload}")
                    messagebox.showerror(code, str(payload), parent=self.root)
                    self.refresh_view()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_result_queue)

    def _set_buttons(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (
            self.load_button,
            self.run_button,
            self.replace_button,
            self.verify_button,
        ):
            button.configure(state=state)

    def _on_close(self) -> None:
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass
        self.root.destroy()


def run_ui(
    *,
    spec_path: str,
    events_path: str,
    db_path: str,
    log_path: str,
) -> int:
    if tk is None or ttk is None:
        print(
            "Tkinter 无法导入。请安装带 Tk 支持的 Python（Windows 官方安装包默认包含），"
            f"并确认图形运行库可用：{TK_IMPORT_ERROR}",
            file=__import__("sys").stderr,
        )
        return 2
    try:
        root = tk.Tk()
    except Exception as exc:
        print(
            "Tkinter 无法启动。请安装带 Tk 支持的 Python（Windows 官方安装包默认包含），"
            f"并确认图形环境可用：{exc}",
            file=__import__("sys").stderr,
        )
        return 2
    style = ttk.Style(root)
    try:
        style.theme_use("vista" if "vista" in style.theme_names() else "clam")
    except tk.TclError:
        pass
    Workbench(
        root,
        spec_path=spec_path,
        events_path=events_path,
        db_path=db_path,
        log_path=log_path,
    )
    root.mainloop()
    return 0
