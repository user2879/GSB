from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from .config import InferenceConfig, load_config
from .engine import DiagnosticSession, RunRequest
from .errors import DiagnosticError


BG = "#0d1720"
PANEL = "#14232e"
PANEL_ALT = "#192c39"
BORDER = "#294455"
TEXT = "#dcecf3"
MUTED = "#89a6b5"
ACCENT = "#4fd1c5"
ACCENT_DARK = "#153e3d"
WARN = "#f2b35d"
FAIL = "#ff6b78"
PASS = "#82d98b"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "-"
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MiB"
    if value >= 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value} B"


def _short_hash(value: str | None) -> str:
    return value[:12] if value else "-"


def _status_color(status: str) -> str:
    if status == "PASS":
        return PASS
    if status == "FAIL":
        return FAIL
    return WARN


class DiagnosticApp(tk.Tk):
    def __init__(
        self,
        *,
        request: RunRequest,
        initial_config: str,
        report_dir: str,
        snapshot_dir: str,
        auto_run: bool = False,
    ) -> None:
        super().__init__()
        self.title("混合精度推理显存管理诊断台")
        self.geometry("1480x940")
        self.minsize(1160, 760)
        self.configure(bg=BG)

        self.request = request
        self.report_dir = report_dir
        self.snapshot_dir = snapshot_dir
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.session: DiagnosticSession | None = None
        self.view: dict[str, Any] | None = None
        self.running = False
        self.curve_points: list[tuple[float, int, int]] = []
        self._ignore_batch_select = False

        self.config_path = tk.StringVar(value=initial_config)
        self.config_summary = tk.StringVar(value="尚未加载配置")
        self.version_var = tk.StringVar(value="v-")
        self.status_var = tk.StringVar(value="READY")
        self.progress_var = tk.StringVar(value="0 / 0 samples")
        self.conclusion_var = tk.StringVar(value="等待运行")
        self.latency_var = tk.StringVar(value="查询延迟: -")
        self.replacement_var = tk.StringVar(value="当前视图: 空")

        self._build_styles()
        self._build_header()
        self._build_toolbar()
        self._build_summary_bar()
        self._build_workspace()
        self._build_tabs()
        self._build_status_bar()

        self._load_config_preview(initial_config, announce=False)
        self._create_session()
        self.after(100, self._poll_queue)
        if auto_run:
            self.after(450, self._run_current)

    def _build_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL_ALT)
        style.configure("App.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL, relief="flat")
        style.configure("Card.TFrame", background=PANEL_ALT, relief="flat")
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 21))
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Value.TLabel", background=PANEL, foreground=ACCENT, font=("Consolas", 11, "bold"))
        style.configure("Status.TLabel", background=PANEL_ALT, foreground=TEXT, font=("Consolas", 10, "bold"), padding=(10, 5))
        style.configure("Accent.TButton", background=ACCENT_DARK, foreground=ACCENT, bordercolor=ACCENT, padding=(12, 7))
        style.map("Accent.TButton", background=[("active", "#1d5652"), ("disabled", "#22323b")])
        style.configure("Tool.TButton", background=PANEL_ALT, foreground=TEXT, bordercolor=BORDER, padding=(10, 7))
        style.map("Tool.TButton", background=[("active", "#233b4b"), ("disabled", "#182630")])
        style.configure("TEntry", padding=6)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_ALT, foreground=MUTED, padding=(14, 7))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure(
            "Treeview",
            background=PANEL,
            fieldbackground=PANEL,
            foreground=TEXT,
            rowheight=25,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background=PANEL_ALT,
            foreground=ACCENT,
            relief="flat",
            font=("Segoe UI Semibold", 9),
        )
        style.map("Treeview", background=[("selected", "#24525a")], foreground=[("selected", "#ffffff")])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=PANEL_ALT, bordercolor=BORDER)

    def _build_header(self) -> None:
        header = ttk.Frame(self, style="App.TFrame", padding=(20, 14, 20, 7))
        header.pack(fill="x")
        left = ttk.Frame(header, style="App.TFrame")
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="MIXED-PRECISION / MEMORY DIAGNOSTIC", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            left,
            text="确定性本地样本 · float32 参考 · 整数定点 · 可重放块池 · Windows/Tkinter",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(3, 0))
        right = ttk.Frame(header, style="App.TFrame")
        right.pack(side="right")
        ttk.Label(right, textvariable=self.version_var, style="Status.TLabel").pack(side="left", padx=(0, 8))
        self.status_badge = ttk.Label(right, textvariable=self.status_var, style="Status.TLabel")
        self.status_badge.pack(side="left", padx=(0, 8))
        self.latency_badge = ttk.Label(right, textvariable=self.latency_var, style="Status.TLabel")
        self.latency_badge.pack(side="left")

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self, style="Panel.TFrame", padding=(18, 10))
        toolbar.pack(fill="x", padx=16, pady=(4, 8))
        ttk.Label(toolbar, text="配置 JSON", style="Muted.TLabel").pack(side="left", padx=(0, 8))
        self.config_entry = ttk.Entry(toolbar, textvariable=self.config_path, width=52)
        self.config_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(toolbar, text="浏览", style="Tool.TButton", command=self._browse_config).pack(side="left", padx=3)
        ttk.Button(toolbar, text="校验配置", style="Tool.TButton", command=self._preview_config).pack(side="left", padx=3)
        self.run_button = ttk.Button(toolbar, text="执行当前", style="Accent.TButton", command=self._run_current)
        self.run_button.pack(side="left", padx=(8, 3))
        self.demo_button = ttk.Button(toolbar, text="三次版本演示", style="Tool.TButton", command=self._run_demo)
        self.demo_button.pack(side="left", padx=3)
        ttk.Button(toolbar, text="刷新查询", style="Tool.TButton", command=self._refresh_from_session).pack(side="left", padx=3)

    def _build_summary_bar(self) -> None:
        summary = ttk.Frame(self, style="Panel.TFrame", padding=(18, 9))
        summary.pack(fill="x", padx=16, pady=(0, 8))
        ttk.Label(summary, textvariable=self.config_summary, style="Panel.TLabel").pack(side="left", fill="x", expand=True)
        ttk.Label(summary, textvariable=self.progress_var, style="Value.TLabel").pack(side="right", padx=(12, 0))

    def _build_workspace(self) -> None:
        workspace = ttk.Frame(self, style="App.TFrame")
        workspace.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        workspace.columnconfigure(0, weight=5)
        workspace.columnconfigure(1, weight=6)
        workspace.rowconfigure(0, weight=3)

        left = ttk.Frame(workspace, style="Panel.TFrame", padding=(12, 10))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        title_row = ttk.Frame(left, style="Panel.TFrame")
        title_row.pack(fill="x")
        ttk.Label(title_row, text="批次列表", style="Panel.TLabel", font=("Segoe UI Semibold", 11)).pack(side="left")
        ttk.Label(title_row, textvariable=self.replacement_var, style="Muted.TLabel").pack(side="right")

        batch_container = ttk.Frame(left, style="Panel.TFrame")
        batch_container.pack(fill="both", expand=True, pady=(8, 0))
        self.batch_tree = ttk.Treeview(
            batch_container,
            columns=("id", "samples", "status", "duration", "peak", "frag", "progress", "digest"),
            show="headings",
            height=9,
        )
        for column, heading, width in (
            ("id", "批次", 72),
            ("samples", "样本", 62),
            ("status", "状态", 82),
            ("duration", "耗时 ms", 84),
            ("peak", "峰值块", 72),
            ("frag", "碎片", 60),
            ("progress", "进度", 70),
            ("digest", "输出摘要", 132),
        ):
            self.batch_tree.heading(column, text=heading)
            self.batch_tree.column(column, width=width, anchor="center", stretch=column == "digest")
        scrollbar = ttk.Scrollbar(batch_container, orient="vertical", command=self.batch_tree.yview)
        self.batch_tree.configure(yscrollcommand=scrollbar.set)
        self.batch_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.batch_tree.tag_configure("complete", foreground=TEXT)
        self.batch_tree.tag_configure("running", foreground=WARN)
        self.batch_tree.tag_configure("fail", foreground=FAIL)
        self.batch_tree.bind("<<TreeviewSelect>>", self._on_batch_selected)

        right = ttk.Frame(workspace, style="Panel.TFrame", padding=(12, 10))
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        chart_header = ttk.Frame(right, style="Panel.TFrame")
        chart_header.pack(fill="x")
        ttk.Label(chart_header, text="显存占用曲线", style="Panel.TLabel", font=("Segoe UI Semibold", 11)).pack(side="left")
        ttk.Label(chart_header, text="块数 / 单调时钟 ms · 事件驱动", style="Muted.TLabel").pack(side="right")
        self.canvas = tk.Canvas(right, bg="#09131a", highlightthickness=0, height=230)
        self.canvas.pack(fill="both", expand=True, pady=(8, 0))
        self.canvas.bind("<Configure>", lambda _event: self._draw_curve())
        self.progress = ttk.Progressbar(right, style="Horizontal.TProgressbar", mode="determinate")
        self.progress.pack(fill="x", pady=(8, 0))
        conclusion = ttk.Frame(right, style="Card.TFrame", padding=(10, 7))
        conclusion.pack(fill="x", pady=(8, 0))
        ttk.Label(conclusion, text="一致性结论", style="Muted.TLabel").pack(anchor="w")
        self.conclusion_label = ttk.Label(conclusion, textvariable=self.conclusion_var, background=PANEL_ALT, foreground=TEXT, font=("Segoe UI", 10))
        self.conclusion_label.pack(anchor="w", pady=(2, 0))

    def _build_tabs(self) -> None:
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True, padx=16, pady=(0, 6))

        log_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=8)
        tabs.add(log_tab, text="运行日志")
        self.log_text = scrolledtext.ScrolledText(
            log_tab,
            bg="#09131a",
            fg=TEXT,
            insertbackground=ACCENT,
            relief="flat",
            font=("Consolas", 9),
            height=8,
        )
        self.log_text.pack(fill="both", expand=True)

        errors_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=8)
        tabs.add(errors_tab, text="逐层误差界")
        self.error_tree = ttk.Treeview(
            errors_tab,
            columns=("layer", "max_error", "bound", "share", "cumulative", "saturations", "valid"),
            show="headings",
            height=7,
        )
        for column, heading, width in (
            ("layer", "层", 56),
            ("max_error", "观测最大误差 (int)", 130),
            ("bound", "整数误差界", 130),
            ("share", "误差占比", 92),
            ("cumulative", "累计占比", 92),
            ("saturations", "饱和次数", 88),
            ("valid", "界有效", 76),
        ):
            self.error_tree.heading(column, text=heading)
            self.error_tree.column(column, width=width, anchor="center")
        self.error_tree.pack(fill="both", expand=True)

        events_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=8)
        tabs.add(events_tab, text="块池分配明细")
        self.event_tree = ttk.Treeview(
            events_tab,
            columns=("seq", "action", "batch", "stage", "blocks", "bytes", "free", "frag", "elapsed"),
            show="headings",
            height=8,
        )
        for column, heading, width in (
            ("seq", "序号", 58),
            ("action", "动作", 76),
            ("batch", "批次", 78),
            ("stage", "阶段", 130),
            ("blocks", "块序号", 160),
            ("bytes", "请求字节", 96),
            ("free", "空闲块", 72),
            ("frag", "碎片", 60),
            ("elapsed", "ms", 82),
        ):
            self.event_tree.heading(column, text=heading)
            self.event_tree.column(column, width=width, anchor="center", stretch=column == "blocks")
        self.event_tree.pack(fill="both", expand=True)

        samples_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=8)
        tabs.add(samples_tab, text="逐样本归因")
        self.sample_tree = ttk.Treeview(
            samples_tab,
            columns=("sample", "batch", "dominant", "output_error", "saturations", "input_digest"),
            show="headings",
            height=8,
        )
        for column, heading, width in (
            ("sample", "样本", 90),
            ("batch", "批次", 78),
            ("dominant", "最大偏差层", 112),
            ("output_error", "输出误差 (int)", 118),
            ("saturations", "饱和次数", 88),
            ("input_digest", "输入摘要", 280),
        ):
            self.sample_tree.heading(column, text=heading)
            self.sample_tree.column(column, width=width, anchor="center", stretch=column == "input_digest")
        self.sample_tree.pack(fill="both", expand=True)

        latency_tab = ttk.Frame(tabs, style="Panel.TFrame", padding=8)
        tabs.add(latency_tab, text="界面查询延迟")
        self.latency_summary = ttk.Label(latency_tab, text="暂无查询记录", style="Panel.TLabel", font=("Consolas", 10))
        self.latency_summary.pack(anchor="w", pady=(0, 6))
        self.latency_tree = ttk.Treeview(
            latency_tab,
            columns=("query", "operation", "batch", "start_ns", "end_ns", "duration", "budget"),
            show="headings",
            height=5,
        )
        for column, heading, width in (
            ("query", "查询", 58),
            ("operation", "操作", 190),
            ("batch", "批次", 80),
            ("start_ns", "起 (ns)", 150),
            ("end_ns", "止 (ns)", 150),
            ("duration", "定点 μs", 94),
            ("budget", "超预算", 78),
        ):
            self.latency_tree.heading(column, text=heading)
            self.latency_tree.column(column, width=width, anchor="center", stretch=column == "operation")
        self.latency_tree.pack(fill="both", expand=True)

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self, style="Panel.TFrame", padding=(16, 6))
        bar.pack(fill="x", side="bottom")
        ttk.Label(
            bar,
            text="本地确定性运行 · time.perf_counter_ns 单调时钟 · 报告真值: ui_latency.queries · 无网络/GPU 依赖",
            style="Muted.TLabel",
        ).pack(side="left")
        ttk.Label(bar, text="v1.0.0", style="Muted.TLabel").pack(side="right")

    def _create_session(self) -> None:
        try:
            self.session = DiagnosticSession(
                self.request,
                report_dir=self.report_dir,
                snapshot_dir=self.snapshot_dir,
                on_event=lambda event: self.queue.put(event),
            )
            self._log(
                f"会话已建立: seed={self.request.seed}, samples={self.request.sample_count}, "
                f"input={self.request.input_dim}, hidden={self.request.hidden_dim}"
            )
            self._log(f"数据摘要 {self.session.artifacts.data_sha256}")
            self._log(f"参数摘要 {self.session.artifacts.params_sha256}")
        except DiagnosticError as exc:
            messagebox.showerror("初始化失败", str(exc))

    def _browse_config(self) -> None:
        path = filedialog.askopenfilename(
            title="选择推理配置 JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
            initialdir=str(Path("configs").resolve()),
        )
        if path:
            self.config_path.set(path)
            self._load_config_preview(path)

    def _preview_config(self) -> None:
        self._load_config_preview(self.config_path.get(), announce=True)

    def _load_config_preview(self, path: str, announce: bool = True) -> None:
        try:
            config = load_config(path)
        except DiagnosticError as exc:
            self.config_summary.set(f"配置拒绝 {exc.code}: {exc.field_path or ''}")
            if announce:
                messagebox.showerror("配置校验失败", str(exc))
            return
        self.config_summary.set(self._format_config(config))
        if announce:
            self._log(f"配置校验通过: {path}")

    @staticmethod
    def _format_config(config: InferenceConfig) -> str:
        return (
            f"预算 {config.memory_budget_mib} MiB → {_format_bytes(config.effective_budget_bytes)} "
            f"({config.total_blocks} blocks × {config.block_size_kb} KiB)  ·  "
            f"{config.precision}  ·  scale 10^{config.quant_scale_power}  ·  "
            f"batch≤{config.batch_size_limit}  ·  {config.preallocation}  ·  "
            f"pressure≥{config.pressure_threshold_ms:g}ms  ·  UI≤{config.ui_query_budget_ms:g}ms"
        )

    def _run_current(self) -> None:
        if self.running or self.session is None:
            return
        path = self.config_path.get()
        try:
            load_config(path)
        except DiagnosticError as exc:
            messagebox.showerror("配置校验失败", str(exc))
            return
        self._start_worker([path])

    def _run_demo(self) -> None:
        if self.running or self.session is None:
            return
        paths = [str(Path("configs") / name) for name in ("float32.json", "fixed.json", "scaled.json")]
        missing = [path for path in paths if not Path(path).exists()]
        if missing:
            messagebox.showerror("演示配置缺失", "\n".join(missing))
            return
        self._start_worker(paths)

    def _start_worker(self, config_paths: list[str]) -> None:
        self.running = True
        self.run_button.configure(state="disabled")
        self.demo_button.configure(state="disabled")
        self.status_var.set("RUNNING")
        self.status_badge.configure(foreground=WARN)
        self.curve_points.clear()
        self._clear_batch_tree()
        self._draw_curve()
        self._log(f"开始运行 {len(config_paths)} 个配置版本")
        threading.Thread(target=self._worker, args=(config_paths,), daemon=True).start()

    def _worker(self, config_paths: list[str]) -> None:
        try:
            for path in config_paths:
                assert self.session is not None
                config = load_config(path)
                self.queue.put({"type": "config_change", "config": config.to_dict(), "path": path})
                self.session.run_config(config)
                self.queue.put({"type": "version_ready", "version": config})
            self.queue.put({"type": "run_complete"})
        except DiagnosticError as exc:
            self.queue.put({"type": "run_error", "error": exc.to_dict(), "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI with a traceback-like message
            self.queue.put(
                {
                    "type": "run_error",
                    "error": {"code": "E_INTERNAL", "message": repr(exc), "field_path": None, "details": {}},
                    "message": repr(exc),
                }
            )

    def _poll_queue(self) -> None:
        try:
            while True:
                event = self.queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        self.after(80, self._poll_queue)

    def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "memory_event":
            memory_event = event["event"]
            self.curve_points.append(
                (
                    float(memory_event["elapsed_ms"]),
                    int(memory_event["used_blocks_after"]),
                    int(memory_event["fragmentation_blocks_after"]),
                )
            )
            self._draw_curve()
            return
        if event_type == "batch_start":
            batch_id = event["batch_id"]
            values = (
                batch_id,
                event["admitted_samples"],
                "RUNNING",
                "-",
                "-",
                "-",
                f"0/{event['admitted_samples']}",
                "-",
            )
            if self.batch_tree.exists(batch_id):
                self.batch_tree.item(batch_id, values=values, tags=("running",))
            else:
                self.batch_tree.insert("", "end", iid=batch_id, values=values, tags=("running",))
            self.status_var.set(f"RUNNING {batch_id}")
            return
        if event_type == "sample_complete":
            batch_id = event.get("batch_id")
            if batch_id and self.batch_tree.exists(batch_id):
                current = list(self.batch_tree.item(batch_id, "values"))
                try:
                    total = int(str(current[1]))
                    done = int(str(current[6]).split("/")[0]) + 1
                    current[6] = f"{done}/{total}"
                    self.batch_tree.item(batch_id, values=current)
                except (ValueError, IndexError):
                    pass
            return
        if event_type == "batch_complete":
            batch = event["batch"]
            batch_id = batch["batch_id"]
            values = (
                batch_id,
                batch["sample_count"],
                batch["status"],
                f"{batch['duration_ms']:.2f}",
                batch["peak_total_used_blocks"],
                batch["fragmentation_at_peak_blocks"],
                f"{batch['sample_count']}/{batch['sample_count']}",
                _short_hash(batch["output_sha256"]),
            )
            if self.batch_tree.exists(batch_id):
                self.batch_tree.item(batch_id, values=values, tags=("complete",))
            else:
                self.batch_tree.insert("", "end", iid=batch_id, values=values, tags=("complete",))
            frac = batch["progress"]["fraction_ppm"] / 10_000.0
            self.progress["value"] = frac
            self.progress_var.set(
                f"{batch['progress']['completed_samples']} / {batch['progress']['total_samples']} samples · "
                f"{frac:.1f}%"
            )
            return
        if event_type == "config_change":
            config = event["config"]
            self.config_path.set(event["path"])
            self.config_summary.set(f"配置载入 · {self._format_config(load_config(event['path']))}")
            self.curve_points.clear()
            self._clear_batch_tree()
            self.progress["value"] = 0
            self.progress_var.set("0 / 0 samples")
            self._draw_curve()
            self._log(f"运行配置 {event['path']}")
            return
        if event_type == "engine_complete":
            return
        if event_type == "version_ready":
            self._refresh_from_session()
            return
        if event_type == "run_complete":
            self.running = False
            self.run_button.configure(state="normal")
            self.demo_button.configure(state="normal")
            self._refresh_from_session()
            self._log("全部版本运行完成")
            return
        if event_type == "run_error":
            self.running = False
            self.run_button.configure(state="normal")
            self.demo_button.configure(state="normal")
            error = event["error"]
            self.status_var.set(error.get("code", "ERROR"))
            self.status_badge.configure(foreground=FAIL)
            self._log(f"ERROR {error}")
            messagebox.showerror("运行失败", event.get("message", "unknown error"))
            return

    def _refresh_from_session(self) -> None:
        if self.session is None or self.session.current_report is None or self.running:
            return
        try:
            view = self.session.query_view()
        except DiagnosticError as exc:
            self._log(f"查询失败: {exc}")
            return
        self.view = view
        self._render_view(view)

    def _render_view(self, view: dict[str, Any]) -> None:
        config = view["config"]
        summary = view["summary"]
        version = view["config_version"]
        self.version_var.set(f"CONFIG v{version}")
        self.status_var.set(summary["status"])
        self.status_badge.configure(foreground=_status_color(summary["status"]))
        self.config_summary.set(self._format_config(load_config(self.config_path.get())) if Path(self.config_path.get()).exists() else self._format_config_from_dict(config))
        completed = sum(batch["sample_count"] for batch in view["batches"])
        total = view["progress"]["total_samples"]
        self.progress_var.set(f"{completed} / {total} samples")
        self.progress["value"] = 100.0 * completed / total if total else 0
        conclusion = summary["conclusion"]
        self.conclusion_var.set(f"{conclusion['status']} · {conclusion['message']}")
        self.conclusion_label.configure(foreground=_status_color(conclusion["status"]))
        replacement = view.get("replacement") or {}
        changed = ", ".join(replacement.get("changed_fields", [])) or "无"
        affected = len(replacement.get("affected_batches", []))
        self.replacement_var.set(
            f"整体替换 v{replacement.get('previous_version') or '-'}→v{version} · 受影响 {affected} 批 · 字段 {changed}"
        )
        self.curve_points = [
            (point["elapsed_ms"], point["used_blocks"], point["fragmentation_blocks"])
            for point in view["memory"]["curve"]
        ]
        self._draw_curve()
        self._render_batches(view["batches"])
        self._render_errors(view.get("layer_error_bounds", []))
        self._render_events(view["memory"]["events"])
        self._render_samples(view.get("samples", []))
        self._render_latency()
        self._log(
            f"v{version} {config['precision']}/{config['preallocation']} "
            f"status={summary['status']} codes={summary['codes']} "
            f"replay={view['memory']['replay']['verified']}"
        )

    def _format_config_from_dict(self, config: dict[str, Any]) -> str:
        return (
            f"预算 {config['memory_budget_mib']} MiB → {_format_bytes(config['effective_budget_bytes'])} "
            f"({config['total_blocks']} blocks × {config['block_size_kb']} KiB) · "
            f"{config['precision']} · 10^{config['quant_scale_power']} · batch≤{config['batch_size_limit']} · "
            f"{config['preallocation']}"
        )

    def _clear_batch_tree(self) -> None:
        for item in self.batch_tree.get_children():
            self.batch_tree.delete(item)

    def _render_batches(self, batches: list[dict[str, Any]]) -> None:
        self._ignore_batch_select = True
        self._clear_batch_tree()
        for batch in batches:
            self.batch_tree.insert(
                "",
                "end",
                iid=batch["batch_id"],
                values=(
                    batch["batch_id"],
                    batch["sample_count"],
                    batch["status"],
                    f"{batch['duration_ms']:.2f}",
                    batch["peak_total_used_blocks"],
                    batch["fragmentation_at_peak_blocks"],
                    f"{batch['sample_count']}/{batch['sample_count']}",
                    _short_hash(batch["output_sha256"]),
                ),
                tags=("complete",),
            )
        self._ignore_batch_select = False

    def _render_errors(self, layers: list[dict[str, Any]]) -> None:
        for item in self.error_tree.get_children():
            self.error_tree.delete(item)
        for layer in layers:
            self.error_tree.insert(
                "",
                "end",
                values=(
                    layer["layer"],
                    layer["observed_max_abs_error_int"],
                    layer["error_bound_int"],
                    f"{layer['error_share_ppm']/10000:.2f}%",
                    f"{layer['cumulative_error_share_ppm']/10000:.2f}%",
                    layer["saturation_count"],
                    "YES" if layer["error_bound_valid"] else "NO",
                ),
            )

    def _render_events(self, events: list[dict[str, Any]]) -> None:
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        for event in events:
            blocks = event.get("blocks", [])
            block_text = _compact_blocks(blocks)
            self.event_tree.insert(
                "",
                "end",
                values=(
                    event["seq"],
                    event["action"],
                    event.get("batch_id") or "-",
                    event.get("stage") or event.get("purpose"),
                    block_text,
                    event.get("requested_bytes", 0),
                    event.get("free_blocks_after", "-"),
                    event.get("fragmentation_blocks_after", "-"),
                    f"{event.get('elapsed_ms', 0):.2f}",
                ),
            )

    def _render_samples(self, samples: list[dict[str, Any]]) -> None:
        for item in self.sample_tree.get_children():
            self.sample_tree.delete(item)
        for sample in samples:
            self.sample_tree.insert(
                "",
                "end",
                iid=sample["sample_id"],
                values=(
                    sample["sample_id"],
                    sample.get("batch_id", "-"),
                    sample.get("dominant_layer") if sample.get("dominant_layer") is not None else "-",
                    sample.get("output_max_abs_error_int", "-"),
                    sample.get("total_saturations", "-"),
                    sample.get("input_sha256", "-"),
                ),
            )

    def _render_latency(self) -> None:
        if self.session is None or self.session.current_report is None:
            return
        latency = self.session.current_report["ui_latency"]
        overall = latency["aggregates"].get("__all__", {})
        exceeded = bool(latency.get("codes"))
        self.latency_var.set(
            f"UI QUERY P95 {overall.get('p95_ms', 0):.3f}ms / MAX {overall.get('max_ms', 0):.3f}ms"
        )
        self.latency_badge.configure(foreground=FAIL if exceeded else PASS)
        summary_codes = self.session.current_report["summary"].get("codes", [])
        base_status = self.session.current_report["summary"].get("status", "READY")
        self.status_var.set(base_status + (f" · {'/'.join(summary_codes)}" if summary_codes else ""))
        self.latency_summary.configure(
            text=(
                f"budget={latency['budget_ms']:g}ms · count={latency['query_count']} · "
                f"mean={overall.get('mean_ms', 0):.3f}ms · p95={overall.get('p95_ms', 0):.3f}ms · "
                f"max={overall.get('max_ms', 0):.3f}ms · fixed μs · "
                f"quantization_error≤{latency['fixed_point']['raw_quantization_error_upper_ms']}ms · "
                f"codes={latency['codes']}"
            )
        )
        for item in self.latency_tree.get_children():
            self.latency_tree.delete(item)
        for record in latency["queries"]:
            self.latency_tree.insert(
                "",
                "end",
                values=(
                    record["query_id"],
                    record["operation"],
                    record["batch_id"] or "ALL",
                    record["start_mono_ns"],
                    record["end_mono_ns"],
                    record["duration_us"],
                    "YES" if record["exceeded_budget"] else "NO",
                ),
                tags=("exceeded",) if record["exceeded_budget"] else (),
            )
        self.latency_tree.tag_configure("exceeded", foreground=FAIL)

    def _on_batch_selected(self, _event: tk.Event[Any]) -> None:
        if self._ignore_batch_select or self.session is None or self.running:
            return
        selection = self.batch_tree.selection()
        if not selection:
            return
        batch_id = selection[0]
        try:
            self.session.query_batch(batch_id)
            samples = self.session.query_samples_for_batch(batch_id)
        except DiagnosticError as exc:
            self._log(f"批次查询失败: {exc}")
            return
        self._render_latency()
        if samples:
            selected_layers = samples[0].get("layer_diagnostics", [])
            if selected_layers:
                layer_count = len(selected_layers)
                absolute_sums = [
                    sum(int(sample["layer_diagnostics"][layer_index]["observed_absolute_error_sum_int"]) for sample in samples)
                    for layer_index in range(layer_count)
                ]
                total_all = sum(absolute_sums)
                cumulative = 0
                synthetic: list[dict[str, Any]] = []
                for layer_index in range(len(selected_layers)):
                    selected = [sample["layer_diagnostics"][layer_index] for sample in samples]
                    layer_sum = absolute_sums[layer_index]
                    cumulative += layer_sum
                    synthetic.append(
                        {
                            "layer": layer_index,
                            "observed_max_abs_error_int": max(int(item["observed_max_abs_error_int"]) for item in selected),
                            "error_bound_int": max(int(item["error_bound_int"]) for item in selected),
                            "saturation_count": sum(int(item["saturation_count"]) for item in selected),
                            "error_bound_valid": all(bool(item["error_bound_valid"]) for item in selected),
                            "error_share_ppm": int(round(1_000_000 * layer_sum / total_all)) if total_all else 0,
                            "cumulative_error_share_ppm": int(round(1_000_000 * cumulative / total_all)) if total_all else 0,
                        }
                    )
                # Replace per-layer tabs with the selected batch's attribution.
                self._render_errors(synthetic)
        self._log(f"已查询批次 {batch_id}: {len(samples)} samples")

    def _draw_curve(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = max(320, canvas.winfo_width())
        height = max(180, canvas.winfo_height())
        left, top, right, bottom = 52, 18, width - 18, height - 32
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = bottom - fraction * (bottom - top)
            canvas.create_line(left, y, right, y, fill="#1a303d", dash=(2, 5))
            canvas.create_text(left - 8, y, text=f"{fraction:.0%}", fill=MUTED, anchor="e", font=("Consolas", 8))
        if not self.curve_points:
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="等待运行事件…",
                fill=MUTED,
                font=("Segoe UI", 10),
            )
            return
        max_ms = max(1.0, max(point[0] for point in self.curve_points))
        max_used = max(1, max(point[1] for point in self.curve_points))
        max_used = max(max_used, 4)
        points: list[float] = []
        frag_points: list[float] = []
        for elapsed_ms, used, fragmentation in self.curve_points:
            x = left + (elapsed_ms / max_ms) * (right - left)
            y = bottom - (used / max_used) * (bottom - top)
            points.extend((x, y))
            fy = bottom - (fragmentation / max_used) * (bottom - top)
            frag_points.extend((x, fy))
        if len(points) >= 4:
            canvas.create_line(*points, fill=ACCENT, width=2, smooth=True)
            canvas.create_line(*frag_points, fill=WARN, width=1, dash=(3, 3))
        elif len(points) == 2:
            canvas.create_oval(points[0] - 3, points[1] - 3, points[0] + 3, points[1] + 3, fill=ACCENT, outline="")
        canvas.create_text(left, top - 8, text=f"used max={max_used} blocks", fill=ACCENT, anchor="w", font=("Consolas", 8))
        canvas.create_text(right, top - 8, text=f"{max_ms:.1f} ms", fill=MUTED, anchor="e", font=("Consolas", 8))
        canvas.create_text(right, bottom + 18, text="— used blocks   --- fragmentation", fill=MUTED, anchor="e", font=("Segoe UI", 8))

    def _log(self, message: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")


def _compact_blocks(blocks: list[int]) -> str:
    if not blocks:
        return "-"
    if len(blocks) > 12:
        return f"{blocks[0]}..{blocks[-1]} ({len(blocks)} blocks)"
    return ",".join(str(block) for block in blocks)


def launch(
    *,
    request: RunRequest,
    initial_config: str,
    report_dir: str,
    snapshot_dir: str,
    auto_run: bool = False,
) -> None:
    app = DiagnosticApp(
        request=request,
        initial_config=initial_config,
        report_dir=report_dir,
        snapshot_dir=snapshot_dir,
        auto_run=auto_run,
    )
    app.mainloop()


if __name__ == "__main__":
    launch(
        request=RunRequest(),
        initial_config="configs/float32.json",
        report_dir="reports",
        snapshot_dir="snapshots",
        auto_run=True,
    )
