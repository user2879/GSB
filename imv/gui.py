from __future__ import annotations

import datetime as _dt
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .errors import BusinessError, as_business_error
from .workbench import Workbench


NAVY = "#0b1220"
NAVY_2 = "#111c2f"
SLATE = "#152238"
PANEL = "#f4f7fb"
CARD = "#ffffff"
TEXT = "#182235"
MUTED = "#64748b"
GREEN = "#166534"
GREEN_BG = "#dcfce7"
AMBER = "#92400e"
AMBER_BG = "#fef3c7"
RED = "#991b1b"
RED_BG = "#fee2e2"
BLUE = "#1d4ed8"
BLUE_BG = "#dbeafe"
BORDER = "#d7e0ea"


def launch(
    base_path: str | Path,
    views_path: str | Path,
    changes_path: str | Path,
    data_dir: str | Path,
    *,
    reset: bool = False,
    auto_rebuild_on_expiry: bool = True,
) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # pragma: no cover - exercised on minimal Python builds
        raise BusinessError(
            "E_STATE",
            "tkinter is unavailable in this Python installation",
            {"module": "tkinter"},
            {"error": str(exc)},
        ) from exc

    root = tk.Tk()
    app = WorkbenchApp(
        root,
        tk,
        ttk,
        base_path,
        views_path,
        changes_path,
        data_dir,
        reset=reset,
        auto_rebuild_on_expiry=auto_rebuild_on_expiry,
    )
    try:
        root.mainloop()
    finally:
        if app.workbench is not None:
            try:
                app.workbench.close()
            except Exception:
                pass


class WorkbenchApp:
    def __init__(
        self,
        root: Any,
        tk: Any,
        ttk: Any,
        base_path: str | Path,
        views_path: str | Path,
        changes_path: str | Path,
        data_dir: str | Path,
        *,
        reset: bool,
        auto_rebuild_on_expiry: bool,
    ):
        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.base_path = Path(base_path)
        self.views_path = Path(views_path)
        self.changes_path = Path(changes_path)
        self.data_dir = Path(data_dir)
        self.reset = reset
        self.auto_rebuild_on_expiry = auto_rebuild_on_expiry
        self.workbench: Workbench | None = None
        self.events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self.action_buttons: list[Any] = []
        self.busy = False
        self.closing = False
        self.last_fast: dict[str, Any] | None = None
        self.selected_view: str | None = None
        self._last_view_values: tuple[str, ...] = ()
        self._last_aggregate_signature: tuple[str, ...] = ()
        self._event_after_id: str | None = None
        self._fast_after_id: str | None = None
        self._init_ui()
        self.root.after(40, self._open_workbench)
        self._event_after_id = self.root.after(120, self._poll_events)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        from tkinter import font as tkfont

        self.root.title("增量物化视图维护工作台 · Incremental Materialized View Workbench")
        self.root.geometry("1440x900")
        self.root.minsize(1120, 720)
        try:
            self.root.state("zoomed")
        except Exception:
            pass
        families = set(tkfont.families(self.root))
        family = "Microsoft YaHei UI" if "Microsoft YaHei UI" in families else ("Segoe UI" if "Segoe UI" in families else "TkDefaultFont")
        self.font_family = family
        self.root.option_add("*Font", (family, 10))
        self.root.configure(bg=PANEL)
        self._configure_styles()
        self._build_header()
        self._build_statusbar()
        self._build_body()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-Return>", lambda _event: self._advance_one())
        self.root.bind("<Control-r>", lambda _event: self._authority_recompute())
        self.root.bind("<F5>", lambda _event: self._refresh_fast())

    def _configure_styles(self) -> None:
        style = self.ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=(self.font_family, 10), background=PANEL, foreground=TEXT)
        style.configure("TFrame", background=PANEL)
        style.configure("Panel.TFrame", background=CARD)
        style.configure("Sidebar.TFrame", background="#eef3f9")
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("SidebarMuted.TLabel", background="#eef3f9", foreground=MUTED)
        style.configure("CardTitle.TLabel", foreground=MUTED, font=(self.font_family, 9))
        style.configure("CardValue.TLabel", foreground=TEXT, font=(self.font_family, 17, "bold"))
        style.configure("Section.TLabelframe", background="#eef3f9", bordercolor=BORDER)
        style.configure("Section.TLabelframe.Label", background="#eef3f9", foreground=NAVY_2, font=(self.font_family, 10, "bold"))
        style.configure("TButton", padding=(9, 6), background="#ffffff", bordercolor=BORDER)
        style.map("TButton", background=[("active", "#e8eef6"), ("disabled", "#eef2f6")])
        style.configure("Accent.TButton", background=GREEN, foreground="#ffffff", bordercolor=GREEN, font=(self.font_family, 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#14532d"), ("disabled", "#86a891")], foreground=[("disabled", "#eef7f0")])
        style.configure("Danger.TButton", foreground=RED)
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 8), background="#e2e9f2", foreground=MUTED)
        style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", NAVY_2)])
        style.configure(
            "Treeview",
            background=CARD,
            fieldbackground=CARD,
            foreground=TEXT,
            rowheight=27,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
        )
        style.configure("Treeview.Heading", background="#e8eef6", foreground=NAVY_2, font=(self.font_family, 9, "bold"), padding=(6, 5))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", NAVY)])
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TSpinbox", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")
        style.configure("Horizontal.TProgressbar", background=GREEN, troughcolor="#e5eaf1", thickness=4)
        style.configure("TLabelframe", background="#eef3f9")

    def _build_header(self) -> None:
        header = self.tk.Frame(self.root, bg=NAVY, height=86)
        header.pack(side="top", fill="x")
        header.pack_propagate(False)
        left = self.tk.Frame(header, bg=NAVY)
        left.pack(side="left", fill="y", padx=(22, 0), pady=14)
        self.tk.Label(
            left,
            text="增量物化视图维护工作台",
            bg=NAVY,
            fg="#f8fafc",
            font=(self.font_family, 19, "bold"),
            anchor="w",
        ).pack(anchor="w")
        self.tk.Label(
            left,
            text="INCREMENTAL MATERIALIZED VIEW WORKBENCH   ·   CACHE-FIRST   ·   LEASE-SAFE   ·   BYTE-AUDITED",
            bg=NAVY,
            fg="#93a4bb",
            font=(self.font_family, 8),
            anchor="w",
        ).pack(anchor="w", pady=(3, 0))
        right = self.tk.Frame(header, bg=NAVY)
        right.pack(side="right", fill="y", padx=22, pady=18)
        self.header_badge = self.tk.Label(
            right,
            text="初始化中",
            bg=AMBER_BG,
            fg=AMBER,
            font=(self.font_family, 10, "bold"),
            padx=13,
            pady=7,
        )
        self.header_badge.pack(anchor="e")
        self.header_detail = self.tk.Label(
            right,
            text="读取本地确定性样本…",
            bg=NAVY,
            fg="#93a4bb",
            font=(self.font_family, 9),
        )
        self.header_detail.pack(anchor="e", pady=(6, 0))

    def _build_body(self) -> None:
        body = self.ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(12, 8))
        sidebar = self.ttk.Frame(body, style="Sidebar.TFrame", width=330)
        sidebar.pack_propagate(False)
        body.add(sidebar, weight=0)
        self.notebook = self.ttk.Notebook(body)
        body.add(self.notebook, weight=1)
        self._build_sidebar(sidebar)
        self._build_dashboard_tab()
        self._build_results_tab()
        self._build_scheduler_tab()
        self._build_ledger_tab()
        self._build_diagnostics_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _build_sidebar(self, parent: Any) -> None:
        dataset = self.ttk.Labelframe(parent, text="本地数据集", style="Section.TLabelframe")
        dataset.pack(fill="x", padx=12, pady=(12, 8))
        for label, path in (
            ("基表", self.base_path),
            ("视图", self.views_path),
            ("变更流", self.changes_path),
            ("持久化", self.data_dir),
        ):
            row = self.ttk.Frame(dataset, style="Sidebar.TFrame")
            row.pack(fill="x", padx=9, pady=2)
            self.ttk.Label(row, text=label, style="SidebarMuted.TLabel", width=7).pack(side="left")
            name = path.name or str(path)
            self.tk.Label(
                row,
                text=name,
                bg="#eef3f9",
                fg=TEXT,
                anchor="w",
                font=(self.font_family, 9),
            ).pack(side="left", fill="x", expand=True)

        batch_box = self.ttk.Labelframe(parent, text="刷新控制", style="Section.TLabelframe")
        batch_box.pack(fill="x", padx=12, pady=8)
        row = self.ttk.Frame(batch_box, style="Sidebar.TFrame")
        row.pack(fill="x", padx=9, pady=(7, 4))
        self.ttk.Label(row, text="批次行数", style="SidebarMuted.TLabel").pack(side="left")
        self.batch_size_var = self.tk.StringVar(value="2")
        self.batch_spin = self.ttk.Spinbox(row, from_=1, to=100, textvariable=self.batch_size_var, width=6)
        self.batch_spin.pack(side="right")
        buttons = self.ttk.Frame(batch_box, style="Sidebar.TFrame")
        buttons.pack(fill="x", padx=9, pady=(2, 8))
        buttons.columnconfigure(0, weight=1)
        buttons.columnconfigure(1, weight=1)
        self._action_button(buttons, "推进批次", self._advance_one, row=0, column=0, accent=True)
        self._action_button(buttons, "推进全部", self._advance_all, row=0, column=1)
        self._action_button(buttons, "权威重算", self._authority_recompute, row=1, column=0)
        self._action_button(buttons, "一致性重放", self._replay_verify, row=1, column=1)
        self._action_button(buttons, "重建所选视图", self._rebuild_selected, row=2, column=0)
        self._action_button(buttons, "模拟租约过期", self._simulate_lease_expiry, row=2, column=1, danger=True)
        self._action_button(buttons, "热替换视图定义…", self._apply_definitions, row=3, column=0, columnspan=2)

        inject_box = self.ttk.Labelframe(parent, text="故障注入", style="Section.TLabelframe")
        inject_box.pack(fill="x", padx=12, pady=8)
        self.inject_var = self.tk.StringVar(value="租约版本冲突 · E_LEASE")
        self.inject_combo = self.ttk.Combobox(
            inject_box,
            textvariable=self.inject_var,
            state="readonly",
            values=[
                "租约版本冲突 · E_LEASE",
                "基表行数据损坏 · E_DATA",
                "增量摘要篡改 · E_SNAPSHOT",
            ],
        )
        self.inject_combo.pack(fill="x", padx=9, pady=(7, 4))
        self._action_button(inject_box, "注入并写结构化日志", self._inject_anomaly, row=0, column=0, pack=True)

        live = self.ttk.Labelframe(parent, text="实时读数", style="Section.TLabelframe")
        live.pack(fill="x", padx=12, pady=8)
        self.live_labels: dict[str, Any] = {}
        for key, label in (
            ("latency", "缓存渲染"),
            ("running", "运行 / 排队"),
            ("cursor", "游标 / 总变更"),
            ("snapshot", "快照版本"),
        ):
            row = self.ttk.Frame(live, style="Sidebar.TFrame")
            row.pack(fill="x", padx=9, pady=2)
            self.ttk.Label(row, text=label, style="SidebarMuted.TLabel").pack(side="left")
            value = self.tk.Label(row, text="—", bg="#eef3f9", fg=TEXT, font=(self.font_family, 9, "bold"))
            value.pack(side="right")
            self.live_labels[key] = value

        self.action_status = self.tk.Label(
            parent,
            text="等待工作台初始化",
            bg="#eef3f9",
            fg=MUTED,
            wraplength=295,
            justify="left",
            anchor="w",
            font=(self.font_family, 9),
        )
        self.action_status.pack(fill="x", padx=20, pady=(8, 4))
        self.progress = self.ttk.Progressbar(parent, mode="indeterminate", style="Horizontal.TProgressbar")
        self.progress.pack(fill="x", padx=20, pady=(2, 10))

    def _action_button(
        self,
        parent: Any,
        text: str,
        command: Callable[[], None],
        *,
        row: int = 0,
        column: int = 0,
        columnspan: int = 1,
        accent: bool = False,
        danger: bool = False,
        pack: bool = False,
    ) -> Any:
        style = "Accent.TButton" if accent else ("Danger.TButton" if danger else "TButton")
        button = self.ttk.Button(parent, text=text, command=command, style=style)
        if pack:
            button.pack(fill="x", padx=9, pady=(3, 8))
        else:
            button.grid(row=row, column=column, columnspan=columnspan, sticky="ew", padx=2, pady=3)
        self.action_buttons.append(button)
        return button

    def _build_dashboard_tab(self) -> None:
        tab = self.ttk.Frame(self.notebook)
        self.notebook.add(tab, text="总览")
        cards = self.ttk.Frame(tab)
        cards.pack(fill="x", padx=4, pady=(4, 12))
        self.kpi_labels: dict[str, Any] = {}
        kpis = [
            ("param_version", "参数版本", "v1"),
            ("cursor", "变更游标", "0 / 0"),
            ("views", "可见视图", "0 / 0"),
            ("jobs", "运行 / 排队", "0 / 0"),
            ("snapshot", "快照 / 格式", "v1 · 1"),
        ]
        for index, (key, title, initial) in enumerate(kpis):
            card = self.tk.Frame(cards, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
            card.grid(row=0, column=index, sticky="nsew", padx=(0 if index == 0 else 8, 0))
            cards.columnconfigure(index, weight=1)
            self.tk.Label(card, text=title, bg=CARD, fg=MUTED, font=(self.font_family, 9)).pack(anchor="w", padx=14, pady=(11, 1))
            value = self.tk.Label(card, text=initial, bg=CARD, fg=TEXT, font=(self.font_family, 18, "bold"))
            value.pack(anchor="w", padx=14, pady=(0, 12))
            self.kpi_labels[key] = value

        header = self.ttk.Frame(tab)
        header.pack(fill="x", padx=4, pady=(0, 6))
        self.ttk.Label(header, text="视图健康度与租约", font=(self.font_family, 12, "bold")).pack(side="left")
        self.ttk.Label(
            header,
            text="界面只读取增量结果缓存；权威重算不会自动介入渲染。",
            style="Muted.TLabel",
        ).pack(side="right")
        tree_frame = self.ttk.Frame(tab, style="Panel.TFrame")
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        columns = ("view", "groups", "mode", "rv", "lease", "remaining", "state", "count", "digest")
        self.view_tree = self.ttk.Treeview(tree_frame, columns=columns, show="headings")
        headings = {
            "view": "视图",
            "groups": "分组键",
            "mode": "刷新模式",
            "rv": "刷新版本",
            "lease": "租约",
            "remaining": "剩余 ms",
            "state": "展示状态",
            "count": "分组数",
            "digest": "结果摘要",
        }
        widths = {"view": 165, "groups": 145, "mode": 100, "rv": 85, "lease": 100, "remaining": 85, "state": 95, "count": 70, "digest": 210}
        for column in columns:
            self.view_tree.heading(column, text=headings[column])
            self.view_tree.column(column, width=widths[column], anchor="w", stretch=column in {"view", "groups", "digest"})
        self.view_tree.tag_configure("ok", foreground=TEXT)
        self.view_tree.tag_configure("warn", foreground=AMBER, background=AMBER_BG)
        self.view_tree.tag_configure("bad", foreground=RED, background=RED_BG)
        self.view_tree.tag_configure("run", foreground=BLUE, background=BLUE_BG)
        self.view_tree.bind("<<TreeviewSelect>>", self._on_view_selected)
        self.view_tree.bind("<Double-1>", lambda _event: self.notebook.select(1))
        scroll = self.ttk.Scrollbar(tree_frame, orient="vertical", command=self.view_tree.yview)
        self.view_tree.configure(yscrollcommand=scroll.set)
        self.view_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_results_tab(self) -> None:
        tab = self.ttk.Frame(self.notebook)
        self.notebook.add(tab, text="聚合结果")
        toolbar = self.ttk.Frame(tab)
        toolbar.pack(fill="x", padx=4, pady=(5, 8))
        self.ttk.Label(toolbar, text="视图", style="Muted.TLabel").pack(side="left", padx=(0, 6))
        self.result_view_var = self.tk.StringVar(value="")
        self.result_view_combo = self.ttk.Combobox(toolbar, textvariable=self.result_view_var, state="readonly", width=24)
        self.result_view_combo.pack(side="left")
        self.result_view_combo.bind("<<ComboboxSelected>>", self._on_result_view_changed)
        self.result_meta = self.ttk.Label(toolbar, text="", style="Muted.TLabel")
        self.result_meta.pack(side="right")
        self.result_banner = self.tk.Label(
            tab,
            text="",
            bg=GREEN_BG,
            fg=GREEN,
            anchor="w",
            padx=10,
            pady=6,
            font=(self.font_family, 9, "bold"),
        )
        self.result_banner.pack(fill="x", padx=4, pady=(0, 8))
        tree_frame = self.ttk.Frame(tab, style="Panel.TFrame")
        tree_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.group_tree = self.ttk.Treeview(tree_frame, columns=("group",), show="headings")
        xscroll = self.ttk.Scrollbar(tree_frame, orient="horizontal", command=self.group_tree.xview)
        yscroll = self.ttk.Scrollbar(tree_frame, orient="vertical", command=self.group_tree.yview)
        self.group_tree.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.group_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self.ttk.Label(
            tab,
            text="所有值均为定点整数；count 的值为行数 × scale_factor，sum/min/max 使用相同的截断与 int64 饱和规则。",
            style="Muted.TLabel",
        ).pack(fill="x", padx=6, pady=(2, 4))

    def _build_scheduler_tab(self) -> None:
        tab = self.ttk.Frame(self.notebook)
        self.notebook.add(tab, text="调度轨迹")
        top = self.ttk.Frame(tab)
        top.pack(fill="x", padx=4, pady=(5, 8))
        self.ttk.Label(top, text="真实工作线程 · 全局与单视图配额双重闸门", font=(self.font_family, 11, "bold")).pack(side="left")
        self.scheduler_summary = self.ttk.Label(top, text="", style="Muted.TLabel")
        self.scheduler_summary.pack(side="right")
        frame = self.ttk.Frame(tab, style="Panel.TFrame")
        frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        columns = ("job", "view", "kind", "queued", "wait", "run", "status", "error", "worker")
        self.scheduler_tree = self.ttk.Treeview(frame, columns=columns, show="headings")
        headings = {
            "job": "作业",
            "view": "视图",
            "kind": "类型",
            "queued": "入队单调 ms",
            "wait": "等待 ms",
            "run": "运行 ms",
            "status": "状态",
            "error": "业务码",
            "worker": "工作线程",
        }
        widths = {"job": 100, "view": 150, "kind": 175, "queued": 125, "wait": 85, "run": 80, "status": 95, "error": 95, "worker": 180}
        for column in columns:
            self.scheduler_tree.heading(column, text=headings[column])
            self.scheduler_tree.column(column, width=widths[column], anchor="w", stretch=column in {"view", "kind", "worker"})
        self.scheduler_tree.tag_configure("failed", foreground=RED, background=RED_BG)
        self.scheduler_tree.tag_configure("running", foreground=BLUE, background=BLUE_BG)
        scroll = self.ttk.Scrollbar(frame, orient="vertical", command=self.scheduler_tree.yview)
        self.scheduler_tree.configure(yscrollcommand=scroll.set)
        self.scheduler_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.ttk.Label(
            tab,
            text="超时不会终止工作线程；未完成作业保持 running/queued，并在调度轨迹中保留真实开始、结束与等待单调时钟。",
            style="Muted.TLabel",
        ).pack(fill="x", padx=6, pady=(2, 4))

    def _build_ledger_tab(self) -> None:
        tab = self.ttk.Frame(self.notebook)
        self.notebook.add(tab, text="账本与重放")
        toolbar = self.ttk.Frame(tab)
        toolbar.pack(fill="x", padx=4, pady=(5, 8))
        self.ttk.Label(toolbar, text="SQLite 单事务批次账本", font=(self.font_family, 11, "bold")).pack(side="left")
        self.ttk.Button(toolbar, text="刷新账本", command=self._refresh_ledger).pack(side="right", padx=(6, 0))
        self.ttk.Button(toolbar, text="一致性重放", command=self._replay_verify).pack(side="right")
        paned = self.ttk.Panedwindow(tab, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        left = self.ttk.Frame(paned, style="Panel.TFrame")
        right = self.ttk.Frame(paned, style="Panel.TFrame")
        paned.add(left, weight=3)
        paned.add(right, weight=2)
        columns = ("batch", "seq", "cursor", "format", "digest", "snapshot")
        self.ledger_tree = self.ttk.Treeview(left, columns=columns, show="headings")
        headings = {"batch": "批次", "seq": "Seq", "cursor": "游标", "format": "格式", "digest": "内容摘要", "snapshot": "快照"}
        widths = {"batch": 65, "seq": 75, "cursor": 65, "format": 55, "digest": 185, "snapshot": 65}
        for column in columns:
            self.ledger_tree.heading(column, text=headings[column])
            self.ledger_tree.column(column, width=widths[column], anchor="w", stretch=column == "digest")
        scroll = self.ttk.Scrollbar(left, orient="vertical", command=self.ledger_tree.yview)
        self.ledger_tree.configure(yscrollcommand=scroll.set)
        self.ledger_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.ledger_detail = self.tk.Text(
            right,
            bg="#0f172a",
            fg="#dbeafe",
            insertbackground="#ffffff",
            font=(self.font_family, 9),
            wrap="word",
            padx=10,
            pady=8,
            relief="flat",
        )
        self.ledger_detail.pack(fill="both", expand=True)

    def _build_diagnostics_tab(self) -> None:
        tab = self.ttk.Frame(self.notebook)
        self.notebook.add(tab, text="诊断日志")
        toolbar = self.ttk.Frame(tab)
        toolbar.pack(fill="x", padx=4, pady=(5, 8))
        self.ttk.Label(toolbar, text="结构化 NDJSON 业务事件", font=(self.font_family, 11, "bold")).pack(side="left")
        self.ttk.Button(toolbar, text="刷新日志", command=self._refresh_logs).pack(side="right")
        self.log_text = self.tk.Text(
            tab,
            bg="#0b1220",
            fg="#cbd5e1",
            insertbackground="#ffffff",
            font=(self.font_family, 9),
            wrap="none",
            padx=10,
            pady=8,
            relief="flat",
        )
        xscroll = self.ttk.Scrollbar(tab, orient="horizontal", command=self.log_text.xview)
        yscroll = self.ttk.Scrollbar(tab, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        self.log_text.pack(fill="both", expand=True, padx=(4, 0), pady=(0, 0))
        yscroll.place(relx=1.0, rely=0.5, anchor="e", relheight=1.0)
        xscroll.pack(side="bottom", fill="x")
        self.log_text.tag_configure("error", foreground="#fca5a5")
        self.log_text.tag_configure("warn", foreground="#fcd34d")
        self.log_text.tag_configure("info", foreground="#93c5fd")

    def _build_statusbar(self) -> None:
        bar = self.tk.Frame(self.root, bg="#e8eef6", height=30)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)
        self.status_text = self.tk.Label(
            bar,
            text="启动本地工作台…",
            bg="#e8eef6",
            fg=MUTED,
            font=(self.font_family, 9),
            anchor="w",
        )
        self.status_text.pack(side="left", fill="x", expand=True, padx=14)
        self.status_right = self.tk.Label(
            bar,
            text="本地 SQLite · 单调时钟租约 · 无网络依赖",
            bg="#e8eef6",
            fg=MUTED,
            font=(self.font_family, 9),
            anchor="e",
        )
        self.status_right.pack(side="right", padx=14)

    # ------------------------------------------------------------------
    # Workbench lifecycle and background dispatch
    # ------------------------------------------------------------------
    def _open_workbench(self) -> None:
        self._set_status("正在校验基表、视图定义与变更流…")

        def worker() -> None:
            started = time.perf_counter()
            try:
                workbench = Workbench(
                    self.base_path,
                    self.views_path,
                    self.changes_path,
                    self.data_dir,
                    reset=self.reset,
                    start_watchdog=True,
                    auto_rebuild_on_expiry=self.auto_rebuild_on_expiry,
                )
                self.events.put(("opened", workbench, None, (time.perf_counter() - started) * 1000.0))
            except BaseException as exc:
                self.events.put(("opened", None, as_business_error(exc), (time.perf_counter() - started) * 1000.0))

        threading.Thread(target=worker, name="imv-gui-open", daemon=True).start()

    def _poll_events(self) -> None:
        drained = False
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            drained = True
            kind = event[0]
            if kind == "opened":
                _kind, workbench, error, elapsed = event
                if error is not None:
                    self._set_busy(False)
                    self._set_header_badge("初始化失败", RED_BG, RED)
                    self._set_status(f"{error.code}: {error.message}")
                    self._show_error(error)
                else:
                    self.workbench = workbench
                    self._set_header_badge("缓存就绪", GREEN_BG, GREEN)
                    self._set_status(f"工作台初始化完成，用时 {elapsed:.1f} ms。界面从增量缓存快速渲染。")
                    self._refresh_fast()
                    self._refresh_ledger()
                    self._refresh_logs()
                    self._fast_after_id = self.root.after(350, self._poll_fast)
            elif kind == "action":
                _kind, label, result, error, elapsed, dispatch_ms = event
                self._set_busy(False)
                if error is not None:
                    self._set_status(f"{label}失败 · {error.code} · {error.message}")
                    self._show_error(error)
                else:
                    self._set_status(f"{label}完成，用时 {elapsed:.1f} ms；界面受理 {dispatch_ms:.2f} ms。")
                self._refresh_fast()
                self._refresh_ledger()
                self._refresh_logs()
            elif kind == "closed":
                self.root.destroy()
                return
        if drained:
            pass
        self._event_after_id = self.root.after(120, self._poll_events)

    def _poll_fast(self) -> None:
        if self.closing:
            return
        if self.workbench is not None and not self.busy:
            try:
                self._refresh_fast()
            except Exception as exc:
                self._set_status(f"快速缓存刷新失败: {exc}")
        if not self.closing:
            self._fast_after_id = self.root.after(400, self._poll_fast)

    def _run_action(self, label: str, fn: Callable[[], Any]) -> None:
        if self.workbench is None:
            self._set_status("工作台尚未就绪")
            return
        if self.busy:
            self._set_status("已有后台操作正在执行；本次请求未排队。")
            return
        dispatch_started = time.perf_counter()
        self._set_busy(True)
        self._set_status(f"{label}已提交到后台线程…")
        dispatch_ms = (time.perf_counter() - dispatch_started) * 1000.0

        def worker() -> None:
            started = time.perf_counter()
            try:
                result = fn()
                self.events.put(("action", label, result, None, (time.perf_counter() - started) * 1000.0, dispatch_ms))
            except BaseException as exc:
                self.events.put(("action", label, None, as_business_error(exc), (time.perf_counter() - started) * 1000.0, dispatch_ms))

        threading.Thread(target=worker, name=f"imv-gui-{label}", daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        for button in self.action_buttons:
            try:
                button.configure(state="disabled" if busy else "normal")
            except Exception:
                pass
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _show_error(self, error: BusinessError) -> None:
        from tkinter import messagebox

        detail = error.message
        if error.location:
            detail += "\n\n位置: " + json.dumps(error.location, ensure_ascii=False, sort_keys=True)
        if error.details:
            detail += "\n\n依据: " + json.dumps(error.details, ensure_ascii=False, sort_keys=True, indent=2)[:1800]
        messagebox.showerror(f"{error.code} · 业务错误", detail, parent=self.root)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _advance_one(self) -> None:
        try:
            size = int(self.batch_size_var.get())
        except ValueError:
            self._set_status("批次行数必须是正整数")
            return
        self._run_action("推进变更批次", lambda: self.workbench.advance_batch(size))

    def _advance_all(self) -> None:
        self._run_action("推进全部变更", lambda: self.workbench.advance_all())

    def _authority_recompute(self) -> None:
        self._run_action("权威逐行重算", lambda: self.workbench.authoritative_recompute())

    def _replay_verify(self) -> None:
        self._run_action("一致性重放校验", lambda: self.workbench.replay_verify())

    def _selected_view_id(self) -> str | None:
        selection = self.view_tree.selection()
        if selection:
            return str(selection[0])
        if self.selected_view:
            return self.selected_view
        return None

    def _rebuild_selected(self) -> None:
        view_id = self._selected_view_id()
        if not view_id:
            self._set_status("请先选择一个视图")
            return
        self._run_action(f"重建 {view_id}", lambda: self.workbench.rebuild_view(view_id, "manual UI rebuild"))

    def _simulate_lease_expiry(self) -> None:
        view_id = self._selected_view_id()
        if not view_id:
            self._set_status("请先选择一个视图")
            return
        self._run_action(
            f"模拟 {view_id} 租约过期",
            lambda: self.workbench.simulate_lease_expiry(view_id, ttl_ms=180),
        )

    def _apply_definitions(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择新的视图定义 JSON",
            initialdir=str(Path(self.views_path).parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        self._run_action("热替换视图定义", lambda: self.workbench.apply_views_file(path))

    def _inject_anomaly(self) -> None:
        choice = self.inject_var.get()
        if choice.startswith("租约"):
            fn = lambda: self.workbench.inject_lease_conflict()
        elif choice.startswith("基表"):
            fn = lambda: self.workbench.inject_base_corruption()
        else:
            fn = lambda: self.workbench.inject_snapshot_tamper()
        self._run_action(f"异常注入 · {choice.split(' · ')[-1]}", fn)

    # ------------------------------------------------------------------
    # Fast-cache rendering
    # ------------------------------------------------------------------
    def _refresh_fast(self) -> None:
        if self.workbench is None:
            return
        snapshot = self.workbench.fast_snapshot()
        self.last_fast = snapshot
        views = snapshot["views"]
        scheduler = snapshot["scheduler"]
        visible = sum(1 for view in views if view["visible"])
        inconsistent = sum(1 for view in views if view["inconsistent"])
        running = scheduler.get("active_total", 0)
        pending = scheduler.get("pending_total", 0)
        self.kpi_labels["param_version"].configure(text=f"v{snapshot['param_version']}")
        self.kpi_labels["cursor"].configure(text=f"{snapshot['cursor']} / {snapshot['total_changes']}")
        self.kpi_labels["views"].configure(text=f"{visible} / {len(views)}")
        self.kpi_labels["jobs"].configure(text=f"{running} / {pending}")
        self.kpi_labels["snapshot"].configure(text=f"v{snapshot['snapshot']['snapshot_version']} · 1")
        self.live_labels["latency"].configure(text=f"{snapshot.get('render_latency_ms', 0):.2f} ms")
        self.live_labels["running"].configure(text=f"{running} / {pending}")
        self.live_labels["cursor"].configure(text=f"{snapshot['cursor']} / {snapshot['total_changes']}")
        self.live_labels["snapshot"].configure(text=f"v{snapshot['snapshot']['snapshot_version']}")

        if inconsistent:
            self._set_header_badge("租约待恢复", RED_BG, RED)
            self.header_detail.configure(text=f"{inconsistent} 个视图已隐藏，等待全量重建")
        elif running or pending or snapshot.get("recovering"):
            self._set_header_badge("刷新中", BLUE_BG, BLUE)
            self.header_detail.configure(text=f"运行 {running} · 排队 {pending} · 恢复 {len(snapshot.get('recovering', []))}")
        else:
            self._set_header_badge("缓存一致", GREEN_BG, GREEN)
            self.header_detail.configure(text=f"游标 {snapshot['cursor']} · 参数 v{snapshot['param_version']}")

        current_selection = self._selected_view_id()
        self.view_tree.delete(*self.view_tree.get_children())
        for view in views:
            lease = view.get("lease") or {}
            state = "不一致" if view["inconsistent"] else ("陈旧" if view["stale"] else ("刷新中" if lease.get("state") == "active" else ("可用" if view["visible"] else "隐藏")))
            tag = "bad" if view["inconsistent"] else ("warn" if view["stale"] else ("run" if lease.get("state") == "active" else "ok"))
            remaining = lease.get("remaining_ms")
            remaining_text = "—" if remaining is None else str(remaining)
            self.view_tree.insert(
                "",
                "end",
                iid=view["view_id"],
                values=(
                    view["view_id"],
                    ", ".join(view["group_by"]),
                    view["refresh_mode"],
                    view["result_version"],
                    lease.get("state", "—"),
                    remaining_text,
                    state,
                    view["group_count"],
                    view["result_digest"][:16] + "…",
                ),
                tags=(tag,),
            )
        ids = [view["view_id"] for view in views]
        if current_selection in ids:
            self.view_tree.selection_set(current_selection)
            self.selected_view = current_selection
        elif ids:
            self.selected_view = ids[0]
            self.view_tree.selection_set(ids[0])
        else:
            self.selected_view = None
        self._update_result_selector(ids)
        self._render_selected_groups()
        self._render_scheduler(scheduler)

    def _update_result_selector(self, ids: list[str]) -> None:
        values = tuple(ids)
        if values != self._last_view_values:
            self.result_view_combo.configure(values=values)
            self._last_view_values = values
        target = self.selected_view or (ids[0] if ids else "")
        if target and self.result_view_var.get() != target:
            self.result_view_var.set(target)

    def _render_selected_groups(self) -> None:
        if self.last_fast is None:
            return
        view_id = self.result_view_var.get() or self.selected_view
        entry = next((item for item in self.last_fast["views"] if item["view_id"] == view_id), None)
        if entry is None:
            self.group_tree.delete(*self.group_tree.get_children())
            return
        aliases = tuple(spec["alias"] for spec in entry.get("aggregates", []))
        signature = ("group",) + aliases
        if signature != self._last_aggregate_signature:
            self.group_tree.configure(columns=signature, show="headings")
            self.group_tree.heading("group", text="分组键")
            self.group_tree.column("group", width=240, anchor="w", stretch=True)
            for spec in entry.get("aggregates", []):
                alias = spec["alias"]
                self.group_tree.heading(alias, text=f"{alias}\n{spec['func']}({spec['column']})")
                self.group_tree.column(alias, width=145, anchor="e")
            self._last_aggregate_signature = signature
        self.group_tree.delete(*self.group_tree.get_children())
        if entry["visible"]:
            self.result_banner.configure(
                text=f"快速缓存可用 · 视图 {view_id} · cursor {entry['cursor']} · 摘要 {entry['result_digest'][:20]}…",
                bg=GREEN_BG,
                fg=GREEN,
            )
        else:
            self.result_banner.configure(
                text=f"结果不可展示 · {entry.get('hide_reason') or '租约或一致性问题'}",
                bg=RED_BG,
                fg=RED,
            )
        for group in entry.get("groups", []):
            key = " / ".join("NULL" if value is None else json.dumps(value, ensure_ascii=False) for value in group.get("key", []))
            values = (key,) + tuple(group.get("values", {}).get(alias) for alias in aliases)
            self.group_tree.insert("", "end", values=values)
        self.result_meta.configure(
            text=(
                f"scale={self.last_fast['fixed_point']['scale_factor']} · "
                f"truncation={self.last_fast['fixed_point']['truncation']} · "
                f"渲染 {self.last_fast.get('render_latency_ms', 0):.2f} ms · 未触发全量重算"
            )
        )

    def _render_scheduler(self, scheduler: dict[str, Any]) -> None:
        self.scheduler_tree.delete(*self.scheduler_tree.get_children())
        jobs = list(reversed(scheduler.get("recent_jobs", [])))
        for job in jobs[:120]:
            tag = "failed" if job.get("status") == "failed" else ("running" if job.get("status") in {"running", "queued"} else "")
            self.scheduler_tree.insert(
                "",
                "end",
                iid=job["job_id"],
                values=(
                    job["job_id"][:10],
                    job["view_id"],
                    job["kind"],
                    job["enqueued_mono_ms"],
                    "—" if job.get("wait_ms") is None else job["wait_ms"],
                    "—" if job.get("run_ms") is None else job["run_ms"],
                    job.get("status"),
                    job.get("error_code") or "—",
                    job.get("worker_thread") or "—",
                ),
                tags=(tag,) if tag else (),
            )
        self.scheduler_summary.configure(
            text=(
                f"全局上限 {scheduler.get('global_limit')} · 活跃 {scheduler.get('active_total')} · "
                f"排队 {scheduler.get('pending_total')} · 实测峰值 {scheduler.get('max_active_total_observed')} · "
                f"单视图峰值 {scheduler.get('max_active_by_view_observed')}"
            )
        )

    # ------------------------------------------------------------------
    # Ledger and diagnostics
    # ------------------------------------------------------------------
    def _refresh_ledger(self) -> None:
        if self.workbench is None:
            return
        batches = self.workbench.ledger_batches(limit=200)
        self.ledger_tree.delete(*self.ledger_tree.get_children())
        for batch in batches:
            self.ledger_tree.insert(
                "",
                "end",
                values=(
                    batch["batch_no"],
                    f"{batch['first_seq']}..{batch['last_seq']}",
                    batch["cursor"],
                    batch["ledger_format_version"],
                    batch["content_digest"][:24] + "…",
                    batch["snapshot_version"],
                ),
            )
        replacement = self.workbench.replacements(limit=1)
        rebuilds = self.workbench.rebuild_traces(limit=5)
        snapshot = self.workbench.store.snapshot_notice()
        detail = {
            "snapshot": snapshot,
            "latest_batch": batches[0] if batches else None,
            "latest_definition_replacement": replacement[0] if replacement else None,
            "recent_rebuild_traces": rebuilds,
            "fixed_point": {
                "scale_factor": self.workbench.definitions.session.scale_factor,
                "truncation": self.workbench.definitions.session.truncation,
                "overflow": self.workbench.definitions.session.overflow,
            },
        }
        self.ledger_detail.configure(state="normal")
        self.ledger_detail.delete("1.0", "end")
        self.ledger_detail.insert("1.0", json.dumps(detail, ensure_ascii=False, indent=2, sort_keys=True))
        self.ledger_detail.configure(state="disabled")

    def _refresh_logs(self) -> None:
        if self.workbench is None:
            return
        events = self.workbench.event_log(limit=350)
        self.log_text.delete("1.0", "end")
        for event in events:
            timestamp = event.get("ts_wall_ms", 0)
            stamp = _dt.datetime.fromtimestamp(timestamp / 1000.0).strftime("%H:%M:%S.%f")[:-3]
            code = event.get("code") or "----"
            location = event.get("location") or {}
            where = ""
            if location:
                where = " " + json.dumps(location, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            line = f"{stamp} {event.get('level', 'INFO'):5s} {code:11s} {event.get('event', ''):24s} {event.get('message', '')}{where}\n"
            tag = "error" if event.get("level") == "ERROR" else ("warn" if event.get("level") == "WARN" else "info")
            self.log_text.insert("end", line, tag)
        self.log_text.see("end")

    # ------------------------------------------------------------------
    # Navigation / status
    # ------------------------------------------------------------------
    def _on_view_selected(self, _event: Any = None) -> None:
        view_id = self._selected_view_id()
        if view_id:
            self.selected_view = view_id
            self.result_view_var.set(view_id)
            self._render_selected_groups()

    def _on_result_view_changed(self, _event: Any = None) -> None:
        view_id = self.result_view_var.get()
        if view_id:
            self.selected_view = view_id
            if self.view_tree.exists(view_id):
                self.view_tree.selection_set(view_id)
            self._render_selected_groups()

    def _on_tab_changed(self, _event: Any = None) -> None:
        try:
            index = self.notebook.index(self.notebook.select())
        except Exception:
            return
        if index == 3:
            self._refresh_ledger()
        elif index == 4:
            self._refresh_logs()

    def _set_status(self, text: str) -> None:
        self.status_text.configure(text=text)
        self.action_status.configure(text=text)

    def _set_header_badge(self, text: str, bg: str, fg: str) -> None:
        self.header_badge.configure(text=text, bg=bg, fg=fg)

    def _on_close(self) -> None:
        if self.closing:
            return
        self.closing = True
        for after_id in (self._event_after_id, self._fast_after_id):
            if after_id:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
        self._event_after_id = self.root.after(50, self._poll_events)
        self._set_status("正在关闭租约监视器与工作线程…")

        def worker() -> None:
            try:
                if self.workbench is not None:
                    self.workbench.close()
            finally:
                self.events.put(("closed",))

        threading.Thread(target=worker, name="imv-gui-close", daemon=True).start()
