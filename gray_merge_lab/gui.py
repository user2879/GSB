from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .errors import LabError
from .viewmodel import DiagnosticSession, SessionPaths


try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox, ttk

    TK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on host Tk libraries
    tk = None  # type: ignore[assignment]
    tkfont = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    TK_IMPORT_ERROR = exc


BG = "#eef2f5"
NAVY = "#12263a"
BLUE = "#1769aa"
GREEN = "#1f7a4d"
AMBER = "#9a6400"
RED = "#b3261e"
CARD = "#ffffff"
MUTED = "#5f6b76"


class GrayMergeDiagnosticsApp:
    def __init__(self, root: Any, session: DiagnosticSession) -> None:
        self.root = root
        self.session = session
        self.merge_map: dict[str, str] = {}
        self._configure_style()
        self._build_layout()
        self.refresh_all()
        if session.snapshot_error:
            self._set_authority_status("快照版本不匹配，已整体替换", AMBER)
            self._log(
                "E_SNAPSHOT 已触发整体替换：\n"
                + json.dumps(session.snapshot_error, ensure_ascii=False, indent=2)
            )

    def _configure_style(self) -> None:
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        default_font = ("Segoe UI", 10) if self.root.tk.call("tk", "windowingsystem") == "win32" else ("DejaVu Sans", 10)
        heading_font = (default_font[0], 11, "bold")
        title_font = (default_font[0], 19, "bold")
        style.configure(".", font=default_font, background=BG, foreground="#1c2733")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD, relief="flat")
        style.configure("TLabel", background=BG, foreground="#1c2733")
        style.configure("Card.TLabel", background=CARD, foreground="#1c2733")
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("CardMuted.TLabel", background=CARD, foreground=MUTED)
        style.configure("Heading.TLabel", font=heading_font)
        style.configure("CardHeading.TLabel", background=CARD, font=heading_font)
        style.configure("TButton", padding=(10, 5))
        style.configure("Accent.TButton", padding=(12, 6), foreground="#ffffff", background=BLUE)
        style.map("Accent.TButton", background=[("active", "#0f588f")])
        style.configure("Treeview", rowheight=25, background=CARD, fieldbackground=CARD)
        style.configure("Treeview.Heading", font=(default_font[0], 9, "bold"))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(16, 7), font=(default_font[0], 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", CARD), ("!selected", "#dde5eb")])
        self.title_font = title_font

    def _build_layout(self) -> None:
        self.root.title("推理网关灰度回执归并诊断台")
        self.root.geometry("1480x900")
        self.root.minsize(1120, 680)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg=NAVY, height=92)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="推理网关 · 灰度回执归并诊断台",
            bg=NAVY,
            fg="#ffffff",
            font=self.title_font,
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(16, 2))
        tk.Label(
            header,
            text="稳定哈希分流  |  h1/h2/h3 回执归并  |  NumPy 本地前向  |  整数定点误差界  |  SQLite 权威账本",
            bg=NAVY,
            fg="#b9cbe0",
            font=("Segoe UI" if self.root.tk.call("tk", "windowingsystem") == "win32" else "DejaVu Sans", 10),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", padx=24)
        self.env_badge = tk.Label(
            header,
            text="加载中",
            bg="#1f3a55",
            fg="#d7e6f3",
            padx=12,
            pady=6,
        )
        self.env_badge.grid(row=0, column=1, rowspan=2, sticky="e", padx=24)

        toolbar = ttk.Frame(self.root, padding=(18, 10, 18, 8))
        toolbar.grid(row=1, column=0, sticky="ew")
        for column in range(9):
            toolbar.grid_columnconfigure(column, weight=0)
        toolbar.grid_columnconfigure(8, weight=1)
        self._toolbar_button(toolbar, "刷新快速视图", self.action_refresh_quick, 0)
        self._toolbar_button(toolbar, "运行剩余样本", self.action_run_remaining, 1)
        self._toolbar_button(toolbar, "热更新 + 重推理", self.action_hot_update, 2)
        self._toolbar_button(toolbar, "权威重算", self.action_authoritative, 3, accent=True)
        self._toolbar_button(toolbar, "恢复重放", self.action_recover, 4)
        self._toolbar_button(toolbar, "异常矩阵", self.action_anomalies, 5)
        self._toolbar_button(toolbar, "重置会话", self.action_reset, 6)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 8))
        self.tab_overview = ttk.Frame(self.notebook)
        self.tab_merge = ttk.Frame(self.notebook)
        self.tab_fixed = ttk.Frame(self.notebook)
        self.tab_meta = ttk.Frame(self.notebook)
        self.tab_anomaly = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_overview, text="运行总览")
        self.notebook.add(self.tab_merge, text="归并详情")
        self.notebook.add(self.tab_fixed, text="定点与权威复核")
        self.notebook.add(self.tab_meta, text="账本与快照")
        self.notebook.add(self.tab_anomaly, text="异常注入")

        self._build_overview()
        self._build_merge()
        self._build_fixed()
        self._build_meta()
        self._build_anomaly()

        status = tk.Frame(self.root, bg="#dce5ec", height=34)
        status.grid(row=3, column=0, sticky="ew")
        status.grid_propagate(False)
        status.grid_columnconfigure(1, weight=1)
        self.quick_status = tk.Label(
            status,
            text="快速视图：SNAPSHOT",
            bg=GREEN,
            fg="#ffffff",
            padx=10,
            pady=4,
            font=("Segoe UI" if self.root.tk.call("tk", "windowingsystem") == "win32" else "DejaVu Sans", 9, "bold"),
        )
        self.quick_status.grid(row=0, column=0, padx=(18, 8), pady=5)
        self.authority_status = tk.Label(
            status,
            text="权威状态：尚未重算",
            bg="#8a96a1",
            fg="#ffffff",
            padx=10,
            pady=4,
            font=("Segoe UI" if self.root.tk.call("tk", "windowingsystem") == "win32" else "DejaVu Sans", 9, "bold"),
        )
        self.authority_status.grid(row=0, column=1, sticky="w")
        self.status_text = tk.Label(
            status,
            text="",
            bg="#dce5ec",
            fg="#33414d",
            anchor="e",
            padx=18,
        )
        self.status_text.grid(row=0, column=2, sticky="e")

    def _toolbar_button(
        self,
        parent: Any,
        text: str,
        command: Callable[[], None],
        column: int,
        accent: bool = False,
    ) -> None:
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if accent else "TButton")
        button.grid(row=0, column=column, padx=(0, 8))

    def _card(self, parent: Any, title: str, subtitle: str | None = None) -> tuple[Any, Any]:
        frame = tk.Frame(parent, bg=CARD, highlightbackground="#d6dee5", highlightthickness=1)
        tk.Label(frame, text=title, bg=CARD, fg=NAVY, font=("Segoe UI" if self.root.tk.call("tk", "windowingsystem") == "win32" else "DejaVu Sans", 12, "bold"), anchor="w").pack(
            fill="x", padx=14, pady=(11, 2)
        )
        if subtitle:
            tk.Label(frame, text=subtitle, bg=CARD, fg=MUTED, anchor="w", justify="left").pack(
                fill="x", padx=14, pady=(0, 7)
            )
        body = tk.Frame(frame, bg=CARD)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return frame, body

    def _tree(self, parent: Any, columns: list[tuple[str, str, int]]) -> tuple[Any, Any]:
        container = tk.Frame(parent, bg=CARD)
        tree = ttk.Treeview(container, columns=[item[0] for item in columns], show="headings")
        for key, label, width in columns:
            tree.heading(key, text=label)
            tree.column(key, width=width, minwidth=max(50, width // 2), anchor="w", stretch=True)
        yscroll = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        xscroll = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        return container, tree

    def _build_overview(self) -> None:
        self.tab_overview.grid_rowconfigure(0, weight=0)
        self.tab_overview.grid_rowconfigure(1, weight=1)
        self.tab_overview.grid_columnconfigure(0, weight=1)

        rule_card, rule_body = self._card(
            self.tab_overview,
            "规则级快速视图",
            "仅读取 snapshot.json；不访问账本重放逻辑。来源列固定标记 snapshot。",
        )
        rule_card.grid(row=0, column=0, sticky="nsew", pady=(4, 8))
        rule_tree_frame, self.rule_tree = self._tree(
            rule_body,
            [
                ("rule", "规则", 110),
                ("source", "来源", 90),
                ("batches", "批数", 70),
                ("stable", "稳定版 批/样本", 130),
                ("canary", "灰度版 批/样本", 130),
                ("raw_ms", "原始耗时和 ms", 120),
                ("avg_dur", "平均耗时 q", 110),
                ("avg_samples", "平均样本 q", 110),
                ("error", "误差影响", 90),
            ],
        )
        rule_tree_frame.pack(fill="both", expand=True)
        self.rule_tree.configure(height=4)

        batch_card, batch_body = self._card(
            self.tab_overview,
            "批次快速视图",
            "批次划分按规则、去向、客户端键与声明批大小规范化；双击批次进入归并详情。",
        )
        batch_card.grid(row=1, column=0, sticky="nsew")
        batch_tree_frame, self.batch_tree = self._tree(
            batch_body,
            [
                ("index", "#", 50),
                ("batch", "批次标识", 330),
                ("rule", "规则", 90),
                ("dest", "去向", 80),
                ("bucket", "分桶", 70),
                ("client", "客户端键", 145),
                ("samples", "样本", 55),
                ("duration", "耗时 q", 95),
                ("sample_q", "样本 q", 95),
                ("param", "参数版本", 85),
                ("seq", "归并序号", 85),
                ("digest", "输出摘要", 260),
            ],
        )
        batch_tree_frame.pack(fill="both", expand=True)
        self.batch_tree.bind("<<TreeviewSelect>>", self._on_batch_selected)
        self.batch_tree.bind("<Double-1>", lambda event: self.notebook.select(self.tab_merge))

    def _build_merge(self) -> None:
        self.tab_merge.grid_rowconfigure(1, weight=1)
        self.tab_merge.grid_rowconfigure(2, weight=1)
        self.tab_merge.grid_columnconfigure(0, weight=1)

        selector = ttk.Frame(self.tab_merge, padding=(4, 8, 4, 6))
        selector.grid(row=0, column=0, sticky="ew")
        ttk.Label(selector, text="批次：").pack(side="left")
        self.merge_combo = ttk.Combobox(selector, state="readonly", width=62)
        self.merge_combo.pack(side="left", padx=(0, 10))
        self.merge_combo.bind("<<ComboboxSelected>>", self._on_merge_selected)
        self.merge_summary = tk.Label(
            selector,
            text="",
            bg=BG,
            fg=NAVY,
            anchor="w",
            justify="left",
        )
        self.merge_summary.pack(side="left", fill="x", expand=True)

        candidates_card, candidates_body = self._card(
            self.tab_merge,
            "候选回执与稳定采纳",
            "比较顺序：merge_seq 升序、参数版本降序、回执摘要升序；采纳最高归并序号，再取最高参数版本。",
        )
        candidates_card.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        candidate_frame, self.candidate_tree = self._tree(
            candidates_body,
            [
                ("format", "格式", 65),
                ("seq", "归并序号", 85),
                ("param", "参数版本", 85),
                ("dest", "去向", 75),
                ("result", "结果", 70),
                ("reason", "否决理由", 230),
                ("receipt", "回执摘要", 330),
            ],
        )
        candidate_frame.pack(fill="both", expand=True)

        discrepancies_card, discrepancies_body = self._card(
            self.tab_merge,
            "样本差异与回执对照",
            "被否决回执与采纳回执不一致的样本标识逐条列出；未知字段保存在于待补齐候选。",
        )
        discrepancies_card.grid(row=2, column=0, sticky="nsew")
        self.merge_text = tk.Text(
            discrepancies_body,
            height=8,
            bg="#f8fafb",
            fg="#24313d",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
        )
        self.merge_text.pack(fill="both", expand=True)

    def _build_fixed(self) -> None:
        self.tab_fixed.grid_rowconfigure(1, weight=1)
        self.tab_fixed.grid_columnconfigure(0, weight=1)
        info_card, info_body = self._card(
            self.tab_fixed,
            "整数定点口径",
            "缩放 S=10^scale_exp；除法向零截断。单值误差 < 1/S；规范聚合先累加原始整数，仅一次截断。",
        )
        info_card.grid(row=0, column=0, sticky="ew", pady=(4, 8))
        self.fixed_info = tk.Label(
            info_body,
            text="",
            bg=CARD,
            fg="#24313d",
            anchor="w",
            justify="left",
        )
        self.fixed_info.pack(fill="x")

        batches_card, batches_body = self._card(
            self.tab_fixed,
            "逐批原始整数与定点值",
            "此表来自快照；点击“权威重算”后，状态栏会切换为账本重放结论。",
        )
        batches_card.grid(row=1, column=0, sticky="nsew")
        fixed_frame, self.fixed_tree = self._tree(
            batches_body,
            [
                ("index", "#", 50),
                ("batch", "批次", 330),
                ("raw_ms", "原始耗时 ms", 115),
                ("dur_q", "耗时 q", 110),
                ("raw_samples", "原始样本数", 115),
                ("samples_q", "样本 q", 110),
                ("scale", "缩放", 80),
                ("bound", "单值误差界", 115),
            ],
        )
        fixed_frame.pack(fill="both", expand=True)
        self.authority_output = tk.Text(
            batches_body,
            height=5,
            bg="#f8fafb",
            fg="#24313d",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
        )
        self.authority_output.pack(fill="x", pady=(8, 0))

    def _build_meta(self) -> None:
        self.tab_meta.grid_rowconfigure(0, weight=1)
        self.tab_meta.grid_columnconfigure(0, weight=1)
        card, body = self._card(
            self.tab_meta,
            "账本、快照与运行环境",
            "账本为权威源；快照仅用于快速视图。版本/摘要不一致时快照整体替换，并产生 E_SNAPSHOT。",
        )
        card.grid(row=0, column=0, sticky="nsew", pady=4)
        self.meta_text = tk.Text(
            body,
            bg="#f8fafb",
            fg="#24313d",
            relief="flat",
            wrap="none",
            font=("Consolas", 9),
        )
        xscroll = ttk.Scrollbar(body, orient="horizontal", command=self.meta_text.xview)
        self.meta_text.configure(xscrollcommand=xscroll.set)
        self.meta_text.pack(fill="both", expand=True)
        xscroll.pack(fill="x")

    def _build_anomaly(self) -> None:
        self.tab_anomaly.grid_rowconfigure(1, weight=1)
        self.tab_anomaly.grid_columnconfigure(0, weight=1)
        top = ttk.Frame(self.tab_anomaly, padding=(4, 8, 4, 6))
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            top,
            text="异常注入只作用于临时副本或不可变输入，不污染当前账本。",
        ).pack(side="left")
        ttk.Button(top, text="运行完整异常矩阵", command=self.action_anomalies).pack(side="right")
        ttk.Button(top, text="仅看摘要损坏", command=lambda: self._focus_anomaly("receipt_digest_corruption")).pack(
            side="right", padx=6
        )
        ttk.Button(top, text="仅看参数字段缺失", command=lambda: self._focus_anomaly("missing_backend_parameter_field")).pack(
            side="right", padx=6
        )
        ttk.Button(top, text="仅看快照摘要不符", command=lambda: self._focus_anomaly("snapshot_digest_mismatch")).pack(
            side="right", padx=6
        )

        card, body = self._card(
            self.tab_anomaly,
            "异常分类结果",
            "覆盖 E_FORMAT、E_PREVIEW、E_REPLAY、E_SNAPSHOT；每项给出稳定首个定位字段。",
        )
        card.grid(row=1, column=0, sticky="nsew")
        frame, self.anomaly_tree = self._tree(
            body,
            [
                ("label", "场景", 250),
                ("expected", "期望", 100),
                ("caught", "实际", 100),
                ("ok", "通过", 65),
                ("field", "首个字段", 230),
                ("batch", "批次", 260),
                ("message", "说明", 360),
            ],
        )
        frame.pack(fill="both", expand=True)
        self.anomaly_text = tk.Text(
            body,
            height=7,
            bg="#f8fafb",
            fg="#24313d",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
        )
        self.anomaly_text.pack(fill="x", pady=(8, 0))

    def _set_authority_status(self, text: str, color: str) -> None:
        self.authority_status.configure(text=f"权威状态：{text}", bg=color)

    def _set_status_text(self, text: str) -> None:
        self.status_text.configure(text=text)

    def _log(self, text: str) -> None:
        self._set_status_text(text.splitlines()[0][:180])
        if hasattr(self, "authority_output"):
            self.authority_output.delete("1.0", "end")
            self.authority_output.insert("end", text)

    def _guard(self, operation: Callable[[], Any], success_message: str | None = None) -> Any:
        try:
            result = operation()
            if success_message:
                self._set_status_text(success_message)
            return result
        except LabError as exc:
            self._set_authority_status(f"{exc.code} @ {exc.field or exc.batch_id or 'unknown'}", RED)
            self._log(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2))
            messagebox.showerror(exc.code, str(exc), parent=self.root)
            return None
        except Exception as exc:
            self._set_authority_status("E_RUNTIME", RED)
            self._log(f"{type(exc).__name__}: {exc}")
            messagebox.showerror("E_RUNTIME", f"{type(exc).__name__}: {exc}", parent=self.root)
            return None

    def refresh_all(self) -> None:
        self._refresh_environment_badge()
        self._refresh_overview()
        self._refresh_merge_list()
        self._refresh_fixed()
        self._refresh_meta()
        self._refresh_anomalies()

    def _refresh_environment_badge(self) -> None:
        env = self.session.environment()
        tk_state = "Tk OK" if env["tkinter"].get("available") else "Tk 不可用"
        self.env_badge.configure(
            text=f"Python {env['python'].split()[0]}  |  NumPy {env['numpy']}  |  {tk_state}"
        )

    def _refresh_overview(self) -> None:
        for tree in (self.rule_tree, self.batch_tree):
            for item in tree.get_children():
                tree.delete(item)
        for row in self.session.quick_rows():
            self.rule_tree.insert(
                "",
                "end",
                values=(
                    row["rule_id"],
                    row["source"],
                    row["batches"],
                    f"{row['stable_batches']} / {row['stable_samples']}",
                    f"{row['canary_batches']} / {row['canary_samples']}",
                    row["raw_duration_sum_ms"],
                    row["avg_duration_q"],
                    row["avg_samples_q"],
                    "是" if row["sampling_error_affected"] else "否",
                ),
            )
        self.merge_map = {}
        for batch in self.session.preview_batches():
            conclusion = self.session.merge_detail(batch["batch_id"])
            seq = conclusion["merge_seq"] if conclusion else ""
            item_id = self.batch_tree.insert(
                "",
                "end",
                values=(
                    batch.get("batch_index", ""),
                    batch["batch_id"],
                    batch["rule_id"],
                    batch["destination"],
                    batch.get("client_bucket", ""),
                    batch.get("client_key", ""),
                    batch.get("raw_sample_count", ""),
                    batch.get("duration_q", ""),
                    batch.get("samples_q", ""),
                    batch.get("backend_param_version", ""),
                    seq,
                    self._short(batch.get("output_digest", ""), 44),
                ),
            )
            self.merge_map[item_id] = batch["batch_id"]
        progress = self.session.quick_payload.get("progress", {})
        self.quick_status.configure(
            text=f"快速视图：SNAPSHOT · {progress.get('completed_batches', 0)} 批"
        )

    def _refresh_merge_list(self) -> None:
        conclusions = self.session.merge_conclusions()
        batch_ids = [item["batch_id"] for item in conclusions]
        self.merge_combo["values"] = batch_ids
        if batch_ids and self.merge_combo.get() not in batch_ids:
            self.merge_combo.set(batch_ids[0])
        if self.merge_combo.get():
            self._show_merge(self.merge_combo.get())

    def _refresh_fixed(self) -> None:
        for item in self.fixed_tree.get_children():
            self.fixed_tree.delete(item)
        for batch in self.session.preview_batches():
            self.fixed_tree.insert(
                "",
                "end",
                values=(
                    batch.get("batch_index", ""),
                    batch["batch_id"],
                    batch.get("raw_duration_ms", ""),
                    batch.get("duration_q", ""),
                    batch.get("raw_sample_count", ""),
                    batch.get("samples_q", ""),
                    batch.get("scale_exp", ""),
                    batch.get("single_value_error_bound_original", ""),
                ),
            )
        scale_exp = self.session.quick_payload.get("scale_exp", self.session.scale_exp)
        stats_rows = self.session.quick_rows()
        raw = sum(int(item["raw_duration_sum_ms"]) for item in stats_rows)
        batches = sum(int(item["batches"]) for item in stats_rows)
        canonical_bound = 10.0 ** (-int(scale_exp))
        aggregate_bound = batches * canonical_bound
        self.fixed_info.configure(
            text=(
                f"scale_exp={scale_exp}   S=10^{scale_exp}={10 ** int(scale_exp)}   截断=toward_zero    "
                f"单值误差界<{canonical_bound}   批次聚合误差上界<{aggregate_bound}（{batches} 批独立再聚合）   "
                f"规范聚合误差界<{canonical_bound}   原始耗时和={raw} ms"
            )
        )

    def _refresh_meta(self) -> None:
        ledger_meta = self.session.ledger_meta()
        snapshot_meta = self.session.snapshot_meta()
        env = self.session.environment()
        payload = {
            "environment": env,
            "ledger": ledger_meta,
            "snapshot": snapshot_meta,
            "snapshot_error_on_load": self.session.snapshot_error,
            "quick_progress": self.session.quick_payload.get("progress", {}),
            "quick_source": self.session.quick_payload.get("source_ledger", {}),
        }
        self.meta_text.delete("1.0", "end")
        self.meta_text.insert("end", json.dumps(payload, ensure_ascii=False, indent=2))

    def _refresh_anomalies(self) -> None:
        for item in self.anomaly_tree.get_children():
            self.anomaly_tree.delete(item)
        for row in self.session.last_anomalies or []:
            self.anomaly_tree.insert(
                "",
                "end",
                values=(
                    row.get("label", ""),
                    row.get("expected", ""),
                    row.get("caught", ""),
                    "通过" if row.get("ok") else "失败",
                    row.get("field", ""),
                    row.get("batch_id", ""),
                    row.get("message", ""),
                ),
            )
        if self.session.last_anomalies:
            self.anomaly_text.delete("1.0", "end")
            self.anomaly_text.insert(
                "end", json.dumps(self.session.last_anomalies, ensure_ascii=False, indent=2)
            )

    def _show_merge(self, batch_id: str) -> None:
        conclusion = self.session.merge_detail(batch_id)
        for item in self.candidate_tree.get_children():
            self.candidate_tree.delete(item)
        if conclusion is None:
            self.merge_summary.configure(text="未找到归并结论")
            return
        adopted_id = conclusion["adopted_receipt"]["receipt_id"]
        rejected_by_id = {item["receipt_id"]: item for item in conclusion.get("rejected", [])}
        for candidate in conclusion.get("comparison_order", []):
            adopted = candidate["receipt_id"] == adopted_id
            rejected = rejected_by_id.get(candidate["receipt_id"])
            self.candidate_tree.insert(
                "",
                "end",
                values=(
                    candidate["format"],
                    candidate["merge_seq"],
                    candidate["param_version"],
                    candidate["destination"],
                    "采纳" if adopted else "否决",
                    "-" if adopted else (rejected or {}).get("rejection_reason", ""),
                    self._short(candidate["receipt_id"], 46),
                ),
            )
        self.merge_summary.configure(
            text=(
                f"去向 {conclusion['destination']}   |   样本 {len(conclusion['sample_ids'])}   |   "
                f"参数版本 {conclusion['backend_param_version']}   |   merge_seq {conclusion['merge_seq']}   |   "
                f"否决 {len(conclusion.get('rejected', []))} 条"
            )
        )
        detail = {
            "batch_id": conclusion["batch_id"],
            "sample_ids": conclusion["sample_ids"],
            "output_digest": conclusion["output_digest"],
            "backend_param_digest": conclusion["backend_param_digest"],
            "decision_rule": conclusion["decision_rule"],
            "arrival_order_independent": conclusion["arrival_order_independent"],
            "sample_discrepancies": [
                discrepancy
                for rejected in conclusion.get("rejected", [])
                for discrepancy in rejected.get("sample_discrepancies", [])
            ],
            "unknown_candidates": {
                rejected["receipt_id"]: rejected.get("unknown_fields", [])
                for rejected in conclusion.get("rejected", [])
                if rejected.get("unknown_fields")
            },
            "rejected_receipts": [
                {
                    "receipt_id": item["receipt_id"],
                    "format": item["format"],
                    "merge_seq": item["merge_seq"],
                    "param_version": item["param_version"],
                    "reason": item["rejection_reason"],
                    "destination_mismatch": item["destination_mismatch"],
                    "output_digest_mismatch": item["output_digest_mismatch"],
                }
                for item in conclusion.get("rejected", [])
            ],
        }
        self.merge_text.delete("1.0", "end")
        self.merge_text.insert("end", json.dumps(detail, ensure_ascii=False, indent=2))

    def _on_batch_selected(self, event: Any = None) -> None:
        del event
        selection = self.batch_tree.selection()
        if not selection:
            return
        batch_id = self.merge_map.get(selection[0])
        if batch_id:
            self.merge_combo.set(batch_id)
            self._show_merge(batch_id)

    def _on_merge_selected(self, event: Any = None) -> None:
        del event
        if self.merge_combo.get():
            self._show_merge(self.merge_combo.get())

    def _focus_anomaly(self, label: str) -> None:
        if not self.session.last_anomalies:
            self.action_anomalies()
        for index, row in enumerate(self.session.last_anomalies or []):
            if row.get("label") == label:
                children = self.anomaly_tree.get_children()
                if index < len(children):
                    self.anomaly_tree.selection_set(children[index])
                    self.anomaly_tree.see(children[index])
                self._log(json.dumps(row, ensure_ascii=False, indent=2))
                break

    def action_refresh_quick(self) -> None:
        result = self._guard(self.session.refresh_quick, "快速视图已从 snapshot.json 刷新")
        if result is not None:
            self.refresh_all()

    def action_run_remaining(self) -> None:
        rows = self._guard(self.session.run_remaining_samples)
        if rows is not None:
            self._set_status_text(f"运行剩余样本完成：{len(rows)} 批")
            self.refresh_all()

    def action_hot_update(self) -> None:
        result = self._guard(self.session.hot_update_and_reinfer)
        if result is not None:
            self._set_authority_status("热更新后尚未权威重算", AMBER)
            self._log(json.dumps(result, ensure_ascii=False, indent=2))
            self.refresh_all()

    def action_authoritative(self) -> None:
        result = self._guard(self.session.recompute_authoritative)
        if result is not None:
            self._set_authority_status(
                f"已与账本一致 · {result['completed_batches']} 批", GREEN
            )
            self._log(
                "权威重算通过\n"
                + json.dumps(
                    {
                        "completed_batches": result["completed_batches"],
                        "ledger_content_digest": result["ledger_content_digest"],
                        "fixed_scale_exp": result["fixed_scale_exp"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            self._refresh_fixed()

    def action_recover(self) -> None:
        result = self._guard(self.session.recover_from_checkpoint)
        if result is not None:
            self._set_status_text(
                f"恢复重放通过：从 {result['resumed_after_batch_index']} 后推进 {result['replayed_batches']} 批"
            )
            self._log(
                json.dumps(
                    {
                        "source": result["source"],
                        "resumed_after_batch_index": result["resumed_after_batch_index"],
                        "replayed_batches": result["replayed_batches"],
                        "batch_ids": result["batch_ids"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            self.refresh_all()

    def action_anomalies(self) -> None:
        result = self._guard(self.session.inject_anomalies)
        if result is not None:
            passed = sum(1 for item in result if item.get("ok"))
            self._set_status_text(f"异常矩阵：{passed}/{len(result)} 通过")
            self._refresh_anomalies()

    def action_reset(self) -> None:
        if not messagebox.askyesno(
            "重置会话",
            "将删除当前会话的 SQLite 账本、快照和检查点，并重新生成演示数据。继续？",
            parent=self.root,
        ):
            return
        result = self._guard(self.session.reset, "会话已重置并重新生成")
        if result is not None:
            self._set_authority_status("尚未重算", "#8a96a1")
            self.refresh_all()

    @staticmethod
    def _short(value: str, maximum: int) -> str:
        if len(value) <= maximum:
            return value
        return value[: maximum - 1] + "…"


def launch_gui(workdir: str | Path = "artifacts/app") -> int:
    if TK_IMPORT_ERROR is not None:
        print(
            "E_RUNTIME: tkinter is unavailable: "
            f"{type(TK_IMPORT_ERROR).__name__}: {TK_IMPORT_ERROR}"
        )
        print("Install the operating-system Tk runtime, then rerun: python -m gray_merge_lab gui")
        return 2
    session = DiagnosticSession(SessionPaths(workdir=Path(workdir))).initialize()
    root = tk.Tk()
    app = GrayMergeDiagnosticsApp(root, session)

    def on_close() -> None:
        session.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_gui())

