from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .config import default_config_dict, load_config
from .errors import SlowDiagError
from .generator import generate_all
from .logging_utils import StructuredLogger
from .pipeline import run_pipeline
from .snapshots import create_snapshot, restore_snapshot
from .storage import Ledger


class WorkbenchApp:
    def __init__(self, root: tk.Tk, workspace: Path, config_path: Path):
        self.root = root
        self.workspace = workspace.resolve(strict=False)
        self.config_path = config_path.resolve(strict=False)
        self.logger = StructuredLogger(self.workspace)
        self.view: dict[str, Any] | None = None
        self.row_payloads: dict[str, dict[str, Any]] = {}
        self.root.title("多源慢查询证据诊断工作台")
        self.root.geometry("1320x820")
        self.root.minsize(980, 640)
        self._build_ui()
        self._load_current_view()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        buttons = (
            ("初始化", self.action_init),
            ("生成样本", self.action_generate),
            ("运行诊断", self.action_run),
            ("创建快照", self.action_snapshot),
            ("恢复最新快照", self.action_restore_latest),
            ("刷新视图", self._load_current_view),
        )
        for text, command in buttons:
            ttk.Button(toolbar, text=text, command=command).pack(
                side=tk.LEFT, padx=(0, 6)
            )
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        paned = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.notebook = ttk.Notebook(paned)
        paned.add(self.notebook, weight=4)
        self.detail = tk.Text(paned, height=12, wrap=tk.NONE)
        self.detail.configure(font=("Consolas", 10))
        paned.add(self.detail, weight=2)

        self.sources_tree = self._make_tree(
            "观测源",
            (
                ("source_id", "源标识", 150),
                ("source_type", "来源类型", 160),
                ("format_version", "声明版本", 90),
                ("format_versions", "记录版本集合", 120),
                ("sample_rows", "样本行数", 90),
                ("sample_period_ms", "采样周期(ms)", 115),
                ("elapsed_semantics", "耗时口径", 95),
                ("scan_semantics", "扫描口径", 95),
                ("content_sha256", "内容摘要", 330),
            ),
        )
        self.conclusions_tree = self._make_tree(
            "合并结论",
            (
                ("query_id", "查询", 100),
                ("slow", "慢查询", 80),
                ("elapsed", "耗时采纳(ms)", 140),
                ("elapsed_source", "耗时源", 140),
                ("scans", "扫描行采纳", 150),
                ("scans_source", "扫描源", 140),
                ("basis", "诊断依据", 360),
            ),
        )
        self.migration_tree = self._make_tree(
            "迁移复核",
            (
                ("pair_id", "迁移对", 220),
                ("query_id", "查询", 90),
                ("status", "判定", 130),
                ("a_slow", "源A慢", 80),
                ("b_slow", "源B慢", 80),
                ("conflicts", "冲突指标", 260),
                ("reasons", "依据", 320),
            ),
        )
        self.attribution_tree = self._make_tree(
            "冲突归因",
            (
                ("pair_id", "迁移对", 210),
                ("query_id", "查询", 90),
                ("metric", "指标", 110),
                ("delta", "偏差量", 100),
                ("semantics", "口径贡献", 130),
                ("format_version", "格式版本贡献", 150),
                ("sample_period", "采样周期贡献", 150),
            ),
        )

        invalidation_frame = ttk.Frame(self.notebook)
        self.invalidation_text = tk.Text(invalidation_frame, wrap=tk.WORD)
        self.invalidation_text.pack(fill=tk.BOTH, expand=True)
        self.invalidation_text.configure(font=("Consolas", 10))
        self.notebook.add(invalidation_frame, text="失效传播")

    def _make_tree(
        self, title: str, columns: tuple[tuple[str, str, int], ...]
    ) -> ttk.Treeview:
        frame = ttk.Frame(self.notebook)
        tree = ttk.Treeview(
            frame,
            columns=tuple(column[0] for column in columns),
            show="headings",
            selectmode="browse",
        )
        for key, label, width in columns:
            tree.heading(key, text=label)
            tree.column(key, width=width, anchor=tk.W, stretch=True)
        vertical = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self.notebook.add(frame, text=title)
        tree.bind("<<TreeviewSelect>>", self._show_selected_detail)
        return tree

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _ensure_config(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.config_path.exists():
            self.config_path.write_text(
                json.dumps(default_config_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, SlowDiagError):
            message = f"{exc.code}: {exc.message}"
            self.logger.log(
                "gui_error",
                code=exc.code,
                severity="error",
                message=exc.message,
                details=exc.to_dict(),
            )
        else:
            message = str(exc)
            self.logger.log(
                "gui_error",
                code="E_INTERNAL",
                severity="critical",
                message=str(exc),
            )
        self._set_status(message)
        messagebox.showerror("诊断工作台", message, parent=self.root)

    def action_init(self) -> None:
        try:
            self._ensure_config()
            config = load_config(self.config_path, self.workspace)
            for source in config.sources:
                source.resolved_data_path.parent.mkdir(parents=True, exist_ok=True)
            Ledger(self.workspace).initialize()
            self._set_status(f"工作区已初始化：{self.workspace}")
        except Exception as exc:
            self._handle_error(exc)

    def action_generate(self) -> None:
        try:
            self._ensure_config()
            config = load_config(self.config_path, self.workspace)
            summaries = generate_all(config)
            self._set_status(f"已生成 {len(summaries)} 个观测源样本")
            self.detail.delete("1.0", tk.END)
            self.detail.insert(tk.END, json.dumps(summaries, ensure_ascii=False, indent=2))
        except Exception as exc:
            self._handle_error(exc)

    def action_run(self) -> None:
        try:
            self._ensure_config()
            config = load_config(self.config_path, self.workspace)
            result = run_pipeline(config, Ledger(self.workspace))
            self.view = result.view
            self._render_view()
            self._set_status(
                f"方案版本 {result.view['plan_version']} 已完成，"
                f"查询 {len(result.view['queries'])} 条"
            )
        except Exception as exc:
            self._handle_error(exc)

    def action_snapshot(self) -> None:
        try:
            if self.view is None:
                self._load_current_view()
            if self.view is None:
                raise SlowDiagError(
                    "E_REPLAY", "当前没有结论视图，请先运行诊断", field="current_view"
                )
            config = load_config(self.config_path, self.workspace)
            path = create_snapshot(self.workspace, self.view, config)
            self._set_status(f"快照已创建：{path}")
        except Exception as exc:
            self._handle_error(exc)

    def action_restore_latest(self) -> None:
        try:
            snapshots = sorted(
                (self.workspace / "snapshots").glob("*.json"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if not snapshots:
                raise SlowDiagError("E_REPLAY", "没有可恢复的快照", field="snapshot")
            result = restore_snapshot(Ledger(self.workspace), snapshots[0], self.workspace)
            self._load_current_view()
            self._set_status(
                f"已恢复 {snapshots[0].name}；差异 {result['snapshot_diff_count']} 项"
            )
        except Exception as exc:
            if isinstance(exc, SlowDiagError) and exc.code == "E_REPLAY":
                self._load_current_view()
            self._handle_error(exc)

    def _load_current_view(self) -> None:
        try:
            current = Ledger(self.workspace).get_current_view()
            if current is None:
                self.view = None
                self._set_status("尚无结论视图；请生成样本并运行诊断")
                return
            self.view = current["view"]
            self._render_view()
            self._set_status(
                f"已加载方案版本 {current['plan_version']}（{current['view_sha256'][:12]}）"
            )
        except Exception as exc:
            self._handle_error(exc)

    def _clear_payloads(self) -> None:
        self.row_payloads.clear()
        for tree in (
            self.sources_tree,
            self.conclusions_tree,
            self.migration_tree,
            self.attribution_tree,
        ):
            for item in tree.get_children():
                tree.delete(item)

    def _insert(
        self, tree: ttk.Treeview, values: tuple[Any, ...], payload: dict[str, Any]
    ) -> None:
        item = tree.insert("", tk.END, values=values)
        self.row_payloads[item] = payload

    def _render_view(self) -> None:
        if self.view is None:
            return
        self._clear_payloads()
        for source in self.view.get("source_summaries", []):
            self._insert(
                self.sources_tree,
                (
                    source["source_id"],
                    source["source_type"],
                    source["format_version"],
                    ",".join(str(value) for value in source["format_versions"]),
                    source["sample_rows"],
                    source["sample_period_ms"],
                    source["elapsed_semantics"],
                    source["scan_semantics"],
                    source["content_sha256"][:24] + "…",
                ),
                source,
            )
        for query in self.view.get("queries", []):
            elapsed = query["metrics"]["elapsed_ms"]
            scans = query["metrics"]["scanned_rows"]
            self._insert(
                self.conclusions_tree,
                (
                    query["query_id"],
                    "是" if query["diagnosis"]["slow"] else "否",
                    elapsed["adopted"]["value"],
                    elapsed["adopted"]["source_id"],
                    scans["adopted"]["value"],
                    scans["adopted"]["source_id"],
                    "; ".join(query["diagnosis"]["basis"]),
                ),
                query,
            )
        for pair in self.view.get("migration", {}).get("pairs", []):
            for result in pair["results"]:
                self._insert(
                    self.migration_tree,
                    (
                        pair["pair_id"],
                        result["query_id"],
                        result["status"],
                        "是" if result["source_a"]["slow"] else "否",
                        "是" if result["source_b"]["slow"] else "否",
                        ", ".join(item["metric"] for item in result["conflicts"]),
                        "; ".join(result["reasons"]),
                    ),
                    result,
                )
        for result in self.view.get("attribution", {}).get("results", []):
            for conflict in result["conflicts"]:
                causes = {item["cause"]: item for item in conflict["causes"]}
                self._insert(
                    self.attribution_tree,
                    (
                        result["pair_id"],
                        result["query_id"],
                        conflict["metric"],
                        conflict["delta"],
                        f"{causes['semantics']['share_bp'] / 100:.2f}%",
                        f"{causes['format_version']['share_bp'] / 100:.2f}%",
                        f"{causes['sample_period']['share_bp'] / 100:.2f}%",
                    ),
                    {"attribution": result, "conflict": conflict},
                )
        invalidation = self.view.get("invalidation", {})
        self.invalidation_text.delete("1.0", tk.END)
        self.invalidation_text.insert(
            tk.END, json.dumps(invalidation, ensure_ascii=False, indent=2)
        )

    def _show_selected_detail(self, _event: tk.Event) -> None:
        notebook = self.notebook
        current = notebook.select()
        index = notebook.index(current)
        trees = (
            self.sources_tree,
            self.conclusions_tree,
            self.migration_tree,
            self.attribution_tree,
        )
        if index < 0 or index >= len(trees):
            return
        selection = trees[index].selection()
        if not selection:
            return
        payload = self.row_payloads.get(selection[0])
        self.detail.delete("1.0", tk.END)
        self.detail.insert(
            tk.END, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        )


def run_gui(workspace: Path | str, config_path: Path | str) -> None:
    workspace_path = Path(workspace).resolve(strict=False)
    config = Path(config_path).resolve(strict=False)
    root = tk.Tk()
    WorkbenchApp(root, workspace_path, config)
    root.mainloop()
