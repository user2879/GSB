"""Tkinter desktop application for local plan editing and verification."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

try:  # Imported lazily by the CLI, but keep a clear diagnostic for Windows installs.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    _TK_IMPORT_ERROR: Exception | None = None
except Exception as _error:  # pragma: no cover - depends on the host install
    _TK_IMPORT_ERROR = _error


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class PlannerApplication:
    def __init__(
        self,
        root: Any,
        engine: Any,
        *,
        workspace_path: str,
        ledger_path: str,
        snapshot_dir: str,
        log_path: str,
    ) -> None:
        self.root = root
        self.engine = engine
        self.workspace_path = workspace_path
        self.ledger_path = ledger_path
        self.snapshot_dir = snapshot_dir
        self.log_path = log_path
        self.active_query = engine.workspace.queries[0].name
        self.pending_verification_batch: int | None = None
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._last_ledger_cursor = self.engine.ledger.latest_cursor(self.engine.workspace_id)
        self._build_widgets()
        self._load_query_into_editor()
        self._load_tables_into_editor()
        self.refresh_view(update_ledger_panels=False)
        self._render_recovery()
        self._poll_worker_queue()

    def _build_widgets(self) -> None:
        self.root.title("SQL 逻辑计划本地编辑器 · 增量一致性")
        self.root.geometry("1320x900")
        self.root.minsize(1040, 720)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(toolbar, text="查询").pack(side=tk.LEFT)
        self.query_var = tk.StringVar(value=self.active_query)
        self.query_combo = ttk.Combobox(
            toolbar,
            textvariable=self.query_var,
            values=[query.name for query in self.engine.workspace.queries],
            state="readonly",
            width=18,
        )
        self.query_combo.pack(side=tk.LEFT, padx=(6, 8))
        self.query_combo.bind("<<ComboboxSelected>>", self._on_query_changed)
        ttk.Button(toolbar, text="刷新视图（不写账本）", command=self.refresh_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="快速查询（记延迟）", command=self.fast_query).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="权威重算", command=self.authoritative).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(toolbar, text="创建快照", command=self.create_snapshot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="恢复快照", command=self.restore_snapshot).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="保存会话 JSON", command=self.save_workspace).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="读取账本延迟", command=self.load_ledger_panels).pack(side=tk.LEFT, padx=2)

        self.cursor_var = tk.StringVar()
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(toolbar, textvariable=self.cursor_var).pack(side=tk.RIGHT, padx=8)
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT, padx=8)
        self._update_cursor_label()

        paned = ttk.Panedwindow(outer, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        editor_notebook = ttk.Notebook(paned)
        paned.add(editor_notebook, weight=2)

        sql_tab = ttk.Frame(editor_notebook, padding=4)
        editor_notebook.add(sql_tab, text="SQL 文本")
        sql_actions = ttk.Frame(sql_tab)
        sql_actions.pack(fill=tk.X)
        ttk.Button(sql_actions, text="应用 SQL（整体参数重建）", command=self.apply_sql).pack(side=tk.LEFT)
        ttk.Label(sql_actions, text="SQL 修改会递增 sql_version，并整体替换所有视图。").pack(side=tk.LEFT, padx=8)
        self.sql_editor = ScrolledText(sql_tab, wrap=tk.WORD, height=10, font=("Consolas", 10))
        self.sql_editor.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        tables_tab = ttk.Frame(editor_notebook, padding=4)
        editor_notebook.add(tables_tab, text="列式表声明（JSON）")
        table_actions = ttk.Frame(tables_tab)
        table_actions.pack(fill=tk.X)
        ttk.Button(table_actions, text="应用表声明（整体参数重建）", command=self.apply_tables).pack(side=tk.LEFT)
        ttk.Label(table_actions, text="必须保持 nulls/values 长度一致且列名、表名唯一。").pack(side=tk.LEFT, padx=8)
        self.tables_editor = ScrolledText(tables_tab, wrap=tk.NONE, height=10, font=("Consolas", 10))
        self.tables_editor.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        edits_tab = ttk.Frame(editor_notebook, padding=4)
        editor_notebook.add(edits_tab, text="编辑操作（JSON 数组）")
        edit_actions = ttk.Frame(edits_tab)
        edit_actions.pack(fill=tk.X)
        ttk.Button(edit_actions, text="提交编辑批次", command=self.submit_edits).pack(side=tk.LEFT)
        self.background_verify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(edit_actions, text="后台执行从零全量校验", variable=self.background_verify_var).pack(
            side=tk.LEFT, padx=10
        )
        ttk.Label(
            edit_actions,
            text="支持 push_filter / prune_columns / replace_scan_order / modify_aggregate",
        ).pack(side=tk.LEFT, padx=8)
        self.edits_editor = ScrolledText(edits_tab, wrap=tk.NONE, height=10, font=("Consolas", 10))
        self.edits_editor.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.edits_editor.insert(
            "1.0",
            _pretty(
                [
                    {
                        "type": "push_filter",
                        "target": "filter",
                    }
                ]
            ),
        )

        middle = ttk.Panedwindow(paned, orient=tk.HORIZONTAL)
        paned.add(middle, weight=3)
        plan_frame = ttk.Labelframe(middle, text="逻辑计划树（稳定顺序：先序 + 子节点 ID 排序）", padding=4)
        middle.add(plan_frame, weight=3)
        columns = ("node", "kind", "version", "rows", "summary")
        self.plan_tree = ttk.Treeview(plan_frame, columns=columns, show="tree headings", height=15)
        self.plan_tree.heading("#0", text="层级")
        self.plan_tree.heading("node", text="节点 ID")
        self.plan_tree.heading("kind", text="类型")
        self.plan_tree.heading("version", text="版本")
        self.plan_tree.heading("rows", text="逻辑行数")
        self.plan_tree.heading("summary", text="结构摘要")
        self.plan_tree.column("#0", width=180, stretch=False)
        self.plan_tree.column("node", width=120, stretch=False)
        self.plan_tree.column("kind", width=100, stretch=False)
        self.plan_tree.column("version", width=70, stretch=False)
        self.plan_tree.column("rows", width=90, stretch=False)
        self.plan_tree.column("summary", width=520)
        plan_scroll_y = ttk.Scrollbar(plan_frame, orient=tk.VERTICAL, command=self.plan_tree.yview)
        plan_scroll_x = ttk.Scrollbar(plan_frame, orient=tk.HORIZONTAL, command=self.plan_tree.xview)
        self.plan_tree.configure(yscrollcommand=plan_scroll_y.set, xscrollcommand=plan_scroll_x.set)
        self.plan_tree.grid(row=0, column=0, sticky="nsew")
        plan_scroll_y.grid(row=0, column=1, sticky="ns")
        plan_scroll_x.grid(row=1, column=0, sticky="ew")
        plan_frame.rowconfigure(0, weight=1)
        plan_frame.columnconfigure(0, weight=1)
        self.plan_tree.bind("<<TreeviewSelect>>", self._show_node_detail)

        detail_frame = ttk.Labelframe(middle, text="当前计划 / 节点详情", padding=4)
        middle.add(detail_frame, weight=2)
        self.summary_label = ttk.Label(detail_frame, text="", justify=tk.LEFT, anchor="nw")
        self.summary_label.pack(fill=tk.X, pady=(0, 4))
        self.node_detail = ScrolledText(detail_frame, wrap=tk.WORD, height=15, font=("Consolas", 9))
        self.node_detail.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Notebook(paned)
        paned.add(bottom, weight=2)
        self.affected_text = self._add_text_tab(bottom, "受影响集合 / 增量差异")
        self.latency_tree = self._add_latency_tab(bottom)
        self.verification_text = self._add_text_tab(bottom, "校验与重放")
        self.log_text = self._add_text_tab(bottom, "结构化日志 / 错误")

    def _add_text_tab(self, notebook: Any, title: str) -> Any:
        frame = ttk.Frame(notebook, padding=4)
        notebook.add(frame, text=title)
        text = ScrolledText(frame, wrap=tk.WORD, height=10, font=("Consolas", 9))
        text.pack(fill=tk.BOTH, expand=True)
        return text

    def _add_latency_tab(self, notebook: Any) -> Any:
        frame = ttk.Frame(notebook, padding=4)
        notebook.add(frame, text="真实延迟记录（账本）")
        columns = ("kind", "query", "budget", "elapsed", "code", "start", "source")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=9)
        for column, title, width in (
            ("kind", "路径", 160),
            ("query", "查询", 100),
            ("budget", "预算 ms", 90),
            ("elapsed", "实测 ms", 100),
            ("code", "业务码", 100),
            ("start", "monotonic 起点 ns", 180),
            ("source", "真值来源", 220),
        ):
            tree.heading(column, text=title)
            tree.column(column, width=width, stretch=column in {"source"})
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        return tree

    def _on_query_changed(self, _event: Any = None) -> None:
        self.active_query = self.query_var.get()
        self._load_query_into_editor()
        self.refresh_view(update_ledger_panels=False)

    def _load_query_into_editor(self) -> None:
        sql = self.engine.workspace.query_map()[self.active_query].sql
        self.sql_editor.delete("1.0", tk.END)
        self.sql_editor.insert("1.0", sql)

    def _load_tables_into_editor(self) -> None:
        self.tables_editor.delete("1.0", tk.END)
        self.tables_editor.insert("1.0", _pretty(self.engine.table_declaration()))

    def _update_cursor_label(self) -> None:
        self._last_ledger_cursor = self.engine.ledger.latest_cursor(self.engine.workspace_id)
        self.cursor_var.set(f"账本游标: {self._last_ledger_cursor} | 预算: {self.engine.budget_ms:g} ms")

    def _render_recovery(self) -> None:
        payload = {
            "recovery": self.engine.recovery,
            "note": "刷新视图不会写账本；只有快速查询、编辑批次、权威重算等显式动作会追加账本记录。",
        }
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", _pretty(payload))

    def refresh_view(self, update_ledger_panels: bool = True) -> None:
        try:
            view = self.engine.current_view(self.active_query)
        except Exception as error:
            self._show_error(error, "refresh_failed")
            return
        self._render_view(view)
        self._update_cursor_label()
        if update_ledger_panels:
            self.load_ledger_panels()
        self.status_var.set(f"视图已整体刷新 · {view['compile_status']}")

    def _render_view(self, view: dict[str, Any]) -> None:
        self.summary_label.configure(
            text=(
                f"摘要 {view['plan_summary']}\n"
                f"逻辑行数 {view['estimated_rows']} · 参数版本 {view['parameter_version']} · "
                f"编译状态 {view['compile_status']} · 节点数 {view['node_count']}"
            )
        )
        for item in self.plan_tree.get_children(""):
            self.plan_tree.delete(item)
        view_by_id = {node["node_id"]: node for node in view["nodes"]}
        children: dict[str, list[str]] = {node_id: [] for node_id in view_by_id}
        roots: list[str] = []
        for node_id, node in view_by_id.items():
            parents = [candidate for candidate, current in view_by_id.items() if node_id in current["children"]]
            if parents:
                children[parents[0]].append(node_id)
            else:
                roots.append(node_id)

        inserted: set[str] = set()

        def insert_node(node_id: str, parent: str = "") -> None:
            if node_id in inserted or node_id not in view_by_id:
                return
            node = view_by_id[node_id]
            tree_id = self.plan_tree.insert(
                parent,
                "end",
                text=node["kind"],
                values=(
                    node["node_id"],
                    node["kind"],
                    node["version"],
                    node["estimated_rows"],
                    node["summary"],
                ),
            )
            inserted.add(node_id)
            for child_id in sorted(children[node_id], key=lambda value: value.encode("utf-8")):
                insert_node(child_id, tree_id)

        for root_id in sorted(roots, key=lambda value: value.encode("utf-8")):
            insert_node(root_id)
        for remaining in sorted(set(view_by_id) - inserted, key=lambda value: value.encode("utf-8")):
            insert_node(remaining)
        first_root = self.plan_tree.get_children("")
        if first_root:
            self.plan_tree.selection_set(first_root[0])
        self.affected_text.delete("1.0", tk.END)
        self.affected_text.insert(
            "1.0",
            _pretty(
                {
                    "affected_nodes": view.get("affected_nodes", []),
                    "propagation": view.get("propagation", []),
                    "whole_view_diff": view.get("diff", {}),
                    "verification": view.get("verification", {}),
                    "latency": view.get("latency", {}),
                    "stale": view.get("stale"),
                }
            ),
        )

    def _show_node_detail(self, _event: Any = None) -> None:
        selection = self.plan_tree.selection()
        if not selection:
            return
        values = self.plan_tree.item(selection[0], "values")
        if not values:
            return
        node_id = values[0]
        try:
            view = self.engine.current_view(self.active_query)
        except Exception:
            return
        node = next((entry for entry in view["nodes"] if entry["node_id"] == node_id), None)
        if node is None:
            return
        details = {"node": node}
        # The full serialised root contains the canonical properties.
        def find(node_data: dict[str, Any]) -> dict[str, Any] | None:
            if node_data.get("id") == node_id:
                return node_data
            for child in node_data.get("children", []):
                found = find(child)
                if found is not None:
                    return found
            return None

        details = find(view["root"]) or details
        self.node_detail.delete("1.0", tk.END)
        self.node_detail.insert("1.0", _pretty(details))

    def fast_query(self) -> None:
        try:
            result = self.engine.quick_query(self.active_query)
        except Exception as error:
            self._show_error(error, "quick_query_failed")
            return
        self._render_view(self.engine.current_view(self.active_query))
        self.load_ledger_panels()
        self._update_cursor_label()
        if result.get("warning_code") == "E_LATENCY":
            self.status_var.set(
                f"E_LATENCY · 实测 {result['elapsed_ms']:.3f} ms > 预算 {result['budget_ms']:.3f} ms"
            )
        else:
            self.status_var.set(
                f"快速查询 {result['elapsed_ms']:.3f} ms · {result['compile_status']} · 未触发全量重编译"
            )

    def apply_sql(self) -> None:
        sql = self.sql_editor.get("1.0", tk.END).strip()
        try:
            result = self.engine.update_query_sql(self.active_query, sql)
        except Exception as error:
            self._show_error(error, "sql_update_rejected")
            return
        self._render_replacement(result)
        self._load_query_into_editor()
        self.status_var.set("SQL 已原子替换；旧参数版本已整体置为陈旧")

    def apply_tables(self) -> None:
        text = self.tables_editor.get("1.0", tk.END).strip()
        try:
            tables = json.loads(text)
            result = self.engine.update_tables(tables)
        except Exception as error:
            self._show_error(error, "table_update_rejected")
            return
        self._render_replacement(result)
        self._load_tables_into_editor()
        self._load_query_into_editor()
        self.status_var.set("表声明已原子替换；旧参数版本已整体置为陈旧")

    def _render_replacement(self, result: dict[str, Any]) -> None:
        self.refresh_view(update_ledger_panels=False)
        self.affected_text.delete("1.0", tk.END)
        self.affected_text.insert(
            "1.0",
            _pretty(
                {
                    "replacement": True,
                    "parameter_version": result.get("parameter_version"),
                    "diffs": result.get("diffs"),
                    "elapsed_ms": result.get("elapsed_ms"),
                }
            ),
        )
        self._update_cursor_label()

    def submit_edits(self) -> None:
        try:
            edits = json.loads(self.edits_editor.get("1.0", tk.END))
            if not isinstance(edits, list):
                raise ValueError("编辑输入必须是 JSON 数组")
            background = bool(self.background_verify_var.get())
            result = self.engine.submit_edits(
                edits,
                query_name=self.active_query,
                background_verify=background,
            )
        except Exception as error:
            self._show_error(error, "edit_rejected")
            return
        self._render_view(result["view"])
        self._update_cursor_label()
        if background:
            self.pending_verification_batch = result["batch_id"]
            self.status_var.set(f"编辑已提交 batch {result['batch_id']}；后台全量校验进行中")
            self._poll_verification()
        else:
            verification = result.get("verification", {})
            self.status_var.set(
                f"编辑已提交 · 增量 {result['incremental_ms']:.3f} ms · 全量校验 {verification.get('status')}"
            )
            self.load_ledger_panels()
        self._render_effects(result)

    def _render_effects(self, result: dict[str, Any]) -> None:
        self.affected_text.delete("1.0", tk.END)
        self.affected_text.insert(
            "1.0",
            _pretty(
                {
                    "batch_id": result.get("batch_id"),
                    "edit_seq": [result.get("edit_seq_start"), result.get("edit_seq_end")],
                    "incremental_ms": result.get("incremental_ms"),
                    "effects": result.get("effects"),
                    "verification": result.get("verification"),
                }
            ),
        )

    def _poll_verification(self) -> None:
        batch_id = self.pending_verification_batch
        if batch_id is None:
            return
        result = self.engine.wait_for_verification(batch_id, timeout=0)
        if result is None:
            self.root.after(100, self._poll_verification)
            return
        self.pending_verification_batch = None
        self.refresh_view(update_ledger_panels=False)
        self.verification_text.delete("1.0", tk.END)
        self.verification_text.insert("1.0", _pretty(result))
        self.load_ledger_panels()
        self.status_var.set(f"后台全量校验 {result.get('status')} · batch {batch_id}")

    def authoritative(self) -> None:
        self.status_var.set("权威重算进行中…")

        def worker() -> None:
            try:
                result = self.engine.authoritative_recompute(self.active_query)
                self.worker_queue.put(("authoritative", result))
            except Exception as error:
                self.worker_queue.put(("error", error))

        threading.Thread(target=worker, name="authoritative-recompute", daemon=True).start()

    def create_snapshot(self) -> None:
        default_name = self.engine.snapshot_dir / f"session-{int(time.time())}.json"
        path = filedialog.asksaveasfilename(
            title="保存会话快照",
            initialdir=str(self.engine.snapshot_dir),
            initialfile=default_name.name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            result = self.engine.create_snapshot(self.active_query, path)
        except Exception as error:
            self._show_error(error, "snapshot_create_failed")
            return
        self.status_var.set(f"快照已保存 · {result['content_digest']}")
        self.verification_text.delete("1.0", tk.END)
        self.verification_text.insert("1.0", _pretty({k: v for k, v in result.items() if k != "document"}))
        self._update_cursor_label()

    def save_workspace(self) -> None:
        path = filedialog.asksaveasfilename(
            title="保存当前会话 JSON",
            initialdir=str(Path(self.workspace_path).parent),
            initialfile=Path(self.workspace_path).name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            result = self.engine.export_workspace(path)
        except Exception as error:
            self._show_error(error, "workspace_export_failed")
            return
        self.workspace_path = path
        self.status_var.set(f"会话 JSON 已保存，可用于同一账本的跨进程恢复: {path}")
        self.verification_text.delete("1.0", tk.END)
        self.verification_text.insert("1.0", _pretty(result))

    def restore_snapshot(self) -> None:
        path = filedialog.askopenfilename(
            title="恢复会话快照",
            initialdir=str(self.engine.snapshot_dir),
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            result = self.engine.restore_snapshot(path)
        except Exception as error:
            self._show_error(error, "snapshot_restore_rejected")
            self.refresh_view(update_ledger_panels=False)
            return
        restored_active = result.get("active_query")
        if restored_active in self.engine.states:
            self.active_query = restored_active
            self.query_var.set(restored_active)
        self.refresh_view(update_ledger_panels=False)
        self._load_query_into_editor()
        self._load_tables_into_editor()
        self.verification_text.delete("1.0", tk.END)
        self.verification_text.insert("1.0", _pretty({k: v for k, v in result.items() if k != "views"}))
        self._update_cursor_label()
        self.status_var.set("快照已整体恢复，并完成账本逐项重放校验")

    def load_ledger_panels(self) -> None:
        try:
            latency = self.engine.ledger.recent_latency(self.engine.workspace_id, 100)
            verification = self.engine.ledger.recent_verifications(self.engine.workspace_id, 100)
        except Exception as error:
            self._show_error(error, "ledger_read_failed")
            return
        for item in self.latency_tree.get_children(""):
            self.latency_tree.delete(item)
        for record in latency:
            self.latency_tree.insert(
                "",
                "end",
                values=(
                    record["kind"],
                    record["query_name"] or "",
                    f"{record['budget_ms']:.3f}",
                    f"{record['elapsed_ms']:.3f}",
                    record["code"],
                    record["mono_start_ns"],
                    record["truth_source"],
                ),
            )
        self.verification_text.delete("1.0", tk.END)
        self.verification_text.insert("1.0", _pretty({"verifications": verification, "latencies": latency[:20]}))
        self._update_cursor_label()

    def _show_error(self, error: Exception, event: str) -> None:
        if hasattr(self.engine.logger, "error"):
            try:
                self.engine.logger.error(error, event, edit_seq=getattr(error, "edit_seq", None))
            except Exception:
                pass
        payload = error.to_dict() if hasattr(error, "to_dict") else {"code": "E_INTERNAL", "message": str(error)}
        payload["traceback"] = traceback.format_exc(limit=8)
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert("1.0", _pretty(payload))
        code = payload.get("code", "E_INTERNAL")
        self.status_var.set(f"{code} · {payload.get('message', '')}")

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "error":
                    self._show_error(payload, "authoritative_failed")
                elif kind == "authoritative":
                    self.load_ledger_panels()
                    self.status_var.set(
                        f"权威重算 MATCH · {payload.get('elapsed_ms', 0):.3f} ms · 真值来自账本"
                    )
                    self.verification_text.delete("1.0", tk.END)
                    self.verification_text.insert("1.0", _pretty(payload))
                    self._update_cursor_label()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker_queue)


def run_ui(
    *,
    workspace_path: str,
    ledger_path: str,
    snapshot_dir: str,
    log_path: str,
    budget_ms: float | None = None,
) -> int:
    if _TK_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Tkinter/Tcl-Tk is unavailable. On Windows reinstall Python with the Tcl/Tk component selected. "
            f"Original error: {_TK_IMPORT_ERROR}"
        )
    if tk is None:  # pragma: no cover
        raise RuntimeError("Tkinter is unavailable")
    from .engine import PlanEngine

    root = tk.Tk()
    try:
        engine = PlanEngine(
            workspace_path,
            ledger_path=ledger_path,
            snapshot_dir=snapshot_dir,
            log_path=log_path,
            budget_ms=budget_ms,
        )
    except Exception as error:
        messagebox.showerror("加载失败", str(error))
        root.destroy()
        return 2
    PlannerApplication(
        root,
        engine,
        workspace_path=workspace_path,
        ledger_path=ledger_path,
        snapshot_dir=snapshot_dir,
        log_path=log_path,
    )
    root.mainloop()
    return 0
