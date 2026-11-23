from __future__ import annotations

import json
from pathlib import Path
from .engine import ConfigEngine
from .errors import DiagnosticError
from .graph import graph_layers
from .recovery import recover_and_replay
from .structured_log import JsonlLogger
from .subscriptions import SubscriptionSimulator
from .validation import load_bundle


def _short_digest(value: str | None) -> str:
    return (value or "")[:12]


def run_gui(
    workspace: Path | str,
    samples_dir: Path | str,
    *,
    db_path: str | None = None,
    log_path: str | None = None,
) -> int:
    """Launch Tkinter lazily so headless CLI tests do not require a Tk build."""

    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        payload = {
            "ok": False,
            "code": "E_GUI",
            "detail": "当前 Python 无法加载 Tkinter；Windows 官方 Python 安装包自带 Tk，CLI 功能不受影响",
            "error": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    workspace_path = Path(workspace).resolve()
    samples_path = Path(samples_dir).resolve()
    logger = JsonlLogger(Path(log_path) if log_path else workspace_path / "logs" / "diagnostics.jsonl")
    engine = ConfigEngine(workspace_path, db_path=db_path, logger=logger)
    simulator: SubscriptionSimulator | None = None

    class DiagnosticsApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("配置中心热更新中间件诊断台")
            self.root.geometry("1380x900")
            self.root.minsize(1100, 720)
            self.status_var = tk.StringVar(value="就绪")
            self.meta_var = tk.StringVar(value="账本未初始化")
            self.stale_var = tk.StringVar(value="陈旧视图: -")

            toolbar = ttk.Frame(root, padding=(10, 8))
            toolbar.pack(fill="x")
            ttk.Button(toolbar, text="刷新", command=self.refresh).pack(side="left", padx=3)
            ttk.Button(toolbar, text="初始化样本", command=self.initialize_samples).pack(side="left", padx=3)
            ttk.Button(toolbar, text="应用下一变更", command=self.apply_next).pack(side="left", padx=3)
            ttk.Button(toolbar, text="计时回收", command=self.expire_now).pack(side="left", padx=3)
            ttk.Button(toolbar, text="模拟轮询", command=self.poll_clients).pack(side="left", padx=3)
            ttk.Button(toolbar, text="恢复重放", command=self.recover).pack(side="left", padx=3)
            ttk.Label(toolbar, textvariable=self.meta_var).pack(side="right", padx=8)

            stale_bar = ttk.Frame(root, padding=(10, 0))
            stale_bar.pack(fill="x")
            ttk.Label(stale_bar, textvariable=self.stale_var, foreground="#a33").pack(side="left")

            upper = ttk.Panedwindow(root, orient="horizontal")
            upper.pack(fill="both", expand=True, padx=10, pady=6)

            key_frame = ttk.LabelFrame(upper, text="配置键列表（按 key 稳定排序）")
            self.key_tree = ttk.Treeview(
                key_frame,
                columns=("key", "value", "version", "status", "digest"),
                show="headings",
                height=18,
            )
            for column, heading, width in (
                ("key", "配置键", 225),
                ("value", "取值", 90),
                ("version", "版本", 60),
                ("status", "状态", 76),
                ("digest", "内容摘要", 120),
            ):
                self.key_tree.heading(column, text=heading)
                self.key_tree.column(column, width=width, anchor="w", stretch=False)
            key_scroll = ttk.Scrollbar(key_frame, orient="vertical", command=self.key_tree.yview)
            self.key_tree.configure(yscrollcommand=key_scroll.set)
            self.key_tree.pack(side="left", fill="both", expand=True)
            key_scroll.pack(side="right", fill="y")
            upper.add(key_frame, weight=3)

            graph_frame = ttk.LabelFrame(upper, text="派生依赖图（稳定分层）")
            self.graph_canvas = tk.Canvas(graph_frame, background="#fbfbfb", height=380)
            graph_x = ttk.Scrollbar(graph_frame, orient="horizontal", command=self.graph_canvas.xview)
            graph_y = ttk.Scrollbar(graph_frame, orient="vertical", command=self.graph_canvas.yview)
            self.graph_canvas.configure(xscrollcommand=graph_x.set, yscrollcommand=graph_y.set)
            self.graph_canvas.grid(row=0, column=0, sticky="nsew")
            graph_y.grid(row=0, column=1, sticky="ns")
            graph_x.grid(row=1, column=0, sticky="ew")
            graph_frame.rowconfigure(0, weight=1)
            graph_frame.columnconfigure(0, weight=1)
            upper.add(graph_frame, weight=5)

            lower = ttk.Panedwindow(root, orient="horizontal")
            lower.pack(fill="both", expand=False, padx=10, pady=(0, 8))

            trace_frame = ttk.LabelFrame(lower, text="最近失效集合与传播链")
            self.trace_text = tk.Text(trace_frame, height=12, wrap="none")
            trace_scroll = ttk.Scrollbar(trace_frame, orient="vertical", command=self.trace_text.yview)
            self.trace_text.configure(yscrollcommand=trace_scroll.set)
            self.trace_text.pack(side="left", fill="both", expand=True)
            trace_scroll.pack(side="right", fill="y")
            lower.add(trace_frame, weight=4)

            snapshot_frame = ttk.LabelFrame(lower, text="账本版本、快照与视图")
            self.snapshot_tree = ttk.Treeview(
                snapshot_frame,
                columns=("kind", "version", "stamp", "status"),
                show="headings",
                height=10,
            )
            for column, heading, width in (
                ("kind", "类型", 80),
                ("version", "版本", 85),
                ("stamp", "创建/游标", 145),
                ("status", "摘要/状态", 245),
            ):
                self.snapshot_tree.heading(column, text=heading)
                self.snapshot_tree.column(column, width=width, anchor="w", stretch=False)
            self.snapshot_tree.pack(fill="both", expand=True)
            lower.add(snapshot_frame, weight=5)

            status_bar = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w", padding=5)
            status_bar.pack(fill="x", side="bottom")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.refresh()
            if engine.open_events:
                codes = ", ".join(item["code"] for item in engine.open_events)
                self._set_status(f"启动恢复提示：{codes}；请执行“恢复重放”查看未完成事务")

        def _set_status(self, text: str) -> None:
            self.status_var.set(text)

        def initialize_samples(self) -> None:
            try:
                spec, clients, _ = load_bundle(samples_path)
                engine.initialize(spec, clients)
                self._set_status("样本初始装载完成")
            except DiagnosticError as exc:
                self._set_status(f"{exc.code} {exc.key or ''}/{exc.field or ''}: {exc.detail}")
            except Exception as exc:
                self._set_status(f"初始化失败: {exc}")
            self.refresh()

        def apply_next(self) -> None:
            try:
                _, _, changes = load_bundle(samples_path)
                cursor = engine.ledger.max_committed_source_sequence()
                pending = [item for item in changes.operations if item.sequence > cursor]
                if not pending:
                    self._set_status("没有待提交变更")
                    return
                result = engine.apply_operation(pending[0])
                self._set_status(f"{result['operation_id']} 已提交，失效 {len(result['affected'])} 键")
            except DiagnosticError as exc:
                self._set_status(f"{exc.code} {exc.key or ''}/{exc.field or ''}: {exc.detail}")
            except Exception as exc:
                self._set_status(f"提交失败: {exc}")
            self.refresh()

        def expire_now(self) -> None:
            try:
                result = engine.collect_expired()
                if result["reclaimed"]:
                    self._set_status(
                        f"回收 {', '.join(result['reclaimed'])}；下游 {', '.join(result['downstream']) or '-'}"
                    )
                else:
                    self._set_status("本轮无过期键")
            except DiagnosticError as exc:
                self._set_status(f"{exc.code}: {exc.detail}")
            except Exception as exc:
                self._set_status(f"回收失败: {exc}")
            self.refresh()

        def poll_clients(self) -> None:
            nonlocal simulator
            try:
                _, clients, _ = engine._initial_models()
                if simulator is None:
                    simulator = SubscriptionSimulator(engine, clients)
                result = simulator.poll(2)
                current = result["current_view_version"]
                stale = len(result["stale_events"])
                visible = sum(
                    client["visible_count"]
                    for round_item in result["rounds"]
                    for client in round_item["clients"]
                )
                self._set_status(f"轮询完成：当前视图 v{current}，可见结论 {visible}，整视图替换 {stale}")
            except Exception as exc:
                self._set_status(f"轮询失败: {exc}")
            self.refresh()

        def recover(self) -> None:
            try:
                result = recover_and_replay(engine, workspace_path, logger=logger)
                suffix = "要求重跑未知事务" if result["needs_rerun"] else "全部已提交事务重放一致"
                self._set_status(f"{result['code']}：{result['keys_compared']} 键，{suffix}")
            except DiagnosticError as exc:
                self._set_status(f"{exc.code} {exc.key or ''}/{exc.field or ''}: {exc.detail}")
            except Exception as exc:
                self._set_status(f"恢复失败: {exc}")
            self.refresh()

        def refresh(self) -> None:
            for item in self.key_tree.get_children():
                self.key_tree.delete(item)
            for item in self.snapshot_tree.get_children():
                self.snapshot_tree.delete(item)
            self.graph_canvas.delete("all")
            self.trace_text.delete("1.0", "end")
            if not engine.ledger.has_state():
                self.meta_var.set("账本未初始化")
                self.stale_var.set("陈旧视图: 不适用")
                self.trace_text.insert("end", "请点击“初始化样本”或运行 python -m config_diag init")
                return
            state = engine.list_state()
            ledger = state["ledger"]
            self.meta_var.set(
                f"账本 F{ledger['format_version']} v{ledger['ledger_version']} | "
                f"集合 v{ledger['config_set_version']} | 口径 v{ledger['derivation_version']} | "
                f"摘要 {_short_digest(ledger['content_digest'])}"
            )
            current = state["current_view"]
            self.stale_var.set(
                f"当前视图 v{current['view_version']} 摘要 {_short_digest(current['view_digest'])} | "
                f"陈旧视图 {state['stale_view_count']} 条 / 陈旧轨迹 {state['stale_trace_count']} 条"
                f"（整体替换，不参与展示与统计）"
            )
            for record in state["keys"]:
                self.key_tree.insert(
                    "",
                    "end",
                    values=(
                        record["key"],
                        "" if record["value"] is None else record["value"],
                        record["version"],
                        record["status"],
                        _short_digest(record["content_digest"]),
                    ),
                )
            refs_by_key = {record["key"]: tuple(record["refs"]) for record in state["keys"]}
            self._draw_graph(refs_by_key, {record["key"]: record["status"] for record in state["keys"]})
            trace = state["latest_trace"]
            self.trace_text.insert(
                "end",
                json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True) if trace else "尚无变更轨迹",
            )
            for snapshot in state["snapshots"]:
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        "快照",
                        snapshot["ledger_version"],
                        f"cursor {snapshot['change_cursor']}",
                        f"{_short_digest(snapshot['content_digest'])} {snapshot['reason']}",
                    ),
                )
            for view in state["views"]:
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        "视图",
                        view["view_version"],
                        f"set {view['config_set_version']} / deriv {view['derivation_version']}",
                        f"{view['status']} {_short_digest(view['view_digest'])}",
                    ),
                )
            for journal in state["journal"]:
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        f"变更 {journal['kind']}",
                        journal["source_sequence"] or "-",
                        f"journal {journal['journal_id']}",
                        f"{journal['status']} / trace {'stale' if journal['trace_stale'] else 'current'}",
                    ),
                )

        def _draw_graph(self, refs_by_key: dict[str, tuple[str, ...]], statuses: dict[str, str]) -> None:
            canvas = self.graph_canvas
            if not refs_by_key:
                return
            layers = graph_layers(refs_by_key)
            positions: dict[str, tuple[float, float]] = {}
            box_width = 200
            box_height = 34
            x_gap = 245
            y_gap = 58
            for layer_index, layer in enumerate(layers):
                for row_index, key in enumerate(layer):
                    x = 20 + layer_index * x_gap
                    y = 20 + row_index * y_gap
                    positions[key] = (x, y)
                    fill = {"active": "#e8f5e9", "invalid": "#ffebee", "expired": "#eceff1"}.get(
                        statuses.get(key), "#ffffff"
                    )
                    canvas.create_rectangle(
                        x,
                        y,
                        x + box_width,
                        y + box_height,
                        fill=fill,
                        outline="#546e7a",
                        width=1,
                    )
                    canvas.create_text(
                        x + 8,
                        y + box_height / 2,
                        text=key,
                        anchor="w",
                        font=("TkDefaultFont", 9),
                    )
            for child in sorted(refs_by_key):
                cx, cy = positions[child]
                for parent in sorted(refs_by_key[child]):
                    px, py = positions[parent]
                    canvas.create_line(
                        px + box_width,
                        py + box_height / 2,
                        cx,
                        cy + box_height / 2,
                        arrow="last",
                        fill="#455a64",
                        smooth=False,
                    )
            width = 40 + len(layers) * x_gap
            height = 40 + max(len(layer) for layer in layers) * y_gap
            canvas.configure(scrollregion=(0, 0, width, height))

        def on_close(self) -> None:
            engine.close()
            self.root.destroy()

    try:
        root = tk.Tk()
        root.diagnostics_app = DiagnosticsApp(root)
        root.mainloop()
        return 0
    except Exception as exc:
        try:
            engine.close()
        except Exception:
            pass
        print(json.dumps({"ok": False, "code": "E_GUI", "detail": str(exc)}, ensure_ascii=False))
        return 1
