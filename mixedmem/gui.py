"""Tkinter diagnostic workbench with read-only, stable refresh."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .errors import DiagnosticError
from .graph import DependencyGraph, build_graph
from .schema import Plan, load_plan
from .snapshot import list_snapshots


@dataclass(frozen=True)
class DiagnosticState:
    plan: Plan
    graph: DependencyGraph
    ledger_path: str
    ledger_content_digest: str | None
    ledger_plan_versions: tuple[int, ...]
    batches: tuple[dict[str, Any], ...]
    leases: tuple[dict[str, Any], ...]
    layer_results: tuple[dict[str, Any], ...]
    attribution: dict[str, Any] | None
    snapshots: tuple[dict[str, Any], ...]


def _readonly_connection(path: str) -> sqlite3.Connection:
    uri = f"file:{quote(str(Path(path).resolve()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


def _rows(connection: sqlite3.Connection, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


def load_diagnostic_state(
    *,
    plan_path: str,
    db_path: str,
    snapshot_dir: str,
) -> DiagnosticState:
    """Read plan, ledger and snapshots without performing any write."""

    plan = load_plan(plan_path)
    graph = build_graph(plan)
    ledger_digest: str | None = None
    versions: list[int] = []
    batches: list[dict[str, Any]] = []
    leases: list[dict[str, Any]] = []
    layer_results: list[dict[str, Any]] = []
    attribution: dict[str, Any] | None = None
    database = Path(db_path)
    if database.exists():
        connection = _readonly_connection(str(database))
        try:
            meta = _rows(connection, "SELECT key,value FROM meta ORDER BY key")
            ledger_digest = next(
                (row["value"] for row in meta if row["key"] == "content_digest"), None
            )
            versions = [
                int(row["plan_version"])
                for row in _rows(
                    connection,
                    "SELECT DISTINCT plan_version FROM batches ORDER BY plan_version",
                )
            ]
            batches = _rows(
                connection,
                "SELECT plan_version,batch_id,sample_ids_json,status,queue_start_ms,"
                "worker_start_ms,end_ms,worker_name,error_json FROM batches "
                "ORDER BY plan_version,batch_id",
            )
            for batch in batches:
                batch["sample_ids"] = json.loads(batch.pop("sample_ids_json"))
                if batch.get("error_json"):
                    batch["error"] = json.loads(batch["error_json"])
            leases = _rows(
                connection,
                "SELECT lease_id,plan_version,batch_id,layer_id,block_start,block_count,"
                "granted_ms,expires_ms,status,released_ms,reason FROM leases "
                "ORDER BY granted_ms,batch_id,layer_id,lease_id",
            )
            layer_results = _rows(
                connection,
                "SELECT plan_version,batch_id,layer_id,layer_version,float_digest,"
                "quant_digest,errors_json,error_bound_json,saturation_count,"
                "occupancy_bytes,occupancy_blocks,lease_id,block_start,block_count,valid,"
                "committed_ms FROM layer_results ORDER BY plan_version,batch_id,layer_id",
            )
            for row in layer_results:
                row["errors"] = json.loads(row.pop("errors_json"))
                row["error_bound"] = json.loads(row.pop("error_bound_json"))
            attribution_rows = _rows(
                connection,
                "SELECT attribution_json FROM attributions WHERE plan_version=? "
                "ORDER BY layer_id LIMIT 1",
                (plan.version,),
            )
            if not attribution_rows and versions:
                attribution_rows = _rows(
                    connection,
                    "SELECT attribution_json FROM attributions WHERE plan_version=? "
                    "ORDER BY layer_id LIMIT 1",
                    (max(versions),),
                )
            if attribution_rows:
                attribution = json.loads(attribution_rows[0]["attribution_json"])
        finally:
            connection.close()
    return DiagnosticState(
        plan=plan,
        graph=graph,
        ledger_path=str(database),
        ledger_content_digest=ledger_digest,
        ledger_plan_versions=tuple(sorted(set(versions))),
        batches=tuple(batches),
        leases=tuple(leases),
        layer_results=tuple(layer_results),
        attribution=attribution,
        snapshots=tuple(list_snapshots(snapshot_dir)),
    )


def _graph_layout(graph: DependencyGraph) -> tuple[dict[str, tuple[int, int]], list[dict[str, str]]]:
    levels: dict[str, int] = {}
    for layer_id in graph.order:
        upstream = graph.upstream(layer_id)
        levels[layer_id] = 0 if not upstream else max(levels[source] + 1 for source in upstream)
    by_level: dict[int, list[str]] = {}
    for layer_id in graph.order:
        by_level.setdefault(levels[layer_id], []).append(layer_id)
    positions: dict[str, tuple[int, int]] = {}
    for level in sorted(by_level):
        for row, layer_id in enumerate(by_level[level]):
            positions[layer_id] = (60 + level * 190, 50 + row * 100)
    return positions, [edge.to_dict() for edge in graph.edges]


def launch_gui(
    *,
    plan_path: str,
    db_path: str,
    snapshot_dir: str,
    sample_count: int,
    batch_size: int,
    workers: int,
) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - platform dependent
        raise DiagnosticError(
            "E_IO",
            "Tkinter is unavailable in this Python runtime; install the official "
            "Windows Python distribution with Tcl/Tk support",
            details={"cause": f"{type(exc).__name__}: {exc}"},
        ) from exc

    class Workbench(tk.Tk):
        def __init__(self) -> None:
            super().__init__()
            self.title("mixedmem · Mixed Precision Lease Diagnostics")
            self.geometry("1240x820")
            self.state_data: DiagnosticState | None = None
            header = ttk.Frame(self, padding=(12, 8))
            header.pack(fill="x")
            self.header_text = tk.StringVar()
            ttk.Label(header, textvariable=self.header_text, font=("TkDefaultFont", 11, "bold")).pack(
                side="left", fill="x", expand=True
            )
            ttk.Button(header, text="只读刷新", command=self.refresh).pack(side="right")
            ttk.Label(
                header,
                text=f"samples={sample_count} batch={batch_size} workers={workers}",
            ).pack(side="right", padx=12)
            self.tabs = ttk.Notebook(self)
            self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            self._build_graph_tab()
            self._build_lease_tab()
            self._build_schedule_tab()
            self._build_attribution_tab()
            self._build_version_tab()
            self.refresh()

        def _tree(self, parent: Any, columns: list[tuple[str, str, int]]) -> Any:
            frame = ttk.Frame(parent)
            tree = ttk.Treeview(
                frame, columns=[item[0] for item in columns], show="headings"
            )
            for key, title, width in columns:
                tree.heading(key, text=title)
                tree.column(key, width=width, anchor="w", stretch=True)
            scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scroll.set)
            tree.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            return frame, tree

        def _build_graph_tab(self) -> None:
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text="层依赖图")
            self.graph_canvas = tk.Canvas(tab, background="#fafafa", highlightthickness=0)
            self.graph_canvas.pack(fill="both", expand=True)
            self.graph_order = tk.Text(tab, height=5, wrap="word")
            self.graph_order.pack(fill="x")

        def _build_lease_tab(self) -> None:
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text="块池/租约时间线")
            self.lease_canvas = tk.Canvas(tab, background="#fafafa", highlightthickness=0, height=330)
            self.lease_canvas.pack(fill="both", expand=True)
            frame, self.lease_tree = self._tree(
                tab,
                [
                    ("lease", "租约", 210),
                    ("batch", "批次", 90),
                    ("layer", "层", 100),
                    ("range", "块区间 [start,end)", 150),
                    ("grant", "授予(ms)", 120),
                    ("expire", "到期(ms)", 120),
                    ("status", "状态", 100),
                ],
            )
            frame.pack(fill="both", expand=True)

        def _build_schedule_tab(self) -> None:
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text="批调度时间线")
            self.schedule_canvas = tk.Canvas(tab, background="#fafafa", highlightthickness=0, height=330)
            self.schedule_canvas.pack(fill="both", expand=True)
            frame, self.schedule_tree = self._tree(
                tab,
                [
                    ("batch", "批次", 90),
                    ("status", "状态", 90),
                    ("queue", "排队(ms)", 90),
                    ("compute", "计算(ms)", 90),
                    ("worker", "工作线程", 180),
                    ("ranges", "块区间", 300),
                ],
            )
            frame.pack(fill="both", expand=True)

        def _build_attribution_tab(self) -> None:
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text="数值偏差/误差归因")
            self.attribution_summary = tk.StringVar()
            ttk.Label(tab, textvariable=self.attribution_summary, padding=8).pack(fill="x")
            frame, self.attribution_tree = self._tree(
                tab,
                [
                    ("layer", "层", 100),
                    ("max_abs", "最大绝对偏差", 150),
                    ("mean_abs", "平均绝对偏差", 150),
                    ("bound", "合成误差界", 150),
                    ("share", "累计占比", 110),
                    ("within", "界内", 80),
                    ("saturation", "饱和次数", 100),
                ],
            )
            frame.pack(fill="both", expand=True)
            self.bound_text = tk.Text(tab, height=7, wrap="word")
            self.bound_text.pack(fill="x")

        def _build_version_tab(self) -> None:
            tab = ttk.Frame(self.tabs)
            self.tabs.add(tab, text="版本/快照")
            self.version_text = tk.StringVar()
            ttk.Label(tab, textvariable=self.version_text, padding=8).pack(fill="x")
            frame, self.snapshot_tree = self._tree(
                tab,
                [
                    ("filename", "快照文件", 380),
                    ("version", "计划版本", 100),
                    ("scale_version", "量化口径版本", 130),
                    ("digest", "内容摘要", 400),
                    ("error", "错误", 200),
                ],
            )
            frame.pack(fill="both", expand=True)

        def refresh(self) -> None:
            state = load_diagnostic_state(
                plan_path=plan_path,
                db_path=db_path,
                snapshot_dir=snapshot_dir,
            )
            self.state_data = state
            self.header_text.set(
                f"plan v{state.plan.version} · scale-version "
                f"{state.plan.quantization_scale_version} · "
                f"ledger {state.ledger_content_digest or '<missing>'}"
            )
            self._render_graph(state)
            self._render_leases(state)
            self._render_schedule(state)
            self._render_attribution(state)
            self._render_versions(state)

        def _render_graph(self, state: DiagnosticState) -> None:
            canvas = self.graph_canvas
            canvas.delete("all")
            positions, edges = _graph_layout(state.graph)
            node_width, node_height = 140, 48
            for edge in sorted(edges, key=lambda item: (item["source"], item["dependent"], item["kind"])):
                if edge["source"] not in positions or edge["dependent"] not in positions:
                    continue
                x1, y1 = positions[edge["source"]]
                x2, y2 = positions[edge["dependent"]]
                color = "#2d6cdf" if edge["kind"] == "activation" else "#c1771b"
                canvas.create_line(
                    x1 + node_width,
                    y1 + node_height / 2,
                    x2,
                    y2 + node_height / 2,
                    arrow="last",
                    fill=color,
                    width=2,
                    dash=None if edge["kind"] == "activation" else (5, 3),
                )
            for layer_id in state.graph.order:
                x, y = positions[layer_id]
                layer = state.plan.layer_by_id[layer_id]
                canvas.create_rectangle(
                    x,
                    y,
                    x + node_width,
                    y + node_height,
                    fill="#ffffff",
                    outline="#35506b",
                    width=2,
                )
                canvas.create_text(
                    x + node_width / 2,
                    y + 15,
                    text=layer_id,
                    font=("TkDefaultFont", 10, "bold"),
                )
                canvas.create_text(
                    x + node_width / 2,
                    y + 34,
                    text=f"{layer.precision} · {layer.input_dim}→{layer.output_dim} · s={layer.scale}",
                )
            width = max((x for x, _ in positions.values()), default=0) + node_width + 60
            height = max((y for _, y in positions.values()), default=0) + node_height + 50
            canvas.configure(scrollregion=(0, 0, width, height))
            self.graph_order.configure(state="normal")
            self.graph_order.delete("1.0", "end")
            self.graph_order.insert(
                "end",
                "稳定拓扑序: "
                + " → ".join(state.graph.order)
                + "\n实线=激活引用，虚线=显存占用派生；刷新只读取计划、SQLite 与快照。",
            )
            self.graph_order.configure(state="disabled")

        @staticmethod
        def _time_range(values: list[int]) -> tuple[int, int]:
            clean = [value for value in values if value is not None]
            if not clean:
                return 0, 1
            low, high = min(clean), max(clean)
            return low, (high if high > low else low + 1)

        def _render_leases(self, state: DiagnosticState) -> None:
            canvas = self.lease_canvas
            canvas.delete("all")
            for item in self.lease_tree.get_children():
                self.lease_tree.delete(item)
            leases = sorted(
                state.leases,
                key=lambda item: (
                    int(item.get("granted_ms") or 0),
                    item["lease_id"],
                ),
            )
            start, end = self._time_range(
                [int(item["granted_ms"]) for item in leases]
                + [int(item["expires_ms"]) for item in leases]
            )
            width = max(900, int(canvas.winfo_width()) - 80)
            for index, lease in enumerate(leases):
                y = 30 + index * 28
                x1 = 250 + (int(lease["granted_ms"]) - start) / (end - start) * (width - 260)
                x2 = 250 + (int(lease["expires_ms"]) - start) / (end - start) * (width - 260)
                x2 = max(x1 + 3, x2)
                status = lease["status"]
                color = {
                    "active": "#2f8f55",
                    "released": "#587a9c",
                    "expired": "#b07b14",
                    "revoked": "#b33a3a",
                    "unknown": "#7a3fa0",
                    "assigned": "#3c6f8f",
                }.get(status, "#777777")
                label = (
                    f"{lease['lease_id']} [{lease['block_start']},"
                    f"{lease['block_start'] + lease['block_count']})"
                )
                canvas.create_text(8, y + 8, anchor="w", text=label)
                canvas.create_rectangle(x1, y, x2, y + 16, fill=color, outline="")
                canvas.create_text(
                    min(x2 + 6, width + 30),
                    y + 8,
                    anchor="w",
                    text=status,
                )
                self.lease_tree.insert(
                    "",
                    "end",
                    values=(
                        lease["lease_id"],
                        lease["batch_id"],
                        lease["layer_id"],
                        f"[{lease['block_start']},{lease['block_start'] + lease['block_count']})",
                        lease["granted_ms"],
                        lease["expires_ms"],
                        lease["status"],
                    ),
                )
            canvas.configure(scrollregion=(0, 0, width + 120, 40 + max(1, len(leases)) * 28))

        def _render_schedule(self, state: DiagnosticState) -> None:
            canvas = self.schedule_canvas
            canvas.delete("all")
            for item in self.schedule_tree.get_children():
                self.schedule_tree.delete(item)
            batches = sorted(state.batches, key=lambda item: (item["plan_version"], item["batch_id"]))
            start, end = self._time_range(
                [
                    int(value)
                    for batch in batches
                    for value in (batch.get("queue_start_ms"), batch.get("end_ms"))
                    if value is not None
                ]
            )
            width = max(900, int(canvas.winfo_width()) - 100)
            ranges_by_batch: dict[tuple[int, str], list[str]] = {}
            for lease in state.leases:
                ranges_by_batch.setdefault(
                    (int(lease["plan_version"]), str(lease["batch_id"])), []
                ).append(
                    f"[{lease['block_start']},{lease['block_start'] + lease['block_count']})"
                )
            for index, batch in enumerate(batches):
                y = 32 + index * 34
                queue_start = int(batch.get("queue_start_ms") or start)
                worker_start = int(batch.get("worker_start_ms") or queue_start)
                batch_end = int(batch.get("end_ms") or worker_start)
                q1 = 230 + (queue_start - start) / (end - start) * (width - 240)
                q2 = 230 + (worker_start - start) / (end - start) * (width - 240)
                c2 = 230 + (batch_end - start) / (end - start) * (width - 240)
                canvas.create_text(8, y + 9, anchor="w", text=f"{batch['batch_id']} · {batch['status']}")
                canvas.create_rectangle(q1, y, max(q1 + 2, q2), y + 18, fill="#d29b2c", outline="")
                canvas.create_rectangle(max(q2, q2 + 1), y, max(q2 + 2, c2), y + 18, fill="#2f6f9f", outline="")
                worker = batch.get("worker_name") or "<none>"
                self.schedule_tree.insert(
                    "",
                    "end",
                    values=(
                        batch["batch_id"],
                        batch["status"],
                        None if worker_start is None else worker_start - queue_start,
                        None if batch_end is None else batch_end - worker_start,
                        worker,
                        ", ".join(
                            sorted(
                                ranges_by_batch.get(
                                    (int(batch["plan_version"]), str(batch["batch_id"])), []
                                )
                            )
                        ),
                    ),
                )
            canvas.configure(scrollregion=(0, 0, width + 120, 40 + max(1, len(batches)) * 34))

        def _render_attribution(self, state: DiagnosticState) -> None:
            for item in self.attribution_tree.get_children():
                self.attribution_tree.delete(item)
            attribution = state.attribution
            if attribution is None:
                self.attribution_summary.set("当前计划版本尚未写入归因账本")
                self.bound_text.configure(state="normal")
                self.bound_text.delete("1.0", "end")
                self.bound_text.insert("end", "运行推理或选择已有快照后刷新。")
                self.bound_text.configure(state="disabled")
                return
            self.attribution_summary.set(
                f"最大偏差层: {attribution.get('max_deviation_layer')} · "
                f"主导段: {attribution.get('dominant_component')} · "
                f"总误差界: {attribution.get('total_error_bound'):.9g}"
            )
            for layer in sorted(attribution.get("layers", []), key=lambda item: item["layer_id"]):
                self.attribution_tree.insert(
                    "",
                    "end",
                    values=(
                        layer["layer_id"],
                        f"{layer['max_abs_error']:.9g}",
                        f"{layer['mean_abs_error']:.9g}",
                        f"{layer['error_bound']['combined']:.9g}",
                        f"{layer['cumulative_share']:.6f}",
                        layer["within_bound"],
                        layer["saturation_count"],
                    ),
                )
            self.bound_text.configure(state="normal")
            self.bound_text.delete("1.0", "end")
            self.bound_text.insert(
                "end",
                json.dumps(
                    {
                        "component_totals": attribution.get("component_totals"),
                        "component_shares": attribution.get("component_shares"),
                        "dominant_rule": attribution.get("dominant_rule"),
                        "tie_rule": attribution.get("tie_rule"),
                        "max_deviation_error_affected": attribution.get(
                            "max_deviation_error_affected"
                        ),
                        "quantization_policy": attribution.get("policy"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
            self.bound_text.configure(state="disabled")

        def _render_versions(self, state: DiagnosticState) -> None:
            self.version_text.set(
                f"当前声明计划版本 {state.plan.version} · 量化口径版本 "
                f"{state.plan.quantization_scale_version} · 计划摘要 "
                f"{state.plan.content_digest} · 账本版本 {state.ledger_plan_versions} · "
                f"账本摘要 {state.ledger_content_digest or '<missing>'}"
            )
            for snapshot in state.snapshots:
                self.snapshot_tree.insert(
                    "",
                    "end",
                    values=(
                        snapshot.get("filename"),
                        snapshot.get("plan_version"),
                        snapshot.get("quantization_scale_version"),
                        snapshot.get("content_digest"),
                        None if "error" not in snapshot else snapshot["error"].get("code"),
                    ),
                )

    app = Workbench()
    app.mainloop()
