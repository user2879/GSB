"""Tkinter 诊断界面。

界面以只读方式展示分片规则与哈希环、路由单元映射与理由、依赖边、本轮失效
集合、并行任务轨迹与每库配额占用、版本与业务码。所有列表按稳定键排序渲染，
刷新动作只读取账本与快照，不执行任何写操作。

若运行环境缺少 tkinter（部分最小化 Linux），``import shardbench.gui`` 会在
导入处抛出 ImportError，由 CLI 给出友好提示；Windows 官方 Python 自带 tkinter。
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from typing import Any

from .config import load_config
from .ledger import Ledger
from .snapshot import load_snapshot


def _read_model(paths: dict[str, str]) -> dict[str, Any]:
    config = load_config(paths["config"], paths["units"])
    ledger = Ledger(paths["ledger"])
    try:
        rounds = ledger.list_committed_rounds()
        last_id = ledger.last_committed_round_id()
        bundle = ledger.load_round(last_id) if last_id is not None else None
    finally:
        ledger.close()
    snapshot = None
    try:
        snapshot = load_snapshot(paths["snapshot"]).to_dict()
    except FileNotFoundError:
        snapshot = None
    return {
        "config": config,
        "rounds": rounds,
        "bundle": bundle,
        "snapshot": snapshot,
    }


class ShardBenchApp(tk.Tk):
    def __init__(self, paths: dict[str, str]) -> None:
        super().__init__()
        self.paths = paths
        self.title("ShardBench 分库分表路由诊断工作台")
        self.geometry("1080x720")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self._build_tabs()
        self.reload()

    def _build_tabs(self) -> None:
        self.tabs: dict[str, tk.Text] = {}
        for name in (
            "规则与哈希环",
            "路由单元",
            "依赖边",
            "失效集合",
            "任务轨迹与配额",
            "版本与业务码",
        ):
            frame = ttk.Frame(self.notebook)
            text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10))
            scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
            scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=text.xview)
            text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
            scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
            scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
            text.pack(fill=tk.BOTH, expand=True)
            self.notebook.add(frame, text=name)
            self.tabs[name] = text
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.X)
        ttk.Button(bottom, text="刷新（只读，不写账本）", command=self.reload).pack(
            side=tk.LEFT, padx=8, pady=6
        )

    def _set_text(self, name: str, content: str) -> None:
        widget = self.tabs[name]
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, content)
        widget.configure(state=tk.DISABLED)

    def reload(self) -> None:
        model = _read_model(self.paths)
        config = model["config"]
        bundle = model["bundle"]
        self._render_rules(config)
        self._render_units(bundle)
        self._render_edges(config)
        self._render_invalidation(bundle)
        self._render_traces(bundle)
        self._render_versions(model, bundle)

    def _render_rules(self, config: Any) -> None:
        lines: list[str] = []
        for table_name in sorted(config.tables):
            table = config.tables[table_name]
            lines.append(
                f"表 {table.name} | 算法={table.algorithm} | 库={table.db_count} "
                f"每库表={table.tables_per_db} | 分片键={list(table.shard_key_columns)}"
            )
            lines.append(f"  参数={json.dumps(table.params, ensure_ascii=False, sort_keys=True)}")
            if table.name in config.rings:
                ring = config.rings[table.name]
                lines.append(f"  环版本={ring.ring_version} 节点={list(ring.node_ids)}")
                for point in ring.describe():
                    lines.append(
                        f"    vnode pos={point['position']} -> {point['node_id']}#{point['vnode_index']}"
                    )
        self._set_text("规则与哈希环", "\n".join(lines))

    def _render_units(self, bundle: dict[str, Any] | None) -> None:
        if bundle is None:
            self._set_text("路由单元", "尚无已提交轮次")
            return
        lines: list[str] = []
        for row in bundle["units"]:
            targets = json.loads(row["targets_json"])
            lines.append(
                f"单元 {row['unit_id']} | 表={row['table_name']} | 路由标识={row['route_key']}"
            )
            lines.append(f"  规范谓词: {row['normalized_predicate']}")
            lines.append(
                f"  广播={bool(row['is_broadcast'])} 原因={row['broadcast_reason']}"
            )
            lines.append(f"  目标库表: {', '.join(targets)}")
            plan = json.loads(row["plan_json"])
            for decision in plan.get("decisions", []):
                lines.append(f"  路由理由: {decision['reason']}")
            for derivation in plan.get("derivation", []):
                lines.append(f"  派生依据: {derivation}")
        self._set_text("路由单元", "\n".join(lines))

    def _render_edges(self, config: Any) -> None:
        lines: list[str] = []
        for unit_id in config.topo_order:
            unit = config.units[unit_id]
            if unit.depends_on:
                for dep in sorted(unit.depends_on):
                    lines.append(f"{dep} -> {unit_id}")
        self._set_text("依赖边", "\n".join(lines) or "无依赖边")

    def _render_invalidation(self, bundle: dict[str, Any] | None) -> None:
        if bundle is None:
            self._set_text("失效集合", "尚无已提交轮次")
            return
        invalidated = json.loads(bundle["round"]["invalidated_json"])
        lines = [f"失效集合（稳定顺序）: {', '.join(invalidated) or '∅'}"]
        for event in bundle["events"]:
            if event["event"] == "invalidation_propagated":
                payload = json.loads(event["payload_json"])
                for step in payload["steps"]:
                    lines.append(
                        f"{step['source']} -> {step['affected']} : {step['reason']}"
                    )
        self._set_text("失效集合", "\n".join(lines))

    def _render_traces(self, bundle: dict[str, Any] | None) -> None:
        if bundle is None:
            self._set_text("任务轨迹与配额", "尚无已提交轮次")
            return
        lines: list[str] = []
        for trace in bundle["traces"]:
            peaks = json.loads(trace["max_per_db_running_json"])
            lines.append(
                f"{trace['task_id']} 单元={trace['unit_id']} 结局={trace['outcome_code']}"
            )
            lines.append(
                "  等待(ms)="
                f"{trace['wait_start_ms']}->{trace['wait_end_ms']} "
                f"运行(ms)={trace['run_start_ms']}->{trace['run_end_ms']}"
            )
            lines.append(f"  每库峰值={json.dumps(peaks, ensure_ascii=False, sort_keys=True)} {trace['detail']}")
        self._set_text("任务轨迹与配额", "\n".join(lines))

    def _render_versions(
        self, model: dict[str, Any], bundle: dict[str, Any] | None
    ) -> None:
        lines = [f"已提交轮次数: {len(model['rounds'])}"]
        if bundle is not None:
            round_row = bundle["round"]
            for key in (
                "config_version",
                "ring_version",
                "dependency_version",
                "budget_version",
            ):
                lines.append(f"{key} = {round_row[key]}")
            lines.append(f"payload_digest = {round_row['payload_digest']}")
        if model["snapshot"] is not None:
            lines.append("快照: " + json.dumps(model["snapshot"], ensure_ascii=False, sort_keys=True))
        outcomes = sorted({trace["outcome_code"] for trace in bundle["traces"]}) if bundle else []
        lines.append("出现过的业务码: " + ", ".join(outcomes))
        self._set_text("版本与业务码", "\n".join(lines))


def launch_gui(paths: dict[str, str]) -> int:
    app = ShardBenchApp(paths)
    app.mainloop()
    return 0
