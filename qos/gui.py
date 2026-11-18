"""Tkinter 诊断界面：依赖关系、失效传播、生效计划与诊断业务码可视化。

Tkinter 仅在实际启动 GUI 时导入；无显示环境或缺少 Tkinter 时抛出
带说明的 RuntimeError，不影响命令行与测试链路。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .changeset import ChangeSet
from .workbench import Workbench


class GuiUnavailable(RuntimeError):
    pass


def _import_tk():
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext
    except Exception as exc:  # pragma: no cover - 环境相关
        raise GuiUnavailable(
            "当前 Python 环境缺少 Tkinter。Windows 官方安装包默认包含；"
            "Linux 请安装 python3-tk。命令行功能不受影响。") from exc
    return tk, ttk, scrolledtext


class WorkbenchGUI:
    def __init__(self, db_path: str = "data/qos_ledger.db",
                 log_path: str = "logs/qos.jsonl"):
        tk, ttk, scrolledtext = _import_tk()
        self.tk = tk
        self.ttk = ttk
        self.scrolledtext = scrolledtext
        self.wb = Workbench(db_path, log_path)
        self.root = tk.Tk()
        self.root.title("QoS 流量整形规则诊断工作台")
        self.root.geometry("1080x760")
        self._build_layout()

    # ---- 布局 -----------------------------------------------------------
    def _build_layout(self) -> None:
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="规则文件:").pack(side=tk.LEFT)
        self.rule_path_var = tk.StringVar(value="samples/rules.json")
        ttk.Entry(top, textvariable=self.rule_path_var, width=48).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(top, text="加载并编译", command=self.on_load).pack(side=tk.LEFT)
        ttk.Button(top, text="增量变更(R1)", command=self.on_change).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(top, text="整形模拟", command=self.on_simulate).pack(side=tk.LEFT)
        ttk.Button(top, text="快照/重放", command=self.on_snapshot_replay).pack(
            side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status_var,
                  foreground="#0a6").pack(anchor=tk.W, padx=10)

        body = ttk.Frame(self.root, padding=8)
        body.pack(fill=tk.BOTH, expand=True)
        self.notebook = ttk.Notebook(body)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_graph = self._text_tab("依赖关系图")
        self.tab_invalid = self._text_tab("失效传播集合")
        self.tab_plan = self._text_tab("当前生效计划")
        self.tab_code = self._text_tab("诊断业务码")

    def _text_tab(self, title: str):
        frame = self.ttk.Frame(self.notebook)
        widget = self.scrolledtext.ScrolledText(frame, wrap="word",
                                                 font=("Consolas", 10))
        widget.pack(fill=self.tk.BOTH, expand=True)
        self.notebook.add(frame, text=title)
        return widget

    def _set_text(self, widget, content: str) -> None:
        widget.delete("1.0", self.tk.END)
        widget.insert(self.tk.END, content)

    # ---- 事件 -----------------------------------------------------------
    def on_load(self) -> None:
        try:
            result = self.wb.load(self.rule_path_var.get())
            self._render_graph()
            self._render_plan(result["plan"])
            self._set_text(self.tab_invalid, "（首次加载，全部规则参与编译）\n")
            self._set_text(self.tab_code, "E_OK：加载与编译成功\n")
            self.status_var.set(
                f"计划版本 v{result['plan']['plan_version']}，"
                f"规则集 {result['ruleset_version'][:10]}")
        except Exception as exc:
            self._show_error("load", exc)

    def on_change(self) -> None:
        try:
            changeset = ChangeSet.from_dict(__import__("json").loads(
                Path("samples/changeset_modify.json").read_text(encoding="utf-8")))
            result = self.wb.apply_change(changeset)
            check = self.wb.verify_incremental()
            self._render_graph()
            self._render_plan(result["plan"])
            lines = ["失效序列（稳定拓扑序）:"]
            lines += [f"  {idx + 1}. {rid}" for idx, rid in
                      enumerate(result["invalidation_sequence"])]
            lines.append("\n传播轨迹:")
            for trace in result["trace"]:
                edge = f" via {trace['via_edge']}" if trace["via_edge"] else ""
                lines.append(
                    f"  step{trace['step']}: {trace['rule_id']} "
                    f"<- {trace['triggered_by']}{edge}")
            lines.append(
                f"\n重编译 {check['recompiled_count']} 条，"
                f"复用 {check['reused_count']} 条；E_OK 增量=全量")
            self._set_text(self.tab_invalid, "\n".join(lines))
            self._set_text(self.tab_code, "E_OK：增量编译与全量编译摘要一致\n")
            self.status_var.set(f"计划版本 v{result['plan']['plan_version']}")
        except Exception as exc:
            self._show_error("change", exc)

    def on_simulate(self) -> None:
        try:
            result = self.wb.simulate()
            lines = ["整形观测摘要:",
                     f"  转发字节: {result['digest']['forwarded_bytes']}",
                     f"  包数: {result['digest']['packet_count']}",
                     f"  持续(us): {result['digest']['duration_us']}",
                     "\n每规则吞吐（承诺 vs 实际，字节/秒）:"]
            for rid, item in sorted(result["throughput"].items()):
                lines.append(
                    f"  {rid}: committed={item['committed_bytes_per_sec']} "
                    f"actual={item['actual_bytes_per_sec']} "
                    f"forwarded={item['forwarded_bytes']}/"
                    f"allowed={item['allowed_total_bytes']}")
            lines.append("\n前 10 个包观测:")
            for rec in result["records"][:10]:
                lines.append(
                    f"  t={rec['timestamp_us']}us seq={rec['seq']} "
                    f"rule={rec['rule_id']} sent={rec['sent_bytes']} "
                    f"delay={rec['delay_us']}us")
            self._set_text(self.tab_code, "\n".join(lines))
            self.status_var.set("E_OK：吞吐未超承诺（含突发容量）")
        except Exception as exc:
            self._show_error("simulate", exc)

    def on_snapshot_replay(self) -> None:
        try:
            path = "data/gui_snapshot.json"
            self.wb.create_snapshot(path)
            diff = self.wb.restore_snapshot(path)
            replay = self.wb.replay(path)
            content = (
                f"快照: {path}\n恢复模式: {diff['diff']['restore_mode']}\n"
                f"计划版本: v{diff['snapshot']['plan_version']}\n"
                f"重放校验: E_OK，计划摘要 {replay['plan_hash'][:16]}...\n"
                f"比对字段: {', '.join(replay['compared_fields'])}")
            self._set_text(self.tab_code, content)
            self.status_var.set("E_OK：快照恢复与重放一致")
        except Exception as exc:
            self._show_error("snapshot", exc)

    # ---- 渲染 -----------------------------------------------------------
    def _render_graph(self) -> None:
        assert self.wb.graph is not None
        topo = self.wb.graph.topo_order()
        lines = ["稳定拓扑顺序:", "  " + " -> ".join(topo), "", "依赖边 (upstream -> downstream):"]
        for rid in topo:
            for child in sorted(self.wb.graph.downstream[rid]):
                lines.append(f"  {rid} -> {child}")
        self._set_text(self.tab_graph, "\n".join(lines))

    def _render_plan(self, plan: dict[str, Any]) -> None:
        lines = [f"计划版本: v{plan['plan_version']}",
                 f"计划摘要: {plan['plan_hash']}",
                 f"条目数: {plan['entry_count']} "
                 f"(重编译 {len(plan['recompiled'])} / 复用 {len(plan['reused'])})",
                 "", "按确定性优先级排序的条目:"]
        for idx, entry in enumerate(plan["payload"]["entries"], start=1):
            lines.append(
                f"{idx:>2}. {entry['rule_id']:<10} "
                f"addr={entry['match']['dst_addr']:<9} "
                f"prefixes={entry['match']['port_prefixes']} "
                f"rate={entry['shaping']['rate_kbps']}kbps "
                f"burst={entry['shaping']['burst_bytes']} "
                f"queue={entry['shaping']['queue_limit_packets']}")
        self._set_text(self.tab_plan, "\n".join(lines))

    def _show_error(self, action: str, exc: Exception) -> None:
        code = getattr(exc, "code", "E_UNKNOWN")
        self._set_text(self.tab_code,
                       f"{code}：{getattr(exc, 'message', str(exc))}\n\n"
                       f"details={getattr(exc, 'details', {})}")
        self.status_var.set(f"{code} @ {action}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    WorkbenchGUI().run()
    return 0
