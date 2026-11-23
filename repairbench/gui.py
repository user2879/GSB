"""Tkinter 诊断工作台（Windows 本地环境，Tkinter 随标准库提供）。

页签：
  概览      副本组 / 版本 / 参数 / 快照状态
  节点与块  节点表与块表（含量化后定点值）
  批次调度  批次状态、带宽占用、等待时长、误差界
  误差界    单块 / 逐节点 / 全局三段贡献与主导段
  裁决      紧迫度排序与三类结论
  演进      版本演进、节点差异与受影响集合、陈旧存档
  账本      已提交批次、进度游标、重放校验
"""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .replay import replay_group
from .workbench import Session


class WorkbenchApp(tk.Tk):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session
        self.title("副本修复预算调度诊断工作台")
        self.geometry("1080x720")
        self._build()
        self.refresh_all()

    # ---------- 骨架 ----------
    def _build(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)
        ttk.Button(toolbar, text="运行调度", command=self.on_run).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="重放校验", command=self.on_replay).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="演进参数…", command=self.on_evolve).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="刷新", command=self.refresh_all).pack(side=tk.LEFT, padx=2)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(toolbar, textvariable=self.status).pack(side=tk.RIGHT, padx=6)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.tabs: dict[str, tk.Widget] = {}
        for name in ("概览", "节点与块", "批次调度", "误差界", "裁决", "演进", "账本"):
            frame = ttk.Frame(self.nb)
            self.nb.add(frame, text=name)
            self.tabs[name] = frame
        self._build_overview()
        self._build_table_tab("节点与块")
        self._build_table_tab("批次调度")
        self._build_text_tab("误差界")
        self._build_text_tab("裁决")
        self._build_text_tab("演进")
        self._build_table_tab("账本")

    def _build_overview(self) -> None:
        self.overview_text = tk.Text(self.tabs["概览"], wrap=tk.NONE)
        self.overview_text.pack(fill=tk.BOTH, expand=True)

    def _build_table_tab(self, name: str) -> None:
        tree = ttk.Treeview(self.tabs[name], show="headings")
        tree.pack(fill=tk.BOTH, expand=True)
        setattr(self, f"tree_{name}", tree)

    def _build_text_tab(self, name: str) -> None:
        text = tk.Text(self.tabs[name], wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True)
        setattr(self, f"text_{name}", text)

    def _fill_table(self, name: str, columns: list, rows: list) -> None:
        tree: ttk.Treeview = getattr(self, f"tree_{name}")
        tree.delete(*tree.get_children())
        tree["columns"] = columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor=tk.W)
        for row in rows:
            tree.insert("", tk.END, values=[row.get(c, "") for c in columns])

    def _fill_text(self, name: str, payload) -> None:
        text: tk.Text = getattr(self, f"text_{name}")
        text.delete("1.0", tk.END)
        text.insert(tk.END, json.dumps(payload, ensure_ascii=False, indent=2))

    # ---------- 数据刷新 ----------
    def refresh_all(self) -> None:
        session = self.session
        overview = {}
        for gid in sorted(session.groups):
            view = session.current_view(gid)
            overview[gid] = {
                "version": view["version"],
                "params": view["spec"]["params"],
                "stale_versions": [e["version"] for e in session.groups[gid].stale.entries],
            }
        self.overview_text.delete("1.0", tk.END)
        self.overview_text.insert(tk.END, json.dumps(overview, ensure_ascii=False, indent=2))

        node_rows, block_rows = [], []
        batch_rows, ledger_rows = [], []
        errors, verdicts, evolution = {}, {}, {}
        for gid in sorted(session.groups):
            ctx = session.groups[gid]
            fp = ctx.err_ledger.fp if ctx.err_ledger else None
            for n in ctx.spec.nodes:
                node_rows.append({
                    "组": gid, "节点": n.node_id, "地址": f"{n.host}:{n.port}",
                    "配额KB/s": n.bandwidth_kbps, "权重": n.priority_weight,
                    "配额定点": fp.quantize_units(n.bandwidth_kbps) if fp else "",
                })
            for b in ctx.spec.blocks:
                block_rows.append({
                    "组": gid, "块": b.block_id, "字节": b.byte_count,
                    "节点": b.node_id, "失配‰": b.mismatch_permille,
                    "字节定点": fp.quantize_bytes(b.byte_count) if fp else "",
                })
            if ctx.last_run:
                for batch in ctx.last_run["batches"]:
                    batch_rows.append({
                        "批次": batch["batch_id"], "块": batch["block_id"],
                        "节点": batch["node_id"], "区间": str(batch["range"]),
                        "状态": batch["state"], "带宽KB/s": batch["bandwidth_kbps"],
                        "等待ms": batch["wait_ms"], "误差界(量子)": batch["err_bound_quanta"],
                    })
            if ctx.err_ledger:
                errors[gid] = ctx.err_ledger.report()
            if ctx.last_ranking:
                verdicts[gid] = ctx.last_ranking
            evolution[gid] = {
                "version": ctx.version,
                "stale_archive": ctx.stale.entries,
            }
            if session.ledger:
                for row in session.ledger.committed_batches():
                    ledger_rows.append({
                        "序号": row["batch_seq"], "批次": row["batch_id"],
                        "块": row["block_id"], "区间": f"[{row['start_byte']},{row['end_byte']})",
                        "状态": row["state"], "带宽KB/s": row["bandwidth_kbps"],
                        "等待ms": row["wait_ms"], "误差界": row["err_bound_quanta"],
                        "版本": row["spec_version"],
                    })
        self._fill_table("节点与块",
                         ["组", "节点", "地址", "配额KB/s", "权重", "配额定点"], node_rows)
        tree = getattr(self, "tree_节点与块")
        for row in block_rows:
            tree.insert("", tk.END, values=[row.get(c, "") for c in
                        ("组", "块", "字节", "节点", "失配‰", "字节定点")])
        self._fill_table("批次调度",
                         ["批次", "块", "节点", "区间", "状态", "带宽KB/s", "等待ms", "误差界(量子)"],
                         batch_rows)
        self._fill_table("账本",
                         ["序号", "批次", "块", "区间", "状态", "带宽KB/s", "等待ms", "误差界", "版本"],
                         ledger_rows)
        self._fill_text("误差界", errors)
        self._fill_text("裁决", verdicts)
        self._fill_text("演进", evolution)

    # ---------- 动作 ----------
    def on_run(self) -> None:
        try:
            self.session.run_all()
            self.status.set("调度完成")
        except Exception as err:  # noqa: BLE001 - GUI 统一兜底
            messagebox.showerror("调度失败", str(err))
            self.status.set("调度失败")
        self.refresh_all()

    def on_replay(self) -> None:
        try:
            reports = [replay_group(self.session, gid) for gid in sorted(self.session.groups)]
            self.status.set("重放一致" if all(r["consistent"] for r in reports) else "重放差异")
            messagebox.showinfo("重放校验", json.dumps(reports, ensure_ascii=False, indent=2))
        except Exception as err:  # noqa: BLE001
            messagebox.showerror("重放失败", str(err))
        self.refresh_all()

    def on_evolve(self) -> None:
        path = filedialog.askopenfilename(title="选择演进后的组规格 JSON")
        if not path:
            return
        gid = sorted(self.session.groups)[0]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                new_spec = json.load(fh)
            gid = new_spec.get("group_id", gid)
            result = self.session.evolve(gid, new_spec)
            self.status.set(f"已演进到版本 {result['version']}")
        except Exception as err:  # noqa: BLE001
            messagebox.showerror("演进失败", str(err))
        self.refresh_all()


def launch(session: Session) -> None:
    app = WorkbenchApp(session)
    app.mainloop()
