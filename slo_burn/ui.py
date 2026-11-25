"""Tkinter 仪表盘：展示账本中的窗口结论、陈旧版本与租约裁决。

无显示环境通过 Xvfb（SLO_DISPLAY，默认 :99）运行；即使无交互显示，
控件树仍可完整构建并 update，供无头冒烟测试截图/断言。
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from .ledger import Ledger
from .loader import load_service_specs


class BurnDashboard(tk.Tk):
    def __init__(self, workspace: str, ledger_path: str) -> None:
        super().__init__()
        self.title("SLO 错误预算燃烧率诊断台")
        self.geometry("1100x680")
        specs = load_service_specs(Path(workspace) / "services.json")
        self.service_ids = [spec.service_id for spec in specs]
        with Ledger(ledger_path) as ledger:
            self.conclusions = ledger.active_conclusions()
            self.latest = ledger.latest_commit()
            adj_rows = ledger.connection.execute(
                "SELECT event_seq, slot_id, service_id, lease_id, version, decision"
                " FROM adjudications ORDER BY event_seq"
            ).fetchall()
        self._build_widgets(adj_rows)

    def _build_widgets(self, adj_rows: list[tuple]) -> None:
        header = tk.Frame(self, padx=10, pady=8)
        header.pack(fill=tk.X)
        digest = self.latest["content_digest"][:16] if self.latest else "none"
        tk.Label(
            header,
            text=f"账本提交 seq={self.latest['commit_seq'] if self.latest else '-'} "
            f"摘要={digest} 格式版本={self.latest['format_version'] if self.latest else '-'}",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side=tk.LEFT)

        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        conclusions_tab = tk.Frame(notebook)
        notebook.add(conclusions_tab, text="窗口结论")
        columns = ("service", "minute", "remaining", "burn_num", "burn_den", "alert", "path")
        tree = ttk.Treeview(conclusions_tab, columns=columns, show="headings")
        headings = {
            "service": "服务",
            "minute": "分钟",
            "remaining": "剩余预算(缩放)",
            "burn_num": "燃烧率分子",
            "burn_den": "燃烧率分母",
            "alert": "告警",
            "path": "路径",
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=130, anchor=tk.CENTER)
        for item in self.conclusions[:500]:
            tree.insert(
                "",
                tk.END,
                values=(
                    item["service_id"],
                    item["minute"],
                    item["remaining_budget_scaled"],
                    item["burn_rate_num"],
                    item["burn_rate_den"],
                    "ALERT" if item["alert"] else "ok",
                    item["path"],
                ),
            )
        scrollbar = ttk.Scrollbar(conclusions_tab, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        lease_tab = tk.Frame(notebook)
        notebook.add(lease_tab, text="租约裁决")
        lease_columns = ("seq", "slot", "service", "lease", "version", "decision")
        lease_tree = ttk.Treeview(lease_tab, columns=lease_columns, show="headings")
        for column, title in zip(
            lease_columns, ("事件序号", "槽位", "服务", "租约", "版本", "裁决")
        ):
            lease_tree.heading(column, text=title)
            lease_tree.column(column, width=160, anchor=tk.CENTER)
        for row in adj_rows:
            lease_tree.insert("", tk.END, values=row)
        lease_tree.pack(fill=tk.BOTH, expand=True)

        stats_tab = tk.Frame(notebook)
        notebook.add(stats_tab, text="服务统计")
        alert_count = sum(1 for item in self.conclusions if item["alert"])
        tk.Label(
            stats_tab,
            text=f"活跃结论 {len(self.conclusions)} 条；告警 {alert_count} 条；"
            f"服务 {', '.join(self.service_ids)}",
            pady=20,
        ).pack()
        self.status = tk.Label(
            self,
            text=f"display={os.environ.get('SLO_DISPLAY', os.environ.get('DISPLAY'))} "
            "仪表盘构建成功",
            anchor=tk.W,
            bd=1,
            relief=tk.SUNKEN,
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)


def launch_dashboard(workspace: str, ledger_path: str, auto_close_ms: int | None = None) -> BurnDashboard:
    app = BurnDashboard(workspace, ledger_path)
    app.update_idletasks()
    app.update()
    if auto_close_ms is not None:
        app.after(auto_close_ms, app.destroy)
    app.mainloop()
    return app
