"""Tkinter 只读诊断界面。

展示各节点租约水位、排队分片、未知批次标记与账本版本。
所有数据经只读查询获得，刷新不执行任何写事务，不改变账本内容。
"""

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # 无 Tk 运行时由调用方决定降级
    tk = None
    ttk = None

from .ledger import FORMAT_VERSION, Ledger
from .spec import load_spec, spec_digest


class DiagnosticUI:
    def __init__(self, root, spec_path, db_path):
        self.root = root
        self.spec_path = spec_path
        self.db_path = db_path
        root.title("边缘缓存分片下发诊断工作台（只读）")
        root.geometry("980x640")

        self.meta_var = tk.StringVar(value="账本未加载")
        top = ttk.Frame(root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.meta_var,
                  font=("monospace", 10)).pack(side=tk.LEFT)
        ttk.Button(top, text="刷新（只读）", command=self.refresh).pack(
            side=tk.RIGHT)

        columns = ("node", "quota", "capacity", "used", "limit", "active",
                   "leases")
        self.tree = ttk.Treeview(root, columns=columns, show="headings",
                                 height=8)
        headers = {"node": "节点", "quota": "配额版本",
                   "capacity": "容量槽位", "used": "已用槽位",
                   "limit": "每节点上限", "active": "活跃租约",
                   "leases": "活跃租约明细"}
        widths = {"node": 100, "quota": 80, "capacity": 80, "used": 80,
                  "limit": 90, "active": 80, "leases": 420}
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=widths[col], anchor=tk.W)
        self.tree.pack(fill=tk.X, padx=8, pady=4)

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        ttk.Label(body, text="排队分片（waits）").grid(row=0, column=0,
                                                        sticky=tk.W)
        self.wait_box = tk.Text(body, height=10, wrap=tk.NONE)
        self.wait_box.grid(row=1, column=0, sticky=tk.NSEW)

        ttk.Label(body, text="未知批次标记（unknown_batches）").grid(
            row=0, column=1, sticky=tk.W)
        self.unknown_box = tk.Text(body, height=10, wrap=tk.NONE)
        self.unknown_box.grid(row=1, column=1, sticky=tk.NSEW)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)
        self.refresh()
        self.root.after(4000, self._auto_refresh)

    def _auto_refresh(self):
        self.refresh()
        self.root.after(4000, self._auto_refresh)

    def refresh(self):
        try:
            spec = load_spec(self.spec_path)
            ledger = Ledger(self.db_path, spec, spec_digest(spec))
            try:
                self._render(ledger, spec)
            finally:
                ledger.close()
            self.status_var.set("刷新完成（只读，未写入账本）")
        except Exception as exc:
            self.status_var.set(f"加载失败: {exc}")

    def _render(self, ledger, spec):
        version = ledger._meta("format_version")
        digest = ledger._meta("spec_digest")[:16]
        last_seq = ledger.last_committed_seq()
        self.meta_var.set(
            f"账本格式版本={version} (期望 {FORMAT_VERSION}) | "
            f"声明摘要={digest}… | 最后提交事件 seq={last_seq} | "
            f"预算已用={ledger.get_budget_used()}/"
            f"{spec['session']['global_byte_budget']}")

        for item in self.tree.get_children():
            self.tree.delete(item)
        water = ledger.snapshot()["water"]
        for node in spec["nodes"]:
            node_id = node["node_id"]
            w = water.get(node_id, {})
            active_rows = ledger.conn.execute(
                "SELECT lease_id,shard_id,slot_index,renew_count,lease_version "
                "FROM leases WHERE node_id=? AND state='active' ORDER BY slot_index",
                (node_id,)).fetchall()
            detail = ", ".join(
                f"slot{r[2]}:{r[1]}(v{r[4]},续{r[3]})" for r in active_rows)
            self.tree.insert("", tk.END, values=(
                node_id, w.get("quota_version"), node["capacity_slots"],
                w.get("used_slots", 0), node["per_node_lease_limit"],
                w.get("active_leases", 0), detail or "-"))

        self.wait_box.delete("1.0", tk.END)
        for row in ledger.all_waits():
            self.wait_box.insert(
                tk.END,
                f"seq={row['seq']} node={row['node_id']} "
                f"shard={row['shard_id']} start={row['wait_start_ms']} "
                f"end={row['wait_end_ms']} granted={row['granted']}\n")

        self.unknown_box.delete("1.0", tk.END)
        for row in ledger.snapshot()["unknown"]:
            self.unknown_box.insert(
                tk.END,
                f"batch={row['batch_id']} seq={row['seq']} "
                f"node={row['node_id']} shard={row['shard_id']} "
                f"slot={row['slot_index']} status={row['resolved']}\n")


def launch_ui(spec_path, db_path):
    root = tk.Tk()
    DiagnosticUI(root, spec_path, db_path)
    root.mainloop()
