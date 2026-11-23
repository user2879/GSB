"""Shard Router Workbench - tkinter desktop app (Windows / Python 3.11+).

Run:  python workbench_app.py
All state lives under the chosen workspace directory (SQLite ledger, snapshot,
structured logs). The GUI is a thin shell over shardbench.engine.
"""

import json
import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from shardbench.engine import WorkbenchEngine
from shardbench.errors import BizError

APP_TITLE = "分库分表路由工作台 Shard Router Workbench"


class WorkbenchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.engine = WorkbenchEngine("workbench_workspace")
        self.plans = {}
        self._ui_queue = queue.Queue()
        self._build_ui()
        self.after(150, self._drain_queue)

    # ---------------- UI construction ----------------
    def _build_ui(self):
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(toolbar, text="加载配置…", command=self.on_load_config).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="加载谓词…", command=self.on_load_predicates).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="编译全部计划", command=self.on_compile_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="并行探查", command=self.on_probe).pack(side=tk.LEFT, padx=4)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="保存快照", command=self.on_save_snapshot).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="恢复快照", command=self.on_recover).pack(side=tk.LEFT, padx=4)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="注入配置损坏", command=self.on_inject_config).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="注入租约冲突", command=self.on_inject_lease).pack(side=tk.LEFT, padx=4)

        self.version_var = tk.StringVar(value="版本: 未加载配置")
        ttk.Label(self, textvariable=self.version_var, anchor=tk.W,
                  relief=tk.SUNKEN).pack(side=tk.BOTTOM, fill=tk.X)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = ttk.Frame(paned, width=320)
        paned.add(left, weight=1)
        ttk.Label(left, text="路由单元 (依赖)").pack(anchor=tk.W)
        self.unit_tree = ttk.Treeview(left, columns=("algo", "slots", "deps"),
                                      show="headings", height=8)
        for col, w in (("algo", 110), ("slots", 60), ("deps", 160)):
            self.unit_tree.heading(col, text=col)
            self.unit_tree.column(col, width=w)
        self.unit_tree.pack(fill=tk.X)
        ttk.Label(left, text="谓词集合").pack(anchor=tk.W, pady=(8, 0))
        self.pred_list = tk.Listbox(left, height=10)
        self.pred_list.pack(fill=tk.BOTH, expand=True)

        right = ttk.Notebook(paned)
        paned.add(right, weight=4)
        self.plan_text = self._tab(right, "路由计划")
        self.ledger_tree = self._ledger_tab(right)
        self.lease_tree = self._lease_tab(right)
        self.inv_text = self._tab(right, "失效传播")
        self.log_text = self._tab(right, "结构化日志")

    def _tab(self, notebook, title):
        frame = ttk.Frame(notebook)
        text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 10))
        text.pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text=title)
        return text

    def _ledger_tab(self, notebook):
        frame = ttk.Frame(notebook)
        cols = ("batch", "unit", "target", "wait_ms", "exec_ms", "lease", "rule_v", "ring_v")
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=90)
        tree.pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text="路由账本")
        return tree

    def _lease_tab(self, notebook):
        frame = ttk.Frame(notebook)
        cols = ("lease_id", "db", "granted", "expires", "version", "state", "reason")
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=110)
        tree.pack(fill=tk.BOTH, expand=True)
        notebook.add(frame, text="租约")
        return tree

    # ---------------- helpers ----------------
    def _say(self, widget, text):
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)

    def _refresh_versions(self):
        if self.engine.versions is None:
            return
        v = self.engine.versions.current()
        stats = self.engine.scheduler.stats()
        self.version_var.set(
            f"rule_v={v.rule_version} ring_v={v.ring_version} "
            f"dep_v={v.dep_graph_version} budget_v={v.budget_version} | "
            f"并发峰值 global={stats['max_observed_global']} "
            f"per_db={stats['max_observed_per_db']}"
        )

    def _refresh_ledger(self):
        self.ledger_tree.delete(*self.ledger_tree.get_children())
        for row in self.engine.ledger.current_tasks():
            targets = ",".join(f"db{t['db']}.t{t['table']}" for t in row["targets"])
            self.ledger_tree.insert("", tk.END, values=(
                row["batch_id"], row["unit_id"], targets, row["wait_ms"],
                row["exec_ms"], row["lease"].get("lease_id", ""),
                row["rule_version"], row["ring_version"]))
        self.lease_tree.delete(*self.lease_tree.get_children())
        for lease in self.engine.ledger.current_leases():
            self.lease_tree.insert("", tk.END, values=(
                lease["lease_id"], lease["db"], lease["granted_ms"],
                lease["expires_ms"], lease["version"], lease["state"],
                lease["reclaim_reason"][:60]))

    def _refresh_log(self):
        path = self.engine.logger.path
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self._say(self.log_text, fh.read())

    def _show_error(self, exc: BizError):
        messagebox.showerror(exc.code, f"{exc.message}\n字段: {exc.field}\n{exc.detail or ''}")

    def _drain_queue(self):
        try:
            while True:
                fn = self._ui_queue.get_nowait()
                fn()
        except queue.Empty:
            pass
        self.after(150, self._drain_queue)

    # ---------------- actions ----------------
    def on_load_config(self):
        path = filedialog.askopenfilename(
            title="选择分片配置 JSON", filetypes=[("JSON", "*.json")],
            initialdir="samples")
        if not path:
            return
        try:
            self.engine.load_config(path)
        except BizError as exc:
            self._show_error(exc)
            return
        self.unit_tree.delete(*self.unit_tree.get_children())
        for unit_id in sorted(self.engine.config.units):
            u = self.engine.config.units[unit_id]
            self.unit_tree.insert("", tk.END, values=(
                f"{unit_id} ({u.algo_type})",
                f"{u.physical_dbs}x{u.tables_per_db}", ",".join(u.depends_on) or "-"))
        diff = self.engine.last_diff
        inv = self.engine.last_invalidation
        self._say(self.inv_text, json.dumps(
            {"diff": diff, "invalidation": inv}, indent=2, ensure_ascii=False))
        self._refresh_versions()
        self._refresh_ledger()
        self._refresh_log()

    def on_load_predicates(self):
        path = filedialog.askopenfilename(
            title="选择谓词集合 JSON", filetypes=[("JSON", "*.json")],
            initialdir="samples")
        if not path:
            return
        doc = json.load(open(path, encoding="utf-8"))
        self.pred_list.delete(0, tk.END)
        self._predicates = doc["predicates"]
        for p in self._predicates:
            self.pred_list.insert(tk.END, f"{p['id']}  [{p['unit']}]")

    def on_compile_all(self):
        if not getattr(self, "_predicates", None):
            messagebox.showinfo("提示", "请先加载谓词集合")
            return
        try:
            out = []
            for p in self._predicates:
                plan = self.engine.compile(p["unit"], p["where"])
                self.plans[p["id"]] = plan
                out.append(f"== {p['id']} [{p['unit']}] digest={plan.plan_digest[:32]}…")
                for t, r in zip(plan.targets, plan.reasons):
                    out.append(f"   db{t['db']}.t{t['table']}  <- {r}")
            self._say(self.plan_text, "\n".join(out))
        except BizError as exc:
            self._show_error(exc)
        self._refresh_versions()

    def on_probe(self):
        if not self.plans:
            messagebox.showinfo("提示", "请先编译计划")
            return

        def run():
            try:
                batch_id, records = self.engine.probe(list(self.plans.values()))
                self._ui_queue.put(lambda: self._after_probe(batch_id, records))
            except BizError as exc:
                self._ui_queue.put(lambda: self._show_error(exc))

        import threading
        threading.Thread(target=run, daemon=True).start()

    def _after_probe(self, batch_id, records):
        waits = [r.wait_ms for r in records]
        messagebox.showinfo(
            "探查完成",
            f"batch={batch_id}, 任务={len(records)}, "
            f"等待ms min={min(waits)} max={max(waits)}")
        self._refresh_versions()
        self._refresh_ledger()
        self._refresh_log()

    def on_save_snapshot(self):
        try:
            snap = self.engine.save_snapshot()
            messagebox.showinfo("快照", f"已保存, 游标={snap['payload']['ledger_cursor']}")
        except BizError as exc:
            self._show_error(exc)

    def on_recover(self):
        try:
            result = self.engine.recover()
            messagebox.showinfo(
                "恢复", f"游标={result['recovered_cursor']}\n"
                f"未知租约(需重跑)={result['unknown_leases']}")
        except BizError as exc:
            self._show_error(exc)
        self._refresh_ledger()
        self._refresh_log()

    def on_inject_config(self):
        path = filedialog.askopenfilename(
            title="选择要损坏的配置副本", filetypes=[("JSON", "*.json")],
            initialdir="samples")
        if not path:
            return
        err = self.engine.inject_config_corruption(path, mode="invalid_json")
        messagebox.showwarning("E_CONFIG", f"{err.message}\n字段: {err.field}")
        self._refresh_log()

    def on_inject_lease(self):
        held = self.engine.lease_manager.held_leases()
        if not held:
            lease = self.engine.lease_manager.acquire(db=0)
        else:
            lease = held[0]
        self.engine.inject_lease_version_conflict(lease.lease_id)
        try:
            self.engine.lease_manager.renew(lease.lease_id,
                                            expected_version=lease.version - 1)
        except BizError as exc:
            messagebox.showwarning("E_LEASE", f"{exc.message}\n{exc.detail}")
        self._refresh_ledger()
        self._refresh_log()


if __name__ == "__main__":
    app = WorkbenchApp()
    app.mainloop()
