"""Tkinter 诊断界面：只读展示账本与调度状态，刷新不改变账本内容。

展示：各工作进程在途批次数与配额余量、逐批三格式回执的采纳与否决理由、
被补齐字段与补齐依据、失效集合与传播轨迹、未知批次标记、账本格式版本与
快照列表。渲染顺序稳定（全部按键排序）。
"""
import json
import os
import sqlite3
import threading
import tkinter as tk
from tkinter import ttk

from .core import load_config, load_ports
from .ledger import Ledger
from .pipeline import run_pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_only_query(path, sql, args=()):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cur = conn.execute(sql, args)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


class DiagApp(tk.Tk):
    def __init__(self, ledger_path, config_path, ports_path):
        super().__init__()
        self.title("推理服务诊断界面")
        self.geometry("1100x700")
        self.ledger_path = ledger_path
        self.config_path = config_path
        self.ports_path = ports_path
        self.scheduler = None
        self.snapshot_dir = os.path.join(
            os.path.dirname(os.path.abspath(ledger_path)), "snapshots")
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Button(top, text="刷新", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(top, text="运行演示流水线",
                   command=self.run_pipeline_async).pack(side=tk.LEFT, padx=4)
        self.status = tk.StringVar(value="就绪")
        ttk.Label(top, textvariable=self.status).pack(side=tk.LEFT, padx=8)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.trees = {}
        self._add_tab("workers", "工作进程", ("worker", "in_flight", "quota_remaining"))
        self._add_tab("batches", "批次", ("batch_id", "worker_id", "param_version",
                                          "status", "adopted_digest"))
        self._add_tab("receipts", "回执", ("batch_id", "request_id", "src_format",
                                           "adopted", "reject_reason", "filled",
                                           "fill_basis"))
        self._add_tab("invalidations", "失效传播", ("batch_id", "request_id", "step",
                                                    "reason"))
        self._add_tab("ledger", "账本/快照", ("key", "value"))
        self.refresh()
        self.after(300, self._poll_stats)

    def _add_tab(self, key, title, columns):
        frame = ttk.Frame(self.nb)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor=tk.W)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.nb.add(frame, text=title)
        self.trees[key] = tree

    def _fill(self, key, rows):
        tree = self.trees[key]
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", tk.END, values=row)

    def refresh(self):
        """只读刷新：不触碰账本内容。"""
        if not os.path.exists(self.ledger_path):
            self.status.set("账本不存在")
            return
        batches = read_only_query(
            self.ledger_path,
            "SELECT batch_id, worker_id, param_version, status, adopted_digest"
            " FROM batches ORDER BY batch_id")
        self._fill("batches", [
            (b["batch_id"], b["worker_id"], b["param_version"], b["status"],
             (b["adopted_digest"] or "-")[:16]) for b in batches])
        receipts = read_only_query(
            self.ledger_path,
            "SELECT batch_id, request_id, src_format, adopted, reject_reason,"
            " filled, fill_basis FROM receipts"
            " ORDER BY batch_id, request_id, src_file, src_line")
        self._fill("receipts", [
            (r["batch_id"], r["request_id"], f"r{r['src_format']}",
             "采纳" if r["adopted"] else "否决", r["reject_reason"] or "-",
             r["filled"], r["fill_basis"]) for r in receipts])
        inv = read_only_query(
            self.ledger_path,
            "SELECT batch_id, request_id, step, reason FROM invalidations"
            " ORDER BY batch_id, request_id, step")
        self._fill("invalidations", [
            (r["batch_id"], r["request_id"], r["step"], r["reason"]) for r in inv])
        meta = read_only_query(self.ledger_path,
                               "SELECT key, value FROM meta ORDER BY key")
        rows = [(m["key"], m["value"][:80]) for m in meta]
        for name in Ledger.list_snapshots(self.snapshot_dir):
            rows.append(("snapshot", name))
        self._fill("ledger", rows)
        self._refresh_workers()
        self.status.set("已刷新（只读）")

    def _refresh_workers(self):
        rows = []
        try:
            cfg = load_config(self.config_path)
            quota = cfg["scheduler"]["per_worker_limit"]
            stats = self.scheduler.stats_snapshot()["workers"] if self.scheduler else {}
            for wid in sorted(cfg["workers_by_id"]):
                st = stats.get(wid, {"in_flight": 0, "remaining": quota})
                rows.append((wid, st["in_flight"], st["remaining"]))
        except Exception:
            pass
        self._fill("workers", rows)

    def _poll_stats(self):
        if self.scheduler is not None:
            self._refresh_workers()
        self.after(300, self._poll_stats)

    def run_pipeline_async(self):
        if self.scheduler is not None:
            self.status.set("流水线运行中")
            return
        self.status.set("流水线运行中…")

        def _run():
            from .scheduler import Scheduler
            cfg = load_config(self.config_path)
            cfg["__config_path__"] = os.path.abspath(self.config_path)
            ports = load_ports(self.ports_path)
            try:
                self.scheduler = Scheduler(cfg, ports)
                run_pipeline(self.config_path, self.ports_path, self.ledger_path,
                             scheduler=self.scheduler)
                self.status.set("流水线完成")
            except Exception as exc:
                self.status.set(f"流水线失败: {exc}")
            finally:
                if self.scheduler is not None:
                    self.scheduler.shutdown()
                self.scheduler = None

        threading.Thread(target=_run, daemon=True).start()


def main():
    import argparse
    parser = argparse.ArgumentParser(prog="ui")
    parser.add_argument("--ledger", default=os.path.join(ROOT, "state", "ledger.db"))
    parser.add_argument("--config", default=os.path.join(ROOT, "config", "pipeline.json"))
    parser.add_argument("--ports", default=os.path.join(ROOT, "config", "ports.json"))
    args = parser.parse_args()
    app = DiagApp(args.ledger, args.config, args.ports)
    app.mainloop()


if __name__ == "__main__":
    main()
