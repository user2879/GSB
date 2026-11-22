"""Tkinter 诊断界面(只读): 刷新不写入账本, 渲染顺序稳定。

展示: 故障域拓扑、实例健康状态与迁移原因、域隔离状态与冷却剩余、
指令排队与并发占用、当前版本与快照列表。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _load_view(workdir: str) -> dict:
    db = Path(workdir) / "ledger.db"
    view = {"transitions": [], "commands": [], "meta": {}, "cursor": (0, 0),
            "marks": {}, "snapshots": []}
    if db.exists():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            view["meta"] = {r["key"]: r["value"] for r in
                            conn.execute("SELECT key, value FROM meta")}
            view["transitions"] = [dict(r) for r in conn.execute(
                "SELECT * FROM transitions ORDER BY id")]
            view["commands"] = [dict(r) for r in conn.execute(
                "SELECT * FROM commands ORDER BY id")]
            row = conn.execute(
                "SELECT event_seq, event_ts_ms FROM cursor WHERE id=1").fetchone()
            if row:
                view["cursor"] = (row["event_seq"], row["event_ts_ms"])
            view["marks"] = {r["domain"]: r["isolated_event_ts"] for r in
                             conn.execute("SELECT * FROM domain_marks ORDER BY domain")}
        finally:
            conn.close()
    from .snapshot import list_snapshots
    view["snapshots"] = list_snapshots(workdir)
    return view


def launch(workdir: str) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("faultdiag 故障隔离诊断")
    root.geometry("980x640")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    trees: dict[str, ttk.Treeview] = {}

    def make_tree(tab: str, columns: tuple[str, ...]) -> ttk.Treeview:
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab)
        tree = ttk.Treeview(frame, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=140, anchor="w")
        tree.pack(fill="both", expand=True)
        trees[tab] = tree
        return tree

    make_tree("故障域拓扑", ("domain", "instance", "health", "weight"))
    make_tree("迁移记录", ("id", "event_seq", "kind", "entity", "from", "to", "reason"))
    make_tree("域隔离与冷却", ("domain", "state", "cooldown_remaining_ms"))
    make_tree("指令与并发", ("cmd_seq", "instance", "action", "attempt",
                           "status", "wait_ms"))
    make_tree("版本与快照", ("key", "value"))

    def refresh() -> None:
        view = _load_view(workdir)
        meta = view["meta"]
        health: dict[str, str] = {}
        domain_state: dict[str, str] = {}
        for tr in view["transitions"]:
            if tr["kind"] == "instance":
                health[tr["entity"]] = tr["to_state"]
            else:
                domain_state[tr["entity"]] = tr["to_state"]
        for tree in trees.values():
            tree.delete(*tree.get_children())
        topo = trees["故障域拓扑"]
        domains = sorted({k.split(":", 1)[1] for k in meta
                          if k.startswith("cooldown:")} | set(domain_state))
        inst_domain: dict[str, str] = {}
        for cmd in view["commands"]:
            inst_domain.setdefault(cmd["instance"], cmd["domain"])
        for domain in domains:
            members = sorted(i for i, d in inst_domain.items() if d == domain) \
                      or sorted(health)
            for iid in members:
                if inst_domain.get(iid, domain) != domain:
                    continue
                topo.insert("", "end", values=(
                    domain, iid, health.get(iid, "unknown"), ""))
        mig = trees["迁移记录"]
        for tr in view["transitions"]:
            mig.insert("", "end", values=(
                tr["id"], tr["event_seq"], tr["kind"], tr["entity"],
                tr["from_state"], tr["to_state"], tr["reason"]))
        cool = trees["域隔离与冷却"]
        _, cursor_ts = view["cursor"]
        for domain in domains:
            remaining = 0
            if domain in view["marks"]:
                cooldown = int(meta.get(f"cooldown:{domain}", "0"))
                remaining = max(0, cooldown - (cursor_ts - view["marks"][domain]))
            cool.insert("", "end", values=(
                domain, domain_state.get(domain, "normal"), remaining))
        que = trees["指令与并发"]
        for cmd in view["commands"]:
            que.insert("", "end", values=(
                cmd["cmd_seq"], cmd["instance"], cmd["action"], cmd["attempt"],
                cmd["status"], cmd["start_mono_ms"] - cmd["enqueue_mono_ms"]))
        ver = trees["版本与快照"]
        for key in sorted(meta):
            ver.insert("", "end", values=(key, meta[key]))
        ver.insert("", "end", values=("cursor", view["cursor"][0]))
        for snap in view["snapshots"]:
            ver.insert("", "end", values=(
                f"snapshot:{snap['name']}",
                f"{'readonly ' if snap['readonly'] else ''}{snap['size']}B"))

    ttk.Button(root, text="刷新(只读)", command=refresh).pack(pady=4)
    refresh()
    root.mainloop()
