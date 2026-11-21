"""Tkinter diagnostic view: fast snapshot view separated from authoritative recompute.

Tkinter is imported lazily so every non-GUI feature works on interpreters
without Tk support. The window shows snapshot-derived statistics (fast view)
and runs the authoritative recompute only when the button is pressed.
"""

import json

from .errors import HotGrayError
from .snapshot import authoritative_recompute, compare_snapshots, snapshot_digest
from .spec import load_declaration
from .storage import LedgerStore


def _snapshot_rows(snapshot):
    rows = []
    for rule_id, rule_body in sorted(snapshot["body"].items()):
        for destination in ("gray", "stable"):
            group = rule_body.get(destination)
            if not group:
                continue
            rows.append({
                "rule_id": rule_id,
                "destination": destination,
                "hit_count": group["hit_count"],
                "gray_ratio_bp": rule_body["gray_ratio_bp"],
                "batch_count": len(group["batch_ids"]),
                "elapsed_ms_avg": group["elapsed_ms_avg"],
                "elapsed_ms_max": group["elapsed_ms_max"],
                "param_versions": ",".join(
                    str(v["backend_seq"]) for v in group["param_versions"]
                ),
                "splits": len(group["splits"]),
            })
    return rows


def build_view_data(db_path):
    store = LedgerStore(db_path)
    try:
        snapshot = store.load_latest_snapshot()
        rebuilds = store.rebuilds()
        if snapshot is None:
            return {"rows": [], "snapshot": None, "rebuilds": rebuilds}
        return {"rows": _snapshot_rows(snapshot), "snapshot": snapshot,
                "rebuilds": rebuilds}
    finally:
        store.close()


def run_authoritative(declaration_path, db_path):
    declaration = load_declaration(declaration_path)
    store = LedgerStore(db_path)
    try:
        snapshot = store.load_latest_snapshot()
        if snapshot is None:
            raise HotGrayError("no persisted snapshot")
        recomputed = authoritative_recompute(
            declaration, store, snapshot["ledger_seq_from"],
            snapshot["ledger_seq_to"],
        )
        compare_snapshots(snapshot["body"], recomputed["groups"])
        digest = snapshot_digest(
            recomputed["groups"], snapshot["param_versions"],
            snapshot["ledger_seq_from"], snapshot["ledger_seq_to"],
        )
        if digest != snapshot["digest"]:
            raise HotGrayError("snapshot envelope digest mismatch")
        return {"verdict": "byte_identical",
                "range": [snapshot["ledger_seq_from"],
                          snapshot["ledger_seq_to"]]}
    finally:
        store.close()


def launch_gui(declaration_path, db_path):
    import tkinter as tk
    from tkinter import messagebox, ttk

    root = tk.Tk()
    root.title("hotgray 灰度快照诊断")
    root.geometry("980x560")

    header = ttk.Frame(root, padding=8)
    header.pack(fill="x")
    ttk.Label(header, text="快速视图（直接读取持久化快照，非权威结果）",
              font=("Segoe UI", 11, "bold")).pack(side="left")

    columns = ("rule_id", "destination", "hit_count", "gray_ratio_bp",
               "batch_count", "elapsed_ms_avg", "elapsed_ms_max",
               "param_versions", "splits")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    headings = {
        "rule_id": "规则", "destination": "去向", "hit_count": "命中数",
        "gray_ratio_bp": "灰度占比(‱)", "batch_count": "批次数",
        "elapsed_ms_avg": "平均耗时(ms)", "elapsed_ms_max": "最大耗时(ms)",
        "param_versions": "参数版本链", "splits": "拆批次数",
    }
    for column in columns:
        tree.heading(column, text=headings[column])
        tree.column(column, width=110, anchor="center")
    tree.pack(fill="both", expand=True, padx=8, pady=8)

    meta = tk.Text(root, height=8, wrap="word")
    meta.pack(fill="x", padx=8)

    def refresh():
        data = build_view_data(db_path)
        tree.delete(*tree.get_children())
        for row in data["rows"]:
            tree.insert("", "end", values=tuple(row[c] for c in columns))
        meta.delete("1.0", "end")
        if data["snapshot"]:
            snap = data["snapshot"]
            meta.insert("end", json.dumps({
                "snapshot_id": snap["snapshot_id"],
                "ledger_range": [snap["ledger_seq_from"],
                                 snap["ledger_seq_to"]],
                "digest": snap["digest"],
                "incoming_during_rebuild": snap["incoming_during_rebuild"],
                "rebuilds": data["rebuilds"],
            }, ensure_ascii=False, indent=2))

    def authoritative():
        try:
            verdict = run_authoritative(declaration_path, db_path)
            messagebox.showinfo("权威重算",
                                f"逐字节一致：{json.dumps(verdict)}")
        except HotGrayError as exc:
            messagebox.showerror(exc.code,
                                 f"{exc}\n字段: {exc.field}")

    buttons = ttk.Frame(root, padding=8)
    buttons.pack(fill="x")
    ttk.Button(buttons, text="刷新快速视图", command=refresh).pack(side="left")
    ttk.Button(buttons, text="权威重算（逐请求重放比对）",
               command=authoritative).pack(side="left", padx=8)

    refresh()
    root.mainloop()
