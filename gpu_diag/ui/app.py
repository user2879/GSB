"""Tkinter fast-view diagnostic window.

The view reads only the cached ledger/snapshot; it never recomputes scheduling.
Tkinter is imported lazily so the rest of the workbench runs headless on
machines without a Tk runtime (the Windows standard installer includes Tk).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def read_cached_view(state_dir: str | Path) -> dict[str, Any]:
    state = Path(state_dir)
    view: dict[str, Any] = {"pools": [], "running": [], "queued": [],
                            "preemptions": [], "unknown": [], "rounds": 0}
    snapshot_path = state / "snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        view["rounds"] = len(snapshot.get("rounds", []))
        view["preemptions"] = snapshot.get("preemption_sequence", [])
        unknown = [sid for sid, outcome in
                   snapshot.get("shard_outcomes", {}).items()
                   if outcome.get("state") == "UNKNOWN"]
        view["unknown"] = unknown
        if snapshot.get("rounds"):
            latest = snapshot["rounds"][-1]
            levels = latest["water_levels"]
            for pid in sorted(k for k in levels if not k.startswith("__")):
                view["pools"].append({"pool_id": pid, **levels[pid]})
            view["global"] = levels.get("__global__")

    db_path = state / "ledger.sqlite"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT round_no, shard_id, state, pool_id FROM shard_traces "
            "ORDER BY round_no DESC, shard_id").fetchall()
        latest_round = rows[0]["round_no"] if rows else 0
        for row in rows:
            if row["round_no"] != latest_round:
                break
            if row["state"] == "RUNNING":
                view["running"].append(row["shard_id"])
            if row["state"] == "QUEUED":
                view["queued"].append(row["shard_id"])
        for row in conn.execute("SELECT shard_id, reason FROM unknown_shards"):
            view["unknown"].append({"shard_id": row["shard_id"],
                                    "reason": row["reason"]})
        conn.close()
    return view


def render_headless(view: dict[str, Any]) -> str:
    lines = ["GPU Diagnostic Fast View (headless fallback)",
             f"committed rounds: {view['rounds']}"]
    for pool in view.get("pools", []):
        lines.append(
            f"  {pool['pool_id']}: gpu {pool['gpu_used']}/{pool['gpu_capacity']} "
            f"free={pool['gpu_free']} ratio_q={pool['water_ratio_q']} "
            f"tasks={pool['running_task_count']} mem_free_mib={pool['memory_free_mib']}")
    if view.get("global"):
        lines.append(f"  GLOBAL: gpu {view['global']['gpu_used']}/"
                     f"{view['global']['gpu_capacity']} "
                     f"ratio_q={view['global']['water_ratio_q']}")
    lines.append(f"  running: {', '.join(view['running']) or '-'}")
    lines.append(f"  queued: {', '.join(view['queued']) or '-'}")
    lines.append(f"  preemptions: {len(view['preemptions'])}")
    lines.append(f"  unknown: {view['unknown'] or '-'}")
    return "\n".join(lines)


def launch_ui(state_dir: str | Path) -> None:
    view = read_cached_view(state_dir)
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # pragma: no cover - environment dependent
        print(render_headless(view))
        print(f"[info] Tk unavailable ({exc}); rendered headless text view.")
        return

    root = tk.Tk()
    root.title("GPU Quota & Preemption Diagnostic Workbench")
    root.geometry("900x640")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    pool_tab = ttk.Frame(notebook)
    notebook.add(pool_tab, text="Pool Water Levels")
    columns = ("pool", "gpu_used", "gpu_free", "ratio_q", "tasks", "mem_free")
    tree = ttk.Treeview(pool_tab, columns=columns, show="headings")
    for col, text in zip(columns, ("Pool", "GPU Used", "GPU Free",
                                   "Ratio (fixed point)", "Tasks",
                                   "Mem Free MiB")):
        tree.heading(col, text=text)
    tree.pack(fill="both", expand=True)
    for pool in view.get("pools", []):
        tree.insert("", "end", values=(
            pool["pool_id"], pool["gpu_used"], pool["gpu_free"],
            pool["water_ratio_q"], pool["running_task_count"],
            pool["memory_free_mib"]))
    if view.get("global"):
        tree.insert("", "end", values=(
            "__global__", view["global"]["gpu_used"], view["global"]["gpu_free"],
            view["global"]["water_ratio_q"], "-", "-"))

    status_tab = ttk.Frame(notebook)
    notebook.add(status_tab, text="Shards / Preemptions / Unknown")
    text = tk.Text(status_tab, wrap="word")
    text.pack(fill="both", expand=True)
    text.insert("end", render_headless(view))
    text.configure(state="disabled")

    root.mainloop()
