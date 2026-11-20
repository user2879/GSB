"""Tkinter diagnostic UI with a headless console fallback.

The UI renders resource-pool quota usage, task/shard states, the waiting queue,
the deterministic preemption sequence, fixed-point statistics with error bounds
and the current versions/snapshot list.  Every refresh rebuilds a read-only view
from the engine and ledger; it never opens a write transaction, so refreshing
cannot change ledger contents.  When Tkinter is unavailable (for example a
headless CI host) :func:`render_console` provides the same ordered information.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from .engine import SimulationEngine
from .stats import full_report

try:  # Tkinter is optional at import time so tests run on headless hosts.
    import tkinter as tk
    from tkinter import ttk
    _TK_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    tk = None
    ttk = None
    _TK_AVAILABLE = False


def build_view_model(engine: SimulationEngine,
                     snapshot_dir: Optional[str] = None) -> dict:
    """Assemble a stable, read-only view model with deterministic ordering."""
    declaration = engine.declaration
    pool_rows = []
    usage_by_pool = {row.pool_id: row for row in engine._pool_usage()}
    for pool in declaration.pools:
        usage = usage_by_pool.get(pool.pool_id)
        pool_rows.append({
            "pool_id": pool.pool_id,
            "gpu": f"{usage.gpu_busy if usage else 0}/{pool.gpu_count}",
            "memory_mib": (
                f"{usage.memory_busy_mib if usage else 0}/"
                f"{pool.total_memory_mib}"),
            "shares": (
                f"{usage.shares_busy if usage else 0}/{pool.quota_shares}"),
            "concurrency": (
                f"{usage.concurrent_tasks if usage else 0}/"
                f"{pool.max_concurrent_tasks}"),
        })
    shard_rows = []
    for item in engine.shard_summary():
        shard_rows.append({
            "shard": f"{item['task_id']}#S{item['shard_index']:02d}",
            "state": item["state"],
            "version": item["version"],
            "pool": item["pool_id"] or "-",
            "started_ms": item["started_ms"],
            "ended_ms": item["ended_ms"],
            "wait_ms": item["wait_duration_ms"],
            "digest": (item["output_digest"] or "")[:12],
        })
    waiting = []
    for record in engine.rt.rounds:
        for shard_id in record.waiting:
            if shard_id not in waiting:
                waiting.append(shard_id)
    preemptions = [record.to_dict() for record in engine.rt.preemptions]
    snapshots = []
    if snapshot_dir and os.path.isdir(snapshot_dir):
        snapshots = sorted(
            name for name in os.listdir(snapshot_dir)
            if name.endswith(".json"))
    return {
        "versions": {
            "pool_version": declaration.pool_version,
            "task_version": declaration.task_version,
            "config_version": declaration.config_version,
            "fixed_point_scale": declaration.config.fixed_point_scale,
        },
        "pools": pool_rows,
        "shards": shard_rows,
        "waiting_queue": waiting,
        "preemptions": preemptions,
        "statistics": full_report(engine),
        "snapshots": snapshots,
    }


def render_console(engine: SimulationEngine,
                   snapshot_dir: Optional[str] = None) -> str:
    model = build_view_model(engine, snapshot_dir)
    lines: List[str] = []
    lines.append("== GPU scheduler diagnostics (read-only) ==")
    lines.append("versions: " + json.dumps(model["versions"], sort_keys=True))
    lines.append("-- pool quota --")
    for row in model["pools"]:
        lines.append(
            f" {row['pool_id']}: gpu {row['gpu']} mem {row['memory_mib']} "
            f"shares {row['shares']} tasks {row['concurrency']}")
    lines.append("-- shard states --")
    for row in model["shards"]:
        lines.append(
            f" {row['shard']} {row['state']} v{row['version']} "
            f"pool={row['pool']} wait={row['wait_ms']} digest={row['digest']}")
    lines.append("-- waiting queue --")
    lines.append(" " + (", ".join(model["waiting_queue"]) or "(empty)"))
    lines.append("-- preemptions --")
    for pre in model["preemptions"]:
        lines.append(
            f" round {pre['round_no']} t={pre['clock_ms']} "
            f"{pre['trigger_task_id']} -> {pre['victim_task_id']}"
            f"#S{pre['victim_shard_index']:02d} released "
            f"gpu={pre['released_gpu']} mem={pre['released_memory_mib']}")
    stats = model["statistics"]
    lines.append("-- fixed-point stats --")
    lines.append(" " + json.dumps(stats["aggregate"], sort_keys=True))
    lines.append(" " + json.dumps(stats["task_completion"], sort_keys=True))
    lines.append("-- snapshots --")
    lines.append(" " + (", ".join(model["snapshots"]) or "(none)"))
    return "\n".join(lines)


def launch_gui(engine: SimulationEngine,
               snapshot_dir: Optional[str] = None) -> None:  # pragma: no cover
    if not _TK_AVAILABLE:
        raise RuntimeError(
            "Tkinter is not available in this Python; use render_console")
    root = tk.Tk()
    root.title("GPU Quota and Shard Scheduler Diagnostics")
    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, sticky="nsew")

    model = build_view_model(engine, snapshot_dir)
    _table_tab(notebook, "Pools", model["pools"],
               ("pool_id", "gpu", "memory_mib", "shares", "concurrency"))
    _table_tab(notebook, "Shards", model["shards"],
               ("shard", "state", "version", "pool", "started_ms",
                "ended_ms", "wait_ms", "digest"))
    text_tab = ttk.Frame(notebook)
    notebook.add(text_tab, text="Queues / Stats")
    text = tk.Text(text_tab, wrap="none")
    text.insert("1.0", json.dumps({
        "waiting_queue": model["waiting_queue"],
        "preemptions": model["preemptions"],
        "statistics": model["statistics"],
        "versions": model["versions"],
        "snapshots": model["snapshots"],
    }, indent=2, sort_keys=True, ensure_ascii=False))
    text.configure(state="disabled")
    text.grid(row=0, column=0, sticky="nsew")
    root.mainloop()


def _table_tab(notebook, title: str, rows: List[dict],
               columns) -> None:  # pragma: no cover
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=title)
    tree = ttk.Treeview(frame, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
        tree.column(column, width=120, anchor="w")
    for row in rows:
        tree.insert("", "end", values=[row[column] for column in columns])
    tree.grid(row=0, column=0, sticky="nsew")
