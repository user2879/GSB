"""Tkinter dashboard for live fan-out sessions and snapshot browsing.

The module imports Tkinter lazily so the rest of the package works on
headless machines without a Tk installation; launching the UI without Tk
raises ``ImportError`` with an actionable message.  All rendered lists use
explicit stable ordering (scene order / sorted snapshot filenames), never
dict or hash iteration.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REFRESH_MS = 700


def _read_state(run_dir: Path) -> dict[str, Any]:
    state: dict[str, Any] = {"snapshots": [], "ledger_meta": {}, "latest": None}
    snapshot_dir = run_dir / "snapshots"
    if snapshot_dir.exists():
        state["snapshots"] = sorted(p.name for p in snapshot_dir.glob("snapshot-*.json"))
        finals = sorted(snapshot_dir.glob("snapshot-*-final.json"))
        if finals:
            try:
                envelope = json.loads(finals[-1].read_text(encoding="utf-8"))
                state["latest"] = envelope["body"]
            except (json.JSONDecodeError, KeyError):
                state["latest"] = None
    return state


def launch_dashboard(run_dir: str | Path) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:  # headless runtime without Tk
        raise ImportError(
            "Tkinter is unavailable. On Windows install the official python.org "
            "build (Tk is bundled); on Debian/Ubuntu install python3-tk."
        ) from exc

    run_dir = Path(run_dir)
    root = tk.Tk()
    root.title(f"Fan-Out Ledger Dashboard - {run_dir}")
    root.geometry("980x680")

    header = ttk.Label(root, text="", font=("TkDefaultFont", 10, "bold"))
    header.pack(anchor="w", padx=10, pady=(10, 4))

    counters = ttk.Label(root, text="")
    counters.pack(anchor="w", padx=10)

    columns = ("node", "samples", "success_rate", "p95_ms", "kbps", "outcomes")
    tree = ttk.Treeview(root, columns=columns, show="headings", height=10)
    for col, width in zip(columns, (120, 90, 130, 90, 110, 260)):
        tree.heading(col, text=col)
        tree.column(col, width=width, anchor="w")
    tree.pack(fill="x", padx=10, pady=8)

    slo = ttk.Label(root, text="", foreground="#0a5")
    slo.pack(anchor="w", padx=10)

    ttk.Label(root, text="Snapshots (stable name order):").pack(
        anchor="w", padx=10, pady=(8, 0))
    snapshot_list = tk.Listbox(root, height=8)
    snapshot_list.pack(fill="both", expand=True, padx=10, pady=(2, 10))

    def refresh() -> None:
        state = _read_state(run_dir)
        body = state.get("latest")
        if not body:
            header.config(text=f"run_dir={run_dir} (no completed snapshot yet)")
            counters.config(text="running/queued/budget: waiting for data ...")
            tree.delete(*tree.get_children())
            slo.config(text="SLO: pending")
        else:
            metrics = body.get("metrics", {})
            g = metrics.get("global", {})
            header.config(
                text=(f"scene_version={body.get('scene_version')}  "
                      f"status={body.get('status')}  "
                      f"payload={str(body.get('payload_digest'))[:16]}...")
            )
            counters.config(
                text=(f"global budget used={body.get('budget_global_ms')} ms | "
                      f"samples={g.get('samples')} | outcome counts={g.get('outcome_counts')}")
            )
            tree.delete(*tree.get_children())
            for block in metrics.get("nodes", []):  # scene order embedded in snapshot
                outcome_text = ",".join(
                    f"{key}:{value}"
                    for key, value in sorted(block.get("outcome_counts", {}).items())
                    if value
                )
                tree.insert("", "end", values=(
                    block.get("node_id"), block.get("samples"),
                    f"{block.get('success_rate_scaled')}/{block.get('success_rate_scale')}",
                    block.get("p95_latency_ms"),
                    f"{block.get('throughput_kbps_scaled')}/{block.get('throughput_scale')}",
                    outcome_text,
                ))
            judgment = body.get("judgment", {})
            if judgment.get("passed"):
                slo.config(text="SLO: PASSED", foreground="#0a5")
            else:
                reasons = " | ".join(
                    f"{r.get('type')} trigger={json.dumps({k: v for k, v in r.items() if 'value' in k}, sort_keys=True)}"
                    for r in judgment.get("reasons", [])
                )
                slo.config(text=f"SLO: FAILED -> {reasons}", foreground="#c22")
        snapshot_list.delete(0, tk.END)
        for name in state["snapshots"]:
            snapshot_list.insert(tk.END, name)
        root.after(REFRESH_MS, refresh)

    refresh()
    root.mainloop()
