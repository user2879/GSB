"""Tkinter diagnostic dashboard with a headless text fallback.

All rendering reads from snapshots/ledger via the supplied read-only state
provider; refreshing never mutates the ledger.  Ordering is fixed (sorted ids)
so repeated renders are stable and reviewable.
"""
from __future__ import annotations


def headless_dashboard(state: dict) -> str:
    lines: list[str] = []
    lines.append("Lease diagnostic dashboard (headless text view)")
    lines.append(f"versions: pool_set={state['versions']['pool_set_version']} "
                 f"rollback={state['versions']['rollback_version']} "
                 f"scale={state['scale']}")
    lines.append("")
    lines.append("Pools (effective leases / capacity, fixed-point watermark):")
    for mark in sorted(state["watermarks"], key=lambda item: item["scope"]):
        if not mark["scope"].startswith("pool:"):
            continue
        lines.append(
            f"  {mark['pool_id']:<6} active={mark['effective_leases']}/"
            f"{mark['capacity']} ratio_fp={mark['ratio_fp']} "
            f"queue={mark['queue_length']} trunc={mark['truncation']} "
            f"err<={mark['value_error_ulp']}ulp")
    for mark in state["watermarks"]:
        if mark["scope"] == "global":
            lines.append(
                f"  GLOBAL active={mark['effective_leases']}/"
                f"{mark['capacity']} ratio_fp={mark['ratio_fp']} "
                f"queue={mark['queue_length']}")
    lines.append("")
    lines.append("Ranking (error-influenced decisions flagged):")
    for row in state["ranking"]:
        flag = " [ERROR-INFLUENCED]" if row["error_influenced"] else ""
        lines.append(
            f"  #{row['rank']} {row['pool_id']} ratio_fp={row['ratio_fp']}"
            f"{flag}")
    lines.append("")
    lines.append("Leases (stable id order):")
    for lease_id in sorted(state["leases"]):
        lease = state["leases"][lease_id]
        lines.append(
            f"  {lease_id}: key={lease['key']} pool={lease['pool_id']} "
            f"state={lease['state']} rev={lease['revision']} "
            f"renew_count={lease['renew_count']} expire_ms={lease['expire_ms']}"
            + (f" revoked_by={lease['revoked_by_lease_id']}"
               if lease.get("revoked_by_lease_id") else "")
            + (f" rolledback_rev={lease['rolledback_rev']}"
               if lease.get("rolledback_rev") is not None else ""))
    lines.append("")
    lines.append(f"Queue observations: {len(state['queue'])}")
    for item in state["queue"][:20]:
        lines.append(f"  {item}")
    lines.append("")
    lines.append(f"Committed batches: {state['batches']}")
    return "\n".join(lines)


def launch_tk(state_provider, refresh_ms: int = 1000):  # pragma: no cover
    """Launch the Tk window; exercised on Windows with the official Python."""
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Lease & Concurrency Diagnostic")
    notebook = ttk.Notebook(root)
    notebook.grid(sticky="nsew", padx=8, pady=8)

    pool_tab = ttk.Frame(notebook)
    lease_tab = ttk.Frame(notebook)
    queue_tab = ttk.Frame(notebook)
    notebook.add(pool_tab, text="Pools & Watermarks")
    notebook.add(lease_tab, text="Leases")
    notebook.add(queue_tab, text="Queue & Batches")

    pool_text = tk.Text(pool_tab, width=90, height=24)
    pool_text.grid(sticky="nsew")
    lease_text = tk.Text(lease_tab, width=100, height=24)
    lease_text.grid(sticky="nsew")
    queue_text = tk.Text(queue_tab, width=100, height=24)
    queue_text.grid(sticky="nsew")
    status = tk.StringVar(value="ready")
    ttk.Label(root, textvariable=status).grid(sticky="w", padx=8)

    def refresh() -> None:
        state = state_provider()
        pool_text.delete("1.0", tk.END)
        pool_text.insert(tk.END, headless_dashboard(state))
        lease_text.delete("1.0", tk.END)
        for lease_id in sorted(state["leases"]):
            lease_text.insert(tk.END, f"{lease_id}: {state['leases'][lease_id]}\n")
        queue_text.delete("1.0", tk.END)
        queue_text.insert(tk.END,
                          f"queue={state['queue']}\nbatches={state['batches']}\n"
                          f"versions={state['versions']}")
        status.set(f"last refresh: ledger unchanged, "
                   f"{len(state['leases'])} leases")
        root.after(refresh_ms, refresh)

    refresh()
    root.mainloop()


def render(state: dict, use_tk: bool = False) -> str | None:
    if use_tk:  # pragma: no cover - GUI path
        launch_tk(lambda: state)
        return None
    return headless_dashboard(state)
