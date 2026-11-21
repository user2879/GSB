"""Tkinter diagnostics UI (read-only over the ledger).

Shows pools/instances with health state and reason, consecutive counters,
the probe dispatch timeline, budget occupancy, the current spec version
and the ledger event list. Rendering order is stable (sorted keys) and
refreshing never writes to the ledger.
"""

from __future__ import annotations

import sqlite3
import tkinter as tk
from tkinter import ttk

REFRESH_MS = 1000


class WorkbenchUI:
    def __init__(self, root, ledger_path, spec_path=None):
        self.root = root
        self.ledger_path = ledger_path
        root.title("LB Health-Check Diagnostic Workbench")
        root.geometry("1100x640")

        self.meta_var = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.meta_var,
                  font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", padx=8, pady=(6, 2))

        panes = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = ttk.Frame(panes)
        panes.add(left, weight=3)
        ttk.Label(left, text="Pools / Instances").pack(anchor="w")
        self.tree = ttk.Treeview(
            left, columns=("state", "reason", "succ", "fail", "last_seq"),
            show="tree headings")
        for col, text, w in (("state", "State", 90),
                             ("reason", "Reason", 130),
                             ("succ", "Succ", 50),
                             ("fail", "Fail", 50),
                             ("last_seq", "Seq", 60)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(panes)
        panes.add(right, weight=4)
        ttk.Label(right, text="Probe dispatch timeline").pack(anchor="w")
        self.timeline = ttk.Treeview(
            right, columns=("seq", "round", "pool", "instance", "wait",
                            "outcome", "reason"),
            show="headings", height=12)
        for col, text, w in (("seq", "Seq", 50), ("round", "Round", 55),
                             ("pool", "Pool", 90),
                             ("instance", "Instance", 80),
                             ("wait", "Wait ms", 80),
                             ("outcome", "Outcome", 80),
                             ("reason", "Reason", 120)):
            self.timeline.heading(col, text=text)
            self.timeline.column(col, width=w, anchor="w")
        self.timeline.pack(fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        ttk.Label(bottom, text="Budget / ledger events").pack(anchor="w")
        self.budget_text = tk.Text(bottom, height=8, wrap="none")
        self.budget_text.pack(fill=tk.BOTH, expand=True)
        self.budget_text.configure(state="disabled")

        self.refresh()

    # -- data (read-only) ---------------------------------------------------
    def _query(self, sql, args=()):
        db = sqlite3.connect(f"file:{self.ledger_path}?mode=ro", uri=True)
        try:
            return db.execute(sql, args).fetchall()
        finally:
            db.close()

    def refresh(self):
        try:
            self._refresh_once()
        except sqlite3.Error as exc:
            self.meta_var.set(f"ledger not readable yet: {exc}")
        self.root.after(REFRESH_MS, self.refresh)

    def _refresh_once(self):
        version_row = self._query(
            "SELECT value FROM meta WHERE key='spec_version'")
        version = int(version_row[0][0]) if version_row else 0
        cursor_row = self._query(
            "SELECT value FROM meta WHERE key='cursor'")
        cursor = cursor_row[0][0] if cursor_row else "-"
        events = self._query("SELECT COUNT(*) FROM events")[0][0]
        self.meta_var.set(
            f"spec version: {version}   progress cursor: {cursor}   "
            f"ledger events: {events}   ledger: {self.ledger_path}")

        self.tree.delete(*self.tree.get_children())
        rows = self._query(
            "SELECT pool_id,instance_id,state,last_reason,consec_success,"
            "consec_failure,last_probe_seq FROM instance_view "
            "WHERE version=? AND stale=0 ORDER BY pool_id,instance_id",
            (version,))
        pools = {}
        for pool_id, iid, state, reason, succ, fail, seq in rows:
            node = pools.get(pool_id)
            if node is None:
                node = self.tree.insert("", "end", text=pool_id,
                                        values=("", "", "", "", ""),
                                        open=True)
                pools[pool_id] = node
            self.tree.insert(node, "end", text=iid,
                             values=(state, reason, succ, fail, seq))

        self.timeline.delete(*self.timeline.get_children())
        for seq, rnd, pool, iid, wait, outcome, reason in self._query(
                "SELECT seq,round,pool_id,instance_id,wait_ms,outcome,"
                "reason FROM probes WHERE version=? ORDER BY seq",
                (version,)):
            self.timeline.insert("", "end", values=(
                seq, rnd, pool, iid, f"{wait:.1f}", outcome, reason))

        lines = []
        lines.append("transitions (stable order):")
        for tseq, iid, frm, to, cat in self._query(
                "SELECT probe_seq,instance_id,from_state,to_state,"
                "reason_category FROM transitions WHERE version=? "
                "ORDER BY probe_seq, id", (version,)):
            lines.append(f"  seq={tseq} {iid}: {frm} -> {to} ({cat})")
        lines.append("")
        lines.append("budget occupancy (per 1s window of probe starts):")
        for (cnt,) in self._query(
                "SELECT COUNT(*) FROM probes WHERE version=? GROUP BY "
                "CAST(started_ms/1000.0 AS INTEGER) ORDER BY 1 DESC "
                "LIMIT 5", (version,)):
            lines.append(f"  peak starts/sec window: {cnt}")
        stale = self._query(
            "SELECT DISTINCT version FROM instance_view WHERE stale=1 "
            "ORDER BY version")
        lines.append(f"stale versions: {[r[0] for r in stale]}")
        self.budget_text.configure(state="normal")
        self.budget_text.delete("1.0", "end")
        self.budget_text.insert("1.0", "\n".join(lines))
        self.budget_text.configure(state="disabled")


def run_ui(ledger_path, spec_path=None):
    root = tk.Tk()
    WorkbenchUI(root, ledger_path, spec_path)
    root.mainloop()
    return 0
