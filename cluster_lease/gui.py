"""Read-only Tkinter diagnostics for the lease watermark ledger."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from .errors import LeaseError
from .ledger import Ledger


class DiagnosticsApp:
    def __init__(self, root: tk.Tk, db_path: str) -> None:
        self.root = root
        self.db_path = db_path
        self.root.title("Cluster Lease Watermark Diagnostics")
        self.root.geometry("1220x760")
        self.ledger = Ledger(db_path, readonly=True)
        self.cluster_ids: list[str] = []
        self._build_widgets()
        self.refresh()

    def _build_widgets(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="SQLite:").pack(side=tk.LEFT)
        self.db_label = ttk.Label(toolbar, text=self.db_path)
        self.db_label.pack(side=tk.LEFT, padx=(6, 18))
        ttk.Button(toolbar, text="Refresh (read-only)", command=self.refresh).pack(side=tk.RIGHT)
        self.digest_label = ttk.Label(toolbar, text="")
        self.digest_label.pack(side=tk.LEFT)

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=4)

        ttk.Label(left, text="Clusters (stable order)").pack(anchor=tk.W)
        self.cluster_tree = ttk.Treeview(
            left,
            columns=("version", "capacity", "used", "watermark"),
            show="tree headings",
            height=20,
        )
        self.cluster_tree.heading("#0", text="cluster")
        self.cluster_tree.heading("version", text="ver")
        self.cluster_tree.heading("capacity", text="cap")
        self.cluster_tree.heading("used", text="used")
        self.cluster_tree.heading("watermark", text="watermark")
        self.cluster_tree.column("#0", width=150)
        self.cluster_tree.column("version", width=45)
        self.cluster_tree.column("capacity", width=45)
        self.cluster_tree.column("used", width=45)
        self.cluster_tree.column("watermark", width=90)
        self.cluster_tree.pack(fill=tk.BOTH, expand=True)
        self.cluster_tree.bind("<<TreeviewSelect>>", lambda _event: self._render_selected())

        notebook = ttk.Notebook(right)
        notebook.pack(fill=tk.BOTH, expand=True)
        self.active_tab = ttk.Frame(notebook)
        self.watermark_tab = ttk.Frame(notebook)
        self.transition_tab = ttk.Frame(notebook)
        self.metadata_tab = ttk.Frame(notebook)
        notebook.add(self.active_tab, text="Active leases")
        notebook.add(self.watermark_tab, text="Watermark curve")
        notebook.add(self.transition_tab, text="Transitions")
        notebook.add(self.metadata_tab, text="Versions & snapshots")
        self.notebook = notebook

        self.active_tree = ttk.Treeview(
            self.active_tab,
            columns=("holder", "state", "priority", "started", "renewed", "expires", "remaining"),
            show="headings",
        )
        for column, heading, width in (
            ("holder", "holder", 150),
            ("state", "state", 85),
            ("priority", "priority", 65),
            ("started", "started ms", 90),
            ("renewed", "last renew ms", 110),
            ("expires", "expires ms", 95),
            ("remaining", "remaining ms", 100),
        ):
            self.active_tree.heading(column, text=heading)
            self.active_tree.column(column, width=width, anchor=tk.W)
        self.active_tree.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.watermark_tab, background="white", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._render_watermark())
        self.watermark_caption = ttk.Label(self.watermark_tab, text="")
        self.watermark_caption.pack(anchor=tk.W, padx=8, pady=4)

        self.transition_tree = ttk.Treeview(
            self.transition_tab,
            columns=("seq", "holder", "before", "after", "reason", "active"),
            show="headings",
        )
        for column, heading, width in (
            ("seq", "event", 70),
            ("holder", "holder", 150),
            ("before", "before", 100),
            ("after", "after", 100),
            ("reason", "reason", 210),
            ("active", "effective active", 120),
        ):
            self.transition_tree.heading(column, text=heading)
            self.transition_tree.column(column, width=width, anchor=tk.W)
        self.transition_tree.pack(fill=tk.BOTH, expand=True)

        self.metadata_text = tk.Text(self.metadata_tab, wrap=tk.WORD)
        self.metadata_text.pack(fill=tk.BOTH, expand=True)

    def refresh(self) -> None:
        selected = self.selected_cluster()
        self.cluster_tree.delete(*self.cluster_tree.get_children())
        self.cluster_ids = sorted(row["cluster_id"] for row in self.ledger.cluster_versions())
        for cluster_id in self.cluster_ids:
            version = self.ledger.cluster_version(cluster_id) or {}
            active = self.ledger.current_active(cluster_id)
            watermarks = self.ledger.current_watermarks(cluster_id)
            watermark = watermarks[-1]["watermark_fixed"] if watermarks else 0
            self.cluster_tree.insert(
                "",
                tk.END,
                iid=cluster_id,
                text=cluster_id,
                values=(
                    version.get("parameter_version", 0),
                    version.get("capacity", 0),
                    len(active),
                    f"{watermark}/{watermarks[-1]['watermark_scale'] if watermarks else 1_000_000}",
                ),
            )
        digest = self.ledger.content_digest()
        self.digest_label.configure(
            text=f"format={self.ledger.get_meta('ledger_format_version', '?')} digest={digest[:24]}…"
        )
        target = selected if selected in self.cluster_ids else (self.cluster_ids[0] if self.cluster_ids else None)
        if target is not None:
            self.cluster_tree.selection_set(target)
            self.cluster_tree.focus(target)
        self._render_selected()

    def selected_cluster(self) -> str | None:
        selection = self.cluster_tree.selection()
        return selection[0] if selection else None

    def _render_selected(self) -> None:
        cluster_id = self.selected_cluster()
        if cluster_id is None:
            return
        progress = self.ledger.progress_for(cluster_id)
        reference_time = int(progress["last_timestamp_ms"])
        self.active_tree.delete(*self.active_tree.get_children())
        for lease in self.ledger.current_active(cluster_id):
            remaining = max(0, int(lease["expires_ms"]) - reference_time)
            self.active_tree.insert(
                "",
                tk.END,
                values=(
                    lease["holder_id"],
                    lease["state"],
                    lease["priority"],
                    lease["started_ms"],
                    lease["last_renewed_ms"],
                    lease["expires_ms"],
                    remaining,
                ),
            )
        self.transition_tree.delete(*self.transition_tree.get_children())
        for decision in self.ledger.current_decisions(cluster_id):
            for transition in decision["payload"].get("transitions", []):
                self.transition_tree.insert(
                    "",
                    tk.END,
                    values=(
                        transition.get("event_seq"),
                        transition.get("holder_id"),
                        transition.get("before_state"),
                        transition.get("after_state"),
                        transition.get("reason"),
                        transition.get("active_count"),
                    ),
                )
        self._render_watermark()
        self._render_metadata(cluster_id)

    def _render_watermark(self) -> None:
        cluster_id = self.selected_cluster()
        self.canvas.delete("all")
        if cluster_id is None:
            return
        rows = self.ledger.current_watermarks(cluster_id)
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        margin = 48
        plot_width = max(1, width - 2 * margin)
        plot_height = max(1, height - 2 * margin)
        self.canvas.create_line(margin, height - margin, width - margin, height - margin, fill="#666")
        self.canvas.create_line(margin, margin, margin, height - margin, fill="#666")
        self.canvas.create_text(margin, margin - 18, text="watermark_fixed (scale=1,000,000)", anchor=tk.W)
        if not rows:
            self.watermark_caption.configure(text="No current watermark records")
            return
        max_seq = max(1, max(int(row["event_seq"]) for row in rows))
        points: list[float] = []
        try:
            import numpy as np

            seqs = np.asarray([int(row["event_seq"]) for row in rows], dtype=float)
            values = np.asarray([int(row["watermark_fixed"]) for row in rows], dtype=float)
            for seq, value in zip(seqs, values):
                x = margin + plot_width * (seq / max_seq)
                y = height - margin - plot_height * (value / 1_000_000.0)
                points.extend((float(x), float(y)))
        except ImportError:
            for row in rows:
                x = margin + plot_width * (int(row["event_seq"]) / max_seq)
                y = height - margin - plot_height * (int(row["watermark_fixed"]) / 1_000_000.0)
                points.extend((float(x), float(y)))
        if len(points) >= 4:
            self.canvas.create_line(*points, fill="#1f77b4", width=2, smooth=False)
        for index in range(0, len(points), 2):
            x, y = points[index], points[index + 1]
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill="#1f77b4", outline="")
        last = rows[-1]
        self.watermark_caption.configure(
            text=(
                f"cluster={cluster_id} events={len(rows)} last_event={last['event_seq']} "
                f"used={last['used_capacity']}/{last['capacity']} fixed={last['watermark_fixed']} "
                "truncation=floor"
            )
        )

    def _render_metadata(self, cluster_id: str) -> None:
        version = self.ledger.cluster_version(cluster_id) or {}
        snapshots = self.ledger.snapshot_catalog()
        self.metadata_text.delete("1.0", tk.END)
        self.metadata_text.insert(tk.END, "Parameter version\n")
        for field in (
            "parameter_version",
            "host",
            "port",
            "lease_duration_ms",
            "renewal_interval_ms",
            "capacity",
            "priority_digest",
        ):
            self.metadata_text.insert(tk.END, f"  {field}: {version.get(field)}\n")
        self.metadata_text.insert(tk.END, "\nSnapshots (stable catalog order)\n")
        for snapshot in snapshots:
            self.metadata_text.insert(
                tk.END,
                f"  {snapshot['snapshot_id']} {snapshot['created_at_utc']} "
                f"format={snapshot['format_version']} digest={snapshot['content_digest']}\n",
            )


def run_gui(db_path: str) -> None:
    if not os.path.exists(db_path):
        raise LeaseError("E_ENV", f"database does not exist: {db_path}", field="db")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise LeaseError(
            "E_ENV",
            f"Tk display is unavailable: {exc}",
            field="DISPLAY",
        ) from None
    app = DiagnosticsApp(root, db_path)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.ledger.close(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read-only Tk lease ledger diagnostics")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()
    run_gui(args.db)
