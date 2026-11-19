"""Tkinter diagnostic workbench with a headless text fallback.

On the supported Windows desktop Python (tkinter bundled by the installer)
launch_workbench() opens the GUI.  In stripped-down environments without the
tkinter shared library it degrades gracefully to the equivalent text report
so diagnostics remain fully usable and testable.
"""

from __future__ import annotations

from .report import render_diagnostic_report

try:  # pragma: no cover - exercised only on desktop runs
    import tkinter as tk
    from tkinter import ttk

    TK_AVAILABLE = True
except Exception:  # pragma: no cover
    tk = None
    ttk = None
    TK_AVAILABLE = False


def launch_workbench(view, *, title="Pipeline Cache Ledger Workbench"):
    if not TK_AVAILABLE:
        return render_diagnostic_report(view)
    return _TkWorkbench(view, title).run()


if TK_AVAILABLE:  # pragma: no cover - desktop only

    class _TkWorkbench:
        def __init__(self, view, title):
            self.view = view
            self.root = tk.Tk()
            self.root.title(title)
            self.root.geometry("1080x720")
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True)

        def _table(self, tab, columns, rows):
            frame = ttk.Frame(tab)
            frame.pack(fill="both", expand=True)
            tree = ttk.Treeview(
                frame, columns=columns, show="headings", height=24
            )
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=max(90, 120 if col != "key" else 140))
            scroll = ttk.Scrollbar(
                frame, orient="vertical", command=tree.yview
            )
            tree.configure(yscrollcommand=scroll.set)
            scroll.pack(side="right", fill="y")
            tree.pack(side="left", fill="both", expand=True)
            for row in rows:
                tree.insert("", "end", values=row)
            return tree

        def run(self):
            view = self.view

            tab1 = ttk.Frame(self.notebook)
            self.notebook.add(tab1, text="Stages & Artifacts")
            stage_rows = []
            for stage in view["stages"]:
                for key in stage["artifacts"]:
                    stage_rows.append(
                        (stage["order"], stage["stage_id"], key)
                    )
            self._table(tab1, ("order", "stage", "key"), stage_rows)

            tab2 = ttk.Frame(self.notebook)
            self.notebook.add(tab2, text="Entries & Hits")
            hit_rows = []
            for art in view["artifacts"]:
                src = view["source_versions"].get(art["key"], "-")
                hit_rows.append(
                    (
                        art["key"], art["stage_id"], art["version"], src,
                        art["status"], art["hit_outcome"], art["hit_reason"],
                        ",".join(art["backfilled"]) or "-",
                    )
                )
            self._table(
                tab2,
                ("key", "stage", "version", "src_fmt", "status",
                 "hit", "reason", "backfilled"),
                hit_rows,
            )

            tab3 = ttk.Frame(self.notebook)
            self.notebook.add(tab3, text="Invalidation")
            trail_rows = []
            for root in sorted(view["trails"]):
                trail_rows.append((root, " -> ".join(view["trails"][root])))
            self._table(tab3, ("root", "propagation order"), trail_rows)

            tab4 = ttk.Frame(self.notebook)
            self.notebook.add(tab4, text="Concurrency Slots")
            slot_rows = []
            for slot in view["slot_timeline"]:
                for occ in slot["occupancy"]:
                    slot_rows.append(
                        (
                            slot["slot"], occ["key"], occ["stage_id"],
                            occ["start_tick"], occ["end_tick"],
                        )
                    )
            self._table(
                tab4,
                ("slot", "key", "stage", "start_tick", "end_tick"),
                slot_rows,
            )

            tab5 = ttk.Frame(self.notebook)
            self.notebook.add(tab5, text="Snapshots")
            snap_rows = [
                (
                    s["name"], s["snapshot_format"],
                    s["pipeline_version"], s["graph_version"],
                    "readonly" if s["readonly"] else "writable",
                    s["payload_digest"],
                )
                for s in view["snapshots"]
            ]
            self._table(
                tab5,
                ("name", "format", "pipeline_v", "graph_v", "mode", "digest"),
                snap_rows,
            )

            self.root.mainloop()
            return "tk-mainloop-exited"
