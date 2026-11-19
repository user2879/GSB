"""Tkinter operator console for the block snapshot component.

The view renders, in stable order: the snapshot chain, a per-snapshot bitmap
overview, the adaptive rollback path plan, the staging/tmp area state and the
structured business-code log.  Tkinter is imported lazily so the rest of the
package keeps working on headless machines without a Tk installation.
"""
from __future__ import annotations

import os

from .bitmap import Bitmap
from .errors import DomainError
from .service import BlockSnapService


class TkUnavailable(RuntimeError):
    pass


def require_tk():  # pragma: no cover - exercised only where tkinter exists
    try:
        import tkinter as tk  # noqa: F401
        from tkinter import ttk  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise TkUnavailable(
            "tkinter is not available in this Python installation; "
            "use scripts/blocksnap_cli.py instead"
        ) from exc
    import tkinter as tk
    from tkinter import ttk
    return tk, ttk


class BlockSnapConsole:  # pragma: no cover - GUI needs a display
    def __init__(self, service: BlockSnapService) -> None:
        self.service = service
        tk, ttk = require_tk()
        self.tk, self.ttk = tk, ttk
        self.root = tk.Tk()
        self.root.title("Block Snapshot & Rollback Console")
        self.root.geometry("980x720")
        self._build_widgets()
        self.refresh_all()

    # -- layout ---------------------------------------------------------
    def _build_widgets(self) -> None:
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Volume:").pack(side=tk.LEFT)
        self.volume_var = tk.StringVar()
        self.volume_entry = ttk.Entry(top, textvariable=self.volume_var, width=24)
        self.volume_entry.pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Refresh", command=self.refresh_all).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Audit", command=self.run_audit).pack(side=tk.LEFT, padx=4)

        body = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        chain_frame = ttk.Labelframe(body, text="Snapshot chain (stable version order)")
        columns = ("version", "name", "snapshot_id", "parent", "changed", "ratio", "refs")
        self.chain_tree = ttk.Treeview(chain_frame, columns=columns, show="headings", height=8)
        for col, width in zip(columns, (70, 120, 170, 170, 90, 80, 70)):
            self.chain_tree.heading(col, text=col)
            self.chain_tree.column(col, width=width, anchor=tk.W)
        self.chain_tree.pack(fill=tk.BOTH, expand=True)
        body.add(chain_frame, weight=3)

        bitmap_frame = ttk.Labelframe(body, text="Changed-block bitmap overview (per selected snapshot)")
        self.bitmap_canvas = tk.Canvas(bitmap_frame, height=90, bg="white")
        self.bitmap_canvas.pack(fill=tk.BOTH, expand=True)
        body.add(bitmap_frame, weight=1)

        plan_frame = ttk.Labelframe(body, text="Rollback path plan & staging area")
        self.plan_text = tk.Text(plan_frame, height=10, wrap=tk.WORD)
        self.plan_text.pack(fill=tk.BOTH, expand=True)
        body.add(plan_frame, weight=2)

        controls = ttk.Frame(self.root, padding=6)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Plan rollback", command=self.plan_rollback).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Rollback (cross-check)", command=self.rollback_selected).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="Recover", command=self.recover).pack(side=tk.LEFT, padx=3)

        log_frame = ttk.Labelframe(self.root, text="Structured business-code log")
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.NONE)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    # -- data binding ---------------------------------------------------
    def _volume(self) -> str:
        return self.volume_var.get().strip()

    def _selected_snapshot(self) -> Optional[str]:
        selection = self.chain_tree.selection()
        if not selection:
            return None
        return str(self.chain_tree.item(selection[0], "values")[2])

    def refresh_all(self) -> None:
        volume = self._volume()
        if not volume:
            return
        try:
            snapshots = self.service.list_snapshots(volume)
        except DomainError:
            snapshots = []
        self.chain_tree.delete(*self.chain_tree.get_children())
        for snap in snapshots:
            self.chain_tree.insert(
                "", self.tk.END,
                values=(snap["version_no"], snap["name"], snap["snapshot_id"], snap["parent_id"] or "-",
                        snap["changed_blocks"], snap["bitmap_ratio"], snap["ref_count"]),
            )
        self.chain_tree.bind("<<TreeviewSelect>>", lambda _event: self.render_bitmap(volume))
        self.render_bitmap(volume)
        self.render_staging(volume)
        self.render_logs()

    def render_bitmap(self, volume: str) -> None:
        self.bitmap_canvas.delete("all")
        snapshot_id = self._selected_snapshot()
        if snapshot_id is None:
            return
        try:
            row = self.service.manager.store.get_snapshot(volume, snapshot_id)
            volume_row = self.service.manager.store.get_volume(volume)
            bitmap = Bitmap(int(volume_row["block_count"]), row["bitmap"])
        except DomainError:
            return
        cell = 14
        for block_no in range(bitmap.block_count):
            x = (block_no % 32) * cell
            y = (block_no // 32) * cell
            color = "#c0392b" if bitmap.get(block_no) else "#ecf0f1"
            self.bitmap_canvas.create_rectangle(x, y, x + cell - 2, y + cell - 2, fill=color, outline="#bdc3c7")

    def render_staging(self, volume: str) -> None:
        tmp_dir = self.service.manager.tmp_dir
        files = sorted(os.listdir(tmp_dir)) if os.path.exists(tmp_dir) else []
        jobs = self.service.manager.store.list_jobs(volume)
        lines = ["Staging directory: " + tmp_dir, "Files: " + (", ".join(files) if files else "(empty)")]
        for job in jobs:
            lines.append(
                f"job={job['job_id']} state={job['state']} path={job['path_used']} "
                f"verified={job['verified_count']}/{job['total_count']} target={job['target_snapshot']}"
            )
        self.plan_text.delete("1.0", self.tk.END)
        self.plan_text.insert(self.tk.END, "\n".join(lines))

    def render_logs(self) -> None:
        entries = self.service.query_logs()
        self.log_text.delete("1.0", self.tk.END)
        for entry in sorted(entries, key=lambda row: (row["ts"], row["event"])):
            self.log_text.insert(self.tk.END, f"{entry['ts']} {entry['level']} {entry['event']} "
                                              f"code={entry['code']} snapshot={entry['snapshot_id']} "
                                              f"block={entry['block_no']} manifest={entry['manifest_version']}\n")

    def plan_rollback(self) -> None:
        volume, snapshot_id = self._volume(), self._selected_snapshot()
        if not volume or not snapshot_id:
            return
        plan = self.service.plan_rollback(volume, snapshot_id)
        self.plan_text.insert(self.tk.END, "\nPATH PLAN: " + str(plan))

    def rollback_selected(self) -> None:
        volume, snapshot_id = self._volume(), self._selected_snapshot()
        if not volume or not snapshot_id:
            return
        try:
            result = self.service.rollback(volume, snapshot_id, verify_all_paths=True)
            self.plan_text.insert(self.tk.END, "\nROLLBACK: " + str(result))
        except DomainError as err:
            self.plan_text.insert(self.tk.END, "\nROLLBACK FAILED: " + str(err.to_dict()))
        self.refresh_all()

    def recover(self) -> None:
        volume = self._volume()
        if volume:
            self.plan_text.insert(self.tk.END, "\nRECOVER: " + str(self.service.recover(volume)))
            self.refresh_all()

    def run_audit(self) -> None:
        volume = self._volume()
        if volume:
            self.plan_text.insert(self.tk.END, "\nAUDIT: " + str(self.service.audit(volume)))
            self.refresh_all()

    def run(self) -> None:  # pragma: no cover
        self.root.mainloop()


def launch(workspace: str) -> None:  # pragma: no cover - GUI entry point
    service = BlockSnapService(workspace)
    BlockSnapConsole(service).run()
