"""tkinter quick-view workbench UI (Windows 10+, Python 3.11 stdlib).

The quick view reads the incremental cache in ONE snapshot and never
triggers a full replay. The authoritative recompute button replays the
ledger request-by-request and reports the first divergence, if any.
"""

import json
import tkinter as tk
from tkinter import ttk

from .engine import Engine
from .errors import DiagError
from .ledger import Ledger
from .replay import (authoritative_recompute,
                     compare_cache_with_authoritative)
from .snapshot import (build_snapshot, restore_snapshot, save_snapshot,
                       verify_replay_after_restore)
from .spec import load_spec
from .upstream_mgr import UpstreamManager


class WorkbenchApp(tk.Tk):
    def __init__(self, data_dir, ledger_path, snapshot_path, scale,
                 batch_size, global_limit, per_service_quota):
        super().__init__()
        self.title("Gateway Rate-Limit & Auth Diagnostic Workbench")
        self.geometry("980x640")
        self.data_dir = data_dir
        self.ledger_path = ledger_path
        self.snapshot_path = snapshot_path
        self.scale = scale
        self.batch_size = batch_size
        self.global_limit = global_limit
        self.per_service_quota = per_service_quota
        self.spec = None
        self.ledger = None
        self.engine = None
        self.upstreams = None
        self._build_widgets()

    def _build_widgets(self):
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        for label, fn in (
                ("Load Spec", self.on_load),
                ("Start Upstreams", self.on_start_upstreams),
                ("Run Batches", self.on_run),
                ("Quick View", self.on_quick_view),
                ("Authoritative Recompute", self.on_recompute),
                ("Save Snapshot", self.on_snapshot_save),
                ("Open Snapshot", self.on_snapshot_open),
                ("Stop Upstreams", self.on_stop_upstreams)):
            ttk.Button(bar, text=label, command=fn).pack(
                side=tk.LEFT, padx=3)
        self.output = tk.Text(self, wrap=tk.NONE, font=("Consolas", 9))
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(self, command=self.output.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.output.configure(yscrollcommand=scroll.set)

    def log(self, text):
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def log_json(self, obj):
        self.log(json.dumps(obj, indent=1, sort_keys=True,
                            ensure_ascii=False))

    def on_load(self):
        try:
            self.spec = load_spec(self.data_dir)
            self.ledger = Ledger(self.ledger_path, scale=self.scale,
                                 services_digest=self.spec.services_digest,
                                 keys_digest=self.spec.keys_digest)
            unknown = self.ledger.recover_uncommitted()
            self.engine = Engine(
                self.spec, self.ledger, global_limit=self.global_limit,
                per_service_quota=self.per_service_quota)
            self.log(f"spec loaded: {len(self.spec.services)} services, "
                     f"{len(self.spec.requests)} requests")
            if unknown:
                self.log(f"recovered UNKNOWN seqs (re-run): {unknown}")
        except DiagError as exc:
            self.log(f"SPEC REJECTED: {exc}")

    def on_start_upstreams(self):
        if self.upstreams is None:
            self.upstreams = UpstreamManager(self.spec)
            self.upstreams.start()
            self.log("upstreams started: "
                     + ", ".join(sorted(self.upstreams.procs)))

    def on_stop_upstreams(self):
        if self.upstreams is not None:
            self.upstreams.stop_all()
            self.upstreams = None
            self.log("upstreams stopped")

    def on_run(self):
        self.on_start_upstreams()
        rows = self.engine.run_all(batch_size=self.batch_size)
        counts = {}
        for row in rows:
            code = row["decision_code"]
            counts[code] = counts.get(code, 0) + 1
        self.log(f"executed {len(rows)} requests: "
                 + json.dumps(counts, sort_keys=True))

    def on_quick_view(self):
        view = self.engine.quick_view()
        self.log("=== QUICK VIEW (incremental cache, one read) ===")
        self.log_json(view)

    def on_recompute(self):
        self.on_start_upstreams()
        result = authoritative_recompute(self.spec, self.ledger)
        if not result.consistent:
            self.log(f"RECOMPUTE DIVERGENCE: {result.divergence}")
            return
        dual = compare_cache_with_authoritative(
            self.engine.cache.snapshot(), result)
        if dual is not None:
            self.log(f"DUAL-PATH DIVERGENCE: {dual}")
            return
        self.log(f"authoritative recompute OK: "
                 f"{len(result.conclusions)} requests, "
                 f"cache byte-identical ({result.elapsed_ms} ms)")
        self.log_json(result.stats)

    def on_snapshot_save(self):
        snap = build_snapshot(self.spec, self.ledger)
        save_snapshot(self.snapshot_path, snap)
        self.log(f"snapshot saved: {self.snapshot_path} "
                 f"(cursor={snap['ledger_cursor']})")

    def on_snapshot_open(self):
        snap, diffs, replaced = restore_snapshot(
            self.snapshot_path, self.spec, self.ledger.scale)
        if replaced:
            self.log("snapshot REPLACED (whole replace); diffs: "
                     + ", ".join(diffs))
        else:
            self.log(f"snapshot accepted (cursor={snap['ledger_cursor']})")
        try:
            n = verify_replay_after_restore(self.spec, self.ledger)
            self.log(f"replay verification OK: {n} requests")
        except DiagError as exc:
            self.log(f"replay verification FAILED: {exc}")

    def destroy(self):
        self.on_stop_upstreams()
        if self.ledger is not None:
            self.ledger.close()
        super().destroy()


def launch(data_dir, ledger_path, snapshot_path, scale, batch_size,
           global_limit, per_service_quota):
    app = WorkbenchApp(data_dir, ledger_path, snapshot_path, scale,
                       batch_size, global_limit, per_service_quota)
    app.mainloop()
