"""Tkinter diagnostic console for snapshot migrations.

The UI is read-mostly and never mutates ledgers while refreshing: every action
that changes state (migrate, cleanup, diagnostic repack) is triggered by an
explicit button.  When tkinter or a display is unavailable, ``launch_ui`` raises
a descriptive ``RuntimeError`` instead of crashing at import time, keeping the
rest of the package usable on headless CI.
"""

import json

from .chain import load_chain
from .engine import MigrationEngine

try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    TK_AVAILABLE = True
    TK_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    tk = None
    ttk = None
    messagebox = None
    TK_AVAILABLE = False
    TK_IMPORT_ERROR = exc


class DiagnosticUI:
    def __init__(self, chain_path, db_path):
        if not TK_AVAILABLE:
            raise RuntimeError(
                f"tkinter is unavailable in this environment: {TK_IMPORT_ERROR}")
        self.chain_path = chain_path
        self.db_path = db_path
        self.chain = load_chain(chain_path)
        self.engine = MigrationEngine(db_path, self.chain)
        self.root = tk.Tk()
        self.root.title("Snapshot Migration Diagnostics")
        self.root.geometry("980x720")
        self._build_widgets()
        self.refresh_all()

    # ------------------------------------------------------------ layout
    def _build_widgets(self):
        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Migrate all",
                   command=self.migrate_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Run cleanup",
                   command=self.run_cleanup).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Save snapshot",
                   command=self.save_snapshot).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Refresh",
                   command=self.refresh_all).pack(side=tk.LEFT, padx=3)

        ttk.Label(toolbar, text="Generation:").pack(side=tk.LEFT, padx=(16, 2))
        self.generation_var = tk.StringVar(
            value=self.chain.generations[-1].generation_id)
        ttk.Combobox(toolbar, width=6, textvariable=self.generation_var,
                     values=[g.generation_id for g in self.chain.generations],
                     state="readonly").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Open scan view",
                   command=self.open_scan).pack(side=tk.LEFT, padx=3)

        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.text_views = {}
        for name in ("Chain & rules", "Fragments", "Split/merge trace",
                     "Inheritance map", "Cleanup ledger", "Scan & cursors"):
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=name)
            text = tk.Text(frame, wrap=tk.NONE, font=("Consolas", 9))
            scroll_y = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                     command=text.yview)
            scroll_x = ttk.Scrollbar(frame, orient=tk.HORIZONTAL,
                                     command=text.xview)
            text.configure(yscrollcommand=scroll_y.set,
                           xscrollcommand=scroll_x.set)
            text.grid(row=0, column=0, sticky="nsew")
            scroll_y.grid(row=0, column=1, sticky="ns")
            scroll_x.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            self.text_views[name] = text

        self.status = ttk.Label(self.root, text="", anchor="w", padding=4)
        self.status.pack(fill=tk.X)

    def _set_text(self, name, content):
        text = self.text_views[name]
        text.configure(state=tk.NORMAL)
        text.delete("1.0", tk.END)
        text.insert(tk.END, content)
        text.configure(state=tk.DISABLED)

    def _set_status(self, message):
        self.status.configure(text=message)

    # ------------------------------------------------------------ actions
    def migrate_all(self):
        try:
            for gen in self.chain.generations:
                self.engine.migrate_generation(gen.generation_id)
            self.refresh_all()
            self._set_status("migration complete")
        except Exception as exc:  # pragma: no cover - interactive guard
            messagebox.showerror(type(exc).__name__, str(exc))

    def run_cleanup(self):
        result = self.engine.run_cleanup()
        self.refresh_all()
        self._set_status(f"cleanup actions: {len(result)}")

    def save_snapshot(self):
        manifest = self.engine.save_snapshot(self.db_path + ".snapshot.json")
        self._set_status(
            f"snapshot saved digest={manifest['content_digest'][:12]}")

    def open_scan(self):
        gen_id = self.generation_var.get()
        with self.engine.open_read_view("ui-view", gen_id) as view:
            page = view.scan(limit=20)
        lines = [f"scan generation={gen_id} limit=20", ""]
        for row in page["rows"]:
            lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
        lines.append("")
        lines.append("next_cursor=" + json.dumps(page["next_cursor"],
                                                  sort_keys=True))
        self._set_text("Scan & cursors", "\n".join(lines))
        self._set_status("scan rendered")

    # ------------------------------------------------------------ rendering
    def refresh_all(self):
        self._render_chain()
        self._render_fragments()
        self._render_events()
        self._render_inheritance()
        self._render_cleanup()

    def _render_chain(self):
        lines = [f"title: {self.chain.title}",
                 f"limits: lower={self.chain.limits.lower} "
                 f"upper={self.chain.limits.upper}", ""]
        for gen in self.chain.generations:
            lines.append(f"[{gen.generation_id}] seq={gen.seq} "
                         f"parent={gen.parent_id}")
            for item in gen.ranges:
                lines.append(f"  {item.range_id:<3} action={item.action:<8} "
                             f"tombstone={item.tombstone} keys={len(item.keys)}")
        self._set_text("Chain & rules", "\n".join(lines))

    def _render_fragments(self):
        lines = []
        for gen in self.chain.generations:
            for frag in self.engine.repo.active_fragments(
                    gen.generation_id):
                lines.append(
                    f"{frag.generation_id} {frag.fragment_id} "
                    f"[{frag.lower},{frag.upper}) keys={frag.key_count} "
                    f"v{frag.version} inherit={frag.inherited_from or '-'} "
                    f"digest={frag.digest[:10]}")
        self._set_text("Fragments", "\n".join(lines) or "(no fragments yet)")

    def _render_events(self):
        lines = []
        for event in self.engine.repo.fragment_events():
            lines.append(
                f"{event['generation_id']} {event['event']:<20} "
                f"{event['fragment_id']} "
                f"{json.dumps(event['detail'], sort_keys=True)}")
        self._set_text("Split/merge trace", "\n".join(lines)
                       or "(no events yet)")

    def _render_inheritance(self):
        manifest = self.engine.build_manifest()
        lines = []
        for frag_id, info in sorted(manifest["inheritance"].items()):
            lines.append(f"{frag_id} <- {info['from_generation']} "
                         f"source={info['source_fragment_id']} "
                         f"digest={info['digest'][:12]}")
        self._set_text("Inheritance map", "\n".join(lines)
                       or "(no inherited fragments)")

    def _render_cleanup(self):
        lines = []
        for row in self.engine.repo.cleanup_ledger():
            lines.append(
                f"{row['generation_id']} {row['range_id']} {row['fragment_id']} "
                f"reason={row['reason']} evidence="
                f"{json.dumps(row['evidence'], sort_keys=True)}")
        self._set_text("Cleanup ledger", "\n".join(lines)
                       or "(no cleanup records)")

    def run(self):
        self.root.mainloop()


def launch_ui(chain_path, db_path):
    if not TK_AVAILABLE:
        raise RuntimeError(
            "Cannot start the diagnostic UI on this host. The Tkinter shared "
            f"library or a graphical display is missing: {TK_IMPORT_ERROR}")
    DiagnosticUI(chain_path, db_path).run()
