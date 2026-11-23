"""Tkinter workbench GUI.

The GUI edits the environment / observation JSON and the view JSON, triggers
arbitration, and renders the per-key multi-environment distribution, authority,
rejected observations with reasons, error bounds and attribution. All list
ordering is sorted and stable; refreshing only re-reads session state and never
writes to the ledger.

If the host Python lacks Tk support (e.g. a headless build without libtk),
:class:`HeadlessUnavailable` is raised with guidance instead of crashing.
"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import WorkbenchError
from .ledger import Ledger
from .render import render_session_text
from .session import Session
from .spec import load_json, load_specification

try:  # Tk is optional in headless validation containers.
    import tkinter as tk
    from tkinter import messagebox, ttk
    TK_AVAILABLE = True
    TK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment dependent
    tk = None
    ttk = None
    messagebox = None
    TK_AVAILABLE = False
    TK_IMPORT_ERROR = exc


class HeadlessUnavailable(RuntimeError):
    pass


def launch_gui(env_path: str | Path, view_path: str | Path,
               ledger_path: str | Path) -> None:
    if not TK_AVAILABLE:
        raise HeadlessUnavailable(
            "Tkinter is not available in this Python build "
            f"({TK_IMPORT_ERROR!r}). On Windows use the standard python.org "
            "installer (Tk bundled); on Debian/Ubuntu install python3-tk. "
            "Headless rendering is available via render.render_session_text."
        )
    root = tk.Tk()
    root.title("Cross-Environment Configuration Arbitration Workbench")
    root.geometry("1180x760")
    WorkbenchApp(root, Path(env_path), Path(view_path), Path(ledger_path))
    root.mainloop()


class WorkbenchApp:
    def __init__(self, root, env_path: Path, view_path: Path, ledger_path: Path):
        self.root = root
        self.env_path = env_path
        self.view_path = view_path
        self.ledger_path = ledger_path
        self.session: Session | None = None

        self._build_editor_pane()
        self._build_control_pane()
        self._build_result_pane()
        self.refresh_editors()

    # ------------------------------------------------------------------
    def _build_editor_pane(self) -> None:
        frame = ttk.Frame(self.root, padding=6)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Environments / observations JSON").pack(anchor=tk.W)
        self.env_text = tk.Text(frame, wrap=tk.NONE, height=24, width=64)
        self.env_text.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Arbitration views JSON").pack(anchor=tk.W, pady=(8, 0))
        self.view_text = tk.Text(frame, wrap=tk.NONE, height=16, width=64)
        self.view_text.pack(fill=tk.BOTH, expand=True)

    def _build_control_pane(self) -> None:
        frame = ttk.Frame(self.root, padding=6)
        frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(frame, text="Save & validate inputs",
                   command=self.save_inputs).pack(fill=tk.X, pady=4)
        ttk.Button(frame, text="Run arbitration",
                   command=self.run_arbitration).pack(fill=tk.X, pady=4)
        ttk.Button(frame, text="Refresh view (no ledger write)",
                   command=self.refresh_results).pack(fill=tk.X, pady=4)
        ttk.Button(frame, text="Recovery / unknown views",
                   command=self.recover_unknown).pack(fill=tk.X, pady=4)
        ttk.Button(frame, text="Save snapshot",
                   command=self.save_snapshot).pack(fill=tk.X, pady=4)
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(frame, text="Views").pack(anchor=tk.W)
        self.view_selector = ttk.Combobox(frame, state="readonly", width=28)
        self.view_selector.pack(fill=tk.X, pady=4)
        self.view_selector.bind("<<ComboboxSelected>>", lambda _e: self.refresh_results())

    def _build_result_pane(self) -> None:
        frame = ttk.Frame(self.root, padding=6)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        columns = ("key", "policy", "authority", "bound", "adj", "flag", "attribution")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        headers = {
            "key": "Key", "policy": "Policy", "authority": "Authority(q/real)",
            "bound": "Bound(q)", "adj": "Adj diff(q)", "flag": "Error flag",
            "attribution": "Attribution",
        }
        for col in columns:
            self.tree.heading(col, text=headers[col])
            self.tree.column(col, width=120 if col != "key" else 150)
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.show_key_detail())
        ttk.Label(frame, text="Key detail / multi-environment distribution").pack(
            anchor=tk.W, pady=(8, 0))
        self.detail_text = tk.Text(frame, wrap=tk.WORD, height=24)
        self.detail_text.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    def refresh_editors(self) -> None:
        for path, widget in ((self.env_path, self.env_text),
                             (self.view_path, self.view_text)):
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, Path(path).read_text(encoding="utf-8"))

    def save_inputs(self) -> None:
        try:
            env_doc = json.loads(self.env_text.get("1.0", tk.END))
            view_doc = json.loads(self.view_text.get("1.0", tk.END))
            self.env_path.write_text(
                json.dumps(env_doc, indent=2, sort_keys=True), encoding="utf-8")
            self.view_path.write_text(
                json.dumps(view_doc, indent=2, sort_keys=True), encoding="utf-8")
            messagebox.showinfo("OK", "Inputs saved and well-formed JSON.")
        except json.JSONDecodeError as exc:
            messagebox.showerror("E_SPEC", f"Malformed JSON: {exc}")

    def _open_session(self) -> Session:
        spec = load_specification(self.env_path, self.view_path)
        ledger = Ledger(self.ledger_path)
        return Session(spec, ledger)

    def run_arbitration(self) -> None:
        try:
            self.session = self._open_session()
            self.session.run_all()
            self._populate_views()
            self.refresh_results()
        except WorkbenchError as exc:
            messagebox.showerror(exc.code, f"{exc.message}\nlocation: {exc.location}")

    def recover_unknown(self) -> None:
        try:
            if self.session is None:
                self.session = self._open_session()
            rerun = self.session.rerun_unknown()
            self._populate_views()
            self.refresh_results()
            messagebox.showinfo("Recovery", f"Re-ran unknown views: {sorted(rerun)}")
        except WorkbenchError as exc:
            messagebox.showerror(exc.code, f"{exc.message}\nlocation: {exc.location}")

    def save_snapshot(self) -> None:
        if self.session is None:
            messagebox.showwarning("No session", "Run arbitration first.")
            return
        path = self.ledger_path.with_name("session_snapshot.json")
        self.session.save_snapshot(path)
        messagebox.showinfo("Snapshot", f"Saved to {path}")

    def _populate_views(self) -> None:
        view_ids = sorted(self.session.conclusions)
        self.view_selector["values"] = view_ids
        if view_ids and self.view_selector.get() not in view_ids:
            self.view_selector.set(view_ids[0])

    def refresh_results(self) -> None:
        # Pure read path: never writes to the ledger.
        if self.session is None:
            return
        view_id = self.view_selector.get()
        if view_id not in self.session.conclusions:
            view_id = sorted(self.session.conclusions)[0]
            self.view_selector.set(view_id)
        vc = self.session.conclusions[view_id]
        for item in self.tree.get_children():
            self.tree.delete(item)
        for key in sorted(vc.keys):
            kc = vc.keys[key]
            self.tree.insert(
                "", tk.END, iid=f"{view_id}|{key}",
                values=(
                    key, kc.policy,
                    f"{kc.authority_q} / {kc.authority_real}",
                    kc.bound_q, kc.adjacent_diff_q,
                    "error" if kc.error_influenced else "ok",
                    ",".join(kc.attribution) or "-",
                ),
            )
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, render_session_text(self.session))

    def show_key_detail(self) -> None:
        selection = self.tree.selection()
        if not selection or self.session is None:
            return
        view_id, key = selection[0].split("|", 1)
        kc = self.session.conclusions[view_id].keys[key]
        lines = [
            f"view={view_id} key={key} policy={kc.policy}",
            f"authority_q={kc.authority_q} authority_real={kc.authority_real}",
            f"bound_q={kc.bound_q} adjacent_diff_q={kc.adjacent_diff_q}",
            f"error_influenced={kc.error_influenced}",
            f"attribution={kc.attribution}",
            f"attribution_contributions={json.dumps(kc.attribution_contributions, sort_keys=True)}",
            f"accepted={kc.accepted_obs_ids}",
            "rejected:",
        ]
        for verdict in kc.rejected:
            lines.append(
                f"  - {verdict.obs_id} reason={verdict.reject_reason} q={verdict.quantized_value}"
            )
        lines.append("distribution:")
        for env_id, rows in sorted(kc.distribution_q.items()):
            for row in rows:
                lines.append(f"  {env_id} {row['obs_id']} seq={row['seq']} "
                             f"ts={row['ts']} q={row['q']}")
        if kc.conflict_detail:
            lines.append(f"conflict: {kc.conflict_detail}")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, "\n".join(lines))
