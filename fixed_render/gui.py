from __future__ import annotations

import ctypes
import json
import os
import traceback
from pathlib import Path
from typing import Any

from .hashing import canonical_json
from .storage import atomic_write_json


def prepare_local_tk() -> None:
    """Load an unprivileged dpkg-deb Tk runtime when python3-tk is absent."""
    candidates = [
        os.environ.get("FIXED_RENDER_TK_ROOT"),
        str(Path(__file__).resolve().parent.parent / ".tk-runtime"),
        str(Path.cwd() / ".tk-runtime"),
        str(Path.home() / ".local" / "fixed-render-tk"),
        "/home/zsl/.local/fixed-render-tk",
    ]
    root = next((Path(item) for item in candidates if item and Path(item).exists()), None)
    if root is None:
        return
    lib_dir = root / "usr" / "lib" / "x86_64-linux-gnu"
    tcl_library = root / "usr" / "share" / "tcltk" / "tcl8.6"
    tk_library = root / "usr" / "share" / "tcltk" / "tk8.6"
    if tcl_library.exists():
        os.environ.setdefault("TCL_LIBRARY", str(tcl_library))
    if tk_library.exists():
        os.environ.setdefault("TK_LIBRARY", str(tk_library))
    if not lib_dir.exists():
        return
    for library in (
        "libXrender.so.1",
        "libXss.so.1",
        "libXft.so.2",
        "libtcl8.6.so",
        "libtk8.6.so",
    ):
        path = lib_dir / library
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                # The normal import error/reporting path handles missing transitive libs.
                pass


def run_gui(report_path: Path, smoke_ms: int = 0) -> int:
    report_path = Path(report_path)
    smoke_path = Path("artifacts/gui_smoke.json")
    prepare_local_tk()
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk
    except Exception as exc:
        atomic_write_json(
            smoke_path,
            {
                "ok": False,
                "code": "TK_IMPORT_FAILED",
                "error": str(exc),
                "hint": "Install Ubuntu package python3-tk and libtk8.6, or extract them locally.",
            },
        )
        print(f"Tkinter unavailable: {exc}")
        return 4

    try:
        root = tk.Tk()
    except Exception as exc:
        atomic_write_json(
            smoke_path,
            {"ok": False, "code": "TK_DISPLAY_FAILED", "error": str(exc), "display": __import__("os").environ.get("DISPLAY")},
        )
        print(f"Tk display unavailable: {exc}")
        return 5

    app = LedgerViewer(root, report_path)
    if smoke_ms > 0:
        def smoke() -> None:
            try:
                root.update_idletasks()
                app.select_first_case()
                root.update_idletasks()
                result = {
                    "ok": True,
                    "tk_version": tk.TkVersion,
                    "report": str(report_path),
                    "cases": len(app.report.get("cases", [])),
                    "text_characters": int(app.text.index("end-1c").split(".")[0]),
                    "selected": app.case_list.get("active"),
                    "title": root.title(),
                }
                atomic_write_json(smoke_path, result)
            except Exception as exc:
                atomic_write_json(
                    smoke_path,
                    {"ok": False, "error": str(exc), "traceback": traceback.format_exc()},
                )
            finally:
                root.destroy()

        root.after(max(50, smoke_ms), smoke)
    root.mainloop()
    return 0


class LedgerViewer:
    def __init__(self, root: Any, report_path: Path) -> None:
        from tkinter import ttk

        self.root = root
        self.report_path = report_path
        self.report: dict[str, Any] = {}
        root.title("Fixed Render — Lease & Content Ledger")
        root.geometry("1280x820")

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        self.title_label = ttk.Label(top, text="Fixed-point multimodal render workbench", font=("sans-serif", 14, "bold"))
        self.title_label.pack(anchor="w")
        self.summary_label = ttk.Label(top, text="Loading report…")
        self.summary_label.pack(anchor="w", pady=(4, 0))

        buttons = ttk.Frame(root, padding=(8, 0))
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Reload report", command=self.reload_report).pack(side="left")
        ttk.Button(buttons, text="Show report JSON", command=self.show_report).pack(side="left", padx=6)
        ttk.Button(buttons, text="Show selected ledger", command=self.show_selected_ledger).pack(side="left")
        ttk.Button(buttons, text="Show lease audit", command=self.show_lease_audit).pack(side="left", padx=6)

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        body.add(left, weight=1)
        ttk.Label(left, text="Injection cases / ledgers").pack(anchor="w")
        self.case_list = tk_listbox(left)
        self.case_list.pack(fill="both", expand=True)
        self.case_list.bind("<<ListboxSelect>>", lambda _event: self.show_selected_ledger())

        right = ttk.Frame(body)
        body.add(right, weight=4)
        self.text = tk_text(right)
        self.text.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(right, command=self.text.yview)
        scrollbar.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scrollbar.set)

        self.status = ttk.Label(root, text="", relief="sunken", anchor="w", padding=4)
        self.status.pack(fill="x")
        self.reload_report()

    def reload_report(self) -> None:
        if self.report_path.exists():
            self.report = json.loads(self.report_path.read_text(encoding="utf-8"))
        else:
            self.report = {"summary": {"failed_cases": "report missing"}, "cases": []}
        summary = self.report.get("summary", {})
        self.summary_label.config(
            text=(
                f"Report: {self.report_path} | injections "
                f"{summary.get('hit_injections', 0)}/{summary.get('total_injections', 0)} hit | "
                f"codes {summary.get('aggregate_business_code_counts', {})}"
            )
        )
        self.case_list.delete(0, "end")
        for case in self.report.get("cases", []):
            self.case_list.insert(
                "end",
                f"{case['injection_id']} {case['fault']} hit={case['hit']} -> {case['expected_code']}",
            )
        self.status.config(text=f"Loaded {self.report_path}")
        self.show_report()

    def select_first_case(self) -> None:
        if self.case_list.size():
            self.case_list.selection_clear(0, "end")
            self.case_list.selection_set(0)
            self.case_list.activate(0)
            self.show_selected_ledger()

    def selected_case(self) -> dict[str, Any] | None:
        selection = self.case_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        cases = self.report.get("cases", [])
        return cases[index] if index < len(cases) else None

    def show_report(self) -> None:
        self._set_text(
            canonical_json(
                {
                    "summary": self.report.get("summary", {}),
                    "lease_reclamation": self.report.get("lease_reclamation", {}),
                    "cases": [
                        {k: v for k, v in case.items() if k not in {"stage_timings_ms"}}
                        for case in self.report.get("cases", [])
                    ],
                }
            )
        )

    def show_selected_ledger(self) -> None:
        case = self.selected_case()
        if not case:
            self.show_report()
            return
        ledger_path = Path(case["ledger_path"])
        if not ledger_path.is_absolute():
            ledger_path = Path.cwd() / ledger_path
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            payload = {
                "selected_case": case["injection_id"],
                "same_ledger_path": str(ledger_path),
                "counts": ledger.get("counts"),
                "business_code_counts": ledger.get("business_code_counts"),
                "stage_order": ledger.get("stage_order"),
                "recompute_order": ledger.get("recompute_order"),
                "reuse_order": ledger.get("reuse_order"),
                "entries": ledger.get("entries"),
                "lease_trajectory": ledger.get("lease_trajectory"),
            }
            self._set_text(canonical_json(payload))
            self.status.config(text=f"Showing same ledger used by diagnostics: {ledger_path}")
        else:
            self._set_text(f"Ledger not found: {ledger_path}")

    def show_lease_audit(self) -> None:
        self._set_text(canonical_json(self.report.get("lease_reclamation", {})))

    def _set_text(self, value: str) -> None:
        self.text.delete("1.0", "end")
        try:
            parsed = json.loads(value)
            value = json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
        self.text.insert("1.0", value)


def tk_listbox(parent: Any) -> Any:
    import tkinter as tk

    return tk.Listbox(parent, exportselection=False, font=("monospace", 10))


def tk_text(parent: Any) -> Any:
    import tkinter as tk

    return tk.Text(parent, wrap="none", font=("monospace", 9))
