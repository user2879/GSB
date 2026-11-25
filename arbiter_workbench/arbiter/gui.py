"""Tkinter diagnostic workbench with a headless text fallback.

The GUI is a read-only diagnostic surface: it runs the full arbitration flow,
shows per-view authority values, error attribution, path-selection rationale
and recovery status. When no X display is available the same content renders as
plain text so the module remains importable and usable over SSH/WSL without a
display server.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .engine import ArbitrationEngine
from .ledger import Ledger
from .sample import DEFAULT_KEYS
from .service_manager import ServiceManager
from .spec import load_declarations


def run_headless(decl_dir: str | Path, scale: int = 2) -> str:
    decls = load_declarations(decl_dir, DEFAULT_KEYS)
    run_dir = Path(decl_dir).parent / "run"
    run_dir.mkdir(exist_ok=True)
    ledger = Ledger(run_dir / "ledger.sqlite", session_scale=scale)
    manager = ServiceManager(list(decls.environments.values()))
    engine = ArbitrationEngine(decls, scale, ledger)
    try:
        manager.start_all()
        engine.start_services_and_ingest(manager)
        outputs = engine.run_full()
        lines = ["# Arbitration diagnostic (headless)", ""]
        for view_id in sorted(outputs):
            item = outputs[view_id]
            decision = item["decision"]
            lines.append(f"## {view_id} version={item['invalidation'].new_version}")
            lines.append(
                f"path={item['path']} selected={decision.selected} "
                f"full_work={decision.full_work} incr_work={decision.incremental_work} "
                f"boundary={decision.threshold_boundary}"
            )
            lines.append(f"reason: {decision.reason}")
            for record in item["records"]:
                lines.append(
                    f"  - {record.key} [{record.policy}] authority_q={record.authority_q} "
                    f"adopted={list(record.adopted_envs)} "
                    f"err(q/b/n)={record.error_quant}/{record.error_boundary}/{record.error_norm}"
                )
            lines.append("")
        lines.append(f"meta_digest={ledger.meta_digest()}")
        return "\n".join(lines)
    finally:
        manager.shutdown_all()
        ledger.close()


def launch_gui(decl_dir: str | Path, scale: int = 2) -> int:
    import tkinter as tk
    from tkinter import ttk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(run_headless(decl_dir, scale))
        print(f"\n[headless fallback] Tk display unavailable: {exc}")
        return 0

    root.title("Cross-Environment Config Arbitration Workbench")
    root.geometry("980x640")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=8)

    summary_tab = ttk.Frame(notebook)
    detail_tab = ttk.Frame(notebook)
    notebook.add(summary_tab, text="Views")
    notebook.add(detail_tab, text="Key Detail")

    text = tk.Text(summary_tab, wrap="word")
    text.pack(fill="both", expand=True)
    detail = tk.Text(detail_tab, wrap="word")
    detail.pack(fill="both", expand=True)

    report = run_headless(decl_dir, scale)
    text.insert("1.0", report)
    detail.insert("1.0", json.dumps(
        _collect_detail(decl_dir, scale), indent=2, sort_keys=True, default=str
    ))

    root.mainloop()
    return 0


def _collect_detail(decl_dir: str | Path, scale: int) -> dict:
    decls = load_declarations(decl_dir, DEFAULT_KEYS)
    run_dir = Path(decl_dir).parent / "run"
    ledger = Ledger(run_dir / "ledger.sqlite", session_scale=scale)
    try:
        latest = ledger.latest_versions()
        payload: dict = {"views": {}}
        for view_id, row in sorted(latest.items()):
            records = ledger.load_records(view_id, row["version"])
            payload["views"][view_id] = {
                "commit": row,
                "records": [asdict(r) for r in records],
            }
        payload["meta_digest"] = ledger.meta_digest()
        return payload
    finally:
        ledger.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arbitration diagnostic GUI")
    parser.add_argument("--decls", default=str(Path(__file__).resolve().parents[1] / "decls"))
    parser.add_argument("--scale", type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.headless:
        print(run_headless(args.decls, args.scale))
    else:
        raise SystemExit(launch_gui(args.decls, args.scale))
