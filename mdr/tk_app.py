from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tk_support import import_tkinter


def launch(workspace: Path) -> None:
    tk = import_tkinter()
    from tkinter import ttk

    report_path = workspace / "reports" / "acceptance-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}

    window = tk.Tk()
    window.title("MDR 多视图重建诊断")
    window.geometry("980x680")

    header = ttk.Label(window, text=f"验收通过：{report.get('passed', '未运行')}", font=("Sans", 14, "bold"))
    header.pack(anchor="w", padx=12, pady=8)

    notebook = ttk.Notebook(window)
    notebook.pack(fill="both", expand=True, padx=12, pady=8)

    summary = tk.Text(notebook, wrap="word")
    summary.insert("1.0", json.dumps({
        "main_run": report.get("main_run"),
        "edit_parity": {k: v for k, v in report.get("edit_parity", {}).items() if k != "step_summaries"},
        "precise_invalidation": report.get("precise_invalidation"),
        "concurrency_and_budgets": report.get("concurrency_and_budgets"),
        "recovery": report.get("recovery"),
        "diagnostics": report.get("diagnostics"),
    }, indent=2, sort_keys=True))
    summary.configure(state="disabled")
    notebook.add(summary, text="验收与归因")

    lineage = tk.Text(notebook, wrap="none")
    ledger_report = workspace / "reports" / "diagnostic-report.json"
    if ledger_report.exists():
        lineage.insert("1.0", ledger_report.read_text(encoding="utf-8")[:200_000])
    lineage.configure(state="disabled")
    notebook.add(lineage, text="诊断 JSON")

    ttk.Button(window, text="关闭", command=window.destroy).pack(pady=8)
    window.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--smoke", action="store_true", help="create and destroy a window immediately")
    args = parser.parse_args()
    if args.smoke:
        tk = import_tkinter()
        root = tk.Tk()
        root.after(100, root.destroy)
        root.mainloop()
        print("tk smoke ok")
        return 0
    launch(Path(args.root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
