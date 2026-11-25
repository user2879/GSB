"""Optional Tkinter report viewer (Tk is optional for headless acceptance)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _flatten(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(value[key], child_prefix))
    elif isinstance(value, list):
        lines.append(f"{prefix}: {len(value)} item(s)")
        for index, item in enumerate(value[:20]):
            lines.extend(_flatten(item, f"{prefix}[{index}]"))
    else:
        lines.append(f"{prefix}: {value}")
    return lines


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tkinter verification report viewer")
    parser.add_argument("--report", default="artifacts/verification.json")
    args = parser.parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError as exc:
        print(f"Tkinter is unavailable in this Python installation: {exc}", file=sys.stderr)
        return 3
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"report not found: {report_path}; run python -m speaker_diag demo first", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    root = tk.Tk()
    root.title("Speaker Diag — batch attribution ledger")
    root.geometry("1040x720")
    header = tk.Label(
        root,
        text=f"status={report.get('status')}  ledger batches={report.get('counts', {}).get('ledger_batches')}",
        anchor="w",
        font=("Consolas", 12, "bold"),
        padx=10,
        pady=8,
    )
    header.pack(fill="x")
    text = scrolledtext.ScrolledText(root, wrap="none", font=("Consolas", 10))
    text.pack(fill="both", expand=True)
    text.insert("1.0", "\n".join(_flatten(report)))
    text.configure(state="disabled")
    root.mainloop()
    return 0

