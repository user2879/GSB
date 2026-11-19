from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import PipelineError
from .ledger import Ledger
from .verify import verify_ledger


def build_text(ledger_path: str) -> str:
    verification = verify_ledger(ledger_path)
    ledger = Ledger(ledger_path, create=False)
    try:
        sessions = ledger.list_sessions()
        snapshots = ledger.list_snapshots()
        active = ledger.active_state()
        lines = [
            "CDN Edge Cache Fixed-Point Diagnostics",
            "=" * 44,
            f"Ledger: {Path(ledger_path).resolve()}",
            f"Self-check: {'PASS' if verification['ok'] else 'FAIL'}",
            f"Sessions: {len(sessions)}  Batches: {verification['batch_count']}  Snapshots: {len(snapshots)}",
            "",
            "Sessions",
            "--------",
        ]
        for session in sessions:
            lines.append(
                f"{session['session_id']}: scale=10^{session['grid_power']} "
                f"digest={session['initial_content_digest'][:16]}..."
            )
        lines.extend(["", "Snapshots", "---------"])
        for snapshot in snapshots:
            cursor = json.loads(snapshot["cursor_json"])
            lines.append(
                f"{snapshot['snapshot_id']}  session={snapshot['session_id']} "
                f"batch={snapshot['batch_id']} kind={snapshot['kind']} "
                f"digest={snapshot['content_digest'][:12]}... "
                f"next_request={cursor.get('next_request_index')}"
            )
        lines.extend(["", "Active recovery state", "---------------------"])
        if active is None:
            lines.append("none")
        else:
            lines.append(
                f"snapshot={active['snapshot_id']} batch={active['batch_id']} "
                f"cache_entries={ledger.active_cache_count()}"
            )
        return "\n".join(lines) + "\n"
    finally:
        ledger.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optional tkinter ledger viewer")
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args(argv)
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        print(
            json.dumps(
                PipelineError(
                    "E_STATE",
                    "tkinter is unavailable; install the CPython tcl/tk component or use the CLI verifier",
                    path="gui",
                    field="tkinter",
                ).to_dict(),
                sort_keys=True,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 11
    text = build_text(args.ledger)
    root = tk.Tk()
    root.title("CDN Edge Cache Diagnostics")
    root.geometry("1120x720")
    widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Consolas", 10))
    widget.pack(fill=tk.BOTH, expand=True)
    widget.insert("1.0", text)
    widget.configure(state=tk.DISABLED)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
