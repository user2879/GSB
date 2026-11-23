"""Optional read-only Tkinter viewer; the CLI remains fully functional alone."""

from __future__ import annotations

import json
from typing import Any

from .store import Store


def _status_document(db_path: str) -> dict[str, Any]:
    store = Store(db_path)
    try:
        snapshot = store.latest_snapshot()
        return {
            "ledger_cursor": store.ledger_cursor,
            "ledger_format_version": store.ledger_format_version,
            "ledger_content_digest": store.ledger_content_digest,
            "ledger_entries": len(store.ledger_entries()),
            "snapshot": None
            if snapshot is None
            else {
                "generation": snapshot["generation"],
                "content_digest": snapshot["content_digest"],
                "phase": snapshot["payload"].get("phase"),
                "latest_revision": snapshot["payload"].get("latest_revision"),
            },
            "stats": store.current_stats(),
            "view": list(store.current_view()),
            "invalidations": list(store.current_invalidations()),
        }
    finally:
        store.close()


def launch_ui(db_path: str) -> int:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": "E_TK",
                    "message": "tkinter is unavailable; use the CLI commands instead",
                },
                ensure_ascii=False,
            )
        )
        return 1

    root = tk.Tk()
    root.title(f"Config Pipeline - {db_path}")
    root.geometry("1100x760")
    header = tk.Label(
        root,
        text="Local deterministic distribution status (read-only)",
        anchor="w",
        padx=10,
        pady=8,
    )
    header.pack(fill="x")
    text = scrolledtext.ScrolledText(root, wrap="none", font=("Consolas", 10))
    text.pack(fill="both", expand=True)

    def refresh() -> None:
        try:
            document = _status_document(db_path)
            rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception as exc:  # noqa: BLE001 - displayed to the local operator
            rendered = f"{type(exc).__name__}: {exc}"
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", rendered)
        text.configure(state="disabled")

    controls = tk.Frame(root)
    controls.pack(fill="x", pady=6)
    tk.Button(controls, text="Refresh", command=refresh).pack(side="left", padx=10)
    tk.Button(controls, text="Close", command=root.destroy).pack(side="left")
    refresh()
    root.mainloop()
    return 0
