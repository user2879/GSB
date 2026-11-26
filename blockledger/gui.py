from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def launch(spec_path: str = "samples/spec.json") -> int:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "gui_check": "skipped",
                    "reason": type(exc).__name__ + ": " + str(exc),
                    "note": "use blockledger validate/run/verify without a display",
                },
                ensure_ascii=False,
            )
        )
        return 0

    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        print(json.dumps({"gui_check": "skipped", "reason": "DISPLAY is not set"}))
        return 0

    from .spec import load_json_path, volume_summary

    root = tk.Tk()
    root.title("BlockLedger Console")
    root.geometry("900x600")

    controls = ttk.Frame(root, padding=8)
    controls.pack(fill="x")
    status = ttk.Label(controls, text=f"Spec: {spec_path}")
    status.pack(side="left")

    text = tk.Text(root, wrap="none", font=("DejaVu Sans Mono", 10))
    text.pack(fill="both", expand=True, padx=8, pady=8)

    def show(payload: Any) -> None:
        text.delete("1.0", "end")
        text.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))

    def reload_spec() -> None:
        try:
            spec = load_json_path(spec_path)
            show({"ok": True, "volumes": volume_summary(spec), "budget": spec["budget"]})
            status.config(text=f"Loaded {spec_path}")
        except Exception as exc:  # noqa: BLE001
            show({"ok": False, "error": str(exc)})
            status.config(text="Spec error")

    ttk.Button(controls, text="Load/Validate", command=reload_spec).pack(side="right", padx=4)

    def show_report(name: str) -> None:
        path = Path("artifacts") / name
        if path.exists():
            show(json.loads(path.read_text(encoding="utf-8")))
        else:
            show({"missing_report": str(path)})

    for label, filename in (
        ("Validation", "validation.json"),
        ("Run", "run-report.json"),
        ("Verify", "verify-report.json"),
        ("Invalidation", "invalidation-write.json"),
    ):
        ttk.Button(controls, text=label, command=lambda f=filename: show_report(f)).pack(side="right", padx=4)

    reload_spec()
    root.mainloop()
    return 0
