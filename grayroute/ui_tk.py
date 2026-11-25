"""Optional Tkinter front-end over the headless view-model.

Tkinter is part of the Python standard library but its native Tcl/Tk runtime
may be absent on a minimal machine.  Importing this module never raises at
import time; :func:`render` reports availability instead.
"""
from __future__ import annotations


def tk_available() -> bool:
    try:
        import tkinter  # noqa: F401
    except Exception:
        return False
    try:
        root = _make_root()
        root.destroy()
    except Exception:
        return False
    return True


def _make_root():
    import tkinter as tk
    return tk.Tk()


def render(conclusion: dict, consistency: dict) -> dict:
    """Show the dashboard if Tk works; otherwise return a headless report."""
    if not tk_available():
        return {
            "rendered": False,
            "reason": "Tkinter native runtime unavailable; use headless output",
            "conclusion": conclusion,
            "consistency": consistency,
        }
    import tkinter as tk
    root = tk.Tk()
    root.title("Gray-Route Receipt Ledger")
    text = tk.Text(root, width=96, height=28)
    text.pack()
    text.insert(tk.END,
                f"byte_identical={consistency['byte_identical']} "
                f"elapsed_ms={consistency['elapsed_ms']}\n")
    text.insert(tk.END,
                f"batches={conclusion['total_batches']} "
                f"invalid={conclusion['invalid_batches']}\n")
    for batch in conclusion["batches"]:
        text.insert(tk.END,
                    f"{batch['status']:7} {batch['batch_id']} "
                    f"{batch['worker_id']} v{batch['param_version_seq']}\n")
    root.mainloop()
    return {"rendered": True}
