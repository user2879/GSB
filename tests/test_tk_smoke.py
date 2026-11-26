from __future__ import annotations

import pytest

from lease_arbitrator.envcheck import check_tkinter


def test_tkinter_import_and_root_smoke() -> None:
    info = check_tkinter()
    if not info.get("available"):
        pytest.skip(f"tkinter unavailable: {info.get('error')}")
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no graphical display: {exc}")
    try:
        root.update_idletasks()
        root.update()
        assert int(root.winfo_exists()) == 1
    finally:
        root.destroy()
