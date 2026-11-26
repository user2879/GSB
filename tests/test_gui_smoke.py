from __future__ import annotations

import tkinter as tk

import pytest

from workbench.gui import WorkbenchGui


def test_gui_constructs_when_display_available():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    try:
        app = WorkbenchGui(root)
        root.update_idletasks()
        root.update()
        app.write({"smoke": "ok"})
        root.update_idletasks()
        root.update()
    finally:
        root.destroy()
