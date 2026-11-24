"""Diagnostics subpackage: headless view model, optional Tk UI, socket."""
from .viewmodel import WindowRow, build_window_rows, render_text_table

__all__ = ["WindowRow", "build_window_rows", "render_text_table"]
