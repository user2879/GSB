"""Tkinter diagnostic UI, imported lazily so headless checks remain usable."""

from .view import format_text_view

__all__ = ["format_text_view", "launch"]


def launch(db_path: str, session_id: str | None = None) -> None:
    from .app import launch as _launch

    _launch(db_path, session_id)
