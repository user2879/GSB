"""Windows-style relative path handling.

The task mandates that *all configured paths* use Windows-style relative paths
(``samples\\tiles\\t00\\R.npy``).  This module is the single place that turns
those strings into native :class:`pathlib.Path` objects, so the workbench runs
identically on Linux/macOS while the on-disk configuration stays Windows style.
"""
from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def to_path(win_relative: str | os.PathLike[str], root: str | os.PathLike[str] | None = None) -> Path:
    """Translate a Windows-style relative path to a native path under ``root``."""
    if isinstance(win_relative, Path):
        text = str(win_relative)
    else:
        text = str(win_relative)
    # Normalise both back- and forward slashes through a PureWindowsPath so that
    # drive letters and separators are handled consistently.
    parts = PureWindowsPath(text.replace("/", "\\"))
    if parts.drive or parts.root:
        # Absolute or drive-qualified input: only take the relative tail.
        rel_parts = parts.parts[1:] if parts.root else parts.parts
        rel = Path(*rel_parts) if rel_parts else Path()
    else:
        rel = Path(*parts.parts)
    if root is not None:
        return Path(root) / rel
    return rel


def to_win(path: str | os.PathLike[str]) -> str:
    """Render a path using back-slash separators (for reports / manifests)."""
    p = Path(path)
    return "\\".join(p.parts)


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
