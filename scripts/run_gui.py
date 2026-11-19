"""Launch the PySide6 workbench GUI.

On a headless machine use the offscreen platform plugin explicitly::

    set QT_QPA_PLATFORM=offscreen   (Windows cmd)
    $env:QT_QPA_PLATFORM="offscreen" (PowerShell)
    python scripts/run_gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loudness_workbench.config import WorkbenchConfig  # noqa: E402
from loudness_workbench.ui.main_window import run_window  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_window(WorkbenchConfig()))
