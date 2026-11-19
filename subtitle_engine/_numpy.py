"""Project-local NumPy import with a small dependency fallback.

The evaluation environment can have a project virtualenv while still invoking
``python -m ...`` with the system interpreter.  Prefer the normal import, but
also make the local ``.venv`` visible when NumPy is otherwise unavailable.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:  # pragma: no cover - normal path
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - environment bootstrap path
    root = Path(__file__).resolve().parent.parent
    candidates = [root / ".deps", *sorted(root.glob(".venv/lib/python*/site-packages"))]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import numpy as np

__all__ = ["np"]
