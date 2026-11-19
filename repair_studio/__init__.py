"""Deterministic local image restoration and style-transfer studio."""

from __future__ import annotations

import sys
from pathlib import Path


_LOCAL_DEPS = Path(__file__).resolve().parent.parent / ".deps"
if _LOCAL_DEPS.is_dir() and str(_LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(_LOCAL_DEPS))

__version__ = "1.0.0"

