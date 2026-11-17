#!/usr/bin/env python3
"""Wrapper that works when launched directly from the repository root."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunnelvpn.peer import main  # noqa: E402


raise SystemExit(main())
