"""Single monotonic clock source so every timing number shares one epoch."""
from __future__ import annotations

import time


def now_ms() -> int:
    """Monotonic clock in whole milliseconds."""
    return int(time.monotonic() * 1000)
