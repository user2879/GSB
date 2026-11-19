"""Timestamp helper kept separate for deterministic imports in tests."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
