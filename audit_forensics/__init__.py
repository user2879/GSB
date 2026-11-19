"""Deterministic offline audit-event correlation and forensics workbench."""

from .canonical import CORE_EVENT_FIELDS, make_sync_marker, normalize_timestamp
from .engine import AuditEngine
from .rules import RuleConfig, load_rules

__all__ = [
    "AuditEngine",
    "CORE_EVENT_FIELDS",
    "RuleConfig",
    "load_rules",
    "make_sync_marker",
    "normalize_timestamp",
]

__version__ = "1.0.0"
