"""Deterministic event-time window aggregation with a SQLite evidence ledger."""

from .errors import BusinessError
from .model import Event, RuleSpec, SessionSpec, SourceSpec
from .spec import load_spec, parse_spec

__all__ = [
    "BusinessError",
    "Event",
    "RuleSpec",
    "SessionSpec",
    "SourceSpec",
    "load_spec",
    "parse_spec",
]

__version__ = "1.0.0"
