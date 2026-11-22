"""Deterministic multi-turn dialogue state management pipeline.

The package is intentionally pure-Python (plus optional NumPy for report
statistics) so that every run over the same events and configuration produces
byte-identical artefacts.
"""

from .errors import PipelineError, ERROR_CODES
from .models import Event, TransitionRule, SlotRecord, TurnCheckpoint, SessionResult, FailureRecord

__all__ = [
    "PipelineError",
    "ERROR_CODES",
    "Event",
    "TransitionRule",
    "SlotRecord",
    "TurnCheckpoint",
    "SessionResult",
    "FailureRecord",
]

