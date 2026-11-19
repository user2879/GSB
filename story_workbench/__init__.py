"""Deterministic narrative branch and event-causality workbench."""

from .errors import WorkbenchError
from .model import NarrativeSpec, SequenceSample
from .spec import load_spec, parse_spec
from .timeline import NarrativeSession

__all__ = [
    "NarrativeSession",
    "NarrativeSpec",
    "SequenceSample",
    "WorkbenchError",
    "load_spec",
    "parse_spec",
]

__version__ = "1.0.0"
