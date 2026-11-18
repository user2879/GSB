"""Slow-query diagnosis and rewrite workbench."""

from .diagnostics import DiagnosisEngine, generate_candidates
from .errors import ERROR_CODES, WorkbenchError, fail
from .query import Query, is_slow, parse, parse_select
from .session import CLOCK_BASIS, Execution, Session
from .workbench import Workbench

__all__ = [
    "CLOCK_BASIS",
    "DiagnosisEngine",
    "ERROR_CODES",
    "Execution",
    "Query",
    "Session",
    "Workbench",
    "WorkbenchError",
    "fail",
    "generate_candidates",
    "is_slow",
    "parse",
    "parse_select",
]

__version__ = "1.0.0"
