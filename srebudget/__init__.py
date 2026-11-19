"""Service-call error-budget and circuit-breaker diagnostics.

The package is intentionally standard-library only.  The public API is exposed
from this module so that local scripts can use it without importing internals.
"""

from .clock import LogicalClock, ManualClock, MonotonicClock
from .engine import DiagnosticEngine
from .errors import SreError
from .loader import load_chain, load_chain_data, load_trace, load_trace_lines
from .models import ChainSpec, Outcome, TraceCall

__all__ = [
    "ChainSpec",
    "DiagnosticEngine",
    "LogicalClock",
    "ManualClock",
    "MonotonicClock",
    "Outcome",
    "SreError",
    "TraceCall",
    "load_chain",
    "load_chain_data",
    "load_trace",
    "load_trace_lines",
]

__version__ = "1.0.0"
