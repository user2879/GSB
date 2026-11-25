"""Deterministic incremental voice-command registration diagnostics."""

from .engine import DiagnosticEngine, run_diagnostics
from .synth import generate_demo

__all__ = ["DiagnosticEngine", "run_diagnostics", "generate_demo"]
__version__ = "0.1.0"
