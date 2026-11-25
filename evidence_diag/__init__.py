"""Conversational evidence-chain diagnostics with deterministic fixed point math."""

from .engine import DiagnosticEngine, run_sequence
from .models import AppConfig, Evidence, Session, Turn
from .validation import validate_config, validate_session

__all__ = [
    "AppConfig",
    "DiagnosticEngine",
    "Evidence",
    "Session",
    "Turn",
    "run_sequence",
    "validate_config",
    "validate_session",
]

__version__ = "1.0.0"
