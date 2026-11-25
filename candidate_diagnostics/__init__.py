"""Deterministic candidate truncation, sampling, exchange and diagnostics."""

__version__ = "1.0.0"
VERSION_FINGERPRINT = "candidate-diagnostics-v1-" + __version__

from .api import CandidateDiagnostics, analyze_jsonl_batch

__all__ = ["CandidateDiagnostics", "analyze_jsonl_batch", "VERSION_FINGERPRINT", "__version__"]
