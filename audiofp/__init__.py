"""Deterministic audio fingerprinting and similar-segment retrieval workbench."""

from .config import DEFAULT_CONFIG, LEGACY_CONFIG, FingerprintConfig
from .errors import (
    AudioFormatError,
    AudioWorkbenchError,
    DedupMergeError,
    FormatDriftError,
    IndexVersionError,
)

__all__ = [
    "AudioFormatError",
    "AudioWorkbenchError",
    "DEFAULT_CONFIG",
    "DedupMergeError",
    "FingerprintConfig",
    "FormatDriftError",
    "IndexVersionError",
    "LEGACY_CONFIG",
]

__version__ = "1.0.0"
