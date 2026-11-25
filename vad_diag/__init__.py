"""Streaming VAD diagnostics component.

A deterministic, content-fingerprinted, snapshot-recoverable pipeline for
streaming Voice Activity Detection.  Everything is built from a global sample
timeline reconstructed from audio *segments* (16 kHz / mono / 16-bit PCM in
RIFF/WAV containers) so that out-of-order arrival, replacement and late
segments produce auditable, reproducible endpoint conclusions.

The public surface is intentionally small; see :mod:`vad_diag.engine` for the
orchestrator and :mod:`vad_diag.cli` for the repeatable demo entry point.
"""

from .errors import (
    VAD_ERROR_CODES,
    VADFieldError,
    VADInjectionError,
    VADValidationError,
)
from .config import Config, load_config, ALLOWED_FRAC_BITS, ALLOWED_QUANT_MODES
from .engine import VADEngine
from .snapshot import SnapshotStore, config_hash
from .signals import install_handlers, GracefulExit

__all__ = [
    "VAD_ERROR_CODES",
    "VADFieldError",
    "VADInjectionError",
    "VADValidationError",
    "Config",
    "load_config",
    "ALLOWED_FRAC_BITS",
    "ALLOWED_QUANT_MODES",
    "VADEngine",
    "SnapshotStore",
    "config_hash",
    "install_handlers",
    "GracefulExit",
]

__version__ = "1.0.0"
