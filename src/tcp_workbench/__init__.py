"""Deterministic TCP state and congestion diagnostics workbench."""

from .errors import (
    E_BUDGET,
    E_CONFIG,
    E_CONNECT,
    E_PORT,
    E_REPLAY,
    E_SCRIPT,
    E_STATE,
    E_VERSION,
    BusinessError,
)

__all__ = [
    "E_BUDGET",
    "E_CONFIG",
    "E_CONNECT",
    "E_PORT",
    "E_REPLAY",
    "E_SCRIPT",
    "E_STATE",
    "E_VERSION",
    "BusinessError",
]

__version__ = "0.1.0"
