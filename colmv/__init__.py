"""Deterministic incremental materialized-view aggregation for local Windows runs.

The public API is intentionally small.  Most integrations use :class:`LedgerStore`
and :class:`MaterializationEngine`; command-line users use ``python -m colmv``.
"""

from .engine import MaterializationEngine
from .errors import ColmvError
from .ledger import LedgerStore
from .schema import RuntimeConfig, load_config

__all__ = [
    "ColmvError",
    "LedgerStore",
    "MaterializationEngine",
    "RuntimeConfig",
    "load_config",
]

__version__ = "1.0.0"
