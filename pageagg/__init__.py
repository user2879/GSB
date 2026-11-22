"""Deterministic columnar page aggregation for local Windows environments."""

from .errors import (
    DomainError,
    EBudget,
    EPage,
    ENullMap,
    EReplay,
    ERoute,
)

__all__ = [
    "DomainError",
    "EBudget",
    "ENullMap",
    "EPage",
    "EReplay",
    "ERoute",
]

__version__ = "1.0.0"
LEDGER_FORMAT_VERSION = 1
RESULT_FORMAT_VERSION = 1

