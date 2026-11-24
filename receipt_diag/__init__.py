"""Inference receipt diagnostics package."""

from .client import WorkerPool
from .engine import DiagnosticsEngine
from .errors import DiagnosticError
from .invalidation import compute_invalidation
from .ledger import Ledger
from .planning import ExecutionPlan, plan_batches
from .receipts import (
    decode_receipt,
    decode_receipt_batch,
    encode_receipt,
    normalize_batch,
    normalize_receipt,
)
from .specs import SampleSpec, load_specs

__all__ = [
    "DiagnosticError",
    "DiagnosticsEngine",
    "ExecutionPlan",
    "Ledger",
    "SampleSpec",
    "WorkerPool",
    "compute_invalidation",
    "decode_receipt",
    "decode_receipt_batch",
    "encode_receipt",
    "load_specs",
    "normalize_batch",
    "normalize_receipt",
    "plan_batches",
]

__version__ = "0.1.0"
