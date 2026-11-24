"""Batch attribution diagnostic component for a local inference service."""

from __future__ import annotations

from .errors import (
    ChainError,
    DiagnosticError,
    FieldError,
    ReplayError,
    SnapshotError,
    SpecError,
    VerifyError,
)
from .inference import ForwardResult, WorkerParameters, build_parameters, forward_batch
from .invalidation import InvalidationReport, PropagationStep, propagate
from .ledger import LEDGER_FORMAT_VERSION, SnapshotLedger
from .planning import Batch, BatchPlan, build_plan
from .receipt import NormalizedReceipt, decode_receipt, encode_receipt
from .spec import BudgetSpec, RequestSpec, SystemSpec, WorkerSpec, load_spec

__version__ = "1.0.0"

__all__ = [
    "DiagnosticError",
    "SpecError",
    "FieldError",
    "ChainError",
    "ReplayError",
    "SnapshotError",
    "VerifyError",
    "SystemSpec",
    "RequestSpec",
    "WorkerSpec",
    "BudgetSpec",
    "load_spec",
    "build_plan",
    "Batch",
    "BatchPlan",
    "build_parameters",
    "forward_batch",
    "ForwardResult",
    "WorkerParameters",
    "propagate",
    "InvalidationReport",
    "PropagationStep",
    "NormalizedReceipt",
    "decode_receipt",
    "encode_receipt",
    "SnapshotLedger",
    "LEDGER_FORMAT_VERSION",
]
