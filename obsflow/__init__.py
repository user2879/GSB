"""obsflow - layered sliding-window aggregation for local observation streams.

Public package interface:

* configuration: :class:`~obsflow.config.SourceSetConfig`,
  :class:`~obsflow.config.EngineConfig`
* record IO: :func:`~obsflow.codec.load_stream`,
  :func:`~obsflow.codec.decode_stream`
* engine: :class:`~obsflow.engine.ObsFlowEngine`
* aggregation / leasing / scheduling / persistence primitives
"""
from __future__ import annotations

from .aggregation import (
    aggregate_all,
    aggregate_window,
    group_by_metric,
    ordered_records,
    quantize_batch,
    quantize_value,
)
from .clock import ManualClock, SystemClock
from .codec import decode_stream, load_stream, write_stream
from .config import (
    EngineConfig,
    SourceSetConfig,
    load_engine_config,
    load_source_set,
)
from .engine import AdvanceReport, ObsFlowEngine
from .errors import (
    FormatError,
    IntegrityError,
    LeaseError,
    ObsFlowError,
    OrderError,
    ReplayError,
    SnapshotError,
)
from .lease import LeaseManager
from .ledger import Ledger, SnapshotIdentity
from .models import (
    Lease,
    NormalizedRecord,
    PropagationEvent,
    SourceDeclaration,
    WindowParams,
    WindowResult,
)
from .propagation import PropagationPlan, derive_propagation, diff_source_sets
from .sample_data import generate_samples
from .scheduler import AggregationTask, BoundedScheduler

__version__ = "1.0.0"

__all__ = [
    "AdvanceReport",
    "AggregationTask",
    "BoundedScheduler",
    "EngineConfig",
    "FormatError",
    "IntegrityError",
    "Lease",
    "LeaseError",
    "LeaseManager",
    "Ledger",
    "ManualClock",
    "NormalizedRecord",
    "ObsFlowEngine",
    "ObsFlowError",
    "OrderError",
    "PropagationEvent",
    "PropagationPlan",
    "ReplayError",
    "SnapshotError",
    "SnapshotIdentity",
    "SourceDeclaration",
    "SourceSetConfig",
    "SystemClock",
    "WindowParams",
    "WindowResult",
    "aggregate_all",
    "aggregate_window",
    "decode_stream",
    "derive_propagation",
    "diff_source_sets",
    "generate_samples",
    "group_by_metric",
    "load_engine_config",
    "load_source_set",
    "load_stream",
    "ordered_records",
    "quantize_batch",
    "quantize_value",
    "write_stream",
]
