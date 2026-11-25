"""qcalib: layer-wise quantization calibration and distillation correction."""
from __future__ import annotations

from .errors import (
    E_BUDGET_PENDING,
    E_INPUT_SCHEMA,
    E_QUOTA_EXCEEDED,
    E_RECEIPT_MALFORMED,
    E_ZERO_SCALE,
    Failure,
    FailureList,
)
from .models import (
    CalibrationSet,
    DistillConfig,
    LayerQuantConfig,
    QuantConfig,
    WeightLayer,
    level_range,
)
from .pipeline import Pipeline, PipelineInputs, config_fingerprint
from .runner import diagnostic_report, recover_ledger, run_pipeline
from .snapshot import ForcedAbort, SnapshotStore

__all__ = [
    "E_BUDGET_PENDING",
    "E_INPUT_SCHEMA",
    "E_QUOTA_EXCEEDED",
    "E_RECEIPT_MALFORMED",
    "E_ZERO_SCALE",
    "Failure",
    "FailureList",
    "CalibrationSet",
    "DistillConfig",
    "LayerQuantConfig",
    "QuantConfig",
    "WeightLayer",
    "level_range",
    "Pipeline",
    "PipelineInputs",
    "config_fingerprint",
    "diagnostic_report",
    "recover_ledger",
    "run_pipeline",
    "ForcedAbort",
    "SnapshotStore",
]
