"""Deterministic spectral-imaging diagnostics workbench."""

from .config import (
    CalibrationConfig,
    CubeConfig,
    DriftConfig,
    InjectionConfig,
    PipelineConfig,
    UnmixConfig,
)
from .errors import ErrorCode, WorkbenchError

__all__ = [
    "CalibrationConfig",
    "CubeConfig",
    "DriftConfig",
    "ErrorCode",
    "InjectionConfig",
    "PipelineConfig",
    "UnmixConfig",
    "WorkbenchError",
]

__version__ = "1.0.0"
