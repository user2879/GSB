"""Long-context LLM position-encoding extrapolation diagnostic workbench.

Public, reusable components:

* :class:`~pe_diag.encoding.PositionEncoder` /
  :class:`~pe_diag.encoding.EncodingResult` -- rotary, linear-interpolation and
  NTK-aware position encodings behind one interface;
* :func:`~pe_diag.scoring.score_sample` -- dot-product attention scoring with
  in/out-of-training layering and single/stream compute paths;
* :func:`~pe_diag.error_tracking.track_sample` -- three-stage extrapolation
  error tracking and precision-attribution;
* :func:`~pe_diag.sensitivity.run_sensitivity` -- base/scale boundary scans;
* :func:`~pe_diag.strategy.select_strategy` -- scale-tier trade-off selector;
* :class:`~pe_diag.pipeline.DiagnosticPipeline` -- deterministic, snapshot-backed
  orchestrator guaranteeing incremental == cold == concurrent results.
"""

from __future__ import annotations

from .config import Config, EffectiveConfig, Injection, Sample, load_samples
from .encoding import EncodingResult, PositionEncoder
from .error_tracking import track_sample
from .errors import (
    E_BASE_ZERO,
    E_KEY_POS_OOR,
    E_NONFINITE_INPUT,
    E_SCALE_INVALID,
    E_SNAPSHOT_STALE,
    DiagError,
)
from .pipeline import DiagnosticPipeline
from .scoring import score_sample
from .sensitivity import run_sensitivity
from .strategy import TierMetrics, select_strategy

__all__ = [
    "Config", "EffectiveConfig", "Sample", "Injection", "load_samples",
    "PositionEncoder", "EncodingResult", "score_sample", "track_sample",
    "run_sensitivity", "select_strategy", "TierMetrics",
    "DiagnosticPipeline", "DiagError", "E_BASE_ZERO", "E_KEY_POS_OOR",
    "E_NONFINITE_INPUT", "E_SCALE_INVALID", "E_SNAPSHOT_STALE",
]
