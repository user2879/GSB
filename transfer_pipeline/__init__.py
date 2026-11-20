"""Deterministic incremental cross-task transfer-learning pipeline."""

import os

# Small CPU matrices do not benefit from BLAS thread fan-out, and a fixed
# single-threaded BLAS improves bitwise repeatability across invocations.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from .pipeline import PipelineConfigMismatchError, PipelineResult, run_pipeline

__all__ = [
    "PipelineConfigMismatchError",
    "PipelineResult",
    "run_pipeline",
]
