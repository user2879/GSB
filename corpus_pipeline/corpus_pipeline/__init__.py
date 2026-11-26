"""Deterministic pretraining corpus pipeline.

The public engine is intentionally small: callers construct a configuration,
feed it to :func:`load_inputs`, and execute :func:`run_pipeline`.  The CLI adds
workspace generation and human-readable reporting on top of that surface.
"""

from .engine import run_pipeline, PipelineResult
from .loading import build_inputs, generate_workspace, load_inputs

__all__ = [
    "PipelineResult",
    "build_inputs",
    "generate_workspace",
    "load_inputs",
    "run_pipeline",
]
