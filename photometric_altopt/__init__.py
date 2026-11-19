"""多视点光度立体与可微渲染交替优化组件（仅依赖 NumPy）。"""

from __future__ import annotations

from .config import ObservationConfig, OptConfig, PipelineConfig, SolveConfig
from .differentiable import SurfaceParams, finite_difference_check
from .errors import BusinessError
from .faults import FaultConfig
from .forward import build_view_table
from .generator import generate_observations, load_observations, load_truth
from .pipeline import resume_run, run_alternating

__all__ = [
    "ObservationConfig",
    "OptConfig",
    "PipelineConfig",
    "SolveConfig",
    "SurfaceParams",
    "BusinessError",
    "FaultConfig",
    "build_view_table",
    "generate_observations",
    "load_observations",
    "load_truth",
    "finite_difference_check",
    "run_alternating",
    "resume_run",
]
