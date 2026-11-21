"""Deterministic multi-view 3D keypoint solver.

The package is intentionally dependency-light: NumPy for geometry and Pillow for
PNG decoding.  Public entry points are re-exported here for library users; the
command line interface lives in :mod:`pose3d.cli`.
"""

from .config import SolverConfig, load_config
from .loader import InputBundle, load_inputs
from .pipeline import SolverPipeline, run_pipeline

__all__ = [
    "SolverConfig",
    "load_config",
    "InputBundle",
    "load_inputs",
    "SolverPipeline",
    "run_pipeline",
]
__version__ = "1.0.0"
