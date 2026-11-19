"""Deterministic vector graphics synthesis and verification."""

from .composite import composite_scene, render_path_layer
from .errors import ErrorCode
from .flatten import flatten_scene
from .inject import apply_injections
from .pipeline import compare_runs, run_case, run_demo
from .raster import rasterize_fill, rasterize_stroke
from .scene import generate_scene, validate_scene
from .verifier import verify_case_dir

__version__ = "1.0.0"

__all__ = [
    "ErrorCode",
    "apply_injections",
    "compare_runs",
    "composite_scene",
    "flatten_scene",
    "generate_scene",
    "rasterize_fill",
    "rasterize_stroke",
    "render_path_layer",
    "run_case",
    "run_demo",
    "validate_scene",
    "verify_case_dir",
]
